"""Sensitivity sweep on TUNING windows only. Never touches holdout windows, never writes results/.

Use it to choose policy-C settings (alpha, buffer, capital charge, ...) before the final run,
then record what you changed and why in docs/ASSUMPTIONS.md section 8.

Usage:
  python scripts/tune_sweep.py --set risk.alpha=0.05,0.10 --set synthetic.capital_rate_annual=0.15,1.0
  python scripts/tune_sweep.py --set risk.buffer_weeks_of_fixed_costs=1,2 --policies A,B,C,D
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import yaml

from _common import base_parser, load_all, load_panel_and_params, to_plain

from cspa.config import as_config, deep_merge, validate_config
from cspa.data.synth_params import make_sku_params
from cspa.forecast.cache import ForecastCache, cache_path
from cspa.sim.backtest import run_scenario, scenario_grid
from cspa.sim.metrics import summarise

_G: dict = {}


def _nest(path: str, value) -> dict:
    out: dict = value
    for part in reversed(path.split(".")):
        out = {part: out}
    return out


def _init(cfg_plain: dict) -> None:
    cfg = as_config(cfg_plain)
    panel, _ = load_panel_and_params(cfg)
    _G.update(panel=panel, cache=ForecastCache.load(cache_path(cfg, str(cfg.forecast.model))))


def _work(job: tuple) -> pd.DataFrame:
    label, cfg_plain, scenario, policies = job
    cfg = as_config(cfg_plain)
    params = make_sku_params(_G["panel"], cfg)  # synthetic overrides take effect here
    m = summarise(run_scenario(_G["panel"], params, _G["cache"], cfg, scenario, policies)["weekly"])
    m.insert(0, "setting", label)
    return m


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=V1,V2", help="config key and the values to try (repeatable)")
    ap.add_argument("--policies", default="A,B,C")
    ap.add_argument("--rule", default=None, help="shortfall rule (default: the headline rule)")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    cfg, exp = load_all(args)
    panel, _ = load_panel_and_params(cfg)
    rule = args.rule or str(exp.headline.shortfall_rule)
    scenarios = [s for s in scenario_grid(exp, panel.n_obs) if s.window.role == "tune" and s.shortfall_rule == rule and s.start_cash_weeks == float(exp.default_start_cash_weeks)]
    if not scenarios:
        raise SystemExit("No tuning scenarios.")
    keys, values = [], []
    for item in args.set:
        k, v = item.split("=", 1)
        keys.append(k.strip())
        values.append([yaml.safe_load(x) for x in v.split(",")])
    policies = [p.strip() for p in args.policies.split(",")]
    jobs = []
    for combo in itertools.product(*values) if keys else [()]:
        over: dict = {}
        for k, v in zip(keys, combo):
            over = deep_merge(over, _nest(k, v))
        plain = deep_merge(to_plain(cfg), over)
        validate_config(plain)
        label = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo)) or "default"
        jobs += [(label, plain, s, policies) for s in scenarios]
    n_jobs = max(1, min(int(args.jobs or exp.n_jobs), len(jobs)))
    print(f"{len(jobs)} runs on tuning windows {sorted({s.window.name for s in scenarios})}, rule {rule}, stress sweep at default opening cash; {n_jobs} worker(s)")
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init, initargs=(to_plain(cfg),)) as pool:
        metrics = pd.concat(list(pool.map(_work, jobs)), ignore_index=True)

    g = metrics.groupby(["setting", "policy"], sort=False).agg(
        weeks=("n_weeks", "sum"), shortfall=("shortfall_weeks", "sum"), insolvent=("insolvency_weeks", "sum"), fill=("fill_rate", "mean"),
        margin=("gross_margin", "sum"), days_inv=("avg_days_inventory", "mean"), gate_binding=("gate_binding_weeks", "sum"), gate_at_risk=("gate_at_risk_weeks", "sum"),
    ).reset_index()
    base = g.set_index(["setting", "policy"])["margin"]
    for ref in ("A", "B"):
        if ref in policies:
            g[f"margin_vs_{ref}_%"] = [100.0 * (m / base[(s, ref)] - 1.0) for s, m in zip(g["setting"], g["margin"])]
    print("TUNING WINDOWS ONLY" + ("  *** SYNTHETIC TEST FIXTURE: not results ***" if panel.meta.get("data_source") != "M5" else ""))
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(g.drop(columns="margin").to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
