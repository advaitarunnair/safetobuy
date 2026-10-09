"""Phases 1/3/5: closed-loop backtest of every policy in the same simulated world.

Usage:
  python scripts/run_backtest.py                  full grid -> results/<run_id>/, updates results/LATEST
  python scripts/run_backtest.py --tune-only      tuning windows only (never touches holdout, never updates LATEST)
  python scripts/run_backtest.py --quick          phase-1 check: one tune window, policies A and B
"""
from __future__ import annotations

import datetime as dt
import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from importlib import metadata

import pandas as pd

from _common import base_parser, config_hash, git_info, load_all, load_panel_and_params, now_utc, processed_dir, results_root, to_plain, write_json

from cspa.config import as_config
from cspa.forecast.cache import ForecastCache, cache_path
from cspa.sim.backtest import Scenario, resolve_windows, run_scenario, scenario_grid
from cspa.sim.bundle import write_bundle
from cspa.sim.metrics import aggregate, headline_set, summarise

_G: dict = {}


def _init(cfg_plain: dict) -> None:
    cfg = as_config(cfg_plain)
    panel, params = load_panel_and_params(cfg)
    _G.update(cfg=cfg, panel=panel, params=params, cache=ForecastCache.load(cache_path(cfg, str(cfg.forecast.model))))


def _work(job: tuple) -> dict:
    scenario, policies, capture = job
    res = run_scenario(_G["panel"], _G["params"], _G["cache"], _G["cfg"], scenario, policies, capture_policy=capture)
    res.pop("start_state")
    return res


def _is_bundle(sc: Scenario, exp) -> bool:
    ab = exp.app_bundle
    return sc.window.name == ab.window and sc.stress in [float(s) for s in ab.stress_levels] and sc.cash_cushion == float(ab.cash_cushion) and sc.shortfall_rule == ab.shortfall_rule


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "lightgbm", "scikit-learn", "scipy", "pyarrow", "streamlit"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--tune-only", action="store_true", help="only windows with role=tune; does not update results/LATEST")
    ap.add_argument("--quick", action="store_true", help="first tune window, default stress, policies A and B only")
    ap.add_argument("--policies", default=None, help="comma-separated subset of policies")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--tag", default=None, help="suffix for the run id")
    ap.add_argument("--no-latest", action="store_true", help="do not point results/LATEST at this run (comparison slices)")
    ap.add_argument("--secondary", action="store_true", help="comparison slice: no app bundle, results/SECONDARY points here, LATEST untouched")
    args = ap.parse_args()
    cfg, exp = load_all(args)
    panel, params = load_panel_and_params(cfg)
    model = str(cfg.forecast.model)
    cache = ForecastCache.load(cache_path(cfg, model))
    if cache.skus != panel.skus:
        raise SystemExit("Forecast cache and processed slice disagree on SKUs. Re-run `make forecast`.")

    policies = [p.strip() for p in args.policies.split(",")] if args.policies else list(exp.policies)
    grid = scenario_grid(exp, panel.n_obs)
    partial = bool(args.tune_only or args.quick or args.policies)
    if args.tune_only:
        grid = [s for s in grid if s.window.role == "tune"]
    if args.quick:
        first = next(w for w in resolve_windows(exp, panel.n_obs) if w.role == "tune")
        grid = [s for s in grid if s.window == first and s.stress == float(exp.default_stress) and s.cash_cushion == float(exp.default_cash_cushion) and s.shortfall_rule == "overdraft"]
        policies = ["A", "B"] if not args.policies else policies
    if not grid:
        raise SystemExit("No scenarios selected.")

    chash = config_hash(cfg, exp)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{chash[:8]}" + (f"_{args.tag}" if args.tag else ("_tune" if args.tune_only else "_quick" if args.quick else ""))
    out = results_root() / run_id
    bundle_policy = "C"
    if args.secondary:
        args.no_latest = True
    jobs = [(s, policies, bundle_policy if (not partial) and (not args.secondary) and bundle_policy in policies and _is_bundle(s, exp) else None) for s in grid]
    n_jobs = max(1, min(int(args.jobs or exp.n_jobs), len(jobs)))
    print(f"run {run_id}: {len(jobs)} scenarios x {len(policies)} policies, forecaster '{model}', {n_jobs} worker(s)")

    t0 = time.time()
    if n_jobs == 1:
        _init(to_plain(cfg))
        results = [_work(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init, initargs=(to_plain(cfg),)) as pool:
            results = list(pool.map(_work, jobs))
    seconds = round(time.time() - t0, 1)

    # Policy fairness: identical demand realisations and parameters for every policy.
    for r in results:
        if len(set(r["demand_hash"].values())) != 1:
            raise SystemExit(f"FAIRNESS VIOLATION in {r['scenario'].id}: policies saw different demand.")
    if len({r["params_hash"] for r in results}) != 1:
        raise SystemExit("FAIRNESS VIOLATION: scenarios used different SKU parameters.")

    out.mkdir(parents=True, exist_ok=True)
    weekly = pd.concat([r["weekly"] for r in results], ignore_index=True)
    metrics = summarise(weekly)
    weekly.to_parquet(out / "weekly.parquet", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    aggregate(metrics, ["role", "shortfall_rule"]).to_csv(out / "pooled_by_role_rule.csv", index=False)
    aggregate(metrics, ["role", "shortfall_rule", "stress", "cash_cushion"]).to_csv(out / "pooled_by_scenario.csv", index=False)
    pd.DataFrame([{**r["scenario"].fields(), **asdict(r["cal"])} for r in results]).to_csv(out / "calibration.csv", index=False)
    headlines = headline_set(weekly, exp, int(cfg.seed)) if {"A", "B", "C"} <= set(policies) else {}
    write_json(out / "headline.json", headlines)

    proc = processed_dir(cfg)
    for f in ("forecast_eval.csv", "forecast_meta.json"):
        if (proc / f).exists():
            shutil.copy(proc / f, out / f)
    bundle_runs = [r for r, j in zip(results, jobs) if j[2]]
    if bundle_runs:
        write_bundle(out / "app", panel, params, cache, cfg, bundle_runs, run_id, bundle_policy)

    c_rows = metrics[metrics["policy"] == "C"]
    manifest = {
        "run_id": run_id,
        "timestamp": now_utc(),
        "partial_run": partial,
        "git": git_info(),
        "config_hash": chash,
        "seed": int(cfg.seed),
        "data_source": panel.meta.get("data_source"),
        "data_slice": {k: v for k, v in panel.meta.items()},
        "forecaster": {"name": model, **{k: v for k, v in cache.meta.items() if k != "feature_importance"}},
        "policies": policies,
        "n_scenarios": len(jobs),
        "scenarios": [r["scenario"].id for r in results],
        "windows": [asdict(w) for w in resolve_windows(exp, panel.n_obs)],
        "mc_paths_backtest": int(cfg.sampling.n_paths_backtest),
        "sampling_method": str(cfg.sampling.method),
        "gate_mode": str(cfg.gate.mode),
        "gate_diagnostics_policy_C": {
            "decisions": int(c_rows["n_weeks"].sum()) if len(c_rows) else 0,
            "binding_weeks": int(c_rows["gate_binding_weeks"].sum()) if len(c_rows) else 0,
            "cash_at_risk_weeks": int(c_rows["gate_at_risk_weeks"].sum()) if len(c_rows) else 0,
            "monotonicity_violations": int(c_rows["gate_violations"].sum()) if len(c_rows) else 0,
        },
        "fairness_check": "passed",
        "runtime_seconds": seconds,
        "versions": _versions(),
        "config": to_plain(cfg),
        "experiments": to_plain(exp),
        "has_app_bundle": bool(bundle_runs),
    }
    write_json(out / "manifest.json", manifest)
    if not partial and not args.no_latest:
        (results_root() / "LATEST").write_text(run_id + "\n")
    if not partial and args.secondary:
        (results_root() / "SECONDARY").write_text(run_id + "\n")

    print("=" * 78)
    print(f"BACKTEST CHECKPOINT  ({seconds}s)  ->  results/{run_id}/")
    if panel.meta.get("data_source") != "M5":
        print("  *** SYNTHETIC TEST FIXTURE: these numbers are not results ***")
    print("=" * 78)
    pooled = aggregate(metrics, ["role", "shortfall_rule"])
    cols = ["role", "shortfall_rule", "policy", "n_runs", "shortfall_weeks", "insolvency_weeks", "fill_rate", "gross_margin", "lost_margin", "total_spend", "end_net_position", "avg_days_inventory"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pooled[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    gd = manifest["gate_diagnostics_policy_C"]
    print(f"\npolicy C gate: binding in {gd['binding_weeks']} of {gd['decisions']} decisions, cash-at-risk in {gd['cash_at_risk_weeks']}, monotonicity violations {gd['monotonicity_violations']}")
    for key, label in (("main", "HEADLINE"),):
        h = headlines.get(key)
        if h:
            print(f"\n{label} [{h['scope']}]:\n  {h['sentence']}")
    for role, h in (headlines.get("other_roles") or {}).items():
        if h:
            print(f"\n[{h['scope']}]:\n  {h['sentence']}")
    for rule, h in (headlines.get("by_rule") or {}).items():
        if h and rule != str(exp.headline.shortfall_rule):
            print(f"\n[{h['scope']}]:\n  {h['sentence']}")
    if partial or args.no_latest:
        print("\n(results/LATEST not updated)")


if __name__ == "__main__":
    sys.exit(main())
