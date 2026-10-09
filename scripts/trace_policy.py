"""Week-by-week trace of policies A, B and C through one TUNING window, with the probability
curve the cash gate sees. Refuses held-out windows. Used to diagnose policy C.

Usage: python scripts/trace_policy.py --window W1 --stress 0.7 [--set gate.continuation=replace_sales ...]

Columns: cash at the end of each week, spend, C's unconstrained plan cost (Cfull), the gate's
budget (B*), its status and P(safe), P(safe) if C's state bought policy A's order instead, and
P(safe) at 0%, 20%, ... 100% of C's unconstrained plan.
"""
from __future__ import annotations

import pandas as pd
import yaml

from _common import base_parser, load_all, load_panel_and_params, to_plain

from cspa.config import as_config, deep_merge, validate_config
from cspa.forecast.cache import ForecastCache, cache_path
from cspa.policies import PlanContext, make_policy
from cspa.sim.backtest import burn_in_state, calibrate_world, make_info, resolve_windows, world_price
from cspa.sim.world import World


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--window", default="W1")
    ap.add_argument("--stress", type=float, default=0.7)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg, exp = load_all(args)
    plain = to_plain(cfg)
    for item in args.set:
        k, v = item.split("=", 1)
        over = yaml.safe_load(v)
        for part in reversed(k.strip().split(".")):
            over = {part: over}
        plain = deep_merge(plain, over)
    validate_config(plain)
    cfg = as_config(plain)
    panel, params = load_panel_and_params(cfg)
    cache = ForecastCache.load(cache_path(cfg, str(cfg.forecast.model)))
    w = next((x for x in resolve_windows(exp, panel.n_obs) if x.name == args.window), None)
    if w is None or w.role != "tune":
        raise SystemExit(f"{args.window} is not a tuning window. Traces are for tuning windows only.")
    H = int(cfg.horizons.cash_horizon_weeks)
    cal = calibrate_world(panel, params, cfg, w.start, args.stress, float(exp.default_cash_cushion))
    start = burn_in_state(panel, cache, params, cfg, w.start)
    start.cash = cal.start_cash
    world = World(params, str(exp.headline.shortfall_rule))
    pols = {k: make_policy(k) for k in ("A", "B", "C")}
    st = {k: start.copy() for k in pols}
    print(f"TUNING WINDOW {w.name}, stress {args.stress:g}, store {panel.meta['store_id']} {panel.meta['dept_id']}; overrides: {args.set or 'none'}")
    print(f"buffer {cal.buffer:,.0f}  opening cash {cal.start_cash:,.0f}  lump {cal.lump:,.0f}  weekly base fixed {cal.fixed_base:,.0f}  R* {cal.repl_star:,.0f}")
    rows = []
    for t in range(w.start, w.end + 1):
        info = make_info(panel, cache, params, cfg, t, cal.schedule(t, H), cal.buffer)
        ctx = PlanContext(st["C"], info)
        alloc = ctx.allocator("marginal")
        curve = [ctx.simulate(alloc(f * ctx.table.total_cost)).prob_above(cal.buffer) for f in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        p_a = ctx.simulate(ctx.need).prob_above(cal.buffer)
        row = {"wk": t - w.start + 1, "fixed": round(cal.fixed(t))}
        for k, pol in pols.items():
            o = pol.decide(st[k], info)
            st[k], rec = world.step(st[k], o, panel.units[t], world_price(panel, params, t), cal.fixed(t), cal.buffer)
            row[f"{k}_cash"], row[f"{k}_spend"] = round(rec["cash_end"]), round(rec["order_cost"])
            row[f"{k}_fill"] = round(rec["units_sold"] / max(rec["units_demand"], 1), 2)
            if k == "C":
                g = o.meta["gate"]
                row.update({"Cfull": round(g.B_max), "B*": round(g.B), "status": g.status[:7], "P*": round(g.prob_safe, 2)})
        row["P(A order)"] = round(p_a, 2)
        row["P at 0/20/40/60/80/100%"] = " ".join(f"{p:.2f}" for p in curve)
        rows.append(row)
    cols = ["wk", "fixed", "A_cash", "B_cash", "C_cash", "A_spend", "B_spend", "C_spend", "Cfull", "B*", "status", "P*", "P(A order)", "P at 0/20/40/60/80/100%", "A_fill", "B_fill", "C_fill"]
    with pd.option_context("display.width", 250):
        print(pd.DataFrame(rows)[cols].to_string(index=False))
    for k in pols:
        short = sum(1 for r in rows if r[f"{k}_cash"] < cal.buffer)
        print(f"policy {k}: {short} weeks below the buffer, mean fill {sum(r[f'{k}_fill'] for r in rows) / len(rows):.2f}, total spend {sum(r[f'{k}_spend'] for r in rows):,.0f}")


if __name__ == "__main__":
    main()
