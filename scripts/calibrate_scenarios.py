"""Do the stress scenarios bind? Counts the weeks policy A (and B) ends below the buffer,
for each stress level and opening-cash cushion, on TUNING windows only.

Usage: python scripts/calibrate_scenarios.py [--cushions 0.2,0.35,0.5,0.75,1.0]
"""
from __future__ import annotations

import itertools

import pandas as pd

from _common import base_parser, load_all, load_panel_and_params

from cspa.forecast.cache import ForecastCache, cache_path
from cspa.sim.backtest import Scenario, resolve_windows, run_scenario
from cspa.sim.metrics import summarise


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--cushions", default="0.15,0.25,0.35,0.5,0.75,1.0")
    args = ap.parse_args()
    cfg, exp = load_all(args)
    panel, params = load_panel_and_params(cfg)
    cache = ForecastCache.load(cache_path(cfg, str(cfg.forecast.model)))
    wins = [w for w in resolve_windows(exp, panel.n_obs) if w.role == "tune"]
    cushions = [float(x) for x in args.cushions.split(",")]
    rows = []
    for w, s, c in itertools.product(wins, [float(x) for x in exp.stress_levels], cushions):
        res = run_scenario(panel, params, cache, cfg, Scenario(w, s, c, str(exp.headline.shortfall_rule)), ["A", "B"])
        m = summarise(res["weekly"]).set_index("policy")
        rows.append({"window": w.name, "stress": s, "cushion": c, "A": int(m.loc["A", "shortfall_weeks"]), "B": int(m.loc["B", "shortfall_weeks"]), "weeks": int(m.loc["A", "n_weeks"])})
    df = pd.DataFrame(rows)
    sl = panel.meta
    total = int(df.groupby(["stress", "cushion"])["weeks"].sum().iloc[0])
    print(f"TUNING WINDOWS ONLY ({', '.join(w.name for w in wins)}); store {sl['store_id']}, department {sl['dept_id']}; rule {exp.headline.shortfall_rule}")
    print("Opening cash = buffer + cushion x stress x R*.  Rows: stress level.  Columns: cushion.")
    for pol in ("A", "B"):
        print(f"\nPolicy {pol}: weeks ending below the buffer, both tuning windows pooled (of {total})")
        print(df.pivot_table(index="stress", columns="cushion", values=pol, aggfunc="sum").to_string())
    for wname, g in df.groupby("window"):
        print(f"\nPolicy A, window {wname} (of {int(g['weeks'].iloc[0])})")
        print(g.pivot(index="stress", columns="cushion", values="A").to_string())
    print(f"\nChosen: experiments.default_cash_cushion = {exp.default_cash_cushion}, cash_cushions = {list(exp.cash_cushions)}")


if __name__ == "__main__":
    main()
