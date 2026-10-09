"""Phase 0: slice M5, aggregate to Walmart weeks, generate synthetic SKU parameters.

Usage: python scripts/prepare_data.py [--config ...] [--experiments ...]
"""
from __future__ import annotations

import pandas as pd

from _common import MissingDataError, base_parser, exit_missing_data, load_all, processed_dir, resolve_path

from cspa.data.load_m5 import REQUIRED_FILES, check_raw, load_slice, slice_summary
from cspa.data.synth_params import make_sku_params, params_summary
from cspa.sim.backtest import resolve_windows, selection_end_week


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg, exp = load_all(args)
    raw_dir = resolve_path(cfg.data.raw_dir)
    try:
        check_raw(raw_dir)
    except MissingDataError as exc:
        exit_missing_data(exc)

    # Number of observed weeks is needed to place the windows; read it from the sales header only.
    header = pd.read_csv(raw_dir / REQUIRED_FILES[0], nrows=0).columns
    cal = pd.read_csv(raw_dir / "calendar.csv", usecols=["d", "wm_yr_wk"])
    days = cal[cal["d"].isin([c for c in header if c.startswith("d_")])].groupby("wm_yr_wk")["d"].count()
    n_obs = int((days == 7).sum())
    sel_end = selection_end_week(exp, cfg, n_obs)

    panel = load_slice(cfg, sel_end)
    params = make_sku_params(panel, cfg)
    out = processed_dir(cfg)
    panel.save(out)
    params.to_frame().to_parquet(out / "sku_params.parquet", index=False)

    print("=" * 78)
    print("PHASE 0 CHECKPOINT: slice summary")
    print("=" * 78)
    print(slice_summary(panel))
    print()
    print(params_summary(params))
    print()
    print("backtest windows (decision weeks, inclusive):")
    for w in resolve_windows(exp, panel.n_obs):
        d0 = panel.weeks["start_date"].iloc[w.start].date()
        d1 = (panel.weeks["start_date"].iloc[w.end] + pd.Timedelta(days=6)).date()
        print(f"  {w.name} [{w.role:7s}] weeks {w.start}..{w.end}  ({d0} .. {d1})")
    print(f"\nwrote {out}/weekly.parquet, weeks.parquet, sku_params.parquet, slice_meta.json")


if __name__ == "__main__":
    main()
