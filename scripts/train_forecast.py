"""Phase 2: rolling-origin quantile forecasts, cached once for every policy to read.

Usage: python scripts/train_forecast.py [--models lgbm,seasonal_naive,naive]
"""
from __future__ import annotations

import time

import pandas as pd

from _common import base_parser, config_hash, ensure_openmp, load_all, load_panel_and_params, now_utc, processed_dir, write_json

from cspa.forecast.cache import build_cache, cache_path
from cspa.forecast.evaluate import evaluate_all
from cspa.sim.backtest import decision_cutoffs, required_cutoffs


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--models", default="lgbm,seasonal_naive,naive")
    args = ap.parse_args()
    cfg, exp = load_all(args)
    if "lgbm" in args.models:
        ensure_openmp()
    panel, _ = load_panel_and_params(cfg)
    cutoffs = required_cutoffs(exp, cfg, panel.n_obs)
    if cutoffs.min() < 60:
        raise SystemExit(f"Not enough history: the first forecast cutoff would be week {cutoffs.min()}.")
    print(f"forecast cutoffs {cutoffs[0]}..{cutoffs[-1]} ({len(cutoffs)} weeks), horizon {cfg.horizons.forecast_horizon_weeks}, refit every {cfg.forecast.refit_every_weeks} weeks")

    caches, timing = {}, {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        t0 = time.time()
        caches[name] = build_cache(panel, cfg, cutoffs, name)
        caches[name].save(cache_path(cfg, name))
        timing[name] = round(time.time() - t0, 1)
        print(f"  cached '{name}' in {timing[name]}s -> {cache_path(cfg, name).name}")

    parts = []
    for scope, roles in (("all", None), ("tune", ("tune",)), ("holdout", ("holdout",))):
        cut = decision_cutoffs(exp, panel.n_obs, roles)
        if cut.size:
            t = evaluate_all(caches, panel.units, cut)
            t.insert(0, "scope", scope)
            parts.append(t)
    table = pd.concat(parts, ignore_index=True)
    out = processed_dir(cfg)
    table.to_csv(out / "forecast_eval.csv", index=False)
    lg = caches.get("lgbm")
    write_json(
        out / "forecast_meta.json",
        {
            "timestamp": now_utc(),
            "config_hash": config_hash(cfg, exp),
            "data_source": panel.meta.get("data_source"),
            "cutoffs": [int(cutoffs[0]), int(cutoffs[-1])],
            "n_cutoffs": int(len(cutoffs)),
            "refit_every_weeks": int(cfg.forecast.refit_every_weeks),
            "n_refits": len(lg.meta["refit_cutoffs"]) if lg else None,
            "first_calibrated_cutoff": lg.meta["first_calibrated_cutoff"] if lg else None,
            "feature_importance": lg.meta.get("feature_importance") if lg else None,
            "seconds": timing,
        },
    )

    print("=" * 78)
    print("PHASE 2 CHECKPOINT: rolling-origin forecast evaluation (backtest decision weeks, all horizons)")
    if panel.meta.get("data_source") != "M5":
        print("  *** SYNTHETIC TEST FIXTURE: these numbers are not results ***")
    print("=" * 78)
    show = table[(table["h"] == "all") & (table["scope"] == "all")][["model", "n", "wql", "pinball_mean", "cov50", "cov80", "cov90"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("nominal coverage: cov50=0.50, cov80=0.80, cov90=0.90.  wQL = 2 x mean pinball / mean demand (lower is better).")


if __name__ == "__main__":
    main()
