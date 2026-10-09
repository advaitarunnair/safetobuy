"""Small results bundle that lets the Streamlit app run without the raw M5 files.

For chosen scenarios it stores policy C's pre-decision state for every week of
the window, the forecasts and recent weekly actuals needed to rebuild that
week's information set, and the SKU parameters. The app then re-runs only the
gate and the allocator for the selected week.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cspa.config import Config, to_plain
from cspa.data.load_m5 import Panel
from cspa.data.synth_params import SkuParams
from cspa.forecast.cache import ForecastCache
from cspa.sim.backtest import WorldCal
from cspa.sim.world import WorldState

HISTORY_WEEKS = 60


def write_bundle(out_dir: Path, panel: Panel, params: SkuParams, cache: ForecastCache, cfg, runs: list[dict], run_id: str, policy: str = "C") -> None:
    """runs: outputs of run_scenario(..., capture_policy=policy) for the bundle scenarios."""
    out_dir.mkdir(parents=True, exist_ok=True)
    H = int(cfg.horizons.forecast_horizon_weeks)
    first_week = min(c["week"] for r in runs for c in r["captured"])
    last_week = max(c["week"] for r in runs for c in r["captured"])
    lo_cut = max(int(cache.cutoffs[0]), first_week - 1 - H - int(cfg.sampling.bank_weeks))
    keep = cache.cutoffs[(cache.cutoffs >= lo_cut) & (cache.cutoffs <= last_week - 1)]
    cache.subset(keep).save(out_dir / "forecasts.parquet")

    trimmed = panel.copy()
    lo_hist = max(0, lo_cut - HISTORY_WEEKS)
    trimmed.units[:lo_hist] = np.nan
    trimmed.price[:lo_hist] = np.nan
    trimmed.meta = {k: v for k, v in panel.meta.items() if k != "raw_files"}
    trimmed.save(out_dir)
    params.to_frame().to_parquet(out_dir / "sku_params.parquet", index=False)

    arrays, scen_meta = {}, []
    for r in runs:
        sc, cap = r["scenario"], r["captured"]
        sid = sc.id
        arrays[f"{sid}__weeks"] = np.array([c["week"] for c in cap])
        arrays[f"{sid}__cash"] = np.array([c["cash"] for c in cap])
        for key in ("on_hand", "pipeline", "payables", "spoil_acc"):
            arrays[f"{sid}__{key}"] = np.stack([c[key] for c in cap])
        scen_meta.append({**sc.fields(), "cal": asdict(r["cal"]), "weeks": [int(c["week"]) for c in cap]})
    np.savez_compressed(out_dir / "states.npz", **arrays)
    meta = {
        "run_id": run_id,
        "policy": policy,
        "data_source": panel.meta.get("data_source"),
        "first_history_week": int(lo_hist),
        "scenarios": scen_meta,
        "config": to_plain(cfg),
    }
    (out_dir / "bundle_meta.json").write_text(json.dumps(meta, indent=2, default=str))


@dataclass
class Bundle:
    meta: dict
    cfg: Config
    panel: Panel
    params: SkuParams
    cache: ForecastCache
    states: dict

    @property
    def policy(self) -> str:
        """The policy whose week-by-week states this bundle holds."""
        return str(self.meta.get("policy", "C"))

    def scenario(self, sid: str) -> dict:
        return next(s for s in self.meta["scenarios"] if s["scenario"] == sid)

    def cal(self, sid: str) -> WorldCal:
        return WorldCal(**self.scenario(sid)["cal"])

    def state(self, sid: str, week: int) -> WorldState:
        weeks = self.states[f"{sid}__weeks"]
        i = int(np.flatnonzero(weeks == week)[0])
        return WorldState(
            week=int(week),
            cash=float(self.states[f"{sid}__cash"][i]),
            on_hand=self.states[f"{sid}__on_hand"][i].astype(np.int64),
            pipeline=self.states[f"{sid}__pipeline"][i].astype(np.int64),
            payables=self.states[f"{sid}__payables"][i].astype(float),
            spoil_acc=self.states[f"{sid}__spoil_acc"][i].astype(float),
        )


def load_bundle(bundle_dir: Path) -> Bundle:
    from cspa.config import _wrap

    meta_path = bundle_dir / "bundle_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No app bundle in {bundle_dir}. Run `make backtest` first.")
    meta = json.loads(meta_path.read_text())
    states = dict(np.load(bundle_dir / "states.npz"))
    return Bundle(
        meta=meta,
        cfg=_wrap(meta["config"]),
        panel=Panel.load(bundle_dir),
        params=SkuParams.from_frame(pd.read_parquet(bundle_dir / "sku_params.parquet")),
        cache=ForecastCache.load(bundle_dir / "forecasts.parquet"),
        states=states,
    )
