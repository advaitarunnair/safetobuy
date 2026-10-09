"""Rolling-origin forecast cache.

Demand is exogenous to the purchasing policy, so forecasts are computed once
per cutoff week and every policy reads the same cached numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cspa.config import QCOLS
from cspa.data.load_m5 import Panel
from cspa.forecast.base import array_to_frame
from cspa.forecast.baselines import make_forecaster
from cspa.forecast.conformal import conformal_adjust


@dataclass
class ForecastCache:
    name: str
    skus: list[str]
    cutoffs: np.ndarray  # [C] sorted cutoff weeks
    raw: np.ndarray  # [C, n, H, 7]
    cal: np.ndarray  # [C, n, H, 7] conformally calibrated (== raw when not applied)
    calibrated: np.ndarray  # [C] bool
    meta: dict = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return int(self.raw.shape[2])

    def idx(self, cutoff: int) -> int:
        i = int(np.searchsorted(self.cutoffs, cutoff))
        if i >= len(self.cutoffs) or self.cutoffs[i] != cutoff:
            raise KeyError(f"no cached '{self.name}' forecast for cutoff week {cutoff}; cached range is {self.cutoffs[0]}..{self.cutoffs[-1]}. Re-run `make forecast`.")
        return i

    def get(self, cutoff: int, kind: str = "cal") -> np.ndarray:
        return (self.cal if kind == "cal" else self.raw)[self.idx(cutoff)]

    def frame(self, cutoff: int, kind: str = "cal") -> pd.DataFrame:
        return array_to_frame(self.skus, self.get(cutoff, kind))

    def subset(self, cutoffs: np.ndarray) -> "ForecastCache":
        ii = np.array([self.idx(int(c)) for c in cutoffs])
        return ForecastCache(self.name, list(self.skus), self.cutoffs[ii], self.raw[ii], self.cal[ii], self.calibrated[ii], dict(self.meta))

    def to_long(self) -> pd.DataFrame:
        C, n, H, Q = self.raw.shape
        df = pd.DataFrame(
            {
                "cutoff": np.repeat(self.cutoffs, n * H),
                "sku": np.tile(np.repeat(np.array(self.skus, dtype=object), H), C),
                "h": np.tile(np.arange(1, H + 1), C * n),
                "calibrated": np.repeat(self.calibrated, n * H),
            }
        )
        for j, col in enumerate(QCOLS):
            df[col] = self.cal[..., j].ravel()
        for j, col in enumerate(QCOLS):
            df[f"raw_{col}"] = self.raw[..., j].ravel()
        return df

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_long().to_parquet(path, index=False)
        path.with_suffix(".meta.json").write_text(json.dumps({"name": self.name, "skus": self.skus, **self.meta}, indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> "ForecastCache":
        if not path.exists():
            raise FileNotFoundError(f"Forecast cache not found: {path}. Run `make forecast` first.")
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        name, skus = meta.pop("name"), meta.pop("skus")
        df = pd.read_parquet(path)
        cutoffs = np.sort(df["cutoff"].unique())
        H = int(df["h"].max())
        C, n = len(cutoffs), len(skus)
        shape = (C, n, H, len(QCOLS))
        cal = df[list(QCOLS)].to_numpy(float).reshape(shape)
        raw = df[[f"raw_{c}" for c in QCOLS]].to_numpy(float).reshape(shape)
        calibrated = df["calibrated"].to_numpy(bool).reshape(C, n * H)[:, 0]
        return cls(name, skus, cutoffs.astype(np.int64), raw, cal, calibrated, meta)


def cache_path(cfg, name: str) -> Path:
    from cspa.config import resolve_path

    return resolve_path(cfg.data.processed_dir) / f"forecasts_{name}.parquet"


def build_cache(panel: Panel, cfg, cutoffs: np.ndarray, name: str, log=print) -> ForecastCache:
    """Roll the forecaster over `cutoffs` (refit every forecast.refit_every_weeks, predict every week)."""
    cutoffs = np.asarray(sorted(int(c) for c in cutoffs), dtype=np.int64)
    if cutoffs.size == 0:
        raise ValueError("no cutoffs to forecast")
    if cutoffs.max() >= panel.n_obs:
        raise ValueError("a forecast cutoff is beyond the last observed week")
    H = int(cfg.horizons.forecast_horizon_weeks)
    refit_every = int(cfg.forecast.refit_every_weeks)
    model = make_forecaster(name, cfg)
    raw = np.empty((len(cutoffs), panel.n_skus, H, len(QCOLS)))
    last_fit, fits = None, []
    for i, c in enumerate(cutoffs):
        c = int(c)
        if last_fit is None or c - last_fit >= refit_every:
            model.fit(panel, c)
            last_fit = c
            fits.append(c)
            if name == "lgbm":
                log(f"  [{name}] refit at cutoff {c} ({model.n_train_rows:,} rows)  [{i + 1}/{len(cutoffs)}]")
        raw[i] = model.predict_array(c, H, panel)

    cal = raw.copy()
    calibrated = np.zeros(len(cutoffs), dtype=bool)
    use_conformal = name == "lgbm" and bool(cfg.forecast.conformal.enabled)
    if use_conformal:
        for i, c in enumerate(cutoffs):
            cal[i], info = conformal_adjust(raw, cutoffs, i, panel.units[: int(c) + 1], cfg)
            calibrated[i] = info["applied"]
    meta = {
        "refit_every_weeks": refit_every,
        "refit_cutoffs": fits,
        "conformal": use_conformal,
        "first_calibrated_cutoff": int(cutoffs[calibrated][0]) if calibrated.any() else None,
        "horizon": H,
        "data_source": panel.meta.get("data_source"),
    }
    if name == "lgbm":
        meta["feature_importance"] = model.feature_importance()
    return ForecastCache(name, list(panel.skus), cutoffs, raw, cal, calibrated, meta)
