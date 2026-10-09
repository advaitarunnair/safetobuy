"""Feature construction for the pooled quantile model.

Strictly past-only: `origin_block(panel, upto)` slices sales and prices to
weeks <= upto before computing anything, so no value after the forecast origin
can influence a feature. Calendar features (events, SNAP days) for the target
week are known in advance, exactly as in the M5 competition, and are the only
future-dated inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cspa.data.load_m5 import Panel

ORIGIN_FEATURES = (
    "lag0", "lag1", "lag2", "lag3",
    "rm4", "rm8", "rm26", "rm52",
    "rs4", "rs13", "rmax13", "zero13", "trend", "log_scale",
    "price", "price_rel13", "price_chg", "price_rel_max", "listed", "age",
)
TARGET_FEATURES = (
    "h", "seas1", "seas3",
    "woy_sin", "woy_cos", "month",
    "snap_days", "event_days", "ev_sporting", "ev_cultural", "ev_national", "ev_religious", "christmas",
    "event_days_next", "christmas_next", "snap_days_origin",
)
FEATURE_NAMES = ORIGIN_FEATURES + TARGET_FEATURES
SEASON = 52


def origin_block(panel: Panel, upto: int) -> dict[str, np.ndarray]:
    """Origin-level features for every origin week 0..upto. Arrays are [upto + 1, n_skus]."""
    if upto >= panel.n_obs:
        raise ValueError(f"origin {upto} is beyond the last observed week {panel.n_obs - 1}")
    u = pd.DataFrame(panel.units[: upto + 1])
    p = pd.DataFrame(panel.price[: upto + 1])

    def roll(df: pd.DataFrame, k: int, fn: str, min_periods: int = 1) -> pd.DataFrame:
        return getattr(df.rolling(k, min_periods=min_periods), fn)()

    rm13 = roll(u, 13, "mean")
    scale = rm13 + 1.0
    out: dict[str, np.ndarray] = {"scale": scale.to_numpy()}
    for k in range(4):
        out[f"lag{k}"] = (u.shift(k) / scale).to_numpy()
    for k in (4, 8, 26, 52):
        out[f"rm{k}"] = (roll(u, k, "mean") / scale).to_numpy()
    out["rs4"] = (roll(u, 4, "std", 2) / scale).fillna(0.0).to_numpy()
    out["rs13"] = (roll(u, 13, "std", 2) / scale).fillna(0.0).to_numpy()
    out["rmax13"] = (roll(u, 13, "max") / scale).to_numpy()
    out["zero13"] = roll((u == 0).astype(float), 13, "mean").to_numpy()
    out["trend"] = ((roll(u, 4, "mean") + 0.5) / (rm13 + 0.5)).to_numpy()
    out["log_scale"] = np.log1p(rm13.to_numpy())
    out["price"] = p.to_numpy()
    out["price_rel13"] = (p / roll(p, 13, "mean")).to_numpy()
    out["price_chg"] = (p / p.shift(1) - 1.0).to_numpy()
    out["price_rel_max"] = (p / p.expanding(min_periods=1).max()).to_numpy()
    listed = p.notna().astype(float)
    out["listed"] = listed.to_numpy()
    out["age"] = (listed.cumsum().clip(upper=104) / 104.0).to_numpy()
    return out


def assemble(panel: Panel, block: dict[str, np.ndarray], origins: np.ndarray, h: int, upto: int, with_target: bool):
    """Stack features for (origin, sku) pairs at horizon h. Row order: origin-major, sku-minor.

    Returns X [len(origins) * n_skus, n_features] float32, scale [rows], y (scaled) or None.
    Only sales weeks <= upto are ever read.
    """
    origins = np.asarray(origins, dtype=np.int64)
    if origins.size and (origins.max() > upto or (with_target and origins.max() + h > upto)):
        raise ValueError("assemble() asked for data beyond the cutoff")
    n = panel.n_skus
    units = panel.units[: upto + 1]
    scale = block["scale"][origins]
    cols: list[np.ndarray] = [block[name][origins] for name in ORIGIN_FEATURES]

    tgt = origins + h
    ones = np.ones((1, n))
    cols.append(np.full((origins.size, n), float(h)))
    s_idx = tgt - SEASON
    seas1 = np.full((origins.size, n), np.nan)
    ok = s_idx >= 0
    seas1[ok] = units[s_idx[ok]] / scale[ok]
    cols.append(seas1)
    seas3 = np.full((origins.size, n), np.nan)
    ok3 = (s_idx - 1 >= 0) & (s_idx + 1 <= origins)
    if ok3.any():
        j = s_idx[ok3]
        seas3[ok3] = (units[j - 1] + units[j] + units[j + 1]) / 3.0 / scale[ok3]
    cols.append(seas3)
    woy = panel.cal("woy", tgt)
    cols.append(np.sin(2 * np.pi * woy / 52.0)[:, None] * ones)
    cols.append(np.cos(2 * np.pi * woy / 52.0)[:, None] * ones)
    for name in ("month", "snap_days", "event_days", "ev_sporting", "ev_cultural", "ev_national", "ev_religious", "christmas"):
        cols.append(panel.cal(name, tgt)[:, None] * ones)
    cols.append(panel.cal("event_days", tgt + 1)[:, None] * ones)
    cols.append(panel.cal("christmas", tgt + 1)[:, None] * ones)
    cols.append(panel.cal("snap_days", origins)[:, None] * ones)

    X = np.stack([c.reshape(-1) for c in cols], axis=1).astype(np.float32)
    y = (units[tgt] / scale).reshape(-1) if with_target else None
    return X, scale.reshape(-1), y
