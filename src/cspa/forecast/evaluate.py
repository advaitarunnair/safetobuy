"""Rolling-origin forecast evaluation: pinball loss and interval coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cspa.config import QCOLS, QUANTILES
from cspa.forecast.cache import ForecastCache

INTERVALS = {"cov50": (0.25, 0.75), "cov80": (0.10, 0.90), "cov90": (0.05, 0.95)}
NOMINAL = {"cov50": 0.50, "cov80": 0.80, "cov90": 0.90}


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = y - q
    return np.maximum(tau * d, (tau - 1.0) * d)


def _pairs(cache: ForecastCache, units: np.ndarray, eval_cutoffs: np.ndarray, kind: str):
    """Yield (h, forecasts [m, n, 7], actuals [m, n]) over cutoffs whose targets are observed."""
    n_obs = units.shape[0]
    arr = cache.cal if kind == "cal" else cache.raw
    for h in range(1, cache.horizon + 1):
        ok = [int(c) for c in eval_cutoffs if c + h < n_obs]
        if not ok:
            continue
        ii = np.array([cache.idx(c) for c in ok])
        yield h, arr[ii, :, h - 1, :], units[np.array(ok) + h]


def evaluate_cache(cache: ForecastCache, units: np.ndarray, eval_cutoffs: np.ndarray, kind: str = "cal", label: str | None = None) -> pd.DataFrame:
    """One row per horizon plus an 'all' row. Pinball is also reported relative to total demand (wQL)."""
    rows = []
    acc = {"pin": np.zeros(len(QUANTILES)), "n": 0, "y": 0.0, **{k: 0.0 for k in INTERVALS}}
    for h, f, y in _pairs(cache, units, eval_cutoffs, kind):
        pin = np.array([pinball(y, f[..., j], tau).sum() for j, tau in enumerate(QUANTILES)])
        cov = {k: float(((y >= f[..., QUANTILES.index(lo)]) & (y <= f[..., QUANTILES.index(hi)])).sum()) for k, (lo, hi) in INTERVALS.items()}
        n = y.size
        rows.append(_row(label or cache.name, str(h), pin, n, float(y.sum()), cov))
        acc["pin"] += pin
        acc["n"] += n
        acc["y"] += float(y.sum())
        for k in INTERVALS:
            acc[k] += cov[k]
    if acc["n"]:
        rows.append(_row(label or cache.name, "all", acc["pin"], acc["n"], acc["y"], {k: acc[k] for k in INTERVALS}))
    return pd.DataFrame(rows)


def _row(model: str, h: str, pin: np.ndarray, n: int, ysum: float, cov: dict) -> dict:
    row = {"model": model, "h": h, "n": int(n)}
    for j, col in enumerate(QCOLS):
        row[f"pinball_{col}"] = float(pin[j] / n)
    row["pinball_mean"] = float(pin.mean() / n)
    row["wql"] = float(2.0 * pin.mean() / max(ysum, 1e-9))
    for k in INTERVALS:
        row[k] = float(cov[k] / n)
    return row


def evaluate_all(caches: dict[str, ForecastCache], units: np.ndarray, eval_cutoffs: np.ndarray) -> pd.DataFrame:
    """Compare forecasters on identical (cutoff, sku, horizon) cells."""
    parts = []
    for name, cache in caches.items():
        if name == "lgbm" and cache.meta.get("conformal"):
            parts.append(evaluate_cache(cache, units, eval_cutoffs, "raw", "lgbm_raw"))
            parts.append(evaluate_cache(cache, units, eval_cutoffs, "cal", "lgbm_conformal"))
        else:
            parts.append(evaluate_cache(cache, units, eval_cutoffs, "cal", name))
    return pd.concat(parts, ignore_index=True)
