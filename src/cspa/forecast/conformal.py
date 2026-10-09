"""Rolling conformal calibration of the quantile forecasts.

For each quantile level tau and horizon h we look at how past *genuinely
out-of-sample* forecasts (issued at earlier cutoffs, already realised) missed:
    r = (y - q_tau) / scale,   scale = max(q75 - q25, floor)
and shift today's quantile by the tau-quantile of r, with the usual finite-sample
correction. This is a per-quantile (one-sided) form of conformalized quantile
regression, pooled across SKUs through the scale normalisation.

The calibration set for cutoff c only contains forecasts whose target week is
<= c, so it ends strictly before anything being predicted.
"""
from __future__ import annotations

import math

import numpy as np

from cspa.config import QUANTILES

_IQR_LO, _IQR_HI = QUANTILES.index(0.25), QUANTILES.index(0.75)


def calibration_origins(cutoffs: np.ndarray, cutoff: int, h: int, calibration_weeks: int) -> np.ndarray:
    """Indices into `cutoffs` of past forecast origins whose h-step target lies in (cutoff - K, cutoff]."""
    tgt = np.asarray(cutoffs) + h
    return np.flatnonzero((tgt <= cutoff) & (tgt > cutoff - calibration_weeks))


def _scale(q: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(q[..., _IQR_HI] - q[..., _IQR_LO], floor)


def _level(tau: float, n: int) -> float:
    if tau > 0.5:
        return min(1.0, math.ceil((n + 1) * tau) / n)
    if tau < 0.5:
        return max(0.0, math.floor((n + 1) * tau) / n)
    return 0.5


def conformal_adjust(raw: np.ndarray, cutoffs: np.ndarray, ci: int, units_upto_cutoff: np.ndarray, cfg) -> tuple[np.ndarray, dict]:
    """Calibrate raw[ci] ([n, H, 7]) using earlier raw forecasts and realised demand <= cutoff.

    units_upto_cutoff must be units[: cutoff + 1]; passing more is an error.
    """
    cutoff = int(cutoffs[ci])
    if units_upto_cutoff.shape[0] != cutoff + 1:
        raise ValueError("conformal_adjust must be given units up to and including the cutoff, nothing later")
    cc = cfg.forecast.conformal
    K, floor, min_n = int(cc.calibration_weeks), float(cc.scale_floor), int(cc.min_samples)
    H = raw.shape[2]
    out = raw[ci].copy()
    info: dict = {"cutoff": cutoff, "h": {}}
    for h in range(1, H + 1):
        oi = calibration_origins(cutoffs, cutoff, h, K)
        n_samples = int(oi.size * raw.shape[1])
        entry = {"n": n_samples, "applied": False}
        if oi.size and n_samples >= min_n:
            past = raw[oi, :, h - 1, :]  # [m, n, 7]
            y = units_upto_cutoff[cutoffs[oi] + h]  # [m, n]
            resid = ((y[..., None] - past) / _scale(past, floor)[..., None]).reshape(-1, len(QUANTILES))
            delta = np.array([np.quantile(resid[:, j], _level(tau, resid.shape[0])) for j, tau in enumerate(QUANTILES)])
            out[:, h - 1, :] = raw[ci, :, h - 1, :] + delta[None, :] * _scale(raw[ci, :, h - 1, :], floor)[:, None]
            entry.update(applied=True, delta=[float(d) for d in delta], last_target_week=int((cutoffs[oi] + h).max()), first_target_week=int((cutoffs[oi] + h).min()))
        info["h"][h] = entry
    out = np.maximum.accumulate(np.clip(out, 0.0, None), axis=-1)
    info["applied"] = all(v["applied"] for v in info["h"].values())
    return out, info
