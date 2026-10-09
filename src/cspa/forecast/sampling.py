"""Joint demand path sampling.

Method (weekly residual-block bootstrap on the PIT scale):
  1. For every past forecast origin o whose targets are already realised, compute
     the PIT u = F_hat(y) of the realised demand under the forecast issued at o,
     for every SKU and horizon. That is the "residual bank": [origins, skus, H].
  2. To draw a path, pick a past origin and reuse its whole cross-section of PITs
     (all SKUs together), so cross-SKU dependence is whatever actually happened.
     Horizons are drawn in consecutive blocks of `block_len` weeks from the same
     origin, which also keeps some week-to-week persistence of forecast errors.
  3. Map each PIT back through today's predicted quantile function for that SKU
     and horizon, round to whole units, clip at zero.

`method="independent"` shuffles origins separately per SKU: identical marginals,
no cross-SKU correlation. It exists to show how much risk independence hides.
"""
from __future__ import annotations

import numpy as np

from cspa.config import QUANTILES

TAUS = np.array(QUANTILES)
KNOT_TAUS = np.concatenate([[0.0], TAUS])
_LN2 = float(np.log(2.0))
_TOP = float(TAUS[-1])


def _knots(q: np.ndarray) -> np.ndarray:
    """Quantile knots with an extra one at tau=0 (linear extrapolation, floored at 0)."""
    q0 = np.clip(2.0 * q[..., 0] - q[..., 1], 0.0, q[..., 0])
    return np.concatenate([q0[..., None], q], axis=-1)


def _tail_scale(q: np.ndarray) -> np.ndarray:
    """Exponential upper-tail scale implied by the q90 -> q95 gap, with a floor."""
    return np.maximum.reduce([q[..., -1] - q[..., -2], 0.1 * q[..., -1], np.full(q.shape[:-1], 0.25)]) / _LN2


def quantile_fn(q: np.ndarray, u: np.ndarray, tail_cap: float = 0.999) -> np.ndarray:
    """Evaluate each SKU's predicted quantile function. q [n, 7], u [m, n] -> [m, n]."""
    u = np.clip(u, 0.0, tail_cap)
    kn = _knots(q).T  # [8, n]
    j = np.clip(np.searchsorted(KNOT_TAUS, u, side="right") - 1, 0, len(KNOT_TAUS) - 2)
    lo, hi = np.take_along_axis(kn, j, axis=0), np.take_along_axis(kn, j + 1, axis=0)
    w = (u - KNOT_TAUS[j]) / (KNOT_TAUS[j + 1] - KNOT_TAUS[j])
    body = lo + np.clip(w, 0.0, 1.0) * (hi - lo)
    tail = q[:, -1][None, :] + _tail_scale(q)[None, :] * np.log((1.0 - _TOP) / np.maximum(1.0 - u, 1e-12))
    return np.where(u > _TOP, tail, body)


def pit_values(q: np.ndarray, y: np.ndarray, tail_cap: float = 0.999) -> np.ndarray:
    """PIT of realised y under the predicted quantile function. q [n, 7], y [n] -> u [n].

    Where the quantile function is flat at y (common for intermittent demand,
    e.g. several quantiles equal to 0), the mid-point of the flat range is used.
    """
    kn = _knots(q)  # [n, 8]
    n_lt = (kn < y[:, None]).sum(axis=1)
    n_le = (kn <= y[:, None]).sum(axis=1)
    last = kn.shape[1] - 1
    u = np.zeros(len(y))
    tail = n_lt == kn.shape[1]
    if tail.any():
        u[tail] = 1.0 - (1.0 - _TOP) * np.exp(-(y[tail] - q[tail, -1]) / _tail_scale(q[tail]))
    tie = (n_le > n_lt) & ~tail
    if tie.any():
        u[tie] = 0.5 * (KNOT_TAUS[n_lt[tie]] + KNOT_TAUS[n_le[tie] - 1])
    mid = (n_le == n_lt) & (n_lt > 0) & ~tail
    if mid.any():
        j = np.clip(n_lt[mid], 1, last)
        rows = np.flatnonzero(mid)
        a, b = kn[rows, j - 1], kn[rows, j]
        u[mid] = KNOT_TAUS[j - 1] + (KNOT_TAUS[j] - KNOT_TAUS[j - 1]) * (y[mid] - a) / np.maximum(b - a, 1e-12)
    return np.clip(u, 1.0 - tail_cap, tail_cap)


def build_pit_bank(cal: np.ndarray, cutoffs: np.ndarray, units_upto_cutoff: np.ndarray, cutoff: int, bank_weeks: int, tail_cap: float = 0.999) -> np.ndarray:
    """Residual bank for decisions made with data <= cutoff. Returns [K, n_skus, H].

    cal: calibrated forecast cache [C, n, H, 7] aligned with `cutoffs`.
    Only origins o with o + H <= cutoff are used, so every PIT is of an already
    realised week. units_upto_cutoff must be units[: cutoff + 1].
    """
    if units_upto_cutoff.shape[0] != cutoff + 1:
        raise ValueError("build_pit_bank must be given units up to and including the cutoff, nothing later")
    H = cal.shape[2]
    oi = np.flatnonzero(np.asarray(cutoffs) + H <= cutoff)[-bank_weeks:]
    bank = np.empty((oi.size, cal.shape[1], H))
    for k, i in enumerate(oi):
        o = int(cutoffs[i])
        for h in range(1, H + 1):
            bank[k, :, h - 1] = pit_values(cal[i, :, h - 1, :], units_upto_cutoff[o + h], tail_cap)
    return bank


def sample_demand(
    q: np.ndarray,
    bank: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
    method: str = "block",
    block_len: int = 2,
    tail_cap: float = 0.999,
) -> np.ndarray:
    """Draw joint demand paths. q [n_skus, H, 7], bank [K, n_skus, H] -> [n_paths, n_skus, H] whole units."""
    n, H, _ = q.shape
    K = bank.shape[0]
    u = np.empty((n_paths, n, H))
    if K == 0:  # no realised forecasts yet: independent uniforms (flagged by the caller)
        u = rng.uniform(0.0, 1.0, size=(n_paths, n, H))
    elif method == "independent":
        for h in range(H):
            k = rng.integers(0, K, size=(n_paths, n))
            u[:, :, h] = np.take_along_axis(bank[:, :, h], k, axis=0)
    elif method == "block":
        for start in range(0, H, block_len):
            k = rng.integers(0, K, size=n_paths)
            for h in range(start, min(start + block_len, H)):
                u[:, :, h] = bank[k, :, h]
    else:
        raise ValueError(f"unknown sampling method '{method}'")
    out = np.empty((n_paths, n, H))
    for h in range(H):
        out[:, :, h] = quantile_fn(q[:, h, :], u[:, :, h], tail_cap)
    return np.clip(np.rint(out), 0.0, None)


def deterministic_path(q: np.ndarray, tau: float) -> np.ndarray:
    """Single demand path at a fixed quantile for the deterministic gate. -> [1, n_skus, H]."""
    return np.rint(q[:, :, QUANTILES.index(tau)])[None, :, :]
