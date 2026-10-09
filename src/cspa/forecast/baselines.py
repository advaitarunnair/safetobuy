"""Naive and seasonal-naive baselines, turned into quantile forecasters.

Quantiles = point forecast + empirical quantiles of that method's own past
errors for the same SKU and horizon, using only weeks <= cutoff.
"""
from __future__ import annotations

import numpy as np

from cspa.config import QUANTILES
from cspa.data.features import SEASON
from cspa.data.load_m5 import Panel
from cspa.forecast.base import QuantileForecaster, enforce_monotone


class NaiveForecaster(QuantileForecaster):
    """Point forecast = last observed week."""

    name = "naive"

    def fit(self, history: Panel, cutoff_week: int) -> None:
        self._hist(history, cutoff_week)
        self.history, self.fit_cutoff = history, cutoff_week

    def predict_array(self, cutoff_week: int, horizon_weeks: int, history: Panel | None = None) -> np.ndarray:
        hist = self._hist(history, cutoff_week)
        u = hist.units[: cutoff_week + 1]
        k = int(self.cfg.forecast.baseline_residual_weeks)
        out = np.empty((hist.n_skus, horizon_weeks, len(QUANTILES)))
        point = u[cutoff_week]
        for h in range(1, horizon_weeks + 1):
            tgt = np.arange(max(h, cutoff_week - k + 1), cutoff_week + 1)
            resid = u[tgt] - u[tgt - h] if tgt.size else np.zeros((1, hist.n_skus))
            out[:, h - 1, :] = (point[None, :] + np.quantile(resid, QUANTILES, axis=0)).T
        return enforce_monotone(out)


class SeasonalNaiveForecaster(QuantileForecaster):
    """Point forecast = the same Walmart week one year (52 weeks) earlier."""

    name = "seasonal_naive"

    def fit(self, history: Panel, cutoff_week: int) -> None:
        self._hist(history, cutoff_week)
        self.history, self.fit_cutoff = history, cutoff_week

    def predict_array(self, cutoff_week: int, horizon_weeks: int, history: Panel | None = None) -> np.ndarray:
        hist = self._hist(history, cutoff_week)
        if cutoff_week + 1 < SEASON:
            raise ValueError("seasonal naive needs at least 52 observed weeks")
        u = hist.units[: cutoff_week + 1]
        k = int(self.cfg.forecast.baseline_residual_weeks)
        tgt = np.arange(max(SEASON, cutoff_week - k + 1), cutoff_week + 1)
        resid = u[tgt] - u[tgt - SEASON] if tgt.size else np.zeros((1, hist.n_skus))
        rq = np.quantile(resid, QUANTILES, axis=0).T
        out = np.empty((hist.n_skus, horizon_weeks, len(QUANTILES)))
        for h in range(1, horizon_weeks + 1):
            out[:, h - 1, :] = u[cutoff_week + h - SEASON][:, None] + rq
        return enforce_monotone(out)


def make_forecaster(name: str, cfg) -> QuantileForecaster:
    if name == "naive":
        return NaiveForecaster(cfg)
    if name == "seasonal_naive":
        return SeasonalNaiveForecaster(cfg)
    if name == "lgbm":
        from cspa.forecast.lgbm_quantile import LGBMQuantileForecaster

        return LGBMQuantileForecaster(cfg)
    raise ValueError(f"unknown forecaster '{name}'")
