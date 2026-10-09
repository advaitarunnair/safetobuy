"""LightGBM quantile forecaster: one pooled model per quantile.

Pooled across SKUs and horizons (direct multi-horizon, h is a feature). Targets
and lag features are divided by each SKU's trailing 13-week mean so SKUs of
very different volume share one model; predictions are scaled back.
"""
from __future__ import annotations

import numpy as np

from cspa.config import QUANTILES
from cspa.data.features import FEATURE_NAMES, SEASON, assemble, origin_block
from cspa.data.load_m5 import Panel
from cspa.forecast.base import QuantileForecaster, enforce_monotone

_LIBOMP_HELP = (
    "LightGBM could not load its OpenMP runtime (libomp).\n"
    "  macOS fix A: run through the Makefile (`make forecast`), which points\n"
    "               DYLD_FALLBACK_LIBRARY_PATH at the libomp bundled with scikit-learn.\n"
    "  macOS fix B: brew install libomp\n"
    "  Linux:       apt-get install libgomp1"
)


def import_lightgbm():
    try:
        import lightgbm as lgb
    except OSError as exc:  # missing libomp on macOS
        raise RuntimeError(f"{_LIBOMP_HELP}\nOriginal error: {str(exc)[:200]}") from exc
    return lgb


class LGBMQuantileForecaster(QuantileForecaster):
    name = "lgbm"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.models: dict[float, object] = {}
        self.n_train_rows = 0

    def _params(self, alpha: float) -> dict:
        p = self.cfg.forecast.lgbm
        return {
            "objective": "quantile",
            "alpha": float(alpha),
            "learning_rate": float(p.learning_rate),
            "num_leaves": int(p.num_leaves),
            "min_data_in_leaf": int(p.min_data_in_leaf),
            "feature_fraction": float(p.feature_fraction),
            "bagging_fraction": float(p.bagging_fraction),
            "bagging_freq": int(p.bagging_freq),
            "lambda_l2": float(p.lambda_l2),
            "num_threads": int(p.num_threads),
            "seed": int(self.cfg.seed),
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }

    def training_frame(self, history: Panel, cutoff_week: int):
        H = int(self.cfg.horizons.forecast_horizon_weeks)
        window = int(self.cfg.forecast.train_window_weeks)
        block = origin_block(history, cutoff_week)
        Xs, ys = [], []
        for h in range(1, H + 1):
            hi = cutoff_week - h
            lo = max(SEASON, hi - window + 1)
            if hi - lo + 1 < int(self.cfg.forecast.min_train_weeks):
                raise ValueError(
                    f"Not enough history to train at cutoff {cutoff_week}: {hi - lo + 1} origin weeks "
                    f"for h={h}, need forecast.min_train_weeks={self.cfg.forecast.min_train_weeks}"
                )
            X, _, y = assemble(history, block, np.arange(lo, hi + 1), h, cutoff_week, with_target=True)
            Xs.append(X)
            ys.append(y)
        X, y = np.concatenate(Xs), np.concatenate(ys)
        keep = (X[:, FEATURE_NAMES.index("listed")] > 0) & np.isfinite(y)
        return X[keep], y[keep]

    def fit(self, history: Panel, cutoff_week: int) -> None:
        lgb = import_lightgbm()
        self._hist(history, cutoff_week)
        X, y = self.training_frame(history, cutoff_week)
        self.n_train_rows = int(len(y))
        self.models = {}
        for q in QUANTILES:
            ds = lgb.Dataset(X, label=y, feature_name=list(FEATURE_NAMES), free_raw_data=False)
            self.models[q] = lgb.train(self._params(q), ds, num_boost_round=int(self.cfg.forecast.lgbm.n_estimators))
        self.history, self.fit_cutoff = history, cutoff_week

    def predict_array(self, cutoff_week: int, horizon_weeks: int, history: Panel | None = None) -> np.ndarray:
        hist = self._hist(history, cutoff_week)
        if not self.models:
            raise RuntimeError("call fit() before predict()")
        if self.fit_cutoff is not None and cutoff_week < self.fit_cutoff:
            raise ValueError("predict cutoff is earlier than the fit cutoff (the model has seen later data)")
        block = origin_block(hist, cutoff_week)
        out = np.empty((hist.n_skus, horizon_weeks, len(QUANTILES)))
        for h in range(1, horizon_weeks + 1):
            X, scale, _ = assemble(hist, block, np.array([cutoff_week]), h, cutoff_week, with_target=False)
            for j, q in enumerate(QUANTILES):
                out[:, h - 1, j] = self.models[q].predict(X, num_threads=int(self.cfg.forecast.lgbm.num_threads)) * scale
        return enforce_monotone(out)

    def feature_importance(self) -> dict[str, float]:
        if not self.models:
            return {}
        gain = np.mean([m.feature_importance(importance_type="gain") for m in self.models.values()], axis=0)
        total = gain.sum() or 1.0
        return {name: float(g / total) for name, g in zip(FEATURE_NAMES, gain)}
