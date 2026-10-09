"""Common interface for quantile forecasters."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from cspa.config import QCOLS
from cspa.data.load_m5 import Panel


def enforce_monotone(arr: np.ndarray) -> np.ndarray:
    """Clip at zero and sort along the quantile axis so quantiles never cross."""
    return np.sort(np.clip(arr, 0.0, None), axis=-1)


def array_to_frame(skus: list[str], arr: np.ndarray) -> pd.DataFrame:
    """[n_skus, H, 7] -> long frame with columns sku, h, q05..q95."""
    n, H, _ = arr.shape
    df = pd.DataFrame(arr.reshape(n * H, -1), columns=list(QCOLS))
    df.insert(0, "h", np.tile(np.arange(1, H + 1), n))
    df.insert(0, "sku", np.repeat(np.array(skus, dtype=object), H))
    return df


class QuantileForecaster(ABC):
    """fit() on data up to a cutoff week, predict() weeks cutoff+1 .. cutoff+H.

    A cutoff week is the last observed week. Nothing after it may be read.
    """

    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.history: Panel | None = None
        self.fit_cutoff: int | None = None

    @abstractmethod
    def fit(self, history: Panel, cutoff_week: int) -> None: ...

    @abstractmethod
    def predict_array(self, cutoff_week: int, horizon_weeks: int, history: Panel | None = None) -> np.ndarray:
        """Return [n_skus, horizon_weeks, 7], non-crossing and non-negative."""

    def predict(self, cutoff_week: int, horizon_weeks: int, history: Panel | None = None) -> pd.DataFrame:
        hist = history if history is not None else self.history
        return array_to_frame(hist.skus, self.predict_array(cutoff_week, horizon_weeks, history))

    def _hist(self, history: Panel | None, cutoff_week: int) -> Panel:
        hist = history if history is not None else self.history
        if hist is None:
            raise RuntimeError("call fit() before predict()")
        if cutoff_week >= hist.n_obs:
            raise ValueError(f"cutoff {cutoff_week} is beyond the last observed week {hist.n_obs - 1}")
        return hist
