"""Pack-size and minimum-order-quantity handling.

Orders must be whole packs, and a SKU is either not ordered or ordered in at
least its MOQ (the MOQ is itself a multiple of the pack size).
"""
from __future__ import annotations

import numpy as np


def round_up_to_pack(need: np.ndarray, pack: np.ndarray, moq: np.ndarray) -> np.ndarray:
    """Smallest feasible order covering `need` (0 stays 0)."""
    need = np.maximum(np.asarray(need, dtype=float), 0.0)
    q = (np.ceil(need / pack - 1e-9) * pack).astype(np.int64)
    return np.where(q > 0, np.maximum(q, moq), 0).astype(np.int64)


def round_down_to_pack(amount: np.ndarray, pack: np.ndarray, moq: np.ndarray) -> np.ndarray:
    """Largest feasible order not exceeding `amount` (below MOQ becomes 0)."""
    amount = np.maximum(np.asarray(amount, dtype=float), 0.0)
    q = (np.floor(amount / pack + 1e-9) * pack).astype(np.int64)
    return np.where(q >= moq, q, 0).astype(np.int64)


def satisfies(qty: np.ndarray, pack: np.ndarray, moq: np.ndarray) -> bool:
    qty = np.asarray(qty)
    if (qty < 0).any() or not np.array_equal(qty, np.rint(qty)):
        return False
    q = np.rint(qty).astype(np.int64)
    return bool(((q % pack == 0) & ((q == 0) | (q >= moq))).all())
