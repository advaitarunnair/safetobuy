"""Greedy marginal-value-per-dollar allocator.

For SKU i with inventory position x (on hand + on order) and demand D over the
cover period (lead time + review period), the (x+1)-th unit is worth

    MV_i(x) = Cu_i * P(D_i > x) - Co_i * P(D_i <= x)

Cu = price - cost (margin if it sells), Co = cost of a leftover unit. MV is
non-increasing in x, so funding units in order of MV / cost until the budget
runs out is optimal for the continuous relaxation. Units with MV <= 0 are
never funded; the point where MV turns non-positive is the unconstrained
newsvendor target.

Packs and MOQs (documented heuristic): units are bought in "chunks". A SKU's
first chunk is its MOQ, every later chunk is one pack. A chunk's value is the
sum of its units' marginal values and it is ranked by value / cost. Chunk
densities are non-increasing within a SKU, so one global sort respects
within-SKU order. After the greedy prefix, a repair pass keeps scanning and
funds any later chunk that still fits the leftover budget.

P(D > x) comes from the joint demand sample paths (the same ones the cash
simulation uses), summed over each SKU's cover period.

Ranking (rank_by):
  "cost"      : value / purchase cost. The formulation in the brief.
  "committed" : value / (purchase cost + cost of the units expected to be left
                unsold at the end of the cover period). A dollar spent on a unit
                that does not sell stays tied up for another period, so under a
                binding budget it is charged twice. This is the fixed point of
                charging leftovers the shadow price of cash: fund a chunk iff
                value - lambda * leftover_cost >= lambda * cost, i.e. iff
                value / (cost + leftover_cost) >= lambda. It only changes the
                ORDER in which units are funded, never which units are worth
                buying, so the unconstrained plan is identical.
  "cash"      : (value - cost of the units expected to be left unsold) / purchase
                cost: the expected cash profit a dollar returns within the cover
                period, counting an unsold unit as cash not yet recovered. This is
                the ranking a cash gate cares about: it puts units that will sell
                soon first, whatever their margin, and deep safety stock last.
                Like "committed" it changes the order only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ChunkTable:
    n_skus: int
    sku: np.ndarray  # [m] SKU index of each chunk, sorted by density (desc)
    qty: np.ndarray  # [m] units in the chunk
    cost: np.ndarray  # [m]
    value: np.ndarray  # [m] expected profit of the chunk
    density: np.ndarray  # [m] value / cost
    cum_cost: np.ndarray  # [m]
    suffix_min_cost: np.ndarray  # [m]
    position: np.ndarray  # [n] inventory position before ordering
    target_qty: np.ndarray  # [n] unconstrained newsvendor order (pack/MOQ feasible)
    cu: np.ndarray
    co: np.ndarray
    unit_cost: np.ndarray

    @property
    def total_cost(self) -> float:
        """Cost of the unconstrained plan (every positive-value chunk)."""
        return float(self.cum_cost[-1]) if self.cum_cost.size else 0.0


def cover_demand(paths: np.ndarray, cover_weeks: np.ndarray) -> np.ndarray:
    """Sum each SKU's sampled demand over its own cover period. paths [P, n, H] -> [P, n]."""
    cum = np.cumsum(paths, axis=2)
    idx = np.clip(np.asarray(cover_weeks, dtype=np.int64) - 1, 0, paths.shape[2] - 1)
    return np.take_along_axis(cum, idx[None, :, None], axis=2)[:, :, 0]


def survival(sorted_cover: np.ndarray, x0: int, n_units: int) -> np.ndarray:
    """P(D > x) for x = x0 .. x0 + n_units - 1 given sorted demand samples."""
    xs = np.arange(x0, x0 + n_units)
    return 1.0 - np.searchsorted(sorted_cover, xs, side="right") / len(sorted_cover)


def marginal_values(sorted_cover: np.ndarray, x0: int, n_units: int, cu: float, co: float) -> np.ndarray:
    """MV of units x0+1 .. x0+n_units given sorted demand samples."""
    surv = survival(sorted_cover, x0, n_units)
    return cu * surv - co * (1.0 - surv)


def build_chunks(cover: np.ndarray, position: np.ndarray, unit_cost: np.ndarray, cu: np.ndarray, co: np.ndarray, pack: np.ndarray, moq: np.ndarray, rank_by: str = "cost") -> ChunkTable:
    """Enumerate every positive-value chunk for every SKU and sort by value per dollar (see rank_by)."""
    if rank_by not in ("cost", "committed", "cash"):
        raise ValueError(f"unknown rank_by '{rank_by}'")
    n = cover.shape[1]
    position = np.asarray(np.rint(position), dtype=np.int64)
    srt = np.sort(cover, axis=0)
    c_sku, c_qty, c_val, c_key = [], [], [], []
    target = np.zeros(n, dtype=np.int64)
    for i in range(n):
        x0 = int(position[i])
        x_max = int(np.ceil(srt[-1, i]))
        if cu[i] <= 0 or x_max <= x0:
            continue
        surv = survival(srt[:, i], x0, x_max - x0)
        mv = float(cu[i]) * surv - float(co[i]) * (1.0 - surv)
        n_pos = int((mv > 0).sum())
        if n_pos == 0:
            continue
        first, pk = int(max(moq[i], pack[i])), int(pack[i])
        n_chunks = 1 + int(np.ceil(max(n_pos - first, 0) / pk))
        ends = first + pk * np.arange(n_chunks)
        starts = np.concatenate([[0], ends[:-1]])
        if ends[-1] > mv.size:  # units beyond the largest sampled demand never sell
            pad = ends[-1] - mv.size
            mv = np.concatenate([mv, np.full(pad, -float(co[i]))])
            surv = np.concatenate([surv, np.zeros(pad)])
        csum = np.concatenate([[0.0], np.cumsum(mv)])
        vals = csum[ends] - csum[starts]
        lsum = np.concatenate([[0.0], np.cumsum(1.0 - surv)])
        left = lsum[ends] - lsum[starts]  # expected unsold units in each chunk
        pos = vals > 1e-12
        n_keep = int(pos.size if pos.all() else np.argmin(pos))
        if n_keep == 0:
            continue
        sizes = (ends - starts)[:n_keep]
        c_sku.append(np.full(n_keep, i, dtype=np.int64))
        c_qty.append(sizes)
        c_val.append(vals[:n_keep])
        # Sort key: density made exactly non-increasing within the SKU, so floating-point
        # ties (e.g. a 2-unit MOQ chunk vs. the single units after it) can never reorder a SKU's chunks.
        spend = sizes * float(unit_cost[i])
        unsold = left[:n_keep] * float(unit_cost[i])
        if rank_by == "committed":
            score = vals[:n_keep] / (spend + unsold)
        elif rank_by == "cash":
            score = (vals[:n_keep] - unsold) / spend
        else:
            score = vals[:n_keep] / spend
        c_key.append(np.minimum.accumulate(score))
        target[i] = int(ends[n_keep - 1])

    if c_sku:
        sku, qty, val, key = np.concatenate(c_sku), np.concatenate(c_qty).astype(np.int64), np.concatenate(c_val), np.concatenate(c_key)
    else:
        sku, qty, val, key = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    cost = qty * unit_cost[sku]
    order = np.lexsort((np.arange(sku.size), -key))  # ties keep within-SKU order
    sku, qty, val, cost, density = sku[order], qty[order], val[order], cost[order], key[order]
    suffix_min = np.minimum.accumulate(cost[::-1])[::-1] if cost.size else cost
    return ChunkTable(n, sku, qty, cost, val, density, np.cumsum(cost), suffix_min, position, target, np.asarray(cu, float), np.asarray(co, float), np.asarray(unit_cost, float))


def allocate(table: ChunkTable, budget: float) -> np.ndarray:
    """Fund chunks greedily by value per dollar within `budget`. Returns whole units per SKU."""
    out = np.zeros(table.n_skus, dtype=np.int64)
    m = table.sku.size
    if m == 0 or budget <= 0:
        return out
    eps = 1e-9
    k = int(np.searchsorted(table.cum_cost, budget + eps, side="right"))
    np.add.at(out, table.sku[:k], table.qty[:k])
    if k >= m:
        return out
    remaining = budget - (table.cum_cost[k - 1] if k else 0.0)
    blocked: set[int] = set()  # a SKU's later chunks need its earlier ones
    for j in range(k, m):
        if remaining + eps < table.suffix_min_cost[j]:
            break
        s = int(table.sku[j])
        if s in blocked:
            continue
        if table.cost[j] <= remaining + eps:
            out[s] += table.qty[j]
            remaining -= table.cost[j]
        else:
            blocked.add(s)
    return out


def plan_report(skus: list[str], qty: np.ndarray, table: ChunkTable, cover: np.ndarray, cover_weeks: np.ndarray) -> pd.DataFrame:
    """Per-SKU decision facts for any plan, measured against the newsvendor target.

    decision: full    = reached the unconstrained newsvendor target
              partial = bought something, but less than the target
              defer   = target > 0 and nothing bought
    """
    qty = np.asarray(qty, dtype=np.int64)
    pos, tgt = table.position, table.target_qty
    after = pos + qty
    full_pos = pos + np.maximum(tgt, qty)
    short_after = np.maximum(cover - after[None, :], 0.0).mean(axis=0)
    short_full = np.maximum(cover - full_pos[None, :], 0.0).mean(axis=0)
    mean_week = cover.mean(axis=0) / np.maximum(cover_weeks, 1)
    decision = np.where(qty >= tgt, "full", np.where(qty > 0, "partial", "defer"))

    last_density = np.full(table.n_skus, np.nan)
    next_density = np.full(table.n_skus, np.nan)
    funded_units = np.zeros(table.n_skus, dtype=np.int64)
    for j in range(table.sku.size):  # chunks of one SKU appear in within-SKU order
        s = table.sku[j]
        if funded_units[s] + table.qty[j] <= qty[s]:
            funded_units[s] += table.qty[j]
            last_density[s] = table.density[j]
        elif np.isnan(next_density[s]):
            next_density[s] = table.density[j]

    return pd.DataFrame(
        {
            "sku": skus,
            "decision": decision,
            "needed": tgt > 0,
            "qty": qty,
            "target_qty": tgt,
            "spend": qty * table.unit_cost,
            "target_spend": tgt * table.unit_cost,
            "position": pos,
            "mv_per_dollar_last": last_density,
            "mv_per_dollar_next": next_density,
            "stockout_prob": (cover > after[None, :]).mean(axis=0),
            "stockout_prob_if_full": (cover > full_pos[None, :]).mean(axis=0),
            "days_cover": 7.0 * after / np.maximum(mean_week, 1e-9),
            "exp_margin_lost": table.cu * np.maximum(short_after - short_full, 0.0),
            "unit_margin": table.cu,
            "unit_cost": table.unit_cost,
            "exp_weekly_demand": mean_week,
        }
    )
