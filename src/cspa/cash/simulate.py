"""Monte Carlo cash projection for a candidate purchase plan.

Weekly recursion on every demand path, in the same order as sim/world.py:

    cash_t = cash_{t-1} + revenue_t - fixed_costs_t - supplier_payments_t

Revenue is sampled demand capped by available stock (sales <= stock), so the
plan feeds back into cash: bought units can only earn revenue after they
arrive, and their payable lands on its due week.

Purchases after this week. The projection covers what is already committed
(stock, pipeline, payables) plus the candidate plan. For the later weeks of the
horizon it needs an assumption about future orders:

  continuation = "replace_sales" (default): in each later week the shop orders
      what it sold the week before (one-for-one replenishment), with the usual
      lead times and payment terms. Stock levels, revenue and supplier payments
      then keep flowing through the horizon, so the projection is a cash runway.
  continuation = "replace_sales_capped": the same, except that a later order is scaled
      down, path by path, to the cash then available above `cash_floor` (the buffer).
      A shop that runs this gate every week will not, next week, spend cash it does not
      have. Without the cap the projection blames this week's order for breaches that
      next week's gate would simply prevent, and it turned out far too pessimistic.
  continuation = "none": no further purchases. Stock runs out inside the
      horizon, revenue collapses while fixed costs continue, and almost any plan
      looks unsafe. Kept for comparison and tests only.

Future orders are still re-decided, and re-gated, when their week arrives.

Bills beyond the horizon. If part of the candidate plan fell due after the last
projected week, buying on those terms would look free to the gate. The config
validator therefore requires every bill of this week's order to fall inside the
cash horizon (max lead time + max payment delay < horizon). The alternative,
charge_beyond_horizon=True, counts such bills in the last projected week; it is
available but double-counts against older bills still in the pipeline, so it is
pessimistic for a shop in steady state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cspa.data.synth_params import SkuParams
from cspa.sim.world import WorldState


@dataclass
class CashPaths:
    cash: np.ndarray  # [n_paths, H] end-of-week cash
    min_cash: np.ndarray  # [n_paths]
    margin: np.ndarray  # [n_paths] gross margin earned over the horizon
    revenue: np.ndarray  # [n_paths]
    plan_cost: float

    def prob_above(self, buffer: float) -> float:
        """P(min_t cash_t >= buffer) across paths."""
        return float(np.mean(self.min_cash >= buffer))

    def quantiles(self, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> np.ndarray:
        """[len(qs), H] cash quantiles per horizon week (for the fan chart)."""
        return np.quantile(self.cash, qs, axis=0)


def simulate_cash(
    state: WorldState,
    plan_qty: np.ndarray,
    demand_paths: np.ndarray,
    price: np.ndarray,
    params: SkuParams,
    fixed_costs: np.ndarray,
    continuation: str = "replace_sales",
    charge_beyond_horizon: bool = False,
    cash_floor: float | None = None,
) -> CashPaths:
    """Project cash over H weeks. demand_paths [n_paths, n_skus, H]; fixed_costs [H].

    cash_floor is only used by continuation="replace_sales_capped"."""
    if continuation not in ("replace_sales", "replace_sales_capped", "none"):
        raise ValueError(f"unknown continuation '{continuation}'")
    capped = continuation == "replace_sales_capped"
    if capped and cash_floor is None:
        raise ValueError("continuation='replace_sales_capped' needs cash_floor")
    n_paths, n, H = demand_paths.shape
    plan = np.asarray(plan_qty, dtype=float)
    c = params.unit_cost
    on_hand = np.broadcast_to(state.on_hand.astype(float), (n_paths, n)).copy()
    cash = np.full(n_paths, float(state.cash))

    pay = np.zeros((n_paths, H))
    k = min(H, len(state.payables))
    pay[:, :k] = state.payables[:k]
    due = params.lead_time + params.pay_delay
    plan_due = np.zeros(H)
    if charge_beyond_horizon:
        np.add.at(plan_due, np.minimum(due, H - 1), plan * c)
    else:
        inside = due < H
        np.add.at(plan_due, due[inside], (plan * c)[inside])
    pay += plan_due[None, :]

    follow = continuation != "none"
    if follow:
        later = np.zeros((n_paths, n, H))  # arrivals from orders placed in later weeks
        groups = [(int(L), int(D), (params.lead_time == L) & (due == D)) for L, D in sorted({(int(a), int(b)) for a, b in zip(params.lead_time, due)})]

    unit_margin = price - c
    perish = bool(params.spoilage_rate.any())
    loss_per_unit = c - params.salvage_value
    out = np.empty((n_paths, H))
    margin = np.zeros(n_paths)
    revenue = np.zeros(n_paths)
    for j in range(H):
        arrive = np.where(params.lead_time == j, plan, 0.0)
        if j < state.pipeline.shape[1]:
            arrive = arrive + state.pipeline[:, j]
        on_hand += arrive
        if follow:
            on_hand += later[:, :, j]
        sales = np.minimum(demand_paths[:, :, j], on_hand)
        on_hand -= sales
        rev = sales @ price
        margin += sales @ unit_margin
        if perish:
            spoiled = on_hand * params.spoilage_rate
            on_hand -= spoiled
            rev = rev + spoiled @ params.salvage_value
            margin -= spoiled @ loss_per_unit
        cash = cash + rev - fixed_costs[j] - pay[:, j]
        revenue += rev
        out[:, j] = cash
        if follow and j + 1 < H:  # next week's order replaces this week's sales
            scale = None
            if capped:  # ... as far as the cash above the floor allows, on each path
                want = sales @ c
                scale = np.clip((cash - cash_floor) / np.maximum(want, 1e-9), 0.0, 1.0)
            for L, D, mask in groups:
                units = sales[:, mask] if scale is None else sales[:, mask] * scale[:, None]
                if j + 1 + L < H:
                    later[:, mask, j + 1 + L] += units
                if j + 1 + D < H:
                    pay[:, j + 1 + D] += units @ c[mask]
    return CashPaths(out, out.min(axis=1), margin, revenue, float(plan @ c))
