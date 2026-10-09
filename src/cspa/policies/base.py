"""Policy interface, the information set policies are allowed to see, and shared building blocks.

Every policy is a (budget rule, allocation rule) pair:

    policy         budget rule     allocation rule
    A              none            order-up-to need
    B              open-to-buy     proportional to A's need
    C (proposed)   cash gate       marginal value per dollar
    D              open-to-buy     full need in priority order
    C_gate_prop    cash gate       proportional to A's need     (ablation)
    OTB_marginal   open-to-buy     marginal value per dollar    (ablation)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np

from cspa.allocate.constraints import round_down_to_pack, round_up_to_pack
from cspa.allocate.marginal import ChunkTable, allocate, build_chunks, cover_demand
from cspa.cash.gate import BudgetResult, safe_budget
from cspa.cash.simulate import CashPaths, simulate_cash
from cspa.config import QUANTILES
from cspa.data.synth_params import SkuParams
from cspa.forecast.sampling import deterministic_path
from cspa.sim.world import Orders, WorldState

_Q10, _Q50, _Q90 = QUANTILES.index(0.10), QUANTILES.index(0.50), QUANTILES.index(0.90)
_Z80 = 2.5631  # width of a normal 10%-90% interval in standard deviations


@dataclass
class InfoSet:
    """Everything a policy may know when deciding for `week`. Nothing here is dated after `cutoff`,
    except the fixed-cost schedule and the calendar, which a shop knows in advance."""

    week: int  # the week being decided
    cutoff: int  # last observed week (= week - 1)
    q: np.ndarray  # [n, H, 7] calibrated quantile forecast for weeks cutoff+1 .. cutoff+H
    price: np.ndarray  # [n] last known sell price
    fixed_costs: np.ndarray  # [H] fixed costs falling due in weeks week .. week+H-1
    buffer: float
    alpha: float
    params: SkuParams
    cfg: object
    paths_fn: Callable[[], np.ndarray]  # lazily samples joint demand paths [P, n, H]

    @cached_property
    def demand_paths(self) -> np.ndarray:
        return self.paths_fn()

    @cached_property
    def det_path(self) -> np.ndarray:
        return deterministic_path(self.q, float(self.cfg.gate.deterministic_quantile))

    @cached_property
    def cover_weeks(self) -> np.ndarray:
        return self.params.lead_time + int(self.cfg.horizons.review_period_weeks)

    @cached_property
    def cover(self) -> np.ndarray:
        """[P, n] sampled demand over each SKU's cover period."""
        return cover_demand(self.demand_paths, self.cover_weeks)

    @property
    def sim_paths(self) -> np.ndarray:
        return self.det_path if self.cfg.gate.mode == "deterministic" else self.demand_paths


class PlanContext:
    """Quantities shared by the budget and allocation rules for one (state, info) pair."""

    def __init__(self, state: WorldState, info: InfoSet):
        self.state, self.info = state, info
        p = info.params
        self.p = p
        self.position = state.inventory_position()
        H = info.q.shape[1]
        hmask = np.arange(H)[None, :] < info.cover_weeks[:, None]
        q50 = info.q[:, :, _Q50]
        sig = (info.q[:, :, _Q90] - info.q[:, :, _Q10]) / _Z80
        self.mu = (q50 * hmask).sum(axis=1)  # median forecast over the cover period
        self.sigma = np.sqrt(((sig * hmask) ** 2).sum(axis=1))
        self.weekly = q50.mean(axis=1)
        self.cu = info.price - p.unit_cost
        self.co = p.overage_cost(info.cfg)

    @cached_property
    def need(self) -> np.ndarray:
        """Policy A's order: up to (forecast over cover period + z * sigma), rounded up to packs / MOQ."""
        z = float(self.info.cfg.policies.reorder_point.z)
        target = self.mu + z * self.sigma
        return round_up_to_pack(np.maximum(target - self.position, 0.0), self.p.pack_size, self.p.moq)

    @cached_property
    def need_cost(self) -> np.ndarray:
        return self.need * self.p.unit_cost

    @cached_property
    def otb_weeks_cover(self) -> float:
        """Planned end-of-period stock, in weeks of planned sales. 'auto' matches policy A's total
        safety stock at cost, so A and B aim for the same amount of stock and differ only in how:
        A per SKU, B as one budget."""
        w = self.info.cfg.policies.otb.target_weeks_cover
        if w != "auto":
            return float(w)
        z = float(self.info.cfg.policies.reorder_point.z)
        denom = float(self.weekly @ self.p.unit_cost)
        return float(z * (self.sigma @ self.p.unit_cost) / denom) if denom > 0 else 0.0

    @cached_property
    def otb_budget(self) -> float:
        """Open-to-buy at cost:
        planned sales + planned markdowns + planned end-of-period stock - beginning stock - on order."""
        otb = self.info.cfg.policies.otb
        planned_sales = self.mu
        planned_end = self.otb_weeks_cover * self.weekly
        units = planned_sales + planned_end - self.position  # position = on hand + on order
        return max(0.0, float(units @ self.p.unit_cost) + float(otb.planned_markdowns))

    @cached_property
    def table(self) -> ChunkTable:
        return build_chunks(self.info.cover, self.position, self.p.unit_cost, self.cu, self.co, self.p.pack_size, self.p.moq, rank_by=str(self.info.cfg.allocator.rank_by))

    @cached_property
    def stockout_prob_now(self) -> np.ndarray:
        """P(demand over the cover period exceeds the current inventory position)."""
        return (self.info.cover > self.position[None, :]).mean(axis=0)

    def simulate(self, qty: np.ndarray, paths: np.ndarray | None = None) -> CashPaths:
        return simulate_cash(self.state, qty, self.info.sim_paths if paths is None else paths, self.info.price, self.p, self.info.fixed_costs, str(self.info.cfg.gate.continuation), bool(self.info.cfg.gate.charge_beyond_horizon))

    # ---- allocation rules: budget -> whole units per SKU ----
    def alloc_need(self) -> Callable[[float], np.ndarray]:
        return lambda B: self.need.copy()

    def alloc_proportional(self) -> Callable[[float], np.ndarray]:
        total = float(self.need_cost.sum())

        def f(B: float) -> np.ndarray:
            if total <= 0:
                return np.zeros(self.p.n, dtype=np.int64)
            scale = min(1.0, max(B, 0.0) / total)  # the budget is a cap, never a target
            return round_down_to_pack(self.need * scale, self.p.pack_size, self.p.moq)

        return f

    def alloc_priority(self) -> Callable[[float], np.ndarray]:
        score = self.stockout_prob_now * np.maximum(self.cu, 0.0)
        idx = np.flatnonzero(self.need > 0)
        order = idx[np.lexsort((idx, -score[idx]))]
        cum = np.cumsum(self.need_cost[order])

        def f(B: float) -> np.ndarray:
            out = np.zeros(self.p.n, dtype=np.int64)
            k = int(np.searchsorted(cum, B + 1e-9, side="right"))
            out[order[:k]] = self.need[order[:k]]
            if k < order.size:  # the first SKU that does not fit gets what is left
                s = order[k]
                rem = B - (cum[k - 1] if k else 0.0)
                out[s] = round_down_to_pack(np.array([rem / self.p.unit_cost[s]]), self.p.pack_size[s : s + 1], self.p.moq[s : s + 1])[0]
            return out

        return f

    def alloc_marginal(self) -> Callable[[float], np.ndarray]:
        return lambda B: allocate(self.table, B)

    def allocator(self, rule: str) -> Callable[[float], np.ndarray]:
        rules = {"need": self.alloc_need, "proportional": self.alloc_proportional, "priority": self.alloc_priority, "marginal": self.alloc_marginal}
        return rules[rule]()

    # ---- budget rules ----
    def gate(self, alloc: Callable[[float], np.ndarray], alpha: float | None = None, buffer: float | None = None, on_infeasible: str | None = None) -> BudgetResult:
        g = self.info.cfg.gate
        buf = self.info.buffer if buffer is None else buffer
        a = self.info.alpha if alpha is None else alpha
        full_cost = float(alloc(np.inf) @ self.p.unit_cost)
        sims: dict[float, CashPaths] = {}

        def sim(B: float) -> CashPaths:
            if B not in sims:
                sims[B] = self.simulate(alloc(B))
            return sims[B]

        return safe_budget(
            lambda B: sim(B).prob_above(buf),
            B_max=full_cost,
            alpha=a,
            tol_frac=float(g.tol_frac),
            tol_abs=float(g.tol_abs),
            max_iter=int(g.max_bisection_iter),
            n_grid=int(g.n_grid),
            on_infeasible=str(g.on_infeasible if on_infeasible is None else on_infeasible),
            tail=lambda B: (float(np.quantile(sim(B).min_cash, a)), float(np.quantile(sim(B).cash[:, -1], a))),
            n_refine=int(g.n_refine),
        )


class Policy:
    """decide(state, info) -> Orders. Subclasses only choose the two rules."""

    name = "base"
    label = ""
    budget_rule = "none"  # none | otb | gate
    alloc_rule = "need"  # need | proportional | priority | marginal
    on_infeasible: str | None = None  # None = use gate.on_infeasible from config

    def decide(self, state: WorldState, info: InfoSet) -> Orders:
        ctx = PlanContext(state, info)
        alloc = ctx.allocator(self.alloc_rule)
        meta: dict = {"policy": self.name, "budget_rule": self.budget_rule, "alloc_rule": self.alloc_rule}
        if self.budget_rule == "none":
            budget = np.inf
        elif self.budget_rule == "otb":
            budget = ctx.otb_budget
        elif self.budget_rule == "gate":
            res = ctx.gate(alloc, on_infeasible=self.on_infeasible)
            budget = res.B
            meta["gate"] = res
        else:
            raise ValueError(f"unknown budget rule '{self.budget_rule}'")
        qty = np.asarray(alloc(budget), dtype=np.int64)
        meta["budget"] = float(budget)
        meta["plan_cost"] = float(qty @ ctx.p.unit_cost)
        return Orders(qty, meta)
