"""The simulated retailer.

Timing. A decision for week t is made before week t's demand is known, using
information up to week t-1. `World.step` then applies, in this fixed order:

  1. receive arrivals scheduled for week t
  2. realise sales = min(demand, stock); unmet demand is lost, not backordered
     (2b. perishable SKUs write off a share of what is left, at salvage value)
  3. receive revenue
  4. pay supplier payables due this week, and fixed costs
  5. place the new orders (arrive at the start of week t + lead time;
     the payable falls due `pay_delay` weeks after delivery)

Cash-shortfall rule (identical for every policy, set by world.shortfall_rule):
  overdraft        : orders are never cancelled; cash may go negative.
  cut_proportional : at placement, if the order's cost exceeds the cash on hand,
                     all lines are scaled down by the same factor (rounded down
                     to packs, lines below MOQ dropped).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cspa.data.synth_params import SkuParams


@dataclass
class WorldState:
    week: int  # the week about to be simulated
    cash: float
    on_hand: np.ndarray  # [n] units, before this week's arrivals
    pipeline: np.ndarray  # [n, P] units arriving at the start of week + k
    payables: np.ndarray  # [Q] dollars falling due in week + k
    spoil_acc: np.ndarray  # [n] fractional spoilage carried between weeks

    def copy(self) -> "WorldState":
        return WorldState(self.week, float(self.cash), self.on_hand.copy(), self.pipeline.copy(), self.payables.copy(), self.spoil_acc.copy())

    def inventory_position(self) -> np.ndarray:
        return self.on_hand + self.pipeline.sum(axis=1)


@dataclass
class Orders:
    qty: np.ndarray  # [n] whole units
    meta: dict = field(default_factory=dict)


class World:
    def __init__(self, params: SkuParams, shortfall_rule: str = "overdraft"):
        if shortfall_rule not in ("overdraft", "cut_proportional"):
            raise ValueError(f"unknown shortfall rule '{shortfall_rule}'")
        if params.lead_time.min() < 1:
            raise ValueError("lead times must be >= 1 week")
        self.p = params
        self.rule = shortfall_rule
        self.n = params.n
        self.P = int(params.lead_time.max()) + 1
        self.Q = int((params.lead_time + params.pay_delay).max()) + 1

    def initial_state(self, week: int, cash: float, on_hand: np.ndarray) -> WorldState:
        return WorldState(
            week=int(week),
            cash=float(cash),
            on_hand=np.asarray(on_hand, dtype=np.int64).copy(),
            pipeline=np.zeros((self.n, self.P), dtype=np.int64),
            payables=np.zeros(self.Q),
            spoil_acc=np.zeros(self.n),
        )

    def step(self, state: WorldState, orders: Orders, demand: np.ndarray, price: np.ndarray, fixed_cost: float, buffer: float) -> tuple[WorldState, dict]:
        p = self.p
        demand = np.asarray(np.rint(demand), dtype=np.int64)
        qty_req = np.asarray(orders.qty)
        if qty_req.shape != (self.n,) or (qty_req < 0).any() or not np.allclose(qty_req, np.rint(qty_req)):
            raise ValueError("orders must be non-negative whole units, one per SKU")
        qty_req = np.rint(qty_req).astype(np.int64)

        # 1. arrivals
        arrivals = state.pipeline[:, 0].copy()
        on_hand = state.on_hand + arrivals
        # 2. sales (lost sales are not backordered)
        sales = np.minimum(demand, on_hand)
        lost = demand - sales
        on_hand = on_hand - sales
        # 2b. spoilage of perishables
        acc = state.spoil_acc + p.spoilage_rate * on_hand
        spoiled = np.minimum(np.floor(acc + 1e-9).astype(np.int64), on_hand)
        acc = acc - spoiled
        on_hand = on_hand - spoiled
        # 3. revenue
        revenue = float(sales @ price)
        salvage = float(spoiled @ p.salvage_value)
        cash = state.cash + revenue + salvage
        # 4. payables and fixed costs
        supplier_paid = float(state.payables[0])
        cash -= supplier_paid + float(fixed_cost)
        # 5. new orders
        qty = qty_req
        cost_req = float(qty_req @ p.unit_cost)
        cut = False
        if self.rule == "cut_proportional" and cost_req > max(cash, 0.0):
            scale = max(cash, 0.0) / cost_req
            qty = (np.floor(qty_req * scale / p.pack_size) * p.pack_size).astype(np.int64)
            qty[qty < p.moq] = 0
            cut = True
        pipeline = state.pipeline.copy()
        payables = state.payables.copy()
        pipeline[np.arange(self.n), p.lead_time] += qty
        np.add.at(payables, p.lead_time + p.pay_delay, qty * p.unit_cost)

        new_pipeline = np.zeros_like(pipeline)
        new_pipeline[:, :-1] = pipeline[:, 1:]
        new_payables = np.zeros_like(payables)
        new_payables[:-1] = payables[1:]
        new_state = WorldState(state.week + 1, cash, on_hand, new_pipeline, new_payables, acc)

        unit_margin = price - p.unit_cost
        spoil_loss = float(spoiled @ (p.unit_cost - p.salvage_value))
        inv_value = float(on_hand @ p.unit_cost)
        pipe_value = float(new_pipeline.sum(axis=1) @ p.unit_cost)
        payables_out = float(new_payables.sum())
        rec = {
            "week": int(state.week),
            "cash_start": float(state.cash),
            "revenue": revenue,
            "salvage": salvage,
            "fixed": float(fixed_cost),
            "supplier_paid": supplier_paid,
            "cash_end": float(cash),
            "order_cost": float(qty @ p.unit_cost),
            "order_cost_requested": cost_req,
            "order_cut": bool(cut),
            "units_demand": int(demand.sum()),
            "units_sold": int(sales.sum()),
            "units_lost": int(lost.sum()),
            "units_arrived": int(arrivals.sum()),
            "units_spoiled": int(spoiled.sum()),
            "lost_margin": float(lost @ unit_margin),
            "gross_margin": float(sales @ unit_margin) - spoil_loss,
            "cogs": float(sales @ p.unit_cost),
            "spoil_loss": spoil_loss,
            "inv_value": inv_value,
            "pipeline_value": pipe_value,
            "payables_out": payables_out,
            "net_position": float(cash) + inv_value + pipe_value - payables_out,
            "buffer": float(buffer),
            "shortfall": bool(cash < buffer),
            "insolvent": bool(cash < 0.0),
            # per-SKU detail, stripped before records are tabulated
            "_sales": sales,
            "_lost": lost,
            "_arrivals": arrivals,
            "_spoiled": spoiled,
            "_qty": qty,
        }
        return new_state, rec
