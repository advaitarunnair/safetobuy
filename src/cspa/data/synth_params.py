"""Per-SKU SYNTHETIC parameters (costs, lead times, packs, terms, perishability).

Everything here is drawn from configs/default.yaml -> synthetic with a seed.
Each SKU gets its own random stream keyed on (seed, sku id), so a SKU's
parameters do not change when the slice changes.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cspa.data.load_m5 import Panel


@dataclass
class SkuParams:
    skus: list[str]
    ref_price: np.ndarray
    margin: np.ndarray
    unit_cost: np.ndarray
    lead_time: np.ndarray  # weeks, int
    pack_size: np.ndarray  # units, int
    moq: np.ndarray  # units, int (multiple of pack_size)
    supplier: list[str]
    pay_delay: np.ndarray  # weeks after delivery, int
    perishable: np.ndarray  # bool
    spoilage_rate: np.ndarray  # per week, 0 for non-perishables
    salvage_value: np.ndarray  # $ per written-off unit
    holding_cost_week: np.ndarray  # $ per unit per week (decision parameter)

    @property
    def n(self) -> int:
        return len(self.skus)

    def overage_cost(self, cfg) -> np.ndarray:
        """Co: cost of one leftover unit at the end of a review period.

        Non-perishables carry over, so a leftover only costs holding + capital
        for one more period. Perishables additionally lose value through spoilage.
        """
        period = float(cfg.horizons.review_period_weeks)
        hold = self.holding_cost_week * period
        loss = np.maximum(self.unit_cost - self.salvage_value, 0.0)
        if cfg.allocator.perishable_overage == "full_loss":
            per = loss + hold
        else:
            per = self.spoilage_rate * period * loss + hold
        return np.where(self.perishable, per, hold)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sku": self.skus,
                "ref_price": self.ref_price,
                "margin": self.margin,
                "unit_cost": self.unit_cost,
                "lead_time": self.lead_time,
                "pack_size": self.pack_size,
                "moq": self.moq,
                "supplier": self.supplier,
                "pay_delay": self.pay_delay,
                "perishable": self.perishable,
                "spoilage_rate": self.spoilage_rate,
                "salvage_value": self.salvage_value,
                "holding_cost_week": self.holding_cost_week,
            }
        )

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "SkuParams":
        return cls(
            skus=[str(s) for s in df["sku"]],
            ref_price=df["ref_price"].to_numpy(float),
            margin=df["margin"].to_numpy(float),
            unit_cost=df["unit_cost"].to_numpy(float),
            lead_time=df["lead_time"].to_numpy(np.int64),
            pack_size=df["pack_size"].to_numpy(np.int64),
            moq=df["moq"].to_numpy(np.int64),
            supplier=[str(s) for s in df["supplier"]],
            pay_delay=df["pay_delay"].to_numpy(np.int64),
            perishable=df["perishable"].to_numpy(bool),
            spoilage_rate=df["spoilage_rate"].to_numpy(float),
            salvage_value=df["salvage_value"].to_numpy(float),
            holding_cost_week=df["holding_cost_week"].to_numpy(float),
        )


def make_sku_params(panel: Panel, cfg) -> SkuParams:
    syn = cfg.synthetic
    s0, s1 = panel.meta["selection_window_weeks"]
    sel_price = panel.price[s0 : s1 + 1]
    sel_units = panel.units[s0 : s1 + 1]
    n = panel.n_skus
    ref_price = np.empty(n)
    for i in range(n):
        col = sel_price[:, i]
        col = col[~np.isnan(col)]
        if col.size == 0:  # never listed inside the selection window: fall back to first known price
            allp = panel.price[:, i]
            allp = allp[~np.isnan(allp)]
            col = allp[:1] if allp.size else np.array([1.0])
        ref_price[i] = float(np.median(col))
    mean_units = sel_units.mean(axis=0)

    lt_vals, lt_p = list(syn.lead_time_weeks["values"]), list(syn.lead_time_weeks.probs)
    moq_vals, moq_p = list(syn.moq_packs["values"]), list(syn.moq_packs.probs)
    pack_vals = sorted(syn.pack_sizes["values"])
    sup_names = [s["name"] for s in syn.suppliers]
    sup_share = [s["share"] for s in syn.suppliers]
    sup_delay = {s["name"]: int(s["pay_delay_weeks"]) for s in syn.suppliers}
    weekly_rate = (float(syn.holding_rate_annual) + float(syn.capital_rate_annual)) / 52.0

    margin, lead, pack, moq, supplier, perishable = np.empty(n), np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64), [], np.zeros(n, bool)
    for i, sku in enumerate(panel.skus):
        rng = np.random.default_rng([int(cfg.seed), zlib.crc32(sku.encode())])
        margin[i] = rng.uniform(syn.margin_range[0], syn.margin_range[1])
        lead[i] = rng.choice(lt_vals, p=lt_p)
        cap = max(1.0, float(syn.pack_sizes.max_share_of_weekly_demand) * mean_units[i])
        allowed = [v for v in pack_vals if v <= cap] or [min(pack_vals)]
        pack[i] = rng.choice(allowed)
        moq[i] = pack[i] * rng.choice(moq_vals, p=moq_p)
        supplier.append(str(rng.choice(sup_names, p=sup_share)))
        perishable[i] = rng.random() < float(syn.perishable_share)

    unit_cost = np.round(ref_price * (1.0 - margin), 2)
    unit_cost = np.maximum(unit_cost, 0.01)
    return SkuParams(
        skus=list(panel.skus),
        ref_price=ref_price,
        margin=margin,
        unit_cost=unit_cost,
        lead_time=lead,
        pack_size=pack,
        moq=moq,
        supplier=supplier,
        pay_delay=np.array([sup_delay[s] for s in supplier], dtype=np.int64),
        perishable=perishable,
        spoilage_rate=np.where(perishable, float(syn.spoilage_rate_per_week), 0.0),
        salvage_value=np.where(perishable, float(syn.salvage_frac_of_cost) * unit_cost, 0.0),
        holding_cost_week=weekly_rate * unit_cost,
    )


def params_summary(p: SkuParams) -> str:
    df = p.to_frame()
    lines = [
        "SYNTHETIC per-SKU parameters (seeded; only demand and prices are real):",
        f"  unit cost   : ${df.unit_cost.min():.2f} .. ${df.unit_cost.max():.2f} (median ${df.unit_cost.median():.2f}); margin {df.margin.min():.0%} .. {df.margin.max():.0%}",
        f"  lead time   : {df.lead_time.value_counts().sort_index().to_dict()} (weeks: count)",
        f"  pack size   : {df.pack_size.value_counts().sort_index().to_dict()}",
        f"  MOQ > 1 pack: {int((df.moq > df.pack_size).sum())} SKUs",
        f"  pay terms   : {df.supplier.value_counts().sort_index().to_dict()}",
        f"  perishable  : {int(df.perishable.sum())} SKUs",
    ]
    return "\n".join(lines)
