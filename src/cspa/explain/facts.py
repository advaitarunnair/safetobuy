"""Structured facts behind every explanation.

Each fact carries the raw number and the exact display string. Templates and
the LLM are only ever given the display strings, and the guardrail in
explain/llm.py checks LLM output against those same strings, so the rounding
a reader sees is the rounding that was verified.
"""
from __future__ import annotations

import math

import pandas as pd

from cspa.cash.gate import BudgetResult


def money(x: float) -> str:
    """$1,240 for amounts of $100 or more, $12.40 below that."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"${x:,.0f}" if abs(x) >= 100 else f"${x:,.2f}"


def pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def num1(x: float) -> str:
    return f"{x:.1f}"


def sku_facts(row: pd.Series, lead_time: int, pack_size: int, moq: int, perishable: bool) -> dict:
    """Facts for one SKU from a plan_report row."""
    qty, target = int(row["qty"]), int(row["target_qty"])
    mv_last, mv_next = row["mv_per_dollar_last"], row["mv_per_dollar_next"]
    return {
        "sku": str(row["sku"]),
        "decision": str(row["decision"]),
        "needed": bool(row["needed"]),
        "perishable": bool(perishable),
        "raw": {
            "qty": qty,
            "target_qty": target,
            "deferred_units": max(target - qty, 0),
            "spend": float(row["spend"]),
            "target_spend": float(row["target_spend"]),
            "stockout_prob": float(row["stockout_prob"]),
            "stockout_prob_if_full": float(row["stockout_prob_if_full"]),
            "days_cover": float(row["days_cover"]),
            "exp_margin_lost": float(row["exp_margin_lost"]),
            "mv_per_dollar_last": None if pd.isna(mv_last) else float(mv_last),
            "mv_per_dollar_next": None if pd.isna(mv_next) else float(mv_next),
            "lead_time_weeks": int(lead_time),
        },
        "fmt": {
            "qty": f"{qty:,}",
            "target_qty": f"{target:,}",
            "deferred_units": f"{max(target - qty, 0):,}",
            "spend": money(float(row["spend"])),
            "deferred_spend": money(max(float(row["target_spend"]) - float(row["spend"]), 0.0)),
            "stockout_pct": pct(float(row["stockout_prob"])),
            "stockout_pct_if_full": pct(float(row["stockout_prob_if_full"])),
            "days_cover": num1(float(row["days_cover"])),
            "cost_of_deferring": money(float(row["exp_margin_lost"])),
            "return_per_dollar_last": "n/a" if pd.isna(mv_last) else f"${mv_last:.2f}",
            "return_per_dollar_next": "n/a" if pd.isna(mv_next) else f"${mv_next:.2f}",
            "lead_time_weeks": f"{int(lead_time)}",
            "pack_size": f"{int(pack_size)}",
            "moq": f"{int(moq)}",
            "per_dollar_unit": "$1",
        },
    }


def week_facts(report: pd.DataFrame, gate: BudgetResult | None, cash: float, buffer: float, alpha: float, prob_safe_plan: float, prob_safe_full: float | None, horizon_weeks: int = 4) -> dict:
    """Facts for the weekly summary."""
    needed = report[report["needed"]]
    n_full = int((needed["decision"] == "full").sum())
    n_partial = int((needed["decision"] == "partial").sum())
    n_defer = int((needed["decision"] == "defer").sum())
    spend = float(report["spend"].sum())
    full_cost = float(report["target_spend"].sum())
    deferred_loss = float(report["exp_margin_lost"].sum())
    status = gate.status if gate is not None else "n/a"
    budget = gate.B if gate is not None else spend
    return {
        "status": status,
        "flag": gate.flag if gate is not None else None,
        "raw": {
            "budget": float(budget),
            "spend": spend,
            "full_cost": full_cost,
            "prob_safe": float(prob_safe_plan),
            "prob_safe_full": None if prob_safe_full is None else float(prob_safe_full),
            "target": 1.0 - alpha,
            "cash": float(cash),
            "buffer": float(buffer),
            "n_full": n_full,
            "n_partial": n_partial,
            "n_defer": n_defer,
            "deferred_loss": deferred_loss,
        },
        "fmt": {
            "budget": money(float(budget)),
            "spend": money(spend),
            "full_cost": money(full_cost),
            "deferred_spend": money(max(full_cost - spend, 0.0)),
            "prob_safe": pct(float(prob_safe_plan)),
            "prob_safe_full": "n/a" if prob_safe_full is None else pct(float(prob_safe_full)),
            "target": pct(1.0 - alpha),
            "cash": money(float(cash)),
            "buffer": money(float(buffer)),
            "n_full": f"{n_full}",
            "n_partial": f"{n_partial}",
            "n_defer": f"{n_defer}",
            "deferred_loss": money(deferred_loss),
            "horizon_weeks": f"{int(horizon_weeks)}",
        },
    }


def build_facts(report: pd.DataFrame, params, gate: BudgetResult | None, cash: float, buffer: float, alpha: float, prob_safe_plan: float, prob_safe_full: float | None, horizon_weeks: int = 4) -> dict:
    """{'week': {...}, 'skus': [{...}, ...]} for SKUs that needed an order this week."""
    idx = {s: i for i, s in enumerate(params.skus)}
    skus = []
    for _, row in report[report["needed"]].iterrows():
        i = idx[row["sku"]]
        skus.append(sku_facts(row, int(params.lead_time[i]), int(params.pack_size[i]), int(params.moq[i]), bool(params.perishable[i])))
    return {"week": week_facts(report, gate, cash, buffer, alpha, prob_safe_plan, prob_safe_full, horizon_weeks), "skus": skus}
