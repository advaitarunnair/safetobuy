"""Deterministic explanations. Used when no API key is set, when the LLM is
disabled, and whenever an LLM line fails the numeric guardrail.

Every number in these sentences is one of the display strings in the facts.
"""
from __future__ import annotations


def sku_line(f: dict) -> str:
    d, x = f["decision"], f["fmt"]
    spoil = " It is perishable, so extra stock risks write-offs." if f["perishable"] else ""
    if not f["needed"]:
        return f"No order needed: {x['days_cover']} days of cover already on hand or on order."
    if d == "full":
        return (
            f"Buy {x['qty']} units ({x['spend']}). Fully funded: this brings stockout risk before the next delivery "
            f"to {x['stockout_pct']}, and the last pack still returns {x['return_per_dollar_last']} of expected profit per {x['per_dollar_unit']}."
        )
    if d == "partial":
        return (
            f"Buy {x['qty']} of {x['target_qty']} units ({x['spend']}). The other {x['deferred_units']} units wait a week "
            f"because cash is tight: that risks {x['cost_of_deferring']} of margin and leaves a {x['stockout_pct']} "
            f"chance of running out before the next delivery.{spoil}"
        )
    return (
        f"Defer all {x['target_qty']} units ({x['deferred_spend']}). With {x['days_cover']} days of cover, waiting a week "
        f"risks {x['cost_of_deferring']} of margin ({x['stockout_pct']} chance of running out); each {x['per_dollar_unit']} here would return "
        f"{x['return_per_dollar_next']}, less than the items funded this week.{spoil}"
    )


def week_summary(w: dict) -> str:
    x, status = w["fmt"], w["status"]
    counts = f"{x['n_full']} items are bought in full, {x['n_partial']} in part and {x['n_defer']} are deferred"
    if status == "unconstrained":
        return (
            f"Everything worth buying this week costs {x['full_cost']}, and that is cash-safe: there is a {x['prob_safe']} "
            f"chance cash stays above the {x['buffer']} buffer over the next {x['horizon_weeks']} weeks (target {x['target']}). {counts}."
        )
    if status == "constrained":
        return (
            f"Safe budget this week: {x['budget']} at {x['target']} confidence. Buying everything worth buying would cost "
            f"{x['full_cost']} and leave only a {x['prob_safe_full']} chance of staying above the {x['buffer']} buffer. "
            f"The plan spends {x['spend']}: {counts}. Deferring is expected to cost {x['deferred_loss']} of margin."
        )
    if status == "cash_at_risk":
        return (
            f"Cash is at risk regardless of purchasing: even with no new orders there is less than a {x['target']} chance "
            f"of staying above the {x['buffer']} buffer. The plan spends {x['spend']} of the {x['full_cost']} worth buying "
            f"({x['prob_safe']} chance of staying above the buffer): {counts}."
        )
    return f"The plan spends {x['spend']} ({x['prob_safe']} chance cash stays above the {x['buffer']} buffer): {counts}."


def explain_with_templates(facts: dict) -> dict:
    return {
        "summary": week_summary(facts["week"]),
        "lines": {f["sku"]: sku_line(f) for f in facts["skus"]},
        "source": {f["sku"]: "template" for f in facts["skus"]},
        "summary_source": "template",
    }
