"""One week's plan, recomputed from the results bundle.

Shared by the Streamlit app and by scripts/render_results.py, so the numbers
quoted in the demo script are the numbers the app shows at its default settings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cspa.allocate.marginal import plan_report
from cspa.explain.facts import build_facts
from cspa.explain.llm import explain
from cspa.explain.templates import explain_with_templates
from cspa.policies import REGISTRY, PlanContext, make_policy
from cspa.sim.backtest import WorldCal, make_info
from cspa.sim.bundle import Bundle

FAN_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def buffer_step(cal: WorldCal) -> float:
    return max(50.0, round(cal.buffer / 40 / 50) * 50.0)


def default_buffer(cal: WorldCal) -> float:
    """The buffer slider's default: the scenario buffer rounded to the slider step."""
    step = buffer_step(cal)
    return float(round(cal.buffer / step) * step)


def default_fallback(b: Bundle) -> str:
    """What policy C does when no budget is safe (gate.on_infeasible in the run's config)."""
    return str(b.cfg.gate.on_infeasible)


def compute_week_plan(b: Bundle, sid: str, week: int, alpha: float, buffer: float, n_paths: int, fallback: str, use_llm: bool = False) -> dict:
    """Re-run the cash gate and the allocator for one decision week from the bundled policy's saved state."""
    cfg, p = b.cfg, b.params
    cal, state = b.cal(sid), b.state(sid, week)
    H = int(cfg.horizons.cash_horizon_weeks)
    sched = cal.schedule(week, H)
    info = make_info(b.panel, b.cache, p, cfg, week, sched, buffer, alpha, n_paths)
    ctx = PlanContext(state, info)
    alloc = ctx.allocator(REGISTRY[b.policy].alloc_rule)
    table = ctx.table
    gate = ctx.gate(alloc, alpha=alpha, buffer=buffer, on_infeasible=fallback)
    qty = alloc(gate.B)
    report = plan_report(p.skus, qty, table, info.cover, info.cover_weeks)

    plans = {"C": qty, "Nothing": np.zeros(p.n, dtype=np.int64), "Everything": table.target_qty}
    for name in ("A", "B", "D"):
        plans[name] = make_policy(name).decide(state, info).qty
    sims = {k: ctx.simulate(v) for k, v in plans.items()}
    compare = pd.DataFrame(
        [
            {
                "plan": k,
                "spend": float(v @ p.unit_cost),
                "p_shortfall": 1.0 - sims[k].prob_above(buffer),
                "exp_margin": float(sims[k].margin.mean()),
                "p05_min_cash": float(np.quantile(sims[k].min_cash, 0.05)),
            }
            for k, v in plans.items()
        ]
    )
    info_ind = make_info(b.panel, b.cache, p, cfg, week, sched, buffer, alpha, n_paths, method="independent")
    p_safe_independent = PlanContext(state, info_ind).simulate(qty).prob_above(buffer)

    facts = build_facts(report, p, gate, state.cash, buffer, alpha, sims["C"].prob_above(buffer), sims["Everything"].prob_above(buffer), H)
    expl = explain(facts, cfg) if use_llm else {**explain_with_templates(facts), "rejected": [], "error": None}
    report["reason"] = report["sku"].map(expl["lines"]).fillna("No order needed this week.")
    report["source"] = report["sku"].map(expl["source"]).fillna("template")
    return {
        "gate": gate.to_dict(),
        "report": report,
        "compare": compare,
        "fan": {k: sims[k].quantiles(FAN_QUANTILES) for k in sims},
        "cash_now": float(state.cash),
        "fixed": sched,
        "payables": state.payables[:H].copy(),
        "p_safe": sims["C"].prob_above(buffer),
        "p_safe_independent": p_safe_independent,
        "facts": facts,
        "summary": expl["summary"],
        "summary_source": expl["summary_source"],
        "rejected": expl["rejected"],
        "llm_error": expl["error"],
        "horizon": H,
    }
