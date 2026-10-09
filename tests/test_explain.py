import json
from types import SimpleNamespace

import numpy as np
import pytest

from cspa.allocate.marginal import plan_report
from cspa.explain.facts import build_facts, money, pct
from cspa.explain.llm import allowed_numbers, explain, extract_numbers, verify_text
from cspa.explain.templates import explain_with_templates, sku_line, week_summary
from cspa.policies import PlanContext, make_policy


@pytest.fixture()
def facts(week_setup):
    """Facts for a week where the gate is binding, so full, partial and deferred lines all occur."""
    state, info = week_setup
    ctx = PlanContext(state, info)
    qty = ctx.allocator("marginal")(0.45 * ctx.table.total_cost)
    report = plan_report(info.params.skus, qty, ctx.table, info.cover, info.cover_weeks)
    gate = make_policy("C").decide(state, info).meta["gate"]
    sim, full = ctx.simulate(qty), ctx.simulate(ctx.table.target_qty)
    f = build_facts(report, info.params, gate, state.cash, info.buffer, info.alpha, sim.prob_above(info.buffer), full.prob_above(info.buffer))
    assert {"full", "partial", "defer"} & {s["decision"] for s in f["skus"]}
    return f


class FakeClient:
    """Stands in for anthropic.Anthropic(); returns whatever text it is given."""

    def __init__(self, payload, stop_reason="end_turn", raises=None):
        self.payload, self.stop_reason, self.raises, self.calls = payload, stop_reason, raises, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=text)])


def good_payload(facts):
    lines = []
    for f in facts["skus"]:
        x = f["fmt"]
        lines.append({"sku": f["sku"], "text": f"Order {x['qty']} units for {x['spend']}; stockout risk is {x['stockout_pct']} and waiting would put {x['cost_of_deferring']} of margin at risk."})
    w = facts["week"]["fmt"]
    return {"summary": f"You can safely spend {w['budget']} this week at {w['target']} confidence, and the plan uses {w['spend']}.", "lines": lines}


def test_number_extraction_and_formatting():
    assert extract_numbers("Buy 1,240 units for $86.50 (31%) of FOODS_3_090", ["FOODS_3_090"]) == ["1240", "86.5", "31"]
    assert extract_numbers("FOODS_3_123 and HOBBIES_1_004") == []
    assert money(1240.4) == "$1,240" and money(12.4) == "$12.40" and pct(0.314) == "31%"
    assert allowed_numbers({"fmt": {"a": "$1,240", "b": "31%", "c": "4.50"}}) == {"1240", "31", "4.5"}


def test_templates_use_only_supplied_numbers(facts):
    out = explain_with_templates(facts)
    week_ok = allowed_numbers(facts["week"])
    assert verify_text(out["summary"], week_ok)[0], verify_text(out["summary"], week_ok)[1]
    for f in facts["skus"]:
        ok, bad = verify_text(out["lines"][f["sku"]], allowed_numbers(f, facts["week"]))
        assert ok, (f["decision"], out["lines"][f["sku"]], bad)
    for status in ("unconstrained", "constrained", "cash_at_risk", "n/a"):
        w = {**facts["week"], "status": status}
        assert verify_text(week_summary(w), week_ok)[0]


def test_faithful_llm_output_is_accepted(facts, cfg):
    client = FakeClient(good_payload(facts))
    out = explain(facts, cfg, client=client)
    assert out["summary_source"] == "llm" and not out["rejected"] and out["error"] is None
    assert all(v == "llm" for v in out["source"].values())
    sent = client.calls[0]
    assert sent["model"] == cfg.llm.model and sent["output_config"]["format"]["type"] == "json_schema"
    assert "FOODS" in sent["messages"][0]["content"]  # facts were sent ...
    assert str(facts["skus"][0]["raw"]["exp_margin_lost"]) not in sent["messages"][0]["content"]  # ... as display strings only


def test_altered_number_is_rejected_and_the_template_is_used(facts, cfg):
    payload = good_payload(facts)
    victim = facts["skus"][0]
    wrong = int(victim["raw"]["qty"]) + 7
    payload["lines"][0]["text"] = f"Order {wrong} units for {victim['fmt']['spend']}."
    out = explain(facts, cfg, client=FakeClient(payload))
    sku = victim["sku"]
    assert out["source"][sku] == "template" and out["lines"][sku] == sku_line(victim)
    assert str(wrong) not in out["lines"][sku]
    assert any(r["sku"] == sku and str(wrong) in r["bad_numbers"] for r in out["rejected"])
    others = [f["sku"] for f in facts["skus"][1:]]
    assert all(out["source"][s] == "llm" for s in others)  # only the offending line falls back


def test_invented_summary_number_and_spelled_out_numbers_are_rejected(facts, cfg):
    payload = good_payload(facts)
    payload["summary"] = "Cash is fine: you will have $999,999 next month."
    payload["lines"][0]["text"] = "Buy two packs now, which is roughly double the usual order."
    out = explain(facts, cfg, client=FakeClient(payload))
    assert out["summary_source"] == "template" and out["summary"] == week_summary(facts["week"])
    assert out["source"][facts["skus"][0]["sku"]] == "template"
    assert not verify_text("about half of it", set())[0] and not verify_text("in 3 weeks", {"4"})[0]
    assert verify_text("in 4 weeks, for FOODS_3_090", {"4"})[0]


def test_rounding_a_supplied_number_differently_is_rejected(facts, cfg):
    f = next(x for x in facts["skus"] if x["raw"]["spend"] >= 1)
    drifted = f"${f['raw']['spend'] + 0.37:,.3f}"
    payload = {"summary": "", "lines": [{"sku": f["sku"], "text": f"Spend {drifted} on this item."}]}
    out = explain(facts, cfg, client=FakeClient(payload))
    assert out["source"][f["sku"]] == "template"


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_refusal_or_truncation_falls_back(stop, facts, cfg):
    out = explain(facts, cfg, client=FakeClient(good_payload(facts), stop_reason=stop))
    assert out["summary_source"] == "template" and set(out["source"].values()) == {"template"}


def test_api_errors_and_garbage_fall_back(facts, cfg):
    out = explain(facts, cfg, client=FakeClient(None, raises=RuntimeError("network down")))
    assert set(out["source"].values()) == {"template"} and "network down" in out["error"]
    out = explain(facts, cfg, client=FakeClient("this is not json"))
    assert set(out["source"].values()) == {"template"}
    out = explain(facts, cfg, client=FakeClient({"summary": 5, "lines": [{"sku": "NOT_A_SKU", "text": "Buy 3"}]}))
    assert set(out["source"].values()) == {"template"} and out["summary_source"] == "template"


def test_explanations_work_with_no_api_key(facts, cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    out = explain(facts, cfg)
    assert out["summary"] and set(out["source"].values()) == {"template"} and out["error"] is None
    assert len(out["lines"]) == len(facts["skus"]) and all(isinstance(v, str) and v for v in out["lines"].values())


def test_deferral_cost_in_facts_matches_the_report(week_setup):
    state, info = week_setup
    ctx = PlanContext(state, info)
    report = plan_report(info.params.skus, np.zeros(info.params.n, dtype=np.int64), ctx.table, info.cover, info.cover_weeks)
    f = build_facts(report, info.params, None, state.cash, info.buffer, info.alpha, 0.5, None)
    row = report[report["needed"]].iloc[0]
    sf = next(s for s in f["skus"] if s["sku"] == row["sku"])
    assert sf["decision"] == "defer" and sf["raw"]["exp_margin_lost"] == pytest.approx(row["exp_margin_lost"])
    assert sf["fmt"]["cost_of_deferring"] == money(row["exp_margin_lost"])
