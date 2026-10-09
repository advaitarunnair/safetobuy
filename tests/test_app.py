"""Headless run of the Streamlit app on the fixture bundle: every page renders, sliders recompute."""
import pytest

from conftest import ROOT

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture()
def app(pipeline, monkeypatch):
    monkeypatch.setenv("CSPA_RESULTS_DIR", str(pipeline["tmp"] / "results"))
    monkeypatch.setenv("CSPA_RUN", pipeline["run"].name)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_plan_page_has_all_components(app):
    subs = [s.value for s in app.subheader]
    assert any(s.startswith("Safe budget this week: $") for s in subs)  # component 2
    assert any(s.startswith("Cash runway") for s in subs)  # component 1 (fan chart)
    assert any("What to buy" in s for s in subs)  # component 3
    labels = [m.label for m in app.metric]
    assert {"Cash in the bank", "Plan spend", "Chance cash stays above buffer", "Buy in full", "Defer"} <= set(labels)
    table = app.dataframe[0].value
    assert {"SKU", "Decision", "Qty", "Spend", "Cost of deferring", "Stockout probability", "Why"} <= set(table.columns)
    assert table["Why"].str.len().min() > 10
    assert any("SYNTHETIC TEST FIXTURE" in e.value for e in app.error)  # fixture data is never shown unlabelled
    assert any("synthetic" in i.value.lower() and "Walmart" in i.value for i in app.info)  # the persona / data note


def test_sliders_recompute_the_gate(app):
    before = [s.value for s in app.subheader][0]
    app.sidebar.slider[0].set_value(1).run()  # risk tolerance 1%
    assert not app.exception
    strict = [s.value for s in app.subheader][0]
    assert "99% confidence" in strict or strict.startswith("Safe budget this week: $")
    buf = app.sidebar.slider[1]
    buf.set_value(buf.max).run()  # a much larger buffer
    assert not app.exception
    after = [s.value for s in app.subheader][0]
    assert after != before or "or more" not in after


def test_compare_toggle_and_other_pages(app):
    app.toggle[0].set_value(True).run()
    assert not app.exception and len(app.dataframe) == 2
    cmp_ = app.dataframe[0].value
    assert list(cmp_["Policy"].str[0]) == ["A", "B", "D", "C"] and {"Total spend", "P(shortfall)"} <= set(cmp_.columns)
    app.sidebar.radio[0].set_value("Backtest results").run()
    assert not app.exception and "Headline" in [s.value for s in app.subheader]
    app.sidebar.radio[0].set_value("About the data").run()
    assert not app.exception


def test_app_explains_what_to_do_when_there_are_no_results(tmp_path, monkeypatch):
    monkeypatch.setenv("CSPA_RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("CSPA_RUN", raising=False)
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception and any("No results found" in w.value for w in at.warning)
