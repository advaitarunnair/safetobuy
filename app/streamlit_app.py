"""Streamlit app. Run with `make app` (or: streamlit run app/streamlit_app.py).

Loads the precomputed bundle in results/<run_id>/app so it starts fast and does
not need the raw M5 files. When the sliders move, only the cash gate and the
allocator are recomputed, for the selected week.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "scripts"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from cspa.config import PROJECT_NAME, as_config  # noqa: E402
from cspa.explain.llm import llm_available  # noqa: E402
from cspa.sim.bundle import load_bundle  # noqa: E402
from cspa.sim.weekplan import buffer_step, compute_week_plan, default_buffer, default_fallback  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
POLICY_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
POLICY_LABEL = {"A": "A: reorder point", "B": "B: open-to-buy", "D": "D: open-to-buy + priority cuts", "C": "C: cash gate + allocator (this plan)"}
DECISION_LABEL = {"full": "Buy in full", "partial": "Buy part", "defer": "Defer"}

st.set_page_config(page_title=PROJECT_NAME, layout="wide")


def results_root() -> Path:
    override = os.environ.get("CSPA_RESULTS_DIR")
    return Path(override) if override else ROOT / "results"


def latest_run() -> Path | None:
    root = results_root()
    name = os.environ.get("CSPA_RUN") or ((root / "LATEST").read_text().strip() if (root / "LATEST").exists() else None)
    if not name or not (root / name / "manifest.json").exists():
        return None
    return root / name


@st.cache_resource(show_spinner="Loading precomputed results...")
def get_bundle(run_dir: str):
    return load_bundle(Path(run_dir) / "app")


@st.cache_data(show_spinner=False)
def get_manifest(run_dir: str) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text())


@st.cache_data(show_spinner="Re-running the cash gate and allocator for this week...")
def compute_week(run_dir: str, sid: str, week: int, alpha: float, buffer: float, n_paths: int, fallback: str, use_llm: bool) -> dict:
    return compute_week_plan(get_bundle(run_dir), sid, week, alpha, buffer, n_paths, fallback, use_llm)


@st.cache_data(show_spinner=False)
def get_blocks(run_dir: str) -> dict:
    from render_results import build_blocks

    return build_blocks(Path(run_dir), include_demo=False)


def usd(x: float) -> str:
    return f"${x:,.0f}"


def md(text: str) -> str:
    """Escape dollar signs: Streamlit's markdown would read a pair of them as LaTeX."""
    return text.replace("\\$", "$").replace("$", "\\$")


def data_banner(manifest: dict, cfg) -> None:
    if manifest["data_source"] != "M5":
        st.error("SYNTHETIC TEST FIXTURE. These numbers are made up for testing the software and are not results.")
    sl = manifest["data_slice"]
    st.info(
        f"**What is real and what is not.** Demo persona: {cfg.project.persona}. Demand and shelf prices are real US Walmart data "
        f"(M5: store {sl['store_id']}, department {sl['dept_id']}, {sl['n_skus']} SKUs). Unit costs, lead times, pack sizes, supplier payment terms, "
        "fixed costs and cash are **synthetic**. The persona is framing only, not a calibrated claim about Singapore. Dollar figures are illustrative; "
        "the comparison between policies in the same simulated world is the meaningful result."
    )


def fan_chart(res: dict, buffer: float, overlay: list[str]) -> go.Figure:
    q = res["fan"]["C"]
    H = q.shape[1]
    x = ["Now"] + [f"Week {j + 1}" for j in range(H)]
    now = res["cash_now"]

    def with_now(row: np.ndarray) -> list[float]:
        return [now, *row.tolist()]

    fig = go.Figure()
    tint = "27,175,122"  # policy C keeps its colour everywhere
    fig.add_trace(go.Scatter(x=x, y=with_now(q[4]), mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=with_now(q[0]), mode="lines", line=dict(width=0), fill="tonexty", fillcolor=f"rgba({tint},0.16)", name="5% to 95% of outcomes", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=with_now(q[3]), mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=with_now(q[1]), mode="lines", line=dict(width=0), fill="tonexty", fillcolor=f"rgba({tint},0.30)", name="25% to 75% of outcomes", hoverinfo="skip"))
    custom = np.column_stack([with_now(q[0]), with_now(q[1]), with_now(q[3]), with_now(q[4])])
    fig.add_trace(
        go.Scatter(
            x=x, y=with_now(q[2]), mode="lines+markers", line=dict(color=POLICY_COLOR["C"], width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
            name="Median, this plan", customdata=custom,
            hovertemplate="<b>%{x}</b><br>median $%{y:,.0f}<br>middle half $%{customdata[1]:,.0f} to $%{customdata[2]:,.0f}<br>5% to 95% $%{customdata[0]:,.0f} to $%{customdata[3]:,.0f}<extra></extra>",
        )
    )
    dash = {"Nothing": "dot", "Everything": "dash"}
    names = {"Nothing": "Median if you buy nothing", "Everything": "Median if you buy everything", "A": "Median under A", "B": "Median under B", "D": "Median under D"}
    colors = {"Nothing": MUTED, "Everything": "#4a3aa7", "A": POLICY_COLOR["A"], "B": POLICY_COLOR["B"], "D": POLICY_COLOR["D"]}
    for k in overlay:
        fig.add_trace(go.Scatter(x=x, y=with_now(res["fan"][k][2]), mode="lines", line=dict(color=colors[k], width=2, dash=dash.get(k, "solid")), name=names[k], hovertemplate="%{x}: $%{y:,.0f}<extra>" + names[k] + "</extra>"))
    fig.add_hline(y=buffer, line=dict(color=INK, width=1.5, dash="dash"), annotation_text=f"cash buffer {usd(buffer)}", annotation_position="bottom right", annotation_font=dict(color=INK2, size=12))
    lows = [q[0].min(), buffer] + [res["fan"][k][2].min() for k in overlay]
    if min(lows) < 0:
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK2, size=13), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), yaxis=dict(title="End-of-week cash", tickprefix="$", tickformat=",.0f", gridcolor=GRID, zeroline=False), xaxis=dict(showgrid=False),
    )
    return fig


def page_plan(run: Path, manifest: dict) -> None:
    b = get_bundle(str(run))
    cfg = b.cfg
    scen = b.meta["scenarios"]
    st.sidebar.header("Shop situation")
    label = {s["scenario"]: f"Budget stress {int(round(s['stress'] * 100))}%" for s in scen}
    sids = [s["scenario"] for s in scen]
    demo = {}
    demo_file = run / "app" / "demo.json"  # written by render_results.py: the week used in the demo script
    if demo_file.exists():
        demo = json.loads(demo_file.read_text())
    try:  # ?stress=70&week=5 preselects a scenario and decision week (used for screenshots and the demo)
        pre = next(i for i, s in enumerate(scen) if int(round(s["stress"] * 100)) == int(st.query_params.get("stress", demo.get("stress_pct", "x"))))
    except (StopIteration, ValueError):
        pre = min(1, len(scen) - 1)
    sid = st.sidebar.selectbox("Scenario", sids, index=pre, format_func=label.get, help="How heavy the monthly lump of fixed costs is. Lower % = tighter.")
    sc = b.scenario(sid)
    cal = b.cal(sid)
    weeks = sc["weeks"]
    dates = b.panel.weeks["start_date"]
    try:
        wk0 = weeks[max(0, min(len(weeks) - 1, int(st.query_params.get("week", demo.get("week", 3))) - 1))]
    except ValueError:
        wk0 = weeks[min(2, len(weeks) - 1)]
    week = st.sidebar.select_slider("Decision week", options=weeks, value=wk0, format_func=lambda w: f"wk {weeks.index(w) + 1} ({pd.Timestamp(dates.iloc[w]).date()})")

    st.sidebar.header("Your risk settings")
    alpha_pct = st.sidebar.slider("Risk tolerance: chance of dipping below the buffer you accept (%)", 1, 30, int(round(cfg.risk.alpha * 100)), help="alpha. The gate keeps P(cash never below the buffer over the horizon) at or above 100% minus this.")
    step = buffer_step(cal)
    buffer = st.sidebar.slider("Cash buffer ($)", 0.0, float(round(3 * cal.buffer / step) * step), default_buffer(cal), step=step, format="$%d", help=f"Default = {cfg.risk.buffer_weeks_of_fixed_costs:g} weeks of average fixed costs.")
    with st.sidebar.expander("Advanced"):
        n_paths = st.select_slider("Monte Carlo demand paths", options=[250, 500, 1000, 2000, 4000], value=int(cfg.sampling.n_paths_app) if int(cfg.sampling.n_paths_app) in (250, 500, 1000, 2000, 4000) else 2000)
        fb = st.radio("If no budget is safe", ["max_prob", "zero"], index=0 if default_fallback(b) == "max_prob" else 1, format_func={"max_prob": "Get as close to safe as possible (default)", "zero": "Buy nothing (original specification)"}.get)
        use_llm = st.checkbox("AI-worded explanations", value=False, disabled=not llm_available(), help="Needs ANTHROPIC_API_KEY. Every number in the AI text is checked against the computed facts; lines that fail fall back to the template.")
        if not llm_available():
            st.caption("No API key found: explanations use the deterministic templates.")

    res = compute_week(str(run), sid, int(week), alpha_pct / 100.0, float(buffer), int(n_paths), fb, bool(use_llm))
    gate, report, cmp_ = res["gate"], res["report"], res["compare"].set_index("plan")
    conf = 1.0 - alpha_pct / 100.0
    H = int(cfg.horizons.cash_horizon_weeks)

    st.title(PROJECT_NAME)
    st.caption("Open-to-buy gives you a budget. We tell you whether you can afford it, and how to spend it best.")
    data_banner(manifest, cfg)

    # ---- 2. safe budget ----
    if gate["status"] == "unconstrained":
        st.subheader(md(f"Safe budget this week: {usd(gate['B'])} or more, at {conf:.0%} confidence"))
        st.success(md(f"Everything worth buying costs {usd(gate['B_max'])} and is cash-safe: {res['p_safe']:.0%} chance cash stays above the buffer over the next {H} weeks."))
    elif gate["status"] == "constrained":
        st.subheader(md(f"Safe budget this week: {usd(gate['B'])} at {conf:.0%} confidence"))
        st.warning(md(f"Buying everything worth buying would cost {usd(gate['B_max'])} and leave only a {1 - cmp_.loc['Everything', 'p_shortfall']:.0%} chance of staying above the buffer. The plan below spends {usd(cmp_.loc['C', 'spend'])} where each dollar earns the most."))
    else:
        st.subheader(md(f"No budget is safe this week. Best-chance budget: {usd(gate['B'])}"))
        st.error(md(f"Cash at risk regardless of purchasing: even with no new orders the chance of staying above the buffer is {1 - cmp_.loc['Nothing', 'p_shortfall']:.0%}, below your {conf:.0%} target." + (f" The plan shown is the one with the best chance: {res['p_safe']:.0%}." if fb == "max_prob" else " With this setting the gate returns a budget of zero.")))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cash in the bank", usd(res["cash_now"]))
    c2.metric("Plan spend", usd(cmp_.loc["C", "spend"]))
    c3.metric("Chance cash stays above buffer", f"{res['p_safe']:.0%}", help=f"Over the next {H} weeks, across {n_paths:,} simulated demand paths.")
    c4.metric("Supplier bills due this week", usd(float(res["payables"][0])))
    c5.metric("Fixed costs this week", usd(float(res["fixed"][0])))
    st.write(md(res["summary"]))
    st.caption(("AI-worded, numbers verified against computed facts." if res["summary_source"] == "llm" else "Deterministic template.") + (f" LLM error: {res['llm_error']}" if res["llm_error"] else "") + (f" {len(res['rejected'])} AI line(s) failed the number check and were replaced by templates." if res["rejected"] else ""))

    # ---- 1. fan chart ----
    st.subheader(f"Cash runway for this plan, next {H} weeks")
    compare_on = st.toggle("Compare this week under policies A, B and D", value=st.query_params.get("compare") == "1")
    overlay = ["A", "B", "D"] if compare_on else ["Nothing", "Everything"]
    st.plotly_chart(fan_chart(res, buffer, overlay), width="stretch", config={"displayModeBar": False})
    lump = [f"week {j + 1} ({usd(v)})" for j, v in enumerate(res["fixed"]) if v > res["fixed"].min() + 1e-6]
    st.caption(md(
        f"Bands show {n_paths:,} joint demand paths (SKUs move together as they did in past forecast errors). Later weeks assume one-for-one replenishment that never spends below the buffer. "
        + (f"Heavier fixed-cost week: {', '.join(lump)}. " if lump else "")
        + f"If SKUs were sampled independently the same plan would look {res['p_safe_independent']:.0%} safe instead of {res['p_safe']:.0%}: independence understates risk."
    ))

    # ---- 4. compare ----
    if compare_on:
        st.subheader("Same week, same cash, four ways to buy")
        rows = []
        for k in ("A", "B", "D", "C"):
            r = cmp_.loc[k]
            rows.append({"Policy": POLICY_LABEL[k], "Total spend ($)": round(r["spend"]), "P(shortfall)": r["p_shortfall"], f"Expected margin, next {H} weeks ($)": round(r["exp_margin"]), "Lowest cash in a bad case, 5th pct ($)": round(r["p05_min_cash"])})
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={
                "Total spend ($)": st.column_config.NumberColumn(format="localized"),
                "P(shortfall)": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
                f"Expected margin, next {H} weeks ($)": st.column_config.NumberColumn(format="localized"),
                "Lowest cash in a bad case, 5th pct ($)": st.column_config.NumberColumn(format="localized"),
            },
        )
        st.caption(md(f"P(shortfall) = chance cash dips below the {usd(buffer)} buffer within {H} weeks. All four plans are evaluated on the same demand paths, from the same starting state. A, B and D do not look at cash."))

    # ---- 3. buy / partial / defer ----
    st.subheader("What to buy, what to trim, what to defer")
    needed = report[report["needed"]].copy()
    counts = needed["decision"].value_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Buy in full", int(counts.get("full", 0)))
    k2.metric("Buy part", int(counts.get("partial", 0)))
    k3.metric("Defer", int(counts.get("defer", 0)))
    k4.metric("Expected cost of deferring", usd(float(needed["exp_margin_lost"].sum())), help="Expected margin lost this cycle versus buying everything worth buying, from the demand distribution.")
    show = st.multiselect("Show", ["full", "partial", "defer"], default=["partial", "defer", "full"], format_func=DECISION_LABEL.get)
    order = {"partial": 0, "defer": 1, "full": 2}
    view = needed[needed["decision"].isin(show)].assign(_o=lambda d: d["decision"].map(order)).sort_values(["_o", "exp_margin_lost", "spend"], ascending=[True, False, False])
    table = pd.DataFrame(
        {
            "SKU": view["sku"],
            "Decision": view["decision"].map(DECISION_LABEL),
            "Qty": view["qty"],
            "Spend ($)": view["spend"].round(0),
            "Cost of deferring ($)": view["exp_margin_lost"].round(2),
            "Stockout probability": view["stockout_prob"],
            "Days of cover": view["days_cover"],
            "Why": view["reason"],
        }
    )
    st.dataframe(
        table, hide_index=True, width="stretch", height=460,
        column_config={
            "Spend ($)": st.column_config.NumberColumn(format="localized"),
            "Cost of deferring ($)": st.column_config.NumberColumn(format="%.2f", help="Expected margin lost by not buying the newsvendor target this week."),
            "Stockout probability": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0, help="Chance demand before the next delivery exceeds stock on hand, on order and bought now."),
            "Days of cover": st.column_config.NumberColumn(format="%.1f"),
            "Why": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(f"{len(report) - len(needed)} other SKUs need no order this week. Gate search: {gate['n_evals']} budget evaluations, {gate['monotonicity_violations']} monotonicity violation(s).")


def page_backtest(run: Path, manifest: dict) -> None:
    cfg = as_config(manifest["config"])
    st.title("Backtest: does it hold up?")
    data_banner(manifest, cfg)
    blocks = get_blocks(str(run))
    st.subheader("Headline")
    st.markdown(md(blocks["headline"]))
    st.caption("Every policy runs in the same simulated shop: same realised demand (actual weekly sales), same synthetic costs and terms, same forecasts. Intervals are paired block-bootstrap intervals over weeks.")
    st.subheader("All policies, headline scope")
    st.markdown(md(blocks["results_table"]))
    figs = run / "figures"
    cols = st.columns(2)
    for i, name in enumerate(("shortfalls_by_stress.png", "margin_by_stress.png", "risk_return.png", "cash_paths.png", "ablation.png", "forecast_coverage.png", "forecast_wql.png")):
        if (figs / name).exists():
            cols[i % 2].image(str(figs / name), width="stretch")
    st.subheader("By budget stress")
    st.markdown(md(blocks["stress_table"]))
    st.subheader("By opening cash")
    st.markdown(md(blocks["start_cash_table"]))
    st.subheader("C against B, stated plainly")
    st.markdown(md(blocks["c_vs_b"]))
    st.subheader("How robust is the headline?")
    st.markdown(md(blocks["robustness_table"]))
    st.caption(md(blocks["gate_diagnostics"]))
    st.subheader("Forecast quality")
    st.markdown(md(blocks["forecast_table"]))
    st.subheader("This run")
    st.markdown(md(blocks["run_info"]))


def page_about(run: Path, manifest: dict) -> None:
    from render_results import synthetic_block

    cfg = as_config(manifest["config"])
    st.title("About the data and the method")
    data_banner(manifest, cfg)
    st.markdown(
        """
**ML for prediction, simulation and optimization for the decision.**

- **Prediction (ML):** a LightGBM quantile model forecasts weekly demand per SKU, and conformal calibration corrects its intervals using past out-of-sample errors.
- **Cash risk (simulation, not ML):** a Monte Carlo cash model replays thousands of joint demand paths through the shop's stock, supplier bills and fixed costs.
- **The decision (optimization, not ML):** the cash gate finds the largest budget that keeps the chance of dipping below the buffer within your tolerance; a greedy allocator spends it where expected profit per dollar is highest.
- **Explanations:** an LLM, if enabled, only rewords numbers the system computed. Each number in its text is checked in code; any mismatch falls back to a template.
"""
    )
    st.subheader("Synthetic parameters used in this run")
    st.markdown(md(synthetic_block(cfg)))


def main() -> None:
    run = latest_run()
    if run is None:
        st.title(PROJECT_NAME)
        st.warning("No results found yet.")
        st.markdown("Put the three M5 files in `data/raw/`, then run:\n\n```bash\nmake all\nmake app\n```\n\nSee the README for the Kaggle download steps.")
        st.stop()
    manifest = get_manifest(str(run))
    pages = ["This week's plan", "Backtest results", "About the data"]
    wanted = {"plan": 0, "backtest": 1, "about": 2}.get(st.query_params.get("page", "plan"), 0)  # ?page=backtest (used for screenshots)
    page = st.sidebar.radio("Page", pages, index=wanted, label_visibility="collapsed")
    if page == "This week's plan":
        if not (run / "app" / "bundle_meta.json").exists():
            st.error("This run has no app bundle (it was a partial run). Run `make backtest`.")
            st.stop()
        page_plan(run, manifest)
    elif page == "Backtest results":
        page_backtest(run, manifest)
    else:
        page_about(run, manifest)
    st.sidebar.caption(f"Run `{manifest['run_id']}`")


main()
