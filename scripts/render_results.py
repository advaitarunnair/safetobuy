"""Fill result blocks in README.md and docs/ from results/<run_id>/.

Every number in the README, the write-up and the demo script comes from here.
Blocks are delimited by  <!-- BEGIN:name -->  and  <!-- END:name -->  markers.

Usage:
  python scripts/render_results.py [--run RUN_ID]
  python scripts/render_results.py --placeholder      # before any real run exists

Refuses to publish numbers from a run that did not use real M5 data.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

from _common import ROOT, base_parser, load_all, resolve_run

from cspa.config import as_config
from cspa.sim.metrics import aggregate, headline_scope

TARGETS = ["README.md", "docs/project_description.md", "docs/demo_script.md", "docs/devpost_submission.md"]
POLICY_ORDER = ["A", "B", "D", "C", "C_gate_prop", "OTB_marginal"]
POLICY_NAME = {
    "A": "A: reorder point",
    "B": "B: open-to-buy, proportional split",
    "D": "D: open-to-buy, priority cuts",
    "C": "**C: cash gate + marginal allocator (proposed)**",
    "C_gate_prop": "ablation: cash gate + proportional split",
    "OTB_marginal": "ablation: OTB budget + marginal allocator",
}
PENDING = "_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._"


def md_table(df: pd.DataFrame, numeric: bool = True) -> str:
    """Markdown table. numeric=True right-aligns every column after the first."""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(" --- " if i == 0 or not numeric else " ---: " for i in range(len(df.columns))) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def usd(x: float) -> str:
    return f"${x:,.0f}"


def policy_table(pooled: pd.DataFrame) -> str:
    pooled = pooled.set_index("policy")
    rows = []
    for p in [p for p in POLICY_ORDER if p in pooled.index]:
        r = pooled.loc[p]
        rows.append(
            {
                "Policy": POLICY_NAME[p],
                "Shortfall weeks": f"{int(r.shortfall_weeks)} / {int(r.n_weeks)}",
                "Insolvency weeks": int(r.insolvency_weeks),
                "Fill rate": f"{r.fill_rate:.1%}",
                "Gross margin": usd(r.gross_margin),
                "Lost margin (stockouts)": usd(r.lost_margin),
                "Ending net position (mean)": usd(r.end_net_position),
                "Days of inventory": f"{r.avg_days_inventory:.1f}",
            }
        )
    return md_table(pd.DataFrame(rows))


def ci_txt(h: dict, key: str) -> str:
    ci = h.get(key)
    return f"{ci[0]:+.1f}% to {ci[1]:+.1f}%" if ci else "n/a"


def headline_rows(named: list[tuple[str, dict | None]]) -> str:
    rows = []
    for label, h in named:
        if not h:
            continue
        sf = h["shortfall_weeks"]
        rows.append(
            {
                "Scope": label,
                "Shortfall weeks A / B / C": f"{sf['A']} / {sf['B']} / {sf['C']}",
                "X: fewer shortfalls than A": "n/a (A had none)" if sf["A"] == 0 else f"{h['x_pct_fewer_shortfalls_vs_A']:+.1f}%",
                f"X {h['ci_level']:.0%} interval": "n/a" if sf["A"] == 0 else ci_txt(h, "x_ci"),
                "Y: margin vs B": f"{h['y_pct_margin_vs_B']:+.1f}%",
                f"Y {h['ci_level']:.0%} interval": ci_txt(h, "y_ci"),
                "Risk <= B": "yes" if h["risk_equal_or_lower_than_B"] else "no",
                "Planned headline form holds": "yes" if h["supports_decided_form"] else "no",
            }
        )
    return md_table(pd.DataFrame(rows)) if rows else PENDING


def synthetic_block(cfg) -> str:
    syn = cfg.synthetic
    sup = "; ".join(f"{s['name']} {s['share']:.0%} of SKUs, pays {s['pay_delay_weeks']} wk after delivery" for s in syn.suppliers)
    rows = [
        ("Unit cost", f"reference price x (1 - margin), margin ~ U({syn.margin_range[0]:.0%}, {syn.margin_range[1]:.0%}) per SKU; reference price = median M5 sell price in the selection window", "synthetic.margin_range"),
        ("Lead time", ", ".join(f"{v} wk ({p:.0%})" for v, p in zip(syn.lead_time_weeks["values"], syn.lead_time_weeks.probs)), "synthetic.lead_time_weeks"),
        ("Pack size", f"one of {list(syn.pack_sizes['values'])} units, never more than {syn.pack_sizes.max_share_of_weekly_demand:.0%} of a SKU's mean weekly units", "synthetic.pack_sizes"),
        ("MOQ", ", ".join(f"{v} pack(s) ({p:.0%})" for v, p in zip(syn.moq_packs["values"], syn.moq_packs.probs)), "synthetic.moq_packs"),
        ("Supplier payment terms", sup, "synthetic.suppliers"),
        ("Perishable SKUs", f"{syn.perishable_share:.0%} of SKUs; {syn.spoilage_rate_per_week:.0%} of leftover stock written off per week at {syn.salvage_frac_of_cost:.0%} of cost", "synthetic.perishable_share, spoilage_rate_per_week, salvage_frac_of_cost"),
        ("Holding + capital cost", f"{syn.holding_rate_annual:.0%} + {syn.capital_rate_annual:.0%} of unit cost per year (decision parameter, not a simulated cash flow)", "synthetic.holding_rate_annual, capital_rate_annual"),
        ("Fixed costs", f"{syn.fixed_cost_share_of_gross_margin:.0%} of pre-window gross margin on average; part is paid as a lump every {syn.fixed_cost_lump_every_weeks} weeks, sized by the stress level", "synthetic.fixed_cost_*"),
        ("Opening cash", "buffer + cushion x s x R*, where s is the stress level and R* the ideal weekly replenishment cost; cushion is set per scenario", "experiments.cash_cushions"),
        ("Cash buffer", f"{cfg.risk.buffer_weeks_of_fixed_costs:g} weeks of average fixed costs", "risk.buffer_weeks_of_fixed_costs"),
        ("Opening stock, pipeline, payables", f"whatever policy A leaves after a {syn.burn_in_weeks}-week burn-in with ample cash", "synthetic.burn_in_weeks"),
        ("Seed", f"{cfg.seed}", "seed"),
    ]
    return md_table(pd.DataFrame(rows, columns=["Parameter", "Value", "Config key"]), numeric=False)


def _iv(h: dict, key: str) -> str:
    ci = h.get(key)
    return f"{h['ci_level']:.0%} interval {ci[0]:+.1f}% to {ci[1]:+.1f}%" if ci else "no interval"


def verdict_lines(h: dict) -> list[str]:
    """What one headline comparison does and does not support, as plain statements."""
    sf, ins, n = h["shortfall_weeks"], h["insolvency_weeks"], h["n_weeks"]
    x, y, va = h["x_pct_fewer_shortfalls_vs_A"], h["y_pct_margin_vs_B"], h["margin_vs_A_pct"]
    out = []
    if sf["A"] == 0:
        out.append(f"- **Fewer cash-shortfall weeks than A: nothing to measure.** A had no shortfall weeks in {n} weeks (C had {sf['C']}).")
    elif sf["A"] < 0.05 * n:
        out.append(f"- **Fewer cash-shortfall weeks than A: not informative.** A ran short in only {sf['A']} of {n} weeks (C: {sf['C']}), too few to support a claim.")
    elif x > 0 and h["x_interval_excludes_zero"]:
        out.append(f"- **Fewer cash-shortfall weeks than A: supported.** C {sf['C']} vs A {sf['A']} of {n} weeks, {x:.1f}% fewer ({_iv(h, 'x_ci')}).")
    else:
        word = "fewer" if x >= 0 else "more"
        out.append(f"- **Fewer cash-shortfall weeks than A: not supported.** C {sf['C']} vs A {sf['A']} of {n} weeks, {abs(x):.1f}% {word} ({_iv(h, 'x_ci')}; the interval includes zero)." if not h["x_interval_excludes_zero"] else f"- **Fewer cash-shortfall weeks than A: not supported.** C had {abs(x):.1f}% more ({sf['C']} vs {sf['A']}; {_iv(h, 'x_ci')}).")
    if y > 0 and h["y_interval_excludes_zero"]:
        out.append(f"- **Higher margin than B: supported.** {y:.1f}% higher ({_iv(h, 'y_ci')}).")
    elif h["y_interval_excludes_zero"]:
        out.append(f"- **Higher margin than B: not supported.** C's margin was {abs(y):.1f}% lower ({_iv(h, 'y_ci')}).")
    else:
        out.append(f"- **Higher margin than B: not supported.** The difference was {y:+.1f}% ({_iv(h, 'y_ci')}; the interval includes zero).")
    p = h["prob_risk_equal_or_lower_than_B"]
    if h["risk_equal_or_lower_than_B"]:
        out.append(f"- **No more shortfall weeks than B: holds.** C {sf['C']} vs B {sf['B']} (C had no more than B in {p:.0%} of bootstrap resamples).")
    else:
        out.append(f"- **No more shortfall weeks than B: does not hold.** C {sf['C']} vs B {sf['B']} (C had no more than B in only {p:.0%} of bootstrap resamples).")
    out.append(f"- **Cost against A:** C's gross margin was {abs(va):.1f}% {'lower' if va < 0 else 'higher'} than A's.")
    out.append(f"- **Weeks with cash below zero:** A {ins['A']}, B {ins['B']}, C {ins['C']}.")
    return out


def verdict_block(hl: dict, exp) -> str:
    main = hl.get("main")
    if not main:
        return PENDING
    roles = "+".join(exp.headline.roles)
    out = [f"**Headline scope** ({roles} windows, the three stress levels, rule `{exp.headline.shortfall_rule}`, {main['n_weeks']} simulated weeks per policy):", ""]
    out += verdict_lines(main)
    out.append(f"- **Planned headline form:** {'supported' if main['supports_decided_form'] else 'not supported'}. It needs fewer shortfall weeks than A, higher margin than B, both with intervals clear of zero, and no more shortfall weeks than B.")
    for rule, h in (hl.get("by_rule") or {}).items():
        if h and rule != str(exp.headline.shortfall_rule):
            out += ["", f"**Same windows under the alternative cash rule `{rule}`:**", ""] + verdict_lines(h)
    parts = []
    for label, group in (("window", hl.get("by_window") or {}), ("stress", hl.get("by_stress") or {})):
        for k, h in group.items():
            if not h:
                continue
            name = f"window {k}" if label == "window" else f"stress {int(round(float(k) * 100))}%"
            sf = h["shortfall_weeks"]
            parts.append(f"{name}: shortfall weeks A {sf['A']} / B {sf['B']} / C {sf['C']}, margin vs B {h['y_pct_margin_vs_B']:+.1f}% ({_iv(h, 'y_ci')}), planned form {'holds' if h['supports_decided_form'] else 'does not hold'}")
    if parts:
        out += ["", "**It is not uniform across the held-out data.** " + "; ".join(parts) + ". These cuts were not chosen in advance and none of them is the headline."]
    for role, h in (hl.get("other_roles") or {}).items():
        if h:
            sf = h["shortfall_weeks"]
            out += ["", f"**{role.capitalize()} windows (used to choose settings, so optimistic by construction):** shortfall weeks A {sf['A']} / B {sf['B']} / C {sf['C']}, margin vs B {h['y_pct_margin_vs_B']:+.1f}% ({_iv(h, 'y_ci')}), vs A {h['margin_vs_A_pct']:+.1f}%."]
    return "\n".join(out)


def secondary_block(exp) -> str:
    """The comparison slice named in results/SECONDARY, summarised from its own run."""
    from _common import results_root

    ptr = results_root() / "SECONDARY"
    if not ptr.exists():
        return "_No comparison slice has been run._"
    run = results_root() / ptr.read_text().strip()
    man = json.loads((run / "manifest.json").read_text())
    hl = json.loads((run / "headline.json").read_text())
    sexp = as_config(man["experiments"])
    sl, h = man["data_slice"], hl.get("main")
    if not h:
        return "_The comparison run has no headline._"
    scope = headline_scope(pd.read_csv(run / "metrics.csv"), sexp)
    sf, n = h["shortfall_weeks"], h["n_weeks"]
    out = [
        f"**Store {sl['store_id']}, department {sl['dept_id']}** ({sl['n_skus']} of {sl['n_skus_in_dept']} SKUs), same settings as the primary slice with nothing retuned. Run `{man['run_id']}`, config hash `{man['config_hash']}`.",
        "",
    ]
    out += verdict_lines(h)
    if sf["A"] < 0.05 * n:
        formal = " The planned headline form holds formally here, but only on that handful of weeks, so we do not lead with it." if h["supports_decided_form"] else ""
        out += ["", f"On this slice the held-out windows were comfortable: policy A ran short in only {sf['A']} of {n} weeks, so it says little about cash shortfalls. What it does show is the margin comparison.{formal}"]
    else:
        out += ["", f"Planned headline form on this slice: {'supported' if h['supports_decided_form'] else 'not supported'}."]
    out += ["", policy_table(aggregate(scope, ["shortfall_rule"]))]
    return "\n".join(out)


def ablation_block(pooled: pd.DataFrame) -> str:
    """What the gate and the allocator each changed, from the headline-scope ablation rows."""
    p = pooled.set_index("policy")
    if not {"B", "C", "C_gate_prop", "OTB_marginal"} <= set(p.index):
        return PENDING

    def step(a: str, b: str) -> str:
        dm = 100.0 * (p.loc[b, "gross_margin"] / p.loc[a, "gross_margin"] - 1.0)
        return f"gross margin {dm:+.1f}%, shortfall weeks {int(p.loc[a, 'shortfall_weeks'])} to {int(p.loc[b, 'shortfall_weeks'])}"

    return (
        f"Swapping the proportional split for the marginal allocator: with an open-to-buy budget, {step('B', 'OTB_marginal')}; "
        f"with the cash gate, {step('C_gate_prop', 'C')}. "
        f"Swapping the open-to-buy budget for the cash gate: with a proportional split, {step('B', 'C_gate_prop')}; "
        f"with the marginal allocator, {step('OTB_marginal', 'C')}."
    )


def c_vs_b_block(hl: dict, exp) -> str:
    """Plain statement of how C compared with B, generated from the run."""
    lines = []
    roles = "+".join(exp.headline.roles)
    named = [(f"{roles} windows, rule `{r}`", h) for r, h in (hl.get("by_rule") or {}).items()]
    named += [(f"{role} windows (used for tuning), rule `{exp.headline.shortfall_rule}`", h) for role, h in (hl.get("other_roles") or {}).items()]
    for label, h in named:
        if not h:
            continue
        y, sf, gm = h["y_pct_margin_vs_B"], h["shortfall_weeks"], h["gross_margin"]
        verdict = "higher" if y > 0 else "lower"
        ci = h.get("y_ci")
        clear = "the interval excludes zero" if ci and (ci[0] > 0 or ci[1] < 0) else "the interval includes zero, so the margin difference is not clear"
        lines.append(
            f"- **{label}:** C's gross margin was {abs(y):.1f}% {verdict} than B's ({usd(gm['C'])} vs {usd(gm['B'])}; {h['ci_level']:.0%} interval {ci_txt(h, 'y_ci')}; {clear}). "
            f"Shortfall weeks: C {sf['C']}, B {sf['B']}, A {sf['A']}. Insolvency weeks: C {h['insolvency_weeks']['C']}, B {h['insolvency_weeks']['B']}, A {h['insolvency_weeks']['A']}."
        )
    return "\n".join(lines) if lines else PENDING


def demo_block(run: Path, cfg, exp) -> dict[str, str]:
    """Pick the demo week from the app bundle and quote its numbers at the app's default settings."""
    from cspa.sim.bundle import load_bundle
    from cspa.sim.weekplan import compute_week_plan, default_buffer, default_fallback

    if not (run / "app" / "bundle_meta.json").exists():
        return {"demo_week": PENDING, "demo_figure": PENDING}
    b = load_bundle(run / "app")
    scen = next((s for s in b.meta["scenarios"] if s["stress"] == float(exp.default_stress)), b.meta["scenarios"][0])
    sid, cal = scen["scenario"], b.cal(scen["scenario"])
    alpha, buf, n_paths, fb = float(cfg.risk.alpha), default_buffer(cal), int(cfg.sampling.n_paths_app), default_fallback(b)
    best = None
    for i, week in enumerate(scen["weeks"]):
        res = compute_week_plan(b, sid, int(week), alpha, buf, n_paths, fb)
        g = res["gate"]
        if g["status"] == "constrained":
            # Most instructive week to show: the gate is binding and the cash-blind order (policy A)
            # would carry the largest shortfall risk. Ties go to the larger gap between full and safe.
            risk_a = float(res["compare"].set_index("plan").loc["A", "p_shortfall"])
            score = (round(risk_a, 3), g["B_max"] - g["B"])
            if best is None or score > best[0]:
                best = (score, i, week, res)
    if best is None:
        txt = f"No week in the bundled scenario (stress {int(round(scen['stress'] * 100))}%) has a binding cash gate at default settings, so there is no constrained week to demo. Lower the risk-tolerance slider or raise the buffer in the app to show the gate binding."
        return {"demo_week": txt, "demo_figure": txt}
    _, i, week, res = best
    w, cmp_ = res["facts"]["week"]["fmt"], res["compare"].set_index("plan")
    date = pd.Timestamp(b.panel.weeks["start_date"].iloc[week]).date()
    figure = f"{w['cash']} in the bank, and everything worth buying this week costs {w['full_cost']}. The safe budget is {w['budget']}."
    txt = (
        f"In the app choose **Budget stress {int(round(scen['stress'] * 100))}%** and **decision week {i + 1}** ({date}), default sliders. "
        f"The shop has {w['cash']} in the bank against a {w['buffer']} buffer. Everything worth buying costs {w['full_cost']}, which would leave only a "
        f"{w['prob_safe_full']} chance of staying above the buffer. The safe budget at {w['target']} confidence is {w['budget']}; the plan spends {w['spend']}: "
        f"{w['n_full']} items in full, {w['n_partial']} in part, {w['n_defer']} deferred, at an expected cost of {w['deferred_loss']} in margin. "
        f"For comparison, from the same position policy A would spend {usd(cmp_.loc['A', 'spend'])} with a {cmp_.loc['A', 'p_shortfall']:.0%} chance of a shortfall, "
        f"and policy B {usd(cmp_.loc['B', 'spend'])} with {cmp_.loc['B', 'p_shortfall']:.0%}; this plan's chance is {cmp_.loc['C', 'p_shortfall']:.0%}."
    )
    return {"demo_week": txt, "demo_figure": figure}


def build_blocks(run: Path, include_demo: bool = True) -> dict[str, str]:
    manifest = json.loads((run / "manifest.json").read_text())
    cfg, exp = as_config(manifest["config"]), as_config(manifest["experiments"])
    hl = json.loads((run / "headline.json").read_text())
    metrics = pd.read_csv(run / "metrics.csv")
    sl, git = manifest["data_slice"], manifest["git"]
    main = hl.get("main")
    roles = "+".join(exp.headline.roles)
    blocks: dict[str, str] = {}

    blocks["project_name"] = cfg.project.name
    blocks["headline"] = f"> **{main['sentence']}**\n>\n> Scope: {main['scope']}; {main['n_scenarios']} scenarios, {main['n_weeks']} simulated weeks. Run `{manifest['run_id']}`." if main else PENDING
    blocks["headline_plain"] = main["sentence"] if main else PENDING

    scope = headline_scope(metrics, exp)
    blocks["results_table"] = policy_table(aggregate(scope, ["shortfall_rule"])) if not scope.empty else PENDING
    blocks["ablation_note"] = ablation_block(aggregate(scope, ["shortfall_rule"])) if not scope.empty else PENDING

    parts = []
    for s, g in aggregate(scope, ["stress"]).groupby("stress"):
        g = g.set_index("policy")
        row = {"Budget stress": f"{int(round(s * 100))}%"}
        for p in [p for p in ("A", "B", "D", "C") if p in g.index]:
            row[f"{p} shortfall wks"] = int(g.loc[p].shortfall_weeks)
        for p in [p for p in ("A", "B", "D", "C") if p in g.index]:
            row[f"{p} margin"] = usd(g.loc[p].gross_margin)
        parts.append(row)
    blocks["stress_table"] = md_table(pd.DataFrame(parts)) if parts else PENDING

    cash_rows = []
    cash_scope = metrics[metrics["role"].isin(list(exp.headline.roles)) & (metrics["shortfall_rule"] == str(exp.headline.shortfall_rule)) & (metrics["stress"] == float(exp.default_stress))]
    for c, g in aggregate(cash_scope, ["cash_cushion"]).groupby("cash_cushion"):
        g = g.set_index("policy")
        row = {"Opening free cash (x stress x R*)": f"{c:g}"}
        for p in [p for p in ("A", "B", "D", "C") if p in g.index]:
            row[f"{p} shortfall wks"] = int(g.loc[p].shortfall_weeks)
        for p in [p for p in ("A", "B", "D", "C") if p in g.index]:
            row[f"{p} margin"] = usd(g.loc[p].gross_margin)
        cash_rows.append(row)
    blocks["start_cash_table"] = md_table(pd.DataFrame(cash_rows)) if cash_rows else PENDING

    named = [(f"Headline: {roles} windows, rule {exp.headline.shortfall_rule}", main)]
    named += [(f"{roles} windows, rule {r}", h) for r, h in (hl.get("by_rule") or {}).items() if r != str(exp.headline.shortfall_rule)]
    named += [(f"{role} windows (used for tuning), rule {exp.headline.shortfall_rule}", h) for role, h in (hl.get("other_roles") or {}).items()]
    named += [(f"Window {w} only", h) for w, h in (hl.get("by_window") or {}).items()]
    named += [(f"Stress {int(round(float(s) * 100))}% only", h) for s, h in (hl.get("by_stress") or {}).items()]
    blocks["robustness_table"] = headline_rows(named)

    fe = run / "forecast_eval.csv"
    if fe.exists():
        ev = pd.read_csv(fe)
        ev = ev[(ev["h"] == "all") & (ev["scope"] == "all")]
        names = {"lgbm_raw": "LightGBM quantile (raw)", "lgbm_conformal": "LightGBM quantile + conformal", "seasonal_naive": "Seasonal naive", "naive": "Naive", "lgbm": "LightGBM quantile"}
        rows = [
            {"Forecaster": names.get(r.model, r.model), "wQL (lower is better)": f"{r.wql:.3f}", "Mean pinball (units)": f"{r.pinball_mean:.2f}", "50% interval coverage": f"{r.cov50:.1%}", "80% interval coverage": f"{r.cov80:.1%}", "90% interval coverage": f"{r.cov90:.1%}"}
            for r in ev.itertuples()
        ]
        blocks["forecast_table"] = md_table(pd.DataFrame(rows)) + f"\n\nRolling-origin over the backtest decision weeks, horizons 1 to {cfg.horizons.forecast_horizon_weeks}, {int(ev['n'].iloc[0]):,} SKU-week-horizon cells per model. LightGBM refit every {cfg.forecast.refit_every_weeks} weeks, predicted weekly."
    else:
        blocks["forecast_table"] = PENDING

    gd = manifest["gate_diagnostics_policy_C"]
    blocks["gate_diagnostics"] = (
        f"Across all {gd['decisions']} policy-C decisions in this run, some budget met the target in {gd['decisions'] - gd['cash_at_risk_weeks']} "
        f"(the gate was binding in {gd['binding_weeks']} of them). In the other {gd['cash_at_risk_weeks']} no budget did, not even zero: the result was flagged "
        f"\"cash at risk regardless of purchasing\" and the best-chance budget was used. The coarse-grid check found "
        f"{gd['monotonicity_violations']} monotonicity violation(s) (grid points whose feasibility disagreed with the bisection)."
    )
    blocks["run_info"] = md_table(
        pd.DataFrame(
            [
                ("Run id", f"`{manifest['run_id']}`"),
                ("Timestamp (UTC)", manifest["timestamp"]),
                ("Git commit", f"`{git['commit']}`" + (" (working tree had uncommitted changes)" if git.get("dirty") else "") if git.get("commit") else "not a git checkout"),
                ("Config hash", f"`{manifest['config_hash']}`"),
                ("Seed", manifest["seed"]),
                ("Data slice", f"store {sl['store_id']}, department {sl['dept_id']}, {sl['n_skus']} of {sl['n_skus_in_dept']} SKUs ({sl['selection_units_share_of_dept']:.0%} of department units in the selection window)"),
                ("History", f"{sl['n_obs_weeks']} complete Walmart weeks, {sl['first_date']} to {sl['last_date']}"),
                ("SKU selection window", f"weeks {sl['selection_window_weeks'][0]} to {sl['selection_window_weeks'][1]} ({sl['selection_mode']})"),
                ("Backtest windows", "; ".join(f"{w['name']} ({w['role']}): weeks {w['start']} to {w['start'] + w['n_weeks'] - 1}" for w in manifest["windows"])),
                ("Scenarios x policies", f"{manifest['n_scenarios']} x {len(manifest['policies'])}"),
                ("Monte Carlo paths per decision", f"{manifest['mc_paths_backtest']} ({manifest['sampling_method']} sampling)"),
                ("Forecaster", f"{manifest['forecaster']['name']}, refit every {manifest['forecaster'].get('refit_every_weeks')} weeks, conformal: {manifest['forecaster'].get('conformal')}"),
                ("Policy fairness check", manifest["fairness_check"]),
            ],
            columns=["Item", "Value"],
        ),
        numeric=False,
    )
    blocks["synthetic_params"] = synthetic_block(cfg)
    blocks["c_vs_b"] = c_vs_b_block(hl, exp)
    blocks["verdict"] = verdict_block(hl, exp)
    blocks["secondary_slice"] = secondary_block(exp)
    if main:
        sfm, insm = main["shortfall_weeks"], main["insolvency_weeks"]
        blocks["headline_numbers"] = (
            f"C had {sfm['C']} cash-shortfall weeks against {sfm['A']} for A and {sfm['B']} for B, out of {main['n_weeks']} held-out weeks each: "
            f"{main['x_pct_fewer_shortfalls_vs_A']:.1f}% fewer than A ({_iv(main, 'x_ci')}). C's gross margin was {main['y_pct_margin_vs_B']:+.1f}% against B ({_iv(main, 'y_ci')}) "
            f"and {main['margin_vs_A_pct']:+.1f}% against A. Weeks with cash below zero: A {insm['A']}, B {insm['B']}, C {insm['C']}."
        )
        x, y, va = main["x_pct_fewer_shortfalls_vs_A"], main["y_pct_margin_vs_B"], main["margin_vs_A_pct"]
        if main["supports_decided_form"]:
            spoken = (
                f"On data we never tuned on, our policy had {x:.1f} percent fewer cash-shortfall weeks than the standard reorder-point rule, "
                f"and {y:.1f} percent more margin than open-to-buy, with no more shortfall weeks than open-to-buy."
            )
        else:
            a = f"{abs(x):.1f} percent {'fewer' if x >= 0 else 'more'} cash-shortfall weeks than the standard reorder-point rule" if sfm["A"] else "no difference in shortfall weeks worth reporting against the reorder-point rule"
            b = f"{abs(y):.1f} percent {'more' if y >= 0 else 'less'} margin than open-to-buy"
            r = f"It was not a clean sweep: it had {sfm['C']} shortfall weeks to open-to-buy's {sfm['B']}" if not main["risk_equal_or_lower_than_B"] else f"It had no more shortfall weeks than open-to-buy, {sfm['C']} against {sfm['B']}"
            spoken = f"On data we never tuned on, our policy had {a}, and {b}. {r}, and it gave up {abs(va):.1f} percent of margin against buying regardless of cash."
        blocks["headline_spoken"] = spoken
    else:
        blocks["headline_numbers"] = PENDING
        blocks["headline_spoken"] = PENDING
    if include_demo:  # recomputes every bundled week, so the app skips it
        blocks.update(demo_block(run, cfg, exp))
    blocks["figures"] = "\n".join(f"![{f.stem}](results/{run.name}/figures/{f.name})" for f in sorted((run / "figures").glob("*.png"))) if (run / "figures").exists() else PENDING
    return blocks


def placeholder_blocks(cfg) -> dict[str, str]:
    keys = ["headline", "headline_plain", "headline_numbers", "headline_spoken", "verdict", "secondary_slice", "ablation_note", "results_table", "stress_table", "start_cash_table", "robustness_table", "forecast_table", "gate_diagnostics", "run_info", "figures", "c_vs_b", "demo_week", "demo_figure"]
    out = {k: PENDING for k in keys}
    out["project_name"] = cfg.project.name
    out["synthetic_params"] = synthetic_block(cfg)
    return out


def fill(text: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    used = []

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name not in blocks:
            return m.group(0)
        used.append(name)
        inline = "\n" not in m.group(2) and m.group(2).strip() == m.group(2)
        body = blocks[name]
        return f"<!-- BEGIN:{name} -->{body}<!-- END:{name} -->" if inline else f"<!-- BEGIN:{name} -->\n{body}\n<!-- END:{name} -->"

    return re.sub(r"<!-- BEGIN:(\w+) -->(.*?)<!-- END:\1 -->", repl, text, flags=re.S), used


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--run", default=None)
    ap.add_argument("--placeholder", action="store_true", help="write 'no results yet' blocks (still fills config-derived tables)")
    ap.add_argument("--out-dir", default=None, help="render copies of the target files here instead of in place")
    ap.add_argument("--allow-fixture", action="store_true", help="tests only: render a fixture run, and only together with --out-dir")
    args = ap.parse_args()

    if args.placeholder:
        cfg, _ = load_all(args)
        blocks = placeholder_blocks(cfg)
    else:
        run = resolve_run(args.run)
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["data_source"] != "M5" and not (args.allow_fixture and args.out_dir):
            print(f"REFUSING to render: run {manifest['run_id']} used data_source={manifest['data_source']}, not real M5 data.", file=sys.stderr)
            sys.exit(3)
        if manifest.get("partial_run"):
            print(f"REFUSING to render: run {manifest['run_id']} is a partial run (tune-only or quick).", file=sys.stderr)
            sys.exit(3)
        blocks = build_blocks(run)

    for rel in TARGETS:
        src = ROOT / rel
        if not src.exists():
            continue
        text, used = fill(src.read_text(), blocks)
        dst = Path(args.out_dir) / rel if args.out_dir else src
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        print(f"rendered {len(used)} block(s) into {dst}")


if __name__ == "__main__":
    main()
