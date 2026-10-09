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

TARGETS = ["README.md", "docs/project_description.md", "docs/demo_script.md"]
POLICY_ORDER = ["A", "B", "D", "C", "C_plus", "C_gate_prop", "OTB_marginal"]
POLICY_NAME = {
    "A": "A: reorder point",
    "B": "B: open-to-buy, proportional split",
    "D": "D: open-to-buy, priority cuts",
    "C": "**C: cash gate + marginal allocator (proposed)**",
    "C_plus": "C+: C with best-effort fallback",
    "C_gate_prop": "ablation: cash gate + proportional split",
    "OTB_marginal": "ablation: OTB budget + marginal allocator",
}
PENDING = "_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._"


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(" --- " if i == 0 else " ---: " for i in range(len(df.columns))) + "|"
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
                "X: fewer shortfalls than A": "n/a (A had none)" if sf["A"] == 0 else f"{h['x_pct_fewer_shortfalls_vs_A']:+.0f}%",
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
        ("Opening cash", "set per scenario, in weeks of average fixed costs", "experiments.start_cash_weeks"),
        ("Cash buffer", f"{cfg.risk.buffer_weeks_of_fixed_costs:g} weeks of average fixed costs", "risk.buffer_weeks_of_fixed_costs"),
        ("Opening stock, pipeline, payables", f"whatever policy A leaves after a {syn.burn_in_weeks}-week burn-in with ample cash", "synthetic.burn_in_weeks"),
        ("Seed", f"{cfg.seed}", "seed"),
    ]
    return md_table(pd.DataFrame(rows, columns=["Parameter", "Value", "Config key"]))


def build_blocks(run: Path) -> dict[str, str]:
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

    parts = []
    for s, g in aggregate(scope, ["stress"]).groupby("stress"):
        g = g.set_index("policy")
        row = {"Budget stress": f"{int(round(s * 100))}%"}
        for p in [p for p in ("A", "B", "D", "C", "C_plus") if p in g.index]:
            row[f"{p} shortfall wks"] = int(g.loc[p].shortfall_weeks)
        for p in [p for p in ("A", "B", "D", "C", "C_plus") if p in g.index]:
            row[f"{p} margin"] = usd(g.loc[p].gross_margin)
        parts.append(row)
    blocks["stress_table"] = md_table(pd.DataFrame(parts)) if parts else PENDING

    cash_rows = []
    cash_scope = metrics[metrics["role"].isin(list(exp.headline.roles)) & (metrics["shortfall_rule"] == str(exp.headline.shortfall_rule)) & (metrics["stress"] == float(exp.default_stress))]
    for c, g in aggregate(cash_scope, ["start_cash_weeks"]).groupby("start_cash_weeks"):
        g = g.set_index("policy")
        row = {"Opening cash (weeks of fixed costs)": f"{c:g}"}
        for p in [p for p in ("A", "B", "C", "C_plus") if p in g.index]:
            row[f"{p} shortfall wks"] = int(g.loc[p].shortfall_weeks)
        for p in [p for p in ("A", "B", "C", "C_plus") if p in g.index]:
            row[f"{p} margin"] = usd(g.loc[p].gross_margin)
        cash_rows.append(row)
    blocks["start_cash_table"] = md_table(pd.DataFrame(cash_rows)) if cash_rows else PENDING

    named = [(f"Headline: {roles} windows, rule {exp.headline.shortfall_rule}", main)]
    named += [(f"{roles} windows, rule {r}", h) for r, h in (hl.get("by_rule") or {}).items() if r != str(exp.headline.shortfall_rule)]
    named += [(f"{role} windows (used for tuning), rule {exp.headline.shortfall_rule}", h) for role, h in (hl.get("other_roles") or {}).items()]
    named += [(f"Window {w} only", h) for w, h in (hl.get("by_window") or {}).items()]
    named += [(f"Stress {int(round(float(s) * 100))}% only", h) for s, h in (hl.get("by_stress") or {}).items()]
    named += [("C replaced by C+ (best-effort fallback)", hl.get("variant_C_plus"))]
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
        f"Across all {gd['decisions']} policy-C decisions in this run the gate was binding in {gd['binding_weeks']}, flagged "
        f"\"cash at risk regardless of purchasing\" in {gd['cash_at_risk_weeks']}, and the coarse-grid check found "
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
        )
    )
    blocks["synthetic_params"] = synthetic_block(cfg)
    blocks["figures"] = "\n".join(f"![{f.stem}](results/{run.name}/figures/{f.name})" for f in sorted((run / "figures").glob("*.png"))) if (run / "figures").exists() else PENDING
    return blocks


def placeholder_blocks(cfg) -> dict[str, str]:
    keys = ["headline", "headline_plain", "results_table", "stress_table", "start_cash_table", "robustness_table", "forecast_table", "gate_diagnostics", "run_info", "figures"]
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
