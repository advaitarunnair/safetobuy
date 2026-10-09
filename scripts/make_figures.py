"""Static figures for the README and write-up, drawn only from results/<run_id>/.

Usage: python scripts/make_figures.py [--run RUN_ID]
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import base_parser, resolve_run  # noqa: E402

from cspa.config import as_config  # noqa: E402
from cspa.sim.metrics import aggregate, headline_scope  # noqa: E402

# Categorical colours are bound to the entity (policy / model), never to its rank.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POLICY_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "C_plus": "#e87ba4", "C_gate_prop": "#008300", "OTB_marginal": "#4a3aa7"}
POLICY_LABEL = {
    "A": "A reorder point",
    "B": "B open-to-buy",
    "C": "C cash gate + allocator",
    "D": "D OTB + priority cuts",
    "C_plus": "C+ capital-aware, best effort",
    "C_gate_prop": "gate + proportional split",
    "OTB_marginal": "OTB + allocator",
}
MODEL_COLOR = {"lgbm_conformal": "#2a78d6", "lgbm_raw": "#eb6834", "seasonal_naive": "#1baf7a", "naive": "#eda100", "lgbm": "#2a78d6"}
MODEL_LABEL = {"lgbm_conformal": "LightGBM + conformal", "lgbm_raw": "LightGBM (raw)", "seasonal_naive": "Seasonal naive", "naive": "Naive", "lgbm": "LightGBM"}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
            "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
            "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
            "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
            "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "ytick.left": False,
            "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0,
        }
    )


def footer(fig, manifest: dict, note: str = "") -> None:
    src = "M5 demand (US Walmart), synthetic costs" if manifest["data_source"] == "M5" else "SYNTHETIC TEST FIXTURE - NOT A RESULT"
    fig.text(0.01, -0.04, f"{note}  Run {manifest['run_id']}. {src}.".strip(), fontsize=7.5, color=MUTED, ha="left", va="top")


def legend_top(ax, ncol: int) -> None:
    """Legend in a row above the plot, with the title lifted clear of it."""
    _, labels = ax.get_legend_handles_labels()
    rows = int(np.ceil(len(labels) / ncol))
    ax.legend(ncol=ncol, loc="lower left", bbox_to_anchor=(-0.01, 1.0), handlelength=1.2, columnspacing=1.4)
    ax.set_title(ax.get_title(loc="left"), pad=10 + 17 * rows)


def save(fig, path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(ax, table: pd.DataFrame, colors: dict, labels: dict, fmt: str) -> None:
    """table: index = groups on the x-axis, columns = series. Value labels only on the first and last group."""
    groups, series = list(table.index), list(table.columns)
    width = 0.8 / len(series)
    for j, s in enumerate(series):
        x = np.arange(len(groups)) + (j - (len(series) - 1) / 2) * width
        bars = ax.bar(x, table[s].to_numpy(), width=width * 0.86, color=colors[s], label=labels.get(s, s), linewidth=0)
        for k, b in enumerate(bars):
            if len(series) <= 4 or k in (0, len(groups) - 1):
                v = b.get_height()
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -9), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups)
    ax.axhline(0, color=AXIS, linewidth=1)


def fig_forecast(run, manifest) -> list[str]:
    path = run / "forecast_eval.csv"
    if not path.exists():
        return []
    ev = pd.read_csv(path)
    ev = ev[ev["scope"] == "all"]
    made = []
    allh = ev[ev["h"] == "all"].set_index("model")
    models = [m for m in ("lgbm_raw", "lgbm_conformal", "seasonal_naive", "naive", "lgbm") if m in allh.index]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    cov = allh.loc[models, ["cov50", "cov80", "cov90"]].T * 100
    cov.index = ["50% interval", "80% interval", "90% interval"]
    grouped_bars(ax, cov, MODEL_COLOR, MODEL_LABEL, "{:.0f}")
    for k, nominal in enumerate((50, 80, 90)):
        ax.hlines(nominal, k - 0.45, k + 0.45, color=INK, linewidth=1.2, linestyles=(0, (4, 2)))
        ax.annotate(f"target {nominal}%", (k + 0.45, nominal), xytext=(2, 0), textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    ax.set_ylabel("Share of actual weekly sales inside the interval (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Forecast intervals: how often reality lands inside them")
    legend_top(ax, 4)
    footer(fig, manifest, "Rolling-origin, backtest weeks, horizons 1-4.")
    save(fig, run / "figures" / "forecast_coverage.png")
    made.append("forecast_coverage.png")

    byh = ev[ev["h"] != "all"].copy()
    byh["h"] = byh["h"].astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in models:
        d = byh[byh["model"] == m].sort_values("h")
        ax.plot(d["h"], d["wql"], color=MODEL_COLOR[m], marker="o", markersize=5, label=MODEL_LABEL[m])
        ax.annotate(MODEL_LABEL[m], (d["h"].iloc[-1], d["wql"].iloc[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax.set_xticks(sorted(byh["h"].unique()))
    ax.set_xlabel("Weeks ahead")
    ax.set_ylabel("Weighted quantile loss (lower is better)")
    ax.set_ylim(0, None)
    ax.set_xlim(0.8, byh["h"].max() + 1.6)
    ax.set_title("Quantile forecast error by horizon")
    legend_top(ax, 4)
    footer(fig, manifest, "Mean pinball loss over 7 quantiles, x2, / mean demand.")
    save(fig, run / "figures" / "forecast_wql.png")
    made.append("forecast_wql.png")
    return made


def fig_backtest(run, manifest, exp) -> list[str]:
    metrics = pd.read_csv(run / "metrics.csv")
    weekly = pd.read_parquet(run / "weekly.parquet")
    scope = headline_scope(metrics, exp)
    if scope.empty:
        return []
    made = []
    pols = [p for p in POLICY_COLOR if p in set(scope["policy"])]
    core = [p for p in ("A", "B", "D", "C", "C_plus") if p in pols]
    by = aggregate(scope, ["stress"])
    scope_txt = f"{'+'.join(exp.headline.roles)} windows, opening cash {exp.default_start_cash_weeks:g} wks of fixed costs, rule: {exp.headline.shortfall_rule}."

    sf = by.pivot(index="stress", columns="policy", values="shortfall_weeks")[core]
    sf.index = [f"stress {int(round(s * 100))}%" for s in sf.index]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    grouped_bars(ax, sf, POLICY_COLOR, POLICY_LABEL, "{:.0f}")
    n_weeks = int(by.groupby("stress")["n_weeks"].first().iloc[0])
    ax.set_ylabel(f"Weeks with cash below the buffer (of {n_weeks})")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Cash-shortfall weeks by policy and budget stress")
    legend_top(ax, 3)
    footer(fig, manifest, scope_txt)
    save(fig, run / "figures" / "shortfalls_by_stress.png")
    made.append("shortfalls_by_stress.png")

    gm = by.pivot(index="stress", columns="policy", values="gross_margin")
    rel = (gm.div(gm["A"], axis=0) - 1.0) * 100
    rel = rel[[p for p in core if p != "A"]]
    rel.index = sf.index
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    grouped_bars(ax, rel, POLICY_COLOR, POLICY_LABEL, "{:+.1f}")
    ax.set_ylabel("Realised gross margin vs policy A (%)")
    ax.set_title("Margin given up (or gained) relative to buying regardless of cash")
    legend_top(ax, 3)
    footer(fig, manifest, scope_txt)
    save(fig, run / "figures" / "margin_by_stress.png")
    made.append("margin_by_stress.png")

    pooled = aggregate(scope, ["shortfall_rule"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for _, r in pooled.iterrows():
        if r["policy"] not in POLICY_COLOR:
            continue
        ax.scatter(r["shortfall_weeks"], r["gross_margin"] / 1000.0, s=90, color=POLICY_COLOR[r["policy"]], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(POLICY_LABEL[r["policy"]], (r["shortfall_weeks"], r["gross_margin"] / 1000.0), xytext=(8, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlabel("Cash-shortfall weeks (fewer is safer)")
    ax.set_ylabel("Realised gross margin ($ thousand)")
    ax.grid(True, axis="both")
    ax.margins(x=0.18, y=0.2)
    ax.set_title("Risk and return of each purchasing policy")
    footer(fig, manifest, scope_txt)
    save(fig, run / "figures" / "risk_return.png")
    made.append("risk_return.png")

    ab = exp.app_bundle
    one = weekly[(weekly["window"] == ab.window) & (weekly["stress"] == float(exp.default_stress)) & (weekly["start_cash_weeks"] == float(ab.start_cash_weeks)) & (weekly["shortfall_rule"] == ab.shortfall_rule)]
    if not one.empty:
        fig, ax = plt.subplots(figsize=(7.6, 3.9))
        for p in core:
            d = one[one["policy"] == p].sort_values("t")
            ax.plot(d["t"] + 1, d["cash_end"] / 1000.0, color=POLICY_COLOR[p], label=POLICY_LABEL[p])
        buf = float(one["buffer"].iloc[0]) / 1000.0
        ax.axhline(buf, color=INK, linewidth=1.2, linestyle=(0, (4, 2)))
        ax.annotate("cash buffer", (one["t"].max() + 1, buf), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK2)
        ax.set_xlabel(f"Week of window {ab.window}")
        ax.set_ylabel("End-of-week cash ($ thousand)")
        ax.set_xlim(1, one["t"].max() + 3.5)
        ax.set_title(f"Cash through one backtest window (stress {int(round(float(exp.default_stress) * 100))}%)")
        legend_top(ax, 3)
        footer(fig, manifest, f"Window {ab.window}, rule: {ab.shortfall_rule}.")
        save(fig, run / "figures" / "cash_paths.png")
        made.append("cash_paths.png")

    abl = [p for p in ("B", "OTB_marginal", "C_gate_prop", "C") if p in pols]
    if len(abl) == 4:
        pooled_all = aggregate(scope, ["shortfall_rule"]).set_index("policy")
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
        labels = ["OTB budget\nproportional\n(B)", "OTB budget\nallocator", "cash gate\nproportional", "cash gate\nallocator\n(C)"]
        gm_b = float(pooled_all.loc["B", "gross_margin"])
        panels = (
            (axes[0], pooled_all.loc[abl, "shortfall_weeks"].to_numpy(dtype=float), "Cash-shortfall weeks", "{:.0f}"),
            (axes[1], (pooled_all.loc[abl, "gross_margin"].to_numpy(dtype=float) / gm_b - 1.0) * 100.0, "Gross margin vs B (%)", "{:+.1f}"),
        )
        for ax, vals, title, fmt in panels:
            bars = ax.bar(range(4), vals, width=0.6, color=[POLICY_COLOR[p] for p in abl], linewidth=0)
            for b, v in zip(bars, vals):
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -10), textcoords="offset points", ha="center", fontsize=8, color=INK2)
            ax.set_xticks(range(4))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(title, fontsize=10.5)
            ax.axhline(0, color=AXIS, linewidth=1)
        axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].set_ylim(0, None)
        fig.suptitle("Ablation: what the cash gate and the allocator each contribute", x=0.01, ha="left", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        footer(fig, manifest, scope_txt)
        save(fig, run / "figures" / "ablation.png")
        made.append("ablation.png")
    return made


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = resolve_run(args.run)
    manifest = json.loads((run / "manifest.json").read_text())
    exp = as_config(manifest["experiments"])
    (run / "figures").mkdir(exist_ok=True)
    style()
    made = fig_forecast(run, manifest) + fig_backtest(run, manifest, exp)
    print(f"wrote {len(made)} figures to {run / 'figures'}: {', '.join(made)}")


if __name__ == "__main__":
    main()
