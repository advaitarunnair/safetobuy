"""Draw docs/architecture.png (static copy of the Mermaid diagram in docs/architecture.md)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from _common import ROOT  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
LANES = [
    ("INPUTS", "#f0efec", 0.80),
    ("PREDICTION  (machine learning)", "#cde2fb", 0.605),
    ("DECISION  (simulation + optimization, not ML)", "#d5f0e5", 0.30),
    ("EVALUATION AND EXPLANATION", "#fbe7c8", 0.07),
]
BOXES = {
    "m5": (0.04, 0.83, 0.30, 0.10, "M5 weekly sales, prices, calendar\n(real, US Walmart)"),
    "syn": (0.38, 0.83, 0.585, 0.10, "Costs, lead times, packs, payment terms, fixed costs, cash\n(SYNTHETIC, seeded) -> cash model, allocator, simulated retailer"),
    "feat": (0.04, 0.635, 0.19, 0.10, "Past-only\nfeatures"),
    "lgb": (0.275, 0.635, 0.21, 0.10, "LightGBM quantile\nmodels (7 quantiles)"),
    "conf": (0.53, 0.635, 0.19, 0.10, "Conformal\ncalibration"),
    "cache": (0.765, 0.635, 0.20, 0.10, "Forecast cache\n(shared by all policies)"),
    "samp": (0.04, 0.42, 0.21, 0.10, "Joint demand sampler\n(cross-SKU correlation)"),
    "cash": (0.30, 0.42, 0.21, 0.10, "Monte Carlo\ncash model"),
    "gate": (0.56, 0.42, 0.19, 0.10, "Cash gate: largest safe\nbudget B (else best chance)"),
    "allocb": (0.30, 0.315, 0.285, 0.065, "Marginal-value allocator\nexpected profit per dollar, packs, MOQs"),
    "plan": (0.795, 0.42, 0.17, 0.10, "Plan per SKU:\nfull / partial / defer"),
    "base": (0.04, 0.10, 0.175, 0.10, "Baselines A, B, D\n+ two ablations"),
    "world": (0.25, 0.10, 0.185, 0.10, "Simulated retailer\n(actual M5 demand)"),
    "met": (0.47, 0.10, 0.165, 0.10, "Metrics, bootstrap,\nrun manifest"),
    "app": (0.67, 0.10, 0.125, 0.10, "Streamlit app,\nREADME"),
    "expl": (0.825, 0.10, 0.145, 0.10, "Facts -> templates\nor LLM (verified)"),
}
ARROWS = [
    ("m5", "feat"), ("feat", "lgb"), ("lgb", "conf"), ("conf", "cache"), ("cache", "samp"), ("samp", "cash"),
    ("cash", "gate"), ("gate", "plan"), ("plan", "expl"), ("base", "world"), ("world", "met"), ("met", "app"), ("expl", "app"), ("plan", "world"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.03, 1.0)
    ax.axis("off")
    for label, color, y in LANES:
        h = 0.17 if label != "DECISION  (simulation + optimization, not ML)" else 0.28
        ax.add_patch(FancyBboxPatch((0.015, y), 0.97, h, boxstyle="round,pad=0.004,rounding_size=0.012", linewidth=0, facecolor=color, alpha=0.55))
        left = label.startswith("EVALUATION")
        ax.text(0.03 if left else 0.975, y + h - 0.012, label, fontsize=9, color=INK2, fontweight="bold", va="top", ha="left" if left else "right")
    centres = {}
    for key, (x, y, w, h, text) in BOXES.items():
        if not text:
            continue
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.01", linewidth=1.1, edgecolor=MUTED, facecolor="white"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, color=INK)
        centres[key] = (x, y, w, h)

    def arrow(a, b, rad=0.0, text=None):
        ax_, ay, aw, ah = centres[a]
        bx, by, bw, bh = centres[b]
        if abs((ay + ah / 2) - (by + bh / 2)) < 0.05:  # same row
            start, end = ((ax_ + aw, ay + ah / 2), (bx, by + bh / 2)) if ax_ < bx else ((ax_, ay + ah / 2), (bx + bw, by + bh / 2))
        elif ay > by:  # downward
            sx = ax_ + aw / 2 + 0.02 if abs((ax_ + aw / 2) - (bx + bw / 2)) < 0.25 else ax_ + aw / 2 - 0.03
            start, end = (sx, ay), (bx + bw / 2, by + bh)
        else:
            start, end = (ax_ + aw / 2, ay + ah), (bx + bw / 2, by)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=INK2, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
        if text:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.012, text, fontsize=7.8, color=INK2, ha="center")

    for a, b in ARROWS:
        arrow(a, b)
    for start, end in (((0.655, 0.42), (0.575, 0.382)), ((0.345, 0.382), (0.405, 0.42))):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=INK2, shrinkA=1, shrinkB=1))
    ax.text(0.665, 0.392, "candidate B", fontsize=7.8, color=INK2)
    ax.text(0.255, 0.392, "plan at B", fontsize=7.8, color=INK2)
    ax.text(0.515, 0.535, "P(safe) at B", fontsize=7.8, color=INK2)
    ax.text(0.5, 0.975, "ML for prediction, simulation and optimization for the decision", ha="center", fontsize=13, fontweight="bold", color=INK)
    out = ROOT / "docs" / "architecture.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
