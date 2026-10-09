"""Backtest metrics, bootstrap intervals and the headline sentence.

Every number that reaches the README, the app or the write-up is produced here
from a real run. Nothing is typed by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIO_KEYS = ["scenario", "window", "role", "stress", "cash_cushion", "shortfall_rule"]


def summarise(weekly: pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario, policy)."""
    rows = []
    for keys, g in weekly.groupby(SCENARIO_KEYS + ["policy"], sort=False):
        g = g.sort_values("week")
        n = len(g)
        cogs_per_day = g["cogs"].sum() / (7.0 * n)
        row = dict(zip(SCENARIO_KEYS + ["policy"], keys))
        row.update(
            n_weeks=n,
            shortfall_weeks=int(g["shortfall"].sum()),
            insolvency_weeks=int(g["insolvent"].sum()),
            fill_rate=float(g["units_sold"].sum() / max(g["units_demand"].sum(), 1)),
            lost_margin=float(g["lost_margin"].sum()),
            gross_margin=float(g["gross_margin"].sum()),
            revenue=float(g["revenue"].sum()),
            total_spend=float(g["order_cost"].sum()),
            end_cash=float(g["cash_end"].iloc[-1]),
            end_net_position=float(g["net_position"].iloc[-1]),
            min_cash=float(g["cash_end"].min()),
            avg_inventory_value=float(g["inv_value"].mean()),
            avg_days_inventory=float(g["inv_value"].mean() / cogs_per_day) if cogs_per_day > 0 else np.nan,
            spoil_loss=float(g["spoil_loss"].sum()),
            order_cut_weeks=int(g["order_cut"].sum()),
            gate_binding_weeks=int((g["gate_status"] == "constrained").sum()),
            gate_at_risk_weeks=int((g["gate_status"] == "cash_at_risk").sum()),
            gate_violations=int(g["gate_violations"].fillna(0).sum()),
            gate_evals_mean=float(g["gate_evals"].mean()) if g["gate_evals"].notna().any() else np.nan,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(metrics: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Pool scenario-level metrics (sums for counts and dollars, weighted fill rate)."""
    g = metrics.groupby(by + ["policy"], sort=False)
    out = g.agg(
        n_runs=("scenario", "nunique"),
        n_weeks=("n_weeks", "sum"),
        shortfall_weeks=("shortfall_weeks", "sum"),
        insolvency_weeks=("insolvency_weeks", "sum"),
        lost_margin=("lost_margin", "sum"),
        gross_margin=("gross_margin", "sum"),
        total_spend=("total_spend", "sum"),
        end_net_position=("end_net_position", "mean"),
        avg_inventory_value=("avg_inventory_value", "mean"),
        avg_days_inventory=("avg_days_inventory", "mean"),
        fill_rate=("fill_rate", "mean"),
        min_cash=("min_cash", "min"),
        gate_binding_weeks=("gate_binding_weeks", "sum"),
        gate_at_risk_weeks=("gate_at_risk_weeks", "sum"),
        gate_violations=("gate_violations", "sum"),
    ).reset_index()
    return out


def _block_indices(T: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap indices for a series of length T."""
    block = max(1, min(block, T))
    starts = rng.integers(0, T - block + 1, size=int(np.ceil(T / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]


def _pct(num: float, den: float) -> float:
    return float(100.0 * num / den) if den else float("nan")


def headline(weekly: pd.DataFrame, n_boot: int, block_weeks: int, ci: float, seed: int, scope: str) -> dict:
    """Headline comparison of C against A (shortfalls) and B (margin), with paired block-bootstrap intervals.

    X = % fewer cash-shortfall weeks under C than under A.
    Y = % higher realised gross margin under C than under B.
    Weeks are resampled in blocks within each scenario, with the same blocks for every policy.
    """
    need = {"A", "B", "C"}
    have = set(weekly["policy"].unique())
    if not need <= have:
        raise ValueError(f"headline needs policies A, B and C (have {sorted(have)})")
    runs = []
    for _, g in weekly.groupby("scenario", sort=False):
        piv_s = g.pivot(index="t", columns="policy", values="shortfall").sort_index().astype(float)
        piv_i = g.pivot(index="t", columns="policy", values="insolvent").sort_index().astype(float)
        piv_m = g.pivot(index="t", columns="policy", values="gross_margin").sort_index()
        runs.append((piv_s[["A", "B", "C"]].to_numpy(), piv_m[["A", "B", "C"]].to_numpy(), piv_i[["A", "B", "C"]].to_numpy()))

    def stats(pick) -> dict:
        s = sum(pick(r[0], k) for k, r in enumerate(runs))
        m = sum(pick(r[1], k) for k, r in enumerate(runs))
        return {"sf": s, "gm": m, "x": _pct(s[0] - s[2], s[0]), "y": _pct(m[2] - m[1], m[1])}

    point = stats(lambda a, k: a.sum(axis=0))
    insolv = sum(r[2].sum(axis=0) for r in runs)
    rng = np.random.default_rng([int(seed), 77])
    xs, ys, risk = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = [_block_indices(r[0].shape[0], block_weeks, rng) for r in runs]
        st = stats(lambda a, k: a[idx[k]].sum(axis=0))
        xs[b], ys[b], risk[b] = st["x"], st["y"], st["sf"][2] - st["sf"][1]
    lo, hi = 100.0 * (1.0 - ci) / 2.0, 100.0 * (1.0 + ci) / 2.0

    def interval(v: np.ndarray) -> list[float] | None:
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if v.size else None

    sf, gm = point["sf"], point["gm"]
    out = {
        "scope": scope,
        "n_scenarios": len(runs),
        "n_weeks": int(sum(r[0].shape[0] for r in runs)),
        "ci_level": ci,
        "n_boot": int(n_boot),
        "block_weeks": int(block_weeks),
        "shortfall_weeks": {"A": int(sf[0]), "B": int(sf[1]), "C": int(sf[2])},
        "insolvency_weeks": {"A": int(insolv[0]), "B": int(insolv[1]), "C": int(insolv[2])},
        "gross_margin": {"A": float(gm[0]), "B": float(gm[1]), "C": float(gm[2])},
        "x_pct_fewer_shortfalls_vs_A": point["x"],
        "x_ci": interval(xs),
        "y_pct_margin_vs_B": point["y"],
        "y_ci": interval(ys),
        "margin_vs_A_pct": _pct(gm[2] - gm[0], gm[0]),
        "risk_equal_or_lower_than_B": bool(sf[2] <= sf[1]),
        "prob_risk_equal_or_lower_than_B": float(np.mean(risk <= 0)),
    }
    out["supports_decided_form"] = bool(sf[0] > 0 and point["x"] > 0 and point["y"] > 0 and out["risk_equal_or_lower_than_B"])
    out["sentence"] = headline_sentence(out)
    return out


def _ci_txt(ci_pair: list[float] | None, level: float, what: str) -> str:
    return f" ({level:.0%} bootstrap interval for the {what}: {ci_pair[0]:+.0f}% to {ci_pair[1]:+.0f}%)" if ci_pair else ""


def headline_sentence(h: dict) -> str:
    """The DECIDED headline form if the run supports it; otherwise a plain statement of what happened."""
    sf, x, y = h["shortfall_weeks"], h["x_pct_fewer_shortfalls_vs_A"], h["y_pct_margin_vs_B"]
    lvl = h["ci_level"]
    if h["supports_decided_form"]:
        return (
            f"{x:.0f}% fewer cash shortfalls than policy A{_ci_txt(h['x_ci'], lvl, 'reduction')}, "
            f"and {y:.1f}% higher margin than policy B{_ci_txt(h['y_ci'], lvl, 'difference')} at equal or lower risk."
        )
    if sf["A"] == 0:
        part_a = f"Policy A had no cash-shortfall weeks in this setting and policy C had {sf['C']}"
    else:
        part_a = f"Policy C had {abs(x):.0f}% {'fewer' if x >= 0 else 'more'} cash-shortfall weeks than policy A ({sf['C']} vs {sf['A']}){_ci_txt(h['x_ci'], lvl, 'reduction')}"
    part_b = f"{abs(y):.1f}% {'higher' if y >= 0 else 'lower'} margin than policy B{_ci_txt(h['y_ci'], lvl, 'difference')}"
    risk = "equal or lower" if h["risk_equal_or_lower_than_B"] else "higher"
    return f"{part_a}, and {part_b}, at {risk} shortfall risk than B ({sf['C']} vs {sf['B']} weeks). This run does not support the planned headline form, so it is reported as it happened."


def headline_scope(df: pd.DataFrame, exp) -> pd.DataFrame:
    """Rows (weekly or per-scenario) that feed the main headline: configured roles and rule,
    the budget-stress sweep at the default opening cash."""
    return df[df["role"].isin(list(exp.headline.roles)) & (df["shortfall_rule"] == str(exp.headline.shortfall_rule)) & (df["cash_cushion"] == float(exp.default_cash_cushion))]


def headline_set(weekly: pd.DataFrame, exp, seed: int) -> dict:
    """Headline for the configured scope, plus the cuts needed to judge how robust it is.

    Scope of the main headline: windows whose role is in exp.headline.roles, the three
    budget-stress levels at the default opening cash, under exp.headline.shortfall_rule.
    """
    bs = exp.bootstrap
    base = weekly[weekly["cash_cushion"] == float(exp.default_cash_cushion)]

    def hl(df: pd.DataFrame, scope: str, proposed: str = "C") -> dict | None:
        if df.empty or not {"A", "B", proposed} <= set(df["policy"].unique()):
            return None
        if proposed != "C":
            df = df[df["policy"] != "C"].copy()
            df["policy"] = df["policy"].replace({proposed: "C"})
        return headline(df, int(bs.n_boot), int(bs.block_weeks), float(bs.ci), seed, scope)

    roles = list(exp.headline.roles)
    main_rule = str(exp.headline.shortfall_rule)
    in_roles = base[base["role"].isin(roles)]
    main = in_roles[in_roles["shortfall_rule"] == main_rule]
    role_txt = "+".join(roles)
    out = {
        "main": hl(main, f"{role_txt} windows, stress sweep, {main_rule}"),
        "by_rule": {str(r): hl(in_roles[in_roles["shortfall_rule"] == r], f"{role_txt} windows, stress sweep, {r}") for r in sorted(in_roles["shortfall_rule"].unique())},
        "by_stress": {f"{s:g}": hl(main[main["stress"] == s], f"{role_txt} windows, stress {s:g}, {main_rule}") for s in sorted(main["stress"].unique())},
        "by_window": {str(w): hl(main[main["window"] == w], f"window {w}, stress sweep, {main_rule}") for w in sorted(main["window"].unique())},
        "other_roles": {},
    }
    for role in sorted(set(base["role"].unique()) - set(roles)):
        sub = base[(base["role"] == role) & (base["shortfall_rule"] == main_rule)]
        out["other_roles"][str(role)] = hl(sub, f"{role} windows, stress sweep, {main_rule}")
    return out
