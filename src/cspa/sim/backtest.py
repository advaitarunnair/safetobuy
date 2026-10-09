"""Closed-loop backtest: every policy acts in the same simulated world.

All policies share the same realised demand (actual M5 weekly units), the same
synthetic parameters, the same forecasts and the same sampled demand paths.
They only differ in what they order.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cspa.data.load_m5 import Panel
from cspa.data.synth_params import SkuParams
from cspa.forecast.cache import ForecastCache
from cspa.forecast.sampling import build_pit_bank, sample_demand
from cspa.policies import InfoSet, make_policy
from cspa.sim.world import World, WorldState


@dataclass(frozen=True)
class Window:
    name: str
    start: int
    n_weeks: int
    role: str

    @property
    def end(self) -> int:
        """Last decision week (inclusive)."""
        return self.start + self.n_weeks - 1


@dataclass(frozen=True)
class Scenario:
    window: Window
    stress: float
    cash_cushion: float
    shortfall_rule: str

    @property
    def id(self) -> str:
        return f"{self.window.name}_s{int(round(self.stress * 100))}_c{self.cash_cushion:g}_{self.shortfall_rule}"

    def fields(self) -> dict:
        return {
            "scenario": self.id,
            "window": self.window.name,
            "role": self.window.role,
            "stress": self.stress,
            "cash_cushion": self.cash_cushion,
            "shortfall_rule": self.shortfall_rule,
        }


def resolve_windows(exp, n_obs: int) -> list[Window]:
    out = []
    for w in exp.windows:
        start = n_obs - int(w["weeks_before_end"])
        if start < 0 or start + int(w["n_weeks"]) > n_obs:
            raise ValueError(f"window {w['name']} does not fit in {n_obs} observed weeks")
        out.append(Window(str(w["name"]), start, int(w["n_weeks"]), str(w["role"])))
    return out


def first_decision_week(exp, cfg, n_obs: int) -> int:
    """Earliest week any policy decides in (burn-in included)."""
    return min(w.start for w in resolve_windows(exp, n_obs)) - int(cfg.synthetic.burn_in_weeks)


def selection_end_week(exp, cfg, n_obs: int) -> int:
    """SKUs are selected on data strictly before the first simulated decision."""
    return first_decision_week(exp, cfg, n_obs) - 1


def required_cutoffs(exp, cfg, n_obs: int) -> np.ndarray:
    """Forecast cutoffs the cache must hold: warm-up, burn-in and every decision week."""
    wins = resolve_windows(exp, n_obs)
    first = first_decision_week(exp, cfg, n_obs) - 1 - int(cfg.forecast.warmup_weeks)
    last = max(w.end for w in wins) - 1
    return np.arange(first, last + 1)


def decision_cutoffs(exp, n_obs: int, roles: tuple[str, ...] | None = None) -> np.ndarray:
    """Cutoffs behind in-window decisions (the forecast evaluation period)."""
    out: set[int] = set()
    for w in resolve_windows(exp, n_obs):
        if roles is None or w.role in roles:
            out.update(range(w.start - 1, w.end))
    return np.array(sorted(out), dtype=np.int64)


def scenario_grid(exp, n_obs: int) -> list[Scenario]:
    """Stress sweep at the default opening cash, plus an opening-cash sweep at the default stress."""
    wins = resolve_windows(exp, n_obs)
    combos = [(float(s), float(exp.default_cash_cushion)) for s in exp.stress_levels]
    combos += [(float(exp.default_stress), float(c)) for c in exp.cash_cushions if float(c) != float(exp.default_cash_cushion)]
    return [Scenario(w, s, c, r) for r in exp.shortfall_rules for w in wins for s, c in combos]


@dataclass
class WorldCal:
    """Scenario-level SYNTHETIC cash parameters, calibrated on weeks before the window."""

    window_start: int
    rev_star: float  # mean weekly revenue at M5 prices, pre-window
    repl_star: float  # ideal weekly replenishment cost R* (COGS at synthetic cost)
    gm_star: float
    fixed_avg: float  # average weekly fixed costs
    fixed_base: float  # paid every week
    lump: float  # paid on top, once per cycle
    lump_every: int
    lump_offset: int
    buffer: float
    start_cash: float
    stress: float
    stress_attainable: bool

    def fixed(self, week: int) -> float:
        return self.fixed_base + (self.lump if (week - self.window_start) % self.lump_every == self.lump_offset else 0.0)

    def schedule(self, week: int, H: int) -> np.ndarray:
        return np.array([self.fixed(week + j) for j in range(H)])


def calibrate_world(panel: Panel, params: SkuParams, cfg, window_start: int, stress: float, cash_cushion: float) -> WorldCal:
    """Turn a stress level and an opening-cash level into dollars.

    R*  = mean weekly cost of goods sold (synthetic unit cost x real units) over the
          `synthetic.calibration_weeks` weeks before the window: what an order-up-to
          policy must spend per week to replace what sells.
    Average weekly fixed costs = fixed_cost_share_of_gross_margin x mean weekly gross margin.
    They are paid as a base amount every week plus a lump once every
    `fixed_cost_lump_every_weeks` weeks (rent / monthly payroll).
    Stress s sizes the lump so that, in a lump week,
        expected revenue - fixed costs = s x R*.
    Buffer       = risk.buffer_weeks_of_fixed_costs x average weekly fixed costs.
    Opening cash = buffer + cash_cushion x s x R*: the free cash above the buffer also
    shrinks with s, so a lower s is tighter in both the lump and the starting position.
    """
    syn = cfg.synthetic
    k = int(syn.calibration_weeks)
    lo = window_start - k
    if lo < 0:
        raise ValueError("not enough history before the window to calibrate the world")
    units = panel.units[lo:window_start]
    price = np.nan_to_num(panel.price[lo:window_start], nan=0.0)
    rev = float((units * price).sum(axis=1).mean())
    repl = float((units @ params.unit_cost).mean())
    gm = rev - repl
    phi = float(syn.fixed_cost_share_of_gross_margin)
    m, off = int(syn.fixed_cost_lump_every_weeks), int(syn.fixed_cost_lump_offset)
    fixed_avg = phi * gm
    attainable = True
    if m >= 2:
        lump = ((1.0 - stress) * repl + (1.0 - phi) * gm) * m / (m - 1.0)
        if lump < 0.0:
            lump, attainable = 0.0, False
        if lump > m * fixed_avg:
            lump, attainable = m * fixed_avg, False
    else:
        lump, attainable = 0.0, False
    base = fixed_avg - lump / m if m >= 2 else fixed_avg
    return WorldCal(
        window_start=window_start,
        rev_star=rev,
        repl_star=repl,
        gm_star=gm,
        fixed_avg=fixed_avg,
        fixed_base=base,
        lump=lump,
        lump_every=m,
        lump_offset=off,
        buffer=float(cfg.risk.buffer_weeks_of_fixed_costs) * fixed_avg,
        start_cash=float(cfg.risk.buffer_weeks_of_fixed_costs) * fixed_avg + float(cash_cushion) * float(stress) * repl,
        stress=float(stress),
        stress_attainable=attainable,
    )


def policy_price(panel: Panel, params: SkuParams, cutoff: int) -> np.ndarray:
    """Last known sell price at the cutoff (reference price if the SKU has never been listed)."""
    p = panel.price[cutoff]
    return np.where(np.isnan(p), params.ref_price, p)


def make_info(panel: Panel, cache: ForecastCache, params: SkuParams, cfg, week: int, fixed_costs: np.ndarray, buffer: float, alpha: float | None = None, n_paths: int | None = None, method: str | None = None) -> InfoSet:
    """Build the information set for the decision in `week`. Reads nothing dated after week - 1."""
    cutoff = week - 1
    units_past = panel.units[: cutoff + 1]
    q = cache.get(cutoff, "cal")
    sm = cfg.sampling
    n_paths = int(sm.n_paths_backtest if n_paths is None else n_paths)
    method = str(sm.method if method is None else method)

    def paths_fn() -> np.ndarray:
        bank = build_pit_bank(cache.cal, cache.cutoffs, units_past, cutoff, int(sm.bank_weeks), float(sm.tail_cap))
        rng = np.random.default_rng([int(cfg.seed), int(week)])
        return sample_demand(q, bank, n_paths, rng, method, int(sm.block_len), float(sm.tail_cap))

    return InfoSet(
        week=week,
        cutoff=cutoff,
        q=q,
        price=policy_price(panel, params, cutoff),
        fixed_costs=np.asarray(fixed_costs, dtype=float),
        buffer=float(buffer),
        alpha=float(cfg.risk.alpha if alpha is None else alpha),
        params=params,
        cfg=cfg,
        paths_fn=paths_fn,
    )


def world_price(panel: Panel, params: SkuParams, week: int) -> np.ndarray:
    """Actual sell price in the simulated week (unlisted SKUs have zero demand anyway)."""
    p = panel.price[week]
    return np.where(np.isnan(p), params.ref_price, p)


def burn_in_state(panel: Panel, cache: ForecastCache, params: SkuParams, cfg, window_start: int) -> WorldState:
    """Run policy A with ample cash for `burn_in_weeks` so every policy starts from a
    realistic stock, pipeline and payables position. Cash is reset by the caller."""
    burn = int(cfg.synthetic.burn_in_weeks)
    t0 = window_start - burn
    H = int(cfg.horizons.cash_horizon_weeks)
    world = World(params, "overdraft")
    lookback = panel.units[max(t0 - 8, 0) : t0]
    mean_week = lookback.mean(axis=0) if lookback.size else np.zeros(panel.n_skus)
    on_hand = np.ceil((params.lead_time + 1) * mean_week).astype(np.int64)
    state = world.initial_state(t0, 1e12, on_hand)
    pol = make_policy("A")
    for t in range(t0, window_start):
        info = make_info(panel, cache, params, cfg, t, np.zeros(H), 0.0)
        state, _ = world.step(state, pol.decide(state, info), panel.units[t], world_price(panel, params, t), 0.0, 0.0)
    return state


def run_scenario(panel: Panel, params: SkuParams, cache: ForecastCache, cfg, scenario: Scenario, policy_names: list[str], capture_policy: str | None = None) -> dict:
    """Simulate every policy through one scenario. Returns weekly records, the world
    calibration, and (optionally) the pre-decision states of `capture_policy` for the app."""
    w = scenario.window
    if w.end >= panel.n_obs:
        raise ValueError("window runs past the observed data")
    cal = calibrate_world(panel, params, cfg, w.start, scenario.stress, scenario.cash_cushion)
    H = int(cfg.horizons.cash_horizon_weeks)
    world = World(params, scenario.shortfall_rule)
    start = burn_in_state(panel, cache, params, cfg, w.start)
    start.cash = cal.start_cash
    policies = {name: make_policy(name) for name in policy_names}
    states = {name: start.copy() for name in policy_names}
    hashes = {name: hashlib.sha1() for name in policy_names}
    rows, captured = [], []
    for t in range(w.start, w.end + 1):
        info = make_info(panel, cache, params, cfg, t, cal.schedule(t, H), cal.buffer)
        demand, price, fixed = panel.units[t], world_price(panel, params, t), cal.fixed(t)
        for name, pol in policies.items():
            st = states[name]
            if name == capture_policy:
                captured.append({"week": t, "cash": st.cash, "on_hand": st.on_hand.copy(), "pipeline": st.pipeline.copy(), "payables": st.payables.copy(), "spoil_acc": st.spoil_acc.copy()})
            orders = pol.decide(st, info)
            hashes[name].update(demand.tobytes() + price.tobytes())
            states[name], rec = world.step(st, orders, demand, price, fixed, cal.buffer)
            row = {k: v for k, v in rec.items() if not k.startswith("_")}
            row.update(scenario.fields())
            row.update(policy=name, t=t - w.start, budget=orders.meta.get("budget"), plan_cost=orders.meta.get("plan_cost"))
            gate = orders.meta.get("gate")
            row.update(
                gate_status=gate.status if gate else None,
                gate_prob_safe=gate.prob_safe if gate else None,
                gate_evals=gate.n_evals if gate else None,
                gate_violations=gate.monotonicity_violations if gate else None,
                gate_full_cost=gate.B_max if gate else None,
            )
            rows.append(row)
    weekly = pd.DataFrame(rows)
    weekly["budget"] = weekly["budget"].replace([np.inf], np.nan)
    return {
        "scenario": scenario,
        "weekly": weekly,
        "cal": cal,
        "captured": captured,
        "start_state": start,
        "demand_hash": {name: h.hexdigest() for name, h in hashes.items()},
        "params_hash": hashlib.sha1(params.to_frame().to_csv(index=False).encode()).hexdigest(),
    }
