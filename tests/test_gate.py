import numpy as np
import pytest

from cspa.cash.gate import STATUS_AT_RISK, STATUS_CONSTRAINED, STATUS_UNCONSTRAINED, safe_budget
from cspa.policies import PlanContext, make_policy

KW = dict(alpha=0.1, tol_frac=0.001, tol_abs=0.5, max_iter=30, n_grid=8)


def test_monotone_problem_finds_the_edge():
    res = safe_budget(lambda B: 1.0 if B <= 620.0 else 0.5, B_max=1000.0, **KW)
    assert res.status == STATUS_CONSTRAINED and res.monotonicity_violations == 0
    assert 620.0 - 1.0 <= res.B <= 620.0 and res.prob_safe >= 0.9
    assert res.prob_at_zero == 1.0 and res.prob_at_max == 0.5


def test_full_plan_safe_returns_it_in_one_evaluation():
    calls = []
    res = safe_budget(lambda B: calls.append(B) or 0.95, B_max=400.0, **KW)
    assert res.status == STATUS_UNCONSTRAINED and res.B == 400.0 and len(calls) == 1 and res.flag is None


def test_zero_budget_edge_case_is_flagged():
    """Even B = 0 misses the target: the result is flagged, and the budget is the one with the best chance."""
    falling = lambda B: 0.4 - B / 1e4  # every dollar spent makes things worse
    res = safe_budget(falling, B_max=800.0, **KW)
    assert res.B == 0.0 and res.status == STATUS_AT_RISK and res.prob_safe == pytest.approx(0.4)
    assert res.flag == "cash at risk regardless of purchasing"
    spec = safe_budget(lambda B: 0.4 + B / 1e4, B_max=800.0, on_infeasible="zero", **KW)  # original specification
    assert spec.B == 0.0 and spec.status == STATUS_AT_RISK and spec.flag is not None
    empty = safe_budget(lambda B: 0.4, B_max=0.0, **KW)
    assert empty.B == 0.0 and empty.status == STATUS_AT_RISK
    fine = safe_budget(lambda B: 0.99, B_max=0.0, **KW)
    assert fine.B == 0.0 and fine.status == STATUS_UNCONSTRAINED
    with pytest.raises(ValueError):
        safe_budget(lambda B: 0.4, B_max=800.0, on_infeasible="shrug", **KW)


def test_non_monotone_feasible_region_is_found_and_counted():
    """Buying a little raises cash (stock earns revenue): infeasible at 0, feasible in [200, 600]."""
    f = lambda B: 0.95 if 200.0 <= B <= 600.0 else 0.6
    res = safe_budget(f, B_max=1000.0, **KW)
    assert res.status == STATUS_CONSTRAINED and f(res.B) >= 0.9 and 550.0 <= res.B <= 600.0
    assert res.monotonicity_violations > 0


def test_grid_rescues_bisection_from_a_second_feasible_region():
    f = lambda B: 0.95 if B <= 100.0 or 700.0 <= B <= 760.0 else 0.5
    res = safe_budget(f, B_max=1000.0, alpha=0.1, tol_frac=0.001, tol_abs=0.5, max_iter=30, n_grid=9)
    assert f(res.B) >= 0.9 and res.B >= 700.0 and res.monotonicity_violations > 0


def test_infeasible_case_maximises_the_probability_of_staying_safe():
    """No budget reaches the target. Zero is not automatically safest: here the peak is at B = 310."""
    prob = lambda B: 0.7 - abs(B - 310.0) / 1000.0
    res = safe_budget(prob, B_max=900.0, n_refine=40, **KW)
    assert res.status == STATUS_AT_RISK and res.flag is not None
    assert abs(res.B - 310.0) <= 5.0 and res.prob_safe > prob(0.0) and res.prob_safe > prob(900.0)
    assert res.prob_safe == max(p for _, p in res.evals)  # the best of everything that was evaluated
    coarse = safe_budget(prob, B_max=900.0, n_refine=0, **KW)
    assert coarse.B in (300.0, 400.0) and res.prob_safe >= coarse.prob_safe
    # probability tied everywhere (already below the buffer): the tail measure decides, then the larger budget
    flat = safe_budget(lambda B: 0.0, B_max=900.0, tail=lambda B: (-abs(B - 500.0), 0.0), **KW)
    assert flat.B == pytest.approx(500.0)
    second = safe_budget(lambda B: 0.0, B_max=900.0, tail=lambda B: (1.0, -abs(B - 200.0)), **KW)
    assert second.B == pytest.approx(200.0)
    tied = safe_budget(lambda B: 0.0, B_max=900.0, tail=lambda B: (1.0, 1.0), **KW)
    assert tied.B == 900.0


@pytest.mark.parametrize("policy", ["C", "C_gate_prop"])
def test_returned_budget_satisfies_the_constraint_on_the_same_random_numbers(policy, week_setup):
    state, info = week_setup
    ctx = PlanContext(state, info)
    orders = make_policy(policy).decide(state, info)
    gate = orders.meta["gate"]
    replay = ctx.simulate(orders.qty).prob_above(info.buffer)  # same demand paths, recomputed from scratch
    assert replay == pytest.approx(gate.prob_safe, abs=1e-12)
    assert orders.meta["plan_cost"] <= gate.B + 1e-6
    if gate.status != STATUS_AT_RISK:
        assert replay >= 1.0 - info.alpha - 1e-12
    assert gate.n_evals >= 1 and gate.monotonicity_violations >= 0


def test_safe_budget_grows_with_cash(week_setup):
    state, info = week_setup
    rich = state.copy()
    rich.cash = 1e9
    res = make_policy("C").decide(rich, info).meta["gate"]
    assert res.status == STATUS_UNCONSTRAINED and res.B == pytest.approx(res.B_max)
    weekly_cost = float(info.cover.mean(axis=0) @ info.params.unit_cost)
    budgets, statuses = [], []
    for cash in np.linspace(-2 * weekly_cost, 3 * weekly_cost, 41):
        st = state.copy()
        st.cash = float(cash)
        g = make_policy("C").decide(st, info).meta["gate"]
        budgets.append(g.B)
        statuses.append(g.status)
    tol = max(info.cfg.gate.tol_abs, info.cfg.gate.tol_frac * res.B_max) + 1e-6
    assert statuses[0] == STATUS_AT_RISK
    assert budgets[-1] == pytest.approx(res.B_max) and statuses[-1] == STATUS_UNCONSTRAINED
    first_ok = next(i for i, s in enumerate(statuses) if s != STATUS_AT_RISK)
    assert all(s != STATUS_AT_RISK for s in statuses[first_ok:])  # once some budget is safe, more cash keeps it so
    safe_budgets = budgets[first_ok:]
    assert all(b >= a - tol for a, b in zip(safe_budgets, safe_budgets[1:]))  # and more cash never shrinks the safe budget


def test_gate_binds_when_purchases_cannot_pay_for_themselves(week_setup, overrides):
    """Deterministic gate on a no-sales path: every dollar spent only drains cash, so the safe
    budget is limited by the cash headroom above the buffer after committed payments."""
    from cspa.config import deep_merge, load_config

    state, info = week_setup
    info.cfg = load_config(overrides=deep_merge(overrides, {"gate": {"mode": "deterministic", "tol_abs": 1.0, "tol_frac": 0.0001, "max_bisection_iter": 40}}))
    info.__dict__["det_path"] = np.zeros((1, info.params.n, info.q.shape[1]))
    ctx = PlanContext(state, info)
    H = len(info.fixed_costs)
    committed = np.cumsum(info.fixed_costs + state.payables[:H])
    st = state.copy()
    st.cash = float(info.buffer + committed[-1] + 0.4 * ctx.table.total_cost)
    orders = make_policy("C").decide(st, info)
    gate = orders.meta["gate"]
    assert gate.status == STATUS_CONSTRAINED and 0.0 < gate.B < gate.B_max
    assert PlanContext(st, info).simulate(orders.qty).min_cash[0] >= info.buffer - 1e-6
    bigger = PlanContext(st, info).allocator("marginal")(min(gate.B_max, gate.B * 1.25 + 200.0))
    assert PlanContext(st, info).simulate(bigger).min_cash[0] < info.buffer  # and a larger budget would breach it


def test_hopeless_cash_position_is_flagged_and_gets_the_best_available_plan(week_setup, overrides):
    from cspa.config import deep_merge, load_config

    state, info = week_setup
    broke = state.copy()
    broke.cash = info.buffer - 0.35 * float(info.cover.mean(axis=0) @ info.params.unit_cost)
    orders = make_policy("C").decide(broke, info)
    gate = orders.meta["gate"]
    assert gate.status == STATUS_AT_RISK and gate.flag == "cash at risk regardless of purchasing"
    ctx = PlanContext(broke, info)
    p_plan = ctx.simulate(orders.qty).prob_above(info.buffer)
    p_zero = ctx.simulate(np.zeros(info.params.n)).prob_above(info.buffer)
    p_full = ctx.simulate(ctx.table.target_qty).prob_above(info.buffer)
    assert p_plan == pytest.approx(gate.prob_safe) and p_plan >= p_zero and p_plan >= p_full  # same random numbers
    assert p_plan < 1.0 - info.alpha
    assert p_plan == max(p for _, p in gate.evals)

    info.cfg = load_config(overrides=deep_merge(overrides, {"gate": {"on_infeasible": "zero"}}))
    spec = make_policy("C").decide(broke, info)
    assert spec.meta["gate"].B == 0.0 and spec.qty.sum() == 0 and spec.meta["gate"].flag is not None


def test_buying_stock_that_sells_can_be_safer_than_buying_nothing(week_setup):
    """The reason zero is not the default: with thin stock, a plan that keeps the shelves full
    brings in more cash than it costs, so P(safe) is higher than with no order at all."""
    state, info = week_setup
    thin = state.copy()
    thin.on_hand = (thin.on_hand * 0.3).astype(np.int64)
    thin.pipeline[:] = 0
    weekly_cost = float(info.cover.mean(axis=0) @ info.params.unit_cost) / 2.4
    found = False
    for cushion in np.linspace(-0.6, 0.6, 13):
        thin.cash = float(info.buffer + cushion * weekly_cost)
        ctx = PlanContext(thin, info)
        gate = ctx.gate(ctx.allocator("marginal"))
        p_zero = ctx.simulate(np.zeros(info.params.n)).prob_above(info.buffer)
        assert gate.prob_safe >= p_zero - 1e-12
        if gate.B > 0 and gate.prob_safe > p_zero + 0.05:
            found = True
    assert found


def test_deterministic_gate_fallback(week_setup, cfg, overrides):
    from cspa.config import deep_merge, load_config

    state, info = week_setup
    info.cfg = load_config(overrides=deep_merge(overrides, {"gate": {"mode": "deterministic"}}))
    assert info.sim_paths.shape[0] == 1
    orders = make_policy("C").decide(state, info)
    gate = orders.meta["gate"]
    assert gate.prob_safe in (0.0, 1.0)
    if gate.status != STATUS_AT_RISK:
        assert PlanContext(state, info).simulate(orders.qty).min_cash[0] >= info.buffer
