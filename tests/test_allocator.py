import itertools

import numpy as np
import pytest

from cspa.allocate.constraints import round_down_to_pack, round_up_to_pack, satisfies
from cspa.allocate.marginal import allocate, build_chunks, cover_demand, marginal_values, plan_report


def instance(seed, n=12, n_paths=300, packs=(1, 6, 12), moq_mult=(1, 2)):
    rng = np.random.default_rng(seed)
    mean = rng.uniform(2, 60, n)
    cover = rng.poisson(mean, size=(n_paths, n)).astype(float)
    pos = np.floor(rng.uniform(0, 0.9, n) * mean).astype(np.int64)
    cost = np.round(rng.uniform(0.5, 9.0, n), 2)
    cu = cost * rng.uniform(0.2, 0.7, n)
    co = cost * rng.uniform(0.005, 0.3, n)
    pack = rng.choice(packs, n).astype(np.int64)
    moq = pack * rng.choice(moq_mult, n)
    return cover, pos, cost, cu, co, pack, moq


def chunk_values(table, qty):
    """Total expected profit of a plan, counting whole chunks in within-SKU order."""
    left, total = qty.copy(), 0.0
    for j in range(table.sku.size):
        s = table.sku[j]
        if left[s] >= table.qty[j]:
            left[s] -= table.qty[j]
            total += table.value[j]
    assert (left == 0).all(), "plan is not a union of leading chunks"
    return total


@pytest.mark.parametrize("seed", range(8))
def test_feasibility(seed):
    cover, pos, cost, cu, co, pack, moq = instance(seed)
    table = build_chunks(cover, pos, cost, cu, co, pack, moq)
    assert (table.value > 0).all()  # no chunk with non-positive marginal value is ever offered
    assert (np.diff(table.density) <= 1e-12).all()
    for B in [0.0, 1.0, 37.5, 250.0, 1000.0, table.total_cost * 0.5, table.total_cost, np.inf]:
        qty = allocate(table, B)
        assert qty.dtype == np.int64 and (qty >= 0).all()
        assert float(qty @ cost) <= B + 1e-6  # spend never exceeds the budget
        assert satisfies(qty, pack, moq)  # whole packs, MOQ respected
        assert (qty <= table.target_qty).all()  # never beyond the newsvendor target
        chunk_values(table, qty)
    assert np.array_equal(allocate(table, np.inf), table.target_qty)
    assert allocate(table, 0.0).sum() == 0


@pytest.mark.parametrize("seed", range(6))
def test_no_unit_with_non_positive_marginal_value_is_funded(seed):
    cover, pos, cost, cu, co, _, _ = instance(seed)
    ones = np.ones(len(cost), dtype=np.int64)
    table = build_chunks(cover, pos, cost, cu, co, ones, ones)
    srt = np.sort(cover, axis=0)
    for B in [50.0, 400.0, np.inf]:
        qty = allocate(table, B)
        for i in np.flatnonzero(qty):
            mv = marginal_values(srt[:, i], int(pos[i]), int(qty[i]), cu[i], co[i])
            assert (mv > 0).all()
        for i in range(len(cost)):  # and the unconstrained plan stops exactly where MV turns non-positive
            nxt = marginal_values(srt[:, i], int(pos[i] + table.target_qty[i]), 1, cu[i], co[i])
            assert nxt[0] <= 1e-12


def brute_force(cover, pos, cost, cu, co, budget, max_q):
    srt = np.sort(cover, axis=0)
    n = len(cost)
    cum = [np.concatenate([[0.0], np.cumsum(marginal_values(srt[:, i], int(pos[i]), max_q, cu[i], co[i]))]) for i in range(n)]
    best, best_q = -1.0, None
    for q in itertools.product(range(max_q + 1), repeat=n):
        if sum(qi * ci for qi, ci in zip(q, cost)) <= budget + 1e-9:
            v = sum(cum[i][q[i]] for i in range(n))
            if v > best + 1e-12:
                best, best_q = v, q
    return best, best_q


@pytest.mark.parametrize("seed", range(10))
def test_greedy_matches_brute_force_on_tiny_instances(seed):
    """No MOQs, single-unit packs. With equal unit costs the greedy is exactly optimal."""
    rng = np.random.default_rng(100 + seed)
    n, max_q = 4, 5
    cover = rng.poisson(rng.uniform(1, 4, n), size=(200, n)).astype(float)
    pos = np.zeros(n, dtype=np.int64)
    cost = np.full(n, 2.0)
    cu, co = rng.uniform(0.3, 1.5, n), rng.uniform(0.05, 0.8, n)
    ones = np.ones(n, dtype=np.int64)
    table = build_chunks(cover, pos, cost, cu, co, ones, ones)
    for budget in [2.0, 6.0, 11.0, 20.0, 1000.0]:
        qty = np.minimum(allocate(table, budget), max_q)
        if (allocate(table, budget) > max_q).any():
            continue  # outside the brute-force box
        best, _ = brute_force(cover, pos, cost, cu, co, budget, max_q)
        assert chunk_values(table, qty) == pytest.approx(best, abs=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_greedy_is_within_one_unit_of_optimal_with_unequal_costs(seed):
    """Unequal unit costs make this a knapsack; greedy-by-density is optimal for the continuous
    relaxation, so it can trail the integer optimum by less than the best single unit."""
    rng = np.random.default_rng(200 + seed)
    n, max_q = 4, 5
    cover = rng.poisson(rng.uniform(1, 4, n), size=(200, n)).astype(float)
    pos = np.zeros(n, dtype=np.int64)
    cost = np.round(rng.uniform(1.0, 5.0, n), 2)
    cu, co = cost * rng.uniform(0.2, 0.6, n), cost * rng.uniform(0.02, 0.3, n)
    ones = np.ones(n, dtype=np.int64)
    table = build_chunks(cover, pos, cost, cu, co, ones, ones)
    for budget in [3.0, 9.0, 17.0]:
        qty = allocate(table, budget)
        if (qty > max_q).any():
            continue
        best, _ = brute_force(cover, pos, cost, cu, co, budget, max_q)
        got = chunk_values(table, qty)
        assert got <= best + 1e-9
        assert got >= best - table.value.max() - 1e-9


def test_more_budget_never_buys_less_value():
    cover, pos, cost, cu, co, pack, moq = instance(3)
    table = build_chunks(cover, pos, cost, cu, co, pack, moq)
    vals = [chunk_values(table, allocate(table, B)) for B in np.linspace(0, table.total_cost, 25)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def test_cover_demand_sums_each_sku_over_its_own_cover_period():
    paths = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    out = cover_demand(paths, np.array([1, 2, 4]))
    assert np.array_equal(out[:, 0], paths[:, 0, 0])
    assert np.array_equal(out[:, 1], paths[:, 1, :2].sum(axis=1))
    assert np.array_equal(out[:, 2], paths[:, 2, :].sum(axis=1))


def test_pack_rounding_helpers():
    pack, moq = np.array([6, 6, 1, 12]), np.array([12, 6, 1, 12])
    assert np.array_equal(round_up_to_pack(np.array([1.0, 0.0, 2.2, 13.0]), pack, moq), [12, 0, 3, 24])
    assert np.array_equal(round_down_to_pack(np.array([11.0, 7.0, 2.9, 30.0]), pack, moq), [0, 6, 2, 24])
    assert satisfies(np.array([12, 0, 3, 24]), pack, moq)
    assert not satisfies(np.array([6, 0, 3, 24]), pack, moq)  # below MOQ
    assert not satisfies(np.array([12, 5, 3, 24]), pack, moq)  # not a whole pack


def test_plan_report_decisions_and_deferral_cost():
    cover, pos, cost, cu, co, pack, moq = instance(5)
    table = build_chunks(cover, pos, cost, cu, co, pack, moq)
    weeks = np.full(len(cost), 2)
    names = [f"S{i}" for i in range(len(cost))]
    full = plan_report(names, allocate(table, np.inf), table, cover, weeks)
    assert (full["decision"] == "full").all() and np.allclose(full["exp_margin_lost"], 0.0)
    none = plan_report(names, allocate(table, 0.0), table, cover, weeks)
    assert set(none.loc[none["needed"], "decision"]) == {"defer"}
    assert (none.loc[none["needed"], "exp_margin_lost"] > 0).all()
    half = plan_report(names, allocate(table, table.total_cost * 0.5), table, cover, weeks)
    assert {"full", "partial", "defer"} >= set(half["decision"])
    assert (half["stockout_prob"] >= half["stockout_prob_if_full"] - 1e-12).all()
    # deferral cost is computed from the demand distribution: Cu x extra expected lost sales
    i = int(np.flatnonzero((half["qty"] < half["target_qty"]).to_numpy())[0])
    a, f = pos[i] + half["qty"][i], pos[i] + half["target_qty"][i]
    expect = cu[i] * (np.maximum(cover[:, i] - a, 0).mean() - np.maximum(cover[:, i] - f, 0).mean())
    assert half["exp_margin_lost"][i] == pytest.approx(expect)
