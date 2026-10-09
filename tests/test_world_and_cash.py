import numpy as np
import pytest

from conftest import make_state

from cspa.cash.simulate import simulate_cash
from cspa.sim.world import Orders, World


def run_weeks(world, state, panel, params, start, n, order_fn, fixed=500.0, buffer=1000.0):
    recs, states = [], [state]
    for t in range(start, start + n):
        price = np.where(np.isnan(panel.price[t]), params.ref_price, panel.price[t])
        state, rec = world.step(state, Orders(order_fn(state, t)), panel.units[t], price, fixed, buffer)
        recs.append(rec)
        states.append(state)
    return recs, states


@pytest.mark.parametrize("rule", ["overdraft", "cut_proportional"])
def test_cash_and_inventory_conservation(rule, panel, params):
    """Every week: cash change = revenue - payments - fixed costs; stock change = arrivals - sales
    (minus write-offs, which only exist for perishable SKUs and earn salvage revenue)."""
    world = World(params, rule)
    start = 200
    rng = np.random.default_rng(0)
    state = make_state(panel, params, start, cash=3000.0)
    order_fn = lambda st, t: params.pack_size * rng.integers(0, 6, params.n)
    prev = state
    for t in range(start, start + 20):
        price = np.where(np.isnan(panel.price[t]), params.ref_price, panel.price[t])
        new, rec = world.step(prev, Orders(order_fn(prev, t)), panel.units[t], price, 450.0, 1000.0)
        assert rec["cash_end"] - rec["cash_start"] == pytest.approx(rec["revenue"] + rec["salvage"] - rec["supplier_paid"] - rec["fixed"], abs=1e-6)
        delta = new.on_hand - prev.on_hand
        assert np.array_equal(delta, rec["_arrivals"] - rec["_sales"] - rec["_spoiled"])
        assert (rec["_spoiled"][~params.perishable] == 0).all()
        assert np.array_equal(delta[~params.perishable], (rec["_arrivals"] - rec["_sales"])[~params.perishable])
        assert (new.on_hand >= 0).all() and (rec["_sales"] <= panel.units[t]).all()
        assert rec["supplier_paid"] == pytest.approx(prev.payables[0])
        assert rec["shortfall"] == (rec["cash_end"] < 1000.0) and rec["insolvent"] == (rec["cash_end"] < 0)
        prev = new


def test_net_position_changes_by_margin_minus_fixed_costs(panel, params):
    """Cash + stock at cost + pipeline at cost - payables moves only by gross margin and fixed costs."""
    world = World(params, "overdraft")
    rng = np.random.default_rng(1)
    state = make_state(panel, params, 210, cash=5000.0)
    recs, _ = run_weeks(world, state, panel, params, 210, 15, lambda st, t: params.pack_size * rng.integers(0, 5, params.n))
    for a, b in zip(recs, recs[1:]):
        assert b["net_position"] - a["net_position"] == pytest.approx(b["gross_margin"] - b["fixed"], abs=1e-6)


def test_orders_arrive_after_lead_time_and_are_paid_on_terms(panel, params):
    world = World(params, "overdraft")
    state = world.initial_state(220, 0.0, np.zeros(params.n))
    qty = params.moq.copy()
    zero_demand = np.zeros(params.n)
    price = params.ref_price
    state, rec = world.step(state, Orders(qty), zero_demand, price, 0.0, 0.0)
    assert rec["order_cost"] == pytest.approx(float(qty @ params.unit_cost)) and rec["cash_end"] == 0.0
    arrived, paid = np.zeros(params.n, dtype=np.int64), {}
    for k in range(1, 9):
        state, rec = world.step(state, Orders(np.zeros(params.n, dtype=np.int64)), zero_demand, price, 0.0, 0.0)
        for L in np.unique(params.lead_time):
            if k == L:
                assert np.array_equal(rec["_arrivals"][params.lead_time == L], qty[params.lead_time == L])
        arrived += rec["_arrivals"]
        paid[k] = rec["supplier_paid"]
    assert np.array_equal(arrived, qty)
    due = params.lead_time + params.pay_delay
    for k, amount in paid.items():
        assert amount == pytest.approx(float((qty * params.unit_cost)[due == k].sum()))


def test_lost_sales_are_not_backordered(params):
    world = World(params, "overdraft")
    state = world.initial_state(0, 0.0, np.full(params.n, 3))
    demand = np.full(params.n, 10)
    state, rec = world.step(state, Orders(np.zeros(params.n, dtype=np.int64)), demand, params.ref_price, 0.0, 0.0)
    assert rec["units_sold"] == 3 * params.n and rec["units_lost"] == 7 * params.n
    state, rec = world.step(state, Orders(np.zeros(params.n, dtype=np.int64)), np.zeros(params.n), params.ref_price, 0.0, 0.0)
    assert rec["units_sold"] == 0  # the 7 lost units did not come back as demand


def test_cut_rule_scales_orders_down_to_cash_on_hand(params):
    qty = params.moq * 4
    cost = float(qty @ params.unit_cost)
    zero = np.zeros(params.n)
    _, free = World(params, "overdraft").step(World(params).initial_state(0, 0.4 * cost, zero), Orders(qty), zero, params.ref_price, 0.0, 0.0)
    assert free["order_cost"] == pytest.approx(cost) and not free["order_cut"]
    w = World(params, "cut_proportional")
    _, rec = w.step(w.initial_state(0, 0.4 * cost, zero), Orders(qty), zero, params.ref_price, 0.0, 0.0)
    assert rec["order_cut"] and rec["order_cost"] <= 0.4 * cost + 1e-9 and rec["order_cost_requested"] == pytest.approx(cost)
    assert (rec["_qty"] % params.pack_size == 0).all() and ((rec["_qty"] == 0) | (rec["_qty"] >= params.moq)).all()
    _, rec = w.step(w.initial_state(0, -50.0, zero), Orders(qty), zero, params.ref_price, 0.0, 0.0)
    assert rec["order_cost"] == 0.0


def test_world_rejects_malformed_orders(params):
    w = World(params)
    st = w.initial_state(0, 0.0, np.zeros(params.n))
    for bad in (np.full(params.n, -1), np.full(params.n, 0.5), np.zeros(params.n + 1)):
        with pytest.raises(ValueError):
            w.step(st, Orders(bad), np.zeros(params.n), params.ref_price, 0.0, 0.0)


def test_cash_projection_matches_the_world_on_a_known_demand_path(panel, params):
    """With no later purchases, simulate_cash must reproduce World.step exactly on the same demand
    (non-perishable SKUs: the projection treats spoilage as a fraction, the world in whole units)."""
    import copy

    p = copy.deepcopy(params)
    p.perishable[:] = False
    p.spoilage_rate[:] = 0.0
    p.salvage_value[:] = 0.0
    world, H, start = World(p), 4, 230
    state = make_state(panel, p, start, cash=2500.0)
    plan = p.moq * 2
    fixed = np.array([300.0, 300.0, 900.0, 300.0])
    price = np.where(np.isnan(panel.price[start - 1]), p.ref_price, panel.price[start - 1])
    demand = panel.units[start : start + H]
    proj = simulate_cash(state, plan, demand.T[None, :, :], price, p, fixed, continuation="none", charge_beyond_horizon=False)
    st, cash = state.copy(), []
    for j in range(H):
        order = plan if j == 0 else np.zeros(p.n, dtype=np.int64)
        st, rec = world.step(st, Orders(order), demand[j], price, fixed[j], 0.0)
        cash.append(rec["cash_end"])
    assert np.allclose(proj.cash[0], cash)
    assert proj.min_cash[0] == pytest.approx(min(cash)) and proj.plan_cost == pytest.approx(float(plan @ p.unit_cost))


def test_late_bills_option_and_validation(week_setup, overrides):
    """Terms that reach past the horizon are rejected by the config, or charged in its last week on request."""
    import copy

    from cspa.config import ConfigError, deep_merge, load_config

    net30 = {"synthetic": {"suppliers": [{"name": "S_net30", "share": 1.0, "pay_delay_weeks": 4}]}}
    with pytest.raises(ConfigError, match="buying on credit looks free"):
        load_config(overrides=deep_merge(overrides, net30))
    load_config(overrides=deep_merge(overrides, deep_merge(net30, {"gate": {"charge_beyond_horizon": True}})))

    state, info = week_setup
    p, H = copy.deepcopy(info.params), info.demand_paths.shape[2]
    p.pay_delay = p.pay_delay + 4  # every bill of this week's order now falls after the horizon
    plan = p.moq * 3
    st = state.copy()
    st.payables = np.zeros(int((p.lead_time + p.pay_delay).max()) + 1)
    args = (st, plan, info.demand_paths, info.price, p, info.fixed_costs)
    ignored = simulate_cash(*args, continuation="none", charge_beyond_horizon=False)
    charged = simulate_cash(*args, continuation="none", charge_beyond_horizon=True)
    assert np.allclose(charged.cash[:, :-1], ignored.cash[:, :-1])
    assert np.allclose(ignored.cash[:, -1] - charged.cash[:, -1], float(plan @ p.unit_cost))


def test_continuation_keeps_revenue_flowing(week_setup):
    state, info = week_setup
    zero = np.zeros(info.params.n)
    stop = simulate_cash(state, zero, info.demand_paths, info.price, info.params, info.fixed_costs, "none")
    go = simulate_cash(state, zero, info.demand_paths, info.price, info.params, info.fixed_costs, "replace_sales")
    assert go.revenue.mean() > stop.revenue.mean()
    assert np.allclose(go.cash[:, 0], stop.cash[:, 0])  # the first week cannot be affected by later orders
    with pytest.raises(ValueError):
        simulate_cash(state, zero, info.demand_paths, info.price, info.params, info.fixed_costs, "something_else")


def test_prob_above_is_monotone_in_the_buffer(week_setup):
    state, info = week_setup
    sim = simulate_cash(state, np.zeros(info.params.n), info.demand_paths, info.price, info.params, info.fixed_costs)
    probs = [sim.prob_above(b) for b in np.linspace(sim.min_cash.min() - 1, sim.min_cash.max() + 1, 12)]
    assert probs[0] == 1.0 and probs[-1] == 0.0 and all(a >= b for a, b in zip(probs, probs[1:]))
    assert sim.quantiles().shape == (5, info.demand_paths.shape[2])
