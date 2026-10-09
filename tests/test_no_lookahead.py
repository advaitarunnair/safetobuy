"""No lookahead: nothing dated after a cutoff week may influence features, forecasts,
calibration, sampled demand, or any policy's decision at that cutoff.

Method: replace every sales and price observation after the cutoff with noise (and,
separately, delete them) and require byte-identical outputs.
"""
import numpy as np
import pytest

from conftest import make_state

from cspa.config import POLICY_NAMES
from cspa.data.features import assemble, origin_block
from cspa.forecast.baselines import make_forecaster
from cspa.forecast.cache import build_cache
from cspa.forecast.conformal import calibration_origins, conformal_adjust
from cspa.forecast.sampling import build_pit_bank
from cspa.policies import make_policy
from cspa.sim.backtest import make_info


@pytest.fixture(scope="module")
def cut(cutoffs):
    return int(cutoffs[-8])


@pytest.fixture(scope="module")
def variants(panel, cut):
    return {"perturbed": panel.perturbed_after(cut, seed=1), "deleted": panel.truncated(cut)}


def test_perturbation_really_changes_the_future(panel, variants, cut):
    assert not np.array_equal(panel.units[cut + 1 :], variants["perturbed"].units[cut + 1 :])
    assert np.array_equal(panel.units[: cut + 1], variants["perturbed"].units[: cut + 1])
    assert variants["deleted"].n_obs == cut + 1


def test_features_are_byte_identical(panel, variants, cut, cfg):
    H = cfg.horizons.forecast_horizon_weeks
    base_block = origin_block(panel, cut)
    for v in variants.values():
        block = origin_block(v, cut)
        for name, arr in base_block.items():
            assert arr.tobytes() == block[name].tobytes(), name
        for h in range(1, H + 1):
            X0, s0, _ = assemble(panel, base_block, np.array([cut]), h, cut, with_target=False)
            X1, s1, _ = assemble(v, block, np.array([cut]), h, cut, with_target=False)
            assert X0.tobytes() == X1.tobytes() and s0.tobytes() == s1.tobytes()
        tr = np.arange(cut - 30, cut - H + 1)
        Xa, _, ya = assemble(panel, base_block, tr, H, cut, with_target=True)
        Xb, _, yb = assemble(v, block, tr, H, cut, with_target=True)
        assert Xa.tobytes() == Xb.tobytes() and ya.tobytes() == yb.tobytes()


def test_features_refuse_to_read_past_the_cutoff(panel, cut):
    block = origin_block(panel, cut)
    with pytest.raises(ValueError):
        assemble(panel, block, np.array([cut]), 1, cut, with_target=True)  # target would be cut + 1
    with pytest.raises(ValueError):
        origin_block(panel.truncated(cut), cut + 1)


@pytest.mark.parametrize("name", ["naive", "seasonal_naive"])
def test_baseline_forecasts_are_byte_identical(name, panel, variants, cut, cfg):
    H = cfg.horizons.forecast_horizon_weeks
    m = make_forecaster(name, cfg)
    m.fit(panel, cut)
    base = m.predict_array(cut, H)
    for v in variants.values():
        m2 = make_forecaster(name, cfg)
        m2.fit(v, cut)
        assert base.tobytes() == m2.predict_array(cut, H).tobytes()


def test_lgbm_forecasts_are_byte_identical(lgb, panel, variants, cut, cfg):
    H = cfg.horizons.forecast_horizon_weeks
    m = make_forecaster("lgbm", cfg)
    m.fit(panel, cut)
    base = m.predict_array(cut, H)
    for v in variants.values():
        m2 = make_forecaster("lgbm", cfg)
        m2.fit(v, cut)
        assert base.tobytes() == m2.predict_array(cut, H).tobytes()


def test_model_cannot_predict_before_its_fit_cutoff(lgb, panel, cut, cfg):
    m = make_forecaster("lgbm", cfg)
    m.fit(panel, cut)
    with pytest.raises(ValueError):
        m.predict_array(cut - 1, 2)


def test_conformal_window_ends_before_the_cutoff(cutoffs, cut, cfg):
    K = cfg.forecast.conformal.calibration_weeks
    for h in range(1, cfg.horizons.forecast_horizon_weeks + 1):
        oi = calibration_origins(cutoffs, cut, h, K)
        assert oi.size > 0
        targets = cutoffs[oi] + h
        assert targets.max() <= cut  # every calibration target is already observed at the cutoff
        assert (cutoffs[oi] < cut).all()  # and every calibration forecast was issued earlier
        assert targets.min() > cut - K


def test_conformal_refuses_future_data(panel, cache, cutoffs, cut, cfg):
    ci = int(np.flatnonzero(cutoffs == cut)[0])
    with pytest.raises(ValueError):
        conformal_adjust(cache.raw, cutoffs, ci, panel.units[: cut + 2], cfg)
    out, info = conformal_adjust(cache.raw, cutoffs, ci, panel.units[: cut + 1], cfg)
    assert info["applied"] and all(v["last_target_week"] <= cut for v in info["h"].values())
    assert (np.diff(out, axis=-1) >= 0).all() and (out >= 0).all()


def test_conformal_output_is_identical_when_the_future_changes(lgb, panel, variants, cutoffs, cut, cfg):
    upto = cutoffs[cutoffs <= cut]
    base = build_cache(panel, cfg, upto, "lgbm", log=lambda *_: None)
    other = build_cache(variants["perturbed"], cfg, upto, "lgbm", log=lambda *_: None)
    assert base.raw.tobytes() == other.raw.tobytes()
    assert base.cal.tobytes() == other.cal.tobytes()
    assert base.calibrated.any()


def test_pit_bank_uses_only_realised_weeks(panel, variants, cache, cutoffs, cut, cfg):
    with pytest.raises(ValueError):
        build_pit_bank(cache.cal, cache.cutoffs, panel.units[: cut + 2], cut, 20)
    a = build_pit_bank(cache.cal, cache.cutoffs, panel.units[: cut + 1], cut, 20)
    b = build_pit_bank(cache.cal, cache.cutoffs, variants["perturbed"].units[: cut + 1], cut, 20)
    assert a.shape[0] > 0 and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("policy", POLICY_NAMES)
def test_every_policy_decision_is_byte_identical(policy, panel, variants, params, cutoffs, cut, cfg):
    """Rebuild forecasts, sampled demand and the decision from perturbed data: same order, to the unit."""
    week = cut + 1
    upto = cutoffs[cutoffs <= cut]
    H = cfg.horizons.cash_horizon_weeks
    weekly_cost = float(panel.units[cut - 7 : cut + 1].mean(axis=0) @ params.unit_cost)
    fixed, buffer = np.full(H, 0.25 * weekly_cost), 0.5 * weekly_cost
    state = make_state(panel, params, week, cash=0.9 * weekly_cost)

    def decide(p):
        c = build_cache(p, cfg, upto, "seasonal_naive")
        info = make_info(p, c, params, cfg, week, fixed, buffer)
        return make_policy(policy).decide(state.copy(), info), info

    base, base_info = decide(panel)
    for v in variants.values():
        other, info = decide(v)
        assert base.qty.tobytes() == other.qty.tobytes()
        assert base_info.q.tobytes() == info.q.tobytes()
        assert base_info.price.tobytes() == info.price.tobytes()
        if policy in ("C", "C_plus", "C_gate_prop", "OTB_marginal", "D"):
            assert base_info.demand_paths.tobytes() == info.demand_paths.tobytes()
    assert base.qty.sum() > 0 or policy in ("C", "C_gate_prop")
