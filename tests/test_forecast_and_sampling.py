import numpy as np
import pytest

from cspa.config import QCOLS, QUANTILES
from cspa.forecast.baselines import make_forecaster
from cspa.forecast.cache import ForecastCache, build_cache
from cspa.forecast.evaluate import evaluate_all, evaluate_cache, pinball
from cspa.forecast.sampling import build_pit_bank, deterministic_path, pit_values, quantile_fn, sample_demand
from cspa.sim.backtest import decision_cutoffs


def check_quantile_array(arr, n, H):
    assert arr.shape == (n, H, len(QUANTILES))
    assert np.isfinite(arr).all() and (arr >= 0).all()
    assert (np.diff(arr, axis=-1) >= 0).all()  # non-crossing


@pytest.mark.parametrize("name", ["naive", "seasonal_naive"])
def test_baselines_are_valid_quantile_forecasters(name, panel, cfg, cutoffs):
    m = make_forecaster(name, cfg)
    c = int(cutoffs[5])
    m.fit(panel, c)
    check_quantile_array(m.predict_array(c, 4), panel.n_skus, 4)
    df = m.predict(c, 4)
    assert list(df.columns) == ["sku", "h", *QCOLS] and len(df) == panel.n_skus * 4
    assert set(df["h"]) == {1, 2, 3, 4}


def test_lgbm_is_a_valid_quantile_forecaster_and_deterministic(lgb, panel, cfg, cutoffs):
    c = int(cutoffs[5])
    a, b = make_forecaster("lgbm", cfg), make_forecaster("lgbm", cfg)
    a.fit(panel, c)
    b.fit(panel, c)
    pa, pb = a.predict_array(c, 4), b.predict_array(c, 4)
    check_quantile_array(pa, panel.n_skus, 4)
    assert pa.tobytes() == pb.tobytes()  # fixed seed -> identical forecasts
    later = a.predict_array(c + 2, 4, panel)  # refit every few weeks, predict weekly with fresh features
    check_quantile_array(later, panel.n_skus, 4)
    assert abs(sum(a.feature_importance().values()) - 1.0) < 1e-9


def test_lgbm_beats_naive_on_the_fixture(lgb, panel, cfg, exp, cutoffs):
    """Sanity check of the learning pipeline on synthetic data with known structure (not a result)."""
    caches = {n: build_cache(panel, cfg, cutoffs, n, log=lambda *_: None) for n in ("lgbm", "naive")}
    table = evaluate_all(caches, panel.units, decision_cutoffs(exp, panel.n_obs))
    wql = table[table["h"] == "all"].set_index("model")["wql"]
    assert wql["lgbm_conformal"] < wql["naive"] and wql["lgbm_raw"] < wql["naive"]
    assert {"lgbm_raw", "lgbm_conformal", "naive"} == set(table["model"])
    cov = table[(table["h"] == "all") & (table["model"] == "lgbm_conformal")].iloc[0]
    assert 0.3 < cov["cov50"] < 0.7 and 0.6 < cov["cov80"] < 0.95 and 0.75 < cov["cov90"] <= 1.0


def test_pinball_loss_definition():
    y, q = np.array([10.0, 10.0]), np.array([8.0, 13.0])
    assert np.allclose(pinball(y, q, 0.9), [0.9 * 2, 0.1 * 3])
    assert np.allclose(pinball(y, q, 0.5), [1.0, 1.5])


def test_cache_roundtrip_and_evaluation(cache, panel, exp, tmp_path):
    path = tmp_path / "fc.parquet"
    cache.save(path)
    back = ForecastCache.load(path)
    assert np.array_equal(back.cutoffs, cache.cutoffs) and back.skus == cache.skus
    assert np.allclose(back.cal, cache.cal) and np.allclose(back.raw, cache.raw)
    with pytest.raises(KeyError):
        cache.get(int(cache.cutoffs[0]) - 1)
    t = evaluate_cache(cache, panel.units, decision_cutoffs(exp, panel.n_obs))
    assert set(t["h"]) == {"1", "2", "3", "4", "all"} and ((t["cov90"] >= t["cov80"]) & (t["cov80"] >= t["cov50"])).all()
    c = int(cache.cutoffs[3])
    assert len(cache.frame(c)) == panel.n_skus * cache.horizon


def test_quantile_function_and_pit_are_inverse():
    q = np.array([[2.0, 3.0, 5.0, 8.0, 12.0, 17.0, 21.0], [0.0, 0.0, 0.0, 1.0, 2.0, 4.0, 6.0]])
    for tau in (0.05, 0.25, 0.5, 0.9, 0.95):
        j = QUANTILES.index(tau)
        assert np.allclose(quantile_fn(q, np.full((1, 2), tau))[0], q[:, j])
    u = np.array([[0.07, 0.6], [0.33, 0.8], [0.97, 0.99]])
    y = quantile_fn(q, u)
    assert (np.diff(y, axis=0) >= 0).all() and (y[-1] > q[:, -1]).all()  # monotone, and the tail extends past q95
    for k in range(3):
        back = pit_values(q, y[k])
        assert np.allclose(back, u[k], atol=1e-9)
    # flat region: quantiles 5%..25% are all zero for SKU 2, so a realised 0 maps to the middle of that range
    assert pit_values(q, np.array([2.5, 0.0]))[1] == pytest.approx(0.125)
    assert ((pit_values(q, np.array([1e6, 1e6])) <= 0.999) & (pit_values(q, np.array([0.0, 0.0])) >= 0.001)).all()


def test_sampling_is_seeded_whole_units_and_respects_marginals(panel, cache, cutoffs, cfg):
    c = int(cutoffs[-3])
    bank = build_pit_bank(cache.cal, cache.cutoffs, panel.units[: c + 1], c, 20)
    assert bank.shape[1:] == (panel.n_skus, 4) and bank.shape[0] > 5 and ((bank > 0) & (bank < 1)).all()
    q = cache.get(c)
    a = sample_demand(q, bank, 400, np.random.default_rng(3), "block", 2)
    b = sample_demand(q, bank, 400, np.random.default_rng(3), "block", 2)
    assert a.shape == (400, panel.n_skus, 4) and a.tobytes() == b.tobytes()
    assert (a >= 0).all() and np.array_equal(a, np.rint(a))
    ind = sample_demand(q, bank, 400, np.random.default_rng(3), "independent", 2)
    assert np.allclose(a.mean(axis=0).sum(), ind.mean(axis=0).sum(), rtol=0.05)  # same marginals on average
    with pytest.raises(ValueError):
        sample_demand(q, bank, 10, np.random.default_rng(0), "copula")
    empty = sample_demand(q, bank[:0], 50, np.random.default_rng(0), "block", 2)
    assert empty.shape == (50, panel.n_skus, 4)
    assert deterministic_path(q, 0.10).shape == (1, panel.n_skus, 4)


def test_block_sampling_preserves_cross_sku_correlation():
    """A bank where every SKU moves together: joint sampling keeps the swing in the total,
    independent sampling averages it away (and so understates risk)."""
    rng = np.random.default_rng(0)
    n, H, K = 40, 4, 30
    common = rng.uniform(0.05, 0.95, size=(K, 1, H))
    bank = np.clip(common + rng.normal(0, 0.02, size=(K, n, H)), 0.01, 0.99)
    q = np.tile(np.array([20.0, 30.0, 45.0, 60.0, 75.0, 90.0, 100.0]), (n, H, 1))
    joint = sample_demand(q, bank, 2000, np.random.default_rng(1), "block", 1).sum(axis=1)[:, 0]
    indep = sample_demand(q, bank, 2000, np.random.default_rng(1), "independent", 1).sum(axis=1)[:, 0]
    assert joint.std() > 3 * indep.std()
    assert joint.mean() == pytest.approx(indep.mean(), rel=0.05)
    assert np.quantile(joint, 0.1) < np.quantile(indep, 0.1)  # the bad case is worse than independence suggests
