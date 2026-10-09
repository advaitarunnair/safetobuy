import numpy as np
import pandas as pd
import pytest

from cspa.config import ConfigError, config_hash, deep_merge, load_config, load_experiments
from cspa.data.load_m5 import FIXTURE_MARKER, SOURCE_FIXTURE, MissingDataError, check_raw, load_slice
from cspa.data.synth_params import make_sku_params
from cspa.sim.backtest import first_decision_week


def test_default_config_is_valid_and_hash_is_stable():
    a, b = load_config(), load_config()
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(load_config(overrides={"seed": a.seed + 1}))
    load_experiments()


@pytest.mark.parametrize(
    "bad",
    [
        {"risk": {"alpha": 2.0}},
        {"risk": {"alpha": 0}},
        {"data": {"n_skus": 0}},
        {"sampling": {"method": "magic"}},
        {"synthetic": {"lead_time_weeks": {"probs": [0.9, 0.9]}}},
        {"synthetic": {"margin_range": [0.5, 0.2]}},
        {"horizons": {"forecast_horizon_weeks": 2}},  # must cover max lead time + review period
        {"forecast": {"quantiles": [0.1, 0.5, 0.9]}},
        {"world": {"shortfall_rule": "cancel_everything"}},
        {"gate": {"on_infeasible": "panic"}},
    ],
)
def test_bad_config_values_fail_loudly(bad):
    with pytest.raises(ConfigError):
        load_config(overrides=bad)


def test_missing_config_key_fails_loudly(tmp_path):
    p = tmp_path / "broken.yaml"
    p.write_text("seed: 1\n")
    with pytest.raises(ConfigError, match="missing"):
        load_config(p)


def test_bad_experiments_fail_loudly():
    with pytest.raises(ConfigError):
        load_experiments(overrides={"policies": ["A", "B"]})  # C is required for the headline
    with pytest.raises(ConfigError):
        load_experiments(overrides={"default_stress": 0.123})


def test_missing_m5_files_stop_with_instructions(tmp_path):
    with pytest.raises(MissingDataError) as err:
        check_raw(tmp_path)
    msg = str(err.value)
    assert "kaggle.com/competitions/m5-forecasting-accuracy" in msg and "sales_train_evaluation.csv" in msg
    assert "Nothing was fabricated" in msg
    assert list(tmp_path.iterdir()) == []  # nothing was created in place of the data


def test_fixture_is_labelled_synthetic(panel, raw_dir):
    assert (raw_dir / FIXTURE_MARKER).exists()
    assert panel.meta["data_source"] == SOURCE_FIXTURE
    assert panel.meta["m5_shape_verified"] is False


def test_slice_filter_and_weekly_aggregation(panel, raw_dir, cfg):
    assert panel.n_obs == 277 and panel.n_skus == cfg.data.n_skus
    assert all(s.startswith("FOODS_3_") for s in panel.skus)
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    row = sales[(sales.store_id == "CA_1") & (sales.item_id == panel.skus[0])]
    daily = row[[f"d_{i}" for i in range(1, 1940)]].to_numpy().ravel()
    assert np.array_equal(panel.units[:, 0], daily.reshape(277, 7).sum(axis=1))
    assert np.nanmax(panel.price) < 50  # the TX_1 decoy price (99.0) was filtered out


def test_sku_selection_precedes_first_decision(panel, cfg, exp):
    assert panel.meta["selection_window_weeks"][1] < first_decision_week(exp, cfg, panel.n_obs)


def test_synthetic_params_are_seeded_and_consistent(panel, cfg, overrides):
    a, b = make_sku_params(panel, cfg), make_sku_params(panel, cfg)
    pd.testing.assert_frame_equal(a.to_frame(), b.to_frame())
    other = make_sku_params(panel, load_config(overrides=deep_merge(overrides, {"seed": 7})))
    assert not a.to_frame().equals(other.to_frame())
    assert (a.moq % a.pack_size == 0).all() and (a.moq >= a.pack_size).all()
    assert (a.unit_cost < a.ref_price).all() and (a.unit_cost > 0).all()
    assert (a.lead_time >= 1).all()
    assert (a.spoilage_rate[~a.perishable] == 0).all()


def test_overage_cost_is_small_for_goods_that_carry_over(params, cfg):
    """Co for non-perishables is holding + capital for one period, not the whole unit cost."""
    co = params.overage_cost(cfg)
    assert (co[~params.perishable] < 0.02 * params.unit_cost[~params.perishable]).all()
    if params.perishable.any():
        assert (co[params.perishable] > co[~params.perishable].max()).all()
        assert (co[params.perishable] < params.unit_cost[params.perishable]).all()


def test_selected_skus_come_from_the_selection_window(panel, cfg, overrides, exp):
    """final_year mode selects on the last weeks instead (and is the documented alternative)."""
    alt = load_slice(load_config(overrides=deep_merge(overrides, {"data": {"selection": {"mode": "final_year"}}})), None)
    assert alt.meta["selection_window_weeks"] == [277 - 52, 276]


def test_config_can_extend_another_file(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text((load_config.__globals__["DEFAULT_CONFIG"]).read_text())
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\ndata:\n  dept_id: HOUSEHOLD_1\n")
    cfg = load_config(child)
    assert cfg.data.dept_id == "HOUSEHOLD_1" and cfg.data.store_id == "CA_1" and "extends" not in cfg
    assert load_config("configs/household_1.yaml").data.processed_dir == "data/processed_household_1"
    (tmp_path / "loop.yaml").write_text("extends: loop.yaml\n")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "loop.yaml")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")
