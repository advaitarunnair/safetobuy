"""Shared fixtures. Everything here runs on the SYNTHETIC unit-test fixture, never on M5."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from synthetic_m5_fixture import TEST_CFG_OVERRIDES, TEST_EXP_OVERRIDES, write_fixture

from cspa.config import deep_merge, load_config, load_experiments
from cspa.data.load_m5 import load_slice
from cspa.data.synth_params import make_sku_params
from cspa.forecast.cache import build_cache
from cspa.sim.backtest import make_info, required_cutoffs, selection_end_week
from cspa.sim.world import World

N_OBS = 277
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp("synthetic_m5") / "raw")


@pytest.fixture(scope="session")
def overrides(raw_dir):
    return deep_merge(TEST_CFG_OVERRIDES, {"data": {"raw_dir": str(raw_dir), "processed_dir": str(raw_dir.parent / "processed")}, "llm": {"cache_dir": str(raw_dir.parent / "llm_cache")}})


@pytest.fixture(scope="session")
def cfg(overrides):
    return load_config(overrides=overrides)


@pytest.fixture(scope="session")
def exp():
    return load_experiments(overrides=TEST_EXP_OVERRIDES)


@pytest.fixture(scope="session")
def panel(cfg, exp):
    return load_slice(cfg, selection_end_week(exp, cfg, N_OBS))


@pytest.fixture(scope="session")
def params(panel, cfg):
    return make_sku_params(panel, cfg)


@pytest.fixture(scope="session")
def cutoffs(panel, cfg, exp):
    return required_cutoffs(exp, cfg, panel.n_obs)


@pytest.fixture(scope="session")
def cache(panel, cfg, cutoffs):
    """Seasonal-naive forecast cache: fast, and needs no LightGBM."""
    return build_cache(panel, cfg, cutoffs, "seasonal_naive")


@pytest.fixture(scope="session")
def lgb():
    try:
        from cspa.forecast.lgbm_quantile import import_lightgbm

        return import_lightgbm()
    except Exception as exc:  # libomp missing: run through `make test`
        pytest.skip(f"LightGBM unavailable: {str(exc)[:120]}")


def make_state(panel, params, week: int, cash: float, cover_weeks: float = 1.5):
    """A plausible mid-run state built only from weeks before `week`."""
    world = World(params)
    mean = panel.units[week - 8 : week].mean(axis=0)
    state = world.initial_state(week, cash, np.ceil(cover_weeks * mean))
    state.pipeline[:, 1] = np.ceil(0.5 * mean).astype(np.int64)
    state.payables[: min(3, len(state.payables))] = float(0.3 * (mean @ params.unit_cost))
    return state


@pytest.fixture()
def week_setup(panel, params, cache, cfg, cutoffs):
    """(state, info) for a decision week in the middle of the cached range."""
    week = int(cutoffs[-6]) + 1
    weekly_cost = float(panel.units[week - 8 : week].mean(axis=0) @ params.unit_cost)
    fixed = np.full(int(cfg.horizons.cash_horizon_weeks), 0.25 * weekly_cost)
    info = make_info(panel, cache, params, cfg, week, fixed, buffer=0.5 * weekly_cost)
    return make_state(panel, params, week, cash=1.2 * weekly_cost), info


def run_script(name, *args, env=None):
    cmd = [sys.executable, str(ROOT / "scripts" / name), *args]
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory, cfg):
    """Run the scripts exactly as `make all` does, on the fixture, into a temp results dir."""
    from cspa.config import to_plain

    tmp = tmp_path_factory.mktemp("pipeline")
    plain = deep_merge(to_plain(cfg), {"data": {"processed_dir": str(tmp / "processed")}})
    exp_plain = deep_merge(yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text()), TEST_EXP_OVERRIDES)
    (tmp / "cfg.yaml").write_text(yaml.safe_dump(plain))
    (tmp / "exp.yaml").write_text(yaml.safe_dump(exp_plain))
    common = ["--config", str(tmp / "cfg.yaml"), "--experiments", str(tmp / "exp.yaml")]
    env = {**os.environ, "CSPA_RESULTS_DIR": str(tmp / "results"), "MPLBACKEND": "Agg"}
    out = {"tmp": tmp, "env": env, "common": common, "steps": {}}
    out["steps"]["data"] = run_script("prepare_data.py", *common, env=env)
    out["steps"]["forecast"] = run_script("train_forecast.py", *common, "--models", "seasonal_naive,naive", env=env)
    out["steps"]["quick"] = run_script("run_backtest.py", *common, "--quick", env=env)
    out["latest_after_quick"] = (tmp / "results" / "LATEST").exists()
    out["steps"]["backtest"] = run_script("run_backtest.py", *common, env=env)
    for name, r in out["steps"].items():
        assert r.returncode == 0, f"{name} failed: {r.stderr[-2000:]}"
    out["run"] = tmp / "results" / (tmp / "results" / "LATEST").read_text().strip()
    out["steps"]["figures"] = run_script("make_figures.py", env=env)
    assert out["steps"]["figures"].returncode == 0, out["steps"]["figures"].stderr[-2000:]
    return out
