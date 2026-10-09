import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import run_script

from cspa.config import POLICY_NAMES, deep_merge, to_plain
from cspa.policies import make_policy
from cspa.sim.backtest import Scenario, calibrate_world, make_info, resolve_windows, run_scenario, scenario_grid
from cspa.sim.bundle import load_bundle, write_bundle
from cspa.sim.metrics import aggregate, headline, headline_set, summarise
from cspa.sim.world import World

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scenario(exp, panel):
    w = resolve_windows(exp, panel.n_obs)[-1]
    return Scenario(w, 0.5, 3.0, "overdraft")


@pytest.fixture(scope="module")
def result(panel, params, cache, cfg, scenario):
    return run_scenario(panel, params, cache, cfg, scenario, list(POLICY_NAMES), capture_policy="C")


def test_every_policy_sees_identical_demand_and_parameters(result, panel, scenario):
    """Policy fairness: same realised demand, same prices, same SKU parameters, same starting state."""
    assert len(set(result["demand_hash"].values())) == 1 and set(result["demand_hash"]) == set(POLICY_NAMES)
    wk = result["weekly"]
    demand = wk.pivot(index="week", columns="policy", values="units_demand")
    assert (demand.nunique(axis=1) == 1).all()
    assert np.array_equal(demand["A"].to_numpy(), panel.units[scenario.window.start : scenario.window.end + 1].sum(axis=1))
    first = wk[wk["t"] == 0]
    assert first["cash_start"].nunique() == 1 and first["fixed"].nunique() == 1 and first["buffer"].nunique() == 1
    assert first["supplier_paid"].nunique() == 1  # identical opening payables
    assert wk.groupby("week")["fixed"].nunique().max() == 1


def test_policies_receive_the_same_infoset_object(monkeypatch, panel, params, cache, cfg, scenario):
    seen = {}
    import cspa.sim.backtest as bt

    real = bt.make_policy

    def spy(name):
        pol = real(name)
        inner = pol.decide

        def decide(state, info):
            seen.setdefault(info.week, set()).add(id(info))
            return inner(state, info)

        pol.decide = decide
        return pol

    monkeypatch.setattr(bt, "make_policy", spy)
    run_scenario(panel, params, cache, cfg, scenario, ["A", "B", "C"])
    in_window = {w: ids for w, ids in seen.items() if w >= scenario.window.start}
    assert len(in_window) == scenario.window.n_weeks and all(len(ids) == 1 for ids in in_window.values())


def test_fixed_seed_reproduces_identical_results(result, panel, params, cache, cfg, scenario):
    again = run_scenario(panel, params, cache, cfg, scenario, list(POLICY_NAMES))
    pd.testing.assert_frame_equal(result["weekly"], again["weekly"])


def test_world_calibration_matches_its_definition(panel, params, cfg):
    start = 240
    for s in (0.5, 0.7, 0.9):
        cal = calibrate_world(panel, params, cfg, start, s, 3.0)
        m, off = cal.lump_every, cal.lump_offset
        sched = cal.schedule(start, 2 * m)
        assert sched.mean() == pytest.approx(cal.fixed_avg)  # lumps redistribute fixed costs, they do not add to them
        assert cal.fixed_avg == pytest.approx(cfg.synthetic.fixed_cost_share_of_gross_margin * cal.gm_star)
        assert cal.stress_attainable
        lump_week = start + off
        assert cal.rev_star - cal.fixed(lump_week) == pytest.approx(s * cal.repl_star)  # the definition of stress
        assert cal.buffer == pytest.approx(cfg.risk.buffer_weeks_of_fixed_costs * cal.fixed_avg)
        assert cal.start_cash == pytest.approx(3.0 * cal.fixed_avg)
        assert cal.gm_star == pytest.approx(cal.rev_star - cal.repl_star) and cal.gm_star > 0


def test_scenario_grid_and_windows(exp, panel):
    wins = resolve_windows(exp, panel.n_obs)
    assert [w.role for w in wins] == ["tune", "holdout"] and wins[-1].end == panel.n_obs - 13
    grid = scenario_grid(exp, panel.n_obs)
    assert len({s.id for s in grid}) == len(grid) == len(wins) * len(exp.stress_levels) * len(exp.shortfall_rules)


def test_metrics_match_the_weekly_records(result):
    wk = result["weekly"]
    m = summarise(wk).set_index("policy")
    for pol, g in wk.groupby("policy"):
        assert m.loc[pol, "shortfall_weeks"] == int((g["cash_end"] < g["buffer"]).sum())
        assert m.loc[pol, "insolvency_weeks"] == int((g["cash_end"] < 0).sum())
        assert m.loc[pol, "fill_rate"] == pytest.approx(g["units_sold"].sum() / g["units_demand"].sum())
        assert m.loc[pol, "gross_margin"] == pytest.approx(g["gross_margin"].sum())
        assert m.loc[pol, "end_net_position"] == pytest.approx(g.sort_values("week")["net_position"].iloc[-1])
    assert m.loc["A", "gate_binding_weeks"] == 0 and np.isnan(m.loc["A", "gate_evals_mean"])
    pooled = aggregate(summarise(wk), ["role"])
    assert set(pooled["policy"]) == set(POLICY_NAMES)


def fake_weekly(sf: dict, gm: dict, weeks: int = 12) -> pd.DataFrame:
    rows = []
    for pol in ("A", "B", "C"):
        for t in range(weeks):
            rows.append({"scenario": "S", "window": "W", "role": "holdout", "stress": 0.7, "start_cash_weeks": 3.0, "shortfall_rule": "overdraft", "policy": pol, "t": t, "shortfall": t < sf[pol], "insolvent": False, "gross_margin": gm[pol]})
    return pd.DataFrame(rows)


def test_headline_uses_the_planned_form_only_when_the_run_supports_it():
    good = headline(fake_weekly({"A": 6, "B": 5, "C": 2}, {"A": 100.0, "B": 100.0, "C": 104.0}), 200, 2, 0.9, 1, "t")
    assert good["supports_decided_form"]
    assert good["x_pct_fewer_shortfalls_vs_A"] == pytest.approx(100 * 4 / 6) and good["y_pct_margin_vs_B"] == pytest.approx(4.0)
    assert good["sentence"].startswith("67% fewer cash shortfalls than policy A") and "4.0% higher margin than policy B" in good["sentence"]
    assert good["sentence"].endswith("at equal or lower risk.")
    assert good["x_ci"][0] <= good["x_pct_fewer_shortfalls_vs_A"] <= good["x_ci"][1]

    lower_margin = headline(fake_weekly({"A": 6, "B": 5, "C": 2}, {"A": 100.0, "B": 100.0, "C": 97.0}), 200, 2, 0.9, 1, "t")
    assert not lower_margin["supports_decided_form"]
    assert "3.0% lower margin than policy B" in lower_margin["sentence"] and "does not support the planned headline form" in lower_margin["sentence"]

    riskier = headline(fake_weekly({"A": 6, "B": 1, "C": 3}, {"A": 100.0, "B": 100.0, "C": 110.0}), 200, 2, 0.9, 1, "t")
    assert not riskier["supports_decided_form"] and "higher shortfall risk than B" in riskier["sentence"]

    no_risk = headline(fake_weekly({"A": 0, "B": 0, "C": 0}, {"A": 100.0, "B": 100.0, "C": 101.0}), 200, 2, 0.9, 1, "t")
    assert not no_risk["supports_decided_form"] and "Policy A had no cash-shortfall weeks" in no_risk["sentence"]
    assert np.isnan(no_risk["x_pct_fewer_shortfalls_vs_A"]) and no_risk["x_ci"] is None

    worse = headline(fake_weekly({"A": 2, "B": 2, "C": 5}, {"A": 100.0, "B": 100.0, "C": 90.0}), 200, 2, 0.9, 1, "t")
    assert "150% more cash-shortfall weeks than policy A" in worse["sentence"]


def test_headline_set_scopes(result, exp):
    hs = headline_set(result["weekly"], exp, 1)
    assert hs["main"] is None  # this scenario's opening cash is not the default one, so it is outside the headline scope
    wk = result["weekly"].assign(start_cash_weeks=float(exp.default_start_cash_weeks))
    hs = headline_set(wk, exp, 1)
    assert hs["main"]["n_scenarios"] == 1 and "holdout" in hs["main"]["scope"]
    assert hs["variant_C_plus"] is not None and set(hs["by_rule"]) == {"overdraft"}


def test_app_bundle_reproduces_the_backtest_decision(result, panel, params, cache, cfg, scenario, tmp_path):
    write_bundle(tmp_path / "app", panel, params, cache, cfg, [result], "test_run")
    b = load_bundle(tmp_path / "app")
    assert b.meta["data_source"] == panel.meta["data_source"] and "raw_files" not in b.panel.meta
    cal = b.cal(scenario.id)
    H = int(b.cfg.horizons.cash_horizon_weeks)
    wk = result["weekly"]
    for week in (scenario.window.start, scenario.window.start + 3, scenario.window.end):
        state = b.state(scenario.id, week)
        info = make_info(b.panel, b.cache, b.params, b.cfg, week, cal.schedule(week, H), cal.buffer)
        orders = make_policy("C").decide(state, info)
        row = wk[(wk["policy"] == "C") & (wk["week"] == week)].iloc[0]
        assert orders.meta["plan_cost"] == pytest.approx(row["order_cost_requested"])
        assert state.cash == pytest.approx(row["cash_start"])
    size = sum(f.stat().st_size for f in (tmp_path / "app").iterdir())
    assert size < 5_000_000


def test_scripts_end_to_end_on_the_fixture(pipeline, cfg):
    """prepare_data -> train_forecast -> run_backtest -> make_figures -> render_results, as `make all` runs them."""
    steps, run, env = pipeline["steps"], pipeline["run"], pipeline["env"]
    assert "SYNTHETIC_TEST_FIXTURE" in steps["data"].stdout and "PHASE 0 CHECKPOINT" in steps["data"].stdout
    assert not pipeline["latest_after_quick"]  # partial runs never become "the" result
    assert "not results" in steps["backtest"].stdout

    manifest = json.loads((run / "manifest.json").read_text())
    for key in ("config_hash", "git", "seed", "timestamp", "data_slice", "run_id", "config", "experiments"):
        assert key in manifest
    assert manifest["data_source"] == "SYNTHETIC_TEST_FIXTURE" and manifest["fairness_check"] == "passed"
    assert manifest["seed"] == cfg.seed and manifest["data_slice"]["store_id"] == "CA_1"
    for f in ("weekly.parquet", "metrics.csv", "headline.json", "calibration.csv", "forecast_eval.csv", "app/bundle_meta.json", "app/states.npz"):
        assert (run / f).exists(), f
    hl = json.loads((run / "headline.json").read_text())
    assert hl["main"]["sentence"] and set(hl["by_rule"]) == {"overdraft", "cut_proportional"}
    assert len(list((run / "figures").glob("*.png"))) >= 5

    r = run_script("render_results.py", env=env)
    assert r.returncode == 3 and "REFUSING" in r.stderr  # fixture numbers can never reach the README
    rendered = pipeline["tmp"] / "rendered"
    r = run_script("render_results.py", "--out-dir", str(rendered), "--allow-fixture", env=env)
    assert r.returncode == 0, r.stderr
    readme = rendered / "README.md"
    if readme.exists():
        text = readme.read_text()
        assert hl["main"]["sentence"] in text and manifest["run_id"] in text and manifest["config_hash"] in text


def test_rerunning_the_backtest_gives_identical_numbers(pipeline):
    r = run_script("run_backtest.py", *pipeline["common"], "--tag", "again", env=pipeline["env"])
    assert r.returncode == 0, r.stderr
    again = pipeline["tmp"] / "results" / (pipeline["tmp"] / "results" / "LATEST").read_text().strip()
    assert again != pipeline["run"]
    cols = ["scenario", "policy", "shortfall_weeks", "insolvency_weeks", "gross_margin", "total_spend", "end_net_position", "fill_rate"]
    pd.testing.assert_frame_equal(pd.read_csv(pipeline["run"] / "metrics.csv")[cols], pd.read_csv(again / "metrics.csv")[cols])
    a, b = (json.loads((d / "headline.json").read_text())["main"] for d in (pipeline["run"], again))
    assert a == b  # including bootstrap intervals
    (pipeline["tmp"] / "results" / "LATEST").write_text(pipeline["run"].name + "\n")


def test_missing_raw_data_makes_the_pipeline_stop(tmp_path, cfg):
    plain = deep_merge(to_plain(cfg), {"data": {"raw_dir": str(tmp_path / "empty_raw"), "processed_dir": str(tmp_path / "p")}})
    (tmp_path / "empty_raw").mkdir()
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(plain))
    r = run_script("prepare_data.py", "--config", str(tmp_path / "cfg.yaml"))
    assert r.returncode == 2 and "kaggle" in r.stderr.lower() and "Nothing was fabricated" in r.stderr
    assert not (tmp_path / "p").exists()
