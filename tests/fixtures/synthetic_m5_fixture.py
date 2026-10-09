"""SYNTHETIC M5-shaped fixture, for unit tests ONLY.

This writes three small CSV files with the same columns as the real M5 files,
filled with made-up numbers, plus a marker file (SYNTHETIC_FIXTURE.txt). The
loader sees the marker and labels everything downstream SYNTHETIC_TEST_FIXTURE,
and scripts/render_results.py refuses to publish numbers from such a run.

It must never be used for results, figures, the README or the app.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MARKER = "SYNTHETIC_FIXTURE.txt"
N_DAYS_SALES = 1941
N_DAYS_CALENDAR = 1969


def write_fixture(out_dir: Path, n_items: int = 30, seed: int = 0, store: str = "CA_1", dept: str = "FOODS_3") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2011-01-29", periods=N_DAYS_CALENDAR, freq="D")
    wk = np.arange(N_DAYS_CALENDAR) // 7
    wm_yr_wk = 11101 + (wk // 52) * 100 + (wk % 52)
    cal = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": wm_yr_wk,
            "weekday": dates.day_name(),
            "wday": ((np.arange(N_DAYS_CALENDAR)) % 7) + 1,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i + 1}" for i in range(N_DAYS_CALENDAR)],
            "event_name_1": None,
            "event_type_1": None,
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": (dates.day <= 10).astype(int),
            "snap_TX": (dates.day % 3 == 0).astype(int),
            "snap_WI": (dates.day % 2 == 0).astype(int),
        }
    )
    events = {(12, 25): ("Christmas", "National"), (7, 4): ("IndependenceDay", "National"), (2, 14): ("ValentinesDay", "Cultural"), (10, 31): ("Halloween", "Cultural")}
    for (m, d), (name, typ) in events.items():
        hit = (dates.month == m) & (dates.day == d)
        cal.loc[hit, "event_name_1"] = name
        cal.loc[hit, "event_type_1"] = typ
    cal.to_csv(out_dir / "calendar.csv", index=False)

    n_weeks = N_DAYS_CALENDAR // 7
    items = [f"{dept}_{i + 1:03d}" for i in range(n_items)]
    base = np.exp(rng.uniform(np.log(0.3), np.log(40.0), size=n_items))  # units per day
    start_week = np.where(rng.random(n_items) < 0.15, rng.integers(20, 90, size=n_items), 0)
    price0 = np.round(rng.uniform(0.8, 9.0, size=n_items), 2)
    common = np.exp(0.12 * rng.standard_normal(n_weeks))  # shared weekly shock -> cross-SKU correlation
    promo = rng.random((n_weeks, n_items)) < 0.06
    price_wk = np.where(promo, np.round(price0 * 0.8, 2), price0)[..., :]

    day = np.arange(N_DAYS_SALES)
    dow = np.array([1.25, 1.2, 0.9, 0.85, 0.85, 0.9, 1.05])[day % 7]
    annual = 1.0 + 0.2 * np.sin(2 * np.pi * day / 364.0)
    snap = 1.0 + 0.15 * cal["snap_CA"].to_numpy()[:N_DAYS_SALES]
    xmas = np.where((dates.month[:N_DAYS_SALES] == 12) & (dates.day[:N_DAYS_SALES] == 25), 0.0, 1.0)
    wk_s = wk[:N_DAYS_SALES]
    lam = base[None, :] * (dow * annual * snap * xmas)[:, None] * common[wk_s][:, None] * np.where(promo[wk_s], 1.6, 1.0)
    lam = lam * (wk_s[:, None] >= start_week[None, :])
    sales = rng.poisson(lam).T  # [items, days]

    meta = pd.DataFrame({"id": [f"{it}_{store}_evaluation" for it in items], "item_id": items, "dept_id": dept, "cat_id": dept.rsplit("_", 1)[0], "store_id": store, "state_id": store.split("_")[0]})
    dcols = [f"d_{i + 1}" for i in range(N_DAYS_SALES)]
    main = pd.concat([meta, pd.DataFrame(sales, columns=dcols)], axis=1)
    # rows from another store and another department, to exercise the slice filter
    other_store = main.head(3).copy()
    other_store["store_id"], other_store["state_id"] = "TX_1", "TX"
    other_store["id"] = other_store["item_id"] + "_TX_1_evaluation"
    other_dept = main.head(3).copy()
    other_dept["dept_id"] = "HOBBIES_1"
    other_dept["item_id"] = [f"HOBBIES_1_{i + 1:03d}" for i in range(3)]
    other_dept["id"] = other_dept["item_id"] + f"_{store}_evaluation"
    pd.concat([main, other_store, other_dept], ignore_index=True).to_csv(out_dir / "sales_train_evaluation.csv", index=False)

    wk_codes = 11101 + (np.arange(n_weeks) // 52) * 100 + (np.arange(n_weeks) % 52)
    rows = []
    for j, it in enumerate(items):
        w = np.arange(start_week[j], n_weeks)
        rows.append(pd.DataFrame({"store_id": store, "item_id": it, "wm_yr_wk": wk_codes[w], "sell_price": price_wk[w, j]}))
    rows.append(pd.DataFrame({"store_id": "TX_1", "item_id": items[0], "wm_yr_wk": wk_codes, "sell_price": 99.0}))
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "sell_prices.csv", index=False)

    (out_dir / MARKER).write_text("SYNTHETIC unit-test fixture. Not M5 data. Never use for results.\n")
    return out_dir


# Small, fast settings used by the test-suite on top of configs/default.yaml.
TEST_CFG_OVERRIDES = {
    "data": {"n_skus": 20},
    "forecast": {
        "model": "seasonal_naive",
        "train_window_weeks": 80,
        "min_train_weeks": 20,
        "warmup_weeks": 12,
        "lgbm": {"n_estimators": 25, "num_threads": 2},
        "conformal": {"calibration_weeks": 8, "min_samples": 40},
    },
    "sampling": {"bank_weeks": 20, "n_paths_backtest": 120, "n_paths_app": 200},
    "synthetic": {"burn_in_weeks": 4},
}

TEST_EXP_OVERRIDES = {
    "windows": [
        {"name": "W1", "weeks_before_end": 40, "n_weeks": 8, "role": "tune"},
        {"name": "W2", "weeks_before_end": 20, "n_weeks": 8, "role": "holdout"},
    ],
    "stress_levels": [0.5, 0.9],
    "default_stress": 0.5,
    "start_cash_weeks": [2.0],
    "default_start_cash_weeks": 2.0,
    "bootstrap": {"n_boot": 50, "block_weeks": 2, "ci": 0.9},
    "app_bundle": {"window": "W2", "stress_levels": [0.5], "start_cash_weeks": 2.0, "shortfall_rule": "overdraft"},
    "n_jobs": 1,
}
