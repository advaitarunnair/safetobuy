"""Load the M5 slice, aggregate to Walmart weeks (wm_yr_wk) and persist it.

Only real M5 files are read here. If they are missing we stop with exact
instructions; sales data is never fabricated.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_FILES = ("sales_train_evaluation.csv", "sell_prices.csv", "calendar.csv")
FIXTURE_MARKER = "SYNTHETIC_FIXTURE.txt"
SOURCE_M5 = "M5"
SOURCE_FIXTURE = "SYNTHETIC_TEST_FIXTURE"
EVENT_TYPES = ("Sporting", "Cultural", "National", "Religious")
CAL_FEATURES = ("snap_days", "event_days", "ev_sporting", "ev_cultural", "ev_national", "ev_religious", "christmas", "month", "woy")


class MissingDataError(FileNotFoundError):
    """Raised when the M5 raw files are not in data/raw/."""


def missing_data_message(raw_dir: Path, missing: list[str]) -> str:
    lines = [
        "",
        "M5 raw data not found. Nothing was fabricated; the pipeline stops here.",
        f"  Looked in: {raw_dir}",
        f"  Missing:   {', '.join(missing)}",
        "",
        "How to get the files (Kaggle login required):",
        "  1. Open https://www.kaggle.com/competitions/m5-forecasting-accuracy/data",
        "  2. Sign in and accept the competition rules (Late Submission is fine).",
        "  3. Download all (m5-forecasting-accuracy.zip) and unzip it.",
        "  4. Copy these three files into the folder above:",
        "       sales_train_evaluation.csv   (~120 MB)",
        "       sell_prices.csv              (~200 MB)",
        "       calendar.csv                 (~100 KB)",
        "  Or with the Kaggle CLI:",
        "       kaggle competitions download -c m5-forecasting-accuracy -p data/raw",
        "       unzip data/raw/m5-forecasting-accuracy.zip -d data/raw",
        "  5. Re-run:  make data",
        "",
    ]
    return "\n".join(lines)


def check_raw(raw_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (raw_dir / f).exists()]
    if missing:
        raise MissingDataError(missing_data_message(raw_dir, missing))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Panel:
    """Weekly demand panel. Rows are weeks (0 = first complete Walmart week), columns are SKUs.

    `weeks` covers every complete calendar week, including a few after the last
    observed sales week (the calendar is published ahead of time). `units` and
    `price` only cover observed weeks.
    """

    skus: list[str]
    weeks: pd.DataFrame
    units: np.ndarray
    price: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return int(self.units.shape[0])

    @property
    def n_skus(self) -> int:
        return int(self.units.shape[1])

    def copy(self) -> "Panel":
        return Panel(list(self.skus), self.weeks.copy(), self.units.copy(), self.price.copy(), dict(self.meta))

    def truncated(self, cutoff: int) -> "Panel":
        """Delete every observation after `cutoff` (inclusive of cutoff is kept)."""
        return Panel(list(self.skus), self.weeks.copy(), self.units[: cutoff + 1].copy(), self.price[: cutoff + 1].copy(), dict(self.meta))

    def perturbed_after(self, cutoff: int, seed: int = 0) -> "Panel":
        """Replace every observation after `cutoff` with noise (for no-lookahead tests)."""
        rng = np.random.default_rng(seed)
        out = self.copy()
        n_after = self.n_obs - (cutoff + 1)
        if n_after > 0:
            out.units[cutoff + 1 :] = rng.integers(0, 5000, size=(n_after, self.n_skus)).astype(float)
            out.price[cutoff + 1 :] = rng.uniform(0.01, 99.0, size=(n_after, self.n_skus))
        return out

    def cal(self, name: str, idx: np.ndarray) -> np.ndarray:
        """Calendar feature lookup that returns 0 beyond the published calendar."""
        col = self.weeks[name].to_numpy(dtype=float)
        idx = np.asarray(idx)
        ok = (idx >= 0) & (idx < len(col))
        out = np.zeros(idx.shape, dtype=float)
        out[ok] = col[idx[ok]]
        return out

    def to_long(self) -> pd.DataFrame:
        n_obs, n = self.units.shape
        return pd.DataFrame(
            {
                "week": np.repeat(np.arange(n_obs), n),
                "sku": np.tile(np.array(self.skus, dtype=object), n_obs),
                "units": self.units.ravel(),
                "price": self.price.ravel(),
            }
        )

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.to_long().to_parquet(out_dir / "weekly.parquet", index=False)
        self.weeks.reset_index().to_parquet(out_dir / "weeks.parquet", index=False)
        (out_dir / "slice_meta.json").write_text(json.dumps({**self.meta, "skus": self.skus}, indent=2, default=str))

    @classmethod
    def load(cls, out_dir: Path) -> "Panel":
        meta_path = out_dir / "slice_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Processed slice not found in {out_dir}. Run `make data` first.")
        meta = json.loads(meta_path.read_text())
        skus = meta.pop("skus")
        long = pd.read_parquet(out_dir / "weekly.parquet")
        weeks = pd.read_parquet(out_dir / "weeks.parquet").set_index("week")
        n_obs = int(long["week"].max()) + 1
        units = long["units"].to_numpy(dtype=float).reshape(n_obs, len(skus))
        price = long["price"].to_numpy(dtype=float).reshape(n_obs, len(skus))
        return cls(skus, weeks, units, price, meta)


def build_weeks(calendar: pd.DataFrame, state: str) -> pd.DataFrame:
    """One row per complete Walmart week with calendar features."""
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    snap_col = f"snap_{state}"
    if snap_col not in cal.columns:
        raise ValueError(f"calendar.csv has no column {snap_col}")
    has_event = cal["event_name_1"].notna()
    agg = {
        "n_days": cal.groupby("wm_yr_wk")["d"].count(),
        "start_date": cal.groupby("wm_yr_wk")["date"].min(),
        "snap_days": cal.groupby("wm_yr_wk")[snap_col].sum(),
        "event_days": has_event.groupby(cal["wm_yr_wk"]).sum(),
    }
    for ev in EVENT_TYPES:
        hit = (cal["event_type_1"] == ev) | (cal["event_type_2"] == ev)
        agg[f"ev_{ev.lower()}"] = hit.groupby(cal["wm_yr_wk"]).sum()
    xmas = (cal["event_name_1"] == "Christmas") | (cal["event_name_2"] == "Christmas")
    agg["christmas"] = xmas.groupby(cal["wm_yr_wk"]).max().astype(int)
    weeks = pd.DataFrame(agg).sort_index()
    weeks = weeks[weeks["n_days"] == 7].drop(columns="n_days").reset_index()
    mid = weeks["start_date"] + pd.Timedelta(days=3)
    weeks["month"] = mid.dt.month.astype(int)
    weeks["woy"] = mid.dt.isocalendar().week.astype(int)
    weeks["year"] = mid.dt.year.astype(int)
    weeks.index.name = "week"
    return weeks


def _read_sales(path: Path, store_id: str, dept_id: str) -> tuple[pd.DataFrame, int]:
    parts, total = [], 0
    for chunk in pd.read_csv(path, chunksize=4000):
        total += len(chunk)
        parts.append(chunk[(chunk["store_id"] == store_id) & (chunk["dept_id"] == dept_id)])
    return pd.concat(parts, ignore_index=True), total


def _read_prices(path: Path, store_id: str, items: set[str]) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, chunksize=1_000_000):
        parts.append(chunk[(chunk["store_id"] == store_id) & (chunk["item_id"].isin(items))])
    return pd.concat(parts, ignore_index=True)


def load_slice(cfg, selection_end_week: int | None = None) -> Panel:
    """Read raw M5 files and return the weekly panel for the configured slice.

    selection_end_week: last week (inclusive) of the SKU-selection window. If
    None, or if data.selection.mode == 'final_year', the final observed weeks are used.
    """
    from cspa.config import resolve_path

    raw_dir = resolve_path(cfg.data.raw_dir)
    check_raw(raw_dir)
    source = SOURCE_FIXTURE if (raw_dir / FIXTURE_MARKER).exists() else SOURCE_M5
    store, dept, state = cfg.data.store_id, cfg.data.dept_id, cfg.data.store_id.split("_")[0]

    calendar = pd.read_csv(raw_dir / "calendar.csv")
    weeks = build_weeks(calendar, state)
    sales, n_sales_rows = _read_sales(raw_dir / "sales_train_evaluation.csv", store, dept)
    if sales.empty:
        raise ValueError(f"No rows for store_id={store}, dept_id={dept} in sales_train_evaluation.csv")
    sales = sales.sort_values("item_id").reset_index(drop=True)

    dcols = [c for c in sales.columns if c.startswith("d_")]
    wk_of_day = calendar.set_index("d")["wm_yr_wk"].reindex(dcols)
    if wk_of_day.isna().any():
        raise ValueError("Some sales day columns are not in calendar.csv")
    codes, uniq = pd.factorize(wk_of_day.to_numpy(), sort=True)
    vals = sales[dcols].to_numpy(dtype=np.float64)
    by_week = np.zeros((len(uniq), vals.shape[0]))
    np.add.at(by_week, codes, vals.T)
    complete = np.bincount(codes, minlength=len(uniq)) == 7
    obs_wk = uniq[complete]
    cal_wk = weeks["wm_yr_wk"].to_numpy()
    n_obs = len(obs_wk)
    if not np.array_equal(cal_wk[:n_obs], obs_wk):
        raise ValueError("Observed sales weeks are not a contiguous prefix of the calendar weeks")
    units_all = by_week[complete]
    weeks["observed"] = np.arange(len(weeks)) < n_obs

    sel_weeks = int(cfg.data.selection.weeks)
    if cfg.data.selection.mode == "final_year" or selection_end_week is None:
        sel_end = n_obs - 1
    else:
        sel_end = int(selection_end_week)
    sel_start = sel_end - sel_weeks + 1
    if sel_start < 0 or sel_end >= n_obs:
        raise ValueError(f"SKU selection window [{sel_start}, {sel_end}] is outside observed weeks 0..{n_obs - 1}")
    sel_units = units_all[sel_start : sel_end + 1].sum(axis=0)
    items = sales["item_id"].to_numpy()
    order = np.lexsort((items, -sel_units))
    n_keep = min(int(cfg.data.n_skus), int((sel_units > 0).sum()))
    keep = np.sort(order[:n_keep])
    skus = [str(x) for x in items[keep]]
    units = units_all[:, keep]

    prices = _read_prices(raw_dir / "sell_prices.csv", store, set(skus))
    price = (
        prices.pivot_table(index="wm_yr_wk", columns="item_id", values="sell_price", aggfunc="last")
        .reindex(index=obs_wk, columns=skus)
        .ffill()
        .to_numpy(dtype=float)
    )

    cal_dates = pd.to_datetime(calendar["date"])
    meta = {
        "data_source": source,
        "store_id": store,
        "dept_id": dept,
        "state": state,
        "n_skus": len(skus),
        "n_skus_in_dept": int(len(items)),
        "n_obs_weeks": int(n_obs),
        "n_calendar_weeks": int(len(weeks)),
        "first_wm_yr_wk": int(obs_wk[0]),
        "last_wm_yr_wk": int(obs_wk[-1]),
        "first_date": str(weeks["start_date"].iloc[0].date()),
        "last_date": str((weeks["start_date"].iloc[n_obs - 1] + pd.Timedelta(days=6)).date()),
        "selection_mode": cfg.data.selection.mode,
        "selection_window_weeks": [int(sel_start), int(sel_end)],
        "selection_units_share_of_dept": float(sel_units[keep].sum() / max(sel_units.sum(), 1.0)),
        "zero_week_share": float((units == 0).mean()),
        "m5_shape_verified": bool(source == SOURCE_M5 and n_sales_rows == 30490 and len(dcols) == 1941 and len(calendar) == 1969),
        "calendar_span": [str(cal_dates.min().date()), str(cal_dates.max().date())],
        "raw_files": {f: {"bytes": (raw_dir / f).stat().st_size, "sha256": _sha256(raw_dir / f)} for f in REQUIRED_FILES},
    }
    return Panel(skus, weeks, units, price, meta)


def slice_summary(panel: Panel) -> str:
    m = panel.meta
    weekly_rev = np.nansum(panel.units * np.nan_to_num(panel.price), axis=1)
    lines = [
        f"data source      : {m['data_source']}" + ("   <-- NOT REAL DATA (unit-test fixture)" if m["data_source"] != SOURCE_M5 else ""),
        f"slice            : store {m['store_id']}, dept {m['dept_id']}",
        f"SKUs             : {m['n_skus']} of {m['n_skus_in_dept']} in dept ({m['selection_units_share_of_dept']:.1%} of dept units in selection window)",
        f"selection window : weeks {m['selection_window_weeks'][0]}..{m['selection_window_weeks'][1]} ({m['selection_mode']})",
        f"observed weeks   : {m['n_obs_weeks']} complete Walmart weeks, {m['first_date']} .. {m['last_date']} (wm_yr_wk {m['first_wm_yr_wk']}..{m['last_wm_yr_wk']})",
        f"units / week     : mean {panel.units.sum(axis=1).mean():,.0f} across the slice; {m['zero_week_share']:.1%} of SKU-weeks are zero",
        f"revenue / week   : mean ${weekly_rev.mean():,.0f} at " + ("M5 sell prices (real prices, US Walmart)" if m["data_source"] == SOURCE_M5 else "fixture prices (made up)"),
        f"M5 shape check   : {'passed' if m['m5_shape_verified'] else 'n/a or failed'}",
    ]
    return "\n".join(lines)
