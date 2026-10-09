"""Load, validate and hash configuration. Bad values fail loudly."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DEFAULT_EXPERIMENTS = ROOT / "configs" / "experiments.yaml"

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QCOLS = tuple(f"q{int(round(q * 100)):02d}" for q in QUANTILES)
POLICY_NAMES = ("A", "B", "C", "D", "C_gate_prop", "OTB_marginal", "C_plus")


class ConfigError(ValueError):
    """Raised when a config value is missing or out of range."""


class Config(dict):
    """A dict with attribute access (cfg.risk.alpha)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"config has no key '{key}'") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo: dict) -> "Config":
        return _wrap(copy.deepcopy(dict(self), memo))


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _unwrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_unwrap(v) for v in obj]
    return obj


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(dict(base))
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _get(cfg: dict, path: str) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def _num(lo: float | None = None, hi: float | None = None, integer: bool = False, lo_open: bool = False):
    def check(v: Any) -> str | None:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return "must be a number"
        if integer and not isinstance(v, int):
            return "must be an integer"
        if lo is not None and (v <= lo if lo_open else v < lo):
            return f"must be {'>' if lo_open else '>='} {lo}"
        if hi is not None and v > hi:
            return f"must be <= {hi}"
        return None

    return check


def _choice(*opts: Any):
    return lambda v: None if v in opts else f"must be one of {list(opts)}"


def _string(v: Any) -> str | None:
    return None if isinstance(v, str) and v.strip() else "must be a non-empty string"


def _probs(v: Any) -> str | None:
    if not isinstance(v, list) or not v or any(not isinstance(p, (int, float)) or p < 0 for p in v):
        return "must be a non-empty list of non-negative numbers"
    return None if abs(sum(v) - 1.0) < 1e-9 else f"must sum to 1 (got {sum(v)})"


def _pos_int_list(v: Any) -> str | None:
    if not isinstance(v, list) or not v or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in v):
        return "must be a non-empty list of integers >= 1"
    return None


_RULES: dict[str, Any] = {
    "project.name": _string,
    "seed": _num(0, None, integer=True),
    "data.raw_dir": _string,
    "data.processed_dir": _string,
    "data.store_id": _string,
    "data.dept_id": _string,
    "data.n_skus": _num(1, 5000, integer=True),
    "data.selection.mode": _choice("pre_backtest", "final_year"),
    "data.selection.weeks": _num(4, 260, integer=True),
    "horizons.review_period_weeks": _num(1, 1, integer=True),
    "horizons.forecast_horizon_weeks": _num(1, 12, integer=True),
    "horizons.cash_horizon_weeks": _num(1, 12, integer=True),
    "forecast.model": _choice("lgbm", "seasonal_naive", "naive"),
    "forecast.refit_every_weeks": _num(1, 52, integer=True),
    "forecast.train_window_weeks": _num(8, 400, integer=True),
    "forecast.min_train_weeks": _num(4, 400, integer=True),
    "forecast.warmup_weeks": _num(0, 104, integer=True),
    "forecast.baseline_residual_weeks": _num(4, 156, integer=True),
    "forecast.lgbm.n_estimators": _num(1, 5000, integer=True),
    "forecast.lgbm.learning_rate": _num(0, 1, lo_open=True),
    "forecast.lgbm.num_leaves": _num(2, 1024, integer=True),
    "forecast.lgbm.min_data_in_leaf": _num(1, None, integer=True),
    "forecast.lgbm.feature_fraction": _num(0, 1, lo_open=True),
    "forecast.lgbm.bagging_fraction": _num(0, 1, lo_open=True),
    "forecast.lgbm.bagging_freq": _num(0, None, integer=True),
    "forecast.lgbm.lambda_l2": _num(0, None),
    "forecast.lgbm.num_threads": _num(1, 64, integer=True),
    "forecast.conformal.enabled": _choice(True, False),
    "forecast.conformal.calibration_weeks": _num(4, 104, integer=True),
    "forecast.conformal.min_samples": _num(1, None, integer=True),
    "forecast.conformal.scale_floor": _num(0, None, lo_open=True),
    "sampling.method": _choice("block", "independent"),
    "sampling.bank_weeks": _num(4, 156, integer=True),
    "sampling.block_len": _num(1, 12, integer=True),
    "sampling.n_paths_backtest": _num(1, 100000, integer=True),
    "sampling.n_paths_app": _num(1, 100000, integer=True),
    "sampling.tail_cap": _num(0.95, 0.99999, lo_open=True),
    "synthetic.lead_time_weeks.values": _pos_int_list,
    "synthetic.lead_time_weeks.probs": _probs,
    "synthetic.pack_sizes.values": _pos_int_list,
    "synthetic.pack_sizes.max_share_of_weekly_demand": _num(0, 10, lo_open=True),
    "synthetic.moq_packs.values": _pos_int_list,
    "synthetic.moq_packs.probs": _probs,
    "synthetic.perishable_share": _num(0, 1),
    "synthetic.spoilage_rate_per_week": _num(0, 1),
    "synthetic.salvage_frac_of_cost": _num(0, 1),
    "synthetic.holding_rate_annual": _num(0, 5),
    "synthetic.capital_rate_annual": _num(0, 5),
    "synthetic.fixed_cost_share_of_gross_margin": _num(0, 3),
    "synthetic.fixed_cost_lump_every_weeks": _num(1, 52, integer=True),
    "synthetic.fixed_cost_lump_offset": _num(0, 51, integer=True),
    "synthetic.calibration_weeks": _num(1, 104, integer=True),
    "synthetic.burn_in_weeks": _num(0, 52, integer=True),
    "risk.alpha": _num(0, 0.5, lo_open=True),
    "risk.buffer_weeks_of_fixed_costs": _num(0, 52),
    "gate.mode": _choice("mc", "deterministic"),
    "gate.deterministic_quantile": _choice(*QUANTILES),
    "gate.continuation": _choice("replace_sales", "none"),
    "gate.charge_beyond_horizon": _choice(True, False),
    "gate.tol_frac": _num(0, 0.5, lo_open=True),
    "gate.tol_abs": _num(0, None),
    "gate.max_bisection_iter": _num(1, 60, integer=True),
    "gate.n_grid": _num(2, 200, integer=True),
    "gate.on_infeasible": _choice("zero", "best_effort"),
    "allocator.perishable_overage": _choice("spoilage", "full_loss"),
    "policies.reorder_point.z": _num(0, 6),
    "policies.otb.target_weeks_cover": lambda v: None if v == "auto" else _num(0, 26)(v),
    "policies.otb.planned_markdowns": _num(0, None),
    "world.shortfall_rule": _choice("overdraft", "cut_proportional"),
    "llm.enabled": _choice(True, False),
    "llm.model": _string,
    "llm.max_tokens": _num(64, 128000, integer=True),
    "llm.effort": _choice("low", "medium", "high", "xhigh", "max"),
    "llm.refusal_fallback": _choice(True, False),
    "llm.max_skus_per_call": _num(1, 500, integer=True),
    "llm.cache_dir": _string,
}


def validate_config(cfg: dict) -> None:
    """Raise ConfigError listing every problem found."""
    errors: list[str] = []
    for path, check in _RULES.items():
        try:
            value = _get(cfg, path)
        except KeyError:
            errors.append(f"{path}: missing")
            continue
        msg = check(value)
        if msg:
            errors.append(f"{path}: {msg} (got {value!r})")

    def safe(path: str, default: Any = None) -> Any:
        try:
            return _get(cfg, path)
        except KeyError:
            return default

    q = safe("forecast.quantiles")
    if q is None or [round(float(x), 4) for x in q] != [round(x, 4) for x in QUANTILES]:
        errors.append(f"forecast.quantiles: must be exactly {list(QUANTILES)} (got {q!r})")
    mr = safe("synthetic.margin_range")
    if not (isinstance(mr, list) and len(mr) == 2 and all(isinstance(x, (int, float)) for x in mr) and 0 < mr[0] <= mr[1] < 1):
        errors.append(f"synthetic.margin_range: must be [lo, hi] with 0 < lo <= hi < 1 (got {mr!r})")
    for key in ("lead_time_weeks", "moq_packs"):
        vals, probs = safe(f"synthetic.{key}.values", []), safe(f"synthetic.{key}.probs", [])
        if isinstance(vals, list) and isinstance(probs, list) and len(vals) != len(probs):
            errors.append(f"synthetic.{key}: values and probs must have the same length")
    sup = safe("synthetic.suppliers")
    if not isinstance(sup, list) or not sup:
        errors.append("synthetic.suppliers: must be a non-empty list")
    else:
        for i, s in enumerate(sup):
            if not isinstance(s, dict) or not {"name", "share", "pay_delay_weeks"} <= set(s):
                errors.append(f"synthetic.suppliers[{i}]: needs name, share, pay_delay_weeks")
            elif not isinstance(s["pay_delay_weeks"], int) or s["pay_delay_weeks"] < 0 or s["share"] < 0:
                errors.append(f"synthetic.suppliers[{i}]: share >= 0 and integer pay_delay_weeks >= 0 required")
        if all(isinstance(s, dict) and "share" in s for s in sup) and abs(sum(s["share"] for s in sup) - 1.0) > 1e-9:
            errors.append("synthetic.suppliers: shares must sum to 1")
    lts = safe("synthetic.lead_time_weeks.values", [])
    review, fh, ch = safe("horizons.review_period_weeks", 1), safe("horizons.forecast_horizon_weeks", 0), safe("horizons.cash_horizon_weeks", 0)
    if isinstance(lts, list) and lts and all(isinstance(x, int) for x in lts) and isinstance(fh, int) and isinstance(review, int):
        if max(lts) + review > fh:
            errors.append(f"horizons.forecast_horizon_weeks ({fh}) must cover max lead time + review period ({max(lts) + review})")
    delays = [x["pay_delay_weeks"] for x in sup if isinstance(x, dict) and isinstance(x.get("pay_delay_weeks"), int)] if isinstance(sup, list) else []
    if isinstance(lts, list) and lts and all(isinstance(x, int) for x in lts) and delays and isinstance(ch, int) and not safe("gate.charge_beyond_horizon", False):
        if max(lts) + max(delays) >= ch:
            errors.append(
                f"horizons.cash_horizon_weeks ({ch}) must exceed max lead time + max payment delay ({max(lts) + max(delays)}): otherwise "
                "bills for this week's order fall outside the projection and buying on credit looks free. Lengthen the horizons, shorten the terms, or set gate.charge_beyond_horizon: true"
            )
    if isinstance(fh, int) and isinstance(ch, int) and ch > fh:
        errors.append("horizons.cash_horizon_weeks must be <= horizons.forecast_horizon_weeks")
    if isinstance(safe("sampling.block_len"), int) and isinstance(fh, int) and safe("sampling.block_len") > fh:
        errors.append("sampling.block_len must be <= horizons.forecast_horizon_weeks")
    lump_every, lump_off = safe("synthetic.fixed_cost_lump_every_weeks", 1), safe("synthetic.fixed_cost_lump_offset", 0)
    if isinstance(lump_every, int) and isinstance(lump_off, int) and lump_off >= lump_every:
        errors.append("synthetic.fixed_cost_lump_offset must be < synthetic.fixed_cost_lump_every_weeks")
    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))


def validate_experiments(exp: dict) -> None:
    errors: list[str] = []
    wins = exp.get("windows")
    if not isinstance(wins, list) or not wins:
        errors.append("windows: must be a non-empty list")
    else:
        names = [w.get("name") for w in wins if isinstance(w, dict)]
        if len(set(names)) != len(wins):
            errors.append("windows: names must be unique")
        for w in wins:
            if not isinstance(w, dict) or not {"name", "weeks_before_end", "n_weeks", "role"} <= set(w):
                errors.append(f"windows: each needs name, weeks_before_end, n_weeks, role (got {w!r})")
                continue
            if w["role"] not in ("tune", "holdout"):
                errors.append(f"windows[{w['name']}].role: must be tune or holdout")
            if not isinstance(w["n_weeks"], int) or w["n_weeks"] < 1 or not isinstance(w["weeks_before_end"], int) or w["n_weeks"] > w["weeks_before_end"]:
                errors.append(f"windows[{w['name']}]: need integer 1 <= n_weeks <= weeks_before_end")
    for key in ("stress_levels", "start_cash_weeks"):
        v = exp.get(key)
        if not isinstance(v, list) or not v or any(not isinstance(x, (int, float)) for x in v):
            errors.append(f"{key}: must be a non-empty list of numbers")
    if isinstance(exp.get("stress_levels"), list) and any(not (0 < s <= 1) for s in exp["stress_levels"] if isinstance(s, (int, float))):
        errors.append("stress_levels: each must be in (0, 1]")
    if exp.get("default_stress") not in (exp.get("stress_levels") or []):
        errors.append("default_stress: must be one of stress_levels")
    if exp.get("default_start_cash_weeks") not in (exp.get("start_cash_weeks") or []):
        errors.append("default_start_cash_weeks: must be one of start_cash_weeks")
    rules = exp.get("shortfall_rules")
    if not isinstance(rules, list) or not rules or any(r not in ("overdraft", "cut_proportional") for r in rules):
        errors.append("shortfall_rules: must be a non-empty subset of [overdraft, cut_proportional]")
    pols = exp.get("policies")
    if not isinstance(pols, list) or not pols or any(p not in POLICY_NAMES for p in pols):
        errors.append(f"policies: must be a non-empty subset of {list(POLICY_NAMES)}")
    elif not {"A", "B", "C"} <= set(pols):
        errors.append("policies: A, B and C are required for the headline")
    bs = exp.get("bootstrap", {})
    if not (isinstance(bs.get("n_boot"), int) and bs["n_boot"] >= 10 and isinstance(bs.get("block_weeks"), int) and bs["block_weeks"] >= 1 and 0.5 <= bs.get("ci", 0) < 1):
        errors.append("bootstrap: need n_boot >= 10, block_weeks >= 1, 0.5 <= ci < 1")
    hl = exp.get("headline", {})
    if not isinstance(hl.get("roles"), list) or not hl.get("roles") or hl.get("shortfall_rule") not in (rules or []):
        errors.append("headline: needs roles (list) and a shortfall_rule that is in shortfall_rules")
    ab = exp.get("app_bundle", {})
    if isinstance(wins, list) and ab.get("window") not in [w.get("name") for w in wins if isinstance(w, dict)]:
        errors.append("app_bundle.window: must name one of the windows")
    if ab.get("policy", "C") not in ("C", "C_plus") or (isinstance(pols, list) and ab.get("policy", "C") not in pols):
        errors.append("app_bundle.policy: must be C or C_plus, and be one of the policies being run")
    if not isinstance(exp.get("n_jobs"), int) or exp["n_jobs"] < 1:
        errors.append("n_jobs: must be an integer >= 1")
    if errors:
        raise ConfigError("Invalid experiments config:\n  - " + "\n  - ".join(errors))


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if overrides:
        raw = deep_merge(raw, overrides)
    validate_config(raw)
    return _wrap(raw)


def load_experiments(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    path = Path(path) if path else DEFAULT_EXPERIMENTS
    if not path.exists():
        raise ConfigError(f"Experiments file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if overrides:
        raw = deep_merge(raw, overrides)
    validate_experiments(raw)
    return _wrap(raw)


def config_hash(*cfgs: dict) -> str:
    """Short stable hash of one or more configs (canonical JSON)."""
    payload = json.dumps([_unwrap(c) for c in cfgs], sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def resolve_path(value: str | Path) -> Path:
    """Paths in config are relative to the repo root unless absolute."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def to_plain(cfg: dict) -> dict:
    return _unwrap(cfg)


def as_config(plain: dict) -> Config:
    return _wrap(plain)


def project_name(cfg: dict | None = None) -> str:
    if cfg is not None:
        return cfg["project"]["name"]
    return (yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}).get("project", {}).get("name", "cspa")


# Single place the display name comes from. Rename in configs/default.yaml -> project.name.
PROJECT_NAME = project_name() if DEFAULT_CONFIG.exists() else "cspa"
