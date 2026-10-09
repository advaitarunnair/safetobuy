"""Shared plumbing for the scripts: import path, CLI arguments, run manifests."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cspa.config import config_hash, load_config, load_experiments, resolve_path, to_plain  # noqa: E402
from cspa.data.load_m5 import MissingDataError, Panel  # noqa: E402
from cspa.data.synth_params import SkuParams  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to config yaml (default configs/default.yaml)")
    p.add_argument("--experiments", default=None, help="path to experiments yaml (default configs/experiments.yaml)")
    return p


def load_all(args):
    return load_config(args.config), load_experiments(args.experiments)


def processed_dir(cfg) -> Path:
    return resolve_path(cfg.data.processed_dir)


def load_panel_and_params(cfg) -> tuple[Panel, SkuParams]:
    import pandas as pd

    d = processed_dir(cfg)
    panel = Panel.load(d)
    path = d / "sku_params.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    return panel, SkuParams.from_frame(pd.read_parquet(path))


def git_info() -> dict:
    def run(*cmd: str) -> str | None:
        try:
            return subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def results_root() -> Path:
    """results/ in the repo, or $CSPA_RESULTS_DIR (used by tests so fixture runs never land in results/)."""
    import os

    override = os.environ.get("CSPA_RESULTS_DIR")
    return Path(override) if override else ROOT / "results"


def resolve_run(run_id: str | None) -> Path:
    """results/<run_id>, defaulting to the run named in results/LATEST."""
    root = results_root()
    if run_id is None:
        latest = root / "LATEST"
        if not latest.exists():
            raise FileNotFoundError("No results yet (results/LATEST missing). Run `make backtest` first.")
        run_id = latest.read_text().strip()
    path = root / run_id
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"{path} is not a completed run (manifest.json missing).")
    return path


def ensure_openmp() -> None:
    """macOS only: if LightGBM cannot find libomp, re-run this script once with the copy bundled
    in the scikit-learn wheel on the dyld fallback path. `make` sets this up front; this covers
    running a script directly."""
    import importlib.util
    import os

    if sys.platform != "darwin" or os.environ.get("CSPA_OMP_RETRY"):
        return
    try:
        import lightgbm  # noqa: F401

        return
    except OSError:
        pass
    spec = importlib.util.find_spec("sklearn")
    dylibs = Path(spec.origin).parent / ".dylibs" if spec else None
    if dylibs is None or not (dylibs / "libomp.dylib").exists():
        return
    env = dict(os.environ, CSPA_OMP_RETRY="1")
    env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{dylibs}:{env.get('DYLD_FALLBACK_LIBRARY_PATH', '/usr/local/lib:/usr/lib')}"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def exit_missing_data(exc: MissingDataError) -> None:
    print(str(exc), file=sys.stderr)
    sys.exit(2)


__all__ = [
    "ROOT", "MissingDataError", "Panel", "SkuParams", "base_parser", "config_hash", "ensure_openmp", "exit_missing_data", "git_info", "load_all",
    "load_panel_and_params", "now_utc", "processed_dir", "resolve_path", "resolve_run", "results_root", "to_plain", "write_json",
]
