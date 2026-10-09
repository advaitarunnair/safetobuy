# Every phase is runnable from here.  make all = data -> forecast -> backtest -> figures -> render
PY ?= python
CONFIG ?= configs/default.yaml
EXPERIMENTS ?= configs/experiments.yaml
ARGS = --config $(CONFIG) --experiments $(EXPERIMENTS)

# macOS: LightGBM needs an OpenMP runtime (libomp). If Homebrew's is not installed, fall back to the
# copy bundled inside the scikit-learn wheel. The variable must be set on the command itself:
# macOS strips DYLD_* from the environment of /bin/sh, so a plain `export` would be lost.
OMP_DIR := $(shell $(PY) -c "import importlib.util as u, pathlib as p; s = u.find_spec('sklearn'); print(p.Path(s.origin).parent / '.dylibs' if s else '')" 2>/dev/null)
RUN = DYLD_FALLBACK_LIBRARY_PATH="$(OMP_DIR):/usr/local/lib:/usr/lib" PYTHONPATH=src $(PY)

.PHONY: help setup data forecast quick tune backtest figures render placeholder app test all clean

help:
	@echo "make setup      create .venv and install pinned requirements (Python 3.11)"
	@echo "make data       Phase 0: slice M5, weekly aggregation, synthetic SKU parameters"
	@echo "make forecast   Phase 2: rolling-origin quantile forecasts + evaluation (cached)"
	@echo "make quick      Phase 1: one window, policies A and B only"
	@echo "make tune       backtest on tuning windows only (never touches holdout)"
	@echo "make backtest   full backtest grid -> results/<run_id>/"
	@echo "make figures    PNG figures from the latest run"
	@echo "make render     fill README / docs result blocks from the latest run"
	@echo "make app        start the Streamlit app"
	@echo "make test       run the test-suite (synthetic fixture only)"
	@echo "make all        data forecast backtest figures render"

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "now: source .venv/bin/activate"

data:
	$(RUN) scripts/prepare_data.py $(ARGS)

forecast:
	$(RUN) scripts/train_forecast.py $(ARGS)

quick:
	$(RUN) scripts/run_backtest.py $(ARGS) --quick

tune:
	$(RUN) scripts/run_backtest.py $(ARGS) --tune-only

backtest:
	$(RUN) scripts/run_backtest.py $(ARGS)

figures:
	$(RUN) scripts/make_figures.py

render:
	$(RUN) scripts/render_results.py

placeholder:
	$(RUN) scripts/render_results.py --placeholder

app:
	$(RUN) -m streamlit run app/streamlit_app.py

test:
	$(RUN) -m pytest

all: data forecast backtest figures render

clean:
	rm -rf data/processed/*.parquet data/processed/*.json data/processed/*.csv data/processed/llm_cache .pytest_cache
