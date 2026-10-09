# <!-- BEGIN:project_name -->SafeToBuy<!-- END:project_name -->

**Open-to-buy gives you a budget. We tell you whether you can afford it, and how to spend it best.**

A purchasing assistant for small retailers. It forecasts demand with calibrated uncertainty,
projects the shop's cash over the coming weeks, finds the largest purchasing budget that keeps
the chance of a cash shortfall within a stated tolerance, spends that budget where expected
profit per dollar is highest, and explains every buy, partial buy and deferral.

ForgeHacks 2026, theme "AI for Real World Problems", track AI + Business.

<!-- BEGIN:headline -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:headline -->

Every number in the Results section is written by `scripts/render_results.py` from a run
manifest. None is typed by hand.

---

## Problem

Small retailers fail from cash crunches, not only from slow sales. Inventory tools recommend
reorder quantities one product at a time, so the recommendations for a week can add up to more
than the shop can pay for. The owner is then left to cut the list by gut feel, at exactly the
moment a wrong cut costs the most: under-buy the fast sellers and revenue drops; buy everything
and the rent cheque bounces.

## Target users

Small retailers, convenience stores and independent e-commerce sellers who need purchasing
guidance without the complexity of enterprise planning software. The demo persona is a small
Singapore minimart. That persona is framing only: the demand data is US Walmart data and the
costs are synthetic (see [Data sources](#data-sources-attribution-and-licence) and
[Synthetic data disclosure](#synthetic-data-disclosure)).

## Illustrative scenario

This is an illustration of the mechanism, not an output of the system.

A shop's replenishment list for the week comes to $3,000, but only $2,000 can be spent without
putting next month's rent at risk. A per-product tool stops at the $3,000 list. This system:

1. **Says how much is safe.** It simulates the next four weeks of sales, supplier bills and fixed costs many times and reports the largest budget that keeps cash above the shop's buffer with the confidence the owner asked for. That is where the $2,000 comes from.
2. **Prioritises.** It ranks every unit on the list by expected profit per dollar: margin if it sells before the next delivery, a small carrying cost if it does not. Units of fast, high-margin products that are nearly sure to sell are funded first. The fifth pack of a slow product comes last.
3. **Defers the rest and prices the deferral.** For each product that is trimmed or skipped it shows the chance of running out before the next delivery and the expected margin lost by waiting a week.
4. **Explains in plain language**, one line per product.

## Approach and architecture

![Architecture](docs/architecture.png)

Mermaid source and the weekly loop: [docs/architecture.md](docs/architecture.md).

| Step | What it does | Where |
| --- | --- | --- |
| 1. Forecast | LightGBM quantile regression, one pooled model per quantile (5% to 95%), rolling conformal calibration of the intervals. Features use only data up to the forecast origin. | `src/cspa/forecast/`, `src/cspa/data/features.py` |
| 2. Sample | Joint demand paths for all SKUs by bootstrapping whole past weeks of forecast errors, so SKUs that were surprised together stay together. | `src/cspa/forecast/sampling.py` |
| 3. Project cash | Monte Carlo cash recursion per path: `cash = cash + revenue - fixed costs - supplier payments`, with sales capped by stock, existing payables and pipeline, and the candidate plan. | `src/cspa/cash/simulate.py` |
| 4. Cash gate | Largest budget B with `P(min cash >= buffer) >= 1 - alpha`. Bisection with common random numbers, then a grid check because the search is not monotone. If no budget meets the target, the budget with the best chance is used and the plan is flagged. | `src/cspa/cash/gate.py` |
| 5. Allocate | Greedy by marginal expected profit per dollar: `Cu * P(D > x) - Co * P(D <= x)` per unit, in pack sizes, with MOQs. | `src/cspa/allocate/` |
| 6. Explain | Facts per SKU and week, then templates or an LLM whose numbers are verified in code. | `src/cspa/explain/` |
| 7. Prove | Closed-loop backtest on real weekly demand against reorder-point, open-to-buy and priority-cut policies, with ablations. | `src/cspa/sim/`, `src/cspa/policies/` |

Policies compared (all act in the same simulated world, on the same demand):

| Policy | Budget | Allocation |
| --- | --- | --- |
| A, reorder point | none: spends what the targets require | order up to forecast over lead time + review period + safety stock |
| B, open-to-buy | planned sales + planned markdowns + planned end-of-period stock - beginning stock - on order, at cost | split in proportion to A's order need |
| **C, proposed** | **cash gate** | **marginal profit per dollar** |
| D, priority cuts | open-to-buy | A's need funded in full, highest (stockout probability x margin) first |
| ablation | cash gate | proportional split |
| ablation | open-to-buy | marginal profit per dollar |

The open-to-buy form used is the standard retail one, planned sales + planned markdowns + planned
end-of-period inventory - beginning-of-period inventory, with on-order stock also subtracted, as
described by [Shopify](https://www.shopify.com/retail/open-to-buy-plans) and
[Inventory Planner](https://www.inventory-planner.com/buying-budgets-with-open-to-buy-tools/).
Here it is computed at cost and weekly; planned markdowns are zero because M5 has no markdown plan.

## Honest AI/ML characterization

**ML for prediction, simulation and optimization for the decision.**

- **Machine learning is used for prediction only:** LightGBM quantile forecasting with conformal calibration.
- **The decision is not ML.** The cash model is a Monte Carlo simulation. The budget search and the allocator are optimization (a constrained search and a greedy knapsack heuristic).
- **An LLM is used only to verbalize numbers** the system has already computed. It never produces a quantity, a budget or a probability. Code checks every number in its output against the supplied facts and falls back to a deterministic template on any mismatch. The app works with no API key.

This is not "an ML system" end to end, and nothing here should be read that way.

## How to run

### The app only (no data needed)

The repo includes a small results bundle, so the app runs straight from a clone:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

To deploy on Streamlit Community Cloud: new app from this repo, branch `main`, main file
`app/streamlit_app.py`, Python 3.11 under Advanced settings. `app/requirements.txt` is picked up
automatically. `ANTHROPIC_API_KEY` is an optional secret; without it explanations use templates.

### The full pipeline

Requirements: Python 3.11, about 400 MB of disk for the M5 files, macOS or Linux.

**1. Get the M5 data** (Kaggle login required; the raw files are not in this repo):

1. Open <https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>, sign in and accept the competition rules.
2. Download `sales_train_evaluation.csv`, `sell_prices.csv` and `calendar.csv` (or the full zip) and put the three CSVs in `data/raw/`.

With the Kaggle CLI: `kaggle competitions download -c m5-forecasting-accuracy -p data/raw && unzip data/raw/m5-forecasting-accuracy.zip -d data/raw`.
If the files are missing, `make data` stops and prints these steps. It never fabricates data.

**2. Install and run:**

```bash
make setup && source .venv/bin/activate   # Python 3.11 venv with pinned requirements
make test                                 # test-suite, runs on a synthetic fixture (no M5 needed)
make data                                 # slice, weekly aggregation, synthetic parameters
make forecast                             # rolling-origin quantile forecasts + evaluation (about 6 minutes)
make calibrate                            # tuning windows only: do the stress scenarios bind for policy A?
make tune                                 # every policy on tuning windows only (never touches holdout)
make sweep SET="--set risk.alpha=0.05,0.10"   # sensitivity on tuning windows only
make backtest                             # full backtest incl. held-out windows -> results/<run_id>/
make figures                              # PNG figures for the run
make render                               # fill the result blocks in this README and docs/
make screenshots                          # app screenshots -> docs/screenshots/ (headless Chrome)
make app                                  # Streamlit app
```

`make all` runs data, forecast, backtest, figures and render in order. The comparison slice uses
the same targets with `CONFIG=configs/household_1.yaml`.

macOS note: LightGBM needs an OpenMP runtime. The Makefile uses the copy bundled with
scikit-learn, so no Homebrew is needed; `brew install libomp` also works. Linux needs nothing extra.

**3. Optional, AI-worded explanations:** `export ANTHROPIC_API_KEY=...` before `make app` and tick
"AI-worded explanations" under Advanced. Model and effort are in `configs/default.yaml -> llm`.

Repository layout:

```
configs/        default.yaml (slice, horizons, synthetic params, risk, MC, seeds), experiments.yaml, household_1.yaml
src/cspa/       data/  forecast/  cash/  allocate/  policies/  sim/  explain/  config.py
scripts/        prepare_data, train_forecast, calibrate_scenarios, tune_sweep, run_backtest, make_figures,
                render_results, capture_screenshots, make_architecture
app/            streamlit_app.py, requirements.txt (lean, for deployment)
tests/          pytest suite + tests/fixtures/synthetic_m5_fixture.py (unit tests only)
docs/           ASSUMPTIONS.md, architecture.md, project_description.md, demo_script.md, tuning/, screenshots/
results/        <run_id>/ : manifest.json, metrics, headline.json, figures/, app/ bundle;  LATEST, SECONDARY
```

## Data sources, attribution and licence

- **Real data:** the [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) competition dataset on Kaggle: Walmart unit sales, sell prices and calendar, made available by Walmart and the Makridakis Open Forecasting Center (MOFC) at the University of Nicosia. Reference: Makridakis, Spiliotis and Assimakopoulos, "M5 accuracy competition: Results, findings, and conclusions", International Journal of Forecasting 38(4), 2022.
- **What this repo contains:** no raw M5 files (`data/raw/` is gitignored). The results bundle under `results/<run_id>/app/` holds derived weekly aggregates for one store and one 300-SKU slice per run, plus forecasts and simulation results, which is what the app needs to run.
- **Licence note:** the M5 data remains subject to the terms on the Kaggle competition page; anyone reusing the bundled aggregates should credit the M5 competition as above. The code in this repository is the team's own.
- **Slice, history, windows, seed, commit:** see [Run details](#run-details), generated from the run manifest.
- **Synthetic:** everything about money other than shelf prices. See the next section.

## Synthetic data disclosure

M5 contains no costs, lead times, supplier terms or cash positions. All of the following are
**synthetic**, generated from `configs/default.yaml` with a fixed seed, and documented one by one
in [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md).

<!-- BEGIN:synthetic_params -->
| Parameter | Value | Config key |
| --- | --- | --- |
| Unit cost | reference price x (1 - margin), margin ~ U(22%, 40%) per SKU; reference price = median M5 sell price in the selection window | synthetic.margin_range |
| Lead time | 1 wk (60%), 2 wk (40%) | synthetic.lead_time_weeks |
| Pack size | one of [1, 6, 12, 24] units, never more than 50% of a SKU's mean weekly units | synthetic.pack_sizes |
| MOQ | 1 pack(s) (70%), 2 pack(s) (30%) | synthetic.moq_packs |
| Supplier payment terms | S1_pay_on_delivery 60% of SKUs, pays 0 wk after delivery; S2_net7 40% of SKUs, pays 1 wk after delivery | synthetic.suppliers |
| Perishable SKUs | 20% of SKUs; 10% of leftover stock written off per week at 20% of cost | synthetic.perishable_share, spoilage_rate_per_week, salvage_frac_of_cost |
| Holding + capital cost | 20% + 15% of unit cost per year (decision parameter, not a simulated cash flow) | synthetic.holding_rate_annual, capital_rate_annual |
| Fixed costs | 95% of pre-window gross margin on average; part is paid as a lump every 4 weeks, sized by the stress level | synthetic.fixed_cost_* |
| Opening cash | buffer + cushion x s x R*, where s is the stress level and R* the ideal weekly replenishment cost; cushion is set per scenario | experiments.cash_cushions |
| Cash buffer | 2 weeks of average fixed costs | risk.buffer_weeks_of_fixed_costs |
| Opening stock, pipeline, payables | whatever policy A leaves after a 8-week burn-in with ample cash | synthetic.burn_in_weeks |
| Seed | 20261009 | seed |
<!-- END:synthetic_params -->

Consequences:

- **Dollar figures are illustrative.** Only demand patterns and shelf prices are real.
- **The meaningful result is the relative comparison between policies** in the same simulated world, on the same demand, with the same parameters.
- The Singapore minimart persona is framing. Nothing here is calibrated to Singapore costs, wages, rents or supplier terms.

## Results

### Headline

<!-- BEGIN:headline -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:headline -->

The headline is computed on **held-out windows only** (never used to choose settings), over the
three budget-stress levels. The planned wording ("X% fewer cash shortfalls than policy A, and Y%
higher margin than policy B at equal or lower risk") is used only if every part holds and both
differences have bootstrap intervals clear of zero. Otherwise the generated sentence says what
happened.

### What the held-out data supports, and what it does not

<!-- BEGIN:verdict -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:verdict -->

### All policies, headline scope

<!-- BEGIN:results_table -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:results_table -->

Shortfall week = a week ending with cash below the buffer. Insolvency week = cash below zero.
Net position = cash + stock at cost + pipeline at cost - payables. Lost margin = unit margin on
demand that met an empty shelf.

### By budget stress

Budget stress s tightens two things. In the heaviest fixed-cost week of each four-week cycle,
expected revenue minus fixed costs covers only s of the ideal weekly replenishment spend R*; and
the shop opens with free cash above its buffer of only `cushion x s x R*`. Lower is tighter.

<!-- BEGIN:stress_table -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:stress_table -->

### By opening cash

<!-- BEGIN:start_cash_table -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:start_cash_table -->

### Robustness of the headline

The same comparison under the alternative cash-shortfall rule, on the tuning windows, per window
and per stress level.

<!-- BEGIN:robustness_table -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:robustness_table -->

### Comparison slice

<!-- BEGIN:secondary_slice -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:secondary_slice -->

### Forecast quality

<!-- BEGIN:forecast_table -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:forecast_table -->

wQL is twice the mean pinball loss over the seven quantiles divided by mean demand. Coverage is
the share of realised weekly sales inside each central interval; the targets are 50%, 80%, 90%.

### Figures

<!-- BEGIN:figures -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:figures -->

### Run details

<!-- BEGIN:run_info -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:run_info -->

## How the settings were chosen

Four consecutive 26-week windows end at the end of the M5 history. The first two were used to
choose every setting; the last two were simulated once, after the settings were frozen, and are
the only source of the headline. The console output of each tuning-window comparison is saved in
[docs/tuning/](docs/tuning/), and [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) section 8 says what
changed and why. In short:

- **Scenarios** were recalibrated until the reorder-point policy really ran short of cash at the 50% and 70% stress levels (`01_scenario_calibration_*.txt`).
- **The cash gate's fallback.** When no budget meets the target, policy C takes the budget with the best chance of staying above the buffer instead of buying nothing (`04_infeasible_rule.txt`).
- **The cash projection** assumes later weeks replace what was sold, but never by spending below the buffer (`02_gate_projection_and_tiebreak.txt`).
- **Risk tolerance, capital charge and ranking rule** were swept and left at their starting values because nothing else did better (`03_alpha_capital_ranking.txt`).

## Tests

`make test` runs on a small synthetic fixture and covers:

- **No lookahead:** all sales and prices after a cutoff are replaced by noise (and, separately, deleted); features, forecasts, conformal calibration, sampled demand and every policy's order must be byte-identical. The conformal window must end before the cutoff.
- **Allocator feasibility:** spend never exceeds the budget; whole units, whole packs, MOQs respected; nothing with non-positive marginal value is funded.
- **Allocator optimality:** greedy equals brute force on tiny instances with equal unit costs, and is within one unit's value otherwise.
- **Cash conservation:** every week, cash change equals revenue minus payments minus fixed costs, and stock change equals arrivals minus sales (minus write-offs for perishables).
- **Gate correctness:** the returned budget satisfies the probability constraint on the same random numbers; when no budget does, the result is flagged and is the best of everything evaluated.
- **LLM guardrail:** a response with an altered number is rejected and the template is used.
- **Policy fairness:** every policy receives identical demand realisations and parameters.
- **Reproducibility:** a fixed seed gives identical results, including bootstrap intervals.
- **Honest headline:** the planned wording is only produced when every part holds with intervals clear of zero.

## Limitations

- **Synthetic costs, lead times, payment terms, fixed costs and cash.** Dollar results are illustrative. Different synthetic parameters would give different dollar figures and could change the ranking of policies.
- **The stress scenarios were built to be tight.** Opening cash and the monthly lump of fixed costs were set, on tuning windows, so that buying regardless of cash runs short. In comfortable scenarios the policies differ much less (see the comparison slice).
- **US data, Singapore persona.** M5 is Walmart data from California. The minimart persona is framing only.
- **M5 sales are censored and intermittent.** Recorded sales are capped by Walmart's own stockouts, so true demand is sometimes higher than what the forecaster learns from and what the backtest replays. Many SKU-weeks are zero. The slice is a set of high-volume items chosen before the backtest period.
- **Correlation handling.** Cross-SKU dependence comes from resampling whole past weeks of forecast errors (about a year of them), mapped through each SKU's predicted quantiles. This assumes the near future co-moves like the recent past, gives a coarse joint distribution, and extrapolates tails beyond the 95% quantile with a fitted exponential.
- **Cash projection assumption.** For weeks after the current order the projection assumes one-for-one replenishment that never spends below the buffer. That is a modelling choice, and the probabilities it produces are estimates, not guarantees.
- **The budget search is not monotone**, because stock bought now earns revenue. The gate checks a coarse grid after bisecting and keeps the largest feasible budget found; it does not guarantee the global maximum.
  <!-- BEGIN:gate_diagnostics -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:gate_diagnostics -->
- **A risk tolerance is a tolerance.** With alpha = 10% the gate accepts up to a 10% chance of a shortfall at each decision, and in the stressed scenarios the target often could not be met at all. Policy C is expected to have shortfall weeks.
- **A chance constraint does not price what it gives up.** The gate protects the buffer; it does not weigh that against lost margin. Deferring stock that would have sold costs margin and, a few weeks later, cash.
- **The two halves do not simply add up.** Ablations on the headline scope:
  <!-- BEGIN:ablation_note -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:ablation_note -->
  The allocator is a greedy heuristic for a knapsack problem with packs and MOQs, not an exact solver, and neither it nor the cash model is ML.
- **Baselines are simple.** Policy B's open-to-buy budget is used as a cap whose target stock matches policy A's, with no markdown plan. A retailer's real OTB process would be tuned to its own plan.
- **Supplier terms are short** (pay on delivery, net-7) so that every bill for this week's order falls inside the 4-week cash horizon.
- **Competitive coverage is partial.** See the next section.
- The LLM path was tested with a fake client only; no live API key was available during the build.

## Related tools and the difference

- Shopify inventory apps such as "Restock: Reorder Forecasting" and "SP | AI Inventory Management" already do per-product ML demand forecasting, reorder quantities, lead times, safety stock and purchase-order suggestions.
- Open-to-buy planning, offered in tools like Inventory Planner, sets a buying budget from a deterministic formula.
- **This project** derives the budget from a probabilistic cash-risk constraint and then allocates it by marginal profit per dollar, with explained deferrals.

We do not claim that no tool does budget allocation. Our competitive review covered only
Shopify-style apps and open-to-buy content, not enterprise planning software.

## More

- [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md): every open choice, the value chosen, why, and how to change it; the tuning log.
- [docs/project_description.md](docs/project_description.md): written project description.
- [docs/demo_script.md](docs/demo_script.md): demo script, under four minutes.
- [docs/devpost_submission.md](docs/devpost_submission.md): ready-to-paste submission text.
