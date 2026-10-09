# Assumptions, open choices and deviations

Every OPEN item from the brief is resolved here with the value chosen, why, and how to change it.
SUGGESTED defaults that were changed are listed with the reason. `configs/default.yaml` and
`configs/experiments.yaml` are the source of truth for values; nothing in `src/` hard-codes them.

Numbers quoted in this file come from tuning-window runs whose console output is saved in
`docs/tuning/`. Held-out results are only in the README, written by `scripts/render_results.py`.

---

## 1. Decisions taken on the team's behalf (10 Oct 2026), and what is still yours

| # | Item | What was done | Where |
| --- | --- | --- | --- |
| 1 | **Project name** | **SafeToBuy.** Web searches for each candidate with "inventory", "retail", "app" (10 Oct 2026). SafeToBuy: no product by that name found in inventory or retail software; nearest hit is Verify2Buy, a consumer product-authenticity app. It also plays on "open-to-buy". Runners-up: **BufferBuy** (no product found; "Buffer" is a social-media scheduler with a Shopify app, a different category), **StockRunway** (nothing found), **CashStock** (no inventory product found, but reads as "Cash App stocks", i.e. share trading), **CashGate** (nothing found in the search; from general knowledge the word is also the name of a Malawian corruption scandal and a Swiss consumer lender), **BudgetLoop** (too close to "InventoryLoop: Reorder & PO", a Shopify reorder app, and to Finaloop), **TillSafe** (crowded by TillTech, TILL POS and a physical till safe by Tidel). **Rejected: RunwayBuy**, which is an existing live-video selling SaaS for fashion retail (gust.com/companies/runwaybuy). A search is not a trademark check. | `configs/default.yaml -> project.name` |
| 2 | **Policy C when no budget is safe** | Changed as instructed: C takes the budget that maximises P(min cash >= buffer) on the same demand paths, and the plan is flagged. C+ is retired. The original rule (budget zero) is kept as `gate.on_infeasible: zero` for comparison only. See 5.4. | `gate.on_infeasible` |
| 3 | **Primary slice** | CA_1 FOODS_3. The comparison slice CA_1 HOUSEHOLD_1 formally supports the planned headline, but only because policy A ran short in a handful of held-out weeks there, so it is reported as a comparison and not as the headline. See 6. | `results/LATEST`, `results/SECONDARY` |
| 4 | **Submission deadline** | 10 Oct 2026 12:00 pm EDT = 11 Oct 2026 00:00 Singapore time. Source: search-engine summaries of the Devpost page (the page itself is behind a bot check that was not bypassed) and the schedule on forgehacks.dev ("Hacking ends Oct 10, 12:00 PM. Submissions lock."), which gives no timezone. Confirm on Devpost before submitting. | n/a |
| 5 | **Results bundle is committed** | As instructed. It holds weekly aggregates of one store and one 300-SKU slice per run, forecasts and simulation results; raw CSVs stay gitignored. README has the attribution note. | `results/<run_id>/app/` |
| 6 | **LLM model** | `claude-haiku-5-5`, low effort, as instructed. Server-side refusal fallback is off because it is not offered for Haiku; a refusal falls back to templates like any other failure. Not tested live (no API key on the build machine). | `configs/default.yaml -> llm` |
| 7 | **Deployment and CSV upload** | Not deployed (needs your Streamlit login); click-by-click steps are in the README. CSV upload is not included. | n/a |

Still yours: installing and authenticating `gh` so the repo can be created and pushed, the
Streamlit deploy click, the video, and the Devpost form.

---

## 2. Data

| Item | Choice | Why | How to change |
| --- | --- | --- | --- |
| Slice | Store `CA_1`, department `FOODS_3`. M5 calls FOODS_3 a department (`dept_id`); its `cat_id` is FOODS. Comparison slice: `CA_1`, `HOUSEHOLD_1`. | SUGGESTED default; comparison slice as instructed. | `data.store_id`, `data.dept_id`; `configs/household_1.yaml` |
| SKU count | Top 300 by units. The count and share of department volume are in the README run details. | SUGGESTED default. | `data.n_skus` |
| **SKU selection window (deviation from SUGGESTED)** | The 52 weeks ending just before the first simulated decision, not the final year of history. | Selecting on the final year uses sales from inside the backtest to choose which SKUs to test on. | `data.selection.mode: final_year` restores the SUGGESTED rule |
| Weekly aggregation | Walmart weeks (`wm_yr_wk`, Saturday to Friday). Only weeks with 7 days of sales are kept; the last 2 days of `sales_train_evaluation.csv` form a partial week and are dropped, leaving 277 weeks. | Decisions are weekly. | `data/load_m5.py` |
| Prices | Real M5 `sell_prices.csv`, forward-filled per SKU. Weeks before a SKU's first listing have no price and zero demand. | No backfill, so no price from the future leaks into features. | n/a |
| Missing data | `make data` stops with exact Kaggle instructions. Nothing is fabricated. | DECIDED. | n/a |
| Test fixture | `tests/fixtures/synthetic_m5_fixture.py` writes M5-shaped made-up CSVs with a marker file. Everything downstream is labelled `SYNTHETIC_TEST_FIXTURE`, the app shows a red banner, and `render_results.py` refuses to publish such a run. | Lets the whole pipeline be tested without M5. | n/a |
| Where the raw files came from | `~/Downloads` on the build machine (two of the three were zipped), copied to `data/raw/`. The manifest records their sizes and SHA-256 hashes. | As instructed. | n/a |

Known properties of M5 (also in README Limitations): sales are censored by Walmart's own
stockouts, and many SKU-weeks are zero.

---

## 3. Timing, horizons and the simulated world

| Item | Choice | Why | How to change |
| --- | --- | --- | --- |
| Review period | 1 week. | SUGGESTED. | `horizons.review_period_weeks` (only 1 is supported) |
| Forecast and cash horizons | 4 weeks each. Config validation requires the forecast horizon to cover max lead time + review period, and the cash horizon to exceed max lead time + max payment delay. | SUGGESTED. | `horizons.*` |
| Decision timing | The decision for week t is made before week t's demand is known, using data up to week t-1. | Matches `decide(state, info)` then `step(orders, realized_demand)`. | n/a |
| Step order | Receive arrivals; realise sales (min of demand and stock, lost sales not backordered); write off spoiled perishables; receive revenue; pay due payables and fixed costs; place new orders. | DECIDED order, plus the spoilage step that the perishability flag requires (see 4). | `sim/world.py` |
| Lead time meaning | An order placed in week t arrives at the start of week t + L and can be sold that week. Its cover period is L + 1 weeks. | Standard periodic review. | n/a |
| Payment timing | The payable is created at order placement and falls due `pay_delay` weeks after delivery. | Weekly granularity. | `synthetic.suppliers` |
| **Cash-shortfall rule (OPEN)** | Default `overdraft`: orders are never cancelled, cash may go negative; a week with cash below the buffer is a shortfall week, below zero an insolvency week. Alternative `cut_proportional`: at placement, if the order's cost exceeds cash on hand, every line is scaled by the same factor (rounded down to packs, lines under MOQ dropped). Both are run; the headline uses `overdraft`, fixed before the held-out windows were simulated. | SUGGESTED default plus the suggested alternative. | `world.shortfall_rule`, `experiments.shortfall_rules`, `experiments.headline` |
| Realised demand | Actual M5 weekly units. Policies see only an `InfoSet` cut off at week t-1. | DECIDED. | n/a |
| Opening state | Policy A runs for 8 burn-in weeks before each window with ample cash. Every policy then starts from that same stock, pipeline and payables; cash is reset to the scenario's opening cash. | A shop mid-operation owes suppliers and has stock in transit. | `synthetic.burn_in_weeks` |

---

## 4. Synthetic parameters (all values in `configs/default.yaml -> synthetic`, seeded)

The README table of these values is generated from the config. Only demand and shelf prices are real.

| Item | Choice | Why |
| --- | --- | --- |
| Unit cost | `reference price x (1 - margin)`, margin drawn per SKU from the configured range. Reference price = median M5 price in the selection window. **Cost is fixed over time**; it does not follow promotional price drops. | Promotions cut margin in real life. |
| Lead times | 1 or 2 weeks, drawn per SKU. | SUGGESTED. |
| Pack sizes / MOQ | Pack drawn from the configured list but never more than half a SKU's mean weekly units; MOQ is 1 or 2 packs. | Avoids a 24-pack for a SKU that sells 3 a week. |
| Payment terms | Two suppliers: pay on delivery (60% of SKUs) and net-7 (40%). **Net-14 and net-30 were dropped.** | With a 4-week horizon a net-30 bill for goods ordered today lands in week 5 or 6, outside the projection, so buying on those terms looked free to the gate. Longer terms need longer horizons (7 weeks for net-30). `gate.charge_beyond_horizon: true` allows them by counting late bills in the last projected week, but that double-counts against older bills. Supplier-specific terms were first on the cut list. |
| Receivables | Not modelled: sales are cash. | Retail; cut-first item. |
| **Perishability** | A share of SKUs is flagged perishable. In the simulated world a fixed share of their leftover stock is written off each week at salvage value. | A flag that changed the allocator's costs but not the world would bias the comparison. This adds a write-off term to the inventory identity; for non-perishables the identity is exactly the one in the brief. Tests cover both. |
| Holding and capital cost | Annual rates on unit cost. **Decision parameters only**: they enter the allocator's overage cost and are not charged as cash flows. | Keeps the cash recursion exactly as specified. |
| Seeds | One global seed. Each SKU's parameters come from a stream keyed on (seed, SKU id). Demand paths for week t are seeded with (seed, t), so every policy and every gate evaluation sees the same paths. | Reproducibility and common random numbers. |

---

## 5. Method choices

### 5.1 Forecasting

| Item | Choice | How to change |
| --- | --- | --- |
| Model | LightGBM, quantile objective, one model per quantile (7), pooled across SKUs and horizons (`h` is a feature). | `forecast.lgbm` |
| Scaling | Target and lag features are divided by the SKU's trailing 13-week mean + 1. Predictions are scaled back. | `data/features.py` |
| Features | Lags 0-3, rolling means and stds, zero share, trend, price level and relative price, listing age, same-week-last-year, and target-week calendar (week of year, month, SNAP days, event counts by type, Christmas). | `data/features.py` |
| Future-dated inputs | Only the calendar of the target week, which is published in advance as in the M5 competition. Sales and prices after the cutoff are never read; the no-lookahead tests replace them with noise and require byte-identical outputs. | n/a |
| Non-crossing | Quantiles are clipped at zero and sorted. | n/a |
| **Refit cadence (OPEN)** | Refit every 4 weeks, predict every week with fresh features. | `forecast.refit_every_weeks` |
| Baselines | Naive and seasonal naive, made into quantile forecasters with empirical quantiles of their own past errors. | `forecast/baselines.py` |
| **Conformal calibration** | Rolling, per quantile: shift each quantile by the matching empirical quantile of normalised past errors from forecasts issued in the previous 26 weeks and already realised. Pooled across SKUs. | `forecast.conformal` |
| Warm-up | 30 weeks of forecasts are issued before the first decision so calibration and the residual bank exist from week one. | `forecast.warmup_weeks` |
| Forecasts were not tuned | The forecaster's settings were fixed before any real data was seen and were not changed afterwards. | n/a |
| Pretrained benchmark (Chronos) | Skipped (cut-first). | n/a |

### 5.2 Joint demand sampling (OPEN: correlation method)

Weekly residual-block bootstrap on the PIT scale, as suggested. For each past forecast origin
whose targets are realised, store the PIT of realised demand under that origin's forecast, for
every SKU and horizon. A path picks a past origin and reuses its whole cross-section, so
cross-SKU dependence is whatever happened. Horizons are drawn in blocks of 2 consecutive weeks.

- Bank size: last 52 origins. Path count: 500 in the backtest, 2000 in the app (OPEN, SUGGESTED values).
- Tails: beyond the 95% quantile an exponential tail fitted to the q90-q95 gap; below 5% linear to zero. Capped at the 99.9% level. Flat regions use the mid-point PIT. Sampled demand is whole units.
- Assumption: dependence in the next four weeks resembles dependence in forecast errors over the past year. With 52 origins the joint distribution is coarse.
- `sampling.method: independent` shuffles origins per SKU (same marginals, no correlation). The app shows both probabilities for the same plan.

### 5.3 Cash projection

`cash_t = cash_{t-1} + revenue_t - fixed_costs_t - supplier_payments_t` on every path, in the
same order as the world, with sales capped by stock.

**Continuation assumption (not in the brief, found necessary, then refined on tuning windows).**
The projection needs to assume something about orders placed in later weeks of the horizon.

1. With no later orders (`none`), stock runs out inside four weeks, revenue collapses and nearly every plan looks unsafe.
2. With one-for-one replenishment (`replace_sales`), the projection blames this week's order for breaches that next week's gate would simply prevent. On tuning windows it was far too pessimistic: tracing one scenario, in 15 decisions where the predicted chance of staying safe was between about 20% and 85%, the shop stayed above the buffer in all 15. Policy C then cut orders deeply and lost about 10% of margin (`docs/tuning/02_gate_projection_and_tiebreak.txt`).
3. **Chosen: `replace_sales_capped`.** Later orders replace what was sold, but on each path they are scaled down to the cash then available above the buffer. A shop that runs this gate every week will not, next week, spend cash it does not have.

**Consequence worth knowing.** Buying stock that sells within the horizon raises cash (price
exceeds cost and revenue arrives before or with the bill). The gate therefore only restrains
units unlikely to sell soon. It also means the probability is not monotone in the budget.

### 5.4 Cash gate

Bisection on B with common random numbers, then a coarse grid of 8 points; the result is the
largest B that is actually feasible. Grid points whose feasibility contradicts the bisection are
counted as monotonicity violations and reported per decision, per run and in the README. If the
unconstrained plan is already safe, it is returned after one evaluation.

**When no budget meets the target, not even zero (changed from the brief, as instructed).**
The plan is flagged "cash at risk regardless of purchasing" and the gate returns the budget that
maximises P(min cash >= buffer) over B = 0, the grid, the full plan, and 6 more points around the
best of those, all on the same demand paths.

- Why not zero: `docs/tuning/04_infeasible_rule.txt`. On tuning windows the original rule ended with a 18% fill rate and 113 insolvency weeks out of 156, because a shop that stops buying stops selling.
- **Ties (my addition; the instruction does not cover them).** In the stressed scenarios the probability is often the same for every budget, usually zero, for example when a lump of fixed costs lands this week. Something has to decide. Probabilities within 0.05 of the best are treated as tied (`gate.max_prob_tol`; with 500 paths drawn from 52 past weeks a few points of probability are noise), and among tied budgets the one with the highest expected cash at the end of the horizon wins (`gate.tie_break: end_mean`). Alternatives tried on tuning windows: the bad-case (alpha-quantile) of minimum cash or of end cash, with and without the tolerance. They buy less and lose more margin for no fewer shortfall weeks (`02_gate_projection_and_tiebreak.txt`).
- How often this happens is reported in the README (gate diagnostics). In the stressed scenarios it is the common case, not the exception.

Deterministic fallback (`gate.mode: deterministic`): the same search on a single demand path at
the 10% quantile.

### 5.5 Allocator

- Marginal value of the (x+1)-th unit: `Cu * P(D > x) - Co * P(D <= x)`, with D the demand over the SKU's cover period, taken from the same joint sample paths the cash model uses.
- **Overage cost (OPEN, verified as the brief asks).** Non-perishables carry over, so `Co = (holding + capital rate) x cost x one review period`, about 0.7% of cost per week at default rates. Perishables: `Co = spoilage rate x (cost - salvage) + holding`, which matches what the simulated world does to leftover stock. `allocator.perishable_overage: full_loss` gives the single-period form. A test checks that non-perishable Co stays below 2% of cost.
- With Co this small the unconstrained target sits near the 99th percentile of cover demand, so C's full plan costs two to three times policy A's weekly spend. A capital charge of 100% a year (about a 95% service level) was tried on tuning windows and made no difference to the results, because the gate, not the target, decides what is bought (`03_alpha_capital_ranking.txt`). Left at 15%.
- **Ranking.** Units are ranked by marginal value per purchase dollar, as in the brief. Two alternatives were built and tried on tuning windows: per dollar of purchase cost plus expected unsold cost (`committed`, the capital-aware ranking from the retired C+), and expected cash profit per dollar (`cash`). Neither helped (`03_alpha_capital_ranking.txt`), so the default stays `cost`. Both remain available as `allocator.rank_by`.
- **MOQ / pack method (OPEN).** Documented heuristic: a SKU's first chunk is its MOQ, later chunks are single packs; a chunk's value is the sum of its units' marginal values; chunks are ranked by value per dollar and funded greedily; a repair pass funds later chunks that still fit. A chunk is only offered if its total value is positive. Tests: budget never exceeded, whole packs, MOQ respected, exact match with brute force when unit costs are equal, within one unit's value of the optimum otherwise.
- Decision labels: `full` = reached the unconstrained target, `partial`, `defer`. Cost of deferring = `Cu x (extra expected lost sales versus the full target)`, from the demand distribution.

### 5.6 Baseline policies

| Policy | Definition | Notes |
| --- | --- | --- |
| A | Order up to `sum of median forecasts over the cover period + z x sigma`, `z = 1.65`, sigma from the 10-90% interval assuming independence across weeks. Rounded up to packs and MOQ. Ignores cash. | `policies.reorder_point.z` |
| B | OTB at cost = planned sales + planned markdowns + planned end-of-period stock - beginning stock - on order. Planned sales = median forecast over the cover period; markdowns = 0; planned ending stock = target weeks of cover x planned weekly sales. **Target weeks of cover = `auto`**: the number that makes B's total planned ending stock equal policy A's total safety stock at cost, recomputed each week. **The budget is a cap**: it is split in proportion to policy A's need and never spends more than that need. | With a fixed week of cover the cap never bound on low-variance data and B was identical to A. `auto` makes A and B aim for the same total stock, so they differ only in how they get there. |
| D | OTB budget; SKUs funded in full in order of stockout probability x unit margin; the first that does not fit gets the remainder. | Meant as a stronger baseline. On real data it was the worst policy: funding some SKUs in full leaves others with nothing. |
| C_gate_prop | Cash gate + proportional split. | Ablation: gate without allocator. |
| OTB_marginal | OTB budget + marginal allocator. | Ablation: allocator without gate. |

---

## 6. Experiment design

| Item | Choice | Why | How to change |
| --- | --- | --- | --- |
| Windows | Four consecutive 26-week windows ending at the end of history. W1, W2 = `tune`; W3, W4 = `holdout`. | 26 decisions each; two windows never touched while tuning. | `experiments.windows` |
| **Headline scope** | Holdout windows only, the three stress levels at the default cash cushion, rule `overdraft`. Fixed before the held-out windows were simulated. | The brief forbids tuning on evaluation windows. | `experiments.headline` |
| Ideal replenishment cost R* | Mean weekly cost of goods sold (synthetic cost x real units) over the 52 weeks before the window. | What an order-up-to policy must spend per week to replace what sells. A full year, so fixed costs are not sized to one season. | `synthetic.calibration_weeks` |
| Fixed costs | On average 95% of pre-window gross margin. Paid as a base amount weekly plus a lump every 4th week (rent, monthly payroll). | Lumps are when small shops hit crunches. | `synthetic.fixed_cost_*` |
| **Budget stress s (OPEN; recalibrated on tuning windows as instructed)** | s tightens two things. (1) The lump is sized so that in a lump week `expected revenue - fixed costs = s x R*`. (2) Opening cash = `buffer + cushion x s x R*`. s = 50%, 70%, 90%; cushion 0.35. | The first definition only sized the lump and opened with 3 weeks of fixed costs. On tuning windows policy A then never ran short in W1, and in W2 it ran short more often at 90% than at 50%: total fixed costs are the same at every stress level, so a bigger lump also means cheaper ordinary weeks. Tying opening cash to s makes lower s tighter in both respects. The other reading of the brief, weekly fixed costs so high that inflows cover only s of replenishment every week, puts fixed costs above gross margin, where no purchasing policy can help. | `experiments.stress_levels`, `sim/backtest.py: calibrate_world` |
| **Cash cushion (OPEN)** | 0.35 by default; 0.2 and 0.75 as the opening-cash sweep. | Chosen from `docs/tuning/01_scenario_calibration_foods3.txt`: at 0.35, policy A is short in 19, 18 and 14 of 52 tuning weeks at stress 50%, 70% and 90%, and in at least 5 weeks of each tuning window at 50% and 70%. At 0.75 the 70% and 90% levels barely bind. | `experiments.cash_cushions`, `default_cash_cushion` |
| **alpha, buffer (OPEN)** | alpha = 0.10; buffer = 2 weeks of average fixed costs. | SUGGESTED. alpha of 0.05 and 0.20 were tried on tuning windows; neither was better on both shortfalls and margin (`03_alpha_capital_ranking.txt`). | `risk.*` |
| Scenario grid | Stress sweep at the default cushion, plus a cushion sweep at stress 70%, for every window and both shortfall rules: 40 scenarios. | Keeps the run short. | `sim/backtest.py: scenario_grid` |
| Uncertainty | Paired moving-block bootstrap: 4-week blocks resampled within each scenario, same blocks for every policy, 2000 resamples, 90% percentile intervals. | Suggested method. | `experiments.bootstrap` |
| Headline wording | The planned form is used only if C has fewer shortfall weeks than A, higher margin than B, both with intervals clear of zero, and no more shortfall weeks than B. Otherwise the generated sentence states what happened. Percentages and intervals are printed to one decimal. | DECIDED, tightened: a difference whose interval includes zero is never called a win. | `sim/metrics.py: headline_sentence` |
| **Primary and comparison slice** | Primary: FOODS_3. Comparison: HOUSEHOLD_1, run once with the same settings, nothing retuned. | On HOUSEHOLD_1 the held-out windows were comfortable (see the README comparison block), so the shortfall comparison there rests on a handful of weeks. Leading with it would be cherry-picking. On tuning windows the two slices behaved alike (`05_all_policies_overdraft.txt`, `06_household1_all_policies_overdraft.txt`). HOUSEHOLD_1 top sellers also turned out not to be slow: about 9 days of stock under policy A, the same as FOODS_3. | `results/LATEST`, `results/SECONDARY` |

---

## 7. Engineering

| Item | Choice |
| --- | --- |
| Python | 3.11 as specified. The build machine had only 3.14 and 3.9, so an isolated 3.11 was installed with `uv`. Versions in `requirements.txt` are what was installed and tested. |
| LightGBM on macOS | Needs `libomp`. `make` points the dyld fallback path at the copy bundled in the scikit-learn wheel, and `scripts/train_forecast.py` retries itself that way when run directly. Linux needs nothing extra. |
| Deployment requirements | `app/requirements.txt` lists only what the app imports (no LightGBM, no matplotlib). Streamlit Community Cloud uses the requirements file next to the app. |
| Results | Each run writes `results/<run_id>/manifest.json` (config hash, git commit, seed, timestamp, data slice, hashes of the raw files, versions). `make render` fills README and docs blocks from it and refuses fixture or partial runs. `results/LATEST` names the primary run, `results/SECONDARY` the comparison run. |
| Config inheritance | A config file may start with `extends: other.yaml`; `configs/household_1.yaml` overrides two keys of the default. |
| Tuning tools | `scripts/calibrate_scenarios.py` and `scripts/tune_sweep.py` read tuning windows only and never write to `results/`. |
| Screenshots | `make screenshots` starts the app, drives headless Chrome over the DevTools protocol and writes one full-page PNG per page. |
| Runtime | On an Apple M5 with 6 workers: forecasts about 6.5 minutes per slice, full backtest (40 scenarios x 6 policies x 26 weeks) about 40 seconds. Exact time is in each manifest. |
| LLM guardrail | Sent: display strings of the facts only. Every number in the output must equal a supplied number after the same rounding; spelled-out numbers are rejected; a failing line is replaced by its template. Tested with a fake client. |
| App bundle | `results/<run_id>/app/` stores calibrated quantiles only (no raw quantiles), about 6.5 MB for the largest file. `render_results.py` also writes `demo.json` there, naming the week the demo script uses; the app opens on it. That week is chosen by code: among weeks where the gate is binding, the one where policy A's order would carry the largest shortfall risk. It is an illustration, not a result. |
| Fresh-clone check | The repo was cloned to a temp folder with no `data/raw`, a new Python 3.11 venv was given only `app/requirements.txt`, and every app page was driven headlessly and the server booted. `uv pip compile` resolved both requirements files to Linux x86-64 wheels for Python 3.11. Nothing was run on an actual Linux machine. |
| Forecast caches | Produced once per slice before the gate settings were tuned and not regenerated afterwards; forecasting settings never changed, and the forecaster is seeded. |
| Git | Local commits only until `gh` is installed and authenticated on the build machine. |

---

## 8. Tuning log

Rule from the brief: if C does not clearly beat B in the first end-to-end run, diagnose before any
UI polish, tune only on `tune` windows, and write down what changed and why. Everything below was
decided on windows W1 and W2. The held-out windows W3 and W4 were simulated once, afterwards.

**Before real data (driven by the synthetic fixture, which only tests mechanics):**

1. Added a continuation assumption to the cash projection (5.3).
2. Fixed a chunk-ordering bug in the allocator: a floating-point tie could place a SKU's MOQ chunk after its later single-pack chunks. Found by the feasibility test.
3. Supplier terms shortened to pay-on-delivery and net-7, with a validator (4).
4. Open-to-buy target cover set to `auto` (5.6).
5. World calibration window lengthened from 13 to 52 weeks (6).

**On real data, tuning windows only (10 Oct 2026):**

6. **Policy C's infeasible case** changed to maximise P(safe), as instructed; C+ retired (5.4). Evidence: `04_infeasible_rule.txt`.
7. **Scenarios did not bind as first defined.** Policy A had no shortfall weeks in W1 at any stress level and the levels were not monotone. Redefined stress to tighten opening cash as well as the lump, and chose cushion 0.35 (6). Evidence: `01_scenario_calibration_foods3.txt`.
8. **First comparison after 6 and 7: C lost to B on both counts.** Roughly 35 shortfall weeks against B's 30 of 156, and about 6.5% less margin than B. Sweeping alpha, the capital charge and the ranking rule changed almost nothing. Diagnosis, from a week-by-week trace: the gate found no safe budget in about 60% of decisions; in that mode it was choosing very small budgets, sometimes a third of normal replenishment, to gain a few points of probability; the projection behind those probabilities was too pessimistic (5.3); and the lost sales cost more cash than the smaller orders saved.
9. **Changes made:** cash-aware continuation (5.3); ties and near-ties in the infeasible case decided by expected end-of-horizon cash (5.4). Evidence: `02_gate_projection_and_tiebreak.txt`.
10. **Tried and not adopted:** alpha 0.05 and 0.20; capital charge 100% a year; `committed` and `cash` ranking; a finer budget grid. Evidence: `03_alpha_capital_ranking.txt`.
11. **Where tuning ended:** on tuning windows C had about a third fewer shortfall weeks than A for under 3% less margin, and was level with B: marginally more margin, a few more shortfall weeks (`05_all_policies_overdraft.txt`). The gate-plus-proportional ablation kept more margin than C. C did not clearly beat B on tuning windows and was not tuned further to make it do so.
12. **Held-out windows simulated once**, with these settings, for both slices. Results are in the README.

Not tuned: the forecaster, the buffer, the baselines' parameters (`z`, OTB cover rule), the
shortfall rule used for the headline, and the headline scope.
