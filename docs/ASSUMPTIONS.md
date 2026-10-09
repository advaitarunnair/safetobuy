# Assumptions, open choices and deviations

Every OPEN item from the brief is resolved here with the value chosen, why, and how to change it.
SUGGESTED defaults that were changed are listed with the reason. `configs/default.yaml` and
`configs/experiments.yaml` are the source of truth for values; nothing in `src/` hard-codes them.

**Status of this file.** Written before any real-data run. The M5 files were not on the build
machine, so every choice below was made without seeing M5 results. Section 8 is the tuning log;
it must be filled in after the first real run, and choices marked *revisit on tuning windows* may
only be changed using windows whose role is `tune`.

---

## 1. Decisions the team needs to make

These are yours, not mine. Defaults are in place so nothing blocks.

| # | Decision | Current state | Where |
| --- | --- | --- | --- |
| 1 | **Project name** | Placeholder "Cash-Safe Purchasing Assistant". "StockWise" avoided as instructed. | `configs/default.yaml` -> `project.name` (one line; README title is filled by `make render`) |
| 2 | **Which policy you present: C as specified, or C+** | C follows the brief exactly. On synthetic rehearsal data it collapsed: once the gate says "cash at risk regardless of purchasing" it buys nothing, sales stop, and cash never recovers. `C_plus` fixes that and is reported next to C in every table. See 5.4. **This decides the headline.** | policy `C_plus`, `gate.on_infeasible` |
| 2b | **Capital charge in the overage cost** | Default 15% a year (plus 20% holding), which puts the unconstrained stock target near the 99th percentile of demand, well above policy A's 95%. C then spends cash up to its risk limit on units that add very little margin. A higher charge may suit a cash-constrained shop. *Tune on tuning windows:* `make sweep SET="--set synthetic.capital_rate_annual=0.15,0.5,1.0 --set risk.alpha=0.05,0.10"`. | `synthetic.capital_rate_annual`, `risk.alpha` |
| 3 | **Deadline timezone** | Unconfirmed in the brief. | n/a |
| 4 | **Publishing the results bundle** | `results/<run_id>/app/` holds weekly aggregates of a 300-SKU M5 slice so the app runs without raw files. Check the M5 competition terms before making the repo public; if in doubt, keep `results/*/app/` out of the public repo and ship run instructions only. | `.gitignore` |
| 5 | **LLM model and cost** | `claude-opus-5-5` at low effort. Optional: without `ANTHROPIC_API_KEY` the app uses templates. | `llm.model` |
| 6 | **Deployment and CSV upload** | Not included (both were cut-first). Run instructions are in the README. | n/a |

---

## 2. Data

| Item | Choice | Why | How to change |
| --- | --- | --- | --- |
| Slice | Store `CA_1`, department `FOODS_3`. M5 calls FOODS_3 a department (`dept_id`); its `cat_id` is FOODS. | SUGGESTED default. | `data.store_id`, `data.dept_id` |
| SKU count | Top 300 by units. Final count and share of department volume are printed by `make data` and stored in the run manifest. | SUGGESTED default. If runtime is a problem, reduce this first. | `data.n_skus` |
| **SKU selection window (deviation from SUGGESTED)** | The 52 weeks ending just before the first simulated decision, not the final year of history. | Selecting on the final year uses sales from inside the backtest to choose which SKUs to test on, a mild survivorship bias. Selecting beforehand means some SKUs may fade or be delisted during the backtest, as in real life. | `data.selection.mode: final_year` restores the SUGGESTED rule |
| Weekly aggregation | Walmart weeks (`wm_yr_wk`, Saturday to Friday). Only weeks with 7 days of sales are kept; the last 2 days of `sales_train_evaluation.csv` form a partial week and are dropped. | Decisions are weekly. | `data/load_m5.py` |
| Prices | Real M5 `sell_prices.csv`, forward-filled per SKU. Weeks before a SKU's first listing have no price (NaN) and zero demand. | No backfill, so no price from the future leaks into features. | n/a |
| Missing data | `make data` stops with exact Kaggle instructions. Nothing is fabricated. | DECIDED. | n/a |
| Test fixture | `tests/fixtures/synthetic_m5_fixture.py` writes M5-shaped made-up CSVs with a marker file. Everything downstream is labelled `SYNTHETIC_TEST_FIXTURE`, the app shows a red banner, and `render_results.py` refuses to publish such a run. | Lets the whole pipeline be tested without M5. | n/a |

Known properties of M5 to keep in mind (also in README Limitations): sales are censored by
Walmart's own stockouts, and many SKU-weeks are zero (intermittent demand).

---

## 3. Timing, horizons and the simulated world

| Item | Choice | Why | How to change |
| --- | --- | --- | --- |
| Review period | 1 week. | SUGGESTED. | `horizons.review_period_weeks` (only 1 is supported) |
| Forecast horizon | 4 weeks. Covers max lead time (2) + review period (1); config validation enforces this. | SUGGESTED. | `horizons.forecast_horizon_weeks` |
| Cash horizon | 4 weeks. | SUGGESTED. | `horizons.cash_horizon_weeks` |
| Decision timing | The decision for week t is made before week t's demand is known, using data up to week t-1. | Matches `decide(state, info)` then `step(orders, realized_demand)`. | n/a |
| Step order | Receive arrivals; realise sales (min of demand and stock, lost sales not backordered); write off spoiled perishables; receive revenue; pay due payables and fixed costs; place new orders. | DECIDED order, plus the spoilage step that the perishability flag requires (see 4). | `sim/world.py` |
| Lead time meaning | An order placed in week t arrives at the start of week t + L and can be sold that week. The cover period of an order is therefore L + 1 weeks (forecast horizons 1..L+1). | Standard periodic review. | n/a |
| Payment timing | The payable is created at order placement and falls due `pay_delay` weeks after delivery. | Weekly granularity. | `synthetic.suppliers` |
| **Cash-shortfall rule (OPEN)** | Default `overdraft`: orders are never cancelled, cash may go negative; a week with cash below the buffer is a shortfall week, below zero an insolvency week. Alternative `cut_proportional`: at placement, if the order's cost exceeds cash on hand, every line is scaled by the same factor (rounded down to packs, lines under MOQ dropped). Both rules are run and the headline is reported under both. | SUGGESTED default plus the suggested alternative. | `world.shortfall_rule`, `experiments.shortfall_rules` |
| Realised demand | Actual M5 weekly units. Policies see only an `InfoSet` cut off at week t-1. | DECIDED. | n/a |
| Opening state | Policy A runs for 8 burn-in weeks before each window with ample cash. Every policy then starts from that same stock, pipeline and payables; cash is reset to the scenario's opening cash. | A shop mid-operation owes suppliers and has stock in transit. Starting from zero payables would hand every policy two weeks of free revenue. | `synthetic.burn_in_weeks` |

---

## 4. Synthetic parameters (all values in `configs/default.yaml -> synthetic`, seeded)

The README table of these values is generated from the config by `make render`. Only demand and
shelf prices are real.

| Item | Choice | Why |
| --- | --- | --- |
| Unit cost | `reference price x (1 - margin)`, margin drawn per SKU from the configured range. Reference price = median M5 price in the selection window. **Cost is fixed over time**; it does not follow promotional price drops. | Promotions cut margin in real life. A cost that tracked price would make every promotion free. |
| Lead times | 1 or 2 weeks, drawn per SKU. | SUGGESTED. |
| Pack sizes / MOQ | Pack drawn from the configured list but never more than half a SKU's mean weekly units; MOQ is 1 or 2 packs. | Avoids a 24-pack for a SKU that sells 3 a week. |
| Payment terms | Two suppliers: pay on delivery (60% of SKUs) and net-7 (40%). **Net-14 and net-30 were dropped.** Every bill for this week's order must fall inside the 4-week cash horizon, and the config validator enforces `max lead time + max payment delay < cash horizon`. | With a 4-week horizon, a net-30 bill for goods ordered today lands in week 5 or 6, outside the projection, so buying on those terms looked free to the gate. Longer terms need a longer forecast and cash horizon (7 weeks for net-30). `gate.charge_beyond_horizon: true` allows them by counting late bills in the last projected week, but that double-counts against older bills and is pessimistic. Supplier-specific terms were first on the cut list. |
| Receivables | Not modelled: sales are cash. | Retail; cut-first item. |
| **Perishability** | A share of SKUs is flagged perishable. In the simulated world a fixed share of their leftover stock is written off each week at salvage value. | The brief asks for a perishability flag. A flag that changes the allocator's costs but not the world would bias the comparison, so the world implements it. This adds a write-off term to the inventory identity (`change = arrivals - sales - write-offs`); for non-perishables the identity is exactly the one in the brief. Tests cover both. |
| Holding and capital cost | Annual rates on unit cost. **Decision parameters only**: they enter the allocator's overage cost and are not charged as cash flows (storage is part of fixed costs). | Keeps the cash recursion exactly as specified. |
| Seeds | One global seed. Each SKU's parameters come from a stream keyed on (seed, SKU id), so they do not change when the slice changes. Demand paths for week t are seeded with (seed, t), so every policy and every gate evaluation sees the same paths. | Reproducibility and common random numbers. |

---

## 5. Method choices

### 5.1 Forecasting

| Item | Choice | How to change |
| --- | --- | --- |
| Model | LightGBM, quantile objective, one model per quantile (7), pooled across SKUs and horizons (direct multi-horizon; `h` is a feature). | `forecast.lgbm` |
| Scaling | Target and lag features are divided by the SKU's trailing 13-week mean + 1, so one model serves SKUs of very different volume. Predictions are scaled back. | `data/features.py` |
| Features | Lags 0-3, rolling means (4/8/26/52) and stds, zero share, trend, price level and relative price, listing age, same-week-last-year, and target-week calendar (week of year, month, SNAP days, event counts by type, Christmas). | `data/features.py` |
| Future-dated inputs | Only the calendar (events, SNAP schedule) of the target week, which is published in advance exactly as in the M5 competition. Sales and prices after the cutoff are never read; the no-lookahead tests replace them with noise and require byte-identical features, forecasts and decisions. | n/a |
| Non-crossing | Quantiles are clipped at zero and sorted. | n/a |
| **Refit cadence (OPEN)** | Refit every 4 weeks, predict every week with fresh features. Logged in the manifest. | `forecast.refit_every_weeks` |
| Training window | Last 156 origin weeks. | `forecast.train_window_weeks` |
| Baselines | Naive (last week) and seasonal naive (same week last year), made into quantile forecasters by adding empirical quantiles of their own past errors. | `forecast/baselines.py` |
| **Conformal calibration** | Rolling, per quantile: for each level and horizon, shift the quantile by the matching empirical quantile of normalised past errors `(y - q) / max(q75 - q25, 1)` from forecasts issued in the previous 26 weeks and already realised. Pooled across SKUs. Coverage before and after is in the forecast table. | `forecast.conformal` |
| Warm-up | 30 weeks of forecasts are issued before the first decision so calibration and the residual bank exist from week one. | `forecast.warmup_weeks` |
| Pretrained benchmark (Chronos) | Skipped (cut-first). | n/a |

### 5.2 Joint demand sampling (OPEN: correlation method)

Weekly residual-block bootstrap on the PIT scale, as suggested. For each past forecast origin
whose targets are realised, store the PIT of realised demand under that origin's forecast, for
every SKU and horizon. A path picks a past origin and reuses its whole cross-section (all SKUs
together), so cross-SKU dependence is whatever happened. Horizons are drawn in blocks of 2
consecutive weeks from one origin, which keeps some persistence of forecast errors.

- Bank size: last 52 origins. Path count: 500 in the backtest, 2000 in the app (OPEN, SUGGESTED values).
- Tails: beyond the 95% quantile an exponential tail fitted to the q90-q95 gap; below 5% linear to zero. Capped at the 99.9% level.
- Flat regions (several quantiles equal, common with zeros) use the mid-point PIT.
- Sampled demand is rounded to whole units.
- Assumption: dependence in the next four weeks resembles dependence in forecast errors over the past year. With 52 origins the joint distribution is coarse.
- `sampling.method: independent` shuffles origins per SKU (same marginals, no correlation). The app shows both probabilities for the same plan.

### 5.3 Cash projection

`cash_t = cash_{t-1} + revenue_t - fixed_costs_t - supplier_payments_t` on every path, in the
same order as the world, with sales capped by stock.

**Continuation assumption (not in the brief, found necessary).** The projection needs to assume
something about orders placed in later weeks of the horizon. With no later orders, stock runs out
inside four weeks, revenue collapses while fixed costs continue, and nearly every plan looks
unsafe: on the test fixture the gate flagged "cash at risk regardless" in 38 of 64 decisions.
Default is therefore `gate.continuation: replace_sales`: in each later week the shop orders what
it sold the week before, with normal lead times and terms. Those future orders are still
re-decided and re-gated when their week arrives. `none` is kept for comparison.

Check done on rehearsal data: for policy A's first order in a window, the projected median cash
over the next four weeks tracked what then happened to A in the simulator, and the realised path
stayed inside the 10%-90% band.

**Consequence worth knowing.** With this projection, buying stock that sells within the horizon
raises cash (price exceeds cost and revenue arrives before or with the bill). The gate therefore
only restrains the units unlikely to sell soon: deep safety stock. How much cash that can free
depends on how much stock the shop holds. Fast-moving grocery turns stock in days, so the lever is
small; see Limitations in the README.

### 5.4 Cash gate

Bisection on B with common random numbers, then a coarse grid of 8 points; the result is the
largest B that is actually feasible. Grid points whose feasibility contradicts the bisection are
counted as monotonicity violations and reported per decision, per run and in the README.
Non-monotonicity is real here: buying units that sell before their bill is due raises cash.
If the unconstrained plan is already safe, it is returned after one evaluation.

**When even B = 0 is infeasible.** The brief says: return 0 and flag. That is the default
(`gate.on_infeasible: zero`) and that is policy C. On the test fixture this rule produced a
death spiral whenever the shop started at its buffer: no purchases, so no revenue, so cash never
recovers (fill rate 28%, insolvency). I did not change the specified behaviour. Instead:

- `C_plus` differs from C in two ways, both found while verifying the specified design:
  1. *Fallback.* `on_infeasible: best_effort`: among candidate budgets take the one with the highest P(safe), ties broken by the alpha-quantile of end-of-horizon cash. It still raises the flag.
  2. *Ranking.* Units are ranked by expected profit per dollar of capital committed (purchase cost plus the cost of units expected to be left unsold), see 5.5.
- Both policies are in every results table. The headline is computed for C as specified; the same headline with C replaced by C_plus is in the robustness table and in "C against B, stated plainly".

Rehearsal evidence (M5-shaped **synthetic** data at full scale, 300 SKUs, so mechanics only):
policy C flagged "cash at risk" in most decisions and ended with a fill rate near 45% and negative
cash; C_plus stayed within a few points of policies A and B on both shortfall weeks and margin.
Expect the same qualitative behaviour from C on real data whenever cash gets near the buffer.

**Team decision 2:** decide which of the two you present as the product. If C_plus is clearly
better on the tuning windows, saying so plainly is stronger than a headline that hides the issue.
If you present C+, set `experiments.yaml -> app_bundle.policy: C_plus` before `make backtest`, so
the app replays C+'s weeks and uses its rules by default. The headline sentence itself is always
computed for C as specified; the C+ version is in the robustness table.

Deterministic fallback (`gate.mode: deterministic`): the same search on a single demand path at
the 10% quantile.

### 5.5 Allocator

- Marginal value of the (x+1)-th unit: `Cu * P(D > x) - Co * P(D <= x)`, with D the demand over the SKU's cover period, taken from the same joint sample paths the cash model uses.
- **Overage cost (OPEN, verified as the brief asks).** Non-perishables carry over, so `Co = (holding + capital rate) x cost x one review period`, about 0.7% of cost per week at default rates. Perishables: `Co = spoilage rate x (cost - salvage) + holding`, which matches what the simulated world actually does to leftover stock. `allocator.perishable_overage: full_loss` gives the single-period form `cost - salvage + holding`. A test checks that non-perishable Co stays below 2% of cost.
- With Co this small the unconstrained target sits near the 99th percentile of cover demand, well above policy A's 95%. Unconstrained, C therefore holds more stock than A; the gate is what limits it.
- **Ranking (verified, as the brief asks).** The brief ranks units by marginal value divided by unit cost. That is the exact solution of the single-period budget-constrained newsvendor, and it is what C, `OTB_marginal` and `C_gate_prop` use. Its weakness under a budget that binds week after week: a unit that does not sell keeps its dollar tied up for another period, and the ranking ignores that. With Co small it prefers a high-margin unit with a 50% chance of selling over a low-margin unit that is certain to sell. Charging leftovers the shadow price of cash gives a closed form: fund a chunk iff `value / (cost + expected leftover cost) >= lambda`. So `rank_by="committed"` ranks by value per dollar of purchase cost plus expected unsold cost. It changes only the order of funding; the unconstrained plan is identical. Used by `C_plus` only. On rehearsal data the two rankings gave almost the same plans at moderate budgets; the difference grows as the budget tightens.
- **MOQ / pack method (OPEN).** Documented heuristic: a SKU's first chunk is its MOQ, later chunks are single packs; a chunk's value is the sum of its units' marginal values; chunks are ranked by value per dollar and funded greedily; a repair pass funds later chunks that still fit. A chunk is only offered if its total value is positive. Tests: budget never exceeded, whole packs, MOQ respected, exact match with brute force when unit costs are equal, within one unit's value of the optimum otherwise (it is a knapsack; greedy is optimal for the continuous relaxation only).
- Decision labels: `full` = reached the unconstrained target, `partial`, `defer`. Cost of deferring = `Cu x (extra expected lost sales versus the full target)`, from the demand distribution.

### 5.6 Baseline policies

| Policy | Definition | Notes |
| --- | --- | --- |
| A | Order up to `sum of median forecasts over the cover period + z x sigma`, `z = 1.65`, sigma from the 10-90% interval assuming independence across weeks. Rounded up to packs and MOQ. Ignores cash. | `policies.reorder_point.z` |
| B | OTB at cost = planned sales + planned markdowns + planned end-of-period stock - beginning stock - on order. Planned sales = median forecast over the cover period; markdowns = 0 (no markdown data); planned ending stock = target weeks of cover x planned weekly sales. **Target weeks of cover = `auto`**: the number that makes B's total planned ending stock equal policy A's total safety stock at cost, recomputed each week. **The budget is a cap**: it is split in proportion to policy A's need and never spends more than that need. | `policies.otb.target_weeks_cover` (a number overrides `auto`). With a fixed 1.0 week the cap never bound on low-variance data and B was identical to A. `auto` makes A and B aim for the same total stock, so they differ only in how they get there: A SKU by SKU, B as one budget in which an overstocked SKU reduces what is left for the others. |
| D | OTB budget; SKUs funded in full in order of stockout probability x unit margin; the first that does not fit gets the remainder. | Stronger baseline than B. |
| C_gate_prop | Cash gate + proportional split. | Ablation: gate without allocator. |
| OTB_marginal | OTB budget + marginal allocator. | Ablation: allocator without gate. |

---

## 6. Experiment design

| Item | Choice | Why | How to change |
| --- | --- | --- | --- |
| Windows | Four consecutive 26-week windows ending at the end of history. W1, W2 = `tune`; W3, W4 = `holdout`. | 26 decisions each; two windows never touched while tuning. | `experiments.windows` |
| **Headline scope** | Holdout windows only, the three stress levels at the default opening cash, default shortfall rule. | The brief forbids tuning on evaluation windows. Tuning windows are reported separately. | `experiments.headline` |
| Ideal replenishment cost R* | Mean weekly cost of goods sold (synthetic cost x real units) over the 52 weeks before the window. | What an order-up-to policy must spend per week to replace what sells. A full year, so the shop's fixed costs are not sized to one season. With 13 weeks, a window that followed a seasonal peak started structurally loss-making and no purchasing policy could help. | `synthetic.calibration_weeks` |
| Fixed costs | On average 95% of pre-window gross margin (a thin net margin). Paid as a base amount weekly plus a lump every 4th week (rent, monthly payroll). | Lumps are when small shops hit crunches. | `synthetic.fixed_cost_*` |
| **Budget stress s (OPEN)** | The lump is sized so that in a lump week, `expected revenue - fixed costs = s x R*`. s = 50%, 70%, 90%. Total fixed costs over a cycle are the same at every stress level; only their lumpiness changes. | Precise, and survivable. The other reading, weekly fixed costs so high that inflows cover only s of replenishment every week, means fixed costs above gross margin: no purchasing policy can save that shop, so it tests nothing. | `experiments.stress_levels` |
| **Opening cash (OPEN)** | 2.5, 3 or 4 weeks of average fixed costs; default 3. | *Revisit on tuning windows.* | `experiments.start_cash_weeks` |
| **alpha, buffer (OPEN)** | alpha = 0.10; buffer = 2 weeks of average fixed costs. | SUGGESTED. Note: a gate at alpha = 0.10 accepts up to a 10% chance of a shortfall at each decision. Where A is never short, C can show more shortfall weeks than A while staying inside its stated tolerance. | `risk.*` |
| Scenario grid | Stress sweep at default opening cash, plus an opening-cash sweep at default stress, for every window and both shortfall rules. | Keeps the run short. | `sim/backtest.py: scenario_grid` |
| Uncertainty | Paired moving-block bootstrap: 4-week blocks resampled within each scenario, same blocks for every policy, 2000 resamples, 90% percentile intervals. | Suggested method. | `experiments.bootstrap` |
| Headline wording | The planned form is used only if C has fewer shortfall weeks than A, higher margin than B, and no more shortfall weeks than B. Otherwise the generated sentence states what happened. | DECIDED. | `sim/metrics.py: headline_sentence` |

---

## 7. Engineering

| Item | Choice |
| --- | --- |
| Python | 3.11 as specified. The build machine had only 3.14 and 3.9, so an isolated 3.11 was installed with `uv`. Versions in `requirements.txt` are what was installed and tested. |
| LightGBM on macOS | Needs `libomp`. `make` points the dyld fallback path at the copy bundled in the scikit-learn wheel, and `scripts/train_forecast.py` retries itself that way when run directly. `brew install libomp` also works. |
| Results | Each run writes `results/<run_id>/manifest.json` (config hash, git commit, seed, timestamp, data slice, file hashes of the raw data, versions). `make render` fills README and docs blocks from it and refuses fixture or partial runs. |
| Git | I initialised a local repository and committed, because the manifest records the commit. Nothing was pushed. |
| LLM | Anthropic API, `claude-opus-5-5`, low effort, JSON schema output, server-side refusal fallback. Sent: display strings of the facts only. Guardrail: every number in the output must equal a supplied number after the same rounding; spelled-out numbers are rejected; a failing line is replaced by its template. Untested against the live API because no key was available; tested with a fake client. |
| Runtime | Rehearsal on M5-shaped synthetic data at full scale (300 SKUs, 277 weeks, default settings, Apple M5, 6 workers): data 1 s, forecasts 5.5 min (36 LightGBM refits x 7 quantiles), full backtest 40 s (40 scenarios x 7 policies x 26 weeks), tuning sweep about 1 s per scenario. The real run's time is in the manifest (`runtime_seconds`). |
| Tuning tool | `scripts/tune_sweep.py` (`make sweep SET="--set key=v1,v2"`) runs any grid of config values on the tuning windows only and prints shortfall weeks, fill rate and margin against A and B. It never reads holdout windows and never writes to `results/`. |
| Screenshots | `make screenshots` starts the app, drives headless Chrome over the DevTools protocol and writes one full-page PNG per page to `docs/screenshots/`. Tested on the fixture bundle. `QUERY="stress=70&week=5"` pins the plan page to a week. |

---

## 8. Tuning log

Rule: if C does not clearly beat B in the first end-to-end run, diagnose before any UI polish,
tune only on `tune` windows, and write down what changed and why.

**Before real data (changes driven by the synthetic fixture, which only tests mechanics):**

1. Added the `replace_sales` continuation to the cash projection (5.3). Without it the gate was structurally pessimistic.
2. Fixed a chunk-ordering bug in the allocator: a floating-point tie could place a SKU's MOQ chunk after its later single-pack chunks, funding less than the MOQ. Found by the feasibility test.
3. Added policy `C_plus` and left C as specified (5.4).
4. Default opening cash moved from 2.0 to 3.0 weeks of fixed costs. At 2.0 the shop starts exactly at its buffer and policy C as specified can never buy.
5. Supplier terms shortened to pay-on-delivery and net-7, with a validator, after finding that net-30 bills fell outside the 4-week horizon (4). A first attempt, charging late bills in the last projected week, was reverted: it double-counted against bills already in the pipeline and made the gate reject plans that were in fact safe.
6. Open-to-buy target cover set to `auto` (5.6) because with a fixed week of cover the cap never bound and B was identical to A.
7. World calibration window lengthened from 13 to 52 weeks (6).
8. Added capital-aware ranking to `C_plus` (5.5).

None of these used M5 data. Items 1, 2, 5, 6 and 7 are corrections to the experiment or the code
and apply to every policy. Items 3 and 8 only affect `C_plus`.

**Suggested order for the first real run:**

1. `make data && make forecast`, then `make tune`. Look at A, B, C, C_plus on the tuning windows only.
2. If A and B have almost no shortfall weeks, the scenarios are too comfortable to test anything: lower `experiments.start_cash_weeks` / `default_start_cash_weeks` and re-run `make tune`.
3. `make sweep` over `risk.alpha` and `synthetic.capital_rate_annual` (decision 2b). Pick values on tuning windows.
4. Decide C versus C_plus (decision 2).
5. Only then `make backtest` (this is the first time holdout windows are simulated), `make figures`, `make render`, `make screenshots`.
6. Write each change and its reason below.

**After the first real run:** _not done yet. Record here: date, run id, what C vs B looked like on the tuning windows, each change, and the reason._
