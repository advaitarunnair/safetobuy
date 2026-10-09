# Project description

**<!-- BEGIN:project_name -->SafeToBuy<!-- END:project_name -->**
ForgeHacks 2026, "AI for Real World Problems", track AI + Business.

> Blocks between `BEGIN` and `END` markers are written by `scripts/render_results.py` from the
> run manifest. Do not type numbers into them.

## One line

Open-to-buy gives you a budget. We tell you whether you can afford it, and how to spend it best.

## The problem and who has it

Small retailers fail from cash crunches, not only from slow sales. A minimart owner reorders
every week across hundreds of products. Inventory apps tell them how much of each product to buy,
one product at a time. Nothing tells them whether the total is affordable once rent, payroll and
the supplier bills already on their way are counted, or what to cut when it is not.

The users are small retailers, convenience stores and independent e-commerce sellers: businesses
with real purchasing decisions and no planning department.

## What we built

A weekly purchasing assistant with five parts.

1. **Demand forecast with calibrated uncertainty (machine learning).** A LightGBM quantile model predicts each product's weekly sales as a range, not a single number. Conformal calibration then corrects those ranges using the model's own recent misses, so a "90% interval" contains the truth about 90% of the time.
2. **Cash projection (simulation).** Hundreds of joint demand scenarios are run through the shop's stock, incoming deliveries, supplier bills and fixed costs for the next four weeks. Products are sampled together, because a slow week is slow for most of the shelf at once.
3. **Cash gate (optimization).** The largest purchasing budget for which the chance of cash falling below the owner's safety buffer stays within the risk they chose. When no budget is that safe, it takes the one with the best chance and says so.
4. **Allocation (optimization).** That budget goes to the units with the highest expected profit per dollar, in whole cases and respecting minimum order quantities.
5. **Explanation.** Every product gets one line: bought in full, bought in part, or deferred, with the chance of running out and the expected margin at stake. An LLM may reword these lines, but code checks every number it writes against the computed facts and falls back to a fixed template on any mismatch.

**ML is used for prediction. Simulation and optimization make the decision.** We do not describe
the whole system as machine learning, because it is not.

## How we tested it

We built a simulated shop and let different purchasing policies run it week by week, on real
demand: actual weekly sales of 300 grocery products from the M5 (Walmart) dataset. Every policy
faces the same demand, the same costs and the same starting position.

- **A, reorder point:** the standard per-product rule. Ignores cash.
- **B, open-to-buy:** a budget from the standard retail formula, split across products in proportion to need.
- **D, priority cuts:** the open-to-buy budget, spent on the most urgent products first.
- **C, ours:** cash gate plus allocation. Two ablations isolate what each half contributes.

Settings were chosen on two 26-week windows and scored once on two later windows that were never
used for tuning. Costs, lead times, supplier terms and cash are synthetic, because M5 has none of
them, so dollar figures are illustrative. The comparison between policies in the same world is
the result.

## Result

<!-- BEGIN:headline -->
> **Policy C had 27.9% fewer cash-shortfall weeks than policy A (62 vs 86) (90% bootstrap interval for the reduction: +18.4% to +42.4%), and 1.0% higher margin than policy B (90% bootstrap interval for the difference: +0.6% to +1.7%), but more shortfall weeks than B (62 vs 58). Against policy A, C's margin was 2.9% lower. The planned headline form is not supported by this run, so the result is stated as it happened.**
>
> Scope: holdout windows, stress sweep, overdraft; 6 scenarios, 156 simulated weeks. Run `20261010-010653_f9b8c73f_foods3`.
<!-- END:headline -->

<!-- BEGIN:verdict -->
**Headline scope** (holdout windows, the three stress levels, rule `overdraft`, 156 simulated weeks per policy):

- **Fewer cash-shortfall weeks than A: supported.** C 62 vs A 86 of 156 weeks, 27.9% fewer (90% interval +18.4% to +42.4%).
- **Higher margin than B: supported.** 1.0% higher (90% interval +0.6% to +1.7%).
- **No more shortfall weeks than B: does not hold.** C 62 vs B 58 (C had no more than B in only 11% of bootstrap resamples).
- **Cost against A:** C's gross margin was 2.9% lower than A's.
- **Weeks with cash below zero:** A 0, B 2, C 0.
- **Planned headline form:** not supported. It needs fewer shortfall weeks than A, higher margin than B, both with intervals clear of zero, and no more shortfall weeks than B.

**Same windows under the alternative cash rule `cut_proportional`:**

- **Fewer cash-shortfall weeks than A: not supported.** C 65 vs A 68 of 156 weeks, 4.4% fewer (90% interval -9.0% to +24.2%; the interval includes zero).
- **Higher margin than B: supported.** 4.2% higher (90% interval +2.0% to +8.9%).
- **No more shortfall weeks than B: does not hold.** C 65 vs B 60 (C had no more than B in only 3% of bootstrap resamples).
- **Cost against A:** C's gross margin was 6.4% lower than A's.
- **Weeks with cash below zero:** A 16, B 23, C 18.

**It is not uniform across the held-out data.** window W3: shortfall weeks A 64 / B 42 / C 44, margin vs B +2.9% (90% interval +2.2% to +3.9%), planned form does not hold; window W4: shortfall weeks A 22 / B 16 / C 18, margin vs B -0.9% (90% interval -1.4% to -0.2%), planned form does not hold; stress 50%: shortfall weeks A 33 / B 22 / C 24, margin vs B +0.8% (90% interval +0.1% to +2.0%), planned form does not hold; stress 70%: shortfall weeks A 29 / B 20 / C 20, margin vs B +1.1% (90% interval +0.4% to +2.3%), planned form holds; stress 90%: shortfall weeks A 24 / B 16 / C 18, margin vs B +1.1% (90% interval +0.3% to +2.2%), planned form does not hold. These cuts were not chosen in advance and none of them is the headline.

**Tune windows (used to choose settings, so optimistic by construction):** shortfall weeks A 51 / B 30 / C 34, margin vs B +0.5% (90% interval -0.0% to +0.7%), vs A -2.7%.
<!-- END:verdict -->

A second slice, run with the same settings:

<!-- BEGIN:secondary_slice -->
**Store CA_1, department HOUSEHOLD_1** (300 of 532 SKUs), same settings as the primary slice with nothing retuned. Run `20261010-010733_e25dab98_household1`, config hash `e25dab98faa9`.

- **Fewer cash-shortfall weeks than A: not informative.** A ran short in only 3 of 156 weeks (C: 0), too few to support a claim.
- **Higher margin than B: supported.** 1.4% higher (90% interval +1.3% to +1.7%).
- **No more shortfall weeks than B: holds.** C 0 vs B 0 (C had no more than B in 100% of bootstrap resamples).
- **Cost against A:** C's gross margin was 0.1% higher than A's.
- **Weeks with cash below zero:** A 0, B 0, C 0.

On this slice the held-out windows were comfortable: policy A ran short in only 3 of 156 weeks, so it says little about cash shortfalls. What it does show is the margin comparison. The planned headline form holds formally here, but only on that handful of weeks, so we do not lead with it.

| Policy | Shortfall weeks | Insolvency weeks | Fill rate | Gross margin | Lost margin (stockouts) | Ending net position (mean) | Days of inventory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: reorder point | 3 / 156 | 0 | 97.9% | $684,624 | $14,705 | $26,843 | 8.7 |
| B: open-to-buy, proportional split | 0 / 156 | 0 | 95.2% | $675,452 | $30,883 | $25,315 | 6.8 |
| D: open-to-buy, priority cuts | 24 / 156 | 0 | 78.8% | $647,923 | $53,859 | $20,727 | 8.5 |
| **C: cash gate + marginal allocator (proposed)** | 0 / 156 | 0 | 96.9% | $685,213 | $21,997 | $26,942 | 10.5 |
| ablation: cash gate + proportional split | 0 / 156 | 0 | 97.2% | $681,751 | $18,645 | $26,365 | 8.4 |
| ablation: OTB budget + marginal allocator | 0 / 156 | 0 | 96.0% | $679,867 | $29,140 | $26,051 | 7.5 |
<!-- END:secondary_slice -->

Forecast quality on the primary slice:

<!-- BEGIN:forecast_table -->
| Forecaster | wQL (lower is better) | Mean pinball (units) | 50% interval coverage | 80% interval coverage | 90% interval coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM quantile (raw) | 0.201 | 3.75 | 47.2% | 76.2% | 87.2% |
| LightGBM quantile + conformal | 0.200 | 3.73 | 51.3% | 80.9% | 90.8% |
| Seasonal naive | 0.354 | 6.59 | 46.2% | 72.1% | 82.5% |
| Naive | 0.239 | 4.44 | 53.5% | 79.3% | 88.3% |

Rolling-origin over the backtest decision weeks, horizons 1 to 4, 123,000 SKU-week-horizon cells per model. LightGBM refit every 4 weeks, predicted weekly.
<!-- END:forecast_table -->

## Real-world impact

- **It answers the question owners actually ask:** not "how much of product X", but "can I afford this week's order, and if not, what do I drop?"
- **It makes the trade-off visible.** Each deferral comes with its cost, so the owner decides with numbers instead of instinct. Our own results show that trade-off is real: protecting the cash buffer costs margin.
- **It is honest about risk.** The owner sets the buffer and the tolerance; the tool reports the chance of a shortfall instead of a single optimistic cash line, and says plainly when no purchase plan is safe.
- **It needs little data:** sales history, prices, supplier terms and a bank balance.

## What it is not

- Not calibrated to Singapore, or to any real shop's costs. The minimart persona is framing.
- Not a guarantee. A 10% risk tolerance means shortfalls will sometimes happen.
- Not a clean win over every baseline. See the result above.
- Not a claim that no other tool allocates a budget. We reviewed Shopify inventory apps (which do per-product forecasting and reorder quantities) and open-to-buy tools (which set a budget from a deterministic formula), not enterprise planning software. Our difference is deriving the budget from a probabilistic cash-risk constraint and allocating it by marginal profit per dollar with explained deferrals.

Full limitations are in the README.
