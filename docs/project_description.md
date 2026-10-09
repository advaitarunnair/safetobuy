# Project description (draft)

**<!-- BEGIN:project_name -->Cash-Safe Purchasing Assistant<!-- END:project_name -->** (working title)
ForgeHacks, "AI for Real World Problems", track AI + Business.

> Draft for the team to edit. Blocks between `BEGIN` and `END` markers are written by
> `scripts/render_results.py` from the run manifest. Do not type numbers into them.

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
2. **Cash projection (simulation).** Thousands of joint demand scenarios are run through the shop's stock, incoming deliveries, supplier bills and fixed costs for the next four weeks. Products are sampled together, because a slow week is slow for most of the shelf at once.
3. **Cash gate (optimization).** The largest purchasing budget for which the chance of cash falling below the owner's safety buffer stays within the risk they chose.
4. **Allocation (optimization).** That budget goes to the units with the highest expected profit per dollar, in whole cases and respecting minimum order quantities.
5. **Explanation.** Every product gets one line: bought in full, bought in part, or deferred, with the chance of running out and the expected margin at stake. An LLM may reword these lines, but code checks every number it writes against the computed facts and falls back to a fixed template on any mismatch.

**ML is used for prediction. Simulation and optimization make the decision.** We do not describe
the whole system as machine learning, because it is not.

## How we tested it

We built a simulated shop and let different purchasing policies run it week by week, on real
demand: actual weekly sales of grocery products from the M5 (Walmart) dataset. Every policy faces
the same demand, the same costs and the same starting position.

- **A, reorder point:** the standard per-product rule. Ignores cash.
- **B, open-to-buy:** a budget from the standard retail formula, split across products in proportion to need.
- **D, priority cuts:** the open-to-buy budget, spent on the most urgent products first.
- **C, ours:** cash gate plus allocation. Two ablations isolate what each half contributes.

Policies were tuned on one set of 26-week windows and scored on others that were never used for
tuning. Costs, lead times, supplier terms and cash are synthetic, because M5 has none of them, so
dollar figures are illustrative. The comparison between policies in the same world is the result.

## Result

<!-- BEGIN:headline -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:headline -->

How policy C compared with open-to-buy, under both cash-shortfall rules and on the tuning windows:

<!-- BEGIN:c_vs_b -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:c_vs_b -->

## Real-world impact

- **It answers the question owners actually ask:** not "how much of product X", but "can I afford this week's order, and if not, what do I drop?"
- **It makes the trade-off visible.** Each deferral comes with its cost, so the owner decides with numbers instead of instinct.
- **It is honest about risk.** The owner sets the buffer and the tolerance; the tool reports the chance of a shortfall instead of a single optimistic cash line.
- **It needs little data:** sales history, prices, supplier terms and a bank balance.

## What it is not

- Not calibrated to Singapore, or to any real shop's costs. The minimart persona is framing.
- Not a guarantee. A 10% risk tolerance means shortfalls will sometimes happen.
- Not a claim that no other tool allocates a budget. We reviewed Shopify inventory apps (which do per-product forecasting and reorder quantities) and open-to-buy tools (which set a budget from a deterministic formula), not enterprise planning software. Our difference is deriving the budget from a probabilistic cash-risk constraint and allocating it by marginal profit per dollar with explained deferrals.

Full limitations are in the README.
