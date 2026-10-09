# Devpost submission (ready to paste)

> Result blocks between `BEGIN` and `END` markers are written by `scripts/render_results.py`.
> Paste the rendered text, not the markers. Track: **AI + Business**.

## Title

<!-- BEGIN:project_name -->SafeToBuy<!-- END:project_name -->

## Tagline

Open-to-buy gives you a budget. We tell you whether you can afford it, and how to spend it best.

## Short description

A purchasing assistant for small retailers. It forecasts demand as a calibrated range, simulates
the shop's cash for the next four weeks, finds the largest order budget that keeps the chance of
a cash shortfall within the owner's tolerance, spends it where profit per dollar is highest, and
explains every buy and every deferral. Tested in a closed-loop simulation on real Walmart sales
against reorder-point and open-to-buy policies.

## Full description

### Problem and target users

Small retailers fail from cash crunches, not only from slow sales. A minimart owner reorders
every week across hundreds of products. Inventory apps recommend a quantity for each product, one
at a time, so a week's recommendations can add up to more than the shop can pay for once rent,
payroll and supplier bills are counted. The owner then cuts the list by gut feel, at the moment a
wrong cut costs the most.

Our users are small retailers, convenience stores and independent e-commerce sellers who need
purchasing guidance without enterprise planning software. The demo persona is a small Singapore
minimart; that is framing only.

### Technical approach and components

ML for prediction, simulation and optimization for the decision.

1. **Forecast (machine learning).** LightGBM quantile regression, one pooled model per quantile, with rolling conformal calibration. Features use only data available at the forecast date; tests replace all later data with noise and require byte-identical forecasts and decisions.
2. **Joint demand sampling.** We bootstrap whole past weeks of forecast errors so products that were surprised together stay together. Treating products as independent understates cash risk.
3. **Monte Carlo cash model (simulation).** Each demand path runs through stock on hand, deliveries in transit, supplier bills and fixed costs: cash = cash + revenue - fixed costs - supplier payments, with sales capped by stock.
4. **Cash gate (optimization).** The largest purchase budget B such that P(cash never falls below the buffer over four weeks) >= 1 - alpha, found by bisection on common random numbers with a grid check, because buying stock that sells can raise cash and the search is not monotone. If no budget is safe, the gate takes the one with the best chance and flags it.
5. **Allocator (optimization).** Units are funded in order of marginal expected profit per dollar, Cu x P(demand > x) - Co x P(demand <= x), in whole packs with minimum order quantities. Every product is labelled full, partial or deferred, with its stockout probability and the expected margin lost by waiting.
6. **Explanations.** Deterministic templates, or an LLM that only rewords computed numbers. Code extracts every number from the LLM's text and checks it against the facts; any mismatch falls back to the template. The app works without an API key.
7. **Evidence.** A simulated retailer replays actual weekly demand for 300 products from the M5 Walmart dataset. Four policies and two ablations act in the same world: reorder point (A), open-to-buy (B), open-to-buy with priority cuts (D), ours (C). Settings were chosen on two 26-week windows and scored once on two later windows. Every number in the README is generated from a run manifest (config hash, git commit, seed).

M5 has no costs, lead times, payment terms or cash, so those are synthetic and disclosed in full.
Dollar figures are illustrative; the comparison between policies is the result.

**Result on held-out data:**

<!-- BEGIN:headline_plain -->
Policy C had 27.9% fewer cash-shortfall weeks than policy A (62 vs 86) (90% bootstrap interval for the reduction: +18.4% to +42.4%), and 1.0% higher margin than policy B (90% bootstrap interval for the difference: +0.6% to +1.7%), but more shortfall weeks than B (62 vs 58). Against policy A, C's margin was 2.9% lower. The planned headline form is not supported by this run, so the result is stated as it happened.
<!-- END:headline_plain -->

<!-- BEGIN:headline_numbers -->
C had 62 cash-shortfall weeks against 86 for A and 58 for B, out of 156 held-out weeks each: 27.9% fewer than A (90% interval +18.4% to +42.4%). C's gross margin was +1.0% against B (90% interval +0.6% to +1.7%) and -2.9% against A. Weeks with cash below zero: A 0, B 2, C 0.
<!-- END:headline_numbers -->

We report this as it happened. Where it did not win, and a second product slice, are in the README.

Stack: Python, LightGBM, NumPy, pandas, Streamlit, Plotly, Anthropic API (optional), pytest.

### Real-world impact

- It answers the question owners actually ask: can I afford this week's order, and if not, what do I drop?
- It puts a price on every deferral, so the owner trades margin against safety with numbers instead of instinct. Our results show that trade-off is real.
- It reports risk honestly: a chance of shortfall the owner chose, not a single optimistic cash line, and a plain warning when no purchase plan is safe.
- It needs only what a small shop already has: sales history, prices, supplier terms and a bank balance.

### What we would do next

Price the trade-off inside the gate (a chance constraint protects the buffer but does not weigh
the margin it gives up), test on a retailer's real costs and payment terms, and add CSV upload.

## Links

- Repository: https://github.com/advaitarunnair/safetobuy
- Demo video: to be recorded and uploaded as a public YouTube video.
- Live app: Streamlit Community Cloud, after the deploy click.
