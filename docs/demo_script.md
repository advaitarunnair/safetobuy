# Demo script (2 to 4 minutes)

> Blocks between `BEGIN` and `END` markers are written by `scripts/render_results.py`.
> Record after `make render`, so the numbers you say are the numbers on screen.
> If the run does not support the planned headline, say what the generated sentence says.

**Setup before recording:** `make app`. Open the plan page with the scenario and week named in
beat 1. Have the "Backtest results" page ready in the sidebar.

---

## Beat 1. The problem and the person (about 20 seconds)

*On screen: the plan page, top of page.*

> "Meet the owner of a small minimart. Every week she reorders a few hundred products. Her
> inventory app tells her how much of each one to buy. It does not tell her whether she can
> afford the total. This week:"

<!-- BEGIN:demo_figure -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:demo_figure -->

> "Small shops do not usually fail because nothing sells. They fail because the rent is due the
> same week as the supplier."

*Point at the blue note:* "One thing up front. The sales data is real, from Walmart. The costs
and the bank balance are synthetic, and the minimart is our framing."

## Beat 2. Data in, forecast with uncertainty, cash runway (about 60 to 75 seconds)

*On screen: scroll to the cash runway chart.*

> "We start from sales history. A machine-learning model, LightGBM with conformal calibration,
> forecasts each product as a range rather than a single number, and we check that the ranges are
> honest: a 90% range should contain the truth 90% of the time."

*Optional cut to the forecast coverage figure on the Backtest page.*

> "Then comes the part that is not machine learning. We simulate the next four weeks two thousand
> times: sales, deliveries, supplier bills, rent. Products are simulated together, because a slow
> week is slow across the whole shelf."

*Point at the fan chart: median line, bands, dashed buffer line, the dip at the rent week.*

> "The dark band is the likely range of her bank balance. The dashed line is the safety buffer
> she chose. You can see the week the rent lands."

*Read the caption under the chart:* "If we pretended products were independent, the same plan
would look safer than it is. That is exactly the mistake that hurts in a bad week."

*Drag the risk-tolerance slider down, then back.*

> "She sets how much risk she accepts. The safe budget moves with it."

## Beat 3. What to buy, with the value quantified (about 60 to 75 seconds)

*On screen: the "Safe budget this week" line, then the table.*

<!-- BEGIN:demo_week -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:demo_week -->

> "So the tool does three things. It says how much is safe to spend. It spends that on the units
> that earn the most per dollar. And for everything it trims or defers, it shows the price of
> waiting."

*Click a deferred row and read its "Why" line. Then a fully funded row.*

*Turn on "Compare this week under policies A, B and D".*

> "Here is the same week, same cash, under the standard reorder-point rule and under open-to-buy.
> They do not look at the bank balance."

*Switch to "Backtest results".*

> "Does it hold up over time? We ran every policy through the same simulated shop on real weekly
> demand, and scored them on periods we never tuned on."

*Read the headline exactly as generated:*

<!-- BEGIN:headline_plain -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:headline_plain -->

> "Open-to-buy gives you a budget. We tell you whether you can afford it, and how to spend it best."

---

## If asked

- **"Is this just a wrapper around an LLM?"** No. The LLM only rewords numbers. Forecasting is LightGBM; the decision is a Monte Carlo cash model, a constrained budget search and an allocator. Every number in LLM text is verified in code. The app runs with no API key.
- **"Is the whole thing ML?"** No. ML for prediction, simulation and optimization for the decision.
- **"Are the dollar figures real?"** Demand and shelf prices are real. Costs, terms and cash are synthetic and disclosed in the README. Read the comparison between policies, not the dollars.
- **"What about when cash is already too low?"** The tool flags "cash at risk regardless of purchasing". See the README limitations and the C+ variant.
