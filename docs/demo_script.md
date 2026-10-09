# Demo script (about 3 minutes 30 seconds)

> Blocks between `BEGIN` and `END` markers are written by `scripts/render_results.py`, so the
> numbers you say are the numbers on screen. Lines in quotation marks are spoken; everything else
> is stage direction. Spoken text is about 470 words: at a normal pace that is under four minutes.

**Before recording:** `make app`. Open the plan page at the scenario and week named in beat 3.
Keep the "Backtest results" page one click away in the sidebar.

---

## Beat 1. The problem and the person (20 seconds)

*On screen: the plan page, top.*

> "This is the owner of a small minimart. Every week she reorders a few hundred products. Her
> inventory app says how much of each to buy. It doesn't say whether she can afford the total."

*Read the figure below from the screen:*

<!-- BEGIN:demo_figure -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:demo_figure -->

> "Small shops rarely fail because nothing sells. They fail because the rent is due the same
> week as the supplier."

## Beat 2. Data in, forecast with uncertainty, cash runway (75 seconds)

*Point at the blue note.*

> "One thing up front. The sales data is real, from Walmart. The costs and the bank balance are
> synthetic, and the minimart is our framing."

*Scroll to the cash runway chart.*

> "We start from sales history. A machine-learning model, LightGBM with conformal calibration,
> forecasts each product as a range, not a single number. We check those ranges are honest: a
> ninety percent range should hold the truth ninety percent of the time, and ours does."

> "The next part is not machine learning. We simulate the next four weeks two thousand times:
> sales, deliveries, supplier bills, rent. Products are simulated together, because a slow week
> is slow across the whole shelf."

*Point at the median line, the bands, the dashed buffer line, the dip in the rent week.*

> "The band is the likely range of her bank balance. The dashed line is the safety buffer she
> chose. You can see the week the rent lands."

*Read the last sentence of the caption under the chart.*

> "Pretend products are independent and the same plan looks safer than it is."

*Drag the risk-tolerance slider down and back.*

> "She sets how much risk she accepts. The safe budget moves with it."

## Beat 3. What to buy, with the value quantified (95 seconds)

*On screen: the "Safe budget this week" line, then the table. The scenario, week and numbers:*

<!-- BEGIN:demo_week -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:demo_week -->

> "So the tool does three things. It says how much is safe to spend. It spends that on the units
> that earn the most per dollar. And for everything it trims or defers, it shows the price of
> waiting."

*Point at one deferred row and read its "Why" line. Then one fully funded row.*

*Turn on "Compare this week under policies A, B and D".*

> "Same week, same cash, under the standard reorder-point rule and under open-to-buy. Neither
> looks at the bank balance."

*Switch to "Backtest results".*

> "Does it hold up? We ran every policy through the same simulated shop on real weekly demand,
> chose our settings on one year, and scored once on the next."

*Read the result exactly as generated:*

<!-- BEGIN:headline_spoken -->
_No results yet. Place the M5 files in `data/raw/`, run `make all`, and this block is filled in by `scripts/render_results.py`._
<!-- END:headline_spoken -->

> "Open-to-buy gives you a budget. We tell you whether you can afford it, and how to spend it
> best."

---

## If asked

- **"Is this just a wrapper around an LLM?"** No. The LLM only rewords numbers. Forecasting is LightGBM; the decision is a Monte Carlo cash model, a constrained budget search and an allocator. Every number in LLM text is verified in code. The app runs with no API key.
- **"Is the whole thing ML?"** No. ML for prediction, simulation and optimization for the decision.
- **"Are the dollar figures real?"** Demand and shelf prices are real. Costs, terms and cash are synthetic and disclosed in the README. Read the comparison between policies, not the dollars.
- **"Did you beat open-to-buy?"** On margin, yes, by a small amount with an interval clear of zero. On shortfall weeks, no. The README says exactly where it won and where it did not.
- **"What about when no budget is safe?"** The tool says so, and recommends the plan with the best chance rather than buying nothing, because a shop that stops buying stops selling.
