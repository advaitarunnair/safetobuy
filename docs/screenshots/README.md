# Screenshots

Produced by `make screenshots` after `make backtest` (headless Chrome, one full-page PNG per app page):

| File | Page |
| --- | --- |
| `01_plan.png` | This week's plan: safe budget, cash runway fan chart, buy / partial / defer table |
| `02_plan_compare.png` | Same page with the A / B / D comparison switched on |
| `03_backtest.png` | Backtest results: headline, tables, figures |
| `04_about.png` | What is real, what is synthetic, and the method |

To pin the plan page to the week used in the demo script: `make screenshots QUERY="stress=70&week=5"`.

No screenshots are committed yet: none can be taken until a backtest has been run on real M5 data.
