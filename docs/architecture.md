# Architecture

**ML for prediction, simulation and optimization for the decision.** The diagram marks which is which.

```mermaid
flowchart TB
    subgraph IN["Inputs"]
        M5["M5 weekly sales, prices, calendar<br/>(real, US Walmart)"]
        SYN["Costs, lead times, packs, payment terms,<br/>fixed costs, opening cash<br/>(SYNTHETIC, seeded config)"]
    end

    subgraph ML["Prediction (machine learning)"]
        FE["Past-only features<br/>data/features.py"]
        LGB["LightGBM quantile models<br/>7 quantiles, pooled across SKUs"]
        CONF["Conformal calibration<br/>from past out-of-sample errors"]
        CACHE[("Forecast cache<br/>one forecast per cutoff week,<br/>shared by all policies")]
        FE --> LGB --> CONF --> CACHE
    end

    subgraph DEC["Decision (simulation + optimization, not ML)"]
        SAMP["Joint demand sampler<br/>residual-block bootstrap,<br/>keeps cross-SKU correlation"]
        CASH["Monte Carlo cash model<br/>cash = cash + revenue - fixed - supplier payments"]
        GATE["Cash gate<br/>largest budget B with<br/>P(cash never below buffer) >= 1 - alpha<br/>(if none is safe: best-chance budget, flagged)"]
        ALLOC["Marginal-value allocator<br/>fund units by expected profit per dollar,<br/>packs and MOQs"]
        PLAN["Purchase plan per SKU<br/>full / partial / defer"]
        SAMP --> CASH
        SAMP --> ALLOC
        GATE -- "candidate budget B" --> ALLOC
        ALLOC -- "plan at B" --> CASH
        CASH -- "P(safe) at B" --> GATE
        GATE --> PLAN
    end

    subgraph EXP["Explanation"]
        FACTS["Structured facts per SKU and week"]
        TPL["Deterministic templates"]
        LLM["LLM rewording<br/>every number checked against the facts,<br/>else template"]
        FACTS --> TPL
        FACTS --> LLM
    end

    subgraph EVAL["Evaluation"]
        WORLD["Simulated retailer<br/>arrivals, sales, revenue, payments, orders"]
        BASE["Baselines in the same world<br/>A reorder point, B open-to-buy,<br/>D priority cuts, two ablations"]
        MET["Metrics, bootstrap intervals,<br/>manifest, generated README numbers"]
        WORLD --> MET
        BASE --> WORLD
    end

    M5 --> FE
    M5 -- "realised demand" --> WORLD
    SYN --> CASH
    SYN --> ALLOC
    SYN --> WORLD
    CACHE --> SAMP
    CACHE --> BASE
    STATE["Shop state<br/>cash, stock, pipeline, payables"] --> CASH
    WORLD --> STATE
    PLAN --> WORLD
    PLAN --> FACTS
    MET --> APP["Streamlit app<br/>fan chart, safe budget, plan table,<br/>policy comparison, backtest page"]
    PLAN --> APP
    TPL --> APP
    LLM --> APP
```

## The weekly loop

1. Forecast: read the cached quantile forecast for the cutoff week (last observed week).
2. Sample: draw joint demand paths for the next four weeks from the forecast and the bank of past forecast errors.
3. Gate: bisect on the budget B. Each candidate B runs the allocator, then the cash simulation on the same demand paths, and reads off P(cash never below the buffer). A coarse grid then checks the result, because buying stock that sells quickly can raise cash and break monotonicity. If no budget meets the target, the gate returns the one with the best chance and flags the plan.
4. Allocate: spend the safe budget on the units with the highest expected profit per dollar.
5. Explain: turn the plan into facts, then sentences.
6. In the backtest only: step the simulated shop with the actual M5 demand for that week and repeat.

A static copy of this diagram is at `docs/architecture.png` (`python scripts/make_architecture.py`).
