"""Cash gate: the largest purchase budget B that keeps

    P( min_t cash_t >= buffer ) >= 1 - alpha.

Evaluating a candidate B is closed-loop: run the allocator at B, simulate cash
with that plan, read off the probability. All candidates are evaluated on the
same demand paths (common random numbers), so `evaluate` is deterministic.

Monotonicity in B is assumed by the bisection but not guaranteed, because
bought stock earns revenue. After bisecting we therefore check a coarse grid
and keep the largest B that is actually feasible; disagreements between the
grid and the monotone assumption are counted and reported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

STATUS_UNCONSTRAINED = "unconstrained"  # the full plan is already cash-safe
STATUS_CONSTRAINED = "constrained"  # the gate is binding
STATUS_AT_RISK = "cash_at_risk"  # no budget meets the target, even B = 0


@dataclass
class BudgetResult:
    B: float
    prob_safe: float
    n_evals: int
    monotonicity_violations: int
    status: str
    target: float
    B_max: float
    prob_at_zero: float | None = None
    prob_at_max: float | None = None
    evals: list = field(default_factory=list)

    @property
    def flag(self) -> str | None:
        return "cash at risk regardless of purchasing" if self.status == STATUS_AT_RISK else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flag"] = self.flag
        return d


def safe_budget(
    evaluate: Callable[[float], float],
    B_max: float,
    alpha: float,
    tol_frac: float = 0.005,
    tol_abs: float = 20.0,
    max_iter: int = 12,
    n_grid: int = 8,
    on_infeasible: str = "zero",
    tail: Callable[[float], float] | None = None,
) -> BudgetResult:
    """Search [0, B_max] for the largest feasible budget.

    evaluate(B) must return P(min cash >= buffer) for the plan the allocator
    produces at budget B. B_max is the cost of the unconstrained plan.

    on_infeasible decides what happens when no budget meets the target, B = 0 included:
      "zero"        : return B = 0 with the flag "cash at risk regardless of purchasing"
                      (the behaviour specified for the gate).
      "best_effort" : same flag, but return the candidate budget that gets closest to
                      safety: highest P(safe | B); ties broken by the highest
                      alpha-quantile of end-of-horizon cash, tail(B), then by larger B.
                      Needs `tail`.
    """
    target = 1.0 - alpha
    seen: dict[float, float] = {}

    def ev(B: float) -> float:
        B = float(B)
        if B not in seen:
            seen[B] = float(evaluate(B))
        return seen[B]

    def ok(B: float) -> bool:
        return ev(B) >= target - 1e-12

    def result(B: float, status: str, violations: int = 0) -> BudgetResult:
        return BudgetResult(
            B=float(B),
            prob_safe=ev(B),
            n_evals=len(seen),
            monotonicity_violations=int(violations),
            status=status,
            target=target,
            B_max=float(B_max),
            prob_at_zero=seen.get(0.0),
            prob_at_max=seen.get(float(B_max)),
            evals=sorted(seen.items()),
        )

    B_max = max(float(B_max), 0.0)
    if B_max <= 0.0:
        return result(0.0, STATUS_UNCONSTRAINED if ok(0.0) else STATUS_AT_RISK)
    if ok(B_max):
        return result(B_max, STATUS_UNCONSTRAINED)

    tol = max(float(tol_abs), float(tol_frac) * B_max)

    def bisect(lo: float, hi: float) -> float:
        """lo feasible, hi infeasible; returns a feasible point within tol of the edge."""
        it = 0
        while hi - lo > tol and it < max_iter:
            mid = 0.5 * (lo + hi)
            if ok(mid):
                lo = mid
            else:
                hi = mid
            it += 1
        return lo

    zero_ok = ok(0.0)
    B_bis = bisect(0.0, B_max) if zero_ok else None

    grid = np.linspace(0.0, B_max, n_grid + 2)[1:-1]
    grid_ok = [ok(g) for g in grid]
    if B_bis is None:
        violations = sum(grid_ok)
    else:
        violations = sum((g < B_bis - 1e-9 and not f) or (g > B_bis + tol and f) for g, f in zip(grid, grid_ok))

    best = B_bis
    feasible_grid = [float(g) for g, f in zip(grid, grid_ok) if f]
    if feasible_grid and (best is None or max(feasible_grid) > best + tol):
        g = max(feasible_grid)
        above = [float(x) for x in grid if x > g] + [B_max]
        best = bisect(g, above[0])

    if best is None:
        best = 0.0
        if on_infeasible == "best_effort":
            if tail is None:
                raise ValueError("on_infeasible='best_effort' needs a tail(B) function")
            cands = [0.0] + [float(g) for g in grid] + [B_max]
            best = max(cands, key=lambda B: (round(ev(B), 9), round(float(tail(B)), 6), B))
        return result(best, STATUS_AT_RISK, violations)
    return result(best, STATUS_CONSTRAINED, violations)
