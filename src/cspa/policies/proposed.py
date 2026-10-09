"""Policy C: cash gate + marginal-value allocator. The proposed method."""
from cspa.policies.base import Policy


class ProposedPolicy(Policy):
    """1. Cash gate: the largest budget B with P(min cash >= buffer) >= 1 - alpha.
    2. Allocator: spend B where expected profit per dollar is highest."""

    name = "C"
    label = "C: cash gate + marginal allocator (proposed)"
    budget_rule = "gate"
    alloc_rule = "marginal"


class GateProportionalPolicy(Policy):
    """Ablation: cash gate, but the budget is split in proportion to policy A's need.
    Isolates what the cash gate contributes without the allocator."""

    name = "C_gate_prop"
    label = "Cash gate + proportional split"
    budget_rule = "gate"
    alloc_rule = "proportional"


class ProposedPlusPolicy(Policy):
    """C+ : policy C with two changes, both found while verifying the specified design.

    1. Ranking. Units are funded by expected profit per dollar of capital committed
       (purchase cost plus the cost of units expected to be left unsold), instead of per
       dollar of purchase cost alone. Under a binding budget the specified ranking funds
       high-margin units that may well not sell ahead of low-margin units that surely will.
    2. Fallback. When no budget is safe, C buys nothing. C+ buys the plan that gets
       closest to safety (see cash/gate.py, "best_effort").

    C+ is reported next to C, never in place of it.
    """

    name = "C_plus"
    label = "C+ (capital-aware ranking, best-effort fallback)"
    budget_rule = "gate"
    alloc_rule = "marginal_committed"
    on_infeasible = "best_effort"
