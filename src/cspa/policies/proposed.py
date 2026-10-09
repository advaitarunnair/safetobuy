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
    """Variant of C that differs only when the gate finds no safe budget at all.

    C as specified then buys nothing. This variant instead buys the plan that gets
    closest to safety (see cash/gate.py, "best_effort"). It is reported next to C,
    never in place of it.
    """

    name = "C_plus"
    label = "C+ (C with best-effort fallback)"
    budget_rule = "gate"
    alloc_rule = "marginal"
    on_infeasible = "best_effort"
