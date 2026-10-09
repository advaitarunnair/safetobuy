"""Policy C: cash gate + marginal-value allocator. The proposed method."""
from cspa.policies.base import Policy


class ProposedPolicy(Policy):
    """1. Cash gate: the largest budget B with P(min cash >= buffer) >= 1 - alpha.
       If no budget meets that, the budget that maximises P(min cash >= buffer), flagged.
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


