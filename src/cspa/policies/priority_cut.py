"""Policy D: open-to-buy budget with priority cuts. A stronger baseline than B."""
from cspa.policies.base import Policy


class PriorityCutPolicy(Policy):
    """Open-to-buy budget, but SKUs are funded in full in priority order
    (stockout probability x unit margin, highest first) until the budget runs out."""

    name = "D"
    label = "D: open-to-buy + priority cuts"
    budget_rule = "otb"
    alloc_rule = "priority"
