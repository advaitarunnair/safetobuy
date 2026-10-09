"""Policy A: reorder point / order-up-to. Baseline."""
from cspa.policies.base import Policy


class ReorderPointPolicy(Policy):
    """Per SKU, order up to: forecast demand over (lead time + review period) + z * sigma.

    Spends whatever that requires, regardless of cash.
    """

    name = "A"
    label = "A: reorder point"
    budget_rule = "none"
    alloc_rule = "need"
