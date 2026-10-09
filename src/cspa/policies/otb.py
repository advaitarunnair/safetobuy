"""Policy B: open-to-buy budget, split proportionally. Baseline."""
from cspa.policies.base import Policy


class OTBPolicy(Policy):
    """Budget = planned sales + planned markdowns + planned end-of-period stock
    - beginning stock - on order, all at cost (standard open-to-buy form).

    Planned sales = median forecast over the cover period; planned markdowns = 0
    (no markdown data); planned ending stock = target weeks of cover x planned
    weekly sales. The budget is a cap and is split across SKUs in proportion to
    each SKU's policy-A order need.
    """

    name = "B"
    label = "B: open-to-buy"
    budget_rule = "otb"
    alloc_rule = "proportional"


class OTBMarginalPolicy(Policy):
    """Ablation: open-to-buy budget, but allocated by marginal value per dollar.
    Isolates what the allocator contributes without the cash gate."""

    name = "OTB_marginal"
    label = "OTB budget + marginal allocator"
    budget_rule = "otb"
    alloc_rule = "marginal"
