from cspa.policies.base import InfoSet, PlanContext, Policy
from cspa.policies.otb import OTBMarginalPolicy, OTBPolicy
from cspa.policies.priority_cut import PriorityCutPolicy
from cspa.policies.proposed import GateProportionalPolicy, ProposedPolicy
from cspa.policies.reorder_point import ReorderPointPolicy

REGISTRY = {p.name: p for p in (ReorderPointPolicy, OTBPolicy, ProposedPolicy, PriorityCutPolicy, GateProportionalPolicy, OTBMarginalPolicy)}


def make_policy(name: str) -> Policy:
    if name not in REGISTRY:
        raise ValueError(f"unknown policy '{name}'; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]()


__all__ = ["InfoSet", "PlanContext", "Policy", "REGISTRY", "make_policy"]
