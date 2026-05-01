"""Monge-gap analysis package: gap utility, adversary push wrappers."""
from monge_gap_utils.monge_gap import (
    monge_gap,
    monge_gap_hungarian,
    monge_gap_sinkhorn,
    median_pairwise_distance_sq,
    sinkhorn_w2_sq,
)
from monge_gap_utils.eval_adversary import REGISTRY

__all__ = [
    "monge_gap",
    "monge_gap_hungarian",
    "monge_gap_sinkhorn",
    "median_pairwise_distance_sq",
    "sinkhorn_w2_sq",
    "REGISTRY",
]
