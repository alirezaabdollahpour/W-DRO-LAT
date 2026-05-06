"""Adversarial training algorithms for input-space CIFAR-10."""

from .base import BaseAdvTrainer, EpochMetrics
from .npf import NPFTrainer
from .nn_dro import NNDROTrainer
from .madry import MadryTrainer
from .wrm import WRMTrainer
from .wfr import WFRTrainer
from .dual import SDRODualTrainer
from .new_ppa import NewPPATrainer

ALGORITHMS = {
    "npf": NPFTrainer,
    "nn_dro": NNDROTrainer,
    "madry": MadryTrainer,
    "wrm": WRMTrainer,
    "wfr": WFRTrainer,
    "dual": SDRODualTrainer,
    "new_ppa": NewPPATrainer,
}

__all__ = [
    "BaseAdvTrainer",
    "EpochMetrics",
    "NPFTrainer",
    "NNDROTrainer",
    "MadryTrainer",
    "WRMTrainer",
    "WFRTrainer",
    "SDRODualTrainer",
    "NewPPATrainer",
    "ALGORITHMS",
]
