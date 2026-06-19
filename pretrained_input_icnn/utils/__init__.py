"""Utilities: device, seeding, image transforms, BB+Armijo, evaluation."""

from .common import (
    cuda_sync,
    frozen_module,
    get_device,
    set_deterministic,
    set_requires_grad,
    set_seed_benchmark_mode,
)
from .transforms import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    clamp_normalized_inputs_,
    clamped_normalized_copy,
    normalized_pixel_bounds,
    pixel_l2_squared,
    to_normalized,
    to_pixel,
)
from .bb_armijo import BBArmijoState, bb_armijo_step_params, bb_armijo_step_tensor
from .muon import MuonState, muon_step_params
from .eval import (
    evaluate_clean,
    evaluate_under_input_pgd,
    evaluate_under_transport,
)
from .projections import free_weight_projection_images
from .adversary_loss import adversary_loss_per_sample

__all__ = [
    "cuda_sync",
    "frozen_module",
    "get_device",
    "set_deterministic",
    "set_seed_benchmark_mode",
    "set_requires_grad",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "to_normalized",
    "to_pixel",
    "clamp_normalized_inputs_",
    "clamped_normalized_copy",
    "normalized_pixel_bounds",
    "pixel_l2_squared",
    "BBArmijoState",
    "bb_armijo_step_params",
    "bb_armijo_step_tensor",
    "MuonState",
    "muon_step_params",
    "evaluate_clean",
    "evaluate_under_input_pgd",
    "evaluate_under_transport",
    "free_weight_projection_images",
    "adversary_loss_per_sample",
]
