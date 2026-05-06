"""New_PPA: free-weight projected particle ascent.

Adapted from ``MNIST_Cuturi/algorithms/new_ppa.py``: round 0 is a WRM
ascent on the inputs; refinement rounds alternate within-class best-
response projection with a constant-LR WRM ascent anchored at the original
inputs. Ranges are clamped to the valid normalized CIFAR-10 pixel bounds
each step. Early-stop the refinement when the marginal gain falls below
``ppa_gain_rtol`` * (running objective scale).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import (
    adversary_loss_per_sample,
    clamped_normalized_copy,
    free_weight_projection_images,
    set_requires_grad,
)
from .base import BaseAdvTrainer


def _wrm_ascent(
    z0: torch.Tensor,
    x_anchor: torch.Tensor,
    classifier: nn.Module,
    y: torch.Tensor,
    lam: float,
    num_steps: int,
    lr: float,
    diminishing: bool,
    use_margin: bool,
    step_offset: int = 0,
) -> torch.Tensor:
    if num_steps <= 0:
        return z0.detach()
    z = z0.detach().clone()
    for s in range(1 + step_offset, num_steps + 1 + step_offset):
        z.requires_grad_(True)
        primary = adversary_loss_per_sample(classifier(z), y, use_margin=use_margin)
        grads = torch.autograd.grad(primary.sum(), z, create_graph=False)[0]
        eta = lr / math.sqrt(s) if diminishing else lr
        with torch.no_grad():
            z = z.detach() + eta * (grads - 2.0 * lam * (z.detach() - x_anchor))
            z = clamped_normalized_copy(z)
    return z.detach()


class NewPPATrainer(BaseAdvTrainer):
    name = "new_ppa"

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        lam = float(cfg.lambda_param)
        use_margin = bool(cfg.use_margin_loss)
        clf = self._classifier_module
        clf.eval()
        set_requires_grad(clf, False)

        # Round 0: diminishing-LR WRM ascent.
        z = _wrm_ascent(
            x, x, clf, y, lam,
            num_steps=int(cfg.ppa_round0_steps),
            lr=float(cfg.ppa_round0_lr),
            diminishing=True,
            use_margin=use_margin,
        )

        # Refinement rounds: alternate free-weight projection + constant-LR
        # WRM ascent, with adaptive early stopping on the projection gain.
        for round_idx in range(1, max(1, int(cfg.ppa_num_rounds))):
            z, _y_proj, gain, obj_scale, _ = free_weight_projection_images(
                z, x, y, clf, lam, use_margin=use_margin
            )
            if (
                round_idx >= int(cfg.ppa_min_rounds)
                and gain <= float(cfg.ppa_gain_rtol) * max(obj_scale, 1e-12)
            ):
                break
            z = _wrm_ascent(
                z, x, clf, y, lam,
                num_steps=int(cfg.ppa_refine_steps),
                lr=float(cfg.ppa_refine_lr),
                diminishing=False,
                use_margin=use_margin,
            )

        # Final projection so the outer step sees within-class best
        # responses (matches MNIST_Cuturi's contract).
        z, _y_proj, _, _, _ = free_weight_projection_images(
            z, x, y, clf, lam, use_margin=use_margin
        )

        with torch.no_grad():
            self._last_inner_loss = float(F.cross_entropy(clf(z), y).item())
        set_requires_grad(clf, True)
        return z.detach()

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        return x.detach()
