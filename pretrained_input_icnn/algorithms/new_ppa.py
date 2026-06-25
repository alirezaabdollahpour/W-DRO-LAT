"""New_PPA: free-weight projected particle ascent — BB+Armijo on z.

Adapted from ``MNIST_Cuturi/algorithms/new_ppa.py``: round 0 is an inner
ascent on the inputs; refinement rounds alternate within-class best-
response projection with another inner ascent anchored at the original
inputs. Ranges are clamped to the valid normalized CIFAR-10 pixel bounds
each step. Early-stop the refinement when the marginal gain falls below
``ppa_gain_rtol`` * (running objective scale).

The original LR-CIFAR10 reference uses constant-LR WRM ascent inside
each round. Here we replace that with the SAME BB+Armijo step rule
used by NPF, applied to z directly. The projection rounds are
unchanged; BB state is fresh at the start of each ascent burst because
z is anchored / re-projected between bursts. Transport costs use the
configured convention, defaulting to legacy normalized-coordinate MSE.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import (
    BBArmijoState,
    adversary_loss_per_sample,
    bb_armijo_step_tensor,
    clamped_normalized_copy,
    free_weight_projection_images,
    frozen_module,
)
from .base import BaseAdvTrainer


def _bb_armijo_ascent(
    z0: torch.Tensor,
    x_anchor: torch.Tensor,
    classifier: nn.Module,
    y: torch.Tensor,
    lam: float,
    num_steps: int,
    use_margin: bool,
    cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    bb_alpha0: float,
    bb_alpha_min: float,
    bb_alpha_max: float,
    bb_ls_c: float,
    bb_ls_shrink: float,
    bb_ls_max_steps: int,
) -> torch.Tensor:
    """BB+Armijo ascent on ``primary(classifier(z), y) - λ ||z - x_anchor||^2``.

    The penalty is anchored to the ORIGINAL inputs (matches the legacy
    ``wrm_ascent_x_anchored`` semantics from MNIST_Cuturi). BB state is
    fresh — between PPA rounds the projection moves z to a different
    sample within its class, breaking any history we'd hope to reuse.
    """
    if num_steps <= 0:
        return z0.detach()
    bb_state = BBArmijoState.create(
        alpha0=bb_alpha0,
        alpha_min=bb_alpha_min,
        alpha_max=bb_alpha_max,
        ls_c=bb_ls_c,
        ls_shrink=bb_ls_shrink,
        ls_max_steps=bb_ls_max_steps,
        reject_on_armijo_failure=True,
    )

    def f_obj(z_var: torch.Tensor, create_graph: bool) -> torch.Tensor:
        primary = adversary_loss_per_sample(classifier(z_var), y, use_margin=use_margin)
        cost = cost_fn(z_var, x_anchor)
        return (primary - lam * cost).mean()

    z = z0.detach().clone()
    for _ in range(int(num_steps)):
        z, bb_state, _, _ = bb_armijo_step_tensor(z, f_obj, bb_state)
        z = clamped_normalized_copy(z)
    return z.detach()


class NewPPATrainer(BaseAdvTrainer):
    name = "new_ppa"

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        lam = float(cfg.lambda_param)
        use_margin = bool(cfg.use_margin_loss)
        clf = self._classifier_module

        bb_kwargs = dict(
            bb_alpha0=cfg.bb_alpha0,
            bb_alpha_min=cfg.bb_alpha_min,
            bb_alpha_max=cfg.bb_alpha_max,
            bb_ls_c=cfg.bb_ls_c,
            bb_ls_shrink=cfg.bb_ls_shrink,
            bb_ls_max_steps=cfg.bb_ls_max_steps,
        )

        with frozen_module(clf):
            # Round 0: BB+Armijo ascent (replaces diminishing-LR WRM).
            z = _bb_armijo_ascent(
                x, x, clf, y, lam,
                num_steps=int(cfg.ppa_round0_steps),
                use_margin=use_margin,
                cost_fn=self._transport_cost,
                **bb_kwargs,
            )

            # Refinement rounds: alternate free-weight projection + BB+Armijo
            # ascent, with adaptive early stopping on the projection gain.
            for round_idx in range(1, max(1, int(cfg.ppa_num_rounds))):
                z, _y_proj, gain, obj_scale, _ = free_weight_projection_images(
                    z, x, y, clf, lam, use_margin=use_margin,
                    transport_cost=getattr(cfg, "transport_cost", "normalized_mse")
                )
                if (
                    round_idx >= int(cfg.ppa_min_rounds)
                    and gain <= float(cfg.ppa_gain_rtol) * max(obj_scale, 1e-12)
                ):
                    break
                z = _bb_armijo_ascent(
                    z, x, clf, y, lam,
                    num_steps=int(cfg.ppa_refine_steps),
                    use_margin=use_margin,
                    cost_fn=self._transport_cost,
                    **bb_kwargs,
                )

            # Final projection so the outer step sees within-class best
            # responses (matches MNIST_Cuturi's contract).
            z, _y_proj, _, _, _ = free_weight_projection_images(
                z, x, y, clf, lam, use_margin=use_margin,
                transport_cost=getattr(cfg, "transport_cost", "normalized_mse")
            )

            with torch.no_grad():
                self._last_inner_loss = float(F.cross_entropy(clf(z), y).item())
        return z.detach()

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        return x.detach()
