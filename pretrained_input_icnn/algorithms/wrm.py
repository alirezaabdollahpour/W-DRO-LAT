"""WRM (Sinha et al., 2018) — exactly MPA with R=1: pure particle ascent.

WRM is the no-reassignment special case of Algorithm ``implicit_mpa``: per
batch, every particle z_i (initialized at x_i) takes K gradient-ascent steps
on the SAME per-sample DRO objective MPA uses,

    primary(clf(z_i), y_i) - λ c(z_i, x_i),

with the transport penalty anchored at the original input x_i, followed by a
clamp to the valid normalized CIFAR pixel box after every step. There is no
reassignment and no other cross-sample interaction.

To make WRM a byte-exact ablation of MPA (same λ via cfg.lambda_param, same
transport-cost geometry via cfg.transport_cost, same margin/CE primary via
cfg.use_margin_loss, same clean-correct masking, same clamp, same eval
semantics), the ascent bursts are the SAME functions the MPA trainer calls:

* ``wrm_step_rule="const_lr"`` (default) — ``_const_lr_ascent`` with the WRM
  paper's diminishing schedule η_s = wrm_inner_lr / sqrt(s), s = 1..K —
  identical to MPA's round-0 burst (MPA round 0 with ppa_round0_lr equal to
  wrm_inner_lr produces bitwise the same particles). Fully per-sample: the
  trajectory is invariant to how the batch is sharded across DDP ranks.
* ``wrm_step_rule="bb_armijo"`` — ``_bb_armijo_ascent`` with the shared
  BB+Armijo step rule (one line-searched step size on the batch-mean
  objective), kept for cross-method parity arms.

Faithfulness properties shared with MPA:

* ``attack_clean_correct_only`` is honored: the attacked batch is the
  clean-correct subset; misclassified samples stay exactly at x.
* ``transport_for_eval`` runs the same WRM attack with predicted labels, so
  the per-epoch transport-eval columns measure robustness to the WRM
  adversary (cap the cost with ``--eval-transport-samples``).
* ``_last_inner_loss`` logs the actual inner objective
  ``mean(primary - λ·cost)`` over the attacked subset.
* Stateless/transductive: the frozen-adversary phase is rejected, and under
  DDP eval-mode adversary forwards are required (same SyncBatchNorm
  collective-uniformity argument as MPA — masked subsets give ranks
  data-dependent forward counts).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..utils import adversary_loss_per_sample, frozen_module
from .base import BaseAdvTrainer
from .new_ppa import _bb_armijo_ascent, _const_lr_ascent


class WRMTrainer(BaseAdvTrainer):
    name = "wrm"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config
        if int(getattr(cfg, "frozen_adversary_epochs", 0) or 0) > 0:
            raise ValueError(
                "wrm is a stateless/transductive adversary: there is no "
                "learned map to freeze. Use --frozen-adversary-epochs 0 with "
                "--algorithm wrm."
            )
        if self.dist.is_distributed and not bool(
            getattr(cfg, "adversary_classifier_eval", True)
        ):
            raise ValueError(
                "wrm under DDP requires eval-mode adversary forwards "
                "(--adversary-classifier-eval): with --adversary-classifier-"
                "train the SyncBatchNorm-converted classifier fires stat "
                "collectives on every train-mode forward, and the masked "
                "per-rank inner loops execute data-dependent forward counts "
                "— the ranks' collective sequences desynchronize. Train-mode "
                "forwards also break the per-sample ascent semantics."
            )

    # ------------------------------------------------------------------
    # Inner maximization: K ascent steps, no reassignment (MPA with R=1)
    # ------------------------------------------------------------------
    def _wrm_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attack_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """K-step per-sample ascent from z = x; unattacked samples stay at x.

        Caller must hold ``frozen_module`` on the classifier. No collectives
        are issued on any path (there is no reassignment pool to gather).
        """
        cfg = self.config
        B = x.size(0)
        if attack_mask is None:
            mask = torch.ones(B, dtype=torch.bool, device=x.device)
        else:
            mask = attack_mask.to(device=x.device, dtype=torch.bool)

        x_att = x[mask].detach()
        y_att = y[mask]
        num_steps = int(cfg.wrm_inner_steps)
        step_rule = str(getattr(cfg, "wrm_step_rule", "const_lr")).lower()

        if x_att.size(0) == 0 or num_steps <= 0:
            z_att = x_att.clone()
        elif step_rule == "const_lr":
            z_att = _const_lr_ascent(
                x_att, x_att, self._classifier_module, y_att,
                float(cfg.lambda_param),
                num_steps=num_steps,
                lr=float(cfg.wrm_inner_lr),
                diminishing=True,
                use_margin=bool(cfg.use_margin_loss),
                cost_fn=self._transport_cost,
            )
        elif step_rule == "bb_armijo":
            z_att = _bb_armijo_ascent(
                x_att, x_att, self._classifier_module, y_att,
                float(cfg.lambda_param),
                num_steps=num_steps,
                use_margin=bool(cfg.use_margin_loss),
                cost_fn=self._transport_cost,
                bb_alpha0=cfg.bb_alpha0,
                bb_alpha_min=cfg.bb_alpha_min,
                bb_alpha_max=cfg.bb_alpha_max,
                bb_ls_c=cfg.bb_ls_c,
                bb_ls_shrink=cfg.bb_ls_shrink,
                bb_ls_max_steps=cfg.bb_ls_max_steps,
            )
        else:
            raise ValueError(f"Unsupported wrm_step_rule: {step_rule}")

        z_full = x.detach().clone()
        z_full[mask] = z_att
        return z_full.detach()

    # ------------------------------------------------------------------
    # Trainer contract (mirrors NewPPATrainer)
    # ------------------------------------------------------------------
    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        lam = float(cfg.lambda_param)
        use_margin = bool(cfg.use_margin_loss)
        clf = self._classifier_module

        attack_mask = None
        if bool(getattr(cfg, "attack_clean_correct_only", True)):
            attack_mask = self._clean_correct_attack_mask(x, y)

        with frozen_module(clf, eval_mode=self._adversary_classifier_eval_mode()):
            z = self._wrm_attack(x, y, attack_mask)
            with torch.no_grad():
                # Inner-objective diagnostic over the ATTACKED subset only —
                # same masked semantics as MPA/NPF/NN-DRO.
                if attack_mask is None or bool(attack_mask.any()):
                    z_m = z if attack_mask is None else z[attack_mask]
                    x_m = x if attack_mask is None else x[attack_mask]
                    y_m = y if attack_mask is None else y[attack_mask]
                    primary = adversary_loss_per_sample(
                        clf(z_m), y_m, use_margin=use_margin
                    )
                    cost = self._transport_cost(z_m, x_m)
                    self._last_inner_loss = float(
                        (primary - lam * cost).mean().item()
                    )
                else:
                    self._last_inner_loss = 0.0
        return z.detach()

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Fresh WRM attack with predicted labels (same convention as MPA).

        Pseudo-labels ``argmax clf(x)`` stand in for y; on clean-correct test
        points the objective coincides with training. adv_acc is a label-free
        metric (see MPA's transport_for_eval for the caveats); read benchmark
        robustness from input_pgd_acc. Cap the per-epoch cost with
        ``--eval-transport-samples``.
        """
        clf = self._classifier_module
        with frozen_module(clf, eval_mode=True):
            with torch.no_grad():
                y_hat = clf(x).argmax(dim=1)
            with torch.enable_grad():
                z = self._wrm_attack(x, y_hat, None)
        return z.detach()
