"""New_PPA / MPA: multi-start particle ascent with within-class reassignment.

Implements Algorithm ``implicit_mpa`` per batch: R rounds of

  (a) K ascent steps on every particle z_i for the objective
      ``primary(clf(z_i), y_i) - λ c(z_i, x_i)`` — the transport penalty is
      ALWAYS anchored at the original input x_i, never the reassigned point;
  (b) within-class free-weight reassignment
      ``z_i <- argmax_{z_j in Z, y_j = y_i} primary(clf(z_j), y_i) - λ c(z_j, x_i)``.

``cfg.ppa_round_order`` selects the within-round order:

* ``ascent_first`` (default, Algorithm 1): rounds of {ascend; reassign};
  R=1 collapses to plain particle ascent (no reassignment).
* ``reassign_first``: mirrored rounds of {reassign; ascend}, closed by a
  final reassignment (R=3: R-A-R-A-R-A-R). The round-0 reassignment fires at
  z = x, so each sample's ascent starts from the best same-class training
  sample under the DRO score; like ascent_first, the theta-update always
  sees within-class best responses.

Two step rules for (a), selected by ``cfg.ppa_step_rule``:

* ``bb_armijo`` — the repo-wide BB+Armijo line-searched ascent on the
  batch-mean objective (one shared step size per step), matching the inner
  step rule of every other DRO adversary here.
* ``const_lr`` — faithful MNIST_Cuturi / paper semantics: fully per-sample
  updates ``z_i += η_s ∇_z(primary_i - λ c_i)`` with round 0 using the WRM
  diminishing schedule ``η_s = ppa_round0_lr / sqrt(s)`` and refinement
  rounds a constant ``ppa_refine_lr`` (``wrm_ascent_x`` /
  ``wrm_ascent_x_anchored_const_lr`` in MNIST_Cuturi, generalized to the
  configured transport cost and margin/CE primary). No cross-sample
  coupling, so per-rank ascent is identical to global-batch ascent.
  (The per-sample property assumes eval-mode adversary forwards — the
  default; train-mode BatchNorm would couple samples through batch
  statistics, which is why __init__ rejects that combination under DDP.)

Faithfulness properties (2026-07-08):

* ``attack_clean_correct_only`` is honored: the MPA batch is the
  clean-correct subset; misclassified samples stay at x and are excluded
  from both the ascent and the reassignment pool.
* Under DDP the reassignment pool is the GLOBAL batch: particles, labels,
  and losses are all-gathered so each sample best-responds over all
  clean-correct candidates in the global batch, regardless of world size.
  The round-count / early-stop decision is all-reduced so every rank
  executes the same number of collectives (no desync).
* ``transport_for_eval`` runs the same MPA attack with predicted labels, so
  the per-epoch transport-eval columns measure robustness to the MPA
  adversary instead of echoing clean accuracy.
* ``_last_inner_loss`` logs the actual DRO inner objective
  ``mean(primary - λ·cost)`` on the returned particles, consistent with the
  configured primary loss.

Ranges are clamped to the valid normalized CIFAR-10 pixel bounds after each
ascent step (projected ascent); reassignment candidates are therefore also
box-feasible.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from .. import distributed as dist_helpers
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
    """BB+Armijo ascent on ``primary(classifier(z), y) - λ c(z, x_anchor)``.

    The penalty is anchored to the ORIGINAL inputs (matches the legacy
    ``wrm_ascent_x_anchored`` semantics from MNIST_Cuturi). BB state is
    fresh — between MPA rounds the reassignment moves z to a different
    sample within its class, breaking any history we'd hope to reuse.
    """
    if num_steps <= 0 or z0.size(0) == 0:
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


def _const_lr_ascent(
    z0: torch.Tensor,
    x_anchor: torch.Tensor,
    classifier: nn.Module,
    y: torch.Tensor,
    lam: float,
    num_steps: int,
    lr: float,
    diminishing: bool,
    use_margin: bool,
    cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Per-sample ascent ``z_i += η_s ∇_z(primary_i - λ c(z_i, x_i))``.

    Mirrors ``MNIST_Cuturi.utils.wrm``: ``diminishing=True`` reproduces
    ``wrm_ascent_x``'s WRM schedule ``η_s = lr / sqrt(s)`` (round 0),
    ``diminishing=False`` reproduces ``wrm_ascent_x_anchored_const_lr``
    (refinement rounds). Differentiating the per-sample SUM yields exact
    per-sample gradients — no shared step size, no batch coupling — so the
    update matches Algorithm 1's ``z_i <- z_i + η(∇f_i - λ∇c_i)`` with the
    CONFIGURED cost (for normalized_mse, ``λ∇c = 2λ(z-x)/D``; the MNIST
    reference hardcoded the unnormalized ``2λ(z-x)``).
    """
    if num_steps <= 0 or z0.size(0) == 0:
        return z0.detach()
    x_anc = x_anchor.detach()
    z = z0.detach().clone()
    for s in range(1, int(num_steps) + 1):
        z = z.detach().requires_grad_(True)
        primary = adversary_loss_per_sample(classifier(z), y, use_margin=use_margin)
        cost = cost_fn(z, x_anc)
        obj = (primary - lam * cost).sum()
        grad = torch.autograd.grad(obj, z, create_graph=False)[0]
        eta = lr / math.sqrt(s) if diminishing else lr
        z = clamped_normalized_copy(z.detach() + eta * grad)
    return z.detach()


class NewPPATrainer(BaseAdvTrainer):
    name = "new_ppa"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config
        if int(getattr(cfg, "frozen_adversary_epochs", 0) or 0) > 0:
            raise ValueError(
                "new_ppa (MPA) is a stateless/transductive adversary: there is "
                "no learned map to freeze, and routing the frozen phase through "
                "transport_for_eval would re-attack with predicted labels on a "
                "rank-local pool (world-size-dependent semantics). Use "
                "--frozen-adversary-epochs 0 with --algorithm new_ppa."
            )
        if self.dist.is_distributed and not bool(
            getattr(cfg, "adversary_classifier_eval", True)
        ):
            raise ValueError(
                "new_ppa (MPA) under DDP requires eval-mode adversary forwards "
                "(--adversary-classifier-eval). With --adversary-classifier-train "
                "the SyncBatchNorm-converted classifier fires stat collectives on "
                "every train-mode forward, and MPA's per-rank inner loops execute "
                "data-dependent forward counts (masked subsets, Armijo trials) — "
                "the ranks' collective sequences desynchronize and the job hangs. "
                "Train-mode adversary forwards also break MPA's per-sample / "
                "shard-invariant ascent semantics."
            )

    # ------------------------------------------------------------------
    # Inner-loop building blocks
    # ------------------------------------------------------------------
    def _ascend(
        self,
        z0: torch.Tensor,
        x_anchor: torch.Tensor,
        y: torch.Tensor,
        num_steps: int,
        *,
        round_idx: int,
    ) -> torch.Tensor:
        cfg = self.config
        if num_steps <= 0 or z0.size(0) == 0:
            return z0.detach()
        step_rule = str(getattr(cfg, "ppa_step_rule", "bb_armijo")).lower()
        if step_rule == "const_lr":
            lr = float(cfg.ppa_round0_lr if round_idx == 0 else cfg.ppa_refine_lr)
            return _const_lr_ascent(
                z0, x_anchor, self._classifier_module, y,
                float(cfg.lambda_param),
                num_steps=num_steps,
                lr=lr,
                diminishing=(round_idx == 0),
                use_margin=bool(cfg.use_margin_loss),
                cost_fn=self._transport_cost,
            )
        if step_rule == "bb_armijo":
            return _bb_armijo_ascent(
                z0, x_anchor, self._classifier_module, y,
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
        raise ValueError(f"Unsupported ppa_step_rule: {step_rule}")

    def _reassign(
        self,
        z_att: torch.Tensor,
        x_att: torch.Tensor,
        y_att: torch.Tensor,
        x_full: torch.Tensor,
        y_full: torch.Tensor,
        mask: torch.Tensor,
        *,
        global_pool: bool,
    ) -> Tuple[torch.Tensor, float, float]:
        """Within-class best-response reassignment of the attacked subset.

        With ``global_pool`` under DDP, the candidate pool is the union of
        every rank's attacked particles: full-batch-shaped tensors (equal
        across ranks) are all-gathered together with the attack mask, then
        filtered — so ranks with differently sized attacked subsets stay
        collective-compatible. The local anchors always have their own
        particle in the pool, preserving per-sample monotonicity.
        """
        cfg = self.config
        clf = self._classifier_module
        lam = float(cfg.lambda_param)
        use_margin = bool(cfg.use_margin_loss)
        transport_cost = str(getattr(cfg, "transport_cost", "normalized_mse"))

        with torch.no_grad():
            if z_att.size(0) > 0:
                loss_att = adversary_loss_per_sample(
                    clf(z_att), y_att, use_margin=use_margin
                )
            else:
                loss_att = torch.zeros(0, device=x_full.device)

        if global_pool and self.dist.is_distributed:
            z_pool_full = x_full.detach().clone()
            z_pool_full[mask] = z_att
            loss_full = torch.zeros(
                x_full.size(0), device=x_full.device, dtype=loss_att.dtype
            )
            loss_full[mask] = loss_att
            z_gathered = dist_helpers.all_gather_concat(z_pool_full)
            y_gathered = dist_helpers.all_gather_concat(y_full)
            loss_gathered = dist_helpers.all_gather_concat(loss_full)
            mask_gathered = dist_helpers.all_gather_concat(mask)
            z_pool = z_gathered[mask_gathered]
            y_pool = y_gathered[mask_gathered]
            pool_loss = loss_gathered[mask_gathered]
        else:
            z_pool, y_pool, pool_loss = z_att, y_att, loss_att

        z_new, _y_proj, gain, obj_scale, _active = free_weight_projection_images(
            z_att, x_att, y_att, clf, lam,
            use_margin=use_margin,
            transport_cost=transport_cost,
            z_pool=z_pool,
            y_pool=y_pool,
            pool_loss=pool_loss,
            z_loss=loss_att,
        )
        return z_new, float(gain), float(obj_scale)

    def _should_stop(
        self,
        round_idx: int,
        gain: float,
        obj_scale: float,
        n_att: int,
        *,
        sync: bool,
    ) -> bool:
        """Adaptive early stop on the (globally averaged) reassignment gain.

        Config-only guards run identically on every rank, so the collective
        below fires the same number of times everywhere. With
        ``ppa_min_rounds >= ppa_num_rounds`` the stop can never trigger and
        the schedule is exactly R fixed rounds (Algorithm 1).
        """
        cfg = self.config
        if round_idx < int(cfg.ppa_min_rounds):
            return False
        gain_sum = gain * n_att
        scale_sum = obj_scale * n_att
        count = float(n_att)
        if sync:
            stats = torch.tensor(
                [gain_sum, scale_sum, count],
                device=self.device,
                dtype=torch.float32,
            )
            stats = dist_helpers.all_reduce_sum_scalar(stats)
            gain_sum, scale_sum, count = (
                float(stats[0]), float(stats[1]), float(stats[2])
            )
        if count <= 0:
            return True
        gain_mean = gain_sum / count
        scale_mean = scale_sum / count
        return gain_mean <= float(cfg.ppa_gain_rtol) * max(scale_mean, 1e-12)

    def _mpa_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attack_mask: Optional[torch.Tensor],
        *,
        global_pool: bool,
    ) -> torch.Tensor:
        """Run the full R-round MPA inner loop; unattacked samples stay at x.

        Caller must hold ``frozen_module`` on the classifier. ``global_pool``
        must be False on rank-local-only paths (evaluation) to avoid
        collectives; during distributed training it is True so the
        reassignment pool is the global batch.
        """
        cfg = self.config
        B = x.size(0)
        if attack_mask is None:
            mask = torch.ones(B, dtype=torch.bool, device=x.device)
        else:
            mask = attack_mask.to(device=x.device, dtype=torch.bool)

        x_att = x[mask].detach()
        y_att = y[mask]
        z_att = x_att.clone()
        sync = bool(global_pool) and self.dist.is_distributed
        num_rounds = max(1, int(cfg.ppa_num_rounds))
        order = str(getattr(cfg, "ppa_round_order", "ascent_first")).lower()

        if order == "reassign_first":
            # Mirrored rounds: {reassign; K ascent steps} x R, closed by a
            # FINAL reassignment (R=3: R-A-R-A-R-A-R). The round-0
            # reassignment happens at z = x, i.e. each sample starts its
            # ascent from the best same-class TRAINING sample under the DRO
            # score, instead of from its own x; the trailing reassignment
            # ensures the outer theta-update sees within-class best
            # responses, same as ascent_first. Round 0 uses the round0 step
            # budget/LR schedule, later rounds the refine ones — identical
            # knobs to ascent_first. _should_stop's round_idx argument is the
            # number of COMPLETED rounds, so ppa_min_rounds guarantees the
            # same number of full rounds in both orders.
            for round_idx in range(num_rounds):
                z_att, gain, obj_scale = self._reassign(
                    z_att, x_att, y_att, x, y, mask, global_pool=global_pool
                )
                if self._should_stop(
                    round_idx, gain, obj_scale, int(z_att.size(0)), sync=sync
                ):
                    break
                z_att = self._ascend(
                    z_att, x_att, y_att,
                    int(cfg.ppa_round0_steps if round_idx == 0
                        else cfg.ppa_refine_steps),
                    round_idx=round_idx,
                )
            # Unconditional (collective counts stay uniform across ranks;
            # idempotent if the early stop already left z reassigned).
            z_att, _, _ = self._reassign(
                z_att, x_att, y_att, x, y, mask, global_pool=global_pool
            )
        elif order == "ascent_first":
            # Algorithm 1: {K ascent steps; reassign} x R.
            z_att = self._ascend(
                z_att, x_att, y_att, int(cfg.ppa_round0_steps), round_idx=0
            )
            for round_idx in range(1, num_rounds):
                z_att, gain, obj_scale = self._reassign(
                    z_att, x_att, y_att, x, y, mask, global_pool=global_pool
                )
                if self._should_stop(
                    round_idx, gain, obj_scale, int(z_att.size(0)), sync=sync
                ):
                    break
                z_att = self._ascend(
                    z_att, x_att, y_att, int(cfg.ppa_refine_steps),
                    round_idx=round_idx,
                )

            if num_rounds > 1:
                # Final reassignment closes round R of Algorithm 1. Skipped
                # for R=1, which the paper defines as plain PA — ascent only,
                # the reassignment steps stay disabled.
                z_att, _, _ = self._reassign(
                    z_att, x_att, y_att, x, y, mask, global_pool=global_pool
                )
        else:
            raise ValueError(f"Unsupported ppa_round_order: {order}")

        z_full = x.detach().clone()
        z_full[mask] = z_att
        return z_full.detach()

    # ------------------------------------------------------------------
    # Trainer contract
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
            z = self._mpa_attack(x, y, attack_mask, global_pool=True)
            with torch.no_grad():
                # Inner-objective diagnostic over the ATTACKED subset only —
                # unattacked samples sit at x with cost 0 and would dilute the
                # metric with clean loss the adversary never touched (matches
                # the masked-objective semantics NPF/NN-DRO log).
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
        """Fresh MPA attack with predicted labels (MPA is transductive).

        The adversary never sees ground truth at eval time: pseudo-labels
        ``argmax clf(x)`` stand in for y, so on clean-correct test points the
        attack objective coincides with the training-time one. On clean-
        MISCLASSIFIED points the attack targets the (wrong) prediction and
        can accidentally restore the true class, so the resulting adv_acc is
        an optimistic label-free metric — unlike the PGD columns, which use
        true labels with clean-correct gating. Compare transport adv_acc
        against clean_acc, and read benchmark robustness from input_pgd_acc.

        Runs rank-locally (``global_pool=False``, no collectives) because
        evaluation is rank-0-only. Cap the per-epoch cost with
        ``--eval-transport-samples`` (this is a full inner maximization per
        eval batch).
        """
        clf = self._classifier_module
        with frozen_module(clf, eval_mode=True):
            with torch.no_grad():
                y_hat = clf(x).argmax(dim=1)
            with torch.enable_grad():
                z = self._mpa_attack(x, y_hat, None, global_pool=False)
        return z.detach()
