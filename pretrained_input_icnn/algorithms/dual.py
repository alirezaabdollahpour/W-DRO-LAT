"""Sinkhorn SDRO dual / entropy-regularised WDRO baseline.

Two operating modes:

* **One-shot** (``dual_langevin_steps=0``): each batch is expanded with
  ``m = 2^level`` Gaussian draws around x, with ``level`` sampled by the
  randomized truncation rule. The loss is the closed-form entropic dual
  using the same lambda convention as the primal DRO trainers:

      L(θ) = 2λε · E_x[ logsumexp_j( ℓ(f_θ(x_j), y) / (2λε) ) − log m ],
      x_j = x + sqrt(ε) · ξ_j,  ξ_j ~ N(0, I).

  The samples are drawn from the *prior* (Gaussian noise), not from the
  Gibbs target, so the inner integral isn't actually solved.

* **Option D — Langevin / MALA** (``dual_langevin_steps > 0``): each
  particle is iteratively refined with a Langevin chain on the Gibbs
  target ``π(z|x) ∝ exp(U(z))`` where

      U(z) = ℓ(f_θ(z), y) / (2λε)  −  ‖z − x‖² / (2ε).

  The classifier sees normalized CIFAR tensors, but the Gaussian prior
  and squared distance above are interpreted in pixel coordinates [0, 1].

  The shared factor 2 makes the zero-temperature limit
  ``max_z ℓ(f_θ(z), y) - λ‖z - x‖²``, matching NPF / NN-DRO / WRM /
  WFR / New_PPA. The ½ factor in the quadratic term comes from the
  Gaussian log-density of the prior N(x, εI), which also matters for the
  MH ratio.

  This is the distribution that the Sinkhorn dual was always meant to
  integrate against — fixing the prior-vs-target mismatch in the legacy
  estimator.

  - Proposal:  z' = z + η · ∇_z U(z) + sqrt(2η) · ξ.
  - With ``--dual-mala`` (default), accept z' with probability
        α = min{1, exp[U(z') − U(z) + log q(z|z') − log q(z'|z)]},
    where q(z'|z) = N(z + η ∇U(z), 2η I). Proposals outside the valid
    normalized pixel box are rejected, which keeps the MH ratio valid
    for the box-constrained target. Otherwise (plain ULA), take z'
    unconditionally after clamping it to the box.
  - After K iterations (optional burn-in discarded), the *final*
    particles are plugged into the same-lambda Sinkhorn dual loss. Only
    the sample distribution changes between one-shot and Langevin modes.

DDP semantics: each rank's particles live on its own batch shard; no
cross-rank sync of the Langevin chain is meaningful (the Gibbs target
factorises over samples). The outer classifier_update is DDP-wrapped as
in every other algorithm.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from ..utils import (
    adversary_loss_per_sample,
    clamped_normalized_copy,
    frozen_module,
    normalized_pixel_bounds,
    pixel_l2_squared,
    set_requires_grad,
    to_normalized,
    to_pixel,
)
from .base import BaseAdvTrainer


class SDRODualTrainer(BaseAdvTrainer):
    name = "dual"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config
        self.epsilon = float(cfg.dual_epsilon)
        self.sample_level = max(1, int(cfg.dual_sample_level))
        self.langevin_steps = max(0, int(cfg.dual_langevin_steps))
        self.langevin_eta = float(cfg.dual_langevin_step_size)
        self.use_mala = bool(cfg.dual_mala)
        self.init_noise_scale = (
            float(cfg.dual_init_noise_scale)
            if cfg.dual_init_noise_scale is not None
            else math.sqrt(self.epsilon)
        )
        self.burn_in = max(0, int(cfg.dual_burn_in))
        if self.burn_in >= self.langevin_steps and self.langevin_steps > 0:
            raise ValueError(
                f"dual_burn_in ({self.burn_in}) must be < dual_langevin_steps "
                f"({self.langevin_steps}); otherwise the chain has no post-burn samples."
            )

    # ------------------------------------------------------------------
    # Inner sampler
    # ------------------------------------------------------------------
    def _expand_batch(
        self, x: torch.Tensor, y: torch.Tensor, m: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tile x and y to m copies per sample. Returns (x_rep, y_rep, z_init)."""
        x_rep = x.repeat_interleave(m, dim=0)
        y_rep = y.repeat_interleave(m, dim=0)
        z_init_pix = (to_pixel(x_rep) + self.init_noise_scale * torch.randn_like(x_rep)).clamp(
            0.0, 1.0
        )
        z_init = to_normalized(z_init_pix)
        return x_rep, y_rep, z_init

    def _U_and_grad(
        self,
        z: torch.Tensor,
        x_rep: torch.Tensor,
        y_rep: torch.Tensor,
        lam_eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-particle U(z) and ∇_z U(z). Detached on return.

        U(z) = primary(f(z), y) / (2λε)  −  ‖z − x‖² / (2ε)
        """
        z_in = z.detach().clone().requires_grad_(True)
        logits = self._classifier_module(z_in)
        ce = adversary_loss_per_sample(
            logits, y_rep, use_margin=bool(self.config.use_margin_loss)
        )
        sq = pixel_l2_squared(z_in, x_rep)
        U_per_particle = ce / lam_eps - 0.5 * sq / self.epsilon
        # autograd.grad wrt z_in returns dU_total/dz_in. We want
        # dU_per_particle / dz_in (per-particle drift). Since U_per_particle
        # is a sum of independent terms over the particle dim, summing and
        # taking grad gives exactly ∇_z U for each row.
        grad = torch.autograd.grad(U_per_particle.sum(), z_in, create_graph=False)[0]
        return U_per_particle.detach(), grad.detach()

    @staticmethod
    def _log_proposal_density(
        z_to: torch.Tensor, z_from: torch.Tensor, drift_from: torch.Tensor, eta: float
    ) -> torch.Tensor:
        """log q(z_to | z_from) = -‖z_to − z_from − η · drift_from‖² / (4η).

        Returned per particle (constants that cancel in the MH ratio omitted).
        """
        diff = (z_to - z_from - eta * drift_from).reshape(z_to.size(0), -1)
        return -diff.pow(2).sum(dim=1) / (4.0 * eta)

    @staticmethod
    def _inside_normalized_box(z: torch.Tensor) -> torch.Tensor:
        lower, upper = normalized_pixel_bounds(z)
        inside = (z >= lower) & (z <= upper)
        return inside.reshape(z.size(0), -1).all(dim=1)

    def _langevin_chain(
        self,
        z: torch.Tensor,
        x_rep: torch.Tensor,
        y_rep: torch.Tensor,
        lam_eps: float,
    ) -> Tuple[torch.Tensor, float]:
        """Run K Langevin (or MALA) iterations. Returns (z_final, mean accept rate)."""
        eta = self.langevin_eta
        n_accepted, n_proposed = 0, 0

        for step in range(self.langevin_steps):
            U_z, drift_z = self._U_and_grad(z, x_rep, y_rep, lam_eps)
            xi = torch.randn_like(z)
            z_prop = z + eta * drift_z + math.sqrt(2.0 * eta) * xi

            if self.use_mala:
                # Need U and drift at the proposal too — extra fwd+bwd.
                U_zp, drift_zp = self._U_and_grad(z_prop, x_rep, y_rep, lam_eps)
                log_q_fw = self._log_proposal_density(z_prop, z, drift_z, eta)
                log_q_bw = self._log_proposal_density(z, z_prop, drift_zp, eta)
                log_alpha = (U_zp - U_z) + (log_q_bw - log_q_fw)
                in_bounds = self._inside_normalized_box(z_prop)
                # Numerically stable accept test in log-space. Drop NaNs
                # or out-of-box proposals to "reject" — preferable to
                # silently changing the MH proposal by clamping it.
                log_alpha = torch.where(
                    torch.isfinite(log_alpha) & in_bounds,
                    log_alpha,
                    torch.full_like(log_alpha, -float("inf")),
                )
                log_u = torch.log(
                    torch.rand(z.size(0), device=z.device, dtype=z.dtype).clamp_min(1e-30)
                )
                accept = log_u < log_alpha
                # Keep z_prop on accept, z on reject — per particle.
                expand = accept.view(-1, *([1] * (z.dim() - 1)))
                z = torch.where(expand, z_prop, z)
                n_accepted += int(accept.sum().item())
                n_proposed += int(accept.numel())
            else:
                # Plain ULA: take the proposal unconditionally.
                z = clamped_normalized_copy(z_prop)

        accept_rate = (n_accepted / n_proposed) if n_proposed > 0 else float("nan")
        return z.detach(), accept_rate

    # ------------------------------------------------------------------
    # Outer step
    # ------------------------------------------------------------------
    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.config

        # m may be sampled stochastically (randomized truncation)
        # or fixed when langevin sampling is on — random m makes the
        # post-burn-in particle count vary mid-batch which complicates
        # MH bookkeeping for no benefit.
        if self.langevin_steps > 0:
            m = 2 ** self.sample_level
        else:
            levels = np.arange(self.sample_level + 1)
            numerators = 2.0 ** (-levels)
            denominator = 2.0 - 2.0 ** (-self.sample_level)
            probabilities = numerators / denominator
            sampled_level = int(np.random.choice(levels, p=probabilities))
            m = 2 ** sampled_level

        x_rep, y_rep, z = self._expand_batch(x, y, m)
        # Temperature 2 * lambda * epsilon: with Gaussian prior
        # exp(-||z-x||^2/(2epsilon)), the small-epsilon limit is
        # max_z primary(f(z), y) - lambda * ||z-x||^2.
        lam_eps = 2.0 * float(cfg.lambda_param) * self.epsilon

        # Freeze classifier during sampling — autograd through z still
        # works; we just don't accumulate gradients on classifier params.
        with frozen_module(self._classifier_module):
            accept_rate = float("nan")
            if self.langevin_steps > 0:
                # Optional burn-in: discard the leading post-init iterations
                # before the dual loss is computed. We split the chain into a
                # burn-in segment (run but ignore) and a sampling segment
                # (run; the final state is what the dual loss sees).
                if self.burn_in > 0:
                    # Run the burn-in by temporarily setting K to burn_in,
                    # then continue from the chain's current state.
                    full_steps = self.langevin_steps
                    try:
                        self.langevin_steps = self.burn_in
                        z, _ = self._langevin_chain(z, x_rep, y_rep, lam_eps)
                        self.langevin_steps = full_steps - self.burn_in
                        z, accept_rate = self._langevin_chain(z, x_rep, y_rep, lam_eps)
                    finally:
                        self.langevin_steps = full_steps
                else:
                    z, accept_rate = self._langevin_chain(z, x_rep, y_rep, lam_eps)

            z = z.detach()

            self._dual_z = z
            self._dual_y_rep = y_rep
            self._dual_m = m
            self._dual_lam_reg = lam_eps
            self._dual_accept_rate = accept_rate

            with torch.no_grad():
                logits = self._classifier_module(z)
                ce = nn.CrossEntropyLoss()(logits, y_rep)
                self._last_inner_loss = float(ce.item())
                primary_each = adversary_loss_per_sample(
                    logits, y_rep, use_margin=bool(cfg.use_margin_loss)
                )
                primary_view = primary_each.view(x.size(0), m)
                top = primary_view.argmax(dim=1, keepdim=True)
                z_view = z.view(x.size(0), m, *x.shape[1:])
                gather_idx = top.view(-1, 1, 1, 1, 1).expand(-1, 1, *x.shape[1:])
                return z_view.gather(1, gather_idx).squeeze(1).detach()

    def classifier_update(self, x_adv, y):
        # The actual outer loss IS the Sinkhorn dual on the (potentially
        # Langevin-refined) particles. ``x_adv`` from step() is just a
        # diagnostic — we use ``self._dual_z`` here.
        self.classifier.train()
        set_requires_grad(self.classifier, True)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.classifier(self._dual_z)
        residuals = adversary_loss_per_sample(
            logits,
            self._dual_y_rep,
            use_margin=bool(self.config.use_margin_loss),
        ) / max(self._dual_lam_reg, 1e-8)
        residual_matrix = residuals.view(-1, self._dual_m).T
        loss = (
            torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(self._dual_m))
            * self._dual_lam_reg
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.classifier.parameters() if p.requires_grad], max_norm=10.0
        )
        self.optimizer.step()
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == self._dual_y_rep).float().mean().item()
        return float(loss.item()), float(acc)

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        return x.detach()
