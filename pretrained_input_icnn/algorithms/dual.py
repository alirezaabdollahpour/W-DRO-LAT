"""Sinkhorn SDRO dual / entropy-regularised WDRO baseline.

Two operating modes:

* **Legacy one-shot** (``dual_langevin_steps=0``): mirrors
  ``Logistic_Regression_CIFAR10/algorithms/dual.py``. Each batch is
  expanded with m=2^sample_level Gaussian draws around x; the loss is
  the closed-form entropic dual

      L(θ) = λε · E_x[ logsumexp_j( ℓ(f_θ(x_j), y) / (λε) ) − log m ],
      x_j = x + sqrt(ε) · ξ_j,  ξ_j ~ N(0, I).

  The samples are drawn from the *prior* (Gaussian noise), not from the
  Gibbs target, so the inner integral isn't actually solved.

* **Option D — Langevin / MALA** (``dual_langevin_steps > 0``): each
  particle is iteratively refined with a Langevin chain on the Gibbs
  target ``π(z|x) ∝ exp(U(z))`` where

      U(z) = ℓ(f_θ(z), y) / (λε)  −  ‖z − x‖² / (2ε).

  (The ½ factor comes from the Gaussian log-density of the prior
  N(x, εI); the entropic dual integrates against that exact prior, so
  getting the factor right matters for the MH ratio.)

  This is the distribution that the Sinkhorn dual was always meant to
  integrate against — fixing the prior-vs-target mismatch in the legacy
  estimator.

  - Proposal:  z' = z + η · ∇_z U(z) + sqrt(2η) · ξ
  - With ``--dual-mala`` (default), accept z' with probability
        α = min{1, exp[U(z') − U(z) + log q(z|z') − log q(z'|z)]},
    where q(z'|z) = N(z + η ∇U(z), 2η I). Otherwise (plain ULA) take z'
    unconditionally.
  - After K iterations (optional burn-in discarded), the *final*
    particles are plugged into the SAME Sinkhorn dual loss as the legacy
    mode. Only the sample distribution is upgraded.

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

from ..utils import clamped_normalized_copy, set_requires_grad
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
        z_init = clamped_normalized_copy(
            x_rep + self.init_noise_scale * torch.randn_like(x_rep)
        )
        return x_rep, y_rep, z_init

    def _U_and_grad(
        self,
        z: torch.Tensor,
        x_rep: torch.Tensor,
        y_rep: torch.Tensor,
        lam_eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-particle U(z) and ∇_z U(z). Detached on return.

        U(z) = CE(f(z), y) / (λε)  −  ‖z − x‖² / (2ε)
        """
        z_in = z.detach().clone().requires_grad_(True)
        logits = self._classifier_module(z_in)
        ce = nn.CrossEntropyLoss(reduction="none")(logits, y_rep)
        sq = (z_in - x_rep).reshape(z_in.size(0), -1).pow(2).sum(dim=1)
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
            z_prop = clamped_normalized_copy(z_prop)

            if self.use_mala:
                # Need U and drift at the proposal too — extra fwd+bwd.
                U_zp, drift_zp = self._U_and_grad(z_prop, x_rep, y_rep, lam_eps)
                log_q_fw = self._log_proposal_density(z_prop, z, drift_z, eta)
                log_q_bw = self._log_proposal_density(z, z_prop, drift_zp, eta)
                log_alpha = (U_zp - U_z) + (log_q_bw - log_q_fw)
                # Numerically stable accept test in log-space. Drop NaNs
                # (e.g. if the classifier produced inf logits at z_prop)
                # to "reject" — preferable to crashing the chain.
                log_alpha = torch.where(
                    torch.isfinite(log_alpha),
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
                z = z_prop

        accept_rate = (n_accepted / n_proposed) if n_proposed > 0 else float("nan")
        return z.detach(), accept_rate

    # ------------------------------------------------------------------
    # Outer step
    # ------------------------------------------------------------------
    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.config

        # m may be sampled stochastically (legacy randomized truncation)
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
        lam_eps = float(cfg.lambda_param) * self.epsilon

        # Freeze classifier during sampling — autograd through z still
        # works; we just don't accumulate gradients on classifier params.
        self._classifier_module.eval()
        set_requires_grad(self._classifier_module, False)

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
                self.langevin_steps = self.burn_in
                z, _ = self._langevin_chain(z, x_rep, y_rep, lam_eps)
                self.langevin_steps = full_steps - self.burn_in
                z, accept_rate = self._langevin_chain(z, x_rep, y_rep, lam_eps)
                self.langevin_steps = full_steps
            else:
                z, accept_rate = self._langevin_chain(z, x_rep, y_rep, lam_eps)

        set_requires_grad(self._classifier_module, True)
        z = z.detach()

        self._dual_z = z
        self._dual_y_rep = y_rep
        self._dual_m = m
        self._dual_lam_reg = lam_eps
        self._dual_accept_rate = accept_rate

        with torch.no_grad():
            ce = nn.CrossEntropyLoss()(self._classifier_module(z), y_rep)
            self._last_inner_loss = float(ce.item())
            ce_each = nn.CrossEntropyLoss(reduction="none")(
                self._classifier_module(z), y_rep
            )
            ce_view = ce_each.view(x.size(0), m)
            top = ce_view.argmax(dim=1, keepdim=True)
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
        criterion = nn.CrossEntropyLoss(reduction="none")
        residuals = criterion(logits, self._dual_y_rep) / max(self._dual_lam_reg, 1e-8)
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
