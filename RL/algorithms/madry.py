"""Madry-style RO adversary: projected L2 PGD in normalized xi-space."""
from __future__ import annotations

from typing import Optional

import torch

from algorithms.base import project_box
from config import InnerConfig
from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.common import make_generator
from utils.gradients import estimate_grad_f_wrt_xi_batch
from utils.xi_ops import f_values


class MadryROAdversary:
    """Projected-gradient adversary over a local L2 ball around each xi anchor.

    The ball is measured in the same normalized geometry used by the WDRO
    objectives: ``||xi - hat_xi||_M <= ro_epsilon`` with diagonal
    ``M = cfg.M_diag``. For the default xi boxes this is equivalent to running
    ordinary L2 PGD after mapping each coordinate to unit box scale, so
    parameters with different physical units receive comparable perturbations.
    """

    def __init__(self, cfg: InnerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.outer_uses_adv_only = True
        self.low = torch.tensor(cfg.xi_low, device=device, dtype=torch.float32)
        self.high = torch.tensor(cfg.xi_high, device=device, dtype=torch.float32)
        self.Mdiag = torch.tensor(cfg.M_diag, device=device, dtype=torch.float32)
        self.sqrtM = self.Mdiag.clamp_min(1e-12).sqrt()
        self.epsilon = float(cfg.ro_epsilon)
        self.pgd_steps = int(cfg.ro_pgd_steps)
        self.pgd_restarts = int(cfg.ro_pgd_restarts)
        step_size: Optional[float] = cfg.ro_pgd_step_size
        self.pgd_step_size = (
            float(step_size)
            if step_size is not None
            else self.epsilon / max(1, self.pgd_steps // 2)
        )

    def _delta_u(self, xi: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        return (xi - anchor) * self.sqrtM

    def _from_delta_u(self, anchor: torch.Tensor, delta_u: torch.Tensor) -> torch.Tensor:
        return anchor + delta_u / self.sqrtM

    def _project_l2_ball(self, delta_u: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(delta_u, dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(norm), self.epsilon / norm)
        return delta_u * scale

    def _project_feasible(self, xi: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        xi = project_box(xi, self.low, self.high)
        delta_u = self._project_l2_ball(self._delta_u(xi, anchor))
        xi = self._from_delta_u(anchor, delta_u)
        return project_box(xi, self.low, self.high)

    def _random_start(
        self,
        anchor: torch.Tensor,
        *,
        seed: int,
        restart_idx: int,
    ) -> torch.Tensor:
        if restart_idx == 0:
            return anchor.detach().clone()

        B, p = anchor.shape
        gen = make_generator(int(seed) + 7919 * restart_idx, anchor.device)
        noise = torch.randn(B, p, generator=gen, device=anchor.device, dtype=anchor.dtype)
        noise_norm = torch.linalg.norm(noise, dim=1, keepdim=True).clamp_min(1e-12)
        direction = noise / noise_norm
        radii = torch.rand(B, 1, generator=gen, device=anchor.device, dtype=anchor.dtype)
        # Uniform in the p-dimensional L2 ball in normalized coordinates.
        radii = radii.pow(1.0 / max(1, p))
        delta_u = self.epsilon * radii * direction
        return self._project_feasible(self._from_delta_u(anchor, delta_u), anchor)

    @torch.no_grad()
    def adversarial_xi(
        self,
        env_eval: VecEnvTorch,
        policy: PolicyNet,
        hat_xi: torch.Tensor,
        seed0: int = 0,
    ) -> torch.Tensor:
        if self.epsilon <= 0.0 or self.pgd_steps <= 0:
            return hat_xi.detach().clone()

        anchor = hat_xi.detach()
        selection_seed = int(seed0) + 10_000
        best_xi = anchor.clone()
        best_obj = f_values(
            env_eval, policy, anchor,
            n_episodes=int(self.cfg.fd_episodes),
            max_steps=int(self.cfg.fd_horizon),
            seed0=selection_seed,
            deterministic=True,
        )

        for restart_idx in range(max(1, self.pgd_restarts)):
            xi = self._random_start(anchor, seed=int(seed0), restart_idx=restart_idx)

            for step_idx in range(self.pgd_steps):
                grad_xi = estimate_grad_f_wrt_xi_batch(
                    env_eval, policy, xi,
                    eps=float(self.cfg.fd_eps),
                    n_episodes=int(self.cfg.fd_episodes),
                    max_steps=int(self.cfg.fd_horizon),
                    seed0=int(seed0) + 1000 * restart_idx + 10 * (step_idx + 1),
                    method=str(self.cfg.grad_method),
                )
                grad_u = grad_xi / self.sqrtM
                grad_norm = torch.linalg.norm(grad_u, dim=1, keepdim=True).clamp_min(1e-12)
                delta_u = self._delta_u(xi, anchor)
                delta_u = delta_u + self.pgd_step_size * grad_u / grad_norm
                delta_u = self._project_l2_ball(delta_u)
                xi = self._project_feasible(self._from_delta_u(anchor, delta_u), anchor)
                xi = torch.where(torch.isfinite(xi), xi, anchor)

            obj = f_values(
                env_eval, policy, xi,
                n_episodes=int(self.cfg.fd_episodes),
                max_steps=int(self.cfg.fd_horizon),
                seed0=selection_seed,
                deterministic=True,
            )
            improved = obj > best_obj
            best_xi = torch.where(improved.unsqueeze(1), xi, best_xi)
            best_obj = torch.where(improved, obj, best_obj)

        return best_xi.detach()
