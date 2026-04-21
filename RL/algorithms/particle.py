"""Particle ascent adversary (batched gradient ascent on the regularized inner objective)."""
from __future__ import annotations

import torch

from algorithms.base import cost_grad_xi, project_box
from config import InnerConfig
from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.gradients import estimate_grad_f_wrt_xi_batch


class ParticleAdversary:
    """Vectorized particle ascent on

        max_xi  f(theta, xi) - lam * ||xi - hat||_M^2,

    with f = -J (return) and diagonal Mahalanobis metric M.
    """

    def __init__(self, cfg: InnerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.low = torch.tensor(cfg.xi_low, device=device, dtype=torch.float32)
        self.high = torch.tensor(cfg.xi_high, device=device, dtype=torch.float32)
        self.Mdiag = torch.tensor(cfg.M_diag, device=device, dtype=torch.float32)

    @torch.no_grad()
    def adversarial_xi(
        self,
        env_eval: VecEnvTorch,
        policy: PolicyNet,
        hat_xi: torch.Tensor,
        seed0: int = 0,
    ) -> torch.Tensor:
        """Particle ascent:  xi <- xi + eta (∇_xi f - lam * ∇_xi c)."""
        xi = hat_xi.clone()

        for k in range(self.cfg.K_part):
            g = estimate_grad_f_wrt_xi_batch(
                env_eval, policy, xi,
                eps=self.cfg.fd_eps,
                n_episodes=self.cfg.fd_episodes,
                max_steps=self.cfg.fd_horizon,
                seed0=seed0 + 1000 * k,
                method=self.cfg.grad_method,
            )
            cost_g = cost_grad_xi(xi, hat_xi, self.Mdiag)
            xi = xi + self.cfg.eta_part * (g - self.cfg.lam * cost_g)
            xi = torch.where(torch.isfinite(xi), xi, hat_xi)
            xi = project_box(xi, self.low, self.high)

        return xi
