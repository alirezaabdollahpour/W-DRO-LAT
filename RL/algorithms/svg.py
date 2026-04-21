"""SVGD (Stein Variational Gradient Descent) WDRO adversary in xi-space.

Each hat_xi spawns S particles that are pushed by the Stein kernel-smoothed
ascent on f(θ, xi) - λ ||xi - hat||_M^2, with AdaGrad-style adaptive step sizes.
"""
from __future__ import annotations

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.common import make_generator
from utils.xi_ops import grad_f_xi, repeat_xi


def _rbf_kernel(particles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RBF kernel and its gradient wrt the first argument.

    particles: (B, S, p).
    Returns:
      K  : (B, S, S)
      grad_K_x: (B, S, S, p)    -- grad of K[i, j] wrt particles[:, i, :]
    """
    B, S, p = particles.shape
    diff = particles.unsqueeze(2) - particles.unsqueeze(1)           # (B, S, S, p)
    sq_dists = (diff * diff).sum(dim=-1)                             # (B, S, S)
    med = torch.median(sq_dists.reshape(B, -1), dim=1).values
    h = med / max(1.0, float(torch.log(torch.tensor(float(S) + 1.0)).item()))
    h = h.clamp_min(1e-6).view(B, 1, 1)
    K = torch.exp(-sq_dists / h)                                     # (B, S, S)
    grad_K_x = -2.0 * diff / h.unsqueeze(-1) * K.unsqueeze(-1)       # (B, S, S, p)
    return K, grad_K_x


class SVGAdversary:
    def __init__(self, cfg: InnerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.low = torch.tensor(cfg.xi_low, device=device, dtype=torch.float32)
        self.high = torch.tensor(cfg.xi_high, device=device, dtype=torch.float32)
        self.Mdiag = torch.tensor(cfg.M_diag, device=device, dtype=torch.float32)

    def _clip(self, xi: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(xi, self.high), self.low)

    def adversarial_xi(
        self,
        env_eval: VecEnvTorch,
        policy: PolicyNet,
        hat_xi: torch.Tensor,
        seed0: int = 0,
    ) -> torch.Tensor:
        B, p = hat_xi.shape
        S = int(self.cfg.particle_num_samples)
        inner_steps = int(self.cfg.particle_inner_steps)
        inner_lr = float(self.cfg.particle_inner_lr)
        epsilon = float(self.cfg.particle_epsilon)
        lam = float(self.cfg.lam)
        decay = float(self.cfg.svg_adagrad_hist_decay)

        anchor = hat_xi.detach().unsqueeze(1).expand(B, S, p).contiguous()
        gen = make_generator(int(seed0) + 1, hat_xi.device)
        init_noise = 0.1 * torch.randn(anchor.shape, generator=gen, device=hat_xi.device, dtype=hat_xi.dtype)
        particles = self._clip((anchor.reshape(B * S, p) + init_noise.reshape(B * S, p))).view(B, S, p)
        hist_grad = torch.zeros_like(particles)

        for s in range(inner_steps):
            xi_flat = particles.reshape(B * S, p)
            g_f = grad_f_xi(
                env_eval, policy, xi_flat,
                n_episodes=int(self.cfg.fd_episodes),
                max_steps=int(self.cfg.fd_horizon),
                seed0=int(seed0) + 11 * (s + 1),
            )
            cost_g = 2.0 * (xi_flat - anchor.reshape(B * S, p)) * self.Mdiag
            score = (g_f - lam * cost_g) / max(lam * epsilon, 1e-8)
            score = score.view(B, S, p)

            K, grad_K_x = _rbf_kernel(particles)                       # (B,S,S), (B,S,S,p)
            K_score = torch.einsum("bij,bjp->bip", K, score)           # (B, S, p)
            sum_grad_K = grad_K_x.sum(dim=2)                            # (B, S, p)
            svg_grad = (K_score + sum_grad_K) / float(S)

            with torch.no_grad():
                hist_grad = decay * hist_grad + (1.0 - decay) * (svg_grad ** 2)
                adj = svg_grad / (1e-6 + torch.sqrt(hist_grad))
                particles = particles + inner_lr * adj
                particles = self._clip(particles.reshape(B * S, p)).view(B, S, p)

        return particles.reshape(B * S, p).detach()
