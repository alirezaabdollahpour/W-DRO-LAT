"""New_PPA: Free-weight Projected Particle Ascent with objective-gain stopping criterion."""
from __future__ import annotations

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.xi_ops import f_values, wrm_ascent_xi, wrm_ascent_xi_const_lr


class NewPPAAdversary:
    """Multi-round ascent that stops early when the adversary-objective gain
    per round falls below a relative tolerance (cf. MNIST_Cuturi new_ppa).
    """

    def __init__(self, cfg: InnerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.low = torch.tensor(cfg.xi_low, device=device, dtype=torch.float32)
        self.high = torch.tensor(cfg.xi_high, device=device, dtype=torch.float32)
        self.Mdiag = torch.tensor(cfg.M_diag, device=device, dtype=torch.float32)

    def _objective(self, xi: torch.Tensor, hat_xi: torch.Tensor, env, policy, seed0: int) -> float:
        """Mean  f(θ, xi) - λ ||xi - hat_xi||_M^2 (deterministic stochastic-return estimate)."""
        f = f_values(
            env, policy, xi,
            n_episodes=int(self.cfg.fd_episodes),
            max_steps=int(self.cfg.fd_horizon),
            seed0=seed0,
            deterministic=True,
        )
        diff = xi - hat_xi
        cost = (diff * diff * self.Mdiag).sum(dim=-1)
        return float((f - float(self.cfg.lam) * cost).mean().item())

    def _clip(self, xi: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(xi, self.high), self.low)

    def adversarial_xi(
        self,
        env_eval: VecEnvTorch,
        policy: PolicyNet,
        hat_xi: torch.Tensor,
        seed0: int = 0,
    ) -> torch.Tensor:
        z = wrm_ascent_xi(
            hat_xi,
            env_eval, policy,
            Mdiag=self.Mdiag, low=self.low, high=self.high,
            lambda_reg=float(self.cfg.lam),
            num_steps=int(self.cfg.new_ppa_inner_steps_round0),
            lr=float(self.cfg.new_ppa_inner_lr_round0),
            n_episodes=int(self.cfg.fd_episodes),
            max_steps=int(self.cfg.fd_horizon),
            seed0=int(seed0),
        )
        prev_obj = self._objective(z, hat_xi, env_eval, policy, seed0=int(seed0) + 1)

        for round_idx in range(1, int(self.cfg.new_ppa_num_rounds)):
            z_proj = self._clip(z)
            obj = self._objective(z_proj, hat_xi, env_eval, policy, seed0=int(seed0) + 2 * round_idx)
            gain = obj - prev_obj
            obj_scale = max(abs(obj), 1e-12)

            if (
                round_idx >= int(self.cfg.new_ppa_min_rounds)
                and gain <= float(self.cfg.new_ppa_gain_rtol) * obj_scale
            ):
                z = z_proj
                break

            z = wrm_ascent_xi_const_lr(
                z_proj, hat_xi,
                env_eval, policy,
                Mdiag=self.Mdiag, low=self.low, high=self.high,
                lambda_reg=float(self.cfg.lam),
                num_steps=int(self.cfg.new_ppa_refine_steps),
                lr=float(self.cfg.new_ppa_refine_lr),
                n_episodes=int(self.cfg.fd_episodes),
                max_steps=int(self.cfg.fd_horizon),
                seed0=int(seed0) + 10_000 * round_idx,
            )
            prev_obj = self._objective(z, hat_xi, env_eval, policy, seed0=int(seed0) + 3 * round_idx)

        return self._clip(z).detach()
