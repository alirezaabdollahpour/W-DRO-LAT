"""NN-DRO adversary: vanilla-MLP inner maximiser (no gradient-of-potential).

The adversary parametrises the transport directly with a plain MLP applied in
the logit-latent space (the same box bijection as ICNN/NPF, so xi_adv stays in
[xi_low, xi_high] by construction). Unlike ICNN/NPF, no autograd is taken
through the network's input; the MLP's output IS the displacement. Inner
maximisation uses a persistent Adam optimiser over the MLP parameters.

Inner objective (maximise):
    E[ J(policy, T(hat_xi)) - lam * ||T(hat_xi) - hat_xi||_M^2 ].
"""
from __future__ import annotations

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.nn_dro import MLPAdversary
from models.policy import PolicyNet
from utils.rollouts import evaluate_return_batch_pathwise


class NNDROAdversary:
    def __init__(self, cfg: InnerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.low = torch.tensor(cfg.xi_low, device=device, dtype=torch.float32)
        self.high = torch.tensor(cfg.xi_high, device=device, dtype=torch.float32)
        self.Mdiag = torch.tensor(cfg.M_diag, device=device, dtype=torch.float32)

        self._box_eps = 1e-6
        self._range = (self.high - self.low).clamp_min(1e-12)
        self._inv_range = 1.0 / self._range

        input_dim = int(self.low.numel())
        self.mlp = MLPAdversary(
            input_dim=input_dim,
            hidden_sizes=cfg.nn_dro_hidden_sizes,
            activation=cfg.nn_dro_activation,
            softplus_beta=cfg.nn_dro_softplus_beta,
            init_scale=cfg.nn_dro_init_scale,
        ).to(device)

        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=float(cfg.lr_nn_dro))

    # -------- box bijection (latent u <-> physical xi) --------

    def _encode(self, xi: torch.Tensor) -> torch.Tensor:
        p = (xi - self.low) * self._inv_range
        p = torch.clamp(p, self._box_eps, 1.0 - self._box_eps)
        return torch.log(p) - torch.log1p(-p)

    def _decode(self, u: torch.Tensor) -> torch.Tensor:
        return self.low + self._range * torch.sigmoid(u)

    # -------- transport map (no autograd through input) --------

    def T(self, hat_xi: torch.Tensor) -> torch.Tensor:
        """xi_adv = decode(encode(hat_xi) + mlp(encode(hat_xi)))."""
        hat_u = self._encode(hat_xi).detach()
        delta = self.mlp(hat_u)
        u_adv = hat_u + delta
        u_adv = torch.nan_to_num(u_adv, nan=0.0, posinf=0.0, neginf=0.0)
        xi_adv = self._decode(u_adv)
        xi_adv = torch.where(torch.isfinite(xi_adv), xi_adv, hat_xi)
        return xi_adv

    def adversarial_xi(
        self,
        env_eval: VecEnvTorch,
        policy: PolicyNet,
        hat_xi: torch.Tensor,
        seed0: int = 0,
    ) -> torch.Tensor:
        if str(self.cfg.grad_method).lower() not in ("pathwise", "autograd"):
            raise ValueError(
                "NN-DRO exact inner optimization requires --grad-method pathwise (autograd)."
            )

        hat_xi = hat_xi.detach()

        with torch.enable_grad():
            for k in range(int(self.cfg.K_nn_dro)):
                seed_k = int(seed0 + 1000 * k)
                xi_local = self.T(hat_xi)

                J = evaluate_return_batch_pathwise(
                    env_eval, policy, xi_local,
                    n_episodes=int(self.cfg.fd_episodes),
                    max_steps=int(self.cfg.fd_horizon),
                    seed0=seed_k,
                )
                f = -J

                diff = xi_local - hat_xi
                cost = (diff * diff * self.Mdiag).sum(dim=-1)

                obj = (f - float(self.cfg.lam) * cost).mean()
                obj = torch.nan_to_num(obj, nan=-1e9, posinf=-1e9, neginf=-1e9)

                self.opt.zero_grad(set_to_none=True)
                (-obj).backward()
                self.opt.step()

        with torch.no_grad():
            return self.T(hat_xi).detach()
