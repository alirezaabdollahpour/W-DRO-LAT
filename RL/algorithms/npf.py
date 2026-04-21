"""NPF-style WDRO adversary (Vesseron & Cuturi, 2024).

Faithful to the MNIST_Cuturi and Least_Squares implementations:

    psi(u) = 0.5 u^T (diag(delta^2) + A^T A) u + a^T u + phi^{NN}(u),

with phi^{NN} a deep ICNN block that re-injects per-layer NPF quadratic
forms Q(u) = ||delta o u||^2 + ||Au||^2, non-negative dense layers with
Hoedt-Klambauer LogNormal init and NEGATIVE bias offset, and an
identity initialisation (init_as_identity) so that T(u) = grad psi(u) ≈ u
at t=0.

RL-specific: we apply the NPF potential in the UNCONSTRAINED latent
space u and decode via the sigmoid box bijection, exactly as the ICNN
adversary does, so xi_adv stays in [xi_low, xi_high] by construction.

Inner maximisation uses a persistent Adam over the NPF parameters for
K_npf steps per outer call.
"""
from __future__ import annotations

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.npf import NPFInputConvexPotential
from models.policy import PolicyNet
from utils.rollouts import evaluate_return_batch_pathwise


class NPFAdversary:
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
        self.icnn = NPFInputConvexPotential(
            input_dim=input_dim,
            hidden_sizes=cfg.npf_hidden_sizes,
            outer_rank=cfg.npf_outer_rank,
            inner_rank=cfg.npf_inner_rank,
            activation=cfg.npf_activation,
            elu_alpha=cfg.npf_elu_alpha,
            softplus_beta=cfg.npf_softplus_beta,
            init_eps=cfg.npf_init_eps,
        ).to(device)
        self.icnn.init_as_identity()

        # Persistent Adam over the NPF parameters (matches MNIST/LS persistence).
        self.opt = torch.optim.Adam(self.icnn.parameters(), lr=float(cfg.lr_npf))

    # -------- box bijection (latent u <-> physical xi) --------

    def _encode(self, xi: torch.Tensor) -> torch.Tensor:
        p = (xi - self.low) * self._inv_range
        p = torch.clamp(p, self._box_eps, 1.0 - self._box_eps)
        return torch.log(p) - torch.log1p(-p)

    def _decode(self, u: torch.Tensor) -> torch.Tensor:
        return self.low + self._range * torch.sigmoid(u)

    # -------- transport map --------

    def T(self, hat_xi: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """xi_adv = decode(grad_u psi(encode(hat_xi)))."""
        hat_u = self._encode(hat_xi)
        with torch.set_grad_enabled(True):
            u_in = hat_u.clone().detach().requires_grad_(True)
            phi = self.icnn(u_in)
            u_adv = torch.autograd.grad(phi.sum(), u_in, create_graph=create_graph)[0]
        u_adv = u_adv.view_as(hat_xi)
        u_adv = torch.nan_to_num(u_adv, nan=0.0, posinf=0.0, neginf=0.0)
        xi_adv = self._decode(u_adv)
        xi_adv = torch.where(torch.isfinite(xi_adv), xi_adv, hat_xi)
        if not create_graph:
            xi_adv = xi_adv.detach()
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
                "NPF exact inner optimization requires --grad-method pathwise (autograd)."
            )

        with torch.enable_grad():
            for k in range(int(self.cfg.K_npf)):
                seed_k = int(seed0 + 1000 * k)
                xi_local = self.T(hat_xi, create_graph=True)

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
            return self.T(hat_xi, create_graph=False)
