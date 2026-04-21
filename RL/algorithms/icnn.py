"""ICNN/Brenier-map adversary: map optimisation on the exact regularized inner WDRO objective.

Implementation details (preserved exactly from the prior RL_minimal.py):
  * Box constraint xi ∈ [xi_low, xi_high] enforced via a smooth sigmoid bijection
    (no hard clamp, which would be many-to-one and cause collapse).
  * BB + Armijo line search over ICNN parameters ψ.
  * Pathwise gradients through a differentiable surrogate return J̃.
"""
from __future__ import annotations

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.icnn import ICNN, initialize_icnn_identity
from models.policy import PolicyNet
from utils.bb_armijo import BBArmijoState, bb_armijo_step_params
from utils.rollouts import evaluate_return_batch_pathwise


class ICNNAdversary:
    """Transport-map adversary:  xi_adv = T_ψ(hat_xi) = ∇φ_ψ(hat_xi)."""

    def __init__(self, cfg: InnerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.low = torch.tensor(cfg.xi_low, device=device, dtype=torch.float32)
        self.high = torch.tensor(cfg.xi_high, device=device, dtype=torch.float32)
        self.Mdiag = torch.tensor(cfg.M_diag, device=device, dtype=torch.float32)

        # Box-squash helpers (work in latent R^d; map back to (low, high)^d).
        self._box_eps = 1e-6
        self._range = (self.high - self.low).clamp_min(1e-12)
        self._inv_range = 1.0 / self._range

        input_dim = int(self.low.numel())
        self.icnn = ICNN(
            input_dim=input_dim,
            hidden_sizes=cfg.icnn_hidden_sizes,
            activation=cfg.icnn_activation,
            strong_convexity=cfg.icnn_strong_convexity,
            nonneg_init=cfg.icnn_nonneg_init,
            softplus_beta=cfg.icnn_softplus_beta,
        ).to(device)
        if cfg.icnn_init.lower() == "identity":
            initialize_icnn_identity(self.icnn, strong_convexity=cfg.icnn_strong_convexity)

        self.bb_state = BBArmijoState.create(
            alpha0=cfg.eta_icnn,
            alpha_min=cfg.bb_alpha_min,
            alpha_max=cfg.bb_alpha_max,
            ls_c=cfg.bb_ls_c,
            ls_shrink=cfg.bb_ls_shrink,
            ls_max_steps=cfg.bb_ls_max_steps,
        )

    def _encode_box(self, xi: torch.Tensor) -> torch.Tensor:
        """Map physical xi ∈ (low, high) to latent u ∈ R via elementwise logit."""
        p = (xi - self.low) * self._inv_range
        p = torch.clamp(p, self._box_eps, 1.0 - self._box_eps)
        return torch.log(p) - torch.log1p(-p)

    def _decode_box(self, u: torch.Tensor) -> torch.Tensor:
        """Map latent u ∈ R to physical xi ∈ (low, high) via elementwise sigmoid."""
        p = torch.sigmoid(u)
        return self.low + self._range * p

    def T_latent(self, hat_u: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """Brenier map in latent coordinates: u_adv = ∇φ_ψ(hat_u)."""
        with torch.set_grad_enabled(True):
            u_in = hat_u.clone().detach().requires_grad_(True)
            phi = self.icnn(u_in)
            u_adv = torch.autograd.grad(
                outputs=phi.sum(),
                inputs=u_in,
                create_graph=create_graph,
            )[0]
        u_adv = u_adv.view_as(hat_u)
        u_adv = torch.nan_to_num(u_adv, nan=0.0, posinf=0.0, neginf=0.0)
        if not create_graph:
            u_adv = u_adv.detach()
        return u_adv

    def T(self, hat_xi: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """Full transport map in physical coordinates with exact box feasibility."""
        hat_u = self._encode_box(hat_xi)
        u_adv = self.T_latent(hat_u, create_graph=create_graph)
        xi_adv = self._decode_box(u_adv)
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
        """Inner maximization over the ICNN parameters ψ.

            max_ψ  E[ f(theta, T_ψ(hat_xi)) - lam * ||T_ψ(hat_xi) - hat_xi||_M^2 ].
        """
        hat = hat_xi

        gm = str(self.cfg.grad_method).lower()
        if gm not in ("pathwise", "autograd"):
            raise ValueError(
                "ICNN exact inner optimization requires --grad-method pathwise (autograd). "
                "For non-differentiable objectives (spsa/fd), you need a parameter-space "
                "zeroth-order optimizer over ICNN weights, which is intentionally not enabled here."
            )

        with torch.enable_grad():
            for k in range(int(self.cfg.K_icnn)):
                seed_k = int(seed0 + 1000 * k)

                def psi_objective(create_graph: bool) -> torch.Tensor:
                    xi_local = self.T(hat, create_graph=create_graph)

                    J = evaluate_return_batch_pathwise(
                        env_eval,
                        policy,
                        xi_local,
                        n_episodes=int(self.cfg.fd_episodes),
                        max_steps=int(self.cfg.fd_horizon),
                        seed0=seed_k,
                    )
                    f = -J

                    diff = xi_local - hat
                    cost = (diff * diff * self.Mdiag).sum(dim=-1)

                    obj = (f - float(self.cfg.lam) * cost).mean()
                    obj = torch.nan_to_num(obj, nan=-1e9, posinf=-1e9, neginf=-1e9)
                    return obj

                _params, self.bb_state, _fval, _gnorm = bb_armijo_step_params(
                    self.icnn.parameters(),
                    psi_objective,
                    self.bb_state,
                )

        with torch.no_grad():
            xi_final = self.T(hat, create_graph=False)

        return xi_final
