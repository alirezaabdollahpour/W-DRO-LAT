"""NPF-style WDRO adversary (Vesseron & Cuturi, 2024).

    psi(u) = 0.5*mu*||u||^2
             + 0.5 u^T (diag(delta^2) + A^T A) u
             + a^T u + phi^{NN}(u),

with phi^{NN} a deep ICNN block that re-injects per-layer NPF quadratic
forms Q(u) = ||delta o u||^2 + ||Au||^2.

Initialization follows the paper exactly and is *not* a choice: the two
schemes are applied jointly.

  1. Principled (Hoedt-Klambauer LogNormal) draws are used for every
     non-negative weight (the hidden-to-hidden cascade {W_y^(l)} and the
     readout w_out). These are NEVER zeroed.
  2. Identity init zeros the remaining parameter groups so that
     T(u) = grad psi(u) approx u at t=0:
        - fixed strong-convexity base mu=1 with epsilon-scale PSD residual,
        - outer/output linear terms a = 0, b_out = 0,
        - input-to-hidden paths W_z^(l) = 0, b^(l) = 0,
        - per-layer quadratic injections at eps scale.

RL-specific: we apply the NPF potential in the UNCONSTRAINED latent
space u and decode via the sigmoid box bijection, exactly as the ICNN
adversary does, so xi_adv stays in [xi_low, xi_high] by construction.

Inner maximisation uses the same BB + Armijo line-search semantics as the
CIFAR-10 NPF implementation, with NPF-specific hyperparameters
(cfg.npf_eta, cfg.npf_bb_*) tuned to the same profile and adapted to the
RL latent-coordinate transport.
"""
from __future__ import annotations

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.npf import NPFInputConvexPotential, npf_T_omega
from models.policy import PolicyNet
from utils.bb_armijo import BBArmijoState, bb_armijo_step_params
from utils.common import freeze_params
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
            strong_convexity=cfg.npf_strong_convexity,
        ).to(device)
        # Identity init is applied on top of the principled (LogNormal) draws
        # already produced inside the constructor's non-negative layers.
        self.icnn.init_as_identity()

        # BB + Armijo state with NPF-specific knobs matching the CIFAR NPF
        # implementation. No optimizer-side weight decay or gradient clipping
        # is injected into the adversary step.
        self.bb_state = BBArmijoState.create(
            alpha0=cfg.npf_eta,
            alpha_min=cfg.npf_bb_alpha_min,
            alpha_max=cfg.npf_bb_alpha_max,
            ls_c=cfg.npf_bb_ls_c,
            ls_shrink=cfg.npf_bb_ls_shrink,
            ls_max_steps=cfg.npf_bb_ls_max_steps,
            reject_on_armijo_failure=True,
        )

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
        u_adv = npf_T_omega(hat_u, self.icnn, create_graph=create_graph)
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

        hat = hat_xi

        with torch.enable_grad(), freeze_params(policy):
            for k in range(int(self.cfg.K_npf)):
                seed_k = int(seed0 + 1000 * k)

                def psi_objective(create_graph: bool) -> torch.Tensor:
                    xi_local = self.T(hat, create_graph=create_graph)
                    J = evaluate_return_batch_pathwise(
                        env_eval, policy, xi_local,
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
            return self.T(hat, create_graph=False)
