"""MPA (Multi-start Particle Ascent) — Algorithm 1 in the paper.

Per round r in [R]:
  1. K steps of WRM ascent anchored at hat_xi  (PA inner loop).
  2. Batch-wide reassignment (highlighted in projblue in the paper algo box):
        Z = {z_j}_{j=1}^B
        z_i <- argmax_{z in Z}  f(theta, z) - lambda * ||z - hat_z_i||_M^2,  for all i.

R = 1 recovers PA (no reassignment); R > 1 activates the MPA reassignment
between rounds. The reassignment closes the Monge gap on the batch by
repairing the cyclical-monotonicity that PA breaks. Cost is O(B^2) per
round in tensor arithmetic plus B policy rollouts to score the pool.

We additionally retain the engineering early-stop from the prior version
of this file: rounds beyond ``new_ppa_min_rounds`` are skipped if the
mean penalized objective stops improving by more than
``new_ppa_gain_rtol`` (relative). This is an engineering knob and not
part of the paper algorithm; set ``new_ppa_min_rounds = new_ppa_num_rounds``
to disable.
"""
from __future__ import annotations

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.xi_ops import f_values, wrm_ascent_xi, wrm_ascent_xi_const_lr


class NewPPAAdversary:
    def __init__(self, cfg: InnerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.low = torch.tensor(cfg.xi_low, device=device, dtype=torch.float32)
        self.high = torch.tensor(cfg.xi_high, device=device, dtype=torch.float32)
        self.Mdiag = torch.tensor(cfg.M_diag, device=device, dtype=torch.float32)

    def _clip(self, xi: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(xi, self.high), self.low)

    def _objective(
        self,
        xi: torch.Tensor,
        hat_xi: torch.Tensor,
        env: VecEnvTorch,
        policy: PolicyNet,
        seed0: int,
    ) -> float:
        """Mean penalized adversary objective  f(theta, xi) - lambda * ||xi - hat_xi||_M^2.

        Used only for the optional early-stop check; not part of the paper
        algorithm. Note ``f`` here is ``-J`` (the adversary maximizes ``f``).
        """
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

    def _reassign_pool(
        self,
        z: torch.Tensor,
        hat_xi: torch.Tensor,
        env: VecEnvTorch,
        policy: PolicyNet,
        seed0: int,
    ) -> torch.Tensor:
        """Batch-wide reassignment step (highlighted line in Algorithm 1):

            Z = {z_j}_{j=1}^B
            z_i <- argmax_{j in [B]}  f(theta, z_j) - lambda * ||z_j - hat_z_i||_M^2.

        ``z`` is the pool — same particles for every anchor — so we evaluate
        ``f`` at the B pool points *once* and reuse the values across all B
        anchors. The Mahalanobis-distance term varies per anchor.

        Returns a fresh tensor (the chosen pool entries gathered into the
        anchor order). Two anchors that select the same pool point produce
        equal rows — this is the transductive coupling MPA induces on the
        batch and is intentional.
        """
        B = z.shape[0]
        if B == 0:
            return z.detach().clone()

        with torch.no_grad():
            # f_pool[j] = f(theta, z_j) — one batched policy rollout, no anchor info.
            f_pool = f_values(
                env, policy, z,
                n_episodes=int(self.cfg.fd_episodes),
                max_steps=int(self.cfg.fd_horizon),
                seed0=int(seed0),
                deterministic=True,
            )
            # D[i, j] = ||z_j - hat_z_i||_M^2  — Mahalanobis sq-distance, B x B.
            # diff[i, j, :] = z[j] - hat_xi[i].  M-cost reduces along the last dim.
            diff = z.unsqueeze(0) - hat_xi.unsqueeze(1)
            D = (diff * diff * self.Mdiag).sum(dim=-1)
            # Score[i, j] = f_pool[j] - lambda * D[i, j]
            scores = f_pool.unsqueeze(0) - float(self.cfg.lam) * D
            best_j = scores.argmax(dim=1)
        # Gather pool entries into the anchor-indexed order. Detach the pool
        # to break any lingering autograd graph (round 0 returns detached
        # tensors anyway, but we make this explicit for safety).
        return z.detach()[best_j].clone()

    def adversarial_xi(
        self,
        env_eval: VecEnvTorch,
        policy: PolicyNet,
        hat_xi: torch.Tensor,
        seed0: int = 0,
    ) -> torch.Tensor:
        # --- Round 0: K steps of PA / WRM ascent anchored at hat_xi ---
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
        # MPA reassignment after round 0 (active iff new_ppa_num_rounds > 1,
        # matching the paper's R > 1 condition).
        if int(self.cfg.new_ppa_num_rounds) > 1:
            z = self._reassign_pool(
                z, hat_xi, env_eval, policy,
                seed0=int(seed0) + 4_242,
            )

        prev_obj = self._objective(z, hat_xi, env_eval, policy, seed0=int(seed0) + 1)

        # --- Rounds 1..R-1: refine + reassign ---
        for round_idx in range(1, int(self.cfg.new_ppa_num_rounds)):
            z_refine = wrm_ascent_xi_const_lr(
                z, hat_xi,
                env_eval, policy,
                Mdiag=self.Mdiag, low=self.low, high=self.high,
                lambda_reg=float(self.cfg.lam),
                num_steps=int(self.cfg.new_ppa_refine_steps),
                lr=float(self.cfg.new_ppa_refine_lr),
                n_episodes=int(self.cfg.fd_episodes),
                max_steps=int(self.cfg.fd_horizon),
                seed0=int(seed0) + 10_000 * round_idx,
            )
            z_refine = self._reassign_pool(
                z_refine, hat_xi, env_eval, policy,
                seed0=int(seed0) + 10_000 * round_idx + 4_242,
            )

            # Optional engineering early-stop: per-round mean objective gain.
            obj = self._objective(
                z_refine, hat_xi, env_eval, policy,
                seed0=int(seed0) + 2 * round_idx,
            )
            gain = obj - prev_obj
            obj_scale = max(abs(obj), 1e-12)
            if (
                round_idx >= int(self.cfg.new_ppa_min_rounds)
                and gain <= float(self.cfg.new_ppa_gain_rtol) * obj_scale
            ):
                z = z_refine
                break
            z = z_refine
            prev_obj = obj

        return self._clip(z).detach()
