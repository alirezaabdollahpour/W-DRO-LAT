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

    def _dedup_targets(
        self,
        z_assigned: torch.Tensor,
        hat_xi: torch.Tensor,
    ) -> torch.Tensor:
        """Replace duplicated pool selections with the (Mahalanobis) centroid
        of the anchors that mapped to them.

        After the argmax-with-replacement reassignment, multiple anchors may
        point to the same pool entry; the resulting pushforward then assigns
        non-uniform mass to those targets, which breaks the uniform-marginal
        assumption of every gap estimator we use (Sinkhorn, Hungarian, etc.).
        Replacing each duplicate cluster's image with the centroid of its
        preimages preserves c-cyclic monotonicity (centroid of a c-monotone
        set's preimages is itself c-monotone-valued) and restores the
        uniform pushforward marginal expected by the gap estimator.

        We use the Mahalanobis-weighted centroid for diagonal M, which for a
        diagonal weight reduces to the arithmetic centroid (the Mahalanobis
        inner-product Karcher mean of a finite set in Euclidean space is the
        arithmetic mean). That suffices for any diagonal M_diag.
        """
        if z_assigned.shape[0] == 0:
            return z_assigned
        z_unique, inv = torch.unique(z_assigned, dim=0, return_inverse=True)
        # If everyone is unique, skip the gather/centroid work.
        if z_unique.shape[0] == z_assigned.shape[0]:
            return z_assigned
        K = int(z_unique.shape[0])
        d = int(z_assigned.shape[1])
        device = z_assigned.device
        dtype = z_assigned.dtype
        sums = torch.zeros((K, d), device=device, dtype=dtype)
        sums.index_add_(0, inv, hat_xi)
        counts = torch.zeros((K,), device=device, dtype=dtype)
        counts.index_add_(0, inv, torch.ones_like(inv, dtype=dtype))
        new_targets = sums / counts.clamp_min(1.0).unsqueeze(-1)
        return new_targets[inv]

    def _reassign_pool(
        self,
        z: torch.Tensor,
        hat_xi: torch.Tensor,
        env: VecEnvTorch,
        policy: PolicyNet,
        seed0: int,
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        """Batch-wide reassignment step (highlighted line in Algorithm 1):

            Z = {z_j}_{j=1}^B
            z_i <- argmax_{j in [B]}  f(theta, z_j) - lambda * ||z_j - hat_z_i||_M^2.

        ``z`` is the pool — same particles for every anchor — so we evaluate
        ``f`` at the B pool points *once* and reuse the values across all B
        anchors. The Mahalanobis-distance term varies per anchor; we form
        ``D`` in chunks of ``chunk_size`` anchors so the peak memory is
        ``O(B * chunk_size * d)`` instead of ``O(B^2 * d)`` (necessary at
        eval batch sizes >= 16K).

        After the per-anchor argmax we de-duplicate: any pool entry that
        is the image of more than one anchor is replaced by the centroid of
        the anchors that mapped to it. This restores uniform marginals on
        the pushforward — required for all gap estimators (Sinkhorn /
        Hungarian) which assume uniform target mass.
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
            best_j_chunks = []
            lam_f = float(self.cfg.lam)
            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)
                anchors_chunk = hat_xi[start:end]  # (Cb, d)
                # diff[c, j, :] = z[j] - anchors_chunk[c]  -> (Cb, B, d)
                diff = z.unsqueeze(0) - anchors_chunk.unsqueeze(1)
                D = (diff * diff * self.Mdiag).sum(dim=-1)        # (Cb, B)
                scores = f_pool.unsqueeze(0) - lam_f * D          # (Cb, B)
                best_j_chunks.append(scores.argmax(dim=1))
            best_j = torch.cat(best_j_chunks, dim=0)
        # Gather pool entries into anchor-indexed order; detach for safety.
        out = z.detach()[best_j].clone()
        return self._dedup_targets(out, hat_xi.detach())

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
