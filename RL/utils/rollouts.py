"""Rollouts, PPO update, and return evaluation (pathwise + stochastic)."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from envs import VecEnvTorch
from envs.cartpole import VecParamCartPoleTorch
from envs.pendulum import VecSwingUpPendulumTorch
from models.policy import PolicyNet
from models.value import ValueNet


# ---------------------------
# Rollout + GAE
# ---------------------------

@torch.no_grad()
def rollout_batch(
    env: VecEnvTorch,
    policy: PolicyNet,
    value: ValueNet,
    xi_samples: torch.Tensor,
    steps_per_xi: int,
    seed0: int = 0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Dict[str, torch.Tensor]:
    """Vectorized rollout across batch of xi."""
    device = xi_samples.device
    B = xi_samples.shape[0]
    T = int(steps_per_xi)
    obs_dim = int(env.obs_dim)

    obs = torch.empty(T, B, obs_dim, device=device, dtype=torch.float32)
    act = torch.empty(T, B, device=device, dtype=torch.int64)
    logp = torch.empty(T, B, device=device, dtype=torch.float32)
    rew = torch.empty(T, B, device=device, dtype=torch.float32)
    val = torch.empty(T, B, device=device, dtype=torch.float32)
    done = torch.empty(T, B, device=device, dtype=torch.float32)
    val_next = torch.empty(T, B, device=device, dtype=torch.float32)

    env.reset(B, seed=seed0)

    for t in range(T):
        obs_t = env.state
        obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=0.0, neginf=0.0)
        env.state = obs_t
        logits = policy(obs_t)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        log_probs = F.log_softmax(logits, dim=-1)

        probs = log_probs.exp()
        probs = torch.clamp(probs, min=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        a_t = torch.multinomial(probs, num_samples=1).squeeze(-1)
        lp_t = log_probs.gather(1, a_t.unsqueeze(-1)).squeeze(-1)
        v_t = value(obs_t)

        next_state, r_t, d_t, _ = env.step(a_t, xi_samples)

        v_next = value(next_state)
        v_next = v_next * (~d_t).to(torch.float32)

        obs[t] = obs_t
        act[t] = a_t
        logp[t] = lp_t
        rew[t] = r_t
        val[t] = v_t
        done[t] = d_t.to(torch.float32)
        val_next[t] = v_next

        env.reset_done(d_t, seed=seed0 + 10_000 + t)

    adv = torch.zeros(T, B, device=device, dtype=torch.float32)
    last_adv = torch.zeros(B, device=device, dtype=torch.float32)

    for t in reversed(range(T)):
        nonterminal = 1.0 - done[t]
        delta = rew[t] + gamma * val_next[t] - val[t]
        last_adv = delta + gamma * gae_lambda * nonterminal * last_adv
        adv[t] = last_adv

    ret = adv + val

    return {
        "obs": obs.reshape(T * B, obs_dim),
        "act": act.reshape(T * B),
        "logp": logp.reshape(T * B),
        "adv": adv.reshape(T * B),
        "ret": ret.reshape(T * B),
        "val": val.reshape(T * B),
    }


def ppo_update(
    policy: PolicyNet,
    value: ValueNet,
    data: Dict[str, torch.Tensor],
    cfg,
    pi_opt: torch.optim.Optimizer,
    vf_opt: torch.optim.Optimizer,
):
    device = next(policy.parameters()).device

    obs = data["obs"]
    act = data["act"]
    old_logp = data["logp"]
    adv = data["adv"]
    ret = data["ret"]

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = obs.shape[0]
    for _ in range(cfg.train_epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, cfg.minibatch_size):
            mb = idx[start:start + cfg.minibatch_size]

            mb_obs = obs[mb]
            mb_act = act[mb]
            mb_old_logp = old_logp[mb]
            mb_adv = adv[mb]
            mb_ret = ret[mb]

            logits = policy(mb_obs)
            log_probs = F.log_softmax(logits, dim=-1)
            new_logp = log_probs.gather(1, mb_act.unsqueeze(-1)).squeeze(-1)

            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
            pi_loss = -torch.min(surr1, surr2).mean()

            v_pred = value(mb_obs)
            vf_loss = F.mse_loss(v_pred, mb_ret)

            probs = log_probs.exp()
            ent = -(probs * log_probs).sum(dim=-1).mean()

            loss = pi_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * ent

            pi_opt.zero_grad(set_to_none=True)
            vf_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(value.parameters(), cfg.max_grad_norm)
            pi_opt.step()
            vf_opt.step()


# ---------------------------
# Return evaluation (batched)
# ---------------------------

@torch.no_grad()
def evaluate_return_batch(
    env: VecEnvTorch,
    policy: PolicyNet,
    xi: torch.Tensor,
    n_episodes: int = 1,
    max_steps: int = 200,
    seed0: int = 0,
    deterministic: bool = True,
) -> torch.Tensor:
    """Vectorized evaluation: returns J(xi) for each xi in batch."""
    device = xi.device
    B = xi.shape[0]
    E = int(n_episodes)
    N = B * E

    xi_rep = xi.repeat_interleave(E, dim=0)

    env.reset(N, seed=seed0)
    alive = torch.ones(N, device=device, dtype=torch.bool)
    ret = torch.zeros(N, device=device, dtype=torch.float32)

    for t in range(int(max_steps)):
        obs = env.state
        logits = policy(obs)

        if deterministic:
            a = torch.argmax(logits, dim=-1)
        else:
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            a = torch.multinomial(probs, num_samples=1).squeeze(-1)

        next_state, r, d, _ = env.step(a, xi_rep)

        ret += r * alive.to(torch.float32)
        alive = alive & (~d)

        if not torch.any(alive):
            break

    ret = ret.reshape(B, E).mean(dim=1)
    return ret


def evaluate_return_batch_pathwise(
    env: VecEnvTorch,
    policy: PolicyNet,
    xi: torch.Tensor,
    n_episodes: int = 1,
    max_steps: int = 200,
    seed0: int = 0,
    survival_beta: float = 50.0,
    straight_through: bool = True,
) -> torch.Tensor:
    """Differentiable surrogate return for ∂J/∂xi estimation via pathwise autograd."""
    if isinstance(env, VecSwingUpPendulumTorch):
        device = xi.device
        B = xi.shape[0]
        E = int(n_episodes)
        N = B * E

        xi_rep = xi.repeat_interleave(E, dim=0)

        env.reset(N, seed=seed0)
        ret = torch.zeros(N, device=device, dtype=xi.dtype)

        beta = float(survival_beta)
        state_clip = 1e3
        action_values = env.action_values.to(device=device, dtype=xi.dtype)
        for _t in range(int(max_steps)):
            obs = torch.nan_to_num(env.state, nan=0.0, posinf=0.0, neginf=0.0)
            obs = torch.clamp(obs, -state_clip, state_clip)
            env.state = obs

            logits = policy(obs)
            probs = F.softmax(logits, dim=-1)

            if straight_through:
                hard = F.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.shape[-1]).to(dtype=probs.dtype)
                probs = hard + probs - probs.detach()

            u = probs @ action_values
            next_state, _r, _d, _info = env.step_continuous(u, xi_rep)

            theta = next_state[:, 0]
            theta_dot = next_state[:, 1]
            delta = env.angle_normalize(theta - math.pi).abs()
            s_theta = torch.sigmoid(beta * (env.theta_tol - delta))
            s_vel = torch.sigmoid(beta * (env.vel_tol - theta_dot.abs()))
            ret = ret + (s_theta * s_vel)

            next_state = torch.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)
            next_state = torch.clamp(next_state, -state_clip, state_clip)
            env.state = next_state

        ret = ret.reshape(B, E).mean(dim=1)
        return ret

    if not isinstance(env, VecParamCartPoleTorch):
        raise TypeError(f"Unsupported env type for pathwise gradients: {type(env).__name__}")

    device = xi.device
    B = xi.shape[0]
    E = int(n_episodes)
    N = B * E

    xi_rep = xi.repeat_interleave(E, dim=0)

    env.reset(N, seed=seed0)
    alive = torch.ones(N, device=device, dtype=xi.dtype)
    ret = torch.zeros(N, device=device, dtype=xi.dtype)

    beta = float(survival_beta)
    alive_eps = 1e-8
    state_clip = 1e3
    action_values = env.action_values.to(device=device, dtype=xi.dtype)
    for _t in range(int(max_steps)):
        obs = torch.nan_to_num(env.state, nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.clamp(obs, -state_clip, state_clip)
        env.state = obs

        logits = policy(obs)
        probs = F.softmax(logits, dim=-1)

        if straight_through:
            hard = F.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.shape[-1]).to(dtype=probs.dtype)
            probs = hard + probs - probs.detach()

        u = probs @ action_values
        next_state, _r, _d, _info = env.step_continuous(u, xi_rep)
        next_state = torch.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)
        next_state = torch.clamp(next_state, -state_clip, state_clip)
        env.state = next_state

        x = next_state[:, 0].abs()
        theta = next_state[:, 2].abs()

        x = torch.nan_to_num(x, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        theta = torch.nan_to_num(theta, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))

        s_x = torch.sigmoid(beta * (env.x_threshold - x))
        s_theta = torch.sigmoid(beta * (env.theta_threshold_radians - theta))
        survive = s_x * s_theta

        alive = alive * survive
        ret = ret + alive

        dead = (alive.detach() < alive_eps)
        keep = (~dead).to(dtype=next_state.dtype).unsqueeze(-1)
        env.state = keep * env.state + (1.0 - keep) * torch.zeros_like(env.state)

    ret = ret.reshape(B, E).mean(dim=1)
    return ret
