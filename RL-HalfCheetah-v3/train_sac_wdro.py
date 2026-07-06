#!/usr/bin/env python3
"""SAC on HalfCheetah with Sinha-Duchi next-state adversaries.

Methods:
  nominal: standard SAC target with empirical replay-buffer next states.
  pgd:     per-batch PGD ascent on next states.
  icnn:    amortized Brenier map T(s') = grad psi(s') with an NPF-LastQuad ICNN.

The robust target uses

    s_adv = arg sup_z {-V(z) - lambda * ||z - s'||_M^2}
    y = r + gamma * V_target(s_adv)

where V is the SAC soft value min(Q1, Q2) - alpha log pi.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.distributions import Normal

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


Tensor = torch.Tensor
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return default if value is None else float(value)


def _env_words(name: str, default: Sequence[int]) -> List[int]:
    value = _env(name)
    if value is None:
        return list(default)
    return [int(tok) for tok in value.replace(",", " ").split()]


def _default_method() -> str:
    raw = (_env("ADVERSARY") or _env("RUN_ONLY_ALGO") or "icnn").lower()
    mapping = {
        "none": "nominal",
        "sac": "nominal",
        "nominal": "nominal",
        "pgd": "pgd",
        "wrm": "pgd",
        "icnn": "icnn",
        "npf": "icnn",
        "npf_lastquad": "icnn",
    }
    return mapping.get(raw, raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def mlp(sizes: Sequence[int], activation: type[nn.Module], output_activation: Optional[type[nn.Module]] = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(int(sizes[j]), int(sizes[j + 1])))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_targ, p in zip(target.parameters(), source.parameters()):
            p_targ.mul_(1.0 - tau).add_(p, alpha=tau)


@contextmanager
def freeze_params(*modules: Optional[nn.Module]):
    old: List[Tuple[nn.Parameter, bool]] = []
    try:
        for module in modules:
            if module is None:
                continue
            for p in module.parameters():
                old.append((p, p.requires_grad))
                p.requires_grad_(False)
        yield
    finally:
        for p, requires_grad in old:
            p.requires_grad_(requires_grad)


def as_tensor(x: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def finite_or(value: float, fallback: float = 0.0) -> float:
    return value if math.isfinite(value) else fallback


# ---------------------------------------------------------------------------
# Gym compatibility and MuJoCo modifiers
# ---------------------------------------------------------------------------


def gym_candidates(env_id: str):
    found = False
    try:
        import gymnasium as gym  # type: ignore

        if str(env_id).endswith(("-v2", "-v3")):
            try:
                import gymnasium_robotics  # noqa: F401  # type: ignore
            except Exception:
                pass
        found = True
        yield "gymnasium", gym
    except Exception:
        pass
    try:
        import gym  # type: ignore

        if not hasattr(np, "bool8"):
            np.bool8 = np.bool_  # type: ignore[attr-defined]
        found = True
        yield "gym", gym
    except Exception as exc:
        if not found:
            raise RuntimeError(
                "HalfCheetah requires gymnasium[mujoco] or gym[mujoco]. "
                "Install it in the environment used for the cluster job."
            ) from exc


def apply_mujoco_modifiers(env: Any, mass_scale: float = 1.0, friction_scale: float = 1.0, damping_scale: float = 1.0) -> None:
    unwrapped = getattr(env, "unwrapped", env)
    model = getattr(unwrapped, "model", None)
    if model is None:
        return
    if hasattr(model, "body_mass"):
        model.body_mass[:] = model.body_mass[:] * float(mass_scale)
    if hasattr(model, "geom_friction"):
        model.geom_friction[:] = model.geom_friction[:] * float(friction_scale)
    if hasattr(model, "dof_damping"):
        model.dof_damping[:] = model.dof_damping[:] * float(damping_scale)


def make_env(
    env_id: str,
    seed: int,
    *,
    mass_scale: float = 1.0,
    friction_scale: float = 1.0,
    damping_scale: float = 1.0,
):
    errors: List[str] = []
    env = None
    for backend_name, gym in gym_candidates(env_id):
        try:
            env = gym.make(env_id)
            break
        except Exception as exc:
            errors.append(f"{backend_name}: {exc}")
    if env is None:
        detail = "; ".join(errors) if errors else "no Gym backend available"
        raise RuntimeError(f"Could not create {env_id}. Tried Gymnasium/Gym backends: {detail}")
    apply_mujoco_modifiers(env, mass_scale=mass_scale, friction_scale=friction_scale, damping_scale=damping_scale)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass
    return env


def reset_env(env: Any, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        try:
            out = env.reset(seed=int(seed))
        except TypeError:
            try:
                env.seed(int(seed))
            except Exception:
                pass
            out = env.reset()
    else:
        out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    return np.asarray(obs, dtype=np.float32)


def step_env(env: Any, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    out = env.step(action)
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        return np.asarray(obs, dtype=np.float32), float(rew), bool(terminated), bool(truncated), dict(info)
    obs, rew, done, info = out
    return np.asarray(obs, dtype=np.float32), float(rew), bool(done), False, dict(info)


# ---------------------------------------------------------------------------
# Replay buffer and running state scale
# ---------------------------------------------------------------------------


class RunningMeanStd:
    def __init__(self, shape: Sequence[int], eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(eps)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def tensors(self, device: torch.device, mode: str) -> Tuple[Tensor, Tensor]:
        if mode == "identity":
            mean = np.zeros_like(self.mean)
            scale = np.ones_like(self.mean)
        elif mode == "running_std":
            mean = self.mean
            scale = np.sqrt(np.maximum(self.var, 1e-6))
        else:
            raise ValueError(f"Unknown state metric mode: {mode}")
        return as_tensor(mean.astype(np.float32), device), as_tensor(scale.astype(np.float32), device).clamp_min(1e-3)


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, seed: int):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.max_size = int(size)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)
        self.next_state_rms = RunningMeanStd((obs_dim,))

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: float) -> None:
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.next_state_rms.update(next_obs)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        idxs = self.rng.integers(0, self.size, size=int(batch_size))
        return {
            "obs": as_tensor(self.obs_buf[idxs], device),
            "act": as_tensor(self.act_buf[idxs], device),
            "rew": as_tensor(self.rew_buf[idxs], device),
            "next_obs": as_tensor(self.next_obs_buf[idxs], device),
            "done": as_tensor(self.done_buf[idxs], device),
        }


# ---------------------------------------------------------------------------
# SAC models
# ---------------------------------------------------------------------------


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int], action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes], nn.ReLU, nn.ReLU)
        last_size = int(hidden_sizes[-1]) if hidden_sizes else int(obs_dim)
        self.mu_layer = nn.Linear(last_size, act_dim)
        self.log_std_layer = nn.Linear(last_size, act_dim)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        raw = mu if deterministic else dist.rsample()
        squashed = torch.tanh(raw)
        action = self.action_bias + self.action_scale * squashed
        log_prob = dist.log_prob(raw)
        correction = torch.log(self.action_scale * (1.0 - squashed.pow(2)) + EPS)
        log_prob = (log_prob - correction).sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: torch.device, deterministic: bool = False) -> np.ndarray:
        obs_t = as_tensor(obs[None, :], device)
        action, _, _ = self.sample(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden_sizes, 1], nn.ReLU, None)

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        return self.q(torch.cat([obs, act], dim=-1))


def sac_soft_value(obs: Tensor, actor: SquashedGaussianActor, q1: QNetwork, q2: QNetwork, alpha: Tensor) -> Tensor:
    act, logp, _ = actor.sample(obs, deterministic=False)
    q = torch.min(q1(obs, act), q2(obs, act))
    return q - alpha * logp


# ---------------------------------------------------------------------------
# NPF-LastQuad ICNN potential
# ---------------------------------------------------------------------------


def _npf_softplus_inverse(y: float) -> float:
    if y <= 0.0:
        return -1e3
    return float(math.log(math.expm1(y)))


def _icnn_principled_moments(fan_in: int) -> Tuple[float, float, float, float, float]:
    if fan_in <= 0:
        raise ValueError(f"ICNN fan-in must be positive; got {fan_in}.")
    denom_offset = 6.0 * (math.pi - 1.0)
    denom_slope = 3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0
    denom = denom_offset + (fan_in - 1.0) * denom_slope
    mu_w = math.sqrt((6.0 * math.pi) / (fan_in * denom))
    sigma_w2 = 1.0 / float(fan_in)
    mu_b = math.sqrt((3.0 * fan_in) / denom)
    mu_w_sq = mu_w * mu_w
    log_var_plus_mean_sq = math.log(sigma_w2 + mu_w_sq)
    log_mean_sq = math.log(mu_w_sq)
    tilde_mu = log_mean_sq - 0.5 * log_var_plus_mean_sq
    tilde_sigma2 = max(log_var_plus_mean_sq - log_mean_sq, 1e-12)
    tilde_sigma = math.sqrt(tilde_sigma2)
    return mu_w, sigma_w2, mu_b, tilde_mu, tilde_sigma


class NPFNonNegativeDense(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_param = nn.Parameter(torch.empty(self.in_features, self.out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _, _, mu_b, tilde_mu, tilde_sigma = _icnn_principled_moments(self.in_features)
        with torch.no_grad():
            self.weight_param.normal_(mean=tilde_mu, std=tilde_sigma)
            if self.bias is not None:
                self.bias.fill_(-mu_b)

    def forward(self, x: Tensor) -> Tensor:
        y = x.matmul(torch.exp(self.weight_param))
        if self.bias is not None:
            y = y + self.bias
        return y


class NPFQuadraticForm(nn.Module):
    def __init__(self, input_dim: int, num_forms: int, rank: int = 0, init_eps: float = 1e-4):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_forms = int(num_forms)
        self.rank = int(rank)
        self.init_eps = float(init_eps)
        self.delta_raw = nn.Parameter(torch.full((self.num_forms, self.input_dim), _npf_softplus_inverse(self.init_eps)))
        if self.rank > 0:
            std = self.init_eps / math.sqrt(max(self.rank * self.input_dim, 1))
            self.A = nn.Parameter(std * torch.randn(self.num_forms, self.rank, self.input_dim))
        else:
            self.register_parameter("A", None)

    @property
    def delta(self) -> Tensor:
        return F.softplus(self.delta_raw)

    def forward(self, z: Tensor) -> Tensor:
        delta = self.delta
        q_diag = ((z.unsqueeze(1) * delta.unsqueeze(0)) ** 2).sum(dim=-1)
        if self.A is not None:
            az = torch.einsum("ord,bd->bor", self.A, z)
            return q_diag + (az ** 2).sum(dim=-1)
        return q_diag


class NPFLastQuadPotential(nn.Module):
    """NPF ICNN with hidden affine/nonnegative blocks and final diagonal quadratic."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "softplus",
        softplus_beta: float = 5.0,
        init_eps: float = 1e-4,
        strong_convexity: float = 1.0,
    ):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("NPFLastQuadPotential needs at least one hidden layer.")
        self.input_dim = int(input_dim)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.activation = activation.lower()
        self.softplus_beta = float(softplus_beta)
        self.init_eps = float(init_eps)
        self.strong_convexity = float(strong_convexity)
        self.outer_a = nn.Parameter(torch.zeros(self.input_dim))

        self.b_linears = nn.ModuleList()
        self.w_linears: nn.ModuleList = nn.ModuleList()
        for layer_idx, width in enumerate(self.hidden_sizes):
            self.b_linears.append(nn.Linear(self.input_dim, width, bias=True))
            if layer_idx == 0:
                self.w_linears.append(None)  # type: ignore[arg-type]
            else:
                self.w_linears.append(NPFNonNegativeDense(self.hidden_sizes[layer_idx - 1], width, use_bias=False))
        self.w_out = NPFNonNegativeDense(self.hidden_sizes[-1], 1, use_bias=False)
        self.q_out = NPFQuadraticForm(self.input_dim, 1, rank=0, init_eps=self.init_eps)
        self.b_out = nn.Linear(self.input_dim, 1, bias=True)
        self.init_as_identity()

    def init_as_identity(self) -> None:
        delta_raw_init = _npf_softplus_inverse(self.init_eps)
        with torch.no_grad():
            self.outer_a.zero_()
            for layer in self.b_linears:
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()
            self.q_out.delta_raw.fill_(delta_raw_init)
            self.b_out.weight.zero_()
            if self.b_out.bias is not None:
                self.b_out.bias.zero_()

    def _act(self, x: Tensor) -> Tensor:
        if self.activation == "softplus":
            beta = self.softplus_beta
            return F.softplus(beta * x) / beta
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "elu":
            return F.elu(x, alpha=1.0)
        raise ValueError(f"Unsupported ICNN activation: {self.activation}")

    def forward(self, z: Tensor) -> Tensor:
        z_flat = z.reshape(z.size(0), -1)
        fixed_q = 0.5 * self.strong_convexity * z_flat.pow(2).sum(dim=-1)
        linear = z_flat @ self.outer_a

        h = self._act(self.b_linears[0](z_flat))
        for layer_idx in range(1, len(self.hidden_sizes)):
            h = self._act(self.w_linears[layer_idx](h) + self.b_linears[layer_idx](z_flat))
        phi = self.w_out(h).squeeze(-1) + self.q_out(z_flat).squeeze(-1) + self.b_out(z_flat).squeeze(-1)
        return fixed_q + linear + phi


def transport_from_potential(z: Tensor, potential: nn.Module, create_graph: bool) -> Tensor:
    z_in = z.clone().detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        psi = potential(z_in)
        grad = torch.autograd.grad(psi.sum(), z_in, create_graph=create_graph)[0]
    grad = torch.where(torch.isfinite(grad), grad, z)
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    if not create_graph:
        grad = grad.detach()
    return grad.view_as(z)


# ---------------------------------------------------------------------------
# BB + Armijo ascent
# ---------------------------------------------------------------------------


@dataclass
class BBArmijoState:
    alpha_min: float
    alpha_max: float
    alpha_prev: float
    ls_c: float
    ls_shrink: float
    ls_max_steps: int
    reject_on_armijo_failure: bool = True
    prev_params_vec: Optional[Tensor] = None
    prev_grad_vec: Optional[Tensor] = None

    @classmethod
    def create(
        cls,
        alpha0: float,
        alpha_min: float,
        alpha_max: float,
        ls_c: float,
        ls_shrink: float,
        ls_max_steps: int,
        reject_on_armijo_failure: bool = True,
    ) -> "BBArmijoState":
        alpha0 = float(max(alpha_min, min(alpha_max, alpha0)))
        return cls(
            alpha_min=float(max(alpha_min, 1e-12)),
            alpha_max=float(max(alpha_max, alpha_min)),
            alpha_prev=alpha0,
            ls_c=float(ls_c),
            ls_shrink=float(ls_shrink),
            ls_max_steps=int(max(ls_max_steps, 1)),
            reject_on_armijo_failure=bool(reject_on_armijo_failure),
        )

    def propose(self, params_vec: Tensor, grad_vec: Tensor) -> float:
        if self.prev_params_vec is None or self.prev_grad_vec is None or self.prev_params_vec.shape != params_vec.shape:
            return self.alpha_prev
        s = params_vec - self.prev_params_vec
        y = grad_vec - self.prev_grad_vec
        denom = torch.dot(s, y)
        num = torch.dot(s, s)
        if torch.isfinite(denom) and float(denom) < -1e-12:
            alpha = float((-num / denom).clamp(self.alpha_min, self.alpha_max).item())
        else:
            alpha = self.alpha_prev
        if not math.isfinite(alpha):
            alpha = self.alpha_prev
        return max(self.alpha_min, min(self.alpha_max, float(alpha)))

    def update_history(self, params_vec: Tensor, grad_vec: Tensor, alpha: float) -> "BBArmijoState":
        return BBArmijoState(
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            alpha_prev=max(self.alpha_min, min(self.alpha_max, float(alpha))),
            ls_c=self.ls_c,
            ls_shrink=self.ls_shrink,
            ls_max_steps=self.ls_max_steps,
            reject_on_armijo_failure=self.reject_on_armijo_failure,
            prev_params_vec=params_vec.detach().clone(),
            prev_grad_vec=grad_vec.detach().clone(),
        )


def bb_armijo_step_params(params: Iterable[nn.Parameter], f_params, bb_state: BBArmijoState) -> Tuple[BBArmijoState, float, float, bool]:
    params = list(params)
    params_vec = nn_utils.parameters_to_vector(params).detach()
    f_val = f_params(True)
    grads = torch.autograd.grad(f_val, params, create_graph=False, retain_graph=False, allow_unused=True)
    grad_vec = torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1) for p, g in zip(params, grads)])
    grad_norm = float(grad_vec.norm().item())
    f_val_float = float(f_val.detach())
    if not math.isfinite(f_val_float) or not torch.isfinite(grad_vec).all() or not math.isfinite(grad_norm):
        return bb_state, f_val_float, grad_norm, False
    g_dot_g = float(torch.dot(grad_vec, grad_vec).item())
    if g_dot_g <= 0.0 or not math.isfinite(g_dot_g):
        return bb_state, f_val_float, grad_norm, False

    alpha = bb_state.propose(params_vec, grad_vec)
    alpha_k = alpha
    armijo_succeeded = False
    for i in range(bb_state.ls_max_steps):
        trial_vec = params_vec + alpha_k * grad_vec
        with torch.no_grad():
            nn_utils.vector_to_parameters(trial_vec, params)
            f_trial = float(f_params(False).detach())
        if math.isfinite(f_trial) and f_trial >= f_val_float + bb_state.ls_c * alpha_k * g_dot_g:
            armijo_succeeded = True
            break
        if i < bb_state.ls_max_steps - 1:
            alpha_k *= bb_state.ls_shrink

    if armijo_succeeded or not bb_state.reject_on_armijo_failure:
        final_vec = params_vec + alpha_k * grad_vec
        with torch.no_grad():
            nn_utils.vector_to_parameters(final_vec, params)
        return bb_state.update_history(params_vec, grad_vec, alpha_k), f_val_float, grad_norm, armijo_succeeded

    with torch.no_grad():
        nn_utils.vector_to_parameters(params_vec, params)
    return bb_state, f_val_float, grad_norm, False


# ---------------------------------------------------------------------------
# Adversaries
# ---------------------------------------------------------------------------


def mahalanobis_cost(z: Tensor, base: Tensor, scale: Tensor) -> Tensor:
    d = (z - base) / scale.clamp_min(1e-6)
    return d.pow(2).sum(dim=-1, keepdim=True)


class ICNNStateAdversary:
    def __init__(self, obs_dim: int, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.potential = NPFLastQuadPotential(
            input_dim=obs_dim,
            hidden_sizes=args.icnn_hidden_sizes,
            activation=args.icnn_activation,
            softplus_beta=args.icnn_softplus_beta,
            init_eps=args.icnn_init_eps,
            strong_convexity=args.icnn_strong_convexity,
        ).to(device)
        self.bb_state = BBArmijoState.create(
            alpha0=args.bb_alpha0,
            alpha_min=args.bb_alpha_min,
            alpha_max=args.bb_alpha_max,
            ls_c=args.bb_ls_c,
            ls_shrink=args.bb_ls_shrink,
            ls_max_steps=args.bb_ls_max_steps,
            reject_on_armijo_failure=True,
        )

    def transport(self, next_obs: Tensor, mean: Tensor, scale: Tensor, create_graph: bool) -> Tensor:
        u = (next_obs - mean) / scale
        u_adv = transport_from_potential(u, self.potential, create_graph=create_graph)
        z_adv = mean + scale * u_adv
        z_adv = torch.where(torch.isfinite(z_adv), z_adv, next_obs)
        if not create_graph:
            z_adv = z_adv.detach()
        return z_adv

    def update(
        self,
        next_obs: Tensor,
        actor: SquashedGaussianActor,
        q1: QNetwork,
        q2: QNetwork,
        alpha: Tensor,
        mean: Tensor,
        scale: Tensor,
    ) -> Dict[str, float]:
        last_obj = 0.0
        last_grad = 0.0
        successes = 0
        with freeze_params(actor, q1, q2):
            for _ in range(int(self.args.icnn_steps)):
                def objective(create_graph: bool) -> Tensor:
                    z_adv = self.transport(next_obs, mean, scale, create_graph=create_graph)
                    v = sac_soft_value(z_adv, actor, q1, q2, alpha.detach())
                    cost = mahalanobis_cost(z_adv, next_obs, scale)
                    obj = (-v - float(self.args.lambda_penalty) * cost).mean()
                    return torch.nan_to_num(obj, nan=-1e12, posinf=-1e12, neginf=-1e12)

                self.bb_state, last_obj, last_grad, ok = bb_armijo_step_params(self.potential.parameters(), objective, self.bb_state)
                successes += int(ok)

        with torch.enable_grad():
            z_final = self.transport(next_obs, mean, scale, create_graph=False)
        cost = mahalanobis_cost(z_final, next_obs, scale).detach()
        delta_l2 = (z_final - next_obs).pow(2).sum(dim=-1).sqrt().detach()
        return {
            "adv_obj": finite_or(float(last_obj)),
            "adv_grad_norm": finite_or(float(last_grad)),
            "adv_armijo_success_rate": successes / max(1, int(self.args.icnn_steps)),
            "adv_cost": finite_or(float(cost.mean().item())),
            "adv_delta_l2": finite_or(float(delta_l2.mean().item())),
        }


def pgd_adversarial_next_obs(
    next_obs: Tensor,
    actor: SquashedGaussianActor,
    q1: QNetwork,
    q2: QNetwork,
    alpha: Tensor,
    args: argparse.Namespace,
    scale: Tensor,
) -> Tuple[Tensor, Dict[str, float]]:
    batch = next_obs.shape[0]
    best_z = next_obs.detach()
    best_score = torch.full((batch, 1), -float("inf"), device=next_obs.device)

    with freeze_params(actor, q1, q2):
        restarts = max(1, int(args.pgd_restarts))
        for restart_idx in range(restarts):
            if restart_idx == 0 or float(args.pgd_noise_scale) <= 0.0:
                z = next_obs.detach().clone()
            else:
                z = next_obs.detach() + float(args.pgd_noise_scale) * scale * torch.randn_like(next_obs)
            for _ in range(int(args.pgd_steps)):
                z = z.detach().requires_grad_(True)
                v = sac_soft_value(z, actor, q1, q2, alpha.detach())
                cost = mahalanobis_cost(z, next_obs, scale)
                score = -v - float(args.lambda_penalty) * cost
                grad = torch.autograd.grad(score.sum(), z, create_graph=False, retain_graph=False)[0]
                z = z + float(args.pgd_step_size) * grad
                if float(args.pgd_max_delta_norm) > 0.0:
                    delta_scaled = (z - next_obs) / scale
                    norm = delta_scaled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    factor = torch.clamp(float(args.pgd_max_delta_norm) / norm, max=1.0)
                    z = next_obs + scale * delta_scaled * factor
                z = torch.where(torch.isfinite(z), z, next_obs)

            with torch.no_grad():
                v = sac_soft_value(z, actor, q1, q2, alpha.detach())
                cost = mahalanobis_cost(z, next_obs, scale)
                score = -v - float(args.lambda_penalty) * cost
                take = score > best_score
                best_score = torch.where(take, score, best_score)
                best_z = torch.where(take, z.detach(), best_z)

    final_cost = mahalanobis_cost(best_z, next_obs, scale)
    delta_l2 = (best_z - next_obs).pow(2).sum(dim=-1).sqrt()
    return best_z.detach(), {
        "adv_obj": finite_or(float(best_score.mean().item())),
        "adv_cost": finite_or(float(final_cost.mean().item())),
        "adv_delta_l2": finite_or(float(delta_l2.mean().item())),
    }


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def evaluate_policy(
    actor: SquashedGaussianActor,
    device: torch.device,
    args: argparse.Namespace,
    *,
    seed: int,
    mass_scale: float = 1.0,
    friction_scale: float = 1.0,
    damping_scale: float = 1.0,
) -> Tuple[float, float]:
    env = make_env(
        args.env_id,
        seed=seed,
        mass_scale=mass_scale,
        friction_scale=friction_scale,
        damping_scale=damping_scale,
    )
    returns: List[float] = []
    lengths: List[int] = []
    try:
        for ep in range(int(args.eval_episodes)):
            obs = reset_env(env, seed=seed + ep)
            ep_ret = 0.0
            ep_len = 0
            while ep_len < int(args.eval_max_steps):
                action = actor.act(obs, device, deterministic=True)
                obs, rew, terminated, truncated, _ = step_env(env, action)
                ep_ret += rew
                ep_len += 1
                if terminated or truncated:
                    break
            returns.append(ep_ret)
            lengths.append(ep_len)
    finally:
        env.close()
    return float(np.mean(returns)), float(np.mean(lengths))


def evaluate_sweeps(actor: SquashedGaussianActor, device: torch.device, args: argparse.Namespace, step: int) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    nom_ret, nom_len = evaluate_policy(actor, device, args, seed=int(args.eval_seed), mass_scale=1.0, friction_scale=1.0, damping_scale=1.0)
    metrics["eval/nominal_return"] = nom_ret
    metrics["eval/nominal_length"] = nom_len
    if bool(args.no_robust_eval):
        return metrics
    for scale in args.eval_mass_scales:
        ret, _ = evaluate_policy(actor, device, args, seed=int(args.eval_seed) + 1000 + step, mass_scale=float(scale))
        metrics[f"eval_mass/{scale:g}"] = ret
    for scale in args.eval_friction_scales:
        ret, _ = evaluate_policy(actor, device, args, seed=int(args.eval_seed) + 2000 + step, friction_scale=float(scale))
        metrics[f"eval_friction/{scale:g}"] = ret
    for scale in args.eval_damping_scales:
        ret, _ = evaluate_policy(actor, device, args, seed=int(args.eval_seed) + 3000 + step, damping_scale=float(scale))
        metrics[f"eval_damping/{scale:g}"] = ret
    return metrics


def make_run_dir(args: argparse.Namespace) -> Path:
    base = Path(args.results_dir) if args.results_dir else Path(__file__).resolve().parent / "runs"
    run_name = args.run_name or f"{args.method}_seed{args.seed}_lam{args.lambda_penalty:g}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out = base / run_name
    suffix = 1
    candidate = out
    while candidate.exists():
        candidate = Path(f"{out}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def save_checkpoint(
    path: Path,
    actor: SquashedGaussianActor,
    q1: QNetwork,
    q2: QNetwork,
    q1_target: QNetwork,
    q2_target: QNetwork,
    log_alpha: Tensor,
    adversary: Optional[ICNNStateAdversary],
    args: argparse.Namespace,
    step: int,
) -> None:
    payload: Dict[str, Any] = {
        "step": int(step),
        "args": vars(args),
        "actor": actor.state_dict(),
        "q1": q1.state_dict(),
        "q2": q2.state_dict(),
        "q1_target": q1_target.state_dict(),
        "q2_target": q2_target.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
    }
    if adversary is not None:
        payload["icnn_potential"] = adversary.potential.state_dict()
        payload["icnn_arch"] = {
            "hidden_sizes": list(args.icnn_hidden_sizes),
            "activation": args.icnn_activation,
            "softplus_beta": float(args.icnn_softplus_beta),
            "init_eps": float(args.icnn_init_eps),
            "strong_convexity": float(args.icnn_strong_convexity),
        }
    torch.save(payload, path)


def train(args: argparse.Namespace) -> Path:
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    run_dir = make_run_dir(args)
    print(f"[run] dir={run_dir}", flush=True)
    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    wandb_run = None
    if bool(args.wandb):
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.run_name or run_dir.name,
            config=vars(args),
            dir=str(run_dir),
        )

    env = make_env(args.env_id, seed=int(args.seed))
    obs = reset_env(env, seed=int(args.seed))
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    actor = SquashedGaussianActor(obs_dim, act_dim, args.hidden_sizes, action_low, action_high).to(device)
    q1 = QNetwork(obs_dim, act_dim, args.hidden_sizes).to(device)
    q2 = QNetwork(obs_dim, act_dim, args.hidden_sizes).to(device)
    q1_target = QNetwork(obs_dim, act_dim, args.hidden_sizes).to(device)
    q2_target = QNetwork(obs_dim, act_dim, args.hidden_sizes).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=float(args.actor_lr))
    q_params = list(q1.parameters()) + list(q2.parameters())
    q_opt = torch.optim.Adam(q_params, lr=float(args.critic_lr))
    log_alpha = torch.tensor(math.log(float(args.init_alpha)), device=device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=float(args.alpha_lr))
    target_entropy = float(args.target_entropy) if args.target_entropy is not None else -float(act_dim)

    replay = ReplayBuffer(obs_dim, act_dim, int(args.replay_size), seed=int(args.seed) + 17)
    adversary = ICNNStateAdversary(obs_dim, args, device) if args.method == "icnn" else None

    print(
        "[config] "
        f"env={args.env_id} method={args.method} device={device} obs_dim={obs_dim} act_dim={act_dim} "
        f"lambda={args.lambda_penalty} batch={args.batch_size}",
        flush=True,
    )
    if adversary is not None:
        n_adv = sum(p.numel() for p in adversary.potential.parameters())
        print(
            "[icnn] "
            f"hidden={tuple(args.icnn_hidden_sizes)} activation={args.icnn_activation} "
            f"beta={args.icnn_softplus_beta} strong_convexity={args.icnn_strong_convexity} "
            f"params={n_adv:,} steps={args.icnn_steps}",
            flush=True,
        )

    log_path = run_dir / "metrics.jsonl"
    ep_ret = 0.0
    ep_len = 0
    episode = 0
    last_metrics: Dict[str, float] = {}
    start_time = time.perf_counter()
    iterator = range(1, int(args.total_steps) + 1)
    if tqdm is not None:
        iterator = tqdm(iterator, dynamic_ncols=True, desc=f"{args.env_id}:{args.method}")  # type: ignore[assignment]

    try:
        for step in iterator:
            if step <= int(args.start_steps):
                action = env.action_space.sample().astype(np.float32)
            else:
                action = actor.act(obs, device, deterministic=False).astype(np.float32)

            next_obs, reward, terminated, truncated, _info = step_env(env, action)
            ep_ret += reward
            ep_len += 1
            replay_done = float(terminated)
            replay.add(obs, action, reward, next_obs, replay_done)
            obs = next_obs

            if terminated or truncated:
                episode += 1
                record = {
                    "step": step,
                    "episode": episode,
                    "train/episode_return": ep_ret,
                    "train/episode_length": ep_len,
                    "time/elapsed_s": time.perf_counter() - start_time,
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
                if wandb_run is not None:
                    wandb_run.log(record, step=step)
                obs = reset_env(env, seed=int(args.seed) + episode)
                ep_ret = 0.0
                ep_len = 0

            if replay.size >= int(args.update_after) and step % int(args.update_every) == 0:
                updates = int(args.update_every) * int(args.updates_per_step)
                update_start = time.perf_counter()
                for _ in range(updates):
                    batch = replay.sample_batch(int(args.batch_size), device)
                    mean, scale = replay.next_state_rms.tensors(device, mode=str(args.state_metric))
                    alpha = log_alpha.exp().detach()

                    adv_metrics: Dict[str, float] = {}
                    if args.method == "nominal":
                        robust_next = batch["next_obs"]
                    elif args.method == "pgd":
                        robust_next, adv_metrics = pgd_adversarial_next_obs(
                            batch["next_obs"], actor, q1, q2, alpha, args, scale
                        )
                    elif args.method == "icnn":
                        assert adversary is not None
                        adv_metrics = adversary.update(batch["next_obs"], actor, q1, q2, alpha, mean, scale)
                        with torch.enable_grad():
                            robust_next = adversary.transport(batch["next_obs"], mean, scale, create_graph=False)
                    else:
                        raise ValueError(f"Unknown method: {args.method}")
                    robust_next = robust_next.detach()

                    with torch.no_grad():
                        next_action, next_logp, _ = actor.sample(robust_next)
                        target_q = torch.min(q1_target(robust_next, next_action), q2_target(robust_next, next_action))
                        target_v = target_q - log_alpha.exp().detach() * next_logp
                        backup = batch["rew"] + float(args.gamma) * (1.0 - batch["done"]) * target_v

                    q1_pred = q1(batch["obs"], batch["act"])
                    q2_pred = q2(batch["obs"], batch["act"])
                    q_loss = F.mse_loss(q1_pred, backup) + F.mse_loss(q2_pred, backup)
                    q_opt.zero_grad(set_to_none=True)
                    q_loss.backward()
                    nn.utils.clip_grad_norm_(q_params, float(args.max_grad_norm))
                    q_opt.step()

                    with freeze_params(q1, q2):
                        pi_action, logp_pi, _ = actor.sample(batch["obs"])
                        q_pi = torch.min(q1(batch["obs"], pi_action), q2(batch["obs"], pi_action))
                        actor_loss = (log_alpha.exp().detach() * logp_pi - q_pi).mean()
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), float(args.max_grad_norm))
                    actor_opt.step()

                    alpha_loss = -(log_alpha * (logp_pi.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    alpha_opt.step()

                    soft_update(q1_target, q1, float(args.tau))
                    soft_update(q2_target, q2, float(args.tau))

                    last_metrics = {
                        "loss/q": finite_or(float(q_loss.detach().item())),
                        "loss/policy": finite_or(float(actor_loss.detach().item())),
                        "loss/alpha": finite_or(float(alpha_loss.detach().item())),
                        "sac/alpha": finite_or(float(log_alpha.exp().detach().item())),
                        "sac/q1_mean": finite_or(float(q1_pred.detach().mean().item())),
                        "sac/target_mean": finite_or(float(backup.detach().mean().item())),
                        **adv_metrics,
                    }

                last_metrics["time/update_s"] = time.perf_counter() - update_start

            should_log = int(args.log_interval) > 0 and step % int(args.log_interval) == 0
            should_eval = int(args.eval_interval) > 0 and step % int(args.eval_interval) == 0
            if should_log or should_eval:
                record = {
                    "step": step,
                    "replay/size": replay.size,
                    "time/elapsed_s": time.perf_counter() - start_time,
                    **last_metrics,
                }
                if should_eval:
                    eval_metrics = evaluate_sweeps(actor, device, args, step)
                    record.update(eval_metrics)
                    print(
                        f"[eval] step={step} nominal_return={eval_metrics.get('eval/nominal_return', float('nan')):.1f}",
                        flush=True,
                    )
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
                if wandb_run is not None:
                    wandb_run.log(record, step=step)
                if tqdm is not None and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(
                        ret=f"{record.get('eval/nominal_return', ep_ret):.1f}",
                        q=f"{record.get('loss/q', 0.0):.2f}",
                        alpha=f"{record.get('sac/alpha', 0.0):.3f}",
                    )

            if int(args.save_interval) > 0 and step % int(args.save_interval) == 0:
                save_checkpoint(run_dir / f"checkpoint_step{step}.pt", actor, q1, q2, q1_target, q2_target, log_alpha, adversary, args, step)

        save_checkpoint(run_dir / "checkpoint_final.pt", actor, q1, q2, q1_target, q2_target, log_alpha, adversary, args, int(args.total_steps))
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()

    print(f"[done] wrote {run_dir}", flush=True)
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HalfCheetah SAC with nominal, PGD, or ICNN WDRO next-state adversaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-id", default=_env("ENV_ID", "HalfCheetah-v5"))
    parser.add_argument("--method", choices=["nominal", "pgd", "icnn"], default=_default_method())
    parser.add_argument("--seed", type=int, default=_env_int("SEED", 0))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=_env("DEVICE", "auto"))
    parser.add_argument("--run-name", default=_env("RUN_NAME"))
    parser.add_argument("--results-dir", default=_env("RESULTS_DIR"))

    parser.add_argument("--total-steps", type=int, default=_env_int("TOTAL_STEPS", 1_000_000))
    parser.add_argument("--replay-size", type=int, default=_env_int("REPLAY_SIZE", 1_000_000))
    parser.add_argument("--start-steps", type=int, default=_env_int("START_STEPS", 10_000))
    parser.add_argument("--update-after", type=int, default=_env_int("UPDATE_AFTER", 1_000))
    parser.add_argument("--update-every", type=int, default=_env_int("UPDATE_EVERY", 1))
    parser.add_argument("--updates-per-step", type=int, default=_env_int("UPDATES_PER_STEP", 1))
    parser.add_argument("--batch-size", type=int, default=_env_int("COMMON_BATCH", 256))
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=_env_words("SAC_HIDDEN", (256, 256)))

    parser.add_argument("--gamma", type=float, default=_env_float("GAMMA", 0.99))
    parser.add_argument("--tau", type=float, default=_env_float("TAU", 0.005))
    parser.add_argument("--actor-lr", type=float, default=_env_float("ACTOR_LR", 3e-4))
    parser.add_argument("--critic-lr", type=float, default=_env_float("CRITIC_LR", 3e-4))
    parser.add_argument("--alpha-lr", type=float, default=_env_float("ALPHA_LR", 3e-4))
    parser.add_argument("--init-alpha", type=float, default=_env_float("INIT_ALPHA", 0.2))
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=_env_float("MAX_GRAD_NORM", 10.0))

    parser.add_argument("--lambda-penalty", type=float, default=_env_float("PENALTY_LAMBDA", 30.0))
    parser.add_argument("--state-metric", choices=["running_std", "identity"], default=_env("STATE_METRIC", "running_std"))

    parser.add_argument("--pgd-steps", type=int, default=_env_int("INP_STEPS", 20))
    parser.add_argument("--pgd-restarts", type=int, default=_env_int("INP_RESTARTS", 5))
    parser.add_argument("--pgd-step-size", type=float, default=_env_float("PGD_STEP_SIZE", 0.03))
    parser.add_argument("--pgd-noise-scale", type=float, default=_env_float("PGD_NOISE_SCALE", 0.01))
    parser.add_argument("--pgd-max-delta-norm", type=float, default=_env_float("PGD_MAX_DELTA_NORM", 0.0))

    parser.add_argument("--icnn-steps", type=int, default=_env_int("OMEGA_STEPS", 20))
    parser.add_argument("--icnn-hidden-sizes", type=int, nargs="+", default=_env_words("NPF_LASTQUAD_HIDDEN", (128, 128, 64, 32)))
    parser.add_argument("--icnn-activation", choices=["softplus", "relu", "elu"], default=_env("NPF_LASTQUAD_ACTIVATION", "softplus"))
    parser.add_argument("--icnn-softplus-beta", type=float, default=_env_float("NPF_LASTQUAD_SOFTPLUS_BETA", 5.0))
    parser.add_argument("--icnn-init-eps", type=float, default=_env_float("NPF_LASTQUAD_INIT_EPS", 1e-4))
    parser.add_argument("--icnn-strong-convexity", type=float, default=_env_float("NPF_LASTQUAD_STRONG_CONVEXITY", 1.0))
    parser.add_argument("--bb-alpha0", type=float, default=_env_float("BB_ALPHA0", 1e-3))
    parser.add_argument("--bb-alpha-min", type=float, default=_env_float("BB_ALPHA_MIN", 1e-6))
    parser.add_argument("--bb-alpha-max", type=float, default=_env_float("BB_ALPHA_MAX", 0.05))
    parser.add_argument("--bb-ls-c", type=float, default=_env_float("BB_LS_C", 1e-5))
    parser.add_argument("--bb-ls-shrink", type=float, default=_env_float("BB_LS_SHRINK", 0.5))
    parser.add_argument("--bb-ls-max-steps", type=int, default=_env_int("BB_LS_MAX_STEPS", 20))

    parser.add_argument("--log-interval", type=int, default=_env_int("LOG_INTERVAL", 1000))
    parser.add_argument("--eval-interval", type=int, default=_env_int("EVAL_INTERVAL", 10_000))
    parser.add_argument("--eval-episodes", type=int, default=_env_int("EVAL_EPISODES", 5))
    parser.add_argument("--eval-max-steps", type=int, default=_env_int("EVAL_MAX_STEPS", 1000))
    parser.add_argument("--eval-seed", type=int, default=_env_int("EVAL_SEED", 100_000))
    parser.add_argument("--eval-mass-scales", type=float, nargs="+", default=[0.5, 0.75, 1.25, 1.5, 2.0])
    parser.add_argument("--eval-friction-scales", type=float, nargs="+", default=[0.5, 0.75, 1.25, 1.5, 2.0])
    parser.add_argument("--eval-damping-scales", type=float, nargs="+", default=[])
    parser.add_argument("--no-robust-eval", action="store_true", default=bool(int(_env("NO_ROBUST_EVAL", "0") or "0")))
    parser.add_argument("--save-interval", type=int, default=_env_int("SAVE_INTERVAL", 100_000))

    parser.add_argument("--wandb", action="store_true", default=bool(int(_env("WANDB", "0") or "0")))
    parser.add_argument("--wandb-project", default=_env("WANDB_PROJECT", "wdro-halfcheetah"))
    parser.add_argument("--wandb-entity", default=_env("WANDB_ENTITY"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.results_dir == "":
        args.results_dir = None
    if len(args.hidden_sizes) == 0:
        raise ValueError("--hidden-sizes must contain at least one width.")
    if len(args.icnn_hidden_sizes) == 0:
        raise ValueError("--icnn-hidden-sizes must contain at least one width.")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
