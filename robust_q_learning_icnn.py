#!/usr/bin/env python3
"""
Robust Q-learning with three training modes on CartPole:

* ``wrm-icnn`` (default): ICNN transport map T(ŝ)=∇φ_ω(ŝ) trained adversarially (WRM-ICNN).
* ``wrm``: first-order Wasserstein robust method via projected gradient ascent on states (no ICNN).
* ``erm``: nominal Q-learning baseline (no adversary).
* ``all``: runs all three sequentially and prints a comparison table.

Inner robust Bellman target (continuous analogue of Sinha et al. 5.3):

    s_adv = argmin_s { r(s) + λ max_a Q(s, a) + γ * ½ ||s - ŝ||^2 }

The ICNN approximates this map T(ŝ) ≈ s_adv via T(ŝ) = ∇φ_ω(ŝ).
CartPole reward is reshaped to r(θ)=exp(-|θ|) as in the described setting.
"""

from __future__ import annotations

import argparse
import collections
import math
import random
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from tqdm.auto import tqdm

from ademamix import AdEMAMix

Tensor = torch.Tensor

# NumPy 2.0 removed ``np.bool8``; Gym 0.26 still references it. Provide a back-compat alias.
if not hasattr(np, "bool8"):  # pragma: no cover - defensive shim
    np.bool8 = np.bool_


# --------------------------
#  ICNN building blocks
# --------------------------
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


def _principled_nonnegative_init(
    weight_param: Tensor,
    bias: Optional[Tensor],
    fan_in: int,
) -> None:
    _, _, mu_b, tilde_mu, tilde_sigma = _icnn_principled_moments(fan_in)
    with torch.no_grad():
        mu_tensor = torch.as_tensor(tilde_mu, dtype=weight_param.dtype, device=weight_param.device)
        if tilde_sigma == 0.0:
            weight_param.fill_(mu_tensor)
        else:
            sigma_tensor = torch.as_tensor(tilde_sigma, dtype=weight_param.dtype, device=weight_param.device)
            noise = torch.randn_like(weight_param)
            weight_param.copy_(noise * sigma_tensor + mu_tensor)
        if bias is not None:
            bias.fill_(torch.as_tensor(mu_b, dtype=bias.dtype, device=bias.device))


class NonNegativeLinear(nn.Module):
    """Linear map with element-wise non-negative weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "principled",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.init_mode = init.lower()
        if self.init_mode not in {"principled", "xavier"}:
            raise ValueError(f"Unsupported initialisation mode '{init}' for NonNegativeLinear.")
        self.parametrisation = "exp" if self.init_mode == "principled" else "softplus"
        self.weight_param = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_mode == "principled":
            _principled_nonnegative_init(self.weight_param, self.bias, self.in_features)
        else:
            nn.init.xavier_uniform_(self.weight_param)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.parametrisation == "exp":
            weight = torch.exp(self.weight_param)
        else:
            weight = F.softplus(self.weight_param)
        return F.linear(x, weight, self.bias)

    @torch.no_grad()
    def project_non_negative(self) -> None:
        # With exp/softplus parametrisation, weights are already non-negative.
        return


class InputConvexPotential(nn.Module):
    """Fully input-convex neural network (FICNN) for low-dimensional states."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "relu",
        strong_convexity: float = 1.0,
        nonneg_init: str = "principled",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_sizes: List[int] = list(hidden_sizes)
        if len(self.hidden_sizes) == 0:
            raise ValueError("ICNN requires at least one hidden layer.")

        if activation == "relu":
            self.nonlin: nn.Module = nn.ReLU()
        elif activation == "softplus":
            self.nonlin = nn.Softplus(beta=20.0)
        else:
            raise ValueError(f"Unsupported ICNN activation: {activation}")

        self.strong_convexity = float(strong_convexity)
        self.nonneg_init = str(nonneg_init).lower()
        if self.nonneg_init not in {"principled", "xavier"}:
            raise ValueError(f"Unsupported ICNN non-negative initialiser: {nonneg_init}")

        self.z_linears = nn.ModuleList()
        self.h_linears = nn.ModuleList()
        prev_hidden = None

        for width in self.hidden_sizes:
            self.z_linears.append(nn.Linear(self.input_dim, width, bias=True))
            if prev_hidden is None:
                self.h_linears.append(None)  # type: ignore
            else:
                self.h_linears.append(
                    NonNegativeLinear(prev_hidden, width, bias=True, init=self.nonneg_init)
                )
            prev_hidden = width
        self.hidden_output = NonNegativeLinear(
            self.hidden_sizes[-1], 1, bias=True, init=self.nonneg_init
        )
        self.input_skip = nn.Linear(self.input_dim, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.z_linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.h_linears:
            if layer is not None:
                layer.reset_parameters()
        self.hidden_output.reset_parameters()
        nn.init.xavier_uniform_(self.input_skip.weight)
        nn.init.zeros_(self.input_skip.bias)

    def forward(self, z: Tensor) -> Tensor:
        batch = z.size(0)
        z_flat = z.view(batch, -1)
        h = None
        for z_linear, h_linear in zip(self.z_linears, self.h_linears):
            z_term = z_linear(z_flat)
            if h is None:
                h = self.nonlin(z_term)
            else:
                h = self.nonlin(z_term + h_linear(h))  # type: ignore[arg-type]

        assert h is not None
        quadratic = 0.5 * self.strong_convexity * (z_flat.pow(2).sum(dim=1, keepdim=True))
        output = quadratic + self.input_skip(z_flat) + self.hidden_output(h)
        return output.squeeze(-1)

    def gradient(self, z: Tensor, create_graph: bool = False) -> Tensor:
        z_in = z
        if not z_in.requires_grad:
            z_in = z_in.detach().clone().requires_grad_(True)
        potential = self.forward(z_in)
        grad = torch.autograd.grad(
            potential.sum(), z_in, create_graph=create_graph, retain_graph=create_graph
        )[0]
        return grad.view_as(z)

    @torch.no_grad()
    def project_convexity(self) -> None:
        for layer in self.h_linears:
            if layer is not None:
                layer.project_non_negative()
        self.hidden_output.project_non_negative()


class BBArmijoState:
    """Simple BB + Armijo state tracker (gradient-ascent version)."""

    def __init__(
        self,
        alpha0: float,
        alpha_min: float,
        alpha_max: float,
        ls_c: float,
        ls_shrink: float,
        ls_max_steps: int,
    ) -> None:
        self.alpha_min = float(max(alpha_min, 1e-12))
        self.alpha_max = float(max(alpha_max, self.alpha_min))
        self.alpha_prev = float(min(max(alpha0, self.alpha_min), self.alpha_max))
        self.ls_c = float(ls_c)
        self.ls_shrink = float(ls_shrink)
        self.ls_max_steps = int(max(ls_max_steps, 1))
        self.prev_params_vec: Optional[Tensor] = None
        self.prev_grad_vec: Optional[Tensor] = None

    def propose(self, params_vec: Tensor, grad_vec: Tensor) -> float:
        if (
            self.prev_params_vec is None
            or self.prev_grad_vec is None
            or self.prev_params_vec.numel() != params_vec.numel()
            or self.prev_grad_vec.numel() != grad_vec.numel()
        ):
            alpha = self.alpha_prev
        else:
            s = params_vec - self.prev_params_vec
            y = grad_vec - self.prev_grad_vec
            denom = torch.dot(s, y)
            if torch.isfinite(denom) and float(denom.abs().item()) > 1e-12:
                num = torch.dot(s, s)
                alpha = float((num / denom).item())
            else:
                alpha = self.alpha_prev
        if not math.isfinite(alpha):
            alpha = self.alpha_prev
        alpha = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return alpha

    def update_history(self, params_vec: Tensor, grad_vec: Tensor, alpha: float) -> None:
        self.prev_params_vec = params_vec.detach().clone()
        self.prev_grad_vec = grad_vec.detach().clone()
        alpha_clamped = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        self.alpha_prev = alpha_clamped


# --------------------------
#  CartPole + replay bits
# --------------------------
def cartpole_reward(state: Tensor) -> Tensor:
    """Reward r(θ) = exp(-|θ|) with θ the pole angle in radians."""
    theta = state[..., 2]
    return torch.exp(-theta.abs())


def make_cartpole(
    env_id: str,
    mass_scale: float = 1.0,
    length_scale: float = 1.0,
    gravity_scale: float = 1.0,
) -> gym.Env:
    env = gym.make(env_id)
    try:
        env.reset(seed=None)
    except TypeError:
        env.reset()
    env_unwrapped = env.unwrapped
    env_unwrapped.masspole *= mass_scale
    env_unwrapped.length *= length_scale
    env_unwrapped.gravity *= gravity_scale
    env_unwrapped.total_mass = env_unwrapped.masspole + env_unwrapped.masscart
    env_unwrapped.polemass_length = env_unwrapped.masspole * env_unwrapped.length
    return env


def to_tensor(arr: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def safe_reset(env: gym.Env, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        obs, info = env.reset(), {}
    if isinstance(obs, tuple):  # gym <0.26 compatibility
        obs, info = obs[0], {}
    return obs, info


def safe_step(env: gym.Env, action: int) -> Tuple[np.ndarray, float, bool, bool, bool, Dict]:
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
        terminated, truncated = bool(done), False
    return obs, float(reward), bool(done), bool(terminated), bool(truncated), info


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer: Deque[Transition] = collections.deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([t.state for t in batch], axis=0)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch], axis=0)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return Transition(states, actions, rewards, next_states, dones)


# --------------------------
#  Models + adversary utils
# --------------------------
class QNetwork(nn.Module):
    """Smooth Q-network (ELU activations) for compatibility with WRM theory."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = state_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ELU())  # smooth activation instead of ReLU
            prev = width
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def clamp_state(x: Tensor, low: Tensor, high: Tensor) -> Tensor:
    if not torch.isfinite(low).all() and not torch.isfinite(high).all():
        return x
    return torch.max(torch.min(x, high), low)


def adversarial_pushforward(
    icnn: InputConvexPotential,
    next_state: Tensor,
    detach_for_model: bool,
    state_low: Tensor,
    state_high: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Return T(ŝ) and delta = T(ŝ) - ŝ."""
    if detach_for_model:
        s_src = next_state.detach()
        s_leaf = s_src.clone().requires_grad_(True)
        with torch.enable_grad():
            s_push = icnn.gradient(s_leaf, create_graph=False)
        s_push = clamp_state(s_push, state_low, state_high).detach()
        delta = (s_push - s_src).detach()
        s_adv = next_state + delta
        return s_adv, delta

    s_leaf = next_state.detach().requires_grad_(True)
    with torch.enable_grad():
        s_adv = icnn.gradient(s_leaf, create_graph=True)
    s_adv = clamp_state(s_adv, state_low, state_high)
    delta = s_adv - s_leaf
    return s_adv, delta


def transport_cost(s_adv: Tensor, s_nom: Tensor) -> Tensor:
    delta = (s_adv - s_nom).view(s_adv.size(0), -1)
    return 0.5 * delta.pow(2).sum(dim=1)


def compute_adv_objective(
    icnn: InputConvexPotential,
    q_net: QNetwork,
    next_state_nom: Tensor,
    done_mask: Tensor,
    discount: float,
    transport_gamma: float,
    state_low: Tensor,
    state_high: Tensor,
    reward_fn: Callable[[Tensor], Tensor],
    ignore_q_term: bool = False,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Compute adversarial objective for ICNN parameters.

    We minimise over ω the expectation of:
        value_for_optim(s) + γ * c(s, ŝ)
    where value_for_optim is either r(s) (if ignore_q_term) or r(s)+λ max_a Q(s,a).
    """
    s_adv, _ = adversarial_pushforward(
        icnn,
        next_state_nom,
        detach_for_model=False,
        state_low=state_low,
        state_high=state_high,
    )
    q_next = q_net(s_adv)
    max_q = q_next.max(dim=1).values
    rewards = reward_fn(s_adv)

    full_value = rewards + discount * (1.0 - done_mask) * max_q
    if ignore_q_term:
        value_for_optim = rewards
    else:
        value_for_optim = full_value

    penalty = transport_gamma * transport_cost(s_adv, next_state_nom)
    adv_objective = -(value_for_optim + penalty).mean()

    metrics = {
        "robust_value": float(full_value.mean().detach().item()),
        "penalty": float(penalty.mean().detach().item()),
    }
    return adv_objective, metrics


def wrm_adversarial_state(
    next_state_nom: Tensor,
    target_net: QNetwork,
    done_mask: Tensor,
    discount: float,
    transport_gamma: float,
    state_low: Tensor,
    state_high: Tensor,
    steps: int,
    lr: float,
    ignore_q_term: bool = False,
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    """
    First-order WRM adversary via projected gradient descent on the next state.

    If ignore_q_term=True, the inner minimisation matches the Sinha et al. RL experiment:
        argmin_s { r(s) + γ c(s, ŝ) }
    otherwise it uses the full WRM objective with r(s)+λ max_a Q(s,a).
    """
    s_current = next_state_nom.detach()
    for _ in range(max(1, steps)):
        s_current.requires_grad_(True)
        q_next = target_net(s_current)
        max_q = q_next.max(dim=1).values
        rewards = cartpole_reward(s_current)

        full_value = rewards + discount * (1.0 - done_mask) * max_q
        if ignore_q_term:
            value_for_optim = rewards
        else:
            value_for_optim = full_value

        penalty = transport_gamma * transport_cost(s_current, next_state_nom)
        adv_obj = -(value_for_optim + penalty).mean()
        adv_obj.backward()
        with torch.no_grad():
            grad = s_current.grad
            if grad is None or not torch.isfinite(grad).all():
                s_next = s_current.detach()
            else:
                # Gradient ascent on adv_obj == gradient descent on value_for_optim + penalty
                s_next = s_current + lr * grad
            s_next = clamp_state(s_next, state_low, state_high)
        s_current = s_next.detach()

    with torch.no_grad():
        q_next = target_net(s_current)
        robust_value = cartpole_reward(s_current) + discount * (1.0 - done_mask) * q_next.max(dim=1).values
        penalty = transport_gamma * transport_cost(s_current, next_state_nom)
        metrics = {
            "robust_value": float(robust_value.mean().item()),
            "penalty": float(penalty.mean().item()),
        }
    delta = s_current - next_state_nom
    return s_current.detach(), delta.detach(), metrics


def epsilon_greedy(q_net: QNetwork, state: Tensor, epsilon: float, action_dim: int) -> int:
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        q_vals = q_net(state.unsqueeze(0))
    return int(q_vals.argmax(dim=1).item())


def linear_schedule(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    mix = min(1.0, step / float(decay_steps))
    return (1.0 - mix) * start + mix * end


# --------------------------
#  Training + evaluation
# --------------------------
def train_batch(
    batch: Transition,
    q_net: QNetwork,
    target_net: QNetwork,
    icnn: Optional[InputConvexPotential],
    opt_q: optim.Optimizer,
    opt_icnn: Optional[optim.Optimizer],
    device: torch.device,
    args,
    state_low: Tensor,
    state_high: Tensor,
    bb_state: Optional[BBArmijoState],
    method: str,
) -> Dict[str, float]:
    states = to_tensor(batch.state, device)
    actions = torch.as_tensor(batch.action, device=device, dtype=torch.int64)
    next_states = to_tensor(batch.next_state, device)
    done = torch.as_tensor(batch.done, device=device, dtype=torch.float32)

    adv_metrics: Dict[str, float] = {}
    delta = torch.zeros_like(next_states)

    if method == "wrm-icnn":
        if icnn is None or opt_icnn is None:
            raise ValueError("ICNN components are required for method 'wrm-icnn'.")
        for _ in range(max(1, args.icnn_ascent_steps)):
            opt_icnn.zero_grad(set_to_none=True)
            adv_obj, adv_metrics = compute_adv_objective(
                icnn,
                target_net,
                next_states,
                done,
                discount=args.discount,
                transport_gamma=args.transport_gamma,
                state_low=state_low,
                state_high=state_high,
                reward_fn=cartpole_reward,
                ignore_q_term=args.wrm_ignore_q,
            )
            # We minimise robust_value + penalty
            adv_loss = -adv_obj
            adv_loss.backward()
            grad_ok = all((p.grad is None) or torch.isfinite(p.grad).all() for p in icnn.parameters())
            if not grad_ok:
                opt_icnn.zero_grad(set_to_none=True)
                continue

            if args.icnn_step_rule == "bb-armijo" and bb_state is not None:
                params_vec = parameters_to_vector([p.detach().clone() for p in icnn.parameters()])
                grad_tensors = []
                for group in opt_icnn.param_groups:
                    wd = float(group.get("weight_decay", 0.0))
                    for param in group["params"]:
                        if param.grad is None:
                            grad_tensors.append(torch.zeros_like(param))
                            continue
                        grad_tensor = param.grad.detach().clone()
                        if wd != 0.0:
                            grad_tensor = grad_tensor + wd * param.detach()
                        grad_tensors.append(grad_tensor)
                grad_vec = parameters_to_vector(grad_tensors)
                grad_norm_sq = float(grad_vec.pow(2).sum().item())
                if grad_norm_sq <= 0.0 or (not math.isfinite(grad_norm_sq)):
                    bb_state.update_history(params_vec, grad_vec, bb_state.alpha_prev)
                    continue

                alpha = bb_state.propose(params_vec, grad_vec)
                params_backup = [p.detach().clone() for p in icnn.parameters()]
                accepted = False
                adv_obj_val = float(adv_obj.detach().item())

                for _ in range(bb_state.ls_max_steps):
                    for group in opt_icnn.param_groups:
                        group["lr"] = alpha
                    opt_icnn.step()
                    icnn.project_convexity()
                    with torch.no_grad():
                        adv_candidate, _ = compute_adv_objective(
                            icnn,
                            target_net,
                            next_states,
                            done,
                            args.discount,
                            args.transport_gamma,
                            state_low,
                            state_high,
                            cartpole_reward,
                            ignore_q_term=args.wrm_ignore_q,
                        )
                    improvement = float(adv_candidate.item() - adv_obj_val)
                    sufficient = improvement >= bb_state.ls_c * alpha * grad_norm_sq
                    if sufficient and math.isfinite(improvement):
                        accepted = True
                        adv_obj_val = float(adv_candidate.item())
                        break
                    alpha *= bb_state.ls_shrink
                    for param, backup in zip(icnn.parameters(), params_backup):
                        param.data.copy_(backup)

                bb_state.update_history(params_vec, grad_vec, alpha if accepted else bb_state.alpha_prev)
                if not accepted:
                    for param, backup in zip(icnn.parameters(), params_backup):
                        param.data.copy_(backup)
            else:
                torch.nn.utils.clip_grad_norm_(icnn.parameters(), max_norm=1.0)
                opt_icnn.step()
                icnn.project_convexity()

        with torch.no_grad():
            s_adv, delta = adversarial_pushforward(
                icnn,
                next_states,
                detach_for_model=True,
                state_low=state_low,
                state_high=state_high,
            )
    elif method == "wrm":
        s_adv, delta, adv_metrics = wrm_adversarial_state(
            next_states,
            target_net,
            done,
            discount=args.discount,
            transport_gamma=args.transport_gamma,
            state_low=state_low,
            state_high=state_high,
            steps=args.wrm_steps,
            lr=args.wrm_lr,
            ignore_q_term=args.wrm_ignore_q,
        )
    elif method == "erm":
        s_adv = next_states.detach()
        with torch.no_grad():
            q_next_nom = target_net(s_adv)
            robust_value = cartpole_reward(s_adv) + args.discount * (1.0 - done) * q_next_nom.max(dim=1).values
            adv_metrics = {"robust_value": float(robust_value.mean().item()), "penalty": 0.0}
    else:
        raise ValueError(f"Unsupported training method: {method}")

    with torch.no_grad():
        q_next = target_net(s_adv)
        target = cartpole_reward(s_adv) + args.discount * (1.0 - done) * q_next.max(dim=1).values

    q_pred = q_net(states).gather(1, actions.view(-1, 1)).squeeze(1)
    td_loss = F.mse_loss(q_pred, target)

    opt_q.zero_grad(set_to_none=True)
    td_loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
    opt_q.step()

    metrics = {
        "td_loss": float(td_loss.item()),
        "delta_mean": float(delta.norm(dim=1).mean().item()),
    }
    metrics.update(adv_metrics)
    return metrics


def evaluate_policy(
    envs: Dict[str, gym.Env],
    q_net: QNetwork,
    device: torch.device,
    episodes: int,
    max_steps: int,
    epsilon_eval: float,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a policy on a collection of environments.

    Returns, for each env name, a dict with:
        * mean_steps:     mean episode length (CartPole-style metric)
        * mean_reward:    mean shaped reward (r(θ)=exp(-|θ|))
    """
    results: Dict[str, Dict[str, float]] = {}
    for name, env in envs.items():
        rewards: List[float] = []
        lengths: List[int] = []
        for _ in range(episodes):
            state_np, _ = safe_reset(env, seed=None)
            episode_reward = 0.0
            steps = 0
            for _ in range(max_steps):
                state = to_tensor(state_np, device)
                action = epsilon_greedy(q_net, state, epsilon_eval, env.action_space.n)
                next_state, _reward_env, done, _terminated, _truncated, _ = safe_step(env, action)
                shaped_reward = float(cartpole_reward(to_tensor(next_state, device)).item())
                episode_reward += shaped_reward
                steps += 1
                state_np = next_state
                if done:
                    break
            rewards.append(episode_reward)
            lengths.append(steps)
        results[name] = {
            "mean_steps": float(np.mean(lengths)),
            "mean_shaped_reward": float(np.mean(rewards)),
        }
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Robust Q-learning on CartPole with ICNN (WRM-ICNN), gradient WRM, or ERM."
    )
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    # Training horizon: Sinha et al. train for ~2000 episodes in Fig. 8.
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--train-every", type=int, default=1, help="Gradient updates per env step.")
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--lr-q", type=float, default=3e-4)
    parser.add_argument("--hidden-q", type=int, nargs="+", default=[256, 256])
    parser.add_argument(
        "--method",
        type=str,
        choices=["wrm-icnn", "wrm", "erm", "all"],
        default="wrm-icnn",
        help="Training method: WRM with ICNN transport, WRM via gradient ascent, ERM baseline, or run all sequentially.",
    )
    parser.add_argument(
        "--transport-gamma",
        type=float,
        default=2.0,
        help="Penalty weight γ on ½||T(s)-ŝ||^2 (smaller => larger adversarial budget).",
    )
    parser.add_argument("--icnn-hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--icnn-activation", type=str, choices=["relu", "softplus"], default="softplus")
    parser.add_argument("--icnn-strong-convexity", type=float, default=0.5)
    parser.add_argument("--icnn-init", type=str, choices=["principled", "xavier"], default="principled")
    parser.add_argument("--lr-omega", type=float, default=1e-3)
    parser.add_argument(
        "--icnn-ascent-steps",
        type=int,
        default=5,
        help="Number of ICNN updates per Q update (inner minimisation iterations).",
    )
    parser.add_argument(
        "--icnn-optimizer",
        type=str,
        choices=["adam", "ademamix", "sgd"],
        default="sgd",
        help="Optimizer for ICNN parameters.",
    )
    parser.add_argument(
        "--icnn-step-rule",
        type=str,
        choices=["constant", "bb-armijo"],
        default="bb-armijo",
        help="Learning-rate rule for ICNN (BB-Armijo only used with SGD).",
    )
    parser.add_argument("--icnn-alpha0", type=float, default=5e-3)
    parser.add_argument("--icnn-alpha-min", type=float, default=1e-6)
    parser.add_argument("--icnn-alpha-max", type=float, default=0.5)
    parser.add_argument("--icnn-ls-c", type=float, default=0.1)
    parser.add_argument("--icnn-ls-shrink", type=float, default=0.5)
    parser.add_argument("--icnn-ls-max-steps", type=int, default=10)

    parser.add_argument(
        "--wrm-steps",
        type=int,
        default=15,
        help="Inner gradient steps for WRM (no ICNN), cf. T_adv≈15 in the original paper.",
    )
    parser.add_argument(
        "--wrm-lr",
        type=float,
        default=0.1,
        help="Inner gradient step size for WRM (no ICNN).",
    )
    parser.add_argument(
        "--wrm-ignore-q",
        action="store_true",
        help="If set, inner WRM objective ignores the Q term (matches Sinha et al. 5.3 tabular experiment).",
    )

    parser.add_argument("--epsilon-start", type=float, default=0.3)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument(
        "--epsilon-decay",
        type=int,
        default=100_000,
        help="Linear decay steps for epsilon-greedy exploration.",
    )
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _build_envs(args, seed: int) -> Tuple[gym.Env, Dict[str, gym.Env]]:
    env = make_cartpole(args.env_id)
    env.action_space.seed(seed)
    safe_reset(env, seed=seed)
    eval_envs = {
        "nominal": make_cartpole(args.env_id),
        "heavy_pole": make_cartpole(args.env_id, mass_scale=2.0),
        "light_pole": make_cartpole(args.env_id, mass_scale=0.5),
        "short_pole": make_cartpole(args.env_id, length_scale=0.5),
        "long_pole": make_cartpole(args.env_id, length_scale=2.0),
        "strong_gravity": make_cartpole(args.env_id, gravity_scale=5.0),
        "weak_gravity": make_cartpole(args.env_id, gravity_scale=0.2),
    }
    return env, eval_envs


def run_single_method(method: str, args, device: torch.device) -> Dict[str, float]:
    _set_seeds(args.seed)
    env, eval_envs = _build_envs(args, args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    state_low = to_tensor(env.observation_space.low, device)
    state_high = to_tensor(env.observation_space.high, device)
    state_low[~torch.isfinite(state_low)] = -1e6
    state_high[~torch.isfinite(state_high)] = 1e6

    q_net = QNetwork(state_dim, action_dim, args.hidden_q).to(device)
    target_net = QNetwork(state_dim, action_dim, args.hidden_q).to(device)
    target_net.load_state_dict(q_net.state_dict())

    icnn: Optional[InputConvexPotential] = None
    opt_icnn: Optional[optim.Optimizer] = None
    bb_state: Optional[BBArmijoState] = None
    if method == "wrm-icnn":
        icnn = InputConvexPotential(
            input_dim=state_dim,
            hidden_sizes=args.icnn_hidden,
            activation=args.icnn_activation,
            strong_convexity=args.icnn_strong_convexity,
            nonneg_init=args.icnn_init,
        ).to(device)
        icnn_param_groups = []
        for name, param in icnn.named_parameters():
            if not param.requires_grad:
                continue
            decay = 0.0 if ("weight_param" in name or "bias" in name) else 1e-4
            icnn_param_groups.append({"params": [param], "weight_decay": decay})
        sgd_lr = args.icnn_alpha0 if args.icnn_step_rule == "bb-armijo" else args.lr_omega
        if args.icnn_optimizer == "adam":
            opt_icnn = optim.Adam(icnn_param_groups, lr=args.lr_omega)
        elif args.icnn_optimizer == "ademamix":
            opt_icnn = AdEMAMix(
                icnn_param_groups,
                lr=args.lr_omega,
                betas=(0.9, 0.999, 0.9999),
                alpha=4.0,
            )
        else:
            opt_icnn = optim.SGD(icnn_param_groups, lr=sgd_lr, momentum=0.0)

        if args.icnn_step_rule == "bb-armijo":
            bb_state = BBArmijoState(
                alpha0=args.icnn_alpha0,
                alpha_min=args.icnn_alpha_min,
                alpha_max=args.icnn_alpha_max,
                ls_c=args.icnn_ls_c,
                ls_shrink=args.icnn_ls_shrink,
                ls_max_steps=args.icnn_ls_max_steps,
            )

    opt_q = optim.Adam(q_net.parameters(), lr=args.lr_q)
    buffer = ReplayBuffer(args.buffer_size)
    total_steps = 0
    total_updates = 0

    progress = tqdm(range(args.episodes), desc=f"{method} episodes")

    for episode in progress:
        state_np, _ = safe_reset(env, seed=None)
        episode_reward = 0.0
        episode_len = 0
        episode_updates = 0
        metric_sums: Dict[str, float] = collections.defaultdict(float)
        last_epsilon = args.epsilon_start

        for _ in range(args.max_steps):
            epsilon = linear_schedule(total_steps, args.epsilon_start, args.epsilon_end, args.epsilon_decay)
            last_epsilon = epsilon
            state = to_tensor(state_np, device)
            action = epsilon_greedy(q_net, state, epsilon, action_dim)
            next_state, _reward_env, done, _terminated, _truncated, _ = safe_step(env, action)
            reward = float(cartpole_reward(to_tensor(next_state, device)).item())

            buffer.push(state_np, action, reward, next_state, done)
            state_np = next_state
            episode_reward += reward
            episode_len += 1
            total_steps += 1

            if (
                len(buffer) >= args.batch_size
                and total_steps > args.warmup_steps
                and total_steps % max(1, args.train_every) == 0
            ):
                for _ in range(args.updates_per_step):
                    batch = buffer.sample(args.batch_size)
                    batch_metrics = train_batch(
                        batch,
                        q_net,
                        target_net,
                        icnn,
                        opt_q,
                        opt_icnn,
                        device,
                        args,
                        state_low,
                        state_high,
                        bb_state,
                        method=method,
                    )
                    episode_updates += 1
                    total_updates += 1
                    for k, v in batch_metrics.items():
                        metric_sums[k] += v

            if total_steps % args.target_update == 0:
                target_net.load_state_dict(q_net.state_dict())
            if done:
                break

        # Average metrics over this episode's optimisation steps
        avg_metrics: Dict[str, float] = {}
        if episode_updates > 0:
            for k, v in metric_sums.items():
                avg_metrics[k] = v / float(episode_updates)

        postfix: Dict[str, object] = {
            "method": method,
            "eps": f"{last_epsilon:.3f}",
            "ep_len": episode_len,
            "ep_rew": f"{episode_reward:.2f}",
            "updates": total_updates,
        }
        if avg_metrics:
            postfix.update(
                {
                    "td": f"{avg_metrics.get('td_loss', float('nan')):.3g}",
                    "delta": f"{avg_metrics.get('delta_mean', float('nan')):.3g}",
                    "robust_v": f"{avg_metrics.get('robust_value', float('nan')):.3g}",
                    "pen": f"{avg_metrics.get('penalty', float('nan')):.3g}",
                }
            )

        if (episode + 1) % args.eval_every == 0 or episode == 0:
            eval_scores = evaluate_policy(
                eval_envs,
                q_net,
                device,
                episodes=args.eval_episodes,
                max_steps=args.max_steps,
                epsilon_eval=0.05,
            )
            postfix.update(
                {
                    "nom_len": f"{eval_scores['nominal']['mean_steps']:.1f}",
                    "heavy_len": f"{eval_scores['heavy_pole']['mean_steps']:.1f}",
                    "strong_g": f"{eval_scores['strong_gravity']['mean_steps']:.1f}",
                }
            )

        progress.set_postfix(postfix, refresh=False)

    # Final evaluation with greedy policy
    final_eval = evaluate_policy(
        eval_envs,
        q_net,
        device,
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
        epsilon_eval=0.0,
    )
    final_scores: Dict[str, float] = {}
    print(f"\nFinal evaluation — {method}:")
    for name, metrics in final_eval.items():
        mean_len = metrics["mean_steps"]
        mean_rew = metrics["mean_shaped_reward"]
        print(f"  {name:15s}: len={mean_len:6.2f}, shaped_reward={mean_rew:8.3f}")
        final_scores[name] = mean_len
    return final_scores


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    methods = ["wrm-icnn", "wrm", "erm"] if args.method == "all" else [args.method]
    all_results: Dict[str, Dict[str, float]] = {}

    for method in methods:
        results = run_single_method(method, args, device)
        all_results[method] = results

    if len(methods) > 1:
        env_names = list(next(iter(all_results.values())).keys())
        print("\nComparison (mean episode length over eval runs):")
        header = ["env"] + methods
        print(" | ".join(f"{h:>15s}" for h in header))
        for env_name in env_names:
            row_vals = [env_name]
            for m in methods:
                row_vals.append(f"{all_results[m][env_name]:15.3f}")
            print(" | ".join(row_vals))


if __name__ == "__main__":
    main()
