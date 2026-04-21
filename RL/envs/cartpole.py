"""Vectorised CartPole dynamics in Torch with per-env physics parameters xi."""
from __future__ import annotations

import math
import random
from typing import Optional

import torch

from utils.common import make_generator


class VecParamCartPoleTorch:
    """Vectorized CartPole dynamics in Torch.

    Supports per-env physics parameters:
      xi[:,0] = masspole
      xi[:,1] = length (half-length)

    State:  [x, x_dot, theta, theta_dot]
    Action: 0 (left), 1 (right)
    Reward: 1 if not terminated else 0  (matches classic CartPole).
    """

    def __init__(self, max_steps: int = 500, device: torch.device = torch.device("cpu")):
        self.device = device
        self.obs_dim = 4
        self.act_dim = 2

        self.gravity = torch.tensor(9.8, device=device)
        self.masscart = torch.tensor(1.0, device=device)
        self.force_mag = torch.tensor(10.0, device=device)
        self.tau = torch.tensor(0.02, device=device)
        self.action_values = torch.tensor([-1.0, 1.0], device=device, dtype=torch.float32) * self.force_mag

        self.x_threshold = torch.tensor(2.4, device=device)
        self.theta_threshold_radians = torch.tensor(12 * math.pi / 180, device=device)

        self.max_steps = int(max_steps)

        self.state = None   # (B,4)
        self.steps = None   # (B,) int32

    def reset(self, batch_size: int, seed: Optional[int] = None) -> torch.Tensor:
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        g = make_generator(seed, self.device)
        self.state = (torch.rand(batch_size, 4, generator=g, device=self.device) * 0.1 - 0.05).to(torch.float32)
        self.steps = torch.zeros(batch_size, device=self.device, dtype=torch.int32)
        return self.state

    def reset_done(self, done: torch.Tensor, seed: int):
        if not torch.any(done):
            return
        idx = torch.nonzero(done, as_tuple=False).squeeze(-1)
        g = make_generator(seed, self.device)
        new_states = (torch.rand(idx.numel(), 4, generator=g, device=self.device) * 0.1 - 0.05).to(torch.float32)
        self.state[idx] = new_states
        self.steps[idx] = 0

    def _step_dynamics(self, force: torch.Tensor, xi: torch.Tensor):
        x, x_dot, theta, theta_dot = self.state[:, 0], self.state[:, 1], self.state[:, 2], self.state[:, 3]

        m_p = xi[:, 0].clamp_min(1e-6)
        l = xi[:, 1].clamp_min(1e-6)

        total_mass = self.masscart + m_p
        polemass_length = m_p * l

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            l * (4.0 / 3.0 - m_p * costheta * costheta / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = torch.stack([x, x_dot, theta, theta_dot], dim=-1)

        self.steps += 1
        terminated = (
            (x < -self.x_threshold) | (x > self.x_threshold) |
            (theta < -self.theta_threshold_radians) | (theta > self.theta_threshold_radians)
        )
        truncated = self.steps >= self.max_steps
        done = terminated | truncated

        reward = torch.where(terminated, torch.zeros_like(x), torch.ones_like(x))
        info = {"terminated": terminated, "truncated": truncated}
        return self.state, reward, done, info

    def step(self, action: torch.Tensor, xi: torch.Tensor):
        force = self.action_values[action]
        return self._step_dynamics(force, xi)

    def step_force(self, force: torch.Tensor, xi: torch.Tensor):
        return self._step_dynamics(force, xi)

    def step_continuous(self, u: torch.Tensor, xi: torch.Tensor):
        return self.step_force(u, xi)
