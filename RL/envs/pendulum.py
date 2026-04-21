"""Vectorised swing-up pendulum dynamics in Torch with per-env physics parameters xi."""
from __future__ import annotations

import math
import random
from typing import Optional

import torch

from utils.common import make_generator


class VecSwingUpPendulumTorch:
    """Vectorized swing-up pendulum in Torch (underactuated).

    Physics parameters (xi):
      xi[:,0] = m : bob mass
      xi[:,1] = l : rod length
      xi[:,2] = b : viscous friction coefficient

    State: [theta, theta_dot] (2D), theta=0 hanging down, theta=pi upright.
    Action: discrete torque index mapped to continuous torque u via `action_values`.

    Dynamics:
      m l^2 theta_ddot + b theta_dot + m g l sin(theta) = u

    Reward (dense shaped):
      (1 + cos(wrap(theta - pi))) / 2 — 0 (hanging) to 1 (upright).
    """

    def __init__(
        self,
        max_steps: int = 500,
        device: torch.device = torch.device("cpu"),
        *,
        dt: float = 0.1,
        u_max: float = 8.0,
        g: float = 9.8,
        max_speed: float = 8.0,
        theta_tol: float = 0.2,
        vel_tol: float = 1.0,
        three_actions: bool = True,
    ):
        self.device = device
        self.obs_dim = 2
        self.act_dim = 3 if bool(three_actions) else 2

        self.gravity = torch.tensor(float(g), device=device)
        self.dt = torch.tensor(float(dt), device=device)
        self.u_max = torch.tensor(float(u_max), device=device)
        self.max_speed = torch.tensor(float(max_speed), device=device)

        self.theta_tol = torch.tensor(float(theta_tol), device=device)
        self.vel_tol = torch.tensor(float(vel_tol), device=device)

        if self.act_dim == 3:
            base = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=torch.float32)
        else:
            base = torch.tensor([-1.0, 1.0], device=device, dtype=torch.float32)
        self.action_values = base * self.u_max

        self.max_steps = int(max_steps)

        self.state = None  # (B,2)
        self.steps = None  # (B,) int32

    @staticmethod
    def angle_normalize(theta: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * math.pi
        return torch.remainder(theta + math.pi, two_pi) - math.pi

    def reset(self, batch_size: int, seed: Optional[int] = None) -> torch.Tensor:
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        g = make_generator(seed, self.device)
        theta = (torch.rand(batch_size, generator=g, device=self.device) * 0.2 - 0.1).to(torch.float32)
        theta_dot = (torch.rand(batch_size, generator=g, device=self.device) * 0.2 - 0.1).to(torch.float32)
        self.state = torch.stack([theta, theta_dot], dim=-1)
        self.steps = torch.zeros(batch_size, device=self.device, dtype=torch.int32)
        return self.state

    def reset_done(self, done: torch.Tensor, seed: int):
        if not torch.any(done):
            return
        idx = torch.nonzero(done, as_tuple=False).squeeze(-1)
        g = make_generator(seed, self.device)
        theta = (torch.rand(idx.numel(), generator=g, device=self.device) * 0.2 - 0.1).to(torch.float32)
        theta_dot = (torch.rand(idx.numel(), generator=g, device=self.device) * 0.2 - 0.1).to(torch.float32)
        self.state[idx] = torch.stack([theta, theta_dot], dim=-1)
        self.steps[idx] = 0

    def _step_dynamics(self, u: torch.Tensor, xi: torch.Tensor):
        theta, theta_dot = self.state[:, 0], self.state[:, 1]

        m = xi[:, 0].clamp_min(1e-6)
        l = xi[:, 1].clamp_min(1e-6)
        b = xi[:, 2].clamp_min(0.0)

        u = torch.clamp(u, -self.u_max, self.u_max)

        thetaacc = (u - b * theta_dot - m * self.gravity * l * torch.sin(theta)) / (m * l * l)
        theta_dot = theta_dot + self.dt * thetaacc
        theta_dot = torch.clamp(theta_dot, -self.max_speed, self.max_speed)
        theta = theta + self.dt * theta_dot
        theta = self.angle_normalize(theta)

        self.state = torch.stack([theta, theta_dot], dim=-1)

        self.steps += 1
        truncated = self.steps >= self.max_steps
        terminated = torch.zeros_like(truncated, dtype=torch.bool)
        done = truncated

        delta = self.angle_normalize(theta - math.pi)
        reward = (1.0 + torch.cos(delta)) * 0.5

        info = {"terminated": terminated, "truncated": truncated}
        return self.state, reward, done, info

    def step(self, action: torch.Tensor, xi: torch.Tensor):
        u = self.action_values[action]
        return self._step_dynamics(u, xi)

    def step_continuous(self, u: torch.Tensor, xi: torch.Tensor):
        return self._step_dynamics(u, xi)
