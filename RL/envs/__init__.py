"""Vectorised Torch environments with per-env physics parameters xi."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import torch

from envs.cartpole import VecParamCartPoleTorch
from envs.pendulum import VecSwingUpPendulumTorch


class VecEnvTorch(Protocol):
    obs_dim: int
    act_dim: int
    state: torch.Tensor

    def reset(self, batch_size: int, seed: Optional[int] = None) -> torch.Tensor: ...
    def reset_done(self, done: torch.Tensor, seed: int) -> None: ...
    def step(self, action: torch.Tensor, xi: torch.Tensor): ...
    def step_continuous(self, u: torch.Tensor, xi: torch.Tensor): ...


@dataclass(frozen=True)
class EnvSpec:
    name: str
    xi_names: Tuple[str, ...]
    xi_low: Tuple[float, ...]
    xi_high: Tuple[float, ...]

    @property
    def xi_dim(self) -> int:
        return len(self.xi_names)


ENV_SPECS: Dict[str, EnvSpec] = {
    "cartpole": EnvSpec(
        name="cartpole",
        xi_names=("masspole", "length"),
        xi_low=(0.05, 0.25),
        xi_high=(0.2, 0.75),
    ),
    "swingup_pendulum": EnvSpec(
        name="swingup_pendulum",
        xi_names=("m", "l", "b"),
        xi_low=(0.5, 0.5, 0.0),
        xi_high=(1.5, 1.5, 0.2),
    ),
}


def canonical_env_name(name: str) -> str:
    n = str(name).strip().lower()
    if n in {"cartpole"}:
        return "cartpole"
    if n in {"pendulum", "swingup", "swingup_pendulum", "swingup-pendulum"}:
        return "swingup_pendulum"
    raise ValueError(f"Unknown env '{name}'. Choices: {sorted(ENV_SPECS.keys())}")


def make_vec_env(
    env_name: str,
    *,
    max_steps: int,
    device: torch.device,
    pendulum_dt: float = 0.1,
    pendulum_u_max: float = 8.0,
    pendulum_max_speed: float = 8.0,
    pendulum_theta_tol: float = 0.2,
    pendulum_vel_tol: float = 1.0,
    pendulum_actions: int = 3,
) -> VecEnvTorch:
    env_name = canonical_env_name(env_name)
    if env_name == "cartpole":
        return VecParamCartPoleTorch(max_steps=int(max_steps), device=device)
    if env_name == "swingup_pendulum":
        return VecSwingUpPendulumTorch(
            max_steps=int(max_steps),
            device=device,
            dt=float(pendulum_dt),
            u_max=float(pendulum_u_max),
            max_speed=float(pendulum_max_speed),
            theta_tol=float(pendulum_theta_tol),
            vel_tol=float(pendulum_vel_tol),
            three_actions=(int(pendulum_actions) == 3),
        )
    raise ValueError(f"Unsupported env '{env_name}'.")


__all__ = [
    "VecEnvTorch",
    "EnvSpec",
    "ENV_SPECS",
    "canonical_env_name",
    "make_vec_env",
    "VecParamCartPoleTorch",
    "VecSwingUpPendulumTorch",
]
