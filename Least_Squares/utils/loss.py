"""Least-squares loss and its gradients w.r.t. theta and xi."""
from __future__ import annotations

import torch


def loss_function(
    theta: torch.Tensor,
    xi: torch.Tensor,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
    dim_m: int,
) -> torch.Tensor:
    """f_theta(xi) = || A(xi) theta - b ||^2 / m, where A(xi) = A0 + xi A1."""
    xi = xi.reshape(-1)
    A_z = A0.unsqueeze(0) + xi.view(-1, 1, 1) * A1.unsqueeze(0)
    residual = torch.matmul(A_z, theta) - b.view(1, -1)
    return (residual ** 2).sum(dim=1) / float(dim_m)


def loss_grad_theta(
    theta: torch.Tensor,
    xi: torch.Tensor,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    xi = xi.reshape(-1)
    A_z = A0.unsqueeze(0) + xi.view(-1, 1, 1) * A1.unsqueeze(0)
    residual = torch.matmul(A_z, theta) - b.view(1, -1)
    dim_m = float(A0.size(0))
    grad = 2.0 * torch.matmul(
        A_z.transpose(1, 2),
        residual.unsqueeze(2),
    ).squeeze(2)
    return grad / dim_m


def loss_grad_xi(
    theta: torch.Tensor,
    xi: torch.Tensor,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    xi = xi.reshape(-1)
    A_z = A0.unsqueeze(0) + xi.view(-1, 1, 1) * A1.unsqueeze(0)
    residual = torch.matmul(A_z, theta) - b.view(1, -1)
    grad_Az = torch.matmul(A1, theta)
    dim_m = float(A0.size(0))
    return 2.0 * (residual * grad_Az.view(1, -1)).sum(dim=1) / dim_m
