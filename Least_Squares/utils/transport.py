"""Gradient-of-potential transport map: T_omega(z) = grad_z psi_omega(z)."""
from __future__ import annotations

import torch
import torch.nn as nn


def transport_map(
    z: torch.Tensor,
    psi: nn.Module,
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute T(z) = grad_z psi(z)."""
    z_in = z.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        val = psi(z_in).sum()
        grad = torch.autograd.grad(val, z_in, create_graph=create_graph)[0]
    return grad.view_as(z)


T_omega = transport_map
