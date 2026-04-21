"""Flatten/unflatten helpers for functional_call-style parameter vectors."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def flatten_params(
    module: nn.Module,
) -> Tuple[torch.Tensor, Tuple[Tuple[str, Tuple[int, ...], int], ...]]:
    names, shapes, tensors = [], [], []
    for name, param in module.named_parameters():
        names.append(name)
        shapes.append(param.shape)
        tensors.append(param.detach().reshape(-1))
    vec = torch.cat(tensors)
    meta = tuple(
        (n, tuple(s), int(torch.prod(torch.tensor(s)).item()))
        for n, s in zip(names, shapes)
    )
    return vec, meta


def unflatten_vector(
    vec: torch.Tensor,
    meta: Tuple[Tuple[str, Tuple[int, ...], int], ...],
) -> Dict[str, torch.Tensor]:
    params = {}
    offset = 0
    for name, shape, size in meta:
        slice_view = vec[offset:offset + size].view(shape)
        params[name] = slice_view
        offset += size
    return params
