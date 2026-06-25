"""Load a pretrained CIFAR-10 classifier.

We delegate to the existing :mod:`pretrained_LAT.load_pretrained_resnet18`
loader (which already understands the checkpoint formats produced by the
EPFL utils-style backbones) but wrap it so callers in this package don't
have to know about that legacy module. This module also supports the
``vit_exp`` SimpleViT checkpoints used by the same training code.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the repository root importable so we can reuse the existing loaders
# without copying the entire ``model.py`` / ``pretrained_LAT.py`` files.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pretrained_LAT import load_pretrained_resnet18  # type: ignore  # noqa: E402
from utils import looks_like_state_dict, unwrap_state_dict  # type: ignore  # noqa: E402

_VIT_CHECKPOINT_KEY_ENV = "PRETRAINED_VIT_STATE_KEY"
_VIT_CHECKPOINT_KEY_FALLBACK_ENV = "VIT_CHECKPOINT_KEY"
_VIT_STATE_PRIORITY = ("swa_last", "last", "best", "swa_best")


def _checkpoint_looks_like_html(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            prefix = f.read(64).strip().lower()
        return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")
    except OSError:
        return False


def _extract_state_dict(raw: Any) -> Optional[Dict[str, torch.Tensor]]:
    try:
        state = unwrap_state_dict(raw)
    except Exception:
        return None
    return state if looks_like_state_dict(state) else None


def _looks_like_simple_vit_state_dict(state: Dict[str, torch.Tensor]) -> bool:
    return all(
        key in state
        for key in (
            "to_patch_embedding.1.weight",
            "transformer.layers.0.0.norm.weight",
            "linear_head.1.weight",
        )
    )


def _extract_named_vit_state_dict(
    raw: Any,
    key: str,
) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(raw, dict) or key not in raw:
        return None
    state = _extract_state_dict(raw[key])
    if state is None or not _looks_like_simple_vit_state_dict(state):
        return None
    return state


def _extract_vit_state_dict(
    raw: Any,
    preferred_key: Optional[str] = None,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[str]]:
    if preferred_key:
        state = _extract_named_vit_state_dict(raw, preferred_key)
        if state is None:
            raise RuntimeError(
                f"Requested ViT checkpoint key {preferred_key!r} was not found "
                "or is not a supported SimpleViT state dict."
            )
        return state, preferred_key

    for key in _VIT_STATE_PRIORITY:
        state = _extract_named_vit_state_dict(raw, key)
        if state is not None:
            return state, key

    state = _extract_state_dict(raw)
    if state is not None and _looks_like_simple_vit_state_dict(state):
        return state, None
    return None, None


def _posemb_sincos_2d(
    h: int,
    w: int,
    dim: int,
    *,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("feature dimension must be a multiple of 4 for sincos emb")
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class _PatchRearrange(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        if x.size(-1) % p != 0 or x.size(-2) % p != 0:
            raise ValueError(f"Image size {tuple(x.shape[-2:])} is not divisible by patch size {p}.")
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        return x.reshape(b, (h // p) * (w // p), p * p * c)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = int(heads)
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        b, n, _ = q.shape
        q = q.reshape(b, n, self.heads, -1).transpose(1, 2)
        k = k.reshape(b, n, self.heads, -1).transpose(1, 2)
        v = v.reshape(b, n, self.heads, -1).transpose(1, 2)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class _TransformerCompat(nn.Module):
    """SimpleViT transformer without final norm, matching older checkpoints."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [_Attention(dim, heads=heads, dim_head=dim_head), _FeedForward(dim, mlp_dim)]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SimpleViTCIFAR(nn.Module):
    """Dependency-free SimpleViT compatible with the CIFAR-10 ``vit_exp`` checkpoint."""

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        patch_dim = channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            _PatchRearrange(patch_size),
            nn.Linear(patch_dim, dim),
        )
        grid = image_size // patch_size
        self.register_buffer(
            "pos_embedding",
            _posemb_sincos_2d(grid, grid, dim),
            persistent=False,
        )
        self.transformer = _TransformerCompat(dim, depth, heads, dim_head, mlp_dim)
        self.pool = "mean"
        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding.to(device=img.device, dtype=x.dtype)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.to_latent(x)
        return self.linear_head(x)


def _infer_simple_vit_config(state: Dict[str, torch.Tensor], num_classes: int) -> Dict[str, int]:
    patch_w = state["to_patch_embedding.1.weight"]
    dim = int(patch_w.shape[0])
    patch_dim = int(patch_w.shape[1])
    patch_size = int(math.isqrt(patch_dim // 3))
    if patch_size * patch_size * 3 != patch_dim:
        raise RuntimeError(f"Cannot infer square RGB patch size from patch_dim={patch_dim}.")
    depth = (
        max(
            int(k.split(".")[2])
            for k in state
            if k.startswith("transformer.layers.") and k.split(".")[2].isdigit()
        )
        + 1
    )
    qkv = state["transformer.layers.0.0.to_qkv.weight"]
    to_out = state["transformer.layers.0.0.to_out.weight"]
    heads_dim_prod = int(to_out.shape[1])
    dim_head = 64 if heads_dim_prod % 64 == 0 else max(1, heads_dim_prod)
    heads = max(1, heads_dim_prod // dim_head)
    ff_w = state["transformer.layers.0.1.net.1.weight"]
    mlp_dim = int(ff_w.shape[0])
    head_w = state["linear_head.1.weight"]
    if int(head_w.shape[0]) != int(num_classes) or int(qkv.shape[1]) != dim:
        raise RuntimeError("ViT checkpoint dimensions do not match the requested classifier.")
    return {
        "image_size": 32,
        "patch_size": patch_size,
        "num_classes": int(num_classes),
        "dim": dim,
        "depth": depth,
        "heads": heads,
        "mlp_dim": mlp_dim,
        "dim_head": dim_head,
    }


def load_pretrained_vit(
    pretrained_path: str,
    num_classes: int = 10,
    strict: bool = False,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    ckpt_path = Path(pretrained_path)
    if _checkpoint_looks_like_html(ckpt_path):
        raise RuntimeError(
            f"Checkpoint at {ckpt_path} looks like an HTML page; the download likely failed."
        )
    raw = torch.load(ckpt_path, map_location=device)
    preferred_key = os.environ.get(_VIT_CHECKPOINT_KEY_ENV) or os.environ.get(
        _VIT_CHECKPOINT_KEY_FALLBACK_ENV
    )
    state, state_key = _extract_vit_state_dict(raw, preferred_key=preferred_key)
    if state is None:
        raise RuntimeError(f"Could not find a supported SimpleViT state dict in {ckpt_path}.")

    model = SimpleViTCIFAR(**_infer_simple_vit_config(state, num_classes))
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"[ViT-simple] Missing keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"[ViT-simple] Unexpected keys ({len(unexpected)}): {unexpected}")
    key_note = f" key={state_key!r}" if state_key is not None else ""
    print(f"[ViT-simple] Loaded vit_exp-compatible weights{key_note} from: {ckpt_path}")
    return model.to(device)


def load_pretrained_classifier(
    pretrained_path: str,
    num_classes: int = 10,
    strict: bool = False,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Return a ``nn.Module`` mapping normalized CIFAR inputs -> class logits."""
    path = Path(pretrained_path)
    prefer_vit = "vit" in path.stem.lower()
    if prefer_vit:
        return load_pretrained_vit(
            pretrained_path=pretrained_path,
            num_classes=num_classes,
            strict=strict,
            device=device,
        )

    base = load_pretrained_resnet18(
        pretrained_path=pretrained_path,
        num_classes=num_classes,
        strict=strict,
        device=device,
    )
    return base.to(device)
