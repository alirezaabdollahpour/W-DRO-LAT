#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a saved WRM-LAT backbone checkpoint on CIFAR-10, CIFAR-10.1 (v6), CIFAR-10.2, and CIFAR-10-W.

What this script does
---------------------
1) Downloads CIFAR-10.1 (v6) and CIFAR-10.2 from the official GitHub repos.
2) Reconstructs your utils-style backbone (PreActResNet18 or ResNet18Plain)
   from a **future-proof checkpoint** you saved (with keys: arch, arch_init, state_dict, cut_layer, args, ...).
3) Applies the **same CIFAR-10 normalization** used in your WRM-LAT code and evaluates clean accuracy.
4) Optionally evaluates clean accuracy on each CIFAR-10-W distribution shift (requires --cifar10w-root).
5) Optionally evaluates **input-space PGD** (pixel units) on CIFAR-10/10.1/10.2.
6) Optionally evaluates **AutoAttack** robustness (CIFAR-10) using the robustbench reference settings.

Notes
-----
* We evaluate using the **backbone forward** directly (no phi/head split needed for clean, PGD, or AutoAttack eval).
* Datasets are normalized with CIFAR10 mean/std. For PGD, we perturb in pixel space [0,1]
  and then re-normalize before feeding into the model (matching your training-eval style).
* The checkpoint loader matches the format you used in your saving block (arch/arch_init/state_dict/... on CPU).

Usage
-----
python evaluate_wrm_lat_cifar10_variants.py \
  --ckpt /path/to/your_saved_wrm_lat.ckpt \
  --batch-size 256 --num-workers 4 \
  --cifar10w-root /datasets/cifar10w \
  --inp-eps 0.031372549 --inp-steps 20 --inp-restarts 5 \
  --autoattack --autoattack-bs 128 \
  --autoattack-version custom --autoattack-attacks apgd-ce apgd-dlr
"""

import os
import sys
import tarfile
import json
import argparse
import csv
import inspect
import importlib.util
from functools import partial
import math
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import grad
from tqdm.auto import tqdm
from typing import Dict, Iterable, List, Optional, Tuple
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from model import ResNet18 as ResNet18Plain  # alias kept consistent with your code
from model import PreActResNet18
from pretrained_LAT import build_split_resnet18

from utils import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    auto_pgd_step_size,
    evaluate_under_input_pgd,
    dataloader_seed,
    get_cifar10_loader,
    get_device,
    parameterized_filename,
    set_deterministic,
    to_pixel,
    to_normalized,
    unwrap_state_dict,
)

DEFAULT_CIFAR10C_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]

DEFAULT_AUTOATTACK_ITERS = 100


class PhiHeadWrapper(nn.Module):
    def __init__(self, phi: nn.Module, head: nn.Module):
        super().__init__()
        self.phi = phi
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.phi(x))


class PixelModelWrapper(nn.Module):
    """Adapter so AutoAttack operates in pixel space while the backbone expects normalized inputs."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x_pix: torch.Tensor) -> torch.Tensor:
        return self.base(to_normalized(x_pix))


class AdvLibProgressLogger:
    """Hybrid logger that mirrors the VisdomLogger API while driving a tqdm progress bar."""

    def __init__(
        self,
        attack_name: str,
        total_steps: int,
        use_tqdm: bool = True,
        visdom_logger: Optional["VisdomLogger"] = None,
    ) -> None:
        self.attack_name = attack_name
        self.total_steps = max(int(total_steps), 1)
        self._visdom = visdom_logger
        self._latest_step = -1
        self._metrics: Dict[str, float] = {}
        self._bar = None
        if use_tqdm:
            self._bar = tqdm(
                total=self.total_steps,
                desc=attack_name,
                leave=False,
                dynamic_ncols=True,
            )

    @staticmethod
    def _to_float(value: object) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float("nan")
            value = value.detach().float()
            return float(value.mean().item())
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)  # type: ignore[arg-type]
        except Exception:
            return float("nan")

    def _advance_bar(self, step: int) -> None:
        if self._bar is None:
            return
        target = min(self.total_steps, int(step) + 1)
        if target > self._bar.n:
            self._bar.n = target

    def accumulate_line(self, names, step, values, title: str = "", **kwargs) -> None:
        if isinstance(names, str):
            names_seq = [names]
        else:
            names_seq = list(names)
        if isinstance(values, (list, tuple)):
            values_seq = list(values)
        else:
            values_seq = [values]
        for key, val in zip(names_seq, values_seq):
            self._metrics[str(key)] = self._to_float(val)
        self._latest_step = max(self._latest_step, int(step))
        self._advance_bar(step)
        if self._visdom is not None:
            self._visdom.accumulate_line(names, step, values, title=title or "", **kwargs)

    def update_lines(self) -> None:
        if self._bar is not None:
            if self._metrics:
                # show up to three metrics to keep the bar readable
                first_items = list(self._metrics.items())[:3]
                postfix = {k: f"{v:.4f}" if math.isfinite(v) else f"{v:.2e}" for k, v in first_items}
                self._bar.set_postfix(postfix, refresh=False)
            self._bar.refresh()
        if self._visdom is not None:
            self._visdom.update_lines()

    def manual_step(self, step: int, metrics: Optional[Dict[str, float]] = None) -> None:
        if metrics:
            for name, value in metrics.items():
                self.accumulate_line(name, step, value)
        else:
            self._latest_step = max(self._latest_step, int(step))
            self._advance_bar(step)
        self.update_lines()

    def get_metrics(self) -> Dict[str, float]:
        return dict(self._metrics)

    def close(self) -> None:
        if self._bar is not None:
            if self._latest_step >= 0:
                self._advance_bar(self._latest_step)
                self._bar.refresh()
            self._bar.close()
            self._bar = None
        if self._visdom is not None:
            try:
                self._visdom.update_lines()
            except Exception:
                pass
        self._metrics.clear()
        self._latest_step = -1


def run_fmn_with_progress(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    norm: float,
    steps: int,
    *,
    targeted: bool = False,
    alpha_init: float = 1.0,
    alpha_final: Optional[float] = None,
    gamma_init: float = 0.05,
    gamma_final: float = 0.001,
    starting_points: Optional[torch.Tensor] = None,
    binary_search_steps: int = 10,
    progress_logger: Optional[AdvLibProgressLogger] = None,
) -> torch.Tensor:
    """FMN attack with a lightweight tqdm/Visdom progress hook."""
    from adv_lib.attacks.fast_minimum_norm import (
        l0_mid_points,
        l0_projection_,
        l1_mid_points,
        l1_projection_,
        l2_mid_points,
        l2_projection_,
        linf_mid_points,
        linf_projection_,
    )
    from adv_lib.utils.losses import difference_of_logits

    _dual_projection_mid_points = {
        0: (None, l0_projection_, l0_mid_points),
        1: (float("inf"), l1_projection_, l1_mid_points),
        2: (2, l2_projection_, l2_mid_points),
        float("inf"): (1, linf_projection_, linf_mid_points),
    }
    if inputs.min() < 0 or inputs.max() > 1:
        raise ValueError("Input values should be in the [0, 1] range.")
    device = inputs.device
    batch_size = inputs.size(0)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    dual, projection, mid_point = _dual_projection_mid_points[norm]
    alpha_final = alpha_init / 100 if alpha_final is None else alpha_final
    multiplier = 1 if targeted else -1

    if starting_points is not None:
        start_preds = model(starting_points).argmax(dim=1)
        is_adv = (start_preds == labels) if targeted else (start_preds != labels)
        if not is_adv.all():
            raise ValueError("Starting points are not all adversarial.")
        lower_bound = torch.zeros(batch_size, device=device)
        upper_bound = torch.ones(batch_size, device=device)
        for _ in range(binary_search_steps):
            eps = (lower_bound + upper_bound) / 2
            mid_points = mid_point(x0=inputs, x1=starting_points, ε=eps)
            pred_labels = model(mid_points).argmax(dim=1)
            is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
            lower_bound = torch.where(is_adv, lower_bound, eps)
            upper_bound = torch.where(is_adv, eps, upper_bound)

        delta = mid_point(x0=inputs, x1=starting_points, ε=upper_bound).sub_(inputs)
    else:
        delta = torch.zeros_like(inputs)
    delta.requires_grad_(True)

    if norm == 0:
        eps = torch.ones(batch_size, device=device) if starting_points is None else delta.flatten(1).norm(p=0, dim=1)
    else:
        eps = torch.full((batch_size,), float("inf"), device=device)

    worst_norm = torch.maximum(inputs, 1 - inputs).flatten(1).norm(p=norm, dim=1)
    best_norm = worst_norm.clone()
    best_adv = inputs.clone()
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    logger = progress_logger

    for i in range(steps):
        cosine = (1 + math.cos(math.pi * i / steps)) / 2
        alpha = alpha_final + (alpha_init - alpha_final) * cosine
        gamma = gamma_final + (gamma_init - gamma_final) * cosine

        delta_norm = delta.data.flatten(1).norm(p=norm, dim=1)
        adv_inputs = inputs + delta
        logits = model(adv_inputs)
        pred_labels = logits.argmax(dim=1)

        if i == 0:
            labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float("inf"))
            logit_diff_func = partial(difference_of_logits, labels=labels, labels_infhot=labels_infhot)

        logit_diffs = logit_diff_func(logits=logits)
        loss = multiplier * logit_diffs
        delta_grad = grad(loss.sum(), delta, only_inputs=True)[0]

        is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
        is_smaller = delta_norm < best_norm
        is_both = is_adv & is_smaller
        adv_found.logical_or_(is_adv)
        best_norm = torch.where(is_both, delta_norm, best_norm)
        best_adv = torch.where(batch_view(is_both), adv_inputs.detach(), best_adv)

        if norm == 0:
            eps = torch.where(
                is_adv,
                torch.minimum(torch.minimum(eps - 1, (eps * (1 - gamma)).floor_()), best_norm),
                torch.maximum(eps + 1, (eps * (1 + gamma)).floor_()),
            )
            eps.clamp_(min=0)
        else:
            distance_to_boundary = loss.detach().abs_().div_(delta_grad.flatten(1).norm(p=dual, dim=1).clamp_(min=1e-12))
            eps = torch.where(
                is_adv,
                torch.minimum(eps * (1 - gamma), best_norm),
                torch.where(adv_found, eps * (1 + gamma), delta_norm + distance_to_boundary),
            )

        eps = torch.minimum(eps, worst_norm)
        grad_l2_norms = delta_grad.flatten(1).norm(p=2, dim=1).clamp_(min=1e-12)
        delta.data.addcdiv_(delta_grad, batch_view(grad_l2_norms), value=alpha)
        projection(δ=delta.data, ε=eps)
        delta.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

        if logger is not None:
            metrics = {
                "success": float(adv_found.float().mean().item()),
                "ε": float(eps.mean().item()),
                "best_norm": float(best_norm.masked_select(adv_found).mean().item())
                if adv_found.any()
                else float("inf"),
            }
            logger.manual_step(i, metrics)

    if logger is not None:
        logger.update_lines()

    return best_adv

# -----------------------------
# Globals & utilities
# -----------------------------
def _infer_arch_from_state_dict(sd: dict) -> str:
    """
    Simple heuristic:
      - if top-level has 'bn1' (root BN at stem), assume ResNet18Plain
      - if top-level has 'bn' (tail BN as in PreAct), assume PreActResNet18
    """
    roots = {k.split(".", 1)[0] for k in sd.keys()}
    if "bn1" in roots:
        return "ResNet18Plain"
    if "bn" in roots:
        return "PreActResNet18"
    # default to PreActResNet18
    return "PreActResNet18"


def _move_normalization_buffers(module: nn.Module, device: torch.device) -> None:
    """Place normalization tensors (mu/std) on the correct device when they are not registered buffers."""
    for attr in ("mu", "std"):
        if hasattr(module, attr):
            tensor = getattr(module, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(module, attr, tensor.to(device))


# -----------------------------
# Checkpoint helpers
# -----------------------------
def _extract_state_dict(ckpt: dict) -> Tuple[dict, Optional[str]]:
    """
    Best-effort extraction of a model state_dict from assorted checkpoint formats.

    Returns: (state_dict, source_key)
    """
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"], "state_dict"

    # Input-ICNN / NPF training checkpoints save the evaluated classifier here.
    if "classifier" in ckpt and isinstance(ckpt["classifier"], dict):
        return ckpt["classifier"], "classifier"

    # Common natural-model checkpoints bundled with multiple versions
    for key in ("best", "swa_best", "swa_last", "last", "model", "net"):
        if key in ckpt and isinstance(ckpt[key], dict):
            payload = ckpt[key]
            if "state_dict" in payload and isinstance(payload["state_dict"], dict):
                return payload["state_dict"], f"{key}.state_dict"
            if any(torch.is_tensor(v) for v in payload.values()):
                return payload, key

    # If the entire checkpoint looks like a state dict (tensor-valued)
    if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt, "top_level"

    raise KeyError("No state_dict-like payload found in checkpoint.")


# -----------------------------
# Checkpoint loader (matches your WRM-LAT save format)
# -----------------------------
def load_backbone_from_ckpt(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")

    # Handle phi/head checkpoints saved by pretrained_LAT.py
    if "phi" in ckpt and "head" in ckpt:
        args_ckpt = ckpt.get("args", {})
        num_classes = int(args_ckpt.get("num_classes", args_ckpt.get("n_cls", 10)))
        cut_layer = args_ckpt.get("cut_layer", "layer4")

        # Input-space checkpoints (phi is identity, head is the full backbone)
        if cut_layer == "input":
            head_sd = ckpt["head"]
            if isinstance(head_sd, dict) and "linear.weight" in head_sd:
                num_classes = int(head_sd["linear.weight"].shape[0])
            arch_from_sd = _infer_arch_from_state_dict(head_sd)
            model_width = args_ckpt.get("model_width", 64)
            if model_width is None:
                model_width = 64
            arch_init = {
                "n_cls": num_classes,
                "model_width": int(model_width),
                "normalize_features": bool(args_ckpt.get("normalize_features", False)),
                "normalize_logits": bool(args_ckpt.get("normalize_logits", False)),
            }

            if arch_from_sd == "PreActResNet18":
                head = PreActResNet18(cuda=device.type == "cuda", **arch_init)
            else:
                head = ResNet18Plain(**arch_init)
            phi = nn.Identity()
        else:
            base_backbone = PreActResNet18(
                n_cls=num_classes,
                model_width=int(args_ckpt.get("model_width", 64)),
                normalize_features=bool(args_ckpt.get("normalize_features", False)),
                normalize_logits=bool(args_ckpt.get("normalize_logits", False)),
                cuda=device.type == "cuda",
            )
            phi, head = build_split_resnet18(num_classes=num_classes, cut_layer=cut_layer, base=base_backbone)

        phi_keys = phi.load_state_dict(ckpt["phi"], strict=False)
        head_keys = head.load_state_dict(ckpt["head"], strict=False)

        if phi_keys.missing_keys or phi_keys.unexpected_keys:
            print(f"[phi] load => missing={phi_keys.missing_keys}, unexpected={phi_keys.unexpected_keys}")
        if head_keys.missing_keys or head_keys.unexpected_keys:
            print(f"[head] load => missing={head_keys.missing_keys}, unexpected={head_keys.unexpected_keys}")

        _move_normalization_buffers(phi, device)
        _move_normalization_buffers(head, device)

        model = PhiHeadWrapper(phi, head).to(device).eval()

        meta = {
            "arch": args_ckpt.get("arch", "PhiHeadWrapper"),
            "arch_init": {
                "num_classes": num_classes,
                "cut_layer": cut_layer,
            },
            "cut_layer": cut_layer,
            "args": args_ckpt,
            "epoch": ckpt.get("epoch", None),
            "date": ckpt.get("date", None),
            "format": "phi_head_input" if cut_layer == "input" else "phi_head",
        }
        return model, meta

    # Generic backbone checkpoint (natural models, WRM baselines, etc.)
    try:
        state_dict, sd_key = _extract_state_dict(ckpt)
    except KeyError as err:
        raise KeyError(f"Checkpoint at {path} does not contain a loadable state_dict.") from err

    arch = ckpt.get("arch") or _infer_arch_from_state_dict(state_dict)
    arch_init = ckpt.get(
        "arch_init",
        {"n_cls": 10, "model_width": 64, "normalize_features": False, "normalize_logits": False},
    )
    if arch == "PreActResNet18" and "cuda" not in arch_init:
        arch_init = dict(arch_init, cuda=device.type == "cuda")

    # Recreate the backbone
    if arch == "PreActResNet18":
        base = PreActResNet18(**arch_init)
    elif arch == "ResNet18Plain":
        base = ResNet18Plain(**arch_init)
    else:
        # best-effort fallback
        print(f"[load] Unknown arch '{arch}', falling back to PreActResNet18 with arch_init={arch_init}")
        base = PreActResNet18(**arch_init)

    missing, unexpected = base.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[load] non-strict load => missing={missing}, unexpected={unexpected}")

    _move_normalization_buffers(base, device)
    base.to(device).eval()

    meta = {
        "cut_layer": ckpt.get("cut_layer", "layer4"),
        "args": ckpt.get("args", {}),
        "epoch": ckpt.get("epoch", None),
        "date": ckpt.get("date", None),
        "pytorch": ckpt.get("pytorch", None),
        "arch": arch,
        "arch_init": arch_init,
        "state_dict_key": sd_key,
        "algorithm": ckpt.get("algorithm", None),
        "robust_input_pgd_acc": ckpt.get("robust_input_pgd_acc", None),
    }
    return base, meta


# -----------------------------
# Download & load CIFAR-10.1 / CIFAR-10.2
# -----------------------------
def download_cifar10_variants(data_dir: str = "./cifar10_variants"):
    os.makedirs(data_dir, exist_ok=True)

    # CIFAR-10.1 (v6)
    c10_1 = {
        "cifar10.1_v6_data.npy": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy",
        "cifar10.1_v6_labels.npy": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy",
    }

    # CIFAR-10.2 (test set)
    c10_2 = {
        "cifar102_test.npy": "https://github.com/modestyachts/cifar-10.2/raw/61b0e3ac09809a2351379fb54331668cc9c975c4/cifar102_test.npy",
    }

    def _dl(url_map, desc):
        for fname, url in tqdm(url_map.items(), desc=desc, leave=False):
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                print(f"Downloading {fname} ...")
                urlretrieve(url, fpath)
                print(f"  ✓ {fname}")
            else:
                print(f"  ✓ {fname} already exists")

    print("[download] CIFAR-10.1 & CIFAR-10.2")
    _dl(c10_1, "CIFAR-10.1 files")
    _dl(c10_2, "CIFAR-10.2 files")

    return {
        "c10_1_data": os.path.join(data_dir, "cifar10.1_v6_data.npy"),
        "c10_1_labels": os.path.join(data_dir, "cifar10.1_v6_labels.npy"),
        "c10_2_test": os.path.join(data_dir, "cifar102_test.npy"),
    }


CIFAR10_CLASS_NAMES: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
_CIFAR10_DIGIT_NAMES: Tuple[str, ...] = tuple(str(i) for i in range(10))


class _RemappedTargets(Dataset):
    """Wrap a dataset and remap its class indices to the canonical CIFAR-10 order."""

    def __init__(self, base: Dataset, mapping: Dict[int, int]):
        self._base = base
        self._mapping = mapping

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int):
        data, target = self._base[index]
        return data, self._mapping.get(int(target), int(target))


def _looks_like_cifar10_class_root(path: Path) -> bool:
    try:
        subdirs = [p for p in path.iterdir() if p.is_dir()]
    except FileNotFoundError:
        return False
    if len(subdirs) < 10:
        return False
    names = {p.name.lower() for p in subdirs}
    if set(name.lower() for name in CIFAR10_CLASS_NAMES).issubset(names):
        return True
    if set(_CIFAR10_DIGIT_NAMES).issubset(names):
        return True
    return False


def locate_cifar10w_class_roots(base_dir: Path, max_depth: int = 4) -> List[Path]:
    """
    Locate every directory that contains the CIFAR-10 class folders inside a CIFAR-10-W split.

    Args:
        base_dir: Directory that should contain the dataset (e.g., bing_cartoon_original).
        max_depth: Maximum BFS depth while searching for class folders.

    Returns:
        Sorted list of paths pointing directly at CIFAR-10 class directory structures.
    """

    base_dir = base_dir.expanduser().resolve()
    roots: List[Path] = []
    queue: List[Tuple[Path, int]] = [(base_dir, 0)]
    visited: set[Path] = set()

    while queue:
        current, depth = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if _looks_like_cifar10_class_root(current):
            roots.append(current)
            # Stop descending once a class root is discovered.
            continue

        if depth >= max_depth:
            continue

        try:
            for child in sorted(current.iterdir()):
                if child.is_dir():
                    queue.append((child, depth + 1))
        except FileNotFoundError:
            continue

    return sorted(set(roots))


def _remap_targets_if_needed(dataset: ImageFolder) -> Dataset:
    class_to_idx = dataset.class_to_idx
    canonical = {name: idx for idx, name in enumerate(CIFAR10_CLASS_NAMES)}
    canonical_lower = {name.lower(): idx for name, idx in canonical.items()}
    digits_map = {name: int(name) for name in _CIFAR10_DIGIT_NAMES}

    mapping: Dict[int, int] = {}
    lowered_lookup = {key.lower(): key for key in class_to_idx.keys()}

    if set(class_to_idx.keys()) == set(canonical.keys()):
        for cls_name, cls_idx in class_to_idx.items():
            mapping[cls_idx] = canonical[cls_name]
    elif set(lowered_lookup.keys()) == set(canonical_lower.keys()):
        for lowered, original in lowered_lookup.items():
            mapping[class_to_idx[original]] = canonical_lower[lowered]
    elif set(class_to_idx.keys()) == set(digits_map.keys()):
        for cls_name, cls_idx in class_to_idx.items():
            mapping[cls_idx] = digits_map[cls_name]
    else:
        return dataset

    return _RemappedTargets(dataset, mapping)


DEFAULT_CIFAR10W_SPLITS: Tuple[str, ...] = (
    "bing_cartoon_original",
    "data_360_cartoon_original",
    "data_360_original",
    "data_baidu_cartoon_original",
    "data_baidu_original",
    "data_bing_original",
    "data_flickr_original",
    "data_google_original",
    "data_pexel_original",
    "data_sougou_original",
    "diffusion_cartoon_original",
    "diffusion_hard_original",
    "diffusion_original",
)


def build_cifar10w_dataset(class_root: Path) -> Dataset:
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    image_folder = ImageFolder(class_root, transform=transform)
    return _remap_targets_if_needed(image_folder)


def build_cifar10w_loader(
    class_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    seed_offset: int,
) -> DataLoader:
    dataset = build_cifar10w_dataset(class_root)
    generator, worker_init = dataloader_seed(seed, offset=seed_offset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
        generator=generator,
    )


def available_cifar10w_splits(root_dir: Path, preferred_order: Iterable[str]) -> List[str]:
    root_dir = root_dir.expanduser()
    ordered: List[str] = []
    seen = set()
    for name in preferred_order:
        if (root_dir / name).is_dir():
            ordered.append(name)
            seen.add(name)
    extra = sorted(
        [
            entry.name
            for entry in root_dir.iterdir()
            if entry.is_dir() and entry.name not in seen and not entry.name.startswith(".")
        ]
    )
    ordered.extend(extra)
    return ordered


def _to_nchw_float01(arr: np.ndarray) -> np.ndarray:
    """Ensure array is (N, C, H, W) in [0,1]."""
    a = np.array(arr)
    if a.max() > 1.0:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32)
    if a.ndim == 4 and a.shape[-1] == 3:  # NHWC -> NCHW
        a = np.transpose(a, (0, 3, 1, 2))
    return a


def load_cifar10_1_as_dataset(data_path: str, labels_path: str) -> TensorDataset:
    data = np.load(data_path)
    labels = np.load(labels_path)
    x = torch.from_numpy(_to_nchw_float01(data)).float()
    y = torch.from_numpy(np.array(labels, dtype=np.int64)).long()
    x = to_normalized(x)
    return TensorDataset(x, y)


def load_cifar10_2_test_as_dataset(file_path: str) -> TensorDataset:
    """Robust loader for CIFAR-10.2 test file."""
    obj = np.load(file_path, allow_pickle=True)

    images = None
    labels = None

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        d = obj.item()
        # try common keys
        if "images" in d:
            images = d["images"]
        elif "data" in d:
            images = d["data"]
        elif "x" in d:
            images = d["x"]
        if "labels" in d:
            labels = d["labels"]
        elif "y" in d:
            labels = d["y"]
        elif "targets" in d:
            labels = d["targets"]
    elif isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] >= 3072:
        # flattened CIFAR-like
        images = obj[:, :3072].reshape(-1, 32, 32, 3)
        labels = obj[:, -1]
    else:
        raise RuntimeError(f"Unsupported cifar10.2 format: type={type(obj)}, shape={getattr(obj,'shape',None)}")

    x = torch.from_numpy(_to_nchw_float01(images)).float()
    y = torch.from_numpy(np.array(labels, dtype=np.int64)).long()
    x = to_normalized(x)
    return TensorDataset(x, y)

# === Download CIFAR-10-C dataset ===
def download_cifar10c():
    """Download CIFAR-10-C dataset."""
    data_dir = "./cifar10c_data"
    os.makedirs(data_dir, exist_ok=True)

    # CIFAR-10-C URL (from the official source)
    cifar10c_url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    cifar10c_path = os.path.join(data_dir, "CIFAR-10-C.tar")

    print("Downloading CIFAR-10-C dataset...")

    # Check if already downloaded and extracted
    extracted_dir = os.path.join(data_dir, "CIFAR-10-C")
    if os.path.exists(extracted_dir) and len(os.listdir(extracted_dir)) > 0:
        print(f"✓ CIFAR-10-C already exists at {extracted_dir}")
        return extracted_dir

    # Download if not exists
    if not os.path.exists(cifar10c_path):
        print(f"Downloading CIFAR-10-C from {cifar10c_url}...")
        try:
            urlretrieve(cifar10c_url, cifar10c_path)
            print(f"✓ Downloaded CIFAR-10-C.tar")
        except Exception as e:
            print(f"✗ Failed to download CIFAR-10-C: {e}")
            return None
    else:
        print(f"✓ CIFAR-10-C.tar already exists")

    # Extract the tar file
    print("Extracting CIFAR-10-C.tar...")
    try:
        with tarfile.open(cifar10c_path, 'r') as tar:
            tar.extractall(data_dir)
        print(f"✓ Extracted to {extracted_dir}")

        # Clean up tar file to save space
        os.remove(cifar10c_path)
        print("✓ Cleaned up tar file")

        return extracted_dir
    except Exception as e:
        print(f"✗ Failed to extract CIFAR-10-C: {e}")
        return None

def load_cifar10c(corruption, severity, n_examples=10000, data_dir=None, download=True):
    """
    Load CIFAR-10-C data for a specific corruption and severity.

    Args:
        corruption (str): Name of the corruption (e.g., 'gaussian_noise')
        severity (int): Severity level (1-5)
        n_examples (int): Number of examples to load (default: 10000, full test set)
        data_dir (str): Directory containing CIFAR-10-C data
        download (bool): Whether to download if not found

    Returns:
        tuple: (images, labels) as numpy arrays
    """
    if data_dir is None:
        if download:
            data_dir = download_cifar10c()
            if data_dir is None:
                raise ValueError("Failed to download CIFAR-10-C")
        else:
            raise ValueError("data_dir must be provided if download=False")

    # Path to corruption file
    corruption_file = os.path.join(data_dir, f"{corruption}.npy")
    labels_file = os.path.join(data_dir, "labels.npy")

    if not os.path.exists(corruption_file):
        raise FileNotFoundError(f"Corruption file not found: {corruption_file}")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    # Load data
    print(f"Loading {corruption} (severity {severity})...")

    # Load corruption data - shape is (50000, 32, 32, 3)
    # Each severity has 10k images: severity 1 = [0:10k], severity 2 = [10k:20k], etc.
    corruption_data = np.load(corruption_file)
    labels_data = np.load(labels_file)

    # Calculate start and end indices for the specific severity
    start_idx = (severity - 1) * 10000
    end_idx = start_idx + min(n_examples, 10000)

    # Extract the relevant slice
    images = corruption_data[start_idx:end_idx]
    labels = labels_data[start_idx:end_idx]

    print(f"✓ Loaded {len(images)} images for {corruption} (severity {severity})")
    print(f"  Images shape: {images.shape}, dtype: {images.dtype}")
    print(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"  Image range: [{images.min()}, {images.max()}]")

    return images, labels


def cifar10c_numpy_to_dataset(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
    """Convert CIFAR-10-C arrays to a normalized TensorDataset."""
    x = torch.from_numpy(_to_nchw_float01(images)).float()
    y = torch.from_numpy(np.array(labels, dtype=np.int64)).long()
    x = to_normalized(x)
    return TensorDataset(x, y)


@torch.no_grad()
def collect_pixels_and_labels(
    loader: DataLoader,
    *,
    max_examples: Optional[int] = None,
    desc: str = "Collect AutoAttack data",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize a dataloader into pixel-space tensors for AutoAttack."""
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    remaining = None if max_examples is None or max_examples < 0 else int(max_examples)
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    progress = tqdm(loader, desc=desc, leave=False, total=total_batches)
    for x_norm, y in progress:
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and x_norm.size(0) > remaining:
            x_norm = x_norm[:remaining]
            y = y[:remaining]
        xs.append(to_pixel(x_norm).cpu())
        ys.append(y.cpu())
        if remaining is not None:
            remaining -= x_norm.size(0)
    progress.close()
    x_pix = torch.cat(xs, dim=0) if xs else torch.empty(0, device="cpu")
    y_lab = torch.cat(ys, dim=0) if ys else torch.empty(0, dtype=torch.long, device="cpu")
    return x_pix.contiguous(), y_lab.contiguous()

# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def eval_clean(base_model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    base_model.eval()
    total, correct = 0, 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    progress = tqdm(loader, desc="Clean Eval", leave=False, total=total_batches)
    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = base_model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else total
    if hasattr(loader, "dataset") and total != dataset_size:
        raise RuntimeError(
            f"eval_clean consumed {total} samples but dataset has {dataset_size}."
            " Ensure DataLoader iterates the full set."
        )
    progress.close()
    return 100.0 * correct / max(1, dataset_size)


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate WRM-LAT checkpoint on CIFAR-10 / 10.1 / 10.2 (+ optional input-PGD)")
    p.add_argument("--ckpt", type=str, required=True, help="Path to saved WRM-LAT checkpoint (.ckpt/.pth)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-json", type=str, default="cifar10_variants_eval.json")
    p.add_argument(
        "--cifar10w-root",
        type=str,
        default="",
        help="Directory containing the extracted CIFAR-10-W splits (as produced by download_cifar10w.py).",
    )
    p.add_argument(
        "--cifar10w-subsets",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of CIFAR-10-W distributions to evaluate (default: canonical 13 and any extra folders).",
    )
    p.add_argument(
        "--skip-cifar10w",
        action="store_true",
        help="Disable CIFAR-10-W evaluation even if a root directory is provided.",
    )

    # Optional: evaluate a clean/natural pretrained backbone instead of WRM-LAT ckpt
    p.add_argument("--natural-model", action="store_true",
                   help="If set, load a natural/clean pretrained backbone from a .pt/.pth file instead of the WRM ckpt")
    p.add_argument("--natural-model-path", type=str, default=".",
                   help="Path to the natural model .pt/.pth file (e.g., /mnt/.../R2.pt)")

    # CIFAR-10-C evaluation
    p.add_argument("--skip-cifar10c", action="store_true", help="Skip CIFAR-10-C evaluation.")
    p.add_argument("--cifar10c-data-dir", type=str, default="",
                   help="Directory containing CIFAR-10-C .npy files (default: auto download to ./cifar10c_data).")
    p.add_argument("--cifar10c-corruptions", type=str, nargs="+", default=None,
                   help="Which CIFAR-10-C corruptions to evaluate (default: all).")
    p.add_argument("--cifar10c-severities", type=int, nargs="+", default=[5],
                   help="Severities (1-5) to evaluate for each corruption (default: 5).")
    p.add_argument("--cifar10c-max-examples", type=int, default=10000,
                   help="Max examples per corruption/severity (<=10000 from CIFAR-10-C).")

    # AutoAttack evaluation
    p.add_argument("--autoattack", action="store_true", help="Enable AutoAttack evaluation.")
    p.add_argument("--autoattack-only", action="store_true",
                   help="Run only AutoAttack (skip clean/PGD/CIFAR-10 variants).")
    p.add_argument("--autoattack-norm", type=str, default="Linf", choices=["Linf", "L2", "L1"],
                   help="Norm ball for AutoAttack.")
    p.add_argument("--autoattack-eps", type=float, default=8/255,
                   help="Epsilon radius for AutoAttack (pixel units).")
    p.add_argument("--autoattack-bs", type=int, default=128,
                   help="Batch size used inside AutoAttack (bs argument).")
    p.add_argument("--autoattack-version", type=str, default="standard",
                   help="AutoAttack version (e.g., 'standard', 'plus', 'rand').")
    p.add_argument("--autoattack-attacks", type=str, nargs="+", default=None,
                   help="Specific attacks_to_run for AutoAttack (requires --autoattack-version custom).")
    p.add_argument("--autoattack-max-examples", type=int, default=-1,
                   help="Limit AutoAttack to the first N samples (<=0 means all).")
    p.add_argument("--autoattack-seed", type=int, default=None,
                   help="Optional seed for AutoAttack's internal RNG.")
    p.add_argument("--autoattack-iters", type=int, default=None,
                   help="Override total iterations for APGD-based AutoAttack components.")
    p.add_argument("--autoattack-restarts", type=int, default=None,
                   help="Override number of random restarts for APGD-based AutoAttack components.")

    # adv_lib evaluation
    p.add_argument("--advlib", action="store_true", help="Enable adv_lib attack evaluation (FMN/PD-PGD).")
    p.add_argument(
        "--advlib-only",
        action="store_true",
        help="Run only adv_lib attacks (skip clean/PGD/CIFAR-10 variants).",
    )
    p.add_argument(
        "--advlib-attacks",
        type=str,
        nargs="+",
        choices=["fmn", "pdpgd"],
        default=["fmn", "pdpgd"],
        help="Subset of adv_lib attacks to run (default: fmn and pdpgd).",
    )
    p.add_argument(
        "--advlib-visdom",
        action="store_true",
        help="Stream adv_lib attack metrics to a Visdom server (requires visdom).",
    )
    p.add_argument(
        "--advlib-visdom-port",
        type=int,
        default=8097,
        help="Visdom server port used for adv_lib monitoring callbacks.",
    )

    # Input-space PGD config (pixel units)
    p.add_argument("--inp-p", type=str, default="inf", choices=["2", "inf"],
                   help="Norm for input-space PGD evaluation")
    p.add_argument("--inp-eps", type=float, default=8/255,
                   help="Epsilon (pixel units) for input-space PGD")
    p.add_argument("--inp-steps", type=int, default=20,
                   help="Steps for input-space PGD (0 disables PGD evaluation)")
    p.add_argument("--inp-step-size", type=float, default=0.0,
                   help="Step size for input-space PGD (pixel units). If <=0, auto = 2*eps/steps")
    p.add_argument("--inp-restarts", type=int, default=5,
                   help="Random restarts for input-space PGD")
    p.add_argument("--inp-eps-sweep", type=float, nargs="+", default=None,
                   help="Optional CIFAR-10 PGD epsilon sweep in pixel units, e.g. 0.1 0.2 ... 0.9")
    p.add_argument("--inp-sweep-max-samples", type=int, default=-1,
                   help="Limit PGD epsilon sweep to first N CIFAR-10 samples (<=0 means full test set).")
    return p.parse_args()


def main():
    args = parse_args()
    set_deterministic(args.seed)
    device = get_device()
    print("Using device:", device)

    if args.autoattack_only and args.advlib_only:
        raise ValueError("Specify at most one of --autoattack-only or --advlib-only.")
    if args.autoattack_only:
        args.autoattack = True
    if args.advlib_only:
        args.advlib = True

    autoattack_requested = bool(args.autoattack)
    advlib_requested = bool(args.advlib)
    perform_standard_eval = not (args.autoattack_only or args.advlib_only)

    print("Python executable:", sys.executable)
    if autoattack_requested:
        autoattack_spec = importlib.util.find_spec("autoattack")
        print("[autoattack] module:", autoattack_spec.origin if autoattack_spec is not None else "not found")

    if autoattack_requested and args.autoattack_bs <= 0:
        raise ValueError("--autoattack-bs must be positive.")
    if autoattack_requested and args.autoattack_eps <= 0:
        raise ValueError("--autoattack-eps must be positive.")
    adv_eval_max_examples = None
    if args.autoattack_max_examples > 0:
        adv_eval_max_examples = int(args.autoattack_max_examples)
    autoattack_max_examples = adv_eval_max_examples if autoattack_requested else None
    autoattack_attacks = None
    if autoattack_requested:
        if args.autoattack_attacks is not None:
            autoattack_attacks = list(dict.fromkeys(args.autoattack_attacks))
        if args.autoattack_version.lower() == "custom":
            if autoattack_attacks is None or len(autoattack_attacks) == 0:
                autoattack_attacks = ["apgd-ce", "apgd-dlr"]
        elif args.autoattack_attacks is not None:
            print("[autoattack] Ignoring --autoattack-attacks because version is not 'custom'.")
            autoattack_attacks = None

    advlib_attacks: List[str] = []
    advlib_norm = None
    advlib_steps = args.autoattack_iters if args.autoattack_iters is not None else DEFAULT_AUTOATTACK_ITERS
    if advlib_requested:
        advlib_attacks = list(dict.fromkeys(args.advlib_attacks))
        if len(advlib_attacks) == 0:
            raise ValueError("At least one adv_lib attack must be selected.")
        norm_key = args.autoattack_norm.lower()
        if norm_key == "linf":
            advlib_norm = float("inf")
        elif norm_key == "l2":
            advlib_norm = 2.0
        elif norm_key == "l1":
            advlib_norm = 1.0
        else:
            raise ValueError(f"Unsupported norm for adv_lib attacks: {args.autoattack_norm}")

    # 1) Load backbone from your WRM-LAT checkpoint
    base, meta = load_backbone_from_ckpt(args.ckpt, device)
    print(f"[ckpt] arch={meta['arch']}  cut_layer={meta['cut_layer']}  epoch={meta['epoch']}  date={meta['date']}")

    # ---- Robust NATURAL (clean) model block ----
    # If requested, REPLACE 'base' with a clean pretrained backbone loaded from a raw .pt/.pth
    if args.natural_model:
        nat_path = args.natural_model_path
        if not os.path.isfile(nat_path):
            raise FileNotFoundError(f"--natural-model was set, but file not found: {nat_path}")

        print(f"[natural] Loading clean pretrained backbone from: {nat_path}")
        ckpt_nat = torch.load(nat_path, map_location="cpu")
        sd_nat = unwrap_state_dict(ckpt_nat)
        nat_arch = _infer_arch_from_state_dict(sd_nat)
        nat_init = dict(n_cls=10, model_width=64, normalize_features=False, normalize_logits=False)

        if nat_arch == "PreActResNet18":
            base_nat = PreActResNet18(**nat_init)
        else:
            base_nat = ResNet18Plain(**nat_init)

        missing, unexpected = base_nat.load_state_dict(sd_nat, strict=False)
        if missing or unexpected:
            print(f"[natural] non-strict load => missing={missing}, unexpected={unexpected}")

        base = base_nat  # replace the model to be evaluated
        base.to(device).eval()
        print(f"[natural] Loaded as {nat_arch}. (Normalization remains as in this eval script.)")

    # 2) CIFAR-10 test loader
    c10_loader, c10_len = get_cifar10_loader(
        "test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=False,
        augment_train=False,
        seed=args.seed,
    )
    print(f"✓ CIFAR-10 test set: {c10_len} samples")

    c101_loader = None
    c102_loader = None
    cifar10w_root_path: Optional[Path] = None
    cifar10w_entries: List[Tuple[str, List[Tuple[str, DataLoader]]]] = []
    if perform_standard_eval:
        # 3) Download + set up CIFAR-10.1 & 10.2
        paths = download_cifar10_variants()

        c101_ds = load_cifar10_1_as_dataset(paths["c10_1_data"], paths["c10_1_labels"]) if os.path.exists(paths["c10_1_data"]) else None
        c102_ds = load_cifar10_2_test_as_dataset(paths["c10_2_test"]) if os.path.exists(paths["c10_2_test"]) else None

        if c101_ds is not None:
            gen_c101, worker_c101 = dataloader_seed(args.seed, offset=2)
            c101_loader = DataLoader(
                c101_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=worker_c101,
                generator=gen_c101,
            )
        if c102_ds is not None:
            gen_c102, worker_c102 = dataloader_seed(args.seed, offset=3)
            c102_loader = DataLoader(
                c102_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=worker_c102,
                generator=gen_c102,
            )

        if not args.skip_cifar10w and args.cifar10w_root:
            cifar10w_root_path = Path(args.cifar10w_root).expanduser()
            if not cifar10w_root_path.is_dir():
                raise FileNotFoundError(
                    f"CIFAR-10-W root directory not found: {cifar10w_root_path}. "
                    "Run download_cifar10w.py to obtain the benchmark."
                )
            requested_splits = args.cifar10w_subsets
            if requested_splits is None:
                requested_splits = available_cifar10w_splits(cifar10w_root_path, DEFAULT_CIFAR10W_SPLITS)
            if len(requested_splits) == 0:
                raise RuntimeError(
                    f"No CIFAR-10-W subsets discovered under {cifar10w_root_path}. "
                    "Ensure the archives are extracted."
                )
            for offset, split_name in enumerate(requested_splits):
                split_dir = cifar10w_root_path / split_name
                if not split_dir.is_dir():
                    print(f"[cifar10w] Skipping '{split_name}': directory not found at {split_dir}")
                    continue
                class_roots = locate_cifar10w_class_roots(split_dir)
                if not class_roots:
                    print(f"[cifar10w] Skipping '{split_name}': could not locate CIFAR-10 class folders.")
                    continue
                collections: List[Tuple[str, DataLoader]] = []
                for coll_idx, class_root in enumerate(class_roots):
                    try:
                        loader = build_cifar10w_loader(
                            class_root,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            seed=args.seed,
                            seed_offset=10 + offset * 100 + coll_idx,
                        )
                    except FileNotFoundError as exc:
                        print(f"[cifar10w] Skipping collection under '{split_name}': {exc}")
                        continue
                    try:
                        rel_path = class_root.relative_to(split_dir)
                        rel_key = rel_path.as_posix() if rel_path != Path(".") else "."
                    except ValueError:
                        rel_key = class_root.name
                    collections.append((rel_key, loader))
                if collections:
                    cifar10w_entries.append((split_name, collections))
                else:
                    print(f"[cifar10w] Skipping '{split_name}': no valid collections found.")
        elif args.cifar10w_subsets and not args.cifar10w_root:
            raise ValueError("--cifar10w-subsets was provided without --cifar10w-root.")

    # 4) Evaluate
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "ckpt": os.path.abspath(args.ckpt),
        "arch": meta["arch"],
        "arch_init": meta["arch_init"],
        "cut_layer": meta["cut_layer"],
        "used_natural_model": bool(args.natural_model),
        "natural_model_path": os.path.abspath(args.natural_model_path) if args.natural_model else None,
        "cifar10w_root": str(cifar10w_root_path) if cifar10w_root_path is not None else None,
        "cifar10w_requested": list(args.cifar10w_subsets) if args.cifar10w_subsets is not None else None,
        "scores": {},
    }

    if perform_standard_eval:
        print("\nEvaluating clean accuracy...")
        acc_c10 = eval_clean(base, c10_loader, device)
        print(f"  CIFAR-10   : {acc_c10:.2f}%")
        results["scores"]["cifar10"] = round(acc_c10, 2)

        if c101_loader is not None:
            acc_c101 = eval_clean(base, c101_loader, device)
            print(f"  CIFAR-10.1 : {acc_c101:.2f}%")
            results["scores"]["cifar10.1_v6"] = round(acc_c101, 2)
        else:
            print("  CIFAR-10.1 : (not available)")

        if c102_loader is not None:
            acc_c102 = eval_clean(base, c102_loader, device)
            print(f"  CIFAR-10.2 : {acc_c102:.2f}%")
            results["scores"]["cifar10.2_test"] = round(acc_c102, 2)
        else:
            print("  CIFAR-10.2 : (not available)")

        if cifar10w_entries:
            print("\nEvaluating CIFAR-10-W benchmark (clean only)...")
            results["scores"]["cifar10w"] = {}
            for split_name, collections in cifar10w_entries:
                print(f"  Dataset: {split_name}")
                split_summary = {}
                for rel_key, loader in collections:
                    display_name = split_name if rel_key in {".", ""} else f"{split_name}/{rel_key}"
                    sample_count = len(loader.dataset)
                    print(f"    Collection: {display_name} ({sample_count} samples)")
                    acc_split = eval_clean(base, loader, device)
                    print(f"      Accuracy: {acc_split:.2f}%")
                    split_summary[display_name] = {
                        "accuracy": round(acc_split, 2),
                        "samples": sample_count,
                    }
                results["scores"]["cifar10w"][split_name] = {
                    "collections": split_summary,
                    "collection_count": len(collections),
                }
        elif cifar10w_root_path is not None:
            print("\nEvaluating CIFAR-10-W benchmark (clean only)...")
            print(f"  No valid CIFAR-10-W subsets were found in {cifar10w_root_path}")
            results["scores"]["cifar10w"] = {"status": "not_found"}

        cifar10c_eval_summary = None
        if args.skip_cifar10c:
            print("  CIFAR-10-C : skipped (--skip-cifar10c)")
            results["scores"]["cifar10c"] = {"status": "skipped"}
        else:
            print("\nEvaluating CIFAR-10-C (clean only)...")
            data_dir = args.cifar10c_data_dir.strip()
            if data_dir:
                if not os.path.isdir(data_dir):
                    candidate = os.path.join(data_dir, "CIFAR-10-C")
                    if os.path.isdir(candidate):
                        data_dir = candidate
                    else:
                        raise FileNotFoundError(
                            f"CIFAR-10-C directory not found: '{data_dir}'. "
                            "Provide the directory containing the corruption .npy files."
                        )
            else:
                data_dir = download_cifar10c()

            if data_dir is None:
                print("  CIFAR-10-C : unavailable (download/extraction failed).")
                results["scores"]["cifar10c"] = {"status": "unavailable"}
            else:
                cifar10c_corruptions = (
                    list(DEFAULT_CIFAR10C_CORRUPTIONS)
                    if args.cifar10c_corruptions is None
                    else list(dict.fromkeys(args.cifar10c_corruptions))
                )
                cifar10c_severities = list(dict.fromkeys(args.cifar10c_severities))
                for sev in cifar10c_severities:
                    if sev < 1 or sev > 5:
                        raise ValueError(f"CIFAR-10-C severities must be between 1 and 5 (got {sev}).")
                if args.cifar10c_max_examples <= 0:
                    raise ValueError("--cifar10c-max-examples must be positive.")
                if args.cifar10c_max_examples > 10000:
                    print("[cifar10c] --cifar10c-max-examples truncated to 10000 (dataset size per severity).")
                cifar10c_max_examples = min(args.cifar10c_max_examples, 10000)

                print(f"[cifar10c] data dir: {data_dir}")
                corruption_summaries = {}
                all_acc = []
                for corr_idx, corruption in enumerate(cifar10c_corruptions):
                    severity_scores = {}
                    severity_vals = []
                    load_error = None
                    for severity in cifar10c_severities:
                        try:
                            images, labels = load_cifar10c(
                                corruption,
                                severity,
                                n_examples=cifar10c_max_examples,
                                data_dir=data_dir,
                                download=False,
                            )
                        except (FileNotFoundError, ValueError) as err:
                            load_error = str(err)
                            break
                        dataset = cifar10c_numpy_to_dataset(images, labels)
                        gen_c, worker_c = dataloader_seed(args.seed, offset=10 + corr_idx * 8 + severity)
                        loader = DataLoader(
                            dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            worker_init_fn=worker_c,
                            generator=gen_c,
                        )
                        acc = eval_clean(base, loader, device)
                        severity_scores[str(severity)] = round(acc, 2)
                        severity_vals.append(acc)

                    if load_error is not None:
                        print(f"  {corruption:<18} error: {load_error}")
                        corruption_summaries[corruption] = {"error": load_error}
                        continue

                    all_acc.extend(severity_vals)
                    if severity_vals:
                        corr_mean = float(np.mean(severity_vals))
                        severity_scores["mean"] = round(corr_mean, 2)
                    else:
                        corr_mean = None
                        severity_scores["mean"] = None
                    corruption_summaries[corruption] = severity_scores

                    sev_line = ", ".join(
                        f"s{sev}:{severity_scores[str(sev)]:.2f}%"
                        for sev in cifar10c_severities
                    )
                    if corr_mean is not None:
                        sev_line = f"{sev_line} | mean={corr_mean:.2f}%"
                    print(f"  {corruption:<18} {sev_line}")

                overall_mean = float(np.mean(all_acc)) if all_acc else None
                if overall_mean is not None:
                    print(f"  CIFAR-10-C mean : {overall_mean:.2f}%")
                else:
                    print("  CIFAR-10-C mean : (no corruptions evaluated)")
                has_error = any(
                    isinstance(summary, dict) and "error" in summary
                    for summary in corruption_summaries.values()
                )
                status = "ok"
                if overall_mean is None:
                    status = "no-data"
                elif has_error:
                    status = "partial"
                cifar10c_eval_summary = {
                    "corruptions": corruption_summaries,
                    "severities": cifar10c_severities,
                    "max_examples": cifar10c_max_examples,
                    "mean": round(overall_mean, 2) if overall_mean is not None else None,
                    "data_dir": os.path.abspath(data_dir),
                    "status": status,
                }
                results["scores"]["cifar10c"] = cifar10c_eval_summary

        # 4.b) Optional: Input-space PGD evaluation (pixel units)
        if args.inp_steps > 0 and args.inp_eps > 0:
            print("\nEvaluating input-space PGD (pixel units)...")
            p_value = 2 if args.inp_p == "2" else float("inf")
            step_size_used = auto_pgd_step_size(p_value, args.inp_eps, args.inp_steps, args.inp_step_size)
            results["scores"]["pgd_params"] = {
                "p": args.inp_p,
                "eps": args.inp_eps,
                "steps": args.inp_steps,
                "step_size": step_size_used,
                "restarts": args.inp_restarts,
            }

            phi_identity = nn.Identity().to(device)

            def run_input_pgd(loader):
                return evaluate_under_input_pgd(
                    phi_identity,
                    base,
                    loader,
                    device,
                    p=p_value,
                    eps=args.inp_eps,
                    steps=args.inp_steps,
                    step_size=step_size_used,
                    restarts=args.inp_restarts,
                )

            acc_pgd_c10, info_c10 = run_input_pgd(c10_loader)
            acc_pgd_c10_pct = acc_pgd_c10 * 100.0
            print(f"  CIFAR-10   PGD: {acc_pgd_c10_pct:.2f}% (avg L2 {info_c10['avg_l2']:.3f}, Linf {info_c10['avg_linf']:.3f})")
            results["scores"]["cifar10_pgd"] = {
                "acc": round(acc_pgd_c10_pct, 2),
                "clean_acc": round(100.0 * float(info_c10.get("clean_acc", 0.0)), 2),
                "clean_correct": int(info_c10.get("clean_correct", 0)),
                "robust_correct": int(info_c10.get("robust_correct", 0)),
                "avg_l2": round(info_c10["avg_l2"], 4),
                "avg_linf": round(info_c10["avg_linf"], 4),
                "max_l2": round(float(info_c10.get("max_l2", 0.0)), 4),
                "max_linf": round(float(info_c10.get("max_linf", 0.0)), 4),
                "samples": int(info_c10.get("samples", 0)),
            }

            if args.inp_eps_sweep:
                sweep_eps = list(dict.fromkeys(float(eps) for eps in args.inp_eps_sweep))
                sweep_max_samples = (
                    int(args.inp_sweep_max_samples)
                    if int(args.inp_sweep_max_samples) > 0
                    else None
                )
                print(
                    "\nEvaluating CIFAR-10 input-space PGD epsilon sweep "
                    f"(L{args.inp_p}, steps={args.inp_steps}, restarts={args.inp_restarts})..."
                )
                sweep_entries = []
                sweep_by_eps = {}
                for eps_i in sweep_eps:
                    if eps_i <= 0:
                        print(f"  eps={eps_i:g}: skipped (epsilon must be > 0)")
                        continue
                    step_i = auto_pgd_step_size(p_value, eps_i, args.inp_steps, args.inp_step_size)
                    acc_i, info_i = evaluate_under_input_pgd(
                        phi_identity,
                        base,
                        c10_loader,
                        device,
                        p=p_value,
                        eps=eps_i,
                        steps=args.inp_steps,
                        step_size=step_i,
                        restarts=args.inp_restarts,
                        max_samples=sweep_max_samples,
                    )
                    entry = {
                        "p": args.inp_p,
                        "eps": float(eps_i),
                        "steps": int(args.inp_steps),
                        "step_size": float(step_i),
                        "restarts": int(args.inp_restarts),
                        "acc": round(100.0 * acc_i, 2),
                        "clean_acc": round(100.0 * float(info_i.get("clean_acc", 0.0)), 2),
                        "clean_correct": int(info_i.get("clean_correct", 0)),
                        "robust_correct": int(info_i.get("robust_correct", 0)),
                        "avg_l2": round(float(info_i["avg_l2"]), 4),
                        "avg_linf": round(float(info_i["avg_linf"]), 4),
                        "max_l2": round(float(info_i.get("max_l2", 0.0)), 4),
                        "max_linf": round(float(info_i.get("max_linf", 0.0)), 4),
                        "samples": int(info_i.get("samples", 0)),
                    }
                    sweep_entries.append(entry)
                    sweep_by_eps[f"{eps_i:g}"] = entry
                    print(
                        f"  eps={eps_i:g}: robust_acc={entry['acc']:.2f}% "
                        f"(clean={entry['clean_acc']:.2f}%, avg L2={entry['avg_l2']:.3f}, "
                        f"max L2={entry['max_l2']:.3f}, n={entry['samples']})"
                    )
                results["scores"]["cifar10_pgd_sweep"] = {
                    "params": {
                        "p": args.inp_p,
                        "steps": int(args.inp_steps),
                        "restarts": int(args.inp_restarts),
                        "step_size_mode": "auto=2*eps/steps" if args.inp_step_size <= 0 else "fixed",
                        "max_samples": sweep_max_samples,
                    },
                    "entries": sweep_entries,
                    "by_eps": sweep_by_eps,
                }

            if c101_loader is not None:
                acc_pgd_c101, info_c101 = run_input_pgd(c101_loader)
                acc_pgd_c101_pct = acc_pgd_c101 * 100.0
                print(f"  CIFAR-10.1 PGD: {acc_pgd_c101_pct:.2f}% (avg L2 {info_c101['avg_l2']:.3f}, Linf {info_c101['avg_linf']:.3f})")
                results["scores"]["cifar10.1_v6_pgd"] = {
                    "acc": round(acc_pgd_c101_pct, 2),
                    "avg_l2": round(info_c101["avg_l2"], 4),
                    "avg_linf": round(info_c101["avg_linf"], 4),
                }
            else:
                print("  CIFAR-10.1 PGD: (not available)")

            if c102_loader is not None:
                acc_pgd_c102, info_c102 = run_input_pgd(c102_loader)
                acc_pgd_c102_pct = acc_pgd_c102 * 100.0
                print(f"  CIFAR-10.2 PGD: {acc_pgd_c102_pct:.2f}% (avg L2 {info_c102['avg_l2']:.3f}, Linf {info_c102['avg_linf']:.3f})")
                results["scores"]["cifar10.2_test_pgd"] = {
                    "acc": round(acc_pgd_c102_pct, 2),
                    "avg_l2": round(info_c102["avg_l2"], 4),
                    "avg_linf": round(info_c102["avg_linf"], 4),
                }
            else:
                print("  CIFAR-10.2 PGD: (not available)")
        else:
            print("\nInput-space PGD: disabled (use --inp-steps > 0 to enable)")
    else:
        if args.autoattack_only:
            skip_reason = "--autoattack-only"
        elif args.advlib_only:
            skip_reason = "--advlib-only"
        else:
            skip_reason = "requested configuration"
        print(f"\nSkipping clean/PGD evaluations ({skip_reason}).")

    if autoattack_requested:
        print("\nEvaluating AutoAttack robustness...")
        try:
            from autoattack import AutoAttack
        except ImportError as err:
            print(f"  AutoAttack unavailable: {err}")
            results["scores"]["autoattack"] = {"status": "unavailable", "error": str(err)}
        else:
            try:
                from adv_lib.utils.visdom_logger import VisdomLogger as AdvLibVisdomLogger
            except Exception:
                AdvLibVisdomLogger = None  # type: ignore[assignment]
            pixel_model = PixelModelWrapper(base)
            pixel_model.eval()
            x_c10_pix, y_c10 = collect_pixels_and_labels(
                c10_loader,
                max_examples=autoattack_max_examples,
                desc="AutoAttack CIFAR-10",
            )
            num_samples = int(y_c10.numel())
            if num_samples == 0:
                print("  CIFAR-10 AutoAttack : no samples collected.")
                results["scores"]["autoattack"] = {"status": "empty"}
            else:
                x_c10_pix = x_c10_pix.to(torch.float32)
                y_c10_cpu = y_c10.clone()
                y_c10_device = y_c10_cpu.to(device)
                if autoattack_attacks is not None:
                    print(f"  AutoAttack attacks_to_run={autoattack_attacks}")
                aa_kwargs = {
                    "model": pixel_model,
                    "norm": args.autoattack_norm,
                    "eps": args.autoattack_eps,
                    "version": args.autoattack_version,
                    "verbose": True,
                    "device": device,
                }
                if autoattack_attacks is not None:
                    aa_kwargs["attacks_to_run"] = autoattack_attacks
                adversary = AutoAttack(**aa_kwargs)
                if args.autoattack_seed is not None:
                    adversary.seed = args.autoattack_seed
                adversary.device = device
                if args.autoattack_iters is not None:
                    if hasattr(adversary, "apgd"):
                        adversary.apgd.n_iter = args.autoattack_iters
                    if hasattr(adversary, "apgd_targeted"):
                        adversary.apgd_targeted.n_iter = args.autoattack_iters
                if args.autoattack_restarts is not None:
                    if hasattr(adversary, "apgd"):
                        adversary.apgd.n_restarts = args.autoattack_restarts
                    if hasattr(adversary, "apgd_targeted"):
                        adversary.apgd_targeted.n_restarts = args.autoattack_restarts
                effective_attacks = list(getattr(adversary, "attacks_to_run", autoattack_attacks or []))
                apgd_n_iter = getattr(getattr(adversary, "apgd", None), "n_iter", None)
                apgd_t_n_iter = getattr(getattr(adversary, "apgd_targeted", None), "n_iter", None)
                print(
                    "  AutoAttack effective config: "
                    f"norm={args.autoattack_norm}, eps={args.autoattack_eps}, "
                    f"version={args.autoattack_version}, attacks={effective_attacks}, "
                    f"apgd_iters={apgd_n_iter}, apgd_t_iters={apgd_t_n_iter}"
                )
                x_adv = adversary.run_standard_evaluation(
                    x_c10_pix,
                    y_c10_cpu,
                    bs=args.autoattack_bs,
                )
                with torch.no_grad():
                    x_adv = x_adv.to(device=device, dtype=torch.float32)
                    x_adv_norm = to_normalized(x_adv)
                    logits_adv = base(x_adv_norm)
                    preds_adv = logits_adv.argmax(dim=1)
                    acc_adv = (preds_adv == y_c10_device).float().mean().item() * 100.0
                print(f"  CIFAR-10 AutoAttack: {acc_adv:.2f}% robust accuracy ({num_samples} samples)")
                results["scores"]["autoattack"] = {
                    "status": "ok",
                    "dataset": "cifar10",
                    "acc": round(acc_adv, 2),
                    "norm": args.autoattack_norm,
                    "eps": args.autoattack_eps,
                    "bs": args.autoattack_bs,
                    "version": args.autoattack_version,
                    "num_examples": num_samples,
                    "max_examples": autoattack_max_examples,
                    "seed": args.autoattack_seed,
                    "attacks": autoattack_attacks,
                    "effective_attacks": effective_attacks,
                    "iters": args.autoattack_iters,
                    "apgd_iters": apgd_n_iter,
                    "apgd_t_iters": apgd_t_n_iter,
                    "restarts": args.autoattack_restarts,
                }

    if advlib_requested:
        print("\nEvaluating adv_lib attacks...")
        try:
            from adv_lib.attacks import pdpgd as advlib_pdpgd
        except ImportError as err:
            print(f"  adv_lib unavailable: {err}")
            results["scores"]["adv_lib"] = {"status": "unavailable", "error": str(err)}
        else:
            try:
                from adv_lib.utils.visdom_logger import VisdomLogger as AdvLibVisdomLogger  # type: ignore
            except Exception:
                AdvLibVisdomLogger = None  # type: ignore[assignment]
            pixel_model = PixelModelWrapper(base)
            pixel_model.eval()
            x_c10_pix_advlib, y_c10_advlib = collect_pixels_and_labels(
                c10_loader,
                max_examples=adv_eval_max_examples,
                desc="adv_lib CIFAR-10",
            )
            num_samples = int(y_c10_advlib.numel())
            if num_samples == 0:
                print("  adv_lib: no samples collected.")
                results["scores"]["adv_lib"] = {"status": "empty"}
            else:
                x_c10_pix_advlib = x_c10_pix_advlib.to(device=device, dtype=torch.float32)
                x_c10_pix_advlib = x_c10_pix_advlib.clamp_(0.0, 1.0)
                y_c10_device = y_c10_advlib.to(device)
                attack_summaries = {}
                det_enabled = (
                    torch.are_deterministic_algorithms_enabled()
                    if hasattr(torch, "are_deterministic_algorithms_enabled")
                    else False
                )
                set_det = getattr(torch, "use_deterministic_algorithms", None)
                visdom_enabled = bool(args.advlib_visdom and AdvLibVisdomLogger is not None)
                visdom_warned = False

                def make_visdom_logger():
                    nonlocal visdom_enabled, visdom_warned
                    if not visdom_enabled or AdvLibVisdomLogger is None:
                        return None
                    try:
                        return AdvLibVisdomLogger(port=args.advlib_visdom_port)
                    except Exception as vis_err:  # pragma: no cover - visualization optional
                        if not visdom_warned:
                            print(
                                f"  adv_lib: Visdom logger unavailable ({vis_err}); disabling Visdom logging."
                            )
                            visdom_warned = True
                        visdom_enabled = False
                        return None

                attack_progress = tqdm(
                    advlib_attacks,
                    desc="adv_lib attacks",
                    leave=False,
                    dynamic_ncols=True,
                )
                for attack_id in attack_progress:
                    attack_inputs = x_c10_pix_advlib.clone()
                    vis_logger = make_visdom_logger()
                    progress_logger = AdvLibProgressLogger(
                        f"adv_lib {attack_id}",
                        total_steps=advlib_steps,
                        visdom_logger=vis_logger,
                    )
                    adv_samples = None
                    metrics_snapshot: Dict[str, float] = {}
                    try:
                        if det_enabled and set_det is not None:
                            set_det(False)
                        if attack_id == "fmn":
                            adv_samples = run_fmn_with_progress(
                                model=pixel_model,
                                inputs=attack_inputs,
                                labels=y_c10_device,
                                norm=advlib_norm,
                                steps=advlib_steps,
                                progress_logger=progress_logger,
                            )
                        elif attack_id == "pdpgd":
                            attack_kwargs = {
                                "model": pixel_model,
                                "inputs": attack_inputs,
                                "labels": y_c10_device,
                                "norm": advlib_norm,
                                "num_steps": advlib_steps,
                            }
                            if "callback" in inspect.signature(advlib_pdpgd).parameters:
                                attack_kwargs["callback"] = progress_logger
                            adv_samples = advlib_pdpgd(**attack_kwargs)
                        else:
                            raise ValueError(f"Unsupported adv_lib attack '{attack_id}'.")
                        metrics_snapshot = progress_logger.get_metrics()
                    except Exception as err:
                        metrics_snapshot = progress_logger.get_metrics()
                        print(f"  adv_lib {attack_id}: error -> {err}")
                        attack_summaries[attack_id] = {"status": "error", "error": str(err)}
                        attack_progress.set_postfix({"attack": attack_id, "status": "error"})
                        continue
                    finally:
                        if det_enabled and set_det is not None:
                            set_det(True)
                        progress_logger.close()
                    if adv_samples is None:
                        continue
                    adv_samples = adv_samples.to(device=device, dtype=torch.float32)
                    adv_samples = adv_samples.clamp_(0.0, 1.0)
                    with torch.no_grad():
                        logits_adv = base(to_normalized(adv_samples))
                        preds_adv = logits_adv.argmax(dim=1)
                        acc_adv = (preds_adv == y_c10_device).float().mean().item() * 100.0
                    success_ratio = metrics_snapshot.get("success")
                    if success_ratio is not None and math.isfinite(success_ratio):
                        print(
                            f"  adv_lib {attack_id}: {acc_adv:.2f}% robust accuracy "
                            f"({num_samples} samples) | success {success_ratio*100:.2f}%"
                        )
                    else:
                        print(f"  adv_lib {attack_id}: {acc_adv:.2f}% robust accuracy ({num_samples} samples)")
                    postfix = {"attack": attack_id, "acc": f"{acc_adv:.2f}"}
                    if success_ratio is not None and math.isfinite(success_ratio):
                        postfix["success"] = f"{success_ratio:.2f}"
                    attack_progress.set_postfix(postfix)
                    summary = {
                        "status": "ok",
                        "acc": round(acc_adv, 2),
                        "norm": args.autoattack_norm,
                        "iters": advlib_steps,
                        "num_examples": num_samples,
                    }
                    metric_key_map = {
                        "success": "success",
                        "ε": "eps",
                        "best_norm": "best_norm",
                    }
                    for metric_key, summary_key in metric_key_map.items():
                        value = metrics_snapshot.get(metric_key)
                        if value is None or not math.isfinite(value):
                            continue
                        summary[summary_key] = round(float(value), 6)
                    attack_summaries[attack_id] = summary
                attack_progress.close()
                if not attack_summaries:
                    results["scores"]["adv_lib"] = {"status": "skipped"}
                else:
                    overall_status = (
                        "ok"
                        if any(v.get("status") == "ok" for v in attack_summaries.values())
                        else "error"
                    )
                    results["scores"]["adv_lib"] = {
                        "status": overall_status,
                        "attacks": attack_summaries,
                        "norm": args.autoattack_norm,
                        "iters": advlib_steps,
                        "max_examples": adv_eval_max_examples,
                    }

    # 5) Save JSON
    save_path = parameterized_filename(
        args.save_json,
        {
            "ckpt": Path(args.ckpt).stem,
            "seed": args.seed,
            "inpP": args.inp_p,
            "eps": args.inp_eps,
            "steps": args.inp_steps,
            "restarts": args.inp_restarts,
            "natural": int(bool(args.natural_model)),
        },
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results["output_file"] = str(save_path)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {save_path}")

    sweep = results.get("scores", {}).get("cifar10_pgd_sweep")
    if isinstance(sweep, dict) and sweep.get("entries"):
        sweep_csv = save_path.with_suffix(".pgd_sweep.csv")
        fieldnames = [
            "p",
            "eps",
            "steps",
            "step_size",
            "restarts",
            "acc",
            "clean_acc",
            "clean_correct",
            "robust_correct",
            "avg_l2",
            "avg_linf",
            "max_l2",
            "max_linf",
            "samples",
        ]
        with sweep_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in sweep["entries"]:
                writer.writerow({key: entry.get(key) for key in fieldnames})
        results["scores"]["cifar10_pgd_sweep"]["csv"] = str(sweep_csv)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved PGD epsilon sweep CSV to: {sweep_csv}")


if __name__ == "__main__":
    main()
