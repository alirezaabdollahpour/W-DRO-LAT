#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot input-space loss landscapes for an INPUT-ICNN-trained CIFAR-10 classifier.

This script reproduces the *input-space* PGD procedure used in
`evaluate_wrm_lat_cifar10_variants.py` (via the shared helpers in `utils.py`):

  - Attack in **pixel space** x_pix ∈ [0,1]
  - Convert to **normalized space** before feeding the model
  - L2-PGD with ε=0.5, 20 iterations, 5 random restarts (by default)

For each selected CIFAR-10 test example, we compute:
  1) Adversarial direction r = (x_adv - x) from PGD (worst-case over restarts).
  2) A random direction u orthogonal to r (in pixel space).
  3) A 2D grid in the plane spanned by (r, u) and evaluate:
       - correctness region (pred == y)
       - cross-entropy loss surface

The output matches the common "decision region + CE-loss surface" visualization
style shown in the prompt (two panels per example).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

# matplotlib.use("inline")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3d projection)
from torch.utils.data import DataLoader, Dataset

from model import PreActResNet18
from model import ResNet18 as ResNet18Plain
from utils import (
    auto_pgd_step_size,
    get_cifar10_loader,
    get_device,
    per_sample_l2_normalize,
    project_onto_lp_ball,
    random_start_input_ball_pix,
    set_deterministic,
    to_normalized,
    to_pixel,
)


@torch.no_grad()
def evaluate_clean_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (avg CE loss, accuracy) on a normalized CIFAR-10 loader."""
    model.eval()
    total = 0
    total_correct = 0
    total_loss = 0.0
    for x_norm, y in loader:
        x_norm = x_norm.to(device)
        y = y.to(device)
        logits = model(x_norm)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_loss += float(F.cross_entropy(logits, y, reduction="sum").item())
        total += int(y.numel())
    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc


def _build_cifar10_head_from_state_dict(sd: dict, num_classes: int = 10) -> nn.Module:
    root_keys = {k.split(".", 1)[0] for k in sd.keys() if isinstance(k, str)}
    if "bn1" in root_keys:
        head = ResNet18Plain(
            n_cls=num_classes,
            model_width=64,
            normalize_features=False,
            normalize_logits=False,
        )
        arch = "ResNet18Plain"
    elif "bn" in root_keys:
        head = PreActResNet18(
            n_cls=num_classes,
            model_width=64,
            normalize_features=False,
            normalize_logits=False,
        )
        arch = "PreActResNet18"
    else:
        head = PreActResNet18(
            n_cls=num_classes,
            model_width=64,
            normalize_features=False,
            normalize_logits=False,
        )
        arch = "PreActResNet18 (fallback)"

    missing, unexpected = head.load_state_dict(sd, strict=False)
    print(f"[Model] Built {arch} and loaded head weights.")
    if missing:
        print(f"[Model] Missing keys (ignored): {list(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[Model] Unexpected keys (ignored): {list(unexpected)[:10]}{'...' if len(unexpected) > 10 else ''}")
    return head


def _pgd_best_delta_pix(
    model: nn.Module,
    x_norm: torch.Tensor,
    y: torch.Tensor,
    *,
    p: int | float,
    eps: float,
    steps: int,
    restarts: int,
    step_size: float | None,
) -> Tuple[torch.Tensor, float]:
    """Return best (max-loss) PGD delta in pixel space for a single sample.

    Mirrors `utils.evaluate_under_input_pgd` exactly (but for a single item and returning delta).
    """
    if x_norm.dim() != 4 or x_norm.size(0) != 1:
        raise ValueError("x_norm must be shape (1, C, H, W)")
    if y.dim() != 1 or y.numel() != 1:
        raise ValueError("y must be shape (1,)")

    p_value = 2 if p == 2 else float("inf")
    step_size_used = auto_pgd_step_size(p_value, eps, steps, step_size)
    restarts = max(1, int(restarts))

    x0_pix = to_pixel(x_norm).detach()
    best_delta = torch.zeros_like(x0_pix)
    best_loss = torch.full((x0_pix.size(0),), -1e9, device=x_norm.device)

    for _ in range(restarts):
        x_pix = random_start_input_ball_pix(x0_pix, eps, p_value).detach().requires_grad_(True)
        for _ in range(steps):
            logits = model(to_normalized(x_pix))
            loss_mean = F.cross_entropy(logits, y, reduction="mean")
            grad_pix = torch.autograd.grad(loss_mean, x_pix, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                step = step_size_used * (
                    per_sample_l2_normalize(grad_pix) if p_value == 2 else torch.sign(grad_pix)
                )
                x_pix = x_pix + step
                delta_pix = x_pix - x0_pix
                delta_pix = project_onto_lp_ball(delta_pix, eps, 2 if p_value == 2 else float("inf"))
                x_pix = (x0_pix + delta_pix).clamp(0.0, 1.0)
                x_pix.requires_grad_(True)

        with torch.no_grad():
            logits = model(to_normalized(x_pix))
            losses = F.cross_entropy(logits, y, reduction="none")
            delta_pix = (x_pix - x0_pix).detach()
            mask = losses > best_loss
            if mask.any():
                best_loss[mask] = losses[mask]
                best_delta[mask] = delta_pix[mask]

    return best_delta.detach(), float(best_loss.item())


def _unit_direction(delta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1).clamp(min=eps)
    reshape = (delta.size(0),) + (1,) * (delta.dim() - 1)
    return delta / norms.view(reshape)


def _random_unit_direction_orthogonal_to(
    ref_unit: torch.Tensor,
    *,
    max_tries: int = 50,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Sample a random unit vector orthogonal to `ref_unit` (batch size = 1)."""
    if ref_unit.dim() != 4 or ref_unit.size(0) != 1:
        raise ValueError("ref_unit must be shape (1, C, H, W)")
    ref_flat = ref_unit.view(1, -1)

    for _ in range(max_tries):
        v = torch.randn_like(ref_unit)
        v_flat = v.view(1, -1)
        proj = (v_flat * ref_flat).sum(dim=1, keepdim=True)  # (1,1)
        v_flat = v_flat - proj * ref_flat
        v_norm = v_flat.norm(p=2, dim=1, keepdim=True)
        if float(v_norm.item()) > eps:
            v_flat = v_flat / v_norm.clamp(min=eps)
            return v_flat.view_as(ref_unit)

    raise RuntimeError("Failed to sample a non-degenerate orthogonal direction.")


@torch.no_grad()
def _eval_plane_grid(
    model: nn.Module,
    x0_pix: torch.Tensor,
    y: torch.Tensor,
    adv_dir_unit: torch.Tensor,
    ortho_dir_unit: torch.Tensor,
    coords: torch.Tensor,
    *,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate correctness + CE loss over a 2D grid."""
    if x0_pix.dim() != 4 or x0_pix.size(0) != 1:
        raise ValueError("x0_pix must be shape (1,C,H,W)")
    if y.dim() != 1 or y.numel() != 1:
        raise ValueError("y must be shape (1,)")

    # Meshgrid in coefficient space: X for random direction, Y for adversarial direction.
    X, Y = torch.meshgrid(coords, coords, indexing="xy")
    X = X.to(x0_pix.device)
    Y = Y.to(x0_pix.device)

    # Build all perturbed points in pixel space.
    pert = (X.reshape(-1, 1, 1, 1) * ortho_dir_unit) + (Y.reshape(-1, 1, 1, 1) * adv_dir_unit)
    x_all = (x0_pix + pert).clamp(0.0, 1.0)  # (N^2, C, H, W)

    n = x_all.size(0)
    losses: List[torch.Tensor] = []
    correct: List[torch.Tensor] = []
    y_all = y.expand(batch_size)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = x_all[start:end]
        yb = y_all[: (end - start)]
        logits = model(to_normalized(xb))
        loss_b = F.cross_entropy(logits, yb, reduction="none")
        pred_b = logits.argmax(dim=1)
        losses.append(loss_b.detach().cpu())
        correct.append((pred_b == yb).detach().cpu())

    loss = torch.cat(losses, dim=0).view(coords.numel(), coords.numel()).numpy()
    corr = torch.cat(correct, dim=0).view(coords.numel(), coords.numel()).numpy().astype(np.float32)
    return corr, loss


def _format_frac255(x: float) -> str:
    if abs(x) < 1e-12:
        return "0"
    n = int(round(x * 255.0))
    return f"{n}/255"


def _format_int255(x: float) -> str:
    if abs(x) < 1e-12:
        return "0"
    n = int(round(x * 255.0))
    return f"{n}"


def _tick_label_mode_from_plot_range(plot_range: float) -> str:
    # Heuristic: for large ranges the "n/255" labels get long and overlap in 3D.
    n = abs(int(round(plot_range * 255.0)))
    return "int" if n >= 64 else "frac"


def _is_long_frac_label(plot_range: float) -> bool:
    # "16/255" (len=6) is usually fine; "230/255" (len=7) often overlaps in 3D.
    return len(_format_frac255(plot_range)) >= 7


def _set_ticks(ax, ticks: Iterable[float], *, mode: str) -> None:
    ticks = list(ticks)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if mode == "int":
        ax.set_xticklabels([_format_int255(t) for t in ticks])
        ax.set_yticklabels([_format_int255(t) for t in ticks])
    else:
        ax.set_xticklabels([_format_frac255(t) for t in ticks])
        ax.set_yticklabels([_format_frac255(t) for t in ticks])


def _plot_example_pair(
    fig: plt.Figure,
    ax_decision: plt.Axes,
    ax_surface: plt.Axes,
    *,
    coords_np: np.ndarray,
    correct: np.ndarray,
    loss: np.ndarray,
    plot_range: float,
    inner_range: float | None,
    constraint_shape: str,
    tick_mode: str,
) -> None:
    # Keep all artists inside the axes box so they don't intrude into neighboring subplots.
    ax_decision.set_anchor("C")
    ax_surface.set_anchor("C")
    # 2D decision region (correct vs incorrect).
    cmap = matplotlib.colors.ListedColormap(["#9ecae1", "#fcbba1"])
    ax_decision.imshow(
        correct,
        origin="lower",
        extent=[-plot_range, plot_range, -plot_range, plot_range],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    if inner_range is not None and inner_range > 0:
        if constraint_shape == "circle":
            ax_decision.add_patch(
                Circle(
                    (0.0, 0.0),
                    radius=inner_range,
                    fill=False,
                    edgecolor="k",
                    linewidth=0.9,
                    linestyle="--",
                )
            )
        else:
            ax_decision.add_patch(
                Rectangle(
                    (-inner_range, -inner_range),
                    2 * inner_range,
                    2 * inner_range,
                    fill=False,
                    edgecolor="k",
                    linewidth=0.9,
                    linestyle="--",
                )
            )
    ax_decision.plot(0.0, 0.0, "k.", markersize=5)
    xlab = r"Random direction $\perp\, r$"
    ylab = r"Adversarial direction $r$"
    if tick_mode == "int":
        xlab += r" ($\times 1/255$)"
        ylab += r" ($\times 1/255$)"
    ax_decision.set_xlabel(xlab, labelpad=3, fontsize=10)
    ax_decision.set_ylabel(ylab, labelpad=3, fontsize=10)
    tick_vals_2d = [-plot_range, -plot_range / 2.0, 0.0, plot_range / 2.0, plot_range]
    _set_ticks(ax_decision, tick_vals_2d, mode=tick_mode)
    ax_decision.tick_params(axis="both", labelsize=9)
    ax_decision.xaxis.label.set_clip_on(True)
    ax_decision.yaxis.label.set_clip_on(True)

    # 3D loss surface.
    Xg, Yg = np.meshgrid(coords_np, coords_np, indexing="xy")
    ax_surface.plot_surface(
        Xg,
        Yg,
        loss,
        cmap="coolwarm",
        rstride=1,
        cstride=1,
        linewidth=0.25,
        edgecolor="white",
        antialiased=True,
    )
    ax_surface.set_xlabel(xlab, labelpad=6, fontsize=9)
    ax_surface.set_ylabel(ylab, labelpad=6, fontsize=9)
    ax_surface.set_zlabel("CE Loss", labelpad=6, fontsize=9)
    ax_surface.xaxis.label.set_clip_on(True)
    ax_surface.yaxis.label.set_clip_on(True)
    ax_surface.zaxis.label.set_clip_on(True)
    ax_surface.set_xlim(-plot_range, plot_range)
    ax_surface.set_ylim(-plot_range, plot_range)

    # 3D tick labels are easy to overlap (especially for long "n/255" strings).
    # Keep the 3D axes uncluttered: only show min/0/max ticks.
    tick_vals_3d = [-plot_range, 0.0, plot_range]
    ax_surface.set_xticks(tick_vals_3d)
    ax_surface.set_yticks(tick_vals_3d)
    if tick_mode == "int":
        ax_surface.set_xticklabels([_format_int255(t) for t in tick_vals_3d])
        ax_surface.set_yticklabels([_format_int255(t) for t in tick_vals_3d])
    else:
        ax_surface.set_xticklabels([_format_frac255(t) for t in tick_vals_3d])
        ax_surface.set_yticklabels([_format_frac255(t) for t in tick_vals_3d])

    # If the fraction labels are long (e.g., 230/255), the x/y corner labels overlap in 3D.
    # Prefer keeping x tick labels and dropping y tick labels (the 2D subplot already shows both).
    if tick_mode == "frac" and _is_long_frac_label(plot_range):
        ax_surface.set_yticklabels([""] * len(tick_vals_3d))
    # Separate x/y tick label pads to reduce overlaps in 3D.
    if tick_mode == "frac":
        ax_surface.tick_params(axis="x", labelsize=7, pad=4)
        ax_surface.tick_params(axis="y", labelsize=7, pad=12)
    else:
        ax_surface.tick_params(axis="x", labelsize=8, pad=4)
        ax_surface.tick_params(axis="y", labelsize=8, pad=10)
    ax_surface.tick_params(axis="z", labelsize=8, pad=2)

    # Reduce overlaps at the front corner in 3D.
    if tick_mode == "frac":
        for lbl in ax_surface.get_xticklabels():
            lbl.set_rotation(25)
            lbl.set_ha("right")
        for lbl in ax_surface.get_yticklabels():
            lbl.set_rotation(-25)
            lbl.set_ha("left")
    try:
        ax_surface.set_proj_type("ortho")
    except Exception:
        pass
    ax_surface.view_init(elev=28, azim=-15)


def _parse_indices(values: List[str]) -> List[int]:
    indices: List[int] = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        indices.append(int(v))
    if not indices:
        raise ValueError("At least one index is required.")
    return indices


def _parse_float_maybe_fraction(value: str) -> float:
    """Parse floats like '0.5' or simple fractions like '16/255'."""
    text = str(value).strip()
    if "/" in text:
        num_text, den_text = text.split("/", 1)
        num = float(num_text.strip())
        den = float(den_text.strip())
        if den == 0:
            raise ValueError("Denominator must be non-zero.")
        return num / den
    return float(text)


@torch.no_grad()
def _pick_two_clean_correct_different_class_indices(
    dataset,
    model: nn.Module,
    device: torch.device,
    *,
    exclude: set[int],
    batch_size: int = 256,
    num_workers: int = 0,
) -> List[int]:
    """Pick two indices with different GT labels that are correctly classified on clean inputs."""

    class _Indexed(Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, i: int):
            x, y = self.base[i]
            return x, y, i

    loader = DataLoader(
        _Indexed(dataset),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )

    chosen: List[Tuple[int, int]] = []
    chosen_labels: set[int] = set()
    for x_norm, y, idx in loader:
        x_norm = x_norm.to(device)
        y = y.to(device)
        idx = idx.detach().cpu().tolist()
        logits = model(x_norm)
        pred = logits.argmax(dim=1)
        correct = pred.eq(y).detach().cpu().numpy()
        if not correct.any():
            continue
        for j, ok in enumerate(correct.tolist()):
            if not ok:
                continue
            i = int(idx[j])
            if i in exclude:
                continue
            lab = int(y[j].item())
            if not chosen:
                chosen.append((i, lab))
                chosen_labels.add(lab)
            elif lab not in chosen_labels:
                chosen.append((i, lab))
                chosen_labels.add(lab)
            if len(chosen) == 2:
                return [chosen[0][0], chosen[1][0]]

    raise RuntimeError(
        "Failed to find two CIFAR-10 test samples that are clean-correct and from different classes. "
        "Try removing exclusions or check model accuracy."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Plot (r, r⊥) loss landscapes for INPUT-ICNN checkpoints.")
    p.add_argument(
        "--ckpt",
        type=str,
        required=False,
        default="/mnt/lts4/scratch/students/aabdolla/LAT/R2_INPUT_icnn_lambda_30_epochs_adv_30_l2_PGD_1024_512_512_256_128_64.pth",
        help="Path to INPUT-ICNN checkpoint (dict with keys: phi/head/icnn/args).",
    )
    p.add_argument("--data-root", type=str, default="./data", help="CIFAR-10 root directory.")
    p.add_argument("--no-download", action="store_true", help="Disable CIFAR-10 download.")
    p.add_argument("--device", type=str, default="", help="Device override, e.g. 'cuda:0' or 'cpu'.")
    p.add_argument("--seed", type=int, default=1, help="Random seed (orthogonal direction reproducibility).")
    p.add_argument(
        "--acc-batch-size",
        type=int,
        default=256,
        help="Batch size for the clean-accuracy sanity check.",
    )
    p.add_argument(
        "--acc-num-workers",
        type=int,
        default=2,
        help="DataLoader workers for the clean-accuracy sanity check.",
    )
    p.add_argument(
        "--allow-misclassified",
        action="store_true",
        help="Allow plotting/attacking selected indices even if misclassified on clean inputs.",
    )

    # PGD parameters (must match evaluate_wrm_lat_cifar10_variants.py defaults used in the prompt).
    p.add_argument(
        "--p",
        type=str,
        default="2",
        choices=["2", "inf"],
        help="Attack norm for input-space PGD (pixel units). Use '2' for L2 (default).",
    )
    p.add_argument("--eps", type=float, default=0.5, help="Epsilon (pixel units) for PGD; for --p 2 this is L2 ε.")
    p.add_argument("--steps", type=int, default=20, help="PGD iterations.")
    p.add_argument("--restarts", type=int, default=5, help="PGD random restarts.")
    p.add_argument(
        "--step-size",
        type=float,
        default=0.0,
        help="PGD step size (pixel units). If <=0, uses auto=2*eps/steps (same as utils.auto_pgd_step_size).",
    )

    # Landscape grid parameters.
    p.add_argument(
        "--plot-range",
        type=_parse_float_maybe_fraction,
        default=16.0 / 255.0,
        help="Coefficient range along each direction (pixel units).",
    )
    p.add_argument(
        "--inner-range",
        type=_parse_float_maybe_fraction,
        default=8.0 / 255.0,
        help="Optional inner dashed constraint marker (pixel units). For --p 2 this is a radius (circle); for --p inf this is half-width (square). Use 0 to disable.",
    )
    p.add_argument("--grid-size", type=int, default=51, help="Grid resolution per axis.")
    p.add_argument("--grid-batch-size", type=int, default=256, help="Batch size for grid evaluation.")
    p.add_argument(
        "--indices",
        nargs="+",
        default=["0", "4"],
        help="Two CIFAR-10 test indices to plot (0-based) with DIFFERENT ground-truth classes. Default: 0 4.",
    )
    p.add_argument(
        "--tick-mode",
        type=str,
        default="auto",
        choices=["auto", "frac", "int"],
        help="Tick label style: 'frac' uses n/255, 'int' uses n with axis label ×1/255, 'auto' picks a readable default.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="fig/loss_landscape.png",
        help="Output path for the combined figure.",
    )

    args = p.parse_args()
    set_deterministic(args.seed)

    device = torch.device(args.device) if args.device else get_device()
    print(f"[Device] Using {device}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "head" not in ckpt:
        raise RuntimeError("Expected checkpoint dict containing key 'head'.")
    if "phi" in ckpt and isinstance(ckpt["phi"], dict) and len(ckpt["phi"]) > 0:
        raise RuntimeError("This script expects INPUT-ICNN checkpoints with phi = Identity (empty state_dict).")

    head = _build_cifar10_head_from_state_dict(ckpt["head"], num_classes=10).to(device)
    head.eval()
    p_value = 2 if args.p == "2" else float("inf")
    print(f"[Attack] PGD-{args.p}: eps={float(args.eps)}, steps={int(args.steps)}, restarts={int(args.restarts)}")

    # Sanity check: clean accuracy on CIFAR-10 test set.
    acc_loader, _ = get_cifar10_loader(
        split="test",
        batch_size=int(args.acc_batch_size),
        num_workers=int(args.acc_num_workers),
        data_root=args.data_root,
        augment_train=False,
        shuffle_train=False,
        pin_memory=(device.type == "cuda"),
        download=not args.no_download,
        seed=args.seed,
    )
    clean_loss, clean_acc = evaluate_clean_accuracy(head, acc_loader, device)
    print(f"[Sanity] Clean CIFAR-10 test — loss: {clean_loss:.4f}, acc: {clean_acc*100:.2f}%")

    test_loader, _ = get_cifar10_loader(
        split="test",
        batch_size=1,
        num_workers=0,
        data_root=args.data_root,
        augment_train=False,
        shuffle_train=False,
        pin_memory=False,
        download=not args.no_download,
        seed=args.seed,
    )
    dataset = test_loader.dataset

    indices = _parse_indices(list(args.indices))
    if len(indices) != 2:
        raise ValueError("To match the target figure layout, provide exactly two indices (e.g., 0 4).")

    def _label_for_index(i: int) -> int:
        if hasattr(dataset, "targets"):
            return int(dataset.targets[i])  # type: ignore[attr-defined]
        return int(dataset[i][1])

    y0 = _label_for_index(indices[0])
    y1 = _label_for_index(indices[1])
    if y0 == y1:
        class_name = None
        if hasattr(dataset, "classes") and 0 <= y0 < len(dataset.classes):  # type: ignore[attr-defined]
            class_name = str(dataset.classes[y0])  # type: ignore[attr-defined]
        suffix = f" ({class_name})" if class_name else ""
        print(
            "[Warn] Provided --indices belong to the same CIFAR-10 class; "
            f"labels={y0}{suffix} for indices {indices[0]} and {indices[1]}. "
            "Selecting two alternative indices with different classes that are clean-correct."
        )
        indices = _pick_two_clean_correct_different_class_indices(
            dataset,
            head,
            device,
            exclude=set(indices),
            batch_size=int(args.acc_batch_size),
            num_workers=0,
        )
        y0 = _label_for_index(indices[0])
        y1 = _label_for_index(indices[1])
        name0 = (
            f" ({dataset.classes[y0]})"  # type: ignore[attr-defined]
            if hasattr(dataset, "classes") and 0 <= y0 < len(dataset.classes)  # type: ignore[attr-defined]
            else ""
        )
        name1 = (
            f" ({dataset.classes[y1]})"  # type: ignore[attr-defined]
            if hasattr(dataset, "classes") and 0 <= y1 < len(dataset.classes)  # type: ignore[attr-defined]
            else ""
        )
        print(
            f"[Pick] Using indices {indices[0]}{name0} and {indices[1]}{name1} "
            f"with labels {y0} and {y1}."
        )

    coords = torch.linspace(-float(args.plot_range), float(args.plot_range), int(args.grid_size), device=device)
    coords_np = coords.detach().cpu().numpy()
    inner_range = None if float(args.inner_range) <= 0 else float(args.inner_range)
    constraint_shape = "circle" if args.p == "2" else "square"
    tick_mode = (
        _tick_label_mode_from_plot_range(float(args.plot_range))
        if args.tick_mode == "auto"
        else str(args.tick_mode)
    )

    results: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in indices:
        x_norm, y_int = dataset[idx]
        x_norm = x_norm.unsqueeze(0).to(device)
        y = torch.tensor([int(y_int)], device=device, dtype=torch.long)

        print(f"[Example] idx={idx} (0-based), label={int(y_int)}")
        with torch.no_grad():
            logits_clean = head(x_norm)
            pred_clean = int(logits_clean.argmax(dim=1).item())
            loss_clean = float(F.cross_entropy(logits_clean, y, reduction="mean").item())
        print(f"[Clean] pred={pred_clean}, CE={loss_clean:.4f}")
        if pred_clean != int(y_int) and not args.allow_misclassified:
            raise RuntimeError(
                f"Selected index {idx} is misclassified on clean input "
                f"(pred={pred_clean}, label={int(y_int)}). "
                "Choose different --indices or pass --allow-misclassified."
            )

        with torch.enable_grad():
            best_delta, best_loss = _pgd_best_delta_pix(
                head,
                x_norm,
                y,
                p=p_value,
                eps=float(args.eps),
                steps=int(args.steps),
                restarts=int(args.restarts),
                step_size=(None if float(args.step_size) <= 0 else float(args.step_size)),
            )
        x0_pix = to_pixel(x_norm).detach()
        r_unit = _unit_direction(best_delta)
        if float(best_delta.view(1, -1).norm(p=2, dim=1).item()) < 1e-10:
            print("[Warn] PGD returned ~zero delta; using a random direction for r.")
            r_unit = per_sample_l2_normalize(torch.randn_like(r_unit))
        u_unit = _random_unit_direction_orthogonal_to(r_unit)

        l2_norm = float(best_delta.view(1, -1).norm(p=2, dim=1).item())
        linf_norm = float(best_delta.abs().view(1, -1).max(dim=1)[0].item())
        print(
            f"[PGD] best CE={best_loss:.4f} | ||delta||2={l2_norm:.4f}, ||delta||inf={linf_norm:.4f}"
        )

        corr, loss = _eval_plane_grid(
            head,
            x0_pix,
            y,
            r_unit,
            u_unit,
            coords,
            batch_size=int(args.grid_batch_size),
        )
        results.append((corr, loss))

    # Plot combined figure: (decision, surface) for each example.
    # Use spacer columns between decision/3D plots (and between pairs) to prevent overlaps.
    
    fig = plt.figure(figsize=(20.5, 7.8), dpi=300)
    gs = fig.add_gridspec(
        1,
        7,
        # Spacer columns prevent axis-label bleed between neighboring subplots.
        width_ratios=[1.0, 0.55, 1.25, 0.65, 1.0, 0.45, 1.00],
        wspace=0.001,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax_sp1 = fig.add_subplot(gs[0, 1])
    ax_sp1.axis("off")
    ax1 = fig.add_subplot(gs[0, 2], projection="3d")
    ax_gap = fig.add_subplot(gs[0, 3])
    ax_gap.axis("off")
    ax2 = fig.add_subplot(gs[0, 4])
    ax_sp2 = fig.add_subplot(gs[0, 5])
    ax_sp2.axis("off")
    ax3 = fig.add_subplot(gs[0, 6], projection="3d")

    _plot_example_pair(
        fig,
        ax0,
        ax1,
        coords_np=coords_np,
        correct=results[0][0],
        loss=results[0][1],
        plot_range=float(args.plot_range),
        inner_range=inner_range,
        constraint_shape=constraint_shape,
        tick_mode=tick_mode,
    )
    _plot_example_pair(
        fig,
        ax2,
        ax3,
        coords_np=coords_np,
        correct=results[1][0],
        loss=results[1][1],
        plot_range=float(args.plot_range),
        inner_range=inner_range,
        constraint_shape=constraint_shape,
        tick_mode=tick_mode,
    )

    # Keep margins consistent and avoid overlaps/cropping.
    fig.subplots_adjust(left=0.05, right=0.9, top=0.98, bottom=0.16)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved: {out_path}")


if __name__ == "__main__":
    main()
