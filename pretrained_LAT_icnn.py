#!/usr/bin/env python3

"""
Latent-space adversarial training with an ICNN adversary.

This script mirrors the overall structure of ``pretrained_LAT.py`` but replaces
the inner maximization routine with an implicit optimal transport map defined
as the gradient of a convex potential :math:`T(z) = \\nabla_z \\phi_\\omega(z)`.
The potential :math:`\\phi_\\omega` is parameterized as a fully input-convex
network (ICNN) with non-negative hidden-to-hidden weights.  Training alternates
between:

* **Adversary step (gradient ascent on ``ω``)**: Maximize the robust loss
  :math:`\\mathbb{E}[\\ell(H_{\\theta_h}(T(z)), y) - \\lambda \\|z - T(z)\\|^2]`.
* **Model step (gradient descent on ``θ``)**: Update encoder ``Φ`` and head
  ``H`` on the adversarial latents, treating the ICNN parameters as constants
  (Danskin-style outer gradient).

Most utilities (data loading, pretrained split, evaluation) are reused from
``pretrained_LAT.py`` and ``utils.py`` to avoid duplication.
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import (
    evaluate_under_input_pgd,
    get_cifar10_loaders,
    get_device,
    parameterized_filename,
    set_deterministic,
)

# Reuse the pretrained split/build helpers and clean evaluation loop.
from pretrained_LAT import (  # type: ignore
    build_split_resnet18,
    evaluate,
    jacobian_aware_latent_eps,
    load_pretrained_resnet18,
)

PENALTY_LAMBDA_MIN = 1e-4
PENALTY_LAMBDA_MAX = 1e4
CALIBRATION_SCALE_MIN = 0.1
CALIBRATION_SCALE_MAX = 10.0
CALIBRATION_SMOOTHING = 0.1


class TransportDeltaTracker:
    """Utility for recording |T(z)-z|^2 statistics during training."""

    def __init__(self, enabled: bool, max_batches: int, plot_dir: str):
        self.enabled = bool(enabled)
        self.max_batches = max(0, int(max_batches))
        self.plot_dir = Path(plot_dir)
        self.per_ascent: Dict[int, List[float]] = defaultdict(list)
        self.epoch_records: List[Dict[str, object]] = []
        self._current_epoch_values: List[float] = []
        self._current_epoch_meta: Optional[Tuple[int, str]] = None

    def start_epoch(self, epoch_idx: int, phase: str) -> None:
        if not self.enabled:
            return
        self._current_epoch_values = []
        self._current_epoch_meta = (int(epoch_idx), str(phase))

    def should_track(self, batch_idx: int) -> bool:
        return self.enabled and batch_idx < self.max_batches

    def record_ascent(self, step_idx: int, delta_sq: torch.Tensor, batch_idx: int) -> None:
        if not self.should_track(batch_idx):
            return
        values = delta_sq.detach().cpu().view(-1).tolist()
        if values:
            self.per_ascent[int(step_idx)].extend(float(v) for v in values)

    def record_epoch_batch(self, delta_sq: torch.Tensor, batch_idx: int) -> None:
        if not self.should_track(batch_idx):
            return
        if self._current_epoch_meta is None:
            return
        values = delta_sq.detach().cpu().view(-1).tolist()
        if values:
            self._current_epoch_values.extend(float(v) for v in values)

    def finish_epoch(self) -> None:
        if not self.enabled or self._current_epoch_meta is None:
            return
        epoch_idx, phase = self._current_epoch_meta
        if self._current_epoch_values:
            mean_val = float(np.mean(self._current_epoch_values))
        else:
            mean_val = float("nan")
        self.epoch_records.append(
            {"epoch": epoch_idx, "phase": phase, "mean_delta_sq": mean_val}
        )
        self._current_epoch_values = []
        self._current_epoch_meta = None

    def has_data(self) -> bool:
        ascent_has = any(len(v) > 0 for v in self.per_ascent.values())
        epoch_has = any(not math.isnan(rec["mean_delta_sq"]) for rec in self.epoch_records)
        return ascent_has or epoch_has

    def ascent_summary(self) -> List[Tuple[int, float]]:
        summary: List[Tuple[int, float]] = []
        for step_idx in sorted(self.per_ascent.keys()):
            values = self.per_ascent[step_idx]
            if not values:
                continue
            arr = np.array(values, dtype=np.float64)
            summary.append((step_idx, float(arr.mean())))
        return summary

    def epoch_summary(self) -> List[Tuple[int, str, float]]:
        summary: List[Tuple[int, str, float]] = []
        for record in self.epoch_records:
            mean_val = float(record["mean_delta_sq"])
            if math.isnan(mean_val):
                continue
            summary.append((int(record["epoch"]), str(record["phase"]), mean_val))
        return summary


def save_transport_delta_plot(
    tracker: TransportDeltaTracker,
    args,
    run_id: str,
) -> None:
    """Generate summary plots for tracked transport delta norms."""
    if not tracker.enabled or not tracker.has_data():
        return

    ascent_summary = tracker.ascent_summary()
    epoch_summary = tracker.epoch_summary()
    if not ascent_summary and not epoch_summary:
        print("Transport delta tracking enabled but no finite statistics collected; skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes = np.atleast_1d(axes).ravel()

    # Left plot: mean delta squared versus ascent iteration.
    ax_iter = axes[0]
    if ascent_summary:
        steps, means = zip(*ascent_summary)
        ax_iter.plot(steps, means, marker="o")
        ax_iter.set_xlabel("Ascent iteration")
        ax_iter.set_ylabel("Mean |T(z) - z|^2")
        ax_iter.set_title("Delta norms per ascent step")
        ax_iter.grid(True, alpha=0.2)
    else:
        ax_iter.axis("off")
        ax_iter.text(
            0.5,
            0.5,
            "No ascent data recorded",
            ha="center",
            va="center",
            transform=ax_iter.transAxes,
        )

    # Right plot: mean delta squared versus epoch index.
    ax_epoch = axes[1]
    if epoch_summary:
        epochs = [item[0] for item in epoch_summary]
        phases = [item[1] for item in epoch_summary]
        values = [item[2] for item in epoch_summary]
        ax_epoch.plot(epochs, values, marker="o")
        ax_epoch.set_xlabel("Epoch")
        ax_epoch.set_ylabel("Mean |T(z) - z|^2")
        ax_epoch.set_title("Delta norms across epochs")
        ax_epoch.grid(True, alpha=0.2)
        ax_epoch.set_xticks(epochs)
        if len(set(phases)) > 1:
            labels = [f"{epoch}\n{phase}" for epoch, phase in zip(epochs, phases)]
            ax_epoch.set_xticklabels(labels)
    else:
        ax_epoch.axis("off")
        ax_epoch.text(
            0.5,
            0.5,
            "No epoch data recorded",
            ha="center",
            va="center",
            transform=ax_epoch.transAxes,
        )

    fig.suptitle("|T(z) - z|^2 tracking", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plot_dir = tracker.plot_dir / run_id
    plot_dir.mkdir(parents=True, exist_ok=True)
    base_path = plot_dir / "transport_delta_tracking.png"
    plot_params = {
        "icnn": "-".join(map(str, args.icnn_hidden)),
        "steps": args.icnn_ascent_steps,
        "tracked_batches": args.track_transport_batches,
    }
    plot_path = parameterized_filename(base_path, plot_params)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved transport delta tracking plot to {plot_path}")


class NonNegativeLinear(nn.Module):
    """Linear map with weights constrained to be element-wise non-negative."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight_raw)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.softplus(self.weight_raw)
        return F.linear(x, weight, self.bias)

    @torch.no_grad()
    def project_non_negative(self) -> None:
        # Forward pass already enforces non-negativity via softplus; no projection needed.
        return


class NonNegativeConv2d(nn.Module):
    """Conv2d module with weights constrained to be element-wise non-negative."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        weight_shape = (out_channels, in_channels, self.kernel_size, self.kernel_size)
        self.weight_raw = nn.Parameter(torch.empty(weight_shape))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.xavier_uniform_(self.weight_raw)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.softplus(self.weight_raw)
        return F.conv2d(x, weight, bias=self.bias, padding=self.padding)

    @torch.no_grad()
    def project_non_negative(self) -> None:
        return


class InputConvexPotential(nn.Module):
    """Fully input-convex neural network (FICNN) for latent potentials."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "relu",
        strong_convexity: float = 1.0,
        input_shape: Optional[Sequence[int]] = None,
        use_convs: bool = False,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_sizes: List[int] = list(hidden_sizes)
        if len(self.hidden_sizes) == 0:
            raise ValueError("ICNN requires at least one hidden layer.")

        self.use_convs = bool(use_convs)
        self.input_shape: Optional[Tuple[int, ...]] = (
            tuple(int(v) for v in input_shape) if input_shape is not None else None
        )
        self.conv_kernel_size = int(conv_kernel_size)
        if self.use_convs:
            if self.input_shape is None or len(self.input_shape) != 3:
                raise ValueError(
                    "Convolutional ICNN requires spatial latent shape (C, H, W)."
                )
            if self.conv_kernel_size <= 0:
                raise ValueError("conv_kernel_size must be positive.")

        if activation == "relu":
            self.nonlin: nn.Module = nn.ReLU()
        elif activation == "softplus":
            self.nonlin = nn.Softplus(beta=1.0)
        else:
            raise ValueError(f"Unsupported ICNN activation: {activation}")

        self.strong_convexity = float(strong_convexity)

        self.z_linears = nn.ModuleList()
        self.h_linears = nn.ModuleList()
        prev_hidden = None

        if self.use_convs:
            in_channels = int(self.input_shape[0])
            padding = self.conv_kernel_size // 2
            for width in self.hidden_sizes:
                self.z_linears.append(
                    nn.Conv2d(
                        in_channels,
                        width,
                        kernel_size=self.conv_kernel_size,
                        padding=padding,
                        bias=True,
                    )
                )
                if prev_hidden is None:
                    self.h_linears.append(None)  # type: ignore
                else:
                    self.h_linears.append(
                        NonNegativeConv2d(
                            prev_hidden,
                            width,
                            kernel_size=self.conv_kernel_size,
                            padding=padding,
                            bias=True,
                        )
                    )
                prev_hidden = width
            self.hidden_output = NonNegativeConv2d(
                self.hidden_sizes[-1], 1, kernel_size=1, padding=0, bias=True
            )
            self.input_skip = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0, bias=True)
        else:
            for width in self.hidden_sizes:
                self.z_linears.append(nn.Linear(self.input_dim, width, bias=True))
                if prev_hidden is None:
                    self.h_linears.append(None)  # type: ignore
                else:
                    self.h_linears.append(NonNegativeLinear(prev_hidden, width, bias=True))
                prev_hidden = width
            self.hidden_output = NonNegativeLinear(self.hidden_sizes[-1], 1, bias=True)
            self.input_skip = nn.Linear(self.input_dim, 1, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.z_linears:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for layer in self.h_linears:
            if layer is not None:
                if isinstance(layer, NonNegativeLinear):
                    nn.init.xavier_uniform_(layer.weight_raw)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, NonNegativeConv2d):
                    nn.init.xavier_uniform_(layer.weight_raw)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        if isinstance(self.hidden_output, NonNegativeLinear):
            nn.init.xavier_uniform_(self.hidden_output.weight_raw)
            nn.init.zeros_(self.hidden_output.bias)
        elif isinstance(self.hidden_output, NonNegativeConv2d):
            nn.init.xavier_uniform_(self.hidden_output.weight_raw)
            if self.hidden_output.bias is not None:
                nn.init.zeros_(self.hidden_output.bias)
        if isinstance(self.input_skip, nn.Linear):
            nn.init.xavier_uniform_(self.input_skip.weight)
            nn.init.zeros_(self.input_skip.bias)
        elif isinstance(self.input_skip, nn.Conv2d):
            nn.init.kaiming_normal_(self.input_skip.weight, nonlinearity="linear")
            if self.input_skip.bias is not None:
                nn.init.zeros_(self.input_skip.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.size(0)
        if self.use_convs:
            z_spatial = z
            h = None
            for z_linear, h_linear in zip(self.z_linears, self.h_linears):
                z_term = z_linear(z_spatial)
                if h is None:
                    h = self.nonlin(z_term)
                else:
                    h = self.nonlin(z_term + h_linear(h))

            assert h is not None
            quadratic = 0.5 * self.strong_convexity * (z_spatial.pow(2).sum(dim=1, keepdim=True))
            output = quadratic + self.input_skip(z_spatial) + self.hidden_output(h)
            return output.view(batch, -1).sum(dim=1)

        z_flat = z.view(batch, -1)
        h = None
        for z_linear, h_linear in zip(self.z_linears, self.h_linears):
            z_term = z_linear(z_flat)
            if h is None:
                h = self.nonlin(z_term)
            else:
                h = self.nonlin(z_term + h_linear(h))

        assert h is not None  # kept for type checkers
        quadratic = 0.5 * self.strong_convexity * (z_flat.pow(2).sum(dim=1, keepdim=True))
        output = quadratic + self.input_skip(z_flat) + self.hidden_output(h)
        return output.squeeze(-1)

    def gradient(self, z: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
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


def _to_device(batch, device: torch.device):
    x, y = batch
    return x.to(device), y.to(device)


def adversarial_pushforward(
    icnn: InputConvexPotential, z: torch.Tensor, detach_for_model: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    if detach_for_model:
        z_source = z.detach()
        z_leaf = z_source.clone().requires_grad_(True)
        with torch.enable_grad():
            z_push = icnn.gradient(z_leaf, create_graph=False)
        z_push = z_push.detach()
        delta = (z_push - z_source).detach()
        z_adv_fixed = z + delta
        return z_adv_fixed, delta

    z_leaf = z.detach().requires_grad_(True)
    with torch.enable_grad():
        z_adv = icnn.gradient(z_leaf, create_graph=True)
    delta = z_adv - z_leaf
    return z_adv, delta


def _parse_hidden_units(token: str) -> int:
    """Parse hidden layer width tokens while tolerating bracket/comma syntax."""
    cleaned = token.strip().strip("[],")
    if cleaned == "":
        raise argparse.ArgumentTypeError(f"Invalid hidden size token: {token!r}")
    try:
        return int(cleaned)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {token!r}") from exc


def _clamp_penalty_lambda(value: float) -> float:
    if not math.isfinite(value):
        return PENALTY_LAMBDA_MIN
    return float(min(max(value, PENALTY_LAMBDA_MIN), PENALTY_LAMBDA_MAX))


def _reduce_latents_for_plot(
    z: torch.Tensor, z_adv: torch.Tensor, method: str, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Project latent pairs into 2D using PCA or t-SNE."""
    z_flat = z.view(z.size(0), -1).cpu().float()
    z_adv_flat = z_adv.view(z_adv.size(0), -1).cpu().float()
    if z_flat.size(0) < 2:
        raise ValueError("Need at least two samples for visualization.")

    if method == "pca":
        z_np = z_flat.numpy()
        z_adv_np = z_adv_flat.numpy()
        if (
            not np.isfinite(z_np).all()
            or not np.isfinite(z_adv_np).all()
        ):
            raise ValueError("Non-finite values encountered when reducing latents.")

        mean = z_np.mean(axis=0, keepdims=True)
        centered = z_np - mean
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            vt = np.eye(centered.shape[1], dtype=np.float64)
        basis = vt[:2].T
        if basis.shape[1] < 2:
            pad = np.zeros((basis.shape[0], 2 - basis.shape[1]), dtype=basis.dtype)
            basis = np.concatenate([basis, pad], axis=1)
        coords_z = centered @ basis
        coords_adv = (z_adv_np - mean) @ basis
        return coords_z.astype(np.float32), coords_adv.astype(np.float32)

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("t-SNE requested but scikit-learn is unavailable; falling back to PCA.")
        return _reduce_latents_for_plot(z, z_adv, "pca", seed)

    stacked = torch.cat([z_flat, z_adv_flat], dim=0).numpy()
    if not np.isfinite(stacked).all():
        raise ValueError("Non-finite values encountered when preparing t-SNE input.")
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=seed)
    embedding = tsne.fit_transform(stacked)
    coords_z = embedding[: z_flat.size(0)]
    coords_adv = embedding[z_flat.size(0) :]
    return coords_z.astype(np.float32), coords_adv.astype(np.float32)


def visualize_transport_map(
    phi: nn.Module,
    icnn: InputConvexPotential,
    loader: DataLoader,
    device: torch.device,
    args,
    run_id: str,
    split_name: str,
) -> None:
    """Generate and save a 2D visualization of the transport map."""
    max_samples = max(1, int(args.transport_viz_samples))
    phi_mode = phi.training
    icnn_mode = icnn.training
    phi.eval()
    icnn.eval()

    z_list: List[torch.Tensor] = []
    z_adv_list: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    collected = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            z = phi(x)
        z_detached = z.detach()
        with torch.enable_grad():
            z_leaf = z_detached.clone().requires_grad_(True)
            z_push = icnn.gradient(z_leaf, create_graph=False).detach()
        z_list.append(z_detached.cpu())
        z_adv_list.append(z_push.cpu())
        labels.append(y.cpu())
        collected += x.size(0)
        if collected >= max_samples:
            break

    phi.train(phi_mode)
    icnn.train(icnn_mode)

    if not z_list:
        print("Transport visualization skipped: no samples collected.")
        return

    z_tensor = torch.cat(z_list, dim=0)[:max_samples]
    z_adv_tensor = torch.cat(z_adv_list, dim=0)[:max_samples]
    label_tensor = torch.cat(labels, dim=0)[:max_samples]

    flat_z = z_tensor.reshape(z_tensor.size(0), -1)
    flat_adv = z_adv_tensor.reshape(z_adv_tensor.size(0), -1)
    finite_mask = torch.isfinite(flat_z).all(dim=1) & torch.isfinite(flat_adv).all(dim=1)
    if finite_mask.sum().item() == 0:
        print("Transport visualization skipped: no finite latent samples available.")
        return
    if finite_mask.sum().item() < z_tensor.size(0):
        dropped = z_tensor.size(0) - finite_mask.sum().item()
        print(f"Transport visualization: dropped {dropped} samples with non-finite values.")
    z_tensor = z_tensor[finite_mask]
    z_adv_tensor = z_adv_tensor[finite_mask]
    label_tensor = label_tensor[finite_mask]

    if z_tensor.size(0) < 2:
        print("Transport visualization skipped: fewer than two valid samples.")
        return

    try:
        coords_z, coords_adv = _reduce_latents_for_plot(
            z_tensor, z_adv_tensor, args.transport_viz_method, seed=args.seed
        )
    except ValueError as err:
        warnings.warn(
            f"Transport visualization encountered an issue ({err}); falling back to PCA.",
            RuntimeWarning,
        )
        coords_z, coords_adv = _reduce_latents_for_plot(
            z_tensor, z_adv_tensor, "pca", seed=args.seed
        )

    class_colors = plt.cm.tab10(label_tensor.numpy() % 10)
    dx = coords_adv[:, 0] - coords_z[:, 0]
    dy = coords_adv[:, 1] - coords_z[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        coords_z[:, 0],
        coords_z[:, 1],
        c=label_tensor.numpy(),
        cmap="tab10",
        s=20,
        alpha=0.9,
        edgecolors="none",
    )
    ax.quiver(
        coords_z[:, 0],
        coords_z[:, 1],
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=class_colors,
        alpha=0.4,
        linewidths=0.5,
    )
    ax.set_title(
        f"Transport map ({args.transport_viz_method.upper()}) on {split_name} split"
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Class label")
    fig.tight_layout()

    viz_dir = Path(args.transport_viz_dir) / run_id
    viz_dir.mkdir(parents=True, exist_ok=True)
    plot_params: Dict[str, object] = {
        "split": split_name,
        "method": args.transport_viz_method,
        "samples": args.transport_viz_samples,
        "seed": args.seed,
        "cut": args.cut_layer,
        "icnn": "-".join(map(str, args.icnn_hidden)),
        "lambda": args.penalty_lambda,
    }
    base_path = viz_dir / "transport_map.png"
    plot_path = parameterized_filename(base_path, plot_params)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved transport map visualization to {plot_path}")


def estimate_mean_grad_norm(
    phi: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 2,
) -> float:
    """Estimate E[||∇_u ℓ||] over a small number of batches."""
    phi_mode = phi.training
    head_mode = head.training
    phi.eval()
    head.eval()

    norms: List[float] = []
    with torch.enable_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            x = x.to(device)
            y = y.to(device)
            z = phi(x).detach().requires_grad_(True)
            logits = head(z)
            loss = F.cross_entropy(logits, y, reduction="mean")
            grad = torch.autograd.grad(loss, z, create_graph=False, retain_graph=False)[0]
            if not torch.isfinite(grad).all():
                continue
            norms.append(grad.reshape(grad.size(0), -1).norm(dim=1).mean().item())

    phi.train(phi_mode)
    head.train(head_mode)
    return float(np.mean(norms)) if norms else 0.0


def compute_avg_delta_norm(
    phi: nn.Module,
    icnn: InputConvexPotential,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 2,
) -> float:
    """Compute the mean L2 norm of T(z)-z over a few minibatches."""
    phi_mode = phi.training
    icnn_mode = icnn.training
    phi.eval()
    icnn.eval()

    norms: List[float] = []
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        x = x.to(device)
        with torch.no_grad():
            z = phi(x)
        z_detached = z.detach()
        with torch.enable_grad():
            z_adv = icnn.gradient(z_detached, create_graph=False)
        z_adv = z_adv.detach()
        delta = (z_adv - z_detached).reshape(z_adv.size(0), -1)
        if not torch.isfinite(delta).all():
            continue
        norms.append(delta.norm(dim=1).mean().item())

    phi.train(phi_mode)
    icnn.train(icnn_mode)
    return float(np.mean(norms)) if norms else 0.0


def estimate_transport_jacobian_sv(
    phi: nn.Module,
    icnn: InputConvexPotential,
    loader: DataLoader,
    device: torch.device,
    args,
) -> Tuple[float, float]:
    """Estimate max/mean largest singular value of the Jacobian of T(z)."""
    phi_mode = phi.training
    icnn_mode = icnn.training
    phi.eval()
    icnn.eval()

    sv_values: List[float] = []
    batches_considered = 0
    max_batches = max(1, args.jacobian_sv_batches)
    max_samples = max(1, args.jacobian_sv_samples)
    power_iters = max(1, args.jacobian_iters)

    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            z = phi(x)
        batch = min(z.size(0), max_samples)
        if batch == 0:
            continue
        indices = torch.randperm(z.size(0), device=z.device)[:batch]
        for idx in indices:
            base = z[idx.item() : idx.item() + 1].detach()
            vec = torch.randn_like(base)
            vec_norm = vec.view(-1).norm().item()
            if vec_norm < 1e-12:
                continue
            vec = vec / vec_norm
            sigma_val = float("nan")
            for _ in range(power_iters):
                z_leaf = base.clone().detach().requires_grad_(True)
                transport = icnn.gradient(z_leaf, create_graph=True)
                hvp = torch.autograd.grad(
                    transport,
                    z_leaf,
                    grad_outputs=vec,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                hvp_norm = hvp.view(-1).norm().item()
                if not math.isfinite(hvp_norm) or hvp_norm < 1e-12:
                    sigma_val = float("nan")
                    break
                sigma_val = hvp_norm
                vec = hvp.detach() / hvp_norm
            if math.isfinite(sigma_val):
                sv_values.append(sigma_val)
        batches_considered += 1
        if batches_considered >= max_batches:
            break
    phi.train(phi_mode)
    icnn.train(icnn_mode)

    if not sv_values:
        return float("nan"), float("nan")
    sv_tensor = torch.tensor(sv_values)
    return float(sv_tensor.max().item()), float(sv_tensor.mean().item())


def train_one_epoch(
    phi: nn.Module,
    head: nn.Module,
    icnn: InputConvexPotential,
    loader: DataLoader,
    opt_theta: optim.Optimizer,
    opt_icnn: optim.Optimizer,
    device: torch.device,
    method: str,
    penalty_lambda: float,
    head_only: bool,
    icnn_ascent_steps: int,
    tracker: Optional[TransportDeltaTracker] = None,
) -> Tuple[float, float, float, float]:
    phi.train(not head_only)
    head.train()
    icnn.train()
    penalty_lambda = _clamp_penalty_lambda(float(penalty_lambda))
    model_params = [p for p in list(phi.parameters()) + list(head.parameters()) if p.requires_grad]

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_penalty = 0.0
    total_adv_obj = 0.0

    progress = tqdm(enumerate(loader), desc="Train", leave=False, total=len(loader))
    for batch_idx, batch in progress:
        x, y = _to_device(batch, device)
        batch_size = x.size(0)

        if method == "erm":
            opt_theta.zero_grad(set_to_none=True)
            logits = head(phi(x))
            loss = F.cross_entropy(logits, y, reduction="mean")
            loss.backward()
            opt_theta.step()

            with torch.no_grad():
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total_samples += batch_size
            continue

        # --- ICNN adversary update ---
        opt_theta.zero_grad(set_to_none=True)
        z = phi(x)
        z_detached = z.detach()
        adv_objective_last: Optional[torch.Tensor] = None
        for ascent_step in range(max(1, icnn_ascent_steps)):
            opt_icnn.zero_grad(set_to_none=True)
            z_adv_ascent, _ = adversarial_pushforward(icnn, z_detached, detach_for_model=False)
            if not torch.isfinite(z_adv_ascent).all():
                warnings.warn(
                    "Encountered non-finite values in adversarial latents during ascent; "
                    "skipping this ascent step.",
                    RuntimeWarning,
                )
                continue
            logits_adv = head(z_adv_ascent)
            ce_adv = F.cross_entropy(logits_adv, y, reduction="mean")
            z_flat = z_detached.reshape(batch_size, -1)
            z_adv_flat = z_adv_ascent.reshape(batch_size, -1)
            penalty = (z_flat - z_adv_flat).pow(2).sum(dim=1).mean()
            if not torch.isfinite(ce_adv) or not torch.isfinite(penalty):
                warnings.warn(
                    "Non-finite adversarial loss components detected; skipping ascent step.",
                    RuntimeWarning,
                )
                continue
            adv_objective = ce_adv - penalty_lambda * penalty
            (-adv_objective).backward()
            if tracker is not None:
                delta_sq_ascent = (z_adv_ascent.detach() - z_detached).reshape(batch_size, -1).pow(2).sum(dim=1)
                tracker.record_ascent(ascent_step, delta_sq_ascent, batch_idx)
            grad_finite = all(
                (p.grad is None) or torch.isfinite(p.grad).all() for p in icnn.parameters()
            )
            if not grad_finite:
                warnings.warn(
                    "Non-finite gradients in ICNN adversary; skipping optimizer step.",
                    RuntimeWarning,
                )
                opt_icnn.zero_grad(set_to_none=True)
                continue

            # Remove stray gradients on the classifier, which acts as a frozen critic.
            for p in head.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            torch.nn.utils.clip_grad_norm_(icnn.parameters(), max_norm=10.0)
            opt_icnn.step()
            for param in icnn.parameters():
                if not torch.isfinite(param).all():
                    warnings.warn(
                        "Detected non-finite ICNN parameters after update; sanitizing values.",
                        RuntimeWarning,
                    )
                    param.data.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)
            icnn.project_convexity()
            adv_objective_last = adv_objective.detach()

        # --- Model update (Danskin-style outer gradient) ---
        opt_theta.zero_grad(set_to_none=True)
        z_adv_fixed, delta = adversarial_pushforward(icnn, z, detach_for_model=True)
        if not torch.isfinite(z_adv_fixed).all():
            warnings.warn(
                "Skipping batch update due to non-finite adversarial latents.",
                RuntimeWarning,
            )
            continue
        logits = head(z_adv_fixed)
        loss = F.cross_entropy(logits, y, reduction="mean")
        if not torch.isfinite(loss):
            warnings.warn("Skipping batch with non-finite loss value.", RuntimeWarning)
            continue
        loss.backward()
        if model_params:
            torch.nn.utils.clip_grad_norm_(model_params, max_norm=10.0)
        opt_theta.step()
        delta_sq = delta.reshape(batch_size, -1).pow(2).sum(dim=1)
        if tracker is not None:
            tracker.record_epoch_batch(delta_sq, batch_idx)

        with torch.no_grad():
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += batch_size
            total_penalty += delta_sq.mean().item()
            if adv_objective_last is not None and torch.isfinite(adv_objective_last).all():
                total_adv_obj += adv_objective_last.item()

        if total_samples > 0:
            mean_loss = total_loss / total_samples
            mean_acc = total_correct / total_samples
            mean_penalty = total_penalty / max(1, len(loader))
            mean_adv_obj = total_adv_obj / max(1, len(loader))
            progress.set_postfix(
                loss=f"{mean_loss:.4f}",
                acc=f"{mean_acc*100:.2f}%",
                penalty=f"{mean_penalty:.4f}",
                adv_obj=f"{mean_adv_obj:.4f}",
                refresh=True,
            )

    progress.close()
    mean_loss = total_loss / max(1, total_samples)
    mean_acc = total_correct / max(1, total_samples)
    mean_penalty = total_penalty / max(1, len(loader))
    mean_adv_obj = total_adv_obj / max(1, len(loader))
    return mean_loss, mean_acc, mean_penalty, mean_adv_obj


def evaluate_under_icnn(
    phi: nn.Module,
    head: nn.Module,
    icnn: InputConvexPotential,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    phi.eval()
    head.eval()
    icnn.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_penalty = 0.0

    progress = tqdm(loader, desc="ICNN Eval", leave=False)
    for x, y in progress:
        x, y = _to_device((x, y), device)
        z = phi(x)
        z_det = z.detach().requires_grad_(True)
        z_adv = icnn.gradient(z_det, create_graph=False).detach()
        if not torch.isfinite(z_adv).all():
            warnings.warn(
                "Skipping batch during ICNN evaluation due to non-finite adversarial latents.",
                RuntimeWarning,
            )
            continue
        logits = head(z_adv)
        ce = F.cross_entropy(logits, y, reduction="sum")
        if not torch.isfinite(ce):
            warnings.warn("Non-finite ICNN evaluation loss encountered; skipping batch.", RuntimeWarning)
            continue
        total_loss += ce.item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)
        delta = z_det - z_adv
        total_penalty += delta.reshape(delta.size(0), -1).pow(2).sum(dim=1).mean().item()
        if total_samples > 0:
            mean_loss = total_loss / total_samples
            mean_acc = total_correct / total_samples
            mean_penalty = total_penalty / max(1, len(loader))
            progress.set_postfix(
                loss=f"{mean_loss:.4f}",
                acc=f"{mean_acc*100:.2f}%",
                penalty=f"{mean_penalty:.4f}",
                refresh=True,
            )

    progress.close()

    mean_loss = total_loss / max(1, total_samples)
    mean_acc = total_correct / max(1, total_samples)
    mean_penalty = total_penalty / max(1, len(loader))
    return mean_loss, mean_acc, mean_penalty


CSV_HEADER = [
    "run_id",
    "time_iso",
    "epoch",
    "phase",
    "train_loss",
    "train_acc",
    "train_penalty",
    "adv_objective",
    "test_loss",
    "test_acc",
    "icnn_loss",
    "icnn_acc",
    "icnn_penalty",
    "input_pgd_acc",
    "input_pgd_avg_l2",
    "input_pgd_avg_linf",
    "penalty_lambda",
    "lr_theta",
    "lr_omega",
    "icnn_hidden",
    "icnn_strong_convexity",
    "epochs_clean",
    "epochs_adv",
    "penalty_ascent_steps",
    "batch_size",
    "seed",
    "cut_layer",
    "head_only",
    "pretrained_path",
]


def append_row(csv_path: str, row: dict) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, None) for k in CSV_HEADER})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adversarial fine-tuning with an ICNN latent adversary."
    )

    parser.add_argument("--pretrained-path", type=str, default="")
    parser.add_argument("--pretrained-strict", action="store_true")

    parser.add_argument("--epochs-clean", type=int, default=0)
    parser.add_argument("--epochs-adv", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=0)  # legacy fallback
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--lr-theta",
        type=float,
        default=0.1,
        help="Step size γ_θ for encoder and classifier parameters.",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--head-only", action="store_true")

    parser.add_argument(
        "--penalty-lambda",
        type=float,
        default=5.0,
        help="Fixed λ multiplying the quadratic transport penalty.",
    )
    parser.add_argument("--icnn-hidden", type=_parse_hidden_units, nargs="+", default=[256, 256])
    parser.add_argument("--icnn-activation", type=str, choices=["relu", "softplus"], default="relu")
    parser.add_argument("--icnn-strong-convexity", type=float, default=1.0)
    parser.add_argument(
        "--lr-omega",
        type=float,
        default=0.001,
        help="Step size γ_ω for ICNN adversary parameters.",
    )
    parser.add_argument("--icnn-beta1", type=float, default=0.9)
    parser.add_argument("--icnn-beta2", type=float, default=0.999)
    parser.add_argument("--icnn-ascent-steps", type=int, default=10)

    parser.add_argument("--cut-layer", type=str, default="layer4",
                        choices=["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"])
    parser.add_argument("--log-csv", type=str, default="./runs_log_icnn.csv")
    parser.add_argument("--save", type=str, default="")

    parser.add_argument("--jacobian-aware", action="store_true")
    parser.add_argument("--jacobian-batches", type=int, default=2)
    parser.add_argument("--jacobian-iters", type=int, default=2)

    parser.add_argument("--inp-p", type=str, default="inf", choices=["2", "inf"])
    parser.add_argument("--inp-eps", type=float, default=8 / 255)
    parser.add_argument("--inp-steps", type=int, default=20)
    parser.add_argument("--inp-step-size", type=float, default=0.0)
    parser.add_argument("--inp-restarts", type=int, default=5)

    parser.add_argument(
        "--visualize-transport",
        action="store_true",
        help="Generate a 2D visualization of the transport map after training.",
    )
    parser.add_argument(
        "--transport-viz-method",
        type=str,
        choices=["pca", "tsne"],
        default="tsne",
        help="Dimensionality reduction to apply before plotting arrows.",
    )
    parser.add_argument(
        "--transport-viz-samples",
        type=int,
        default=512,
        help="Number of latent samples to include in the transport visualization.",
    )
    parser.add_argument(
        "--transport-viz-dir",
        type=str,
        default="fig/transport_maps",
        help="Directory where transport visualization figures are stored.",
    )
    parser.add_argument(
        "--transport-viz-split",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Dataset split to draw samples from for the transport visualization.",
    )
    parser.add_argument(
        "--track-transport-deltas",
        action="store_true",
        help="Record |T(z)-z|^2 statistics during adversary updates and generate summary plots.",
    )
    parser.add_argument(
        "--track-transport-batches",
        type=int,
        default=4,
        help="Number of minibatches per epoch to monitor when tracking transport deltas.",
    )
    parser.add_argument(
        "--transport-delta-plot-dir",
        type=str,
        default="fig/transport_deltas",
        help="Directory where transport delta tracking plots are stored.",
    )
    parser.add_argument(
        "--estimate-transport-jacobian",
        action="store_true",
        help="Estimate the largest singular value of the transport Jacobian after training.",
    )
    parser.add_argument(
        "--jacobian-sv-batches",
        type=int,
        default=4,
        help="Number of minibatches to sample when estimating transport Jacobian singular values.",
    )
    parser.add_argument(
        "--jacobian-sv-samples",
        type=int,
        default=512,
        help="Maximum number of latent samples used per batch for Jacobian SV estimation.",
    )
    parser.add_argument(
        "--jacobian-sv-split",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Dataset split to sample latents from when estimating transport Jacobian singular values.",
    )
    parser.add_argument(
        "--icnn-conv",
        action="store_true",
        help="Use convolutional ICNN layers when latent features are spatial.",
    )
    parser.add_argument(
        "--icnn-kernel-size",
        type=int,
        default=3,
        help="Kernel size for convolutional ICNN layers (requires --icnn-conv).",
    )
    parser.add_argument(
        "--latent-eps-target",
        type=float,
        default=None,
        help="Desired average L2 norm of T(z)-z; used to calibrate penalty_lambda.",
    )
    parser.add_argument(
        "--calibrate-penalty",
        action="store_true",
        help="Enable automatic calibration of penalty_lambda to match latent-eps-target.",
    )
    parser.add_argument(
        "--gamma-calibration-batches",
        type=int,
        default=4,
        help="Number of minibatches used to estimate gradient norms or delta norms.",
    )

    return parser.parse_args()


def determine_schedule(args) -> Tuple[int, List[str]]:
    if args.epochs_clean > 0 or args.epochs_adv > 0:
        schedule = ["erm"] * args.epochs_clean + ["icnn"] * args.epochs_adv
        total_epochs = len(schedule)
    elif args.epochs > 0:
        schedule = ["icnn"] * args.epochs
        total_epochs = args.epochs
    else:
        raise ValueError("Specify either --epochs or ( --epochs-clean + --epochs-adv ).")
    return total_epochs, schedule


def main():
    args = parse_args()
    set_deterministic(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    total_epochs, schedule = determine_schedule(args)
    print(f"Training schedule: {' → '.join(schedule)}")

    trainloader, testloader = get_cifar10_loaders(batch_size=args.batch_size, seed=args.seed)

    log_csv_path = parameterized_filename(
        args.log_csv,
        {
            "pre": Path(args.pretrained_path).stem if args.pretrained_path else "none",
            "seed": args.seed,
            "cut": args.cut_layer,
            "icnn": "-".join(map(str, args.icnn_hidden)),
        },
    )
    print(f"Logging to: {log_csv_path}")

    base_pretrained = load_pretrained_resnet18(
        pretrained_path=args.pretrained_path,
        num_classes=10,
        strict=args.pretrained_strict,
        device=device,
    )
    phi, head = build_split_resnet18(num_classes=10, cut_layer=args.cut_layer, base=base_pretrained)
    phi.to(device)
    head.to(device)
    if args.head_only:
        for p in phi.parameters():
            p.requires_grad = False

    was_training = phi.training
    phi.eval()
    with torch.no_grad():
        example_batch = next(iter(trainloader))[0][:1].to(device)
        latent_example = phi(example_batch)
        latent_dim = latent_example[0].numel()
        latent_shape = latent_example.shape[1:]
    phi.train(was_training)
    print(f"Latent shape at cut-layer '{args.cut_layer}': {latent_shape} (dim={latent_dim})")

    icnn = InputConvexPotential(
        input_dim=latent_dim,
        hidden_sizes=args.icnn_hidden,
        activation=args.icnn_activation,
        strong_convexity=args.icnn_strong_convexity,
        input_shape=latent_shape,
        use_convs=args.icnn_conv,
        conv_kernel_size=args.icnn_kernel_size,
    ).to(device)

    theta_params = [p for p in list(phi.parameters()) + list(head.parameters()) if p.requires_grad]
    opt_theta = optim.SGD(
        theta_params,
        lr=args.lr_theta,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    icnn_param_groups = []
    # Apply decay selectively: only unconstrained input-to-hidden weights get L2 penalty.
    for name, param in icnn.named_parameters():
        if not param.requires_grad:
            continue
        decay = 0.0 if ("weight_raw" in name or "bias" in name) else 1e-4
        icnn_param_groups.append({"params": [param], "weight_decay": decay})
    opt_icnn = optim.Adam(
        icnn_param_groups,
        lr=args.lr_omega,
        betas=(args.icnn_beta1, args.icnn_beta2),
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_theta, T_max=total_epochs, last_epoch=-1)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    tracker = TransportDeltaTracker(
        enabled=args.track_transport_deltas,
        max_batches=max(0, args.track_transport_batches),
        plot_dir=args.transport_delta_plot_dir,
    )

    if args.calibrate_penalty and args.latent_eps_target and args.latent_eps_target > 0:
        mean_grad_norm = estimate_mean_grad_norm(
            phi,
            head,
            trainloader,
            device,
            num_batches=args.gamma_calibration_batches,
        )
        if math.isfinite(mean_grad_norm) and mean_grad_norm > 0:
            initialized_lambda = mean_grad_norm / args.latent_eps_target
            args.penalty_lambda = _clamp_penalty_lambda(initialized_lambda)
            print(
                f"Initialized penalty_lambda to {args.penalty_lambda:.6f} "
                f"from mean grad norm {mean_grad_norm:.6f} targeting ε_u={args.latent_eps_target:.6f}"
            )
        else:
            print(
                "Calibration requested but gradient norm estimate was non-finite; "
                "keeping existing penalty_lambda."
            )
    elif args.calibrate_penalty:
        print("Calibration requested but latent_eps_target not set or non-positive; skipping.")

    args.penalty_lambda = _clamp_penalty_lambda(float(args.penalty_lambda))

    p_input = 2 if args.inp_p == "2" else float("inf")
    jacobian_ready = False
    L_hat_used = None

    for epoch, phase in enumerate(schedule, start=1):
        print(f"\n=== Epoch {epoch:02d}/{total_epochs} | Phase: {phase.upper()} ===")
        tracker.start_epoch(epoch, phase)

        if phase == "icnn" and args.jacobian_aware and not jacobian_ready:
            L_hat, eps_latent = jacobian_aware_latent_eps(
                phi,
                trainloader,
                device,
                inp_eps=args.inp_eps,
                p_input=p_input,
                n_batches=args.jacobian_batches,
                power_iters=args.jacobian_iters,
            )
            print(f"[Jacobian] L_hat ≈ {L_hat:.6f}, eps_latent_target ≈ {eps_latent:.6f}")
            L_hat_used = L_hat
            jacobian_ready = True

        train_loss, train_acc, penalty_avg, adv_obj = train_one_epoch(
            phi,
            head,
            icnn,
            trainloader,
            opt_theta,
            opt_icnn,
            device,
            method=phase,
            penalty_lambda=args.penalty_lambda,
            head_only=args.head_only,
            icnn_ascent_steps=args.icnn_ascent_steps,
            tracker=tracker,
        )

        test_loss, test_acc = evaluate(phi, head, testloader, device)
        icnn_loss, icnn_acc, icnn_penalty = evaluate_under_icnn(phi, head, icnn, testloader, device)

        if args.inp_steps > 0 and args.inp_eps > 0:
            input_pgd_acc, pgd_info = evaluate_under_input_pgd(
                phi,
                head,
                testloader,
                device,
                p=p_input,
                eps=args.inp_eps,
                steps=args.inp_steps,
                step_size=args.inp_step_size,
                restarts=args.inp_restarts,
            )
            ipgd_l2 = pgd_info["avg_l2"]
            ipgd_linf = pgd_info["avg_linf"]
        else:
            input_pgd_acc, ipgd_l2, ipgd_linf = None, None, None

        msg = (
            f"[Epoch {epoch:02d} | {phase}] train {train_loss:.4f}/{train_acc*100:.2f}% | "
            f"penalty {penalty_avg:.4f} | adv_obj {adv_obj:.4f} | "
            f"test {test_loss:.4f}/{test_acc*100:.2f}% | "
            f"icnn {icnn_loss:.4f}/{icnn_acc*100:.2f}% (pen {icnn_penalty:.4f})"
        )
        if input_pgd_acc is not None:
            msg += f" | input-PGD {input_pgd_acc*100:.2f}% (L2 {ipgd_l2:.4f}, Linf {ipgd_linf:.4f})"
        if jacobian_ready and L_hat_used is not None:
            msg += f" | L_hat {L_hat_used:.4f}"
        print(msg)

        append_row(
            log_csv_path,
            {
                "run_id": run_id,
                "time_iso": datetime.now().isoformat(timespec="seconds"),
                "epoch": epoch,
                "phase": phase,
                "train_loss": round(train_loss, 6),
                "train_acc": round(float(train_acc), 6),
                "train_penalty": round(float(penalty_avg), 6),
                "adv_objective": round(float(adv_obj), 6),
                "test_loss": round(test_loss, 6),
                "test_acc": round(float(test_acc), 6),
                "icnn_loss": round(icnn_loss, 6),
                "icnn_acc": round(float(icnn_acc), 6),
                "icnn_penalty": round(float(icnn_penalty), 6),
                "input_pgd_acc": None if input_pgd_acc is None else round(float(input_pgd_acc), 6),
                "input_pgd_avg_l2": None if ipgd_l2 is None else round(float(ipgd_l2), 6),
                "input_pgd_avg_linf": None if ipgd_linf is None else round(float(ipgd_linf), 6),
                "penalty_lambda": args.penalty_lambda,
                "lr_theta": args.lr_theta,
                "lr_omega": args.lr_omega,
                "icnn_hidden": "-".join(map(str, args.icnn_hidden)),
                "icnn_strong_convexity": args.icnn_strong_convexity,
                "epochs_clean": args.epochs_clean,
                "epochs_adv": args.epochs_adv,
                "penalty_ascent_steps": args.icnn_ascent_steps,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "cut_layer": args.cut_layer,
                "head_only": bool(args.head_only),
                "pretrained_path": args.pretrained_path if args.pretrained_path else "",
            },
        )

        lr_scheduler.step()

        if (
            args.calibrate_penalty
            and args.latent_eps_target
            and args.latent_eps_target > 0
            and phase == "icnn"
        ):
            avg_delta = compute_avg_delta_norm(
                phi,
                icnn,
                trainloader,
                device,
                num_batches=args.gamma_calibration_batches,
            )
            if not math.isfinite(avg_delta) or avg_delta <= 0:
                print("[Calibration] Average delta was non-finite; penalty_lambda unchanged.")
            else:
                ratio = float(avg_delta / args.latent_eps_target)
                ratio = float(
                    min(max(ratio, CALIBRATION_SCALE_MIN), CALIBRATION_SCALE_MAX)
                )
                smoothing = 1.0 + CALIBRATION_SMOOTHING * (ratio - 1.0)
                new_lambda = _clamp_penalty_lambda(args.penalty_lambda * smoothing)
                print(
                    f"[Calibration] Average delta {avg_delta:.4f}, ratio {ratio:.4f}, "
                    f"smoothing {smoothing:.4f}, penalty_lambda {args.penalty_lambda:.6f} → {new_lambda:.6f}"
                )
                args.penalty_lambda = new_lambda
        tracker.finish_epoch()

    save_transport_delta_plot(tracker, args, run_id)

    if args.estimate_transport_jacobian:
        jac_loader = trainloader if args.jacobian_sv_split == "train" else testloader
        max_sv, mean_sv = estimate_transport_jacobian_sv(
            phi, icnn, jac_loader, device, args
        )
        if math.isnan(max_sv) or math.isnan(mean_sv):
            print("Transport Jacobian estimation returned NaN; consider adjusting parameters.")
        else:
            print(
                f"Estimated transport Jacobian σ_max ≈ {max_sv:.4f}, σ_mean ≈ {mean_sv:.4f}"
            )

    if args.visualize_transport:
        viz_loader = trainloader if args.transport_viz_split == "train" else testloader
        visualize_transport_map(
            phi,
            icnn,
            viz_loader,
            device,
            args,
            run_id=run_id,
            split_name=args.transport_viz_split,
        )

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "phi": phi.state_dict(),
                "head": head.state_dict(),
                "icnn": icnn.state_dict(),
                "args": vars(args),
            },
            args.save,
        )
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
