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
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ademamix import AdEMAMix
from torch.nn.utils import parameters_to_vector
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
DEFAULT_CALIBRATION_RATIO_MIN = 1.0
DEFAULT_CALIBRATION_RATIO_MAX = 6.0
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
        ax_iter.set_ylabel("Mean per-feature |T(z) - z|^2")
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
        ax_epoch.set_ylabel("Mean per-feature |T(z) - z|^2")
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


def per_sample_mean_square_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return per-sample squared difference sum between tensors a and b."""
    if a.shape != b.shape:
        raise ValueError("Tensors must have identical shapes to compute squared difference.")
    diff = (a - b).reshape(a.size(0), -1)
    return diff.pow(2).sum(dim=1)


def _extract_penalty_features(
    z_src: torch.Tensor,
    z_adv: torch.Tensor,
    head: nn.Module,
    feature_type: str,
    logits_adv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if feature_type == "latent":
        feat_src = z_src.view(z_src.size(0), -1)
        feat_adv = z_adv.view(z_adv.size(0), -1)
        return feat_src, feat_adv, logits_adv
    if feature_type == "head_logits":
        with torch.no_grad():
            feat_src = head(z_src)
        if logits_adv is None:
            logits_adv = head(z_adv)
        feat_adv = logits_adv
        return feat_src, feat_adv, logits_adv
    raise ValueError(f"Unsupported cosine feature extractor: {feature_type}")


def _cosine_distance(feat_src: torch.Tensor, feat_adv: torch.Tensor, eps: float) -> torch.Tensor:
    src_flat = feat_src.reshape(feat_src.size(0), -1)
    adv_flat = feat_adv.reshape(feat_adv.size(0), -1)
    denom = (
        src_flat.norm(dim=1).clamp_min(eps)
        * adv_flat.norm(dim=1).clamp_min(eps)
    )
    cos_sim = (src_flat * adv_flat).sum(dim=1) / denom
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    return 1.0 - cos_sim


def compute_transport_penalty(
    z_src: torch.Tensor,
    z_adv: torch.Tensor,
    head: nn.Module,
    penalty_lambda: float,
    mse_per_sample: torch.Tensor,
    cosine_cfg: Optional[Dict[str, Any]],
    logits_adv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    quad_scale = float(cosine_cfg.get("quadratic_scale", 1.0)) if cosine_cfg else 1.0
    mse_mean = mse_per_sample.mean()
    mse_term = quad_scale * mse_mean
    if cosine_cfg and cosine_cfg.get("enabled", False):
        feature_type = str(cosine_cfg.get("feature", "latent"))
        eps = float(cosine_cfg.get("eps", 1e-8))
        feat_src, feat_adv, logits_adv = _extract_penalty_features(
            z_src, z_adv, head, feature_type, logits_adv=logits_adv
        )
        cos_dist = _cosine_distance(feat_src, feat_adv, eps)
        cos_mean = cos_dist.mean()
        cos_weight = penalty_lambda * float(cosine_cfg.get("lambda", 1.0))
        quad_weight = penalty_lambda * float(cosine_cfg.get("quadratic_weight", 0.0))
        penalty = cos_weight * cos_mean + quad_weight * mse_term
        monitor = mse_mean
        return penalty, monitor, logits_adv

    penalty = penalty_lambda * mse_term
    monitor = mse_mean
    return penalty, monitor, logits_adv


def _icnn_principled_moments(fan_in: int) -> Tuple[float, float, float, float, float]:
    if fan_in <= 0:
        raise ValueError(f"ICNN fan-in must be positive; got {fan_in}.")
    denom_offset = 6.0 * (math.pi - 1.0)
    denom_slope = 3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0
    denom = denom_offset + (fan_in - 1.0) * denom_slope
    mu_w = math.sqrt((6.0 * math.pi) / (fan_in * denom))
    sigma_w2 = 1.0 / float(fan_in)
    mu_b = math.sqrt((3.0 * fan_in) / denom)
    mu_w_sq = mu_w * mu_w
    log_var_plus_mean_sq = math.log(sigma_w2 + mu_w_sq)
    log_mean_sq = math.log(mu_w_sq)
    tilde_mu = log_mean_sq - 0.5 * log_var_plus_mean_sq
    tilde_sigma2 = max(log_var_plus_mean_sq - log_mean_sq, 1e-12)
    tilde_sigma = math.sqrt(tilde_sigma2)
    return mu_w, sigma_w2, mu_b, tilde_mu, tilde_sigma


def _principled_nonnegative_init(
    weight_param: torch.Tensor,
    bias: Optional[torch.Tensor],
    fan_in: int,
) -> None:
    _, _, mu_b, tilde_mu, tilde_sigma = _icnn_principled_moments(fan_in)
    with torch.no_grad():
        mu_tensor = torch.as_tensor(tilde_mu, dtype=weight_param.dtype, device=weight_param.device)
        if tilde_sigma == 0.0:
            weight_param.fill_(mu_tensor)
        else:
            sigma_tensor = torch.as_tensor(tilde_sigma, dtype=weight_param.dtype, device=weight_param.device)
            noise = torch.randn_like(weight_param)
            weight_param.copy_(noise * sigma_tensor + mu_tensor)
        if bias is not None:
            bias.fill_(torch.as_tensor(mu_b, dtype=bias.dtype, device=bias.device))


class NonNegativeLinear(nn.Module):
    """Linear map with weights constrained to be element-wise non-negative."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "principled",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.init_mode = init.lower()
        if self.init_mode not in {"principled", "xavier"}:
            raise ValueError(f"Unsupported initialisation mode '{init}' for NonNegativeLinear.")
        self.parametrisation = "exp" if self.init_mode == "principled" else "softplus"
        self.weight_param = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_mode == "principled":
            _principled_nonnegative_init(self.weight_param, self.bias, self.in_features)
        else:
            nn.init.xavier_uniform_(self.weight_param)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parametrisation == "exp":
            weight = torch.exp(self.weight_param)
        else:
            weight = F.softplus(self.weight_param)
        return F.linear(x, weight, self.bias)

    @torch.no_grad()
    def project_non_negative(self) -> None:
        return


class NonNegativeConv2d(nn.Module):
    """Conv2d module with weights constrained to be element-wise non-negative."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int]] = 0,
        bias: bool = True,
        init: str = "principled",
    ):
        super().__init__()
        self.init_mode = init.lower()
        if self.init_mode not in {"principled", "xavier"}:
            raise ValueError(f"Unsupported initialisation mode '{init}' for NonNegativeConv2d.")
        if isinstance(kernel_size, int):
            kernel_tuple = (int(kernel_size), int(kernel_size))
        elif isinstance(kernel_size, (tuple, list)):
            kernel_tuple = tuple(int(k) for k in kernel_size)
            if len(kernel_tuple) != 2:
                raise ValueError("kernel_size must have 2 elements for NonNegativeConv2d.")
        else:
            raise TypeError("kernel_size must be an int or a length-2 sequence of ints.")
        self.kernel_size = kernel_tuple
        if isinstance(padding, int):
            self.padding = int(padding)
        elif isinstance(padding, (tuple, list)):
            padding_tuple = tuple(int(p) for p in padding)
            if len(padding_tuple) != 2:
                raise ValueError("padding must have 2 elements when provided as a sequence.")
            self.padding = padding_tuple
        else:
            raise TypeError("padding must be an int or a length-2 sequence of ints.")
        kernel_area = self.kernel_size[0] * self.kernel_size[1]
        weight_shape = (out_channels, in_channels, *self.kernel_size)
        self.weight_param = nn.Parameter(torch.empty(weight_shape))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.parametrisation = "exp" if self.init_mode == "principled" else "softplus"
        self._fan_in = int(in_channels * kernel_area)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_mode == "principled":
            _principled_nonnegative_init(self.weight_param, self.bias, self._fan_in)
        else:
            nn.init.xavier_uniform_(self.weight_param)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parametrisation == "exp":
            weight = torch.exp(self.weight_param)
        else:
            weight = F.softplus(self.weight_param)
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
        nonneg_init: str = "principled",
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
        self.nonneg_init = str(nonneg_init).lower()
        if self.nonneg_init not in {"principled", "xavier"}:
            raise ValueError(f"Unsupported ICNN non-negative initialiser: {nonneg_init}")

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
                            init=self.nonneg_init,
                        )
                    )
                prev_hidden = width
            self.hidden_output = NonNegativeConv2d(
                self.hidden_sizes[-1], 1, kernel_size=1, padding=0, bias=True, init=self.nonneg_init
            )
            self.input_skip = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0, bias=True)
        else:
            for width in self.hidden_sizes:
                self.z_linears.append(nn.Linear(self.input_dim, width, bias=True))
                if prev_hidden is None:
                    self.h_linears.append(None)  # type: ignore
                else:
                    self.h_linears.append(
                        NonNegativeLinear(prev_hidden, width, bias=True, init=self.nonneg_init)
                    )
                prev_hidden = width
            self.hidden_output = NonNegativeLinear(
                self.hidden_sizes[-1], 1, bias=True, init=self.nonneg_init
            )
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
                layer.reset_parameters()
        self.hidden_output.reset_parameters()
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
    rng: Optional[torch.Generator] = None,
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

    total_batches_hint = max_batches
    if hasattr(loader, "__len__"):
        total_batches_hint = min(len(loader), max_batches)
    progress_bar = tqdm(
        loader,
        desc="Jacobian SV Eval",
        leave=False,
        total=total_batches_hint,
        dynamic_ncols=True,
    )
    for x, _ in progress_bar:
        x = x.to(device)
        with torch.no_grad():
            z = phi(x)
        batch = min(z.size(0), max_samples)
        if batch == 0:
            continue
        if rng is not None:
            indices = torch.randperm(z.size(0), generator=rng, device=z.device)[:batch]
        else:
            indices = torch.randperm(z.size(0), device=z.device)[:batch]
        for idx in indices:
            base = z[idx.item() : idx.item() + 1].detach()
            if rng is not None:
                vec = torch.randn(base.shape, generator=rng, device=base.device, dtype=base.dtype)
            else:
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
        if sv_values:
            current_max = max(sv_values)
            current_mean = sum(sv_values) / len(sv_values)
            progress_bar.set_postfix(
                max=f"{current_max:.4f}", mean=f"{current_mean:.4f}", refresh=False
            )
        if batches_considered >= max_batches:
            break
    progress_bar.close()
    phi.train(phi_mode)
    icnn.train(icnn_mode)

    if not sv_values:
        return float("nan"), float("nan")
    sv_tensor = torch.tensor(sv_values)
    return float(sv_tensor.max().item()), float(sv_tensor.mean().item())


def _jacobian_sv_for_sample(
    icnn: InputConvexPotential,
    base: torch.Tensor,
    power_iters: int,
    rng: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    """Return spectral norm estimate for a single latent sample (differentiable)."""
    if base.ndim == 0:
        return None
    sample = base.detach().unsqueeze(0)
    if rng is not None:
        vec = torch.empty_like(sample).normal_(mean=0.0, std=1.0, generator=rng)
    else:
        vec = torch.randn_like(sample)
    vec_flat = vec.view(vec.size(0), -1)
    vec_norm = vec_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    vec = vec / vec_norm.view(vec.size(0), *([1] * (vec.ndim - 1)))
    sigma: Optional[torch.Tensor] = None
    for _ in range(max(1, int(power_iters))):
        z_leaf = sample.clone().detach().requires_grad_(True)
        transport = icnn.gradient(z_leaf, create_graph=True)
        hvp = torch.autograd.grad(
            transport,
            z_leaf,
            grad_outputs=vec,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        if hvp is None:
            return None
        hvp_flat = hvp.view(hvp.size(0), -1)
        sigma = hvp_flat.norm(dim=1)
        if not torch.isfinite(sigma).all():
            return None
        sigma_clamped = sigma.detach().clamp_min(1e-12)
        reshaped = sigma_clamped.view(hvp.size(0), *([1] * (hvp.ndim - 1)))
        vec = hvp.detach() / reshaped
    if sigma is None:
        return None
    return sigma.mean()


def estimate_local_jacobian_sv(
    icnn: InputConvexPotential,
    z_batch: torch.Tensor,
    num_samples: int,
    power_iters: int,
    rng: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    """Estimate average largest singular value over a minibatch (differentiable)."""
    if z_batch.ndim == 0 or z_batch.size(0) == 0:
        return None
    batch = min(z_batch.size(0), max(1, int(num_samples)))
    if rng is not None:
        indices = torch.randperm(z_batch.size(0), generator=rng, device=z_batch.device)[:batch]
    else:
        indices = torch.randperm(z_batch.size(0), device=z_batch.device)[:batch]
    estimates: List[torch.Tensor] = []
    indices_list = indices.tolist()
    use_bar = len(indices_list) > 1
    bar = None
    iterator = indices_list
    if use_bar:
        bar = tqdm(
            indices_list,
            desc="Jacobian SV Samples",
            leave=False,
            dynamic_ncols=True,
        )
        iterator = bar
    for idx in iterator:
        sample = z_batch[idx]
        sigma = _jacobian_sv_for_sample(icnn, sample, power_iters, rng=rng)
        if sigma is None:
            continue
        if not torch.isfinite(sigma):
            continue
        estimates.append(sigma)
        if bar is not None and torch.isfinite(sigma).all():
            bar.set_postfix(sigma=float(sigma.mean().item()), refresh=False)
    if bar is not None:
        bar.close()
    if not estimates:
        return None
    return torch.stack(estimates).mean()


class BBArmijoState:
    def __init__(
        self,
        alpha0: float,
        alpha_min: float,
        alpha_max: float,
        ls_c: float,
        ls_shrink: float,
        ls_max_steps: int,
    ) -> None:
        self.alpha_min = float(max(alpha_min, 1e-12))
        self.alpha_max = float(max(alpha_max, self.alpha_min))
        self.alpha_prev = float(min(max(alpha0, self.alpha_min), self.alpha_max))
        self.ls_c = float(ls_c)
        self.ls_shrink = float(ls_shrink)
        self.ls_max_steps = int(max(ls_max_steps, 1))
        self.prev_params_vec: Optional[torch.Tensor] = None
        self.prev_grad_vec: Optional[torch.Tensor] = None

    def propose(self, params_vec: torch.Tensor, grad_vec: torch.Tensor) -> float:
        if (
            self.prev_params_vec is None
            or self.prev_grad_vec is None
            or self.prev_params_vec.numel() != params_vec.numel()
            or self.prev_grad_vec.numel() != grad_vec.numel()
        ):
            alpha = self.alpha_prev
        else:
            s = params_vec - self.prev_params_vec
            y = grad_vec - self.prev_grad_vec
            denom = torch.dot(s, y)
            if torch.isfinite(denom) and float(denom.abs().item()) > 1e-12:
                num = torch.dot(s, s)
                alpha = float((num / denom).item())
            else:
                alpha = self.alpha_prev
        if not math.isfinite(alpha):
            alpha = self.alpha_prev
        alpha = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return alpha

    def update_history(self, params_vec: torch.Tensor, grad_vec: torch.Tensor, alpha: float) -> None:
        self.prev_params_vec = params_vec.detach().clone()
        self.prev_grad_vec = grad_vec.detach().clone()
        alpha_clamped = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        self.alpha_prev = alpha_clamped


def _evaluate_icnn_adv_objective(
    icnn: InputConvexPotential,
    head: nn.Module,
    z_source: torch.Tensor,
    y: torch.Tensor,
    penalty_lambda: float,
    cosine_cfg: Optional[Dict[str, Any]],
    use_margin_adv: bool,
    jacobian_reg_weight: float,
    jacobian_reg_samples: int,
    jacobian_reg_iters: int,
    rng: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        z_adv_eval, _ = adversarial_pushforward(icnn, z_source, detach_for_model=False)
        logits_eval = head(z_adv_eval)
        if use_margin_adv:
            logits_correct = logits_eval.gather(1, y.unsqueeze(1)).squeeze(1)
            margins = logits_eval - logits_correct.unsqueeze(1)
            num_classes = logits_eval.size(1)
            correct_mask = F.one_hot(y, num_classes=num_classes).bool()
            margins = margins.masked_fill(correct_mask, float("-inf"))
            adv_primary = torch.logsumexp(margins, dim=1).mean()
        else:
            adv_primary = F.cross_entropy(logits_eval, y, reduction="mean")
        per_sample_mse = per_sample_mean_square_diff(z_adv_eval, z_source)
        penalty_term, _, _ = compute_transport_penalty(
            z_source,
            z_adv_eval,
            head,
            penalty_lambda,
            per_sample_mse,
            cosine_cfg,
            logits_adv=logits_eval,
        )
        sv_penalty = torch.zeros_like(adv_primary)
        if jacobian_reg_weight > 0.0:
            sv_estimate = estimate_local_jacobian_sv(
                icnn,
                z_source,
                num_samples=jacobian_reg_samples,
                power_iters=jacobian_reg_iters,
                rng=rng,
            )
            if sv_estimate is not None and torch.isfinite(sv_estimate):
                sv_penalty = jacobian_reg_weight * sv_estimate.pow(2)
        adv_objective = adv_primary - penalty_term - sv_penalty
    return adv_objective.detach(), per_sample_mse.detach()


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
    cosine_cfg: Optional[Dict[str, Any]] = None,
    jacobian_reg_weight: float = 0.0,
    jacobian_reg_samples: int = 1,
    jacobian_reg_iters: int = 1,
    rng: Optional[torch.Generator] = None,
    use_margin_adv: bool = True,
    icnn_step_rule: str = "constant",
    icnn_bb_config: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, float, float, float]:
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
    total_jacobian_penalty = 0.0
    total_sv_sq = 0.0
    jacobian_counts = 0
    cosine_cfg = cosine_cfg or {"enabled": False}

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
        penalty_monitor = torch.zeros((), device=z_detached.device, dtype=z_detached.dtype)
        sv_estimate_last: Optional[torch.Tensor] = None
        ascent_steps = max(1, icnn_ascent_steps)
        bb_state: Optional[BBArmijoState] = None
        if icnn_step_rule == "bb-armijo" and icnn_bb_config is not None:
            bb_state = BBArmijoState(
                icnn_bb_config["alpha0"],
                icnn_bb_config["alpha_min"],
                icnn_bb_config["alpha_max"],
                icnn_bb_config["ls_c"],
                icnn_bb_config["ls_shrink"],
                icnn_bb_config["ls_max_steps"],
            )
        ascent_bar = tqdm(
            range(ascent_steps),
            desc="ICNN Ascent",
            leave=False,
            dynamic_ncols=True,
        )
        try:
            for ascent_step in ascent_bar:
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
                if use_margin_adv:
                    logits_correct = logits_adv.gather(1, y.unsqueeze(1)).squeeze(1)
                    margins = logits_adv - logits_correct.unsqueeze(1)
                    num_classes = logits_adv.size(1)
                    correct_mask = F.one_hot(y, num_classes=num_classes).bool()
                    margins = margins.masked_fill(correct_mask, float("-inf"))
                    adv_primary = torch.logsumexp(margins, dim=1).mean()
                    metric_key = "margin"
                else:
                    adv_primary = F.cross_entropy(logits_adv, y, reduction="mean")
                    metric_key = "ce"
                per_sample_mse = per_sample_mean_square_diff(z_adv_ascent, z_detached)
                penalty_term, penalty_monitor, logits_adv = compute_transport_penalty(
                    z_detached,
                    z_adv_ascent,
                    head,
                    penalty_lambda,
                    per_sample_mse,
                    cosine_cfg,
                    logits_adv=logits_adv,
                )
                sv_estimate: Optional[torch.Tensor] = None
                sv_penalty = torch.zeros((), device=adv_primary.device, dtype=adv_primary.dtype)
                if jacobian_reg_weight > 0.0:
                    sv_estimate = estimate_local_jacobian_sv(
                        icnn,
                        z_detached,
                        num_samples=jacobian_reg_samples,
                        power_iters=jacobian_reg_iters,
                        rng=rng,
                    )
                    if sv_estimate is not None and torch.isfinite(sv_estimate):
                        sv_penalty = jacobian_reg_weight * sv_estimate.pow(2)
                    else:
                        sv_estimate = None
                if (
                    not torch.isfinite(adv_primary)
                    or not torch.isfinite(penalty_term)
                    or not torch.isfinite(penalty_monitor)
                    or not torch.isfinite(sv_penalty)
                ):
                    warnings.warn(
                        "Non-finite adversarial loss components detected; skipping ascent step.",
                        RuntimeWarning,
                    )
                    continue
                adv_objective = adv_primary - penalty_term - sv_penalty
                if sv_estimate is not None:
                    sv_estimate_last = sv_estimate.detach()
                if torch.isfinite(adv_primary) and torch.isfinite(penalty_term):
                    bar_postfix: Dict[str, str] = {
                        metric_key: f"{float(adv_primary.item()):.4f}",
                        "pen": f"{float(penalty_term.item()):.4f}",
                    }
                    if sv_estimate is not None and torch.isfinite(sv_estimate):
                        bar_postfix["jac_sv"] = f"{float(sv_estimate.item()):.4f}"
                    ascent_bar.set_postfix(bar_postfix, refresh=False)
                (-adv_objective).backward()
                if tracker is not None:
                    tracker.record_ascent(ascent_step, per_sample_mse.detach(), batch_idx)
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
                torch.nn.utils.clip_grad_norm_(icnn.parameters(), max_norm=1.0)
                if icnn_step_rule == "bb-armijo" and bb_state is not None:
                    params_tensors: List[torch.Tensor] = []
                    grad_tensors: List[torch.Tensor] = []
                    for group in opt_icnn.param_groups:
                        weight_decay = float(group.get("weight_decay", 0.0))
                        for param in group["params"]:
                            params_tensors.append(param.detach().clone())
                            grad_tensor = torch.zeros_like(param)
                            if param.grad is not None:
                                grad_tensor = param.grad.detach().clone()
                            if weight_decay != 0.0:
                                grad_tensor = grad_tensor + weight_decay * param.detach()
                            grad_tensors.append(grad_tensor)
                    if not grad_tensors:
                        adv_objective_last = adv_objective.detach()
                        continue
                    grad_vec = parameters_to_vector(grad_tensors)
                    params_vec = parameters_to_vector(params_tensors)
                    grad_norm_sq = float(grad_vec.pow(2).sum().item())
                    adv_obj_current = float(adv_objective.detach().item())
                    if (not math.isfinite(grad_norm_sq)) or grad_norm_sq <= 0.0:
                        bb_state.update_history(params_vec, grad_vec, bb_state.alpha_prev)
                        adv_objective_last = adv_objective.detach()
                        continue

                    alpha_candidate = bb_state.propose(params_vec, grad_vec)
                    params_backup = [param.detach().clone() for param in icnn.parameters()]
                    accepted = False
                    adv_objective_candidate: Optional[torch.Tensor] = None

                    for _ in range(bb_state.ls_max_steps):
                        for group in opt_icnn.param_groups:
                            group["lr"] = alpha_candidate
                        opt_icnn.step()
                        icnn.project_convexity()
                        adv_objective_candidate, _ = _evaluate_icnn_adv_objective(
                            icnn,
                            head,
                            z_detached,
                            y,
                            penalty_lambda,
                            cosine_cfg,
                            use_margin_adv,
                            jacobian_reg_weight,
                            jacobian_reg_samples,
                            jacobian_reg_iters,
                            rng,
                        )
                        if torch.isfinite(adv_objective_candidate):
                            lhs = float(adv_objective_candidate.item())
                            rhs = adv_obj_current + bb_state.ls_c * alpha_candidate * grad_norm_sq
                            if lhs >= rhs:
                                accepted = True
                                break
                        with torch.no_grad():
                            for param, saved in zip(icnn.parameters(), params_backup):
                                param.copy_(saved)
                        alpha_candidate *= bb_state.ls_shrink
                        if alpha_candidate < bb_state.alpha_min:
                            break

                    if not accepted:
                        alpha_candidate = max(bb_state.alpha_min, min(bb_state.alpha_max, alpha_candidate))
                        for group in opt_icnn.param_groups:
                            group["lr"] = alpha_candidate
                        opt_icnn.step()
                        icnn.project_convexity()
                        adv_objective_candidate, _ = _evaluate_icnn_adv_objective(
                            icnn,
                            head,
                            z_detached,
                            y,
                            penalty_lambda,
                            cosine_cfg,
                            use_margin_adv,
                            jacobian_reg_weight,
                            jacobian_reg_samples,
                            jacobian_reg_iters,
                            rng,
                        )
                    for param in icnn.parameters():
                        if not torch.isfinite(param).all():
                            warnings.warn(
                                "Detected non-finite ICNN parameters after update; sanitizing values.",
                                RuntimeWarning,
                            )
                            param.data.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)
                    if adv_objective_candidate is not None and torch.isfinite(adv_objective_candidate):
                        adv_objective_last = adv_objective_candidate
                    else:
                        adv_objective_last = adv_objective.detach()
                    bb_state.update_history(params_vec, grad_vec, alpha_candidate)
                else:
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
        finally:
            ascent_bar.close()

        # Clear any stray gradients created during adversary ascent before the outer update.
        for param in model_params:
            if param.grad is not None:
                param.grad = None

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
            delta_sq = per_sample_mean_square_diff(z_adv_fixed, z)
            if tracker is not None:
                tracker.record_epoch_batch(delta_sq, batch_idx)

        with torch.no_grad():
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += batch_size
            total_penalty += float(penalty_monitor.detach().item())
            if adv_objective_last is not None and torch.isfinite(adv_objective_last).all():
                total_adv_obj += adv_objective_last.item()
            if sv_estimate_last is not None:
                sv_sq = float(sv_estimate_last.pow(2).item())
                total_sv_sq += sv_sq
                jacobian_counts += 1
                total_jacobian_penalty += float(jacobian_reg_weight * sv_sq)

            if total_samples > 0:
                mean_loss = total_loss / total_samples
                mean_acc = total_correct / total_samples
                mean_penalty = total_penalty / max(1, len(loader))
                mean_adv_obj = total_adv_obj / max(1, len(loader))
                postfix: Dict[str, str] = {
                    "loss": f"{mean_loss:.4f}",
                    "acc": f"{mean_acc*100:.2f}%",
                    "penalty": f"{mean_penalty:.4f}",
                    "adv_obj": f"{mean_adv_obj:.4f}",
                }
                if jacobian_reg_weight > 0.0 and jacobian_counts > 0:
                    mean_sv_sq = total_sv_sq / jacobian_counts
                    mean_sv = math.sqrt(max(mean_sv_sq, 0.0))
                    mean_jac_pen = total_jacobian_penalty / jacobian_counts
                    postfix["jac_sv"] = f"{mean_sv:.4f}"
                    postfix["jac_pen"] = f"{mean_jac_pen:.4f}"
                progress.set_postfix(refresh=True, **postfix)

    progress.close()
    mean_loss = total_loss / max(1, total_samples)
    mean_acc = total_correct / max(1, total_samples)
    mean_penalty = total_penalty / max(1, len(loader))
    mean_adv_obj = total_adv_obj / max(1, len(loader))
    if jacobian_counts > 0:
        mean_sv_sq = total_sv_sq / jacobian_counts
        mean_sv = math.sqrt(max(mean_sv_sq, 0.0))
        mean_jacobian_penalty = total_jacobian_penalty / jacobian_counts
    else:
        mean_sv = 0.0
        mean_jacobian_penalty = 0.0
    return (
        mean_loss,
        mean_acc,
        mean_penalty,
        mean_adv_obj,
        mean_jacobian_penalty,
        mean_sv,
        jacobian_counts,
    )


def evaluate_under_icnn(
    phi: nn.Module,
    head: nn.Module,
    icnn: InputConvexPotential,
    loader: DataLoader,
    device: torch.device,
    penalty_lambda: float,
    cosine_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, float]:
    phi.eval()
    head.eval()
    icnn.eval()
    cosine_cfg = cosine_cfg or {"enabled": False}

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
        per_sample_mse = per_sample_mean_square_diff(z_adv, z_det)
        _, penalty_monitor, _ = compute_transport_penalty(
            z_det,
            z_adv,
            head,
            penalty_lambda,
            per_sample_mse,
            cosine_cfg,
            logits_adv=logits,
        )
        total_penalty += float(penalty_monitor.detach().item())
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
    "train_jac_penalty",
    "train_jac_sv",
    "adv_objective",
    "test_loss",
    "test_acc",
    "icnn_loss",
    "icnn_acc",
    "icnn_penalty",
    "input_pgd_acc",
    "input_pgd_avg_l2",
    "input_pgd_avg_linf",
    "input_pgd_samples",
    "penalty_lambda",
    "jacobian_reg_weight",
    "jacobian_reg_samples",
    "jacobian_reg_iters",
    "jacobian_reg_counts",
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
    parser.add_argument("--batch-size", type=int, default=256)
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
    parser.add_argument(
        "--use-margin-loss",
        action="store_true",
        help="Use the log-sum-exp margin objective for the ICNN adversary (non-zero-sum view).",
    )
    parser.add_argument("--icnn-hidden", type=_parse_hidden_units, nargs="+", default=[512, 256])
    parser.add_argument("--icnn-activation", type=str, choices=["relu", "softplus"], default="softplus")
    parser.add_argument("--icnn-strong-convexity", type=float, default=1.0)
    parser.add_argument(
        "--icnn-init",
        type=str,
        choices=["principled", "xavier"],
        default="principled",
        help="Initialisation scheme for non-negative ICNN weights.",
    )
    parser.add_argument(
        "--lr-omega",
        type=float,
        default=0.0005,
        help="Step size γ_ω for ICNN adversary parameters.",
    )
    parser.add_argument(
        "--icnn-step-rule",
        type=str,
        choices=["constant", "bb-armijo"],
        default="constant",
        help="Adaptive step-size rule for ICNN ascent (constant keeps optimizer LR; bb-armijo uses BB + Armijo).",
    )
    parser.add_argument(
        "--icnn-alpha0",
        type=float,
        default=0.0005,
        help="Initial step size guess for ICNN BB/Armijo rule.",
    )
    parser.add_argument(
        "--icnn-ls-c",
        type=float,
        default=0.1,
        help="Armijo sufficient increase constant for ICNN BB rule.",
    )
    parser.add_argument(
        "--icnn-ls-shrink",
        type=float,
        default=0.5,
        help="Backtracking shrink factor for ICNN Armijo search.",
    )
    parser.add_argument(
        "--icnn-ls-max-steps",
        type=int,
        default=10,
        help="Maximum Armijo backtracking iterations for ICNN BB rule.",
    )
    parser.add_argument(
        "--icnn-alpha-min",
        type=float,
        default=1e-6,
        help="Minimum clamp on ICNN BB step size.",
    )
    parser.add_argument(
        "--icnn-alpha-max",
        type=float,
        default=1.0,
        help="Maximum clamp on ICNN BB step size.",
    )
    parser.add_argument(
        "--icnn-optimizer",
        type=str,
        choices=["adam", "ademamix", "sgd"],
        default="ademamix",
        help="Optimizer used for ICNN weights (adam, ademamix, or sgd).",
    )
    parser.add_argument("--icnn-beta1", type=float, default=0.9)
    parser.add_argument("--icnn-beta2", type=float, default=0.999)
    parser.add_argument(
        "--ademamix-beta3",
        type=float,
        default=0.9999,
        help="Third beta coefficient for AdEMAMix (ignored for Adam).",
    )
    parser.add_argument(
        "--ademamix-alpha",
        type=float,
        default=4.0,
        help="Mixing factor α for AdEMAMix (ignored for Adam).",
    )
    parser.add_argument(
        "--ademamix-beta3-warmup",
        type=int,
        default=5,
        help="Number of warmup steps for β3 in AdEMAMix (ignored for Adam).",
    )
    parser.add_argument(
        "--ademamix-alpha-warmup",
        type=int,
        default=3,
        help="Number of warmup steps for α in AdEMAMix (ignored for Adam).",
    )
    parser.add_argument("--icnn-ascent-steps", type=int, default=5)
    parser.add_argument(
        "--cosine-penalty",
        action="store_true",
        help="Enable cosine-similarity transport penalty instead of the default quadratic penalty.",
    )
    parser.add_argument(
        "--cosine-feature",
        type=str,
        choices=["latent", "head_logits"],
        default="latent",
        help="Feature space used for cosine penalty (latent vectors or classifier logits).",
    )
    parser.add_argument(
        "--cosine-lambda",
        type=float,
        default=1.0,
        help="Scale applied to the cosine distance component of the penalty.",
    )
    parser.add_argument(
        "--cosine-quadratic-weight",
        type=float,
        default=1.0,
        help="Additional quadratic weight α ensuring strong convexity in the cosine penalty.",
    )
    parser.add_argument(
        "--cosine-eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon to stabilise cosine similarity denominator.",
    )

    parser.add_argument("--cut-layer", type=str, default="layer4",
                        choices=["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"])
    parser.add_argument("--log-csv", type=str, default="./runs_log_icnn.csv")
    parser.add_argument("--save", type=str, default="")

    parser.add_argument("--jacobian-aware", action="store_true")
    parser.add_argument("--jacobian-batches", type=int, default=4)
    parser.add_argument("--jacobian-iters", type=int, default=10)
    parser.add_argument(
        "--jacobian-reg-weight",
        type=float,
        default=0.0,
        help="Weight for Jacobian spectral norm regularization on the ICNN adversary.",
    )
    parser.add_argument(
        "--jacobian-reg-samples",
        type=int,
        default=1,
        help="Number of latent samples per batch used for the Jacobian spectral penalty.",
    )
    parser.add_argument(
        "--jacobian-reg-iters",
        type=int,
        default=1,
        help="Number of power iterations used to estimate the Jacobian spectral penalty.",
    )
    parser.add_argument(
        "--jacobian-log-dir",
        type=str,
        default="results/jacobian_logs",
        help="Directory used to store Jacobian regularization summaries for each run.",
    )

    parser.add_argument("--inp-p", type=str, default="inf", choices=["2", "inf"])
    parser.add_argument("--inp-eps", type=float, default=8 / 255)
    parser.add_argument("--inp-steps", type=int, default=20)
    parser.add_argument("--inp-step-size", type=float, default=0.0)
    parser.add_argument("--inp-restarts", type=int, default=5)
    parser.add_argument(
        "--eval-input-pgd",
        dest="eval_input_pgd",
        action="store_true",
        help="Evaluate robustness with input-space PGD each epoch (uses subset if samples specified).",
    )
    parser.add_argument(
        "--no-eval-input-pgd",
        dest="eval_input_pgd",
        action="store_false",
        help="Disable input-space PGD evaluation at epoch end.",
    )
    parser.add_argument(
        "--eval-input-pgd-samples",
        type=int,
        default=1000,
        help="Number of test samples to use for input-space PGD evaluation (<=0 uses full set).",
    )

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
        default=10,
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
        default=2.0,
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
    parser.add_argument(
        "--calibration-ratio-min",
        type=float,
        default=DEFAULT_CALIBRATION_RATIO_MIN,
        help="Minimum clamp applied to avg_delta / latent_eps_target during penalty calibration.",
    )
    parser.add_argument(
        "--calibration-ratio-max",
        type=float,
        default=DEFAULT_CALIBRATION_RATIO_MAX,
        help="Maximum clamp applied to avg_delta / latent_eps_target during penalty calibration.",
    )

    parser.set_defaults(eval_input_pgd=True)
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
    ratio_min = max(float(args.calibration_ratio_min), 1e-6)
    ratio_max = max(float(args.calibration_ratio_max), ratio_min)
    if ratio_min != args.calibration_ratio_min or ratio_max != args.calibration_ratio_max:
        warnings.warn(
            "Adjusted calibration ratio bounds to ensure 0 < min ≤ max.",
            RuntimeWarning,
        )
    args.calibration_ratio_min = ratio_min
    args.calibration_ratio_max = ratio_max

    set_deterministic(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Global seed: {args.seed}")
    if device.type == "cuda":
        global_rng = torch.Generator(device=device)
    else:
        global_rng = torch.Generator()
    global_rng.manual_seed(int(args.seed))

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

    was_training = phi.training
    phi.eval()
    with torch.no_grad():
        example_batch = next(iter(trainloader))[0][:1].to(device)
        latent_example = phi(example_batch)
        latent_dim = latent_example[0].numel()
        latent_shape = latent_example.shape[1:]
    phi.train(was_training)
    print(f"Latent shape at cut-layer '{args.cut_layer}': {latent_shape} (dim={latent_dim})")

    cosine_cfg = {
        "enabled": bool(args.cosine_penalty),
        "feature": args.cosine_feature,
        "lambda": float(args.cosine_lambda),
        "quadratic_weight": float(args.cosine_quadratic_weight),
        "eps": float(args.cosine_eps),
    }
    if cosine_cfg["enabled"]:
        print(
            f"Cosine penalty enabled: feature={cosine_cfg['feature']}, "
            f"λ={cosine_cfg['lambda']}, α={cosine_cfg['quadratic_weight']}"
        )
    if args.calibrate_penalty:
        print(
            "Penalty calibration enabled: "
            f"ε_target={args.latent_eps_target}, "
            f"ratio clamp=[{args.calibration_ratio_min:.3f}, {args.calibration_ratio_max:.3f}]"
        )
    if args.jacobian_reg_weight > 0.0:
        print(
            "Jacobian spectral regularizer enabled: "
            f"weight={args.jacobian_reg_weight}, samples={args.jacobian_reg_samples}, "
            f"iters={args.jacobian_reg_iters}"
        )

    if args.head_only:
        for p in phi.parameters():
            p.requires_grad = False

    icnn = InputConvexPotential(
        input_dim=latent_dim,
        hidden_sizes=args.icnn_hidden,
        activation=args.icnn_activation,
        strong_convexity=args.icnn_strong_convexity,
        nonneg_init=args.icnn_init,
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
        decay = 0.0 if ("weight_param" in name or "bias" in name) else 1e-4
        icnn_param_groups.append({"params": [param], "weight_decay": decay})
    use_bb_armijo = args.icnn_step_rule == "bb-armijo"
    icnn_optimizer_name = args.icnn_optimizer
    if use_bb_armijo and icnn_optimizer_name != "sgd":
        print(
            "ICNN step rule 'bb-armijo' requires SGD-style updates; overriding optimizer to SGD."
        )
        icnn_optimizer_name = "sgd"
    if use_bb_armijo and args.icnn_alpha0 <= 0.0:
        args.icnn_alpha0 = max(args.lr_omega, 1e-6)
        print(
            f"Adjusted icnn_alpha0 to {args.icnn_alpha0:.6f} to ensure positive initial step size."
        )
    icnn_bb_config = None
    if use_bb_armijo:
        icnn_bb_config = {
            "alpha0": float(args.icnn_alpha0),
            "alpha_min": float(args.icnn_alpha_min),
            "alpha_max": float(args.icnn_alpha_max),
            "ls_c": float(args.icnn_ls_c),
            "ls_shrink": float(args.icnn_ls_shrink),
            "ls_max_steps": int(args.icnn_ls_max_steps),
        }
    if icnn_optimizer_name == "adam":
        opt_icnn = optim.Adam(
            icnn_param_groups,
            lr=args.lr_omega,
            betas=(args.icnn_beta1, args.icnn_beta2),
        )
    elif icnn_optimizer_name == "ademamix":
        ademamix_kwargs = {
            "lr": args.lr_omega,
            "betas": (args.icnn_beta1, args.icnn_beta2, args.ademamix_beta3),
            "alpha": args.ademamix_alpha,
            "beta3_warmup": args.ademamix_beta3_warmup,
            "alpha_warmup": args.ademamix_alpha_warmup,
        }
        opt_icnn = AdEMAMix(icnn_param_groups, **ademamix_kwargs)
    else:
        sgd_lr = args.icnn_alpha0 if use_bb_armijo else args.lr_omega
        opt_icnn = optim.SGD(
            icnn_param_groups,
            lr=sgd_lr,
            momentum=0.0,
        )
    print(f"ICNN optimizer: {icnn_optimizer_name}")
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_theta, T_max=total_epochs, last_epoch=-1)

    
    

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    tracker = TransportDeltaTracker(
        enabled=args.track_transport_deltas,
        max_batches=max(0, args.track_transport_batches),
        plot_dir=args.transport_delta_plot_dir,
    )
    jacobian_run_dir = Path(args.jacobian_log_dir) / run_id
    jacobian_run_dir.mkdir(parents=True, exist_ok=True)
    jacobian_epoch_records: List[Dict[str, object]] = []
    jacobian_summary = {
        "run_id": run_id,
        "jacobian_reg_weight": args.jacobian_reg_weight,
        "jacobian_reg_samples": args.jacobian_reg_samples,
        "jacobian_reg_iters": args.jacobian_reg_iters,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (jacobian_run_dir / "config.json").write_text(json.dumps(jacobian_summary, indent=2))

    if cosine_cfg["enabled"] and args.calibrate_penalty:
        print("Cosine penalty active; skipping penalty_lambda calibration.")
    elif args.calibrate_penalty and args.latent_eps_target and args.latent_eps_target > 0:
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

        (
            train_loss,
            train_acc,
            penalty_avg,
            adv_obj,
            jac_penalty,
            jac_sv,
            jac_counts,
        ) = train_one_epoch(
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
            cosine_cfg=cosine_cfg,
            jacobian_reg_weight=args.jacobian_reg_weight,
            jacobian_reg_samples=args.jacobian_reg_samples,
            jacobian_reg_iters=args.jacobian_reg_iters,
            rng=global_rng,
            use_margin_adv=args.use_margin_loss,
            icnn_step_rule=args.icnn_step_rule,
            icnn_bb_config=icnn_bb_config,
        )

        test_loss, test_acc = evaluate(phi, head, testloader, device)
        icnn_loss, icnn_acc, icnn_penalty = evaluate_under_icnn(
            phi,
            head,
            icnn,
            testloader,
            device,
            penalty_lambda=args.penalty_lambda,
            cosine_cfg=cosine_cfg,
        )

        if (
            args.eval_input_pgd
            and args.inp_steps > 0
            and args.inp_eps > 0
        ):
            sample_limit = args.eval_input_pgd_samples
            max_batches = None
            if sample_limit is not None and sample_limit > 0 and hasattr(testloader, "dataset"):
                samples_available = len(testloader.dataset)
                sample_limit = min(sample_limit, samples_available)
                batch_size = getattr(testloader, "batch_size", sample_limit)
                if batch_size <= 0:
                    batch_size = sample_limit
                max_batches = max(1, math.ceil(sample_limit / batch_size))
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
                max_batches=max_batches,
            )
            ipgd_l2 = pgd_info["avg_l2"]
            ipgd_linf = pgd_info["avg_linf"]
            ipgd_samples = pgd_info.get("samples")
        else:
            input_pgd_acc, ipgd_l2, ipgd_linf, ipgd_samples = None, None, None, None

        msg = (
            f"[Epoch {epoch:02d} | {phase}] train {train_loss:.4f}/{train_acc*100:.2f}% | "
            f"penalty {penalty_avg:.4f} | jac_pen {jac_penalty:.4f} | jac_sv {jac_sv:.4f} | adv_obj {adv_obj:.4f} | "
            f"test {test_loss:.4f}/{test_acc*100:.2f}% | "
            f"icnn {icnn_loss:.4f}/{icnn_acc*100:.2f}% (pen {icnn_penalty:.4f})"
        )
        if input_pgd_acc is not None:
            sample_note = f", n={ipgd_samples}" if ipgd_samples is not None else ""
            msg += f" | input-PGD {input_pgd_acc*100:.2f}% (L2 {ipgd_l2:.4f}, Linf {ipgd_linf:.4f}{sample_note})"
        if jacobian_ready and L_hat_used is not None:
            msg += f" | L_hat {L_hat_used:.4f}"
        print(msg)

        jacobian_epoch_records.append(
            {
                "epoch": epoch,
                "phase": phase,
                "mean_jac_penalty": float(jac_penalty),
                "mean_jac_sv": float(jac_sv),
                "jacobian_counts": int(jac_counts),
                "reg_weight": float(args.jacobian_reg_weight),
                "reg_samples": int(args.jacobian_reg_samples),
                "reg_iters": int(args.jacobian_reg_iters),
                "adv_objective": float(adv_obj),
                "train_penalty": float(penalty_avg),
            }
        )

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
                "train_jac_penalty": round(float(jac_penalty), 6),
                "train_jac_sv": round(float(jac_sv), 6),
                "adv_objective": round(float(adv_obj), 6),
                "test_loss": round(test_loss, 6),
                "test_acc": round(float(test_acc), 6),
                "icnn_loss": round(icnn_loss, 6),
                "icnn_acc": round(float(icnn_acc), 6),
                "icnn_penalty": round(float(icnn_penalty), 6),
                "input_pgd_acc": None if input_pgd_acc is None else round(float(input_pgd_acc), 6),
                "input_pgd_avg_l2": None if ipgd_l2 is None else round(float(ipgd_l2), 6),
                "input_pgd_avg_linf": None if ipgd_linf is None else round(float(ipgd_linf), 6),
                "input_pgd_samples": None if ipgd_samples is None else int(ipgd_samples),
                "penalty_lambda": args.penalty_lambda,
                "jacobian_reg_weight": args.jacobian_reg_weight,
                "jacobian_reg_samples": args.jacobian_reg_samples,
                "jacobian_reg_iters": args.jacobian_reg_iters,
                "jacobian_reg_counts": jac_counts,
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
            and not cosine_cfg["enabled"]
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
                ratio_raw = float(avg_delta / args.latent_eps_target)
                ratio_clamped = float(
                    min(
                        max(ratio_raw, args.calibration_ratio_min),
                        args.calibration_ratio_max,
                    )
                )
                smoothing = 1.0 + CALIBRATION_SMOOTHING * (ratio_clamped - 1.0)
                new_lambda = _clamp_penalty_lambda(args.penalty_lambda * smoothing)
                print(
                    f"[Calibration] Average delta {avg_delta:.4f}, ratio {ratio_raw:.4f} → {ratio_clamped:.4f}, "
                    f"smoothing {smoothing:.4f}, penalty_lambda {args.penalty_lambda:.6f} → {new_lambda:.6f}"
                )
                args.penalty_lambda = new_lambda
        tracker.finish_epoch()

    save_transport_delta_plot(tracker, args, run_id)

    if jacobian_epoch_records:
        epoch_fields = [
            "epoch",
            "phase",
            "mean_jac_penalty",
            "mean_jac_sv",
            "jacobian_counts",
            "reg_weight",
            "reg_samples",
            "reg_iters",
            "adv_objective",
            "train_penalty",
        ]
        jac_epoch_path = jacobian_run_dir / "jacobian_epoch_stats.csv"
        with jac_epoch_path.open("w", newline="") as f_epoch:
            writer = csv.DictWriter(f_epoch, fieldnames=epoch_fields)
            writer.writeheader()
            for record in jacobian_epoch_records:
                writer.writerow(record)
        overall_mean_sv = sum(rec["mean_jac_sv"] for rec in jacobian_epoch_records) / max(
            1, len(jacobian_epoch_records)
        )
        overall_max_sv = max(rec["mean_jac_sv"] for rec in jacobian_epoch_records)
        jacobian_summary.update(
            {
                "epochs_recorded": len(jacobian_epoch_records),
                "mean_jac_sv_overall": overall_mean_sv,
                "max_jac_sv_overall": overall_max_sv,
                "epoch_stats_path": str(jac_epoch_path),
            }
        )
        summary_path = jacobian_run_dir / "summary.json"
        summary_path.write_text(json.dumps(jacobian_summary, indent=2))
        print(f"Saved Jacobian epoch statistics to {jac_epoch_path}")

    if args.estimate_transport_jacobian:
        jac_loader = trainloader if args.jacobian_sv_split == "train" else testloader
        max_sv, mean_sv = estimate_transport_jacobian_sv(
            phi, icnn, jac_loader, device, args, rng=global_rng
        )
        if math.isnan(max_sv) or math.isnan(mean_sv):
            print("Transport Jacobian estimation returned NaN; consider adjusting parameters.")
        else:
            print(
                f"Estimated transport Jacobian σ_max ≈ {max_sv:.4f}, σ_mean ≈ {mean_sv:.4f}"
            )
            jacobian_summary.update(
                {
                    "transport_jacobian_max": max_sv,
                    "transport_jacobian_mean": mean_sv,
                }
            )
            summary_path = jacobian_run_dir / "summary.json"
            summary_path.write_text(json.dumps(jacobian_summary, indent=2))

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
 
