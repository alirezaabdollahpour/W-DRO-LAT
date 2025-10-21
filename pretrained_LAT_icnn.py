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
from datetime import datetime
import math
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


class InputConvexPotential(nn.Module):
    """Fully input-convex neural network (FICNN) for latent potentials."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "relu",
        strong_convexity: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_sizes: List[int] = list(hidden_sizes)
        if len(self.hidden_sizes) == 0:
            raise ValueError("ICNN requires at least one hidden layer.")

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
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.h_linears:
            if layer is not None:
                nn.init.xavier_uniform_(layer.weight_raw)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.hidden_output.weight_raw)
        nn.init.zeros_(self.hidden_output.bias)
        nn.init.xavier_uniform_(self.input_skip.weight)
        nn.init.zeros_(self.input_skip.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.size(0)
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


def _reduce_latents_for_plot(
    z: torch.Tensor, z_adv: torch.Tensor, method: str, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Project latent pairs into 2D using PCA or t-SNE."""
    z_flat = z.view(z.size(0), -1).cpu().float()
    z_adv_flat = z_adv.view(z_adv.size(0), -1).cpu().float()

    if method == "pca":
        z_np = z_flat.numpy()
        z_adv_np = z_adv_flat.numpy()

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

    coords_z, coords_adv = _reduce_latents_for_plot(
        z_tensor, z_adv_tensor, args.transport_viz_method, seed=args.seed
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
            norms.append(grad.view(grad.size(0), -1).norm(dim=1).mean().item())

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
        delta = z_adv - z_detached
        norms.append(delta.view(delta.size(0), -1).norm(dim=1).mean().item())

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
) -> Tuple[float, float, float, float]:
    phi.train(not head_only)
    head.train()
    icnn.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_penalty = 0.0
    total_adv_obj = 0.0

    progress = tqdm(loader, desc="Train", leave=False)
    for batch in progress:
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
        for _ in range(max(1, icnn_ascent_steps)):
            opt_icnn.zero_grad(set_to_none=True)
            z_adv_ascent, _ = adversarial_pushforward(icnn, z_detached, detach_for_model=False)
            logits_adv = head(z_adv_ascent)
            ce_adv = F.cross_entropy(logits_adv, y, reduction="mean")
            penalty = (z_detached - z_adv_ascent).view(batch_size, -1).pow(2).sum(dim=1).mean()
            adv_objective = ce_adv - penalty_lambda * penalty
            (-adv_objective).backward()

            # Remove stray gradients on the classifier, which acts as a frozen critic.
            for p in head.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            opt_icnn.step()
            icnn.project_convexity()

        # --- Model update (Danskin-style outer gradient) ---
        opt_theta.zero_grad(set_to_none=True)
        z_adv_fixed, delta = adversarial_pushforward(icnn, z, detach_for_model=True)
        logits = head(z_adv_fixed)
        loss = F.cross_entropy(logits, y, reduction="mean")
        loss.backward()
        opt_theta.step()

        with torch.no_grad():
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += batch_size
            total_penalty += delta.view(batch_size, -1).pow(2).sum(dim=1).mean().item()
            total_adv_obj += adv_objective.item()

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
        logits = head(z_adv)
        ce = F.cross_entropy(logits, y, reduction="sum")
        total_loss += ce.item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)
        total_penalty += (z_det - z_adv).view(z.size(0), -1).pow(2).sum(dim=1).mean().item()
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
        default=1e-3,
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
        default="pca",
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
        default=128,
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
        default=2,
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

    if args.calibrate_penalty and args.latent_eps_target and args.latent_eps_target > 0:
        mean_grad_norm = estimate_mean_grad_norm(
            phi,
            head,
            trainloader,
            device,
            num_batches=args.gamma_calibration_batches,
        )
        if mean_grad_norm > 0:
            args.penalty_lambda = mean_grad_norm / args.latent_eps_target
            print(
                f"Initialized penalty_lambda to {args.penalty_lambda:.6f} "
                f"from mean grad norm {mean_grad_norm:.6f} targeting ε_u={args.latent_eps_target:.6f}"
            )
        else:
            print(
                "Calibration requested but gradient norm estimate was zero; "
                "keeping existing penalty_lambda."
            )
    elif args.calibrate_penalty:
        print("Calibration requested but latent_eps_target not set or non-positive; skipping.")

    p_input = 2 if args.inp_p == "2" else float("inf")
    jacobian_ready = False
    L_hat_used = None

    for epoch, phase in enumerate(schedule, start=1):
        print(f"\n=== Epoch {epoch:02d}/{total_epochs} | Phase: {phase.upper()} ===")

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
            if avg_delta > 0:
                scale = avg_delta / args.latent_eps_target
                args.penalty_lambda *= scale
                print(
                    f"[Calibration] Average delta {avg_delta:.4f}, scaling penalty_lambda by {scale:.4f} "
                    f"to {args.penalty_lambda:.6f}"
                )
            else:
                print("[Calibration] Average delta was zero; penalty_lambda unchanged.")

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
