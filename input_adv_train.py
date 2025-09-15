
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
W-DRO (WRM) **in input pixel space** for CIFAR-10 with ResNet-18, CSV logging,
and test-time input-space PGD evaluation (multi-restart).

This mirrors the structure of your previous script but moves ALL inner
adversaries (PGD/WRM) to the **pixel domain** x_pix ∈ [0,1]. The model always
sees normalized inputs via an explicit to_normalized(x_pix) in the forward path.

Important:
- --eps and step sizes are in **pixel units** (e.g., 8/255 ≈ 0.03137).
- Projections/clamps are performed in pixel space; we clamp to [0,1].
- WRM penalty and γ initialization are computed in pixel space.
"""

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# -----------------------------
# Utilities
# -----------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def set_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def per_sample_l2_normalize(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    B = t.size(0)
    flat = t.view(B, -1)
    norms = flat.norm(p=2, dim=1).clamp(min=eps)
    reshape = (B,) + (1,) * (t.dim() - 1)
    return t / norms.view(reshape)

def to_pixel(x_norm: torch.Tensor) -> torch.Tensor:
    """Convert normalized tensor to pixel space [0,1]."""
    mean = torch.tensor(CIFAR10_MEAN, device=x_norm.device).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  device=x_norm.device).view(1, 3, 1, 1)
    return (x_norm * std) + mean

def to_normalized(x_pix: torch.Tensor) -> torch.Tensor:
    """Convert pixel tensor [0,1] to normalized space."""
    mean = torch.tensor(CIFAR10_MEAN, device=x_pix.device).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  device=x_pix.device).view(1, 3, 1, 1)
    return (x_pix - mean) / std

def auto_pgd_step_size(p, eps: float, steps: int, user_step_size: float) -> float:
    if user_step_size is not None and user_step_size > 0:
        return float(user_step_size)
    steps = max(1, int(steps))
    return float(2.0 * eps / steps)  # in pixel units


# -----------------------------
# Model
# -----------------------------

def build_resnet18(num_classes: int = 10):
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    return model


# -----------------------------
# Input-space adversary primitives (PIXEL SPACE)
# -----------------------------

def project_onto_lp_ball(delta_pix: torch.Tensor, eps: float, p: int) -> torch.Tensor:
    if p == 2:
        flat = delta_pix.view(delta_pix.size(0), -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        factors = (eps / norms).clamp(max=1.0)
        flat = flat * factors
        return flat.view_as(delta_pix)
    elif p == float('inf') or p == "inf":
        return delta_pix.clamp(min=-eps, max=eps)
    else:
        raise ValueError("Only L2 and Linf norms are supported for projection.")

def _random_start_input_ball_pix(x0_pix: torch.Tensor, eps: float, p) -> torch.Tensor:
    """Random start in pixel space; clamp to [0,1]."""
    if p == 2:
        z = torch.randn_like(x0_pix)
        z = per_sample_l2_normalize(z)
        B = x0_pix.size(0)
        r = torch.rand(B, device=x0_pix.device).view(B, *([1] * (x0_pix.dim() - 1)))
        delta0 = z * (r * eps)
    else:
        delta0 = torch.empty_like(x0_pix).uniform_(-eps, eps)
    return (x0_pix + delta0).clamp(0.0, 1.0)


@dataclass
class InnerConfig:
    # Inner max method in INPUT space
    method: str  # {'pgd-input','wrm-input'}
    # PGD controls (pixel units)
    p: int = float('inf')
    eps: float = 8/255
    pgd_steps: int = 7
    pgd_step_size: float = 0.0  # 0.0 => auto (2*eps/steps)
    # WRM controls
    gamma: float = 10.0
    inner_steps: int = 7
    inner_step_size: float = 0.8  # used if wrm-step-rule=fixed
    # Adaptive step rules (for WRM)
    wrm_step_rule: str = "armijo"   # {'armijo','fixed','bb'}
    wrm_alpha0: float = 1.0
    wrm_ls_c: float = 0.1
    wrm_ls_shrink: float = 0.5
    wrm_ls_max_steps: int = 10
    wrm_alpha_min: float = 1e-6
    wrm_alpha_max: float = 10.0


def _wrm_objective_input_pix(CE_mean: torch.Tensor, x_pix: torch.Tensor, x0_pix: torch.Tensor, gamma: float):
    """
    f(x_pix) = CE_mean - 0.5*gamma*||x_pix - x0_pix||^2   (batch-averaged)
    Returns (f_value, penalty_value).
    """
    B = x_pix.size(0)
    l2sq = (x_pix - x0_pix).view(B, -1).pow(2).sum(dim=1).mean()
    penalty = 0.5 * gamma * l2sq
    return CE_mean - penalty, penalty


def inner_max_x(x_norm0: torch.Tensor,
                y: torch.Tensor,
                model: nn.Module,
                cfg: InnerConfig) -> Tuple[torch.Tensor, dict]:
    """
    Inner maximization over **pixel-space** input x_pix (PGD or WRM with optional adaptive α).
    Returns x_adv_norm (normalized view for the model) and info dict.
    """
    device = x_norm0.device
    # Convert starting batch to pixel space and keep it fixed as the center
    x0_pix = to_pixel(x_norm0).detach()

    if cfg.method == "pgd-input":
        x_pix = _random_start_input_ball_pix(x0_pix, cfg.eps, cfg.p).detach().requires_grad_(True)
        step_size = auto_pgd_step_size(cfg.p, cfg.eps, cfg.pgd_steps, cfg.pgd_step_size)

        for _ in range(cfg.pgd_steps):
            logits = model(to_normalized(x_pix))
            loss = F.cross_entropy(logits, y, reduction='mean')
            g_pix = torch.autograd.grad(loss, x_pix, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                if cfg.p == 2:
                    step = step_size * per_sample_l2_normalize(g_pix)
                else:  # Linf
                    step = step_size * torch.sign(g_pix)
                x_pix = x_pix + step
                delta_pix = x_pix - x0_pix
                delta_pix = project_onto_lp_ball(delta_pix, cfg.eps, 2 if cfg.p == 2 else float('inf'))
                x_pix = (x0_pix + delta_pix).clamp(0.0, 1.0).detach().requires_grad_(True)

        with torch.no_grad():
            d = (x_pix - x0_pix)
            avg_l2 = d.view(d.size(0), -1).norm(p=2, dim=1).mean().item()
            avg_linf = d.abs().view(d.size(0), -1).max(dim=1)[0].mean().item()
        return to_normalized(x_pix).detach(), {"avg_l2": avg_l2, "avg_linf": avg_linf, "penalty": None}

    elif cfg.method == "wrm-input":
        # Penalized ascent with adaptive α on pixel x
        x_pix = x0_pix.clone().detach().requires_grad_(True)

        g_prev = None
        x_prev = None
        alpha_prev = cfg.wrm_alpha0

        for t in range(cfg.inner_steps):
            logits = model(to_normalized(x_pix))
            CE = F.cross_entropy(logits, y, reduction='mean')
            f_val, penalty = _wrm_objective_input_pix(CE, x_pix, x0_pix, cfg.gamma)
            g_pix = torch.autograd.grad(f_val, x_pix, retain_graph=False, create_graph=False)[0]

            if cfg.wrm_step_rule == "fixed":
                alpha = cfg.inner_step_size

            elif cfg.wrm_step_rule == "bb":
                if t == 0 or g_prev is None or x_prev is None:
                    alpha = alpha_prev
                else:
                    s = (x_pix - x_prev).view(x_pix.size(0), -1)
                    yk = (g_pix - g_prev).view(g_pix.size(0), -1)
                    num = (s * s).sum(dim=1).mean()
                    den = (s * yk).sum(dim=1).mean().clamp(min=1e-12)
                    alpha = (num / den).item()
                    alpha = float(max(cfg.wrm_alpha_min, min(cfg.wrm_alpha_max, alpha)))
                # Armijo safeguard in pixel space
                alpha_try = alpha
                accepted = False
                for _ in range(cfg.wrm_ls_max_steps):
                    x_try = (x_pix + alpha_try * g_pix).clamp(0.0, 1.0).detach().requires_grad_(True)
                    logits_try = model(to_normalized(x_try))
                    CE_try = F.cross_entropy(logits_try, y, reduction='mean')
                    f_try, _ = _wrm_objective_input_pix(CE_try, x_try, x0_pix, cfg.gamma)
                    rhs = f_val + cfg.wrm_ls_c * alpha_try * (g_pix.view(g_pix.size(0), -1).pow(2).sum(dim=1).mean())
                    if f_try >= rhs:
                        accepted = True
                        break
                    alpha_try *= cfg.wrm_ls_shrink
                    if alpha_try < cfg.wrm_alpha_min:
                        break
                alpha = alpha_try
                if not accepted:
                    alpha = cfg.wrm_alpha_min
                alpha_prev = alpha

            else:
                # Armijo backtracking ascent (default)
                alpha = alpha_prev if t > 0 else cfg.wrm_alpha0
                for _ in range(cfg.wrm_ls_max_steps):
                    x_try = (x_pix + alpha * g_pix).clamp(0.0, 1.0).detach().requires_grad_(True)
                    logits_try = model(to_normalized(x_try))
                    CE_try = F.cross_entropy(logits_try, y, reduction='mean')
                    f_try, _ = _wrm_objective_input_pix(CE_try, x_try, x0_pix, cfg.gamma)
                    rhs = f_val + cfg.wrm_ls_c * alpha * (g_pix.view(g_pix.size(0), -1).pow(2).sum(dim=1).mean())
                    if f_try >= rhs:
                        break
                    alpha *= cfg.wrm_ls_shrink
                    if alpha < cfg.wrm_alpha_min:
                        break
                alpha_prev = alpha

            with torch.no_grad():
                x_prev = x_pix.detach()
                g_prev = g_pix.detach()
                x_pix = (x_pix + alpha * g_pix).clamp(0.0, 1.0).detach().requires_grad_(True)

        with torch.no_grad():
            d = (x_pix - x0_pix)
            penalty_val = 0.5 * cfg.gamma * d.view(d.size(0), -1).pow(2).sum(dim=1).mean().item()
            avg_l2 = d.view(d.size(0), -1).norm(p=2, dim=1).mean().item()
            avg_linf = d.abs().view(d.size(0), -1).max(dim=1)[0].mean().item()
        return to_normalized(x_pix).detach(), {"avg_l2": avg_l2, "avg_linf": avg_linf, "penalty": penalty_val}

    else:
        raise ValueError("inner_max_x: method must be 'pgd-input' or 'wrm-input'")


# -----------------------------
# WRM γ helpers (init + online calibration) in INPUT PIXEL space
# -----------------------------

@torch.no_grad()
def _take_first_batches(loader, n):
    it = iter(loader)
    batches = []
    for _ in range(n):
        try:
            batches.append(next(it))
        except StopIteration:
            break
    return batches

def wrm_init_gamma_input(model, loader, device, target_eps: float, n_batches: int = 2) -> float:
    """
    γ₀ ≈ E[||∇_{x_pix} L||₂] / target_eps  on clean batches.
    """
    model.eval()
    batches = _take_first_batches(loader, max(1, n_batches))
    norms = []
    for x_norm, y in batches:
        x_norm = x_norm.to(device)
        y = y.to(device)
        with torch.enable_grad():
            x_pix = to_pixel(x_norm).detach().requires_grad_(True)
            logits = model(to_normalized(x_pix))
            loss = F.cross_entropy(logits, y, reduction='mean')
            gx_pix = torch.autograd.grad(loss, x_pix, retain_graph=False, create_graph=False)[0]
            n = gx_pix.view(gx_pix.size(0), -1).norm(p=2, dim=1).mean().item()
            norms.append(n)
    if len(norms) == 0:
        return 10.0 / max(target_eps, 1e-6)
    gbar = sum(norms) / len(norms)
    return max(gbar / max(target_eps, 1e-8), 1e-8)


def wrm_estimate_avg_l2_input(model, loader, device, inner_cfg: InnerConfig, n_batches: int = 1) -> float:
    """
    Run a tiny WRM inner max on a few batches to estimate avg ||x_pix - x0_pix||_2.
    """
    model.eval()
    batches = _take_first_batches(loader, max(1, n_batches))
    vals = []
    for x_norm, y in batches:
        x_norm = x_norm.to(device)
        y = y.to(device)
        with torch.enable_grad():
            cfg = InnerConfig(
                method="wrm-input",
                p=inner_cfg.p, eps=inner_cfg.eps,
                pgd_steps=inner_cfg.pgd_steps, pgd_step_size=inner_cfg.pgd_step_size,
                gamma=inner_cfg.gamma, inner_steps=inner_cfg.inner_steps, inner_step_size=inner_cfg.inner_step_size,
                wrm_step_rule=inner_cfg.wrm_step_rule, wrm_alpha0=inner_cfg.wrm_alpha0,
                wrm_ls_c=inner_cfg.wrm_ls_c, wrm_ls_shrink=inner_cfg.wrm_ls_shrink,
                wrm_ls_max_steps=inner_cfg.wrm_ls_max_steps,
                wrm_alpha_min=inner_cfg.wrm_alpha_min, wrm_alpha_max=inner_cfg.wrm_alpha_max
            )
            x_adv_norm, info = inner_max_x(x_norm, y, model, cfg)
            vals.append(info["avg_l2"])
    if len(vals) == 0:
        return 0.0
    return sum(vals) / len(vals)


# -----------------------------
# Training & evaluation loops
# -----------------------------

def train_one_epoch(model, loader, optimizer, device, method, inner_cfg: InnerConfig):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_norm, y in loader:
        x_norm = x_norm.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        if method == "erm":
            logits = model(x_norm)
            loss = F.cross_entropy(logits, y)
            loss.backward()

        else:
            # Inner adversary in **pixel space** (returns normalized adversarial for the model)
            x_adv_norm, _ = inner_max_x(x_norm, y, model, inner_cfg)
            logits = model(x_adv_norm)
            if method == "wrm-input":
                # Recompute pixel-space delta for penalty
                with torch.no_grad():
                    x_pix0 = to_pixel(x_norm)
                    x_pix_adv = to_pixel(x_adv_norm)
                    d = (x_pix_adv - x_pix0)
                penalty = 0.5 * inner_cfg.gamma * d.view(d.size(0), -1).pow(2).sum(dim=1).mean()
                loss = F.cross_entropy(logits, y) - penalty
            else:
                loss = F.cross_entropy(logits, y)
            loss.backward()

        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item() * x_norm.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x_norm.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_norm, y in loader:
        x_norm = x_norm.to(device)
        y = y.to(device)
        logits = model(x_norm)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x_norm.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate_under_input_attack(model, loader, device, method_for_eval: str, inner_cfg: InnerConfig):
    """
    Robust accuracy under an **input-space** attack produced by the given method (PGD/WRM in pixel space).
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    avg_l2, avg_linf, avg_penalty = 0.0, 0.0, 0.0
    n_batches = 0

    for x_norm, y in loader:
        x_norm = x_norm.to(device)
        y = y.to(device)

        with torch.enable_grad():
            cfg = InnerConfig(
                method=method_for_eval,
                p=inner_cfg.p, eps=inner_cfg.eps,
                pgd_steps=inner_cfg.pgd_steps,
                pgd_step_size=auto_pgd_step_size(inner_cfg.p, inner_cfg.eps, inner_cfg.pgd_steps, inner_cfg.pgd_step_size),
                gamma=inner_cfg.gamma,
                inner_steps=inner_cfg.inner_steps,
                inner_step_size=inner_cfg.inner_step_size,
                wrm_step_rule=inner_cfg.wrm_step_rule, wrm_alpha0=inner_cfg.wrm_alpha0,
                wrm_ls_c=inner_cfg.wrm_ls_c, wrm_ls_shrink=inner_cfg.wrm_ls_shrink,
                wrm_ls_max_steps=inner_cfg.wrm_ls_max_steps,
                wrm_alpha_min=inner_cfg.wrm_alpha_min, wrm_alpha_max=inner_cfg.wrm_alpha_max
            )
            x_adv_norm, info = inner_max_x(x_norm, y, model, cfg)
            logits = model(x_adv_norm)

        with torch.no_grad():
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += x_norm.size(0)
            n_batches += 1
            avg_l2 += info["avg_l2"]
            avg_linf += info["avg_linf"]
            if info["penalty"] is not None:
                avg_penalty += info["penalty"]

    robust_acc = total_correct / total_samples
    avg_l2 /= max(1, n_batches)
    avg_linf /= max(1, n_batches)
    avg_penalty = (avg_penalty / max(1, n_batches)) if avg_penalty != 0.0 else None
    return robust_acc, {"avg_l2": avg_l2, "avg_linf": avg_linf, "avg_penalty": avg_penalty}


# -----------------------------
# Test-time input-space PGD with restarts (PIXEL SPACE)
# -----------------------------

def evaluate_under_input_pgd(model, loader, device, p, eps, steps, step_size, restarts: int = 1):
    model.eval()
    total_correct = 0
    total = 0
    avg_l2 = 0.0
    avg_linf = 0.0
    n_batches = 0

    step_size = auto_pgd_step_size(p, eps, steps, step_size)
    restarts = max(1, int(restarts))

    for x_norm, y in loader:
        x_norm = x_norm.to(device)
        y = y.to(device)
        x0_pix = to_pixel(x_norm).detach()

        best_delta = torch.zeros_like(x0_pix)
        best_loss = torch.full((x0_pix.size(0),), -1e9, device=device)

        for _ in range(restarts):
            x_pix = _random_start_input_ball_pix(x0_pix, eps, p).detach().requires_grad_(True)

            for _ in range(steps):
                logits = model(to_normalized(x_pix))
                loss_mean = F.cross_entropy(logits, y, reduction='mean')
                g_pix = torch.autograd.grad(loss_mean, x_pix, retain_graph=False, create_graph=False)[0]
                with torch.no_grad():
                    if p == 2:
                        step = step_size * per_sample_l2_normalize(g_pix)
                    else:
                        step = step_size * torch.sign(g_pix)
                    x_pix = x_pix + step
                    delta_pix = x_pix - x0_pix
                    delta_pix = project_onto_lp_ball(delta_pix, eps, 2 if p == 2 else float('inf'))
                    x_pix = (x0_pix + delta_pix).clamp(0.0, 1.0).detach().requires_grad_(True)

            with torch.no_grad():
                logits = model(to_normalized(x_pix))
                loss_vec = F.cross_entropy(logits, y, reduction='none')
                delta_pix = (x_pix - x0_pix).detach()
                mask = loss_vec > best_loss
                if mask.any():
                    best_loss[mask] = loss_vec[mask]
                    best_delta[mask] = delta_pix[mask]

        with torch.no_grad():
            x_adv_best = (x0_pix + best_delta).clamp(0.0, 1.0)
            logits = model(to_normalized(x_adv_best))
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x_norm.size(0)
            d = best_delta
            avg_l2 += d.view(d.size(0), -1).norm(p=2, dim=1).mean().item()
            avg_linf += d.abs().view(d.size(0), -1).max(dim=1)[0].mean().item()
            n_batches += 1

    acc = total_correct / total
    avg_l2 /= max(1, n_batches)
    avg_linf /= max(1, n_batches)
    return acc, {"avg_l2": avg_l2, "avg_linf": avg_linf}


# -----------------------------
# Data
# -----------------------------

def get_cifar10_loaders(batch_size=128, num_workers=2):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader


# -----------------------------
# CSV logging helpers
# -----------------------------

CSV_HEADER = [
    "run_id", "time_iso", "epoch",
    # Metrics
    "train_loss", "train_acc", "test_loss", "test_acc",
    "robust_acc", "avg_penalty", "avg_l2", "avg_linf",
    # Input-PGD metrics
    "input_pgd_acc", "input_pgd_avg_l2", "input_pgd_avg_linf",
    # Core config/hparams
    "method", "eval_attack",
    "p", "eps", "pgd_steps", "pgd_step_size",
    "gamma", "inner_steps", "inner_step_size",
    "wrm_step_rule", "wrm_alpha0", "wrm_ls_c", "wrm_ls_shrink",
    "wrm_ls_max_steps", "wrm_alpha_min", "wrm_alpha_max",
    "batch_size", "lr", "momentum", "weight_decay", "seed",
    # Input-PGD config
    "inp_p", "inp_eps", "inp_steps", "inp_step_size", "inp_restarts",
]

def append_row(csv_path: str, row_dict: dict):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            w.writeheader()
        row = {k: row_dict.get(k, None) for k in CSV_HEADER}
        w.writerow(row)


# -----------------------------
# Main
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Input-space W-DRO (WRM) on CIFAR-10 (ResNet-18) in PIXEL SPACE.")
    parser.add_argument("--method", type=str, default="erm",
                        choices=["erm", "pgd-input", "wrm-input"],
                        help="Training method.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1)

    # Inner problem config (input) — **pixel units**
    parser.add_argument("--p", type=str, default="inf", help="Norm for PGD ball: '2' or 'inf'")
    parser.add_argument("--eps", type=float, default=8/255, help="Radius for input PGD and target WRM radius (pixel units).")
    parser.add_argument("--pgd-steps", type=int, default=7)
    parser.add_argument("--pgd-step-size", type=float, default=0.0, help="If <=0, auto = 2*eps/steps (pixel units).")

    parser.add_argument("--gamma", type=float, default=10.0, help="WRM penalty coefficient γ for 0.5||x_pix-x0_pix||^2.")
    parser.add_argument("--inner-steps", type=int, default=7, help="Ascent steps for WRM inner maximization.")
    parser.add_argument("--inner-step-size", type=float, default=0.8, help="Used if --wrm-step-rule=fixed")

    # WRM adaptive α flags (on input)
    parser.add_argument("--wrm-step-rule", type=str, default="fixed",
                        choices=["armijo", "fixed", "bb"],
                        help="Adaptive rule for WRM ascent step-size α.")
    parser.add_argument("--wrm-alpha0", type=float, default=1.0, help="Initial α guess for Armijo/BB.")
    parser.add_argument("--wrm-ls-c", type=float, default=0.1, help="Armijo c in (0,0.5).")
    parser.add_argument("--wrm-ls-shrink", type=float, default=0.5, help="Armijo backtracking factor.")
    parser.add_argument("--wrm-ls-max-steps", type=int, default=10)
    parser.add_argument("--wrm-alpha-min", type=float, default=1e-6)
    parser.add_argument("--wrm-alpha-max", type=float, default=10.0)

    # WRM auto-γ options (input)
    parser.add_argument("--wrm-auto-gamma", action="store_true",
                        help="Auto-initialize WRM gamma from clean ∥∇_{x_pix} L∥ (γ₀≈E||∇x_pix L||/eps).")
    parser.add_argument("--wrm-init-batches", type=int, default=2,
                        help="Number of clean batches used to initialize γ.")
    parser.add_argument("--wrm-calibrate-online", action="store_true",
                        help="After each epoch: γ←γ*(avg_l2/eps) using a tiny WRM solve.")
    parser.add_argument("--wrm-calibrate-every", type=int, default=1)
    parser.add_argument("--wrm-calibrate-batches", type=int, default=1)

    parser.add_argument("--eval-attack", type=str, default="pgd-input",
                        choices=["pgd-input", "wrm-input"],
                        help="Input-space attack used for robust evaluation each epoch.")
    parser.add_argument("--save", type=str, default="", help="Path to save final model (optional).")

    # CSV logging
    parser.add_argument("--log-csv", type=str, default="./runs_wdro_input_pixel.csv",
                        help="Path to CSV file where per-epoch results are appended.")

    # Input-space PGD eval config (test-time) — **pixel units**
    parser.add_argument("--inp-p", type=str, default="inf", choices=["2", "inf"],
                        help="Norm for input-space PGD at test time.")
    parser.add_argument("--inp-eps", type=float, default=8/255,
                        help="Radius for input-space PGD (pixel units).")
    parser.add_argument("--inp-steps", type=int, default=20, help="Steps for input-space PGD (0 to disable).")
    parser.add_argument("--inp-step-size", type=float, default=0.0, help="If <=0, auto = 2*eps/steps (pixel units).")
    parser.add_argument("--inp-restarts", type=int, default=5, help="Random restarts for input-space PGD.")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    p_input = float('inf') if str(args.p).lower() in ["inf", "linf"] else int(args.p)
    p_eval  = float('inf') if str(args.inp_p).lower() in ["inf", "linf"] else int(args.inp_p)

    trainloader, testloader = get_cifar10_loaders(batch_size=args.batch_size)

    model = build_resnet18(num_classes=10).to(device)

    # Auto-calibrate PGD step size if requested (used for 'pgd-input' inner + test PGD) — pixel units
    pgd_step_size_calibrated = auto_pgd_step_size(p_input, args.eps, args.pgd_steps, args.pgd_step_size)

    params = list(filter(lambda t: t.requires_grad, model.parameters()))
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=-1)

    inner_cfg = InnerConfig(
        method=args.method if args.method in ["pgd-input", "wrm-input"] else "pgd-input",
        p=p_input, eps=args.eps,
        pgd_steps=args.pgd_steps, pgd_step_size=pgd_step_size_calibrated,
        gamma=args.gamma, inner_steps=args.inner_steps, inner_step_size=args.inner_step_size,
        wrm_step_rule=args.wrm_step_rule, wrm_alpha0=args.wrm_alpha0,
        wrm_ls_c=args.wrm_ls_c, wrm_ls_shrink=args.wrm_ls_shrink,
        wrm_ls_max_steps=args.wrm_ls_max_steps,
        wrm_alpha_min=args.wrm_alpha_min, wrm_alpha_max=args.wrm_alpha_max
    )

    # ---- WRM γ auto-initialization (pixel) ----
    if inner_cfg.method == "wrm-input" and args.wrm_auto_gamma:
        gamma0 = wrm_init_gamma_input(model, trainloader, device, target_eps=args.eps, n_batches=args.wrm_init_batches)
        inner_cfg.gamma = gamma0
        print(f"[WRM-INPUT] Auto-initialized gamma to {inner_cfg.gamma:.6f} using {args.wrm_init_batches} batch(es) (target eps={args.eps}).")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, device, args.method, inner_cfg)
        test_loss, test_acc = evaluate(model, testloader, device)
        robust_acc, rinfo = evaluate_under_input_attack(model, testloader, device, args.eval_attack, inner_cfg)

        # Optional: WRM online γ calibration every K epochs
        if (inner_cfg.method == "wrm-input"
            and args.wrm_calibrate_online
            and (epoch % max(1, args.wrm_calibrate_every) == 0)):
            with torch.no_grad():
                est_avg_l2 = wrm_estimate_avg_l2_input(model, trainloader, device, inner_cfg, n_batches=args.wrm_calibrate_batches)
            if est_avg_l2 > 0 and args.eps > 0:
                new_gamma = max(inner_cfg.gamma * (est_avg_l2 / args.eps), 1e-8)
                print(f"[WRM-INPUT] Calibrate gamma: avg_l2={est_avg_l2:.6f}, target eps={args.eps}, "
                      f"gamma {inner_cfg.gamma:.6f} -> {new_gamma:.6f}")
                inner_cfg.gamma = new_gamma

        # Input-space PGD evaluation (with restarts; disable with --inp-steps 0) — pixel units
        if args.inp_steps > 0 and args.inp_eps > 0:
            input_pgd_acc, ipgd_info = evaluate_under_input_pgd(
                model, testloader, device,
                p=p_eval, eps=args.inp_eps,
                steps=args.inp_steps, step_size=args.inp_step_size,
                restarts=args.inp_restarts
            )
            ipgd_l2, ipgd_linf = ipgd_info["avg_l2"], ipgd_info["avg_linf"]
        else:
            input_pgd_acc, ipgd_l2, ipgd_linf = None, None, None

        msg = f"[Epoch {epoch:02d}] train loss {train_loss:.4f} acc {train_acc*100:.2f}% | " \
              f"test loss {test_loss:.4f} acc {test_acc*100:.2f}% | " \
              f"robust({args.eval_attack}) acc {robust_acc*100:.2f}%"
        if rinfo["avg_penalty"] is not None:
            msg += f" | avg_penalty {rinfo['avg_penalty']:.4f}"
        msg += f" | adv avg_l2 {rinfo['avg_l2']:.3f} avg_linf {rinfo['avg_linf']:.3f}"
        if input_pgd_acc is not None:
            msg += f" | input-PGD acc {input_pgd_acc*100:.2f}% (R={args.inp_restarts}, L2 {ipgd_l2:.3f}, Linf {ipgd_linf:.3f})"
        print(msg)

        # CSV logging
        append_row(args.log_csv, {
            "run_id": run_id,
            "time_iso": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(float(train_acc), 6),
            "test_loss": round(test_loss, 6),
            "test_acc": round(float(test_acc), 6),
            "robust_acc": round(float(robust_acc), 6),
            "avg_penalty": None if rinfo["avg_penalty"] is None else round(float(rinfo["avg_penalty"]), 6),
            "avg_l2": round(float(rinfo["avg_l2"]), 6),
            "avg_linf": round(float(rinfo["avg_linf"]), 6),
            "input_pgd_acc": None if input_pgd_acc is None else round(float(input_pgd_acc), 6),
            "input_pgd_avg_l2": None if ipgd_l2 is None else round(float(ipgd_l2), 6),
            "input_pgd_avg_linf": None if ipgd_linf is None else round(float(ipgd_linf), 6),
            "method": args.method,
            "eval_attack": args.eval_attack,
            "p": str(args.p),
            "eps": args.eps,
            "pgd_steps": args.pgd_steps,
            "pgd_step_size": inner_cfg.pgd_step_size,
            "gamma": inner_cfg.gamma,
            "inner_steps": args.inner_steps,
            "inner_step_size": args.inner_step_size,
            "wrm_step_rule": inner_cfg.wrm_step_rule,
            "wrm_alpha0": inner_cfg.wrm_alpha0,
            "wrm_ls_c": inner_cfg.wrm_ls_c,
            "wrm_ls_shrink": inner_cfg.wrm_ls_shrink,
            "wrm_ls_max_steps": inner_cfg.wrm_ls_max_steps,
            "wrm_alpha_min": inner_cfg.wrm_alpha_min,
            "wrm_alpha_max": inner_cfg.wrm_alpha_max,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "inp_p": str(args.inp_p),
            "inp_eps": args.inp_eps,
            "inp_steps": args.inp_steps,
            "inp_step_size": auto_pgd_step_size(p_eval, args.inp_eps, args.inp_steps, args.inp_step_size),
            "inp_restarts": args.inp_restarts,
        })

        lr_scheduler.step()

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.save)
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
