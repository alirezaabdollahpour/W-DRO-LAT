#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Latent-space adversarial training on CIFAR-10 (ResNet-18) with **Jacobian-aware** mapping
from **pixel ε** to **latent ε_target** and **test-time input-space PGD in PIXEL SPACE**.

Highlights
----------
- Split schedule: clean ERM for --epochs-clean, then adversarial (--adv-method) for --epochs-adv.
- Pixel-space multi-restart PGD evaluation (attack and projection in [0,1] pixel units).
- Jacobian-aware latent ε_target:
    * Estimate L_hat ≈ E||J_Φ(x)||_2 on a few batches (vector-Jacobian product).
    * Convert input ε to an L2 radius:
          inp_p==2:  ε_in_L2 = inp_eps
          inp_p==∞: ε_in_L2 = inp_eps * sqrt(d)   (d = 3*H*W)
    * Set ε_latent_target = L_hat * ε_in_L2
- Use ε_latent_target for:
    * latent-PGD radius (eps) during adversarial epochs
    * WRM γ auto-init: γ0 ≈ E||∇_u L||_2 / ε_latent_target
    * WRM online calibration: γ ← γ * (avg_l2 / ε_latent_target)

Notes
-----
* --inp-eps, --inp-step-size are in **pixel units** (e.g., 8/255).
* Latent adversary is unchanged structurally (PGD/WRM), only its ε/γ are Jacobian-aware.
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np


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
    import random
    random.seed(seed)
    np.random.seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def per_sample_l2_normalize(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    B = t.size(0)
    flat = t.view(B, -1)
    norms = flat.norm(p=2, dim=1).clamp(min=eps)  # (B,)
    reshape = (B,) + (1,) * (t.dim() - 1)
    return t / norms.view(reshape)

def to_pixel(x_norm: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=x_norm.device).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  device=x_norm.device).view(1, 3, 1, 1)
    return (x_norm * std) + mean

def to_normalized(x_pix: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=x_pix.device).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  device=x_pix.device).view(1, 3, 1, 1)
    return (x_pix - mean) / std

def auto_pgd_step_size(p, eps: float, steps: int, user_step_size: float) -> float:
    if user_step_size is not None and user_step_size > 0:
        return float(user_step_size)
    steps = max(1, int(steps))
    return float(2.0 * eps / steps)


# -----------------------------
# Model split: ResNet-18 -> (Phi, Head)
# -----------------------------

class HeadFromResNet18(nn.Module):
    def __init__(self, base: torchvision.models.ResNet, cut_layer: str):
        super().__init__()
        self.cut_layer = cut_layer
        if cut_layer == "conv1":
            self.tail = nn.Sequential(base.bn1, base.relu, base.maxpool,
                                      base.layer1, base.layer2, base.layer3, base.layer4,
                                      base.avgpool)
            self.expect_shape = "after_conv1"
        elif cut_layer == "layer1":
            self.tail = nn.Sequential(base.layer2, base.layer3, base.layer4, base.avgpool)
            self.expect_shape = "after_layer1"
        elif cut_layer == "layer2":
            self.tail = nn.Sequential(base.layer3, base.layer4, base.avgpool)
            self.expect_shape = "after_layer2"
        elif cut_layer == "layer3":
            self.tail = nn.Sequential(base.layer4, base.avgpool)
            self.expect_shape = "after_layer3"
        elif cut_layer == "layer4":
            self.tail = nn.Sequential(base.avgpool)
            self.expect_shape = "after_layer4"
        elif cut_layer == "avgpool":
            self.tail = nn.Identity()
            self.expect_shape = "after_avgpool"
        else:
            raise ValueError(f"Unsupported cut layer: {cut_layer}")
        self.fc = base.fc

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if self.expect_shape == "after_avgpool":
            x = u if u.dim() == 2 else torch.flatten(u, 1)
            return self.fc(x)
        x = self.tail(u)
        x = torch.flatten(x, 1)
        return self.fc(x)

class PhiFromResNet18(nn.Module):
    def __init__(self, base: torchvision.models.ResNet, cut_layer: str):
        super().__init__()
        self.cut_layer = cut_layer
        if cut_layer == "conv1":
            self.head = nn.Sequential(base.conv1)
        elif cut_layer == "layer1":
            self.head = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1)
        elif cut_layer == "layer2":
            self.head = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2)
        elif cut_layer == "layer3":
            self.head = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2, base.layer3)
        elif cut_layer == "layer4":
            self.head = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2, base.layer3, base.layer4)
        elif cut_layer == "avgpool":
            self.head = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                      base.layer1, base.layer2, base.layer3, base.layer4, base.avgpool)
        else:
            raise ValueError(f"Unsupported cut layer: {cut_layer}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

def build_split_resnet18(num_classes: int = 10, cut_layer: str = "layer2"):
    base = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    phi = PhiFromResNet18(base, cut_layer=cut_layer)
    head = HeadFromResNet18(base, cut_layer=cut_layer)
    return phi, head


# -----------------------------
# Latent adversary primitives
# -----------------------------

def project_onto_lp_ball(delta: torch.Tensor, eps: float, p: int) -> torch.Tensor:
    if p == 2:
        flat = delta.view(delta.size(0), -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        factors = (eps / norms).clamp(max=1.0)
        flat = flat * factors
        return flat.view_as(delta)
    elif p == float('inf') or p == "inf":
        return delta.clamp(min=-eps, max=eps)
    else:
        raise ValueError("Only L2 and Linf norms are supported for projection.")

@dataclass
class InnerConfig:
    method: str  # 'pgd-latent' or 'wrm-latent'
    p: int = float('inf')
    eps: float = 0.031
    pgd_steps: int = 7
    pgd_step_size: float = 0.0
    gamma: float = 10.0
    inner_steps: int = 7
    inner_step_size: float = 0.8
    wrm_step_rule: str = "bb"   # {'armijo','fixed','bb'}
    wrm_alpha0: float = 1.0
    wrm_ls_c: float = 0.1
    wrm_ls_shrink: float = 0.5
    wrm_ls_max_steps: int = 10
    wrm_alpha_min: float = 1e-6
    wrm_alpha_max: float = 10.0

def _random_start_in_lp_ball(shape_like: torch.Tensor, eps: float, p: int) -> torch.Tensor:
    if p == 2:
        z = torch.randn_like(shape_like)
        z = per_sample_l2_normalize(z)
        B = shape_like.size(0)
        r = torch.rand(B, device=shape_like.device).view(B, *([1] * (shape_like.dim() - 1)))
        return z * (r * eps)
    elif p == float('inf') or p == "inf":
        return torch.empty_like(shape_like).uniform_(-eps, eps)
    else:
        raise ValueError("Unsupported p for random start (use 2 or inf).")

def _wrm_objective(loss_mean: torch.Tensor, delta: torch.Tensor, gamma: float):
    B = delta.size(0)
    l2sq = delta.view(B, -1).pow(2).sum(dim=1).mean()
    penalty = 0.5 * gamma * l2sq
    return loss_mean - penalty, penalty

def inner_max_delta(u0: torch.Tensor,
                    head: nn.Module,
                    y: torch.Tensor,
                    cfg: InnerConfig) -> Tuple[torch.Tensor, dict]:
    if cfg.method == "pgd-latent":
        delta = _random_start_in_lp_ball(u0, cfg.eps, cfg.p).detach().requires_grad_(True)
        step_size = auto_pgd_step_size(cfg.p, cfg.eps, cfg.pgd_steps, cfg.pgd_step_size)
        for _ in range(cfg.pgd_steps):
            logits = head(u0 + delta)
            loss = F.cross_entropy(logits, y, reduction='mean')
            grads = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                step = step_size * (per_sample_l2_normalize(grads) if cfg.p == 2 else torch.sign(grads))
                delta = delta + step
                delta = project_onto_lp_ball(delta, cfg.eps, 2 if cfg.p == 2 else float('inf'))
                delta.requires_grad_(True)
        info = {
            "avg_l2": delta.view(delta.size(0), -1).norm(p=2, dim=1).mean().item(),
            "avg_linf": delta.abs().view(delta.size(0), -1).max(dim=1)[0].mean().item(),
            "penalty": None,
        }
        return delta.detach(), info

    elif cfg.method == "wrm-latent":
        delta = torch.zeros_like(u0, requires_grad=True)
        g_prev = None
        delta_prev = None
        alpha_prev = cfg.wrm_alpha0
        for t in range(cfg.inner_steps):
            logits = head(u0 + delta)
            CE = F.cross_entropy(logits, y, reduction='mean')
            f_val, _ = _wrm_objective(CE, delta, cfg.gamma)
            g = torch.autograd.grad(f_val, delta, retain_graph=False, create_graph=False)[0]
            if cfg.wrm_step_rule == "fixed":
                alpha = cfg.inner_step_size
            elif cfg.wrm_step_rule == "bb":
                if t == 0 or g_prev is None or delta_prev is None:
                    alpha = alpha_prev
                else:
                    s = (delta - delta_prev).view(delta.size(0), -1)
                    yk = (g - g_prev).view(g.size(0), -1)
                    num = (s * s).sum(dim=1).mean()
                    den = (s * yk).sum(dim=1).mean().clamp(min=1e-12)
                    alpha = float(max(cfg.wrm_alpha_min, min(cfg.wrm_alpha_max, (num / den).item())))
                # Armijo safeguard
                alpha_try = alpha
                accepted = False
                for _ in range(cfg.wrm_ls_max_steps):
                    delta_try = (delta + alpha_try * g).detach().requires_grad_(True)
                    logits_try = head(u0 + delta_try)
                    CE_try = F.cross_entropy(logits_try, y, reduction='mean')
                    f_try, _ = _wrm_objective(CE_try, delta_try, cfg.gamma)
                    rhs = f_val + cfg.wrm_ls_c * alpha_try * (g.view(g.size(0), -1).pow(2).sum(dim=1).mean())
                    if f_try >= rhs:
                        accepted = True
                        break
                    alpha_try *= cfg.wrm_ls_shrink
                    if alpha_try < cfg.wrm_alpha_min:
                        break
                alpha = alpha_try if accepted else cfg.wrm_alpha_min
                alpha_prev = alpha
            else:  # armijo
                alpha = alpha_prev if t > 0 else cfg.wrm_alpha0
                for _ in range(cfg.wrm_ls_max_steps):
                    delta_try = (delta + alpha * g).detach().requires_grad_(True)
                    logits_try = head(u0 + delta_try)
                    CE_try = F.cross_entropy(logits_try, y, reduction='mean')
                    f_try, _ = _wrm_objective(CE_try, delta_try, cfg.gamma)
                    rhs = f_val + cfg.wrm_ls_c * alpha * (g.view(g.size(0), -1).pow(2).sum(dim=1).mean())
                    if f_try >= rhs:
                        break
                    alpha *= cfg.wrm_ls_shrink
                    if alpha < cfg.wrm_alpha_min:
                        break
                alpha_prev = alpha
            with torch.no_grad():
                delta_prev = delta.detach()
                g_prev = g.detach()
                delta = (delta + alpha * g).detach().requires_grad_(True)

        with torch.no_grad():
            avg_l2 = delta.view(delta.size(0), -1).norm(p=2, dim=1).mean().item()
            avg_linf = delta.abs().view(delta.size(0), -1).max(dim=1)[0].mean().item()
            penalty_val = 0.5 * cfg.gamma * (delta.view(delta.size(0), -1).pow(2).sum(dim=1)).mean().item()
        return delta.detach(), {"avg_l2": avg_l2, "avg_linf": avg_linf, "penalty": penalty_val}
    else:
        raise ValueError("inner_max_delta: method must be 'pgd-latent' or 'wrm-latent'")


# -----------------------------
# WRM γ helpers (init + online calibration)
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

def wrm_init_gamma(phi, head, loader, device, target_eps_latent: float, n_batches: int = 2) -> float:
    """
    γ₀ ≈ E[||∇_u L||₂] / target_eps_latent   (computed on clean batches).
    """
    phi.eval(); head.eval()
    batches = _take_first_batches(loader, max(1, n_batches))
    norms = []
    for x, y in batches:
        x = x.to(device)
        y = y.to(device)
        with torch.enable_grad():
            u = phi(x).detach().requires_grad_(True)
            logits = head(u)
            loss = F.cross_entropy(logits, y, reduction='mean')
            gu = torch.autograd.grad(loss, u, retain_graph=False, create_graph=False)[0]
            n = gu.view(gu.size(0), -1).norm(p=2, dim=1).mean().item()
            norms.append(n)
    if len(norms) == 0:
        return 10.0 / max(target_eps_latent, 1e-6)
    gbar = sum(norms) / len(norms)
    return max(gbar / max(target_eps_latent, 1e-8), 1e-8)

def wrm_estimate_avg_l2(phi, head, loader, device, inner_cfg: InnerConfig, n_batches: int = 1) -> float:
    phi.eval(); head.eval()
    batches = _take_first_batches(loader, max(1, n_batches))
    vals = []
    for x, y in batches:
        x = x.to(device); y = y.to(device)
        with torch.enable_grad():
            u = phi(x)
            cfg = InnerConfig(
                method="wrm-latent",
                p=inner_cfg.p, eps=inner_cfg.eps,
                pgd_steps=inner_cfg.pgd_steps, pgd_step_size=inner_cfg.pgd_step_size,
                gamma=inner_cfg.gamma, inner_steps=inner_cfg.inner_steps, inner_step_size=inner_cfg.inner_step_size,
                wrm_step_rule=inner_cfg.wrm_step_rule, wrm_alpha0=inner_cfg.wrm_alpha0,
                wrm_ls_c=inner_cfg.wrm_ls_c, wrm_ls_shrink=inner_cfg.wrm_ls_shrink,
                wrm_ls_max_steps=inner_cfg.wrm_ls_max_steps,
                wrm_alpha_min=inner_cfg.wrm_alpha_min, wrm_alpha_max=inner_cfg.wrm_alpha_max
            )
            _, info = inner_max_delta(u, head, y, cfg)
            vals.append(info["avg_l2"])
    return 0.0 if len(vals) == 0 else sum(vals) / len(vals)


# -----------------------------
# Jacobian-aware mapping helpers
# -----------------------------

@torch.no_grad()
def _num_input_dims_from_loader(loader) -> int:
    # Peek one batch to infer dimensionality d = C*H*W
    for x, _ in _take_first_batches(loader, 1):
        return int(np.prod(list(x.shape[1:])))  # C,H,W
    return 3 * 32 * 32  # fallback for CIFAR-10

def estimate_L_hat(phi, loader, device, n_batches: int = 2, power_iters: int = 1) -> float:
    """
    Estimate L_hat ≈ E||J_Φ(x)||_2 using vector-Jacobian products.
    To avoid graph reuse issues, we **rebuild the graph every iteration**:
      For t in 1..power_iters:
        - sample fresh v in latent (unit L2 per-sample)
        - compute u = Φ(x) from x that requires_grad
        - compute g = ∂(⟨u,v⟩)/∂x = J^T v
        - accumulate ||g||_2
    Averaging over both batches and 'power' iterations stabilizes the estimate.
    """
    phi.eval()
    L_vals = []
    iters = max(1, int(power_iters))
    batches = _take_first_batches(loader, max(1, n_batches))
    for x, _ in batches:
        # keep original input tensor; rebuild leaf each inner iteration
        x_init = x.to(device).detach()
        for _ in range(iters):
            x_leaf = x_init.detach().requires_grad_(True)
            with torch.enable_grad():
                u = phi(x_leaf)
                v = torch.randn_like(u)
                v = per_sample_l2_normalize(v)
                s = (u * v).view(u.size(0), -1).sum(dim=1).mean()
                g = torch.autograd.grad(s, x_leaf, retain_graph=False, create_graph=False)[0]  # g = J^T v
                Lb = g.view(g.size(0), -1).norm(p=2, dim=1).mean().item()
                L_vals.append(Lb)
    return 0.0 if len(L_vals) == 0 else (sum(L_vals) / len(L_vals))

def input_eps_to_l2(eps_pix: float, p_input, input_dim: int) -> float:
    """
    Map pixel ε (L2 or Linf) to an approximate **L2** radius in input space.
    If p=2: identity; if p=∞: ε_L2 ≈ ε * sqrt(d).
    """
    if p_input == 2:
        return float(eps_pix)
    else:
        return float(eps_pix * math.sqrt(max(1, input_dim)))

def jacobian_aware_latent_eps(phi, loader, device, inp_eps: float, p_input, n_batches=2, power_iters=1) -> Tuple[float, float]:
    """
    Returns (L_hat, eps_latent_target) using L_hat * (input eps mapped to L2).
    """
    d = _num_input_dims_from_loader(loader)
    L_hat = estimate_L_hat(phi, loader, device, n_batches=n_batches, power_iters=power_iters)
    eps_in_l2 = input_eps_to_l2(inp_eps, p_input, d)
    # eps_latent_target = L_hat * eps_in_l2
    eps_latent_target = L_hat 
    return L_hat, eps_latent_target


# -----------------------------
# Training & evaluation loops
# -----------------------------

def train_one_epoch(phi, head, loader, optimizer, device, method, inner_cfg: InnerConfig, head_only: bool):
    phi.train(not head_only)
    head.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()
        u0 = phi(x)
        if method == "erm":
            logits = head(u0)
            loss = F.cross_entropy(logits, y)
            loss.backward()
        else:
            delta_star, _ = inner_max_delta(u0.detach() if head_only else u0, head, y, inner_cfg)
            logits = head(u0 + delta_star.detach())
            if method == "wrm-latent":
                penalty = 0.5 * inner_cfg.gamma * (delta_star.view(delta_star.size(0), -1).pow(2).sum(dim=1)).mean()
                loss = F.cross_entropy(logits, y) - penalty
            else:
                loss = F.cross_entropy(logits, y)
            loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)
    return total_loss / total_samples, total_correct / total_samples

@torch.no_grad()
def evaluate(phi, head, loader, device):
    phi.eval(); head.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = head(phi(x))
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)
    return total_loss / total_samples, total_correct / total_samples

def evaluate_under_latent_attack(phi, head, loader, device, method_for_eval: str, inner_cfg: InnerConfig):
    phi.eval(); head.eval()
    total_correct = 0
    total_samples = 0
    avg_l2 = avg_linf = avg_penalty = 0.0
    n_batches = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        with torch.enable_grad():
            u0 = phi(x)
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
            delta_star, info = inner_max_delta(u0, head, y, cfg)
            logits = head(u0 + delta_star)
        with torch.no_grad():
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)
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
# Input-space PGD eval — **PIXEL SPACE**
# -----------------------------

def _random_start_input_ball_pix(x0_pix: torch.Tensor, eps: float, p) -> torch.Tensor:
    if p == 2:
        z = torch.randn_like(x0_pix)
        z = per_sample_l2_normalize(z)
        B = x0_pix.size(0)
        r = torch.rand(B, device=x0_pix.device).view(B, *([1] * (x0_pix.dim() - 1)))
        delta0 = z * (r * eps)
    else:
        delta0 = torch.empty_like(x0_pix).uniform_(-eps, eps)
    return (x0_pix + delta0).clamp(0.0, 1.0)

def evaluate_under_input_pgd(phi, head, loader, device, p, eps, steps, step_size, restarts: int = 1):
    phi.eval(); head.eval()
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
                logits = head(phi(to_normalized(x_pix)))
                loss_mean = F.cross_entropy(logits, y, reduction='mean')
                g_pix = torch.autograd.grad(loss_mean, x_pix, retain_graph=False, create_graph=False)[0]
                with torch.no_grad():
                    step = step_size * (per_sample_l2_normalize(g_pix) if p == 2 else torch.sign(g_pix))
                    x_pix = x_pix + step
                    delta_pix = x_pix - x0_pix
                    delta_pix = project_onto_lp_ball(delta_pix, eps, 2 if p == 2 else float('inf'))
                    x_pix = (x0_pix + delta_pix).clamp(0.0, 1.0)
                    x_pix.requires_grad_(True)

            with torch.no_grad():
                logits = head(phi(to_normalized(x_pix)))
                loss_vec = F.cross_entropy(logits, y, reduction='none')
                delta_pix = (x_pix - x0_pix).detach()
                mask = loss_vec > best_loss
                if mask.any():
                    best_loss[mask] = loss_vec[mask]
                    best_delta[mask] = delta_pix[mask]

        with torch.no_grad():
            x_adv_best_pix = (x0_pix + best_delta).clamp(0.0, 1.0)
            logits = head(phi(to_normalized(x_adv_best_pix)))
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
    "phase_method", "eval_attack", "cut_layer", "head_only",
    "p", "eps_latent_used", "pgd_steps", "pgd_step_size",
    "gamma", "inner_steps", "inner_step_size",
    "wrm_step_rule", "wrm_alpha0", "wrm_ls_c", "wrm_ls_shrink",
    "wrm_ls_max_steps", "wrm_alpha_min", "wrm_alpha_max",
    "batch_size", "lr", "momentum", "weight_decay", "seed",
    # Input-PGD config
    "inp_p", "inp_eps", "inp_steps", "inp_step_size", "inp_restarts",
    # Jacobian-aware
    "L_hat", "eps_latent_target",
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
    parser = argparse.ArgumentParser(
        description="Latent-space adversarial training on CIFAR-10 (ResNet-18) with PIXEL-SPACE input-PGD eval and Jacobian-aware latent ε."
    )

    # --- Training schedule (clean -> adv) ---
    parser.add_argument("--epochs-clean", type=int, default=0,
                        help="Number of initial epochs of **clean ERM** training. If >0, enables the clean->adv schedule.")
    parser.add_argument("--epochs-adv", type=int, default=0,
                        help="Number of subsequent epochs of **adversarial** training (latent). Total epochs = epochs-clean + epochs-adv.")
    parser.add_argument("--adv-method", type=str, default="wrm-latent",
                        choices=["pgd-latent", "wrm-latent"],
                        help="Adversarial method during the adversarial phase.")

    # --- Legacy controls (used only if epochs-clean=epochs-adv=0) ---
    parser.add_argument("--epochs", type=int, default=10,
                        help="(Legacy) Total epochs if no split schedule is specified.")
    parser.add_argument("--method", type=str, default="erm",
                        choices=["erm", "pgd-latent", "wrm-latent"],
                        help="(Legacy) Training method if no split schedule is specified.")

    # Model / optimization
    parser.add_argument("--cut-layer", type=str, default="layer2",
        choices=["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"],
        help="Layer where the model is split; adversary lives in this latent space.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.21)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1)

    # Inner problem config (latent)
    parser.add_argument("--p", type=str, default="inf", help="Norm for latent PGD ball: '2' or 'inf'")
    parser.add_argument("--eps", type=float, default=0.031, help="(Fallback) latent radius; overridden by Jacobian-aware mapping during adv phase.")
    parser.add_argument("--pgd-steps", type=int, default=7)
    parser.add_argument("--pgd-step-size", type=float, default=0.0, help="If <=0, auto = 2*eps/steps.")

    parser.add_argument("--gamma", type=float, default=10.0, help="(Fallback) WRM γ; may be auto-initialized when adv phase starts.")
    parser.add_argument("--inner-steps", type=int, default=7, help="Ascent steps for WRM inner maximization.")
    parser.add_argument("--inner-step-size", type=float, default=0.8, help="Used if --wrm-step-rule=fixed")

    # WRM adaptive α flags
    parser.add_argument("--wrm-step-rule", type=str, default="bb",
                        choices=["armijo", "fixed", "bb"],
                        help="Adaptive rule for WRM ascent step-size α.")
    parser.add_argument("--wrm-alpha0", type=float, default=1.0, help="Initial α guess for Armijo/BB.")
    parser.add_argument("--wrm-ls-c", type=float, default=0.1, help="Armijo c in (0,0.5).")
    parser.add_argument("--wrm-ls-shrink", type=float, default=0.5, help="Armijo backtracking factor.")
    parser.add_argument("--wrm-ls-max-steps", type=int, default=10)
    parser.add_argument("--wrm-alpha-min", type=float, default=1e-6)
    parser.add_argument("--wrm-alpha-max", type=float, default=10.0)

    # WRM γ options
    parser.add_argument("--wrm-auto-gamma", action="store_true",
                        help="Auto-initialize WRM gamma when the adversarial WRM phase begins.")
    parser.add_argument("--wrm-init-batches", type=int, default=2,
                        help="Batches used to initialize γ.")
    parser.add_argument("--wrm-calibrate-online", action="store_true",
                        help="After each WRM epoch: γ←γ*(avg_l2/eps_latent_target).")
    parser.add_argument("--wrm-calibrate-every", type=int, default=1,
                        help="Epoch interval for online γ calibration (if enabled).")
    parser.add_argument("--wrm-calibrate-batches", type=int, default=1,
                        help="Batches used for online γ calibration each time it runs.")

    parser.add_argument("--head-only", action="store_true", help="Freeze Phi and train only the Head.")
    parser.add_argument("--eval-attack", type=str, default="pgd-latent",
                        choices=["pgd-latent", "wrm-latent"],
                        help="Latent attack used for robust evaluation after each epoch.")
    parser.add_argument("--save", type=str, default="", help="Path to save final model (optional).")

    # CSV logging
    parser.add_argument("--log-csv", type=str, default="./runs_log.csv",
                        help="Path to CSV file where per-epoch results are appended.")

    # Input-space PGD eval config (test-time) — **PIXEL units**
    parser.add_argument("--inp-p", type=str, default="inf", choices=["2", "inf"],
                        help="Norm for input-space PGD at test time.")
    parser.add_argument("--inp-eps", type=float, default=8/255,
                        help="Radius for input-space PGD (in pixel units).")
    parser.add_argument("--inp-steps", type=int, default=20,
                        help="Steps for input-space PGD (set 0 to disable).")
    parser.add_argument("--inp-step-size", type=float, default=0.0,
                        help="Step size for input-space PGD (pixel units). If <=0, auto = 2*eps/steps.")
    parser.add_argument("--inp-restarts", type=int, default=5,
                        help="Random restarts for input-space PGD at test time.")

    # Jacobian-aware controls
    parser.add_argument("--jacobian-aware", action="store_true",
                        help="Enable Jacobian-aware mapping from pixel ε to latent ε_target for the adversarial phase.")
    parser.add_argument("--jacobian-batches", type=int, default=2,
                        help="Batches used to estimate L_hat.")
    parser.add_argument("--jacobian-iters", type=int, default=2,
                        help="Independent vJP samples per batch to stabilize L_hat (no graph reuse).")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    # Determine schedule
    use_split_schedule = (args.epochs_clean > 0 or args.epochs_adv > 0)
    if use_split_schedule:
        total_epochs = int(args.epochs_clean + args.epochs_adv)
        print(f"[Schedule] Clean ERM: {args.epochs_clean} -> Adversarial ({args.adv_method}): {args.epochs_adv}  (Total={total_epochs}).")
    else:
        total_epochs = int(args.epochs)
        print(f"[Schedule] Legacy mode: {args.method} for {total_epochs} epoch(s).")

    # Parse norms
    p_latent = float('inf') if str(args.p).lower() in ["inf", "linf"] else int(args.p)
    p_input  = 2 if str(args.inp_p) == "2" else float('inf')

    # Data
    trainloader, testloader = get_cifar10_loaders(batch_size=args.batch_size)

    # Model
    phi, head = build_split_resnet18(num_classes=10, cut_layer=args.cut_layer)
    phi.to(device); head.to(device)
    if args.head_only:
        for p_param in phi.parameters():
            p_param.requires_grad = False

    # Optimizer / LR
    params = list(filter(lambda t: t.requires_grad, list(phi.parameters()) + list(head.parameters())))
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, last_epoch=-1)

    # Base inner cfg (method/eps/gamma will be set at runtime)
    base_inner_cfg = InnerConfig(
        method="pgd-latent",
        p=p_latent, eps=args.eps,
        pgd_steps=args.pgd_steps, pgd_step_size=auto_pgd_step_size(p_latent, args.eps, args.pgd_steps, args.pgd_step_size),
        gamma=args.gamma, inner_steps=args.inner_steps, inner_step_size=args.inner_step_size,
        wrm_step_rule=args.wrm_step_rule, wrm_alpha0=args.wrm_alpha0,
        wrm_ls_c=args.wrm_ls_c, wrm_ls_shrink=args.wrm_ls_shrink,
        wrm_ls_max_steps=args.wrm_ls_max_steps,
        wrm_alpha_min=args.wrm_alpha_min, wrm_alpha_max=args.wrm_alpha_max
    )

    # Jacobian-aware state (populated when adversarial phase starts)
    L_hat_used: Optional[float] = None
    eps_latent_target_used: Optional[float] = None
    jacobian_ready = False

    # run_id for CSV
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    for epoch in range(1, total_epochs + 1):
        # Decide method for this epoch
        if use_split_schedule:
            current_method = "erm" if epoch <= args.epochs_clean else args.adv_method
            in_adv_phase = (epoch > args.epochs_clean) and (args.epochs_adv > 0)
        else:
            current_method = args.method
            in_adv_phase = (current_method in ["pgd-latent", "wrm-latent"])

        # If entering adversarial phase and Jacobian-aware enabled, set latent ε_target (and γ if WRM)
        if in_adv_phase and args.jacobian_aware and (not jacobian_ready):
            L_hat_used, eps_latent_target_used = jacobian_aware_latent_eps(
                phi, trainloader, device,
                inp_eps=args.inp_eps, p_input=p_input,
                n_batches=args.jacobian_batches, power_iters=args.jacobian_iters
            )
            if L_hat_used <= 0.0:
                print("[Jacobian] Warning: L_hat estimate <= 0; falling back to provided latent eps.")
                eps_latent_target_used = args.eps
            print(f"[Jacobian] L_hat ≈ {L_hat_used:.6f}, eps_input={args.inp_eps:.6f} (p={str(args.inp_p)}) "
                  f"→ eps_latent_target ≈ {eps_latent_target_used:.6f}")

            base_inner_cfg.eps = float(eps_latent_target_used)
            base_inner_cfg.pgd_step_size = auto_pgd_step_size(p_latent, base_inner_cfg.eps, base_inner_cfg.pgd_steps, args.pgd_step_size)

            if args.adv_method == "wrm-latent" and args.wrm_auto_gamma:
                gamma0 = wrm_init_gamma(phi, head, trainloader, device,
                                        target_eps_latent=base_inner_cfg.eps,
                                        n_batches=args.wrm_init_batches)
                base_inner_cfg.gamma = gamma0
                print(f"[WRM] Auto-initialized gamma to {base_inner_cfg.gamma:.6f} "
                      f"using {args.wrm_init_batches} batch(es) (target latent eps={base_inner_cfg.eps:.6f}).")
            jacobian_ready = True

        # Legacy adversarial start at epoch 1 (no split)
        if (not use_split_schedule) and in_adv_phase and args.jacobian_aware and (not jacobian_ready):
            L_hat_used, eps_latent_target_used = jacobian_aware_latent_eps(
                phi, trainloader, device,
                inp_eps=args.inp_eps, p_input=p_input,
                n_batches=args.jacobian_batches, power_iters=args.jacobian_iters
            )
            if L_hat_used <= 0.0:
                print("[Jacobian] Warning: L_hat estimate <= 0; falling back to provided latent eps.")
                eps_latent_target_used = args.eps
            print(f"[Jacobian] L_hat ≈ {L_hat_used:.6f}, eps_input={args.inp_eps:.6f} (p={str(args.inp_p)}) "
                  f"→ eps_latent_target ≈ {eps_latent_target_used:.6f}")

            base_inner_cfg.eps = float(eps_latent_target_used)
            base_inner_cfg.pgd_step_size = auto_pgd_step_size(p_latent, base_inner_cfg.eps, base_inner_cfg.pgd_steps, args.pgd_step_size)

            if current_method == "wrm-latent" and args.wrm_auto_gamma:
                gamma0 = wrm_init_gamma(phi, head, trainloader, device,
                                        target_eps_latent=base_inner_cfg.eps,
                                        n_batches=args.wrm_init_batches)
                base_inner_cfg.gamma = gamma0
                print(f"[WRM] Auto-initialized gamma to {base_inner_cfg.gamma:.6f} "
                      f"(target latent eps={base_inner_cfg.eps:.6f}).")
            jacobian_ready = True

        # Build per-epoch inner cfg
        epoch_inner_cfg = InnerConfig(
            method=current_method if current_method in ["pgd-latent", "wrm-latent"] else "pgd-latent",
            p=base_inner_cfg.p, eps=base_inner_cfg.eps,
            pgd_steps=base_inner_cfg.pgd_steps, pgd_step_size=base_inner_cfg.pgd_step_size,
            gamma=base_inner_cfg.gamma, inner_steps=base_inner_cfg.inner_steps, inner_step_size=base_inner_cfg.inner_step_size,
            wrm_step_rule=base_inner_cfg.wrm_step_rule, wrm_alpha0=base_inner_cfg.wrm_alpha0,
            wrm_ls_c=base_inner_cfg.wrm_ls_c, wrm_ls_shrink=base_inner_cfg.wrm_ls_shrink,
            wrm_ls_max_steps=base_inner_cfg.wrm_ls_max_steps,
            wrm_alpha_min=base_inner_cfg.wrm_alpha_min, wrm_alpha_max=base_inner_cfg.wrm_alpha_max
        )

        # Train + eval
        train_loss, train_acc = train_one_epoch(phi, head, trainloader, optimizer, device, current_method, epoch_inner_cfg, args.head_only)
        test_loss, test_acc = evaluate(phi, head, testloader, device)
        robust_acc, rinfo = evaluate_under_latent_attack(phi, head, testloader, device, args.eval_attack, epoch_inner_cfg)

        # γ online calibration (WRM only) against **latent target ε**
        if (current_method == "wrm-latent"
            and args.wrm_calibrate_online
            and (epoch % max(1, args.wrm_calibrate_every) == 0)):
            with torch.no_grad():
                est_avg_l2 = wrm_estimate_avg_l2(phi, head, trainloader, device, epoch_inner_cfg, n_batches=args.wrm_calibrate_batches)
            target_eps_for_gamma = base_inner_cfg.eps  # this is eps_latent_target if Jacobian-aware
            if est_avg_l2 > 0 and target_eps_for_gamma > 0:
                new_gamma = max(epoch_inner_cfg.gamma * (est_avg_l2 / target_eps_for_gamma), 1e-8)
                print(f"[WRM] Calibrate gamma: avg_l2={est_avg_l2:.6f}, target_latent_eps={target_eps_for_gamma:.6f}, "
                      f"gamma {epoch_inner_cfg.gamma:.6f} -> {new_gamma:.6f}")
                base_inner_cfg.gamma = new_gamma  # persist

        # Pixel-space PGD eval
        if args.inp_steps > 0 and args.inp_eps > 0:
            input_pgd_acc, ipgd_info = evaluate_under_input_pgd(
                phi, head, testloader, device,
                p=p_input, eps=args.inp_eps,
                steps=args.inp_steps, step_size=args.inp_step_size,
                restarts=args.inp_restarts
            )
            ipgd_l2, ipgd_linf = ipgd_info["avg_l2"], ipgd_info["avg_linf"]
        else:
            input_pgd_acc, ipgd_l2, ipgd_linf = None, None, None

        phase_str = "CLEAN/ERM" if current_method == "erm" else f"ADV/{current_method}"
        msg = f"[Epoch {epoch:02d} | {phase_str}] train {train_loss:.4f}/{train_acc*100:.2f}% | " \
              f"test {test_loss:.4f}/{test_acc*100:.2f}% | " \
              f"latent-robust({args.eval_attack}) {robust_acc*100:.2f}%"
        if rinfo["avg_penalty"] is not None:
            msg += f" | avg_penalty {rinfo['avg_penalty']:.4f}"
        msg += f" | latent avg_l2 {rinfo['avg_l2']:.3f} avg_linf {rinfo['avg_linf']:.3f}"
        if input_pgd_acc is not None:
            msg += f" | input-PGD {input_pgd_acc*100:.2f}% (R={args.inp_restarts}, L2 {ipgd_l2:.3f}, Linf {ipgd_linf:.3f})"
        if jacobian_ready:
            msg += f" | L_hat {L_hat_used:.4f} → eps_latent {base_inner_cfg.eps:.5f}"
        print(msg)

        # CSV
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
            "phase_method": current_method,
            "eval_attack": args.eval_attack,
            "cut_layer": args.cut_layer,
            "head_only": bool(args.head_only),
            "p": "inf" if p_latent == float('inf') else str(p_latent),
            "eps_latent_used": base_inner_cfg.eps,
            "pgd_steps": base_inner_cfg.pgd_steps,
            "pgd_step_size": base_inner_cfg.pgd_step_size,
            "gamma": base_inner_cfg.gamma,
            "inner_steps": args.inner_steps,
            "inner_step_size": args.inner_step_size,
            "wrm_step_rule": base_inner_cfg.wrm_step_rule,
            "wrm_alpha0": base_inner_cfg.wrm_alpha0,
            "wrm_ls_c": base_inner_cfg.wrm_ls_c,
            "wrm_ls_shrink": base_inner_cfg.wrm_ls_shrink,
            "wrm_ls_max_steps": base_inner_cfg.wrm_ls_max_steps,
            "wrm_alpha_min": base_inner_cfg.wrm_alpha_min,
            "wrm_alpha_max": base_inner_cfg.wrm_alpha_max,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "inp_p": str(args.inp_p),
            "inp_eps": args.inp_eps,
            "inp_steps": args.inp_steps,
            "inp_step_size": auto_pgd_step_size(p_input, args.inp_eps, args.inp_steps, args.inp_step_size),
            "inp_restarts": args.inp_restarts,
            "L_hat": None if not jacobian_ready else round(float(L_hat_used), 6),
            "eps_latent_target": None if not jacobian_ready else round(float(base_inner_cfg.eps), 6),
        })

        lr_scheduler.step()

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save({"phi": phi.state_dict(), "head": head.state_dict(), "args": vars(args)}, args.save)
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
