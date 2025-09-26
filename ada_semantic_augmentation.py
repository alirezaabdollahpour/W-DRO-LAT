#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Data Augmentation (ADA) in the **semantic space** per
Volpi et al., NeurIPS 2018 — "Generalizing to Unseen Domains via Adversarial Data Augmentation".

This script implements **Algorithm 1** with your exact **Φ/Head split** and CIFAR-10
pipeline style, and adds the paper’s **γ-ensembles** for classification:
- Train **one model per γ** (semantic penalty multiplier) via ADA.
- At test time, follow the paper’s rule: **pick the model with the largest max-softmax
  confidence** on each sample (clean or attacked) and use its prediction.

It also adds **PIXEL-SPACE input PGD evaluation** identical in spirit to your
reference code: attack & projection in [0,1] pixel units, auto step size
(2·ε/steps if not given), multiple random restarts, and reporting of avg L2/L∞.

No other methods are re-implemented (no latent-PGD, no WRM, no Jacobian mapping).
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import torchvision
import torchvision.transforms as T

# -----------------------------
# Constants & helpers
# -----------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_sample_l2_normalize(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    B = t.size(0)
    flat = t.view(B, -1)
    norms = flat.norm(p=2, dim=1).clamp(min=eps)
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


@torch.no_grad()
def clamp_to_valid_normalized(x_norm: torch.Tensor) -> torch.Tensor:
    """Clamp in pixel space then re-normalize, like your reference PGD eval."""
    x_pix = to_pixel(x_norm)
    x_pix = x_pix.clamp(0.0, 1.0)
    return to_normalized(x_pix)


# -----------------------------
# Import your utils-style models (same interface as your code)
# -----------------------------
from model import ResNet18 as ResNet18Plain
from model import PreActResNet18


# -----------------------------
# Split model (Phi, Head)
# -----------------------------
class HeadFromBackbone(nn.Module):
    def __init__(self, base: nn.Module, cut_layer: str):
        super().__init__()
        self.base = base
        self.cut_layer = cut_layer
        self.has_tail_bn  = hasattr(base, "bn")
        self.has_root_bn1 = hasattr(base, "bn1")

        for req in ["conv1", "layer1", "layer2", "layer3", "layer4", "linear"]:
            if not hasattr(base, req):
                raise AttributeError(f"Backbone missing '{req}'")
        if cut_layer not in {"conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"}:
            raise ValueError(f"Unsupported cut layer: {cut_layer}")

    def _finish_from_layer4(self, z: torch.Tensor) -> torch.Tensor:
        if self.has_tail_bn:
            z = F.relu(self.base.bn(z))
        z = F.avg_pool2d(z, 4)
        z = z.view(z.size(0), -1)
        if getattr(self.base, "normalize_features", False):
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-10)
        logits = self.base.linear(z)
        if getattr(self.base, "normalize_logits", False):
            logits = logits - logits.mean(dim=-1, keepdim=True)
            norms = logits.norm(dim=-1, keepdim=True)
            norms = torch.max(norms, (10.0 ** -10) * torch.ones_like(norms))
            logits = logits / norms
        if logits.shape[1] == 1:
            logits = torch.cat([torch.zeros_like(logits), logits], dim=1)
        return logits

    def _apply_linear_on_feats(self, feats: torch.Tensor) -> torch.Tensor:
        if getattr(self.base, "normalize_features", False):
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-10)
        logits = self.base.linear(feats)
        if getattr(self.base, "normalize_logits", False):
            logits = logits - logits.mean(dim=-1, keepdim=True)
            norms = logits.norm(dim=-1, keepdim=True)
            norms = torch.max(norms, (10.0 ** -10) * torch.ones_like(norms))
            logits = logits / norms
        if logits.shape[1] == 1:
            logits = torch.cat([torch.zeros_like(logits), logits], dim=1)
        return logits

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        cl = self.cut_layer
        if cl == "conv1":
            z = self.base.layer1(u)
            z = self.base.layer2(z)
            z = self.base.layer3(z)
            z = self.base.layer4(z)
            return self._finish_from_layer4(z)
        elif cl == "layer1":
            z = self.base.layer2(u); z = self.base.layer3(z); z = self.base.layer4(z)
            return self._finish_from_layer4(z)
        elif cl == "layer2":
            z = self.base.layer3(u); z = self.base.layer4(z)
            return self._finish_from_layer4(z)
        elif cl == "layer3":
            z = self.base.layer4(u)
            return self._finish_from_layer4(z)
        elif cl == "layer4":
            return self._finish_from_layer4(u)
        elif cl == "avgpool":
            if u.dim() == 4:
                u = F.avg_pool2d(u, 4)
                u = u.view(u.size(0), -1)
            return self._apply_linear_on_feats(u)
        else:
            raise ValueError(f"Unsupported cut layer: {cl}")


class PhiFromBackbone(nn.Module):
    def __init__(self, base: nn.Module, cut_layer: str):
        super().__init__()
        self.base = base
        self.cut_layer = cut_layer
        if cut_layer not in {"conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"}:
            raise ValueError(f"Unsupported cut layer: {cut_layer}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cl = self.cut_layer
        if cl == "conv1":
            if hasattr(self.base, "normalize"):
                x = self.base.normalize(x)
            x = self.base.conv1(x)
            if hasattr(self.base, "bn1"):
                x = F.relu(self.base.bn1(x))
            return x
        block_map = {"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4, "avgpool": 5}
        rb = block_map[cl]
        return self.base(x, return_features=True, return_block=rb)


def build_split_resnet18(num_classes: int = 10, cut_layer: str = "layer4", base: Optional[nn.Module] = None):
    if base is None:
        base = PreActResNet18(n_cls=num_classes, model_width=64)
    phi = PhiFromBackbone(base, cut_layer=cut_layer)
    head = HeadFromBackbone(base, cut_layer=cut_layer)
    return phi, head


# -----------------------------
# Pretrained loading (robust unwrap like your code)
# -----------------------------
def _looks_like_state_dict(obj) -> bool:
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    tensorish = 0
    total = 0
    for k, v in obj.items():
        if not isinstance(k, str):
            return False
        total += 1
        if isinstance(v, (torch.Tensor, nn.Parameter)):
            tensorish += 1
    return (tensorish >= max(1, total // 2)) and any("." in k for k in obj.keys())


def _unwrap_state_dict(maybe_sd):
    if _looks_like_state_dict(maybe_sd):
        sd = maybe_sd
    elif isinstance(maybe_sd, dict):
        priority = [
            "state_dict", "model", "net",
            "ema_state", "model_ema", "ema",
            "model_state", "model_state_dict",
            "weights", "params",
            "last", "best", "swa_last", "swa_best",
        ]
        sd = None
        for k in priority:
            if k in maybe_sd:
                cand = maybe_sd[k]
                if _looks_like_state_dict(cand):
                    sd = cand; break
                if isinstance(cand, dict) and "state_dict" in cand and _looks_like_state_dict(cand["state_dict"]):
                    sd = cand["state_dict"]; break
        if sd is None:
            best = None; best_len = -1
            for v in maybe_sd.values():
                if _looks_like_state_dict(v) and len(v) > best_len:
                    best = v; best_len = len(v)
                elif isinstance(v, dict) and "state_dict" in v and _looks_like_state_dict(v["state_dict"]):
                    if len(v["state_dict"]) > best_len:
                        best = v["state_dict"]; best_len = len(v["state_dict"])
            sd = best if best is not None else maybe_sd
    else:
        sd = maybe_sd

    cleaned = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        if k.startswith("module."):
            k2 = k[len("module."):]
        elif k.startswith("model."):
            k2 = k[len("model."):]
        else:
            k2 = k
        cleaned[k2] = v

    remapped = {}
    for k, v in cleaned.items():
        if k.startswith("fc."):
            remapped["linear." + k[len("fc."):]] = v
        else:
            remapped[k] = v
    return remapped


def load_pretrained_resnet18(pretrained_path: str, num_classes: int = 10, strict: bool = False, device: torch.device = torch.device("cpu")):
    if not pretrained_path:
        return PreActResNet18(n_cls=num_classes, model_width=64)
    ckpt = torch.load(pretrained_path, map_location=device)
    sd = _unwrap_state_dict(ckpt)

    root_keys = {k.split('.', 1)[0] for k in sd.keys()}
    if "bn1" in root_keys:
        base = ResNet18Plain(n_cls=num_classes, model_width=64, normalize_features=False, normalize_logits=False)
    elif "bn" in root_keys:
        base = PreActResNet18(n_cls=num_classes, model_width=64, normalize_features=False, normalize_logits=False)
    else:
        base = PreActResNet18(n_cls=num_classes, model_width=64, normalize_features=False, normalize_logits=False)

    base.load_state_dict(sd, strict=False)
    return base


# -----------------------------
# Data: Indexed CIFAR-10 (returns index)
# -----------------------------
class IndexedCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_cifar10_datasets():
    t_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    t_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = IndexedCIFAR10(root="./data", train=True, download=True, transform=t_train)
    testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=t_test)
    return trainset, testset


# -----------------------------
# ADA maximization: generate X^k by ascent on x with semantic penalty
# -----------------------------
@dataclass
class AscentCfg:
    steps: int = 15
    step_size: float = 1.0
    gamma: float = 1.0


def generate_adversarial_bank(
    phi: nn.Module,
    head: nn.Module,
    trainset: IndexedCIFAR10,
    device: torch.device,
    batch_size: int,
    anchor_bank: torch.Tensor,  # (N,3,32,32) normalized, CPU tensor
    ascent_cfg: AscentCfg,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (adv_images_norm_cpu[N,...], labels_cpu[N]) for a full pass over the dataset.
    The generated X^k are also used to update the anchor bank for the next round.
    """
    phi.eval(); head.eval()
    for p in phi.parameters(): p.requires_grad_(False)
    for p in head.parameters(): p.requires_grad_(False)

    loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    N = len(trainset)
    adv_bank = torch.empty_like(anchor_bank)  # normalized, CPU
    labels_bank = torch.empty(N, dtype=torch.long)

    for x_norm, y, idx in loader:
        x_norm = x_norm.to(device)
        y = y.to(device)
        idx = idx.to(device)

        anchors = anchor_bank[idx.cpu()].to(device)
        x_adv = anchors.clone().detach().requires_grad_(True)

        for _ in range(ascent_cfg.steps):
            z_adv = phi(x_adv)
            with torch.no_grad():
                z_anchor = phi(anchors)
            ce = F.cross_entropy(head(z_adv), y, reduction='mean')
            diff = (z_adv - z_anchor).view(z_adv.size(0), -1)
            c_sem = 0.5 * (diff.pow(2).sum(dim=1).mean())
            obj = ce - ascent_cfg.gamma * c_sem
            grad_x = torch.autograd.grad(obj, x_adv, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                x_adv = x_adv + ascent_cfg.step_size * grad_x
                x_adv = clamp_to_valid_normalized(x_adv)
                x_adv.requires_grad_(True)

        with torch.no_grad():
            adv_bank[idx.cpu()] = x_adv.detach().cpu()
            labels_bank[idx.cpu()] = y.detach().cpu()

    for p in phi.parameters(): p.requires_grad_(True)
    for p in head.parameters(): p.requires_grad_(True)

    return adv_bank, labels_bank


# -----------------------------
# Training (ERM) utilities
# -----------------------------
def sgd_steps(
    phi: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_steps: int,
):
    """Performs SGD steps on any loader that yields either (x, y) *or* (x, y, idx)."""
    def _xy(batch):
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("Batch does not contain (x, y) or (x, y, idx)")

    phi.train(); head.train()
    it = iter(loader)
    total = 0; correct = 0
    for step in range(1, num_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x, y = _xy(batch)
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()
        logits = head(phi(x))
        loss = F.cross_entropy(logits, y, reduction='mean')
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total += x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
        if step % max(1, num_steps // 10) == 0:
            acc = 100.0 * correct / max(1, total)
            print(f"  [ERM step {step}/{num_steps}] loss={loss.item():.4f} acc={acc:.2f}%")


def evaluate_clean(phi: nn.Module, head: nn.Module, loader: DataLoader, device: torch.device):
    phi.eval(); head.eval()
    total = 0; correct = 0; loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = head(phi(x))
            loss = F.cross_entropy(logits, y, reduction='sum')
            loss_sum += loss.item()
            total += x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
    return loss_sum / total, correct / total


# -----------------------------
# Input-space PGD evaluation (PIXEL units) — per-model and ensemble
# -----------------------------
def auto_pgd_step_size(p, eps: float, steps: int, user_step_size: float) -> float:
    if user_step_size is not None and user_step_size > 0:
        return float(user_step_size)
    steps = max(1, int(steps))
    return float(2.0 * eps / steps)


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


def _random_start_input_ball_pix(x0_pix: torch.Tensor, eps: float, p):
    if p == 2:
        z = torch.randn_like(x0_pix)
        z = per_sample_l2_normalize(z)
        B = x0_pix.size(0)
        r = torch.rand(B, device=x0_pix.device).view(B, *([1] * (x0_pix.dim() - 1)))
        delta0 = z * (r * eps)
    else:
        delta0 = torch.empty_like(x0_pix).uniform_(-eps, eps)
    return (x0_pix + delta0).clamp(0.0, 1.0)


def evaluate_under_input_pgd(phi: nn.Module, head: nn.Module, loader: DataLoader, device: torch.device,
                             p, eps, steps, step_size, restarts: int = 1):
    """Identical structure to your reference implementation (per-model)."""
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


def evaluate_clean_ensemble(models: List[Tuple[nn.Module, nn.Module]], loader: DataLoader, device: torch.device):
    """Max-softmax selection across models on clean inputs."""
    total = 0; correct = 0; loss_sum = 0.0
    for phi, head in models:
        phi.eval(); head.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits_list = [head(phi(x)) for (phi, head) in models]
            confs = [F.softmax(logits, dim=1).max(dim=1).values for logits in logits_list]  # (M*[B])
            confs_stack = torch.stack(confs, dim=0)  # (M,B)
            sel = confs_stack.argmax(dim=0)  # (B,)
            preds = torch.empty_like(y)
            for m_idx, logits in enumerate(logits_list):
                mask = (sel == m_idx)
                if mask.any():
                    preds[mask] = logits.argmax(dim=1)[mask]
            loss_sum += sum(F.cross_entropy(logits, y, reduction='sum').item() for logits in logits_list) / len(models)
            total += x.size(0)
            correct += (preds == y).sum().item()
    return loss_sum / max(1, total), correct / max(1, total)


def evaluate_under_input_pgd_ensemble(models: List[Tuple[nn.Module, nn.Module]], loader: DataLoader, device: torch.device,
                                      p, eps, steps, step_size, restarts: int = 1):
    """Pixel-space PGD **against the selection rule** (white-box approx)."""
    for phi, head in models:
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
                logits_list = [head(phi(to_normalized(x_pix))) for (phi, head) in models]
                confs = [F.softmax(logits, dim=1).max(dim=1).values for logits in logits_list]
                confs_stack = torch.stack(confs, dim=0)
                sel = confs_stack.argmax(dim=0)  # (B,)
                losses_stack = torch.stack([F.cross_entropy(logits, y, reduction='none') for logits in logits_list], dim=0)  # (M,B)
                loss_vec = losses_stack.gather(dim=0, index=sel.unsqueeze(0)).squeeze(0)  # (B,)
                loss_mean = loss_vec.mean()
                g_pix = torch.autograd.grad(loss_mean, x_pix, retain_graph=False, create_graph=False)[0]
                with torch.no_grad():
                    step = step_size * (per_sample_l2_normalize(g_pix) if p == 2 else torch.sign(g_pix))
                    x_pix = x_pix + step
                    delta_pix = x_pix - x0_pix
                    delta_pix = project_onto_lp_ball(delta_pix, eps, 2 if p == 2 else float('inf'))
                    x_pix = (x0_pix + delta_pix).clamp(0.0, 1.0)
                    x_pix.requires_grad_(True)

            with torch.no_grad():
                logits_list = [head(phi(to_normalized(x_pix))) for (phi, head) in models]
                losses_stack = torch.stack([F.cross_entropy(logits, y, reduction='none') for logits in logits_list], dim=0)
                confs = [F.softmax(logits, dim=1).max(dim=1).values for logits in logits_list]
                confs_stack = torch.stack(confs, dim=0)
                sel = confs_stack.argmax(dim=0)
                loss_vec = losses_stack.gather(dim=0, index=sel.unsqueeze(0)).squeeze(0)
                delta_pix = (x_pix - x0_pix).detach()
                mask = loss_vec > best_loss
                if mask.any():
                    best_loss[mask] = loss_vec[mask]
                    best_delta[mask] = delta_pix[mask]

        with torch.no_grad():
            x_adv_best_pix = (x0_pix + best_delta).clamp(0.0, 1.0)
            logits_list = [head(phi(to_normalized(x_adv_best_pix))) for (phi, head) in models]
            confs = [F.softmax(logits, dim=1).max(dim=1).values for logits in logits_list]
            confs_stack = torch.stack(confs, dim=0)
            sel = confs_stack.argmax(dim=0)
            preds = torch.empty_like(y)
            for m_idx, logits in enumerate(logits_list):
                mask = (sel == m_idx)
                if mask.any():
                    preds[mask] = logits.argmax(dim=1)[mask]
            total_correct += (preds == y).sum().item()
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
# Main ADA loop (Algorithm 1) + γ-ensembles
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Semantic-space ADA (Volpi et al. 2018) with Φ/Head split and γ-ensembles")

    # Pretrained
    p.add_argument("--pretrained-path", type=str, default="", help="Path to pretrained CIFAR-10 checkpoint")
    p.add_argument("--pretrained-strict", action="store_true")

    # Model & optimization
    p.add_argument("--cut-layer", type=str, default="layer4", choices=["conv1","layer1","layer2","layer3","layer4","avgpool"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)

    # ADA hyperparameters (Algorithm 1)
    p.add_argument("--K", type=int, default=1, help="Number of ADA outer rounds")
    p.add_argument("--gamma", type=float, default=1.0, help="Semantic penalty multiplier (used if --gamma-list not set)")
    p.add_argument("--gamma-list", type=str, default="", help="Comma-separated list of gammas for γ-ensemble, e.g. '0.5,1.0,2.0'")
    p.add_argument("--tmin", type=int, default=0, help="Initial ERM steps before ADA (T_min in paper)")
    p.add_argument("--t", type=int, default=2000, help="ERM steps after each append (T in paper)")
    p.add_argument("--ascent-steps", type=int, default=15, help="Inner ascent steps (T_max in paper)")
    p.add_argument("--ascent-step-size", type=float, default=1.0, help="Inner ascent step size (eta in paper)")

    # Input-PGD eval config (test-time) — PIXEL units
    p.add_argument("--inp-p", type=str, default="inf", choices=["2", "inf"], help="Norm for input-space PGD at test time")
    p.add_argument("--inp-eps", type=float, default=8/255, help="Radius for input-space PGD (pixel units)")
    p.add_argument("--inp-steps", type=int, default=20, help="Steps for input-space PGD (set 0 to disable)")
    p.add_argument("--inp-step-size", type=float, default=0.0, help="Step size (pixel units). If <=0, auto = 2*eps/steps")
    p.add_argument("--inp-restarts", type=int, default=5, help="Random restarts for input-space PGD")

    # Training epochs wrapper (optional convenience)
    p.add_argument("--epochs", type=int, default=0, help="If >0, run this many full-epoch ERM passes after final ADA")

    # Misc
    p.add_argument("--save", type=str, default="", help="Optional path to save final ensemble state")
    return p.parse_args()


def train_single_gamma(gamma: float, base_ckpt_path: str, cut_layer: str, batch_size: int, lr: float, momentum: float,
                        weight_decay: float, K: int, tmin: int, t: int, ascent_steps: int, ascent_step_size: float,
                        device: torch.device):
    # Data (shared dataset; separate loaders as needed)
    trainset, _ = get_cifar10_datasets()
    train_loader_plain = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Build model from pretrained
    base = load_pretrained_resnet18(base_ckpt_path, num_classes=10, strict=False, device=device)
    phi, head = build_split_resnet18(num_classes=10, cut_layer=cut_layer, base=base)
    phi.to(device); head.to(device)

    optimizer = optim.SGD(list(filter(lambda t: t.requires_grad, list(phi.parameters()) + list(head.parameters()))),
                          lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    # Warmup (ERM on original data)
    if tmin > 0:
        print(f"[Warmup γ={gamma}] {tmin} ERM steps on original data …")
        sgd_steps(phi, head, train_loader_plain, optimizer, device, num_steps=tmin)

    # Prepare anchor bank (original images)
    print(f"[ADA γ={gamma}] Preparing anchor bank …")
    N = len(trainset)
    anchor_bank = torch.empty((N, 3, 32, 32), dtype=torch.float32)
    labels_all = torch.empty(N, dtype=torch.long)
    with torch.no_grad():
        for x, y, idx in DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True):
            anchor_bank[idx] = x
            labels_all[idx] = y

    ascent_cfg = AscentCfg(steps=ascent_steps, step_size=ascent_step_size, gamma=gamma)
    memory_adv_list: List[torch.Tensor] = []
    memory_lbl_list: List[torch.Tensor] = []

    for k in range(1, K + 1):
        print(f"[ADA γ={gamma}] Round {k}/{K}: generating adversarial set …")
        adv_bank_k, labels_k = generate_adversarial_bank(phi, head, trainset, device, batch_size, anchor_bank, ascent_cfg)
        anchor_bank = adv_bank_k.clone()  # update anchors for next round
        memory_adv_list.append(adv_bank_k)
        memory_lbl_list.append(labels_k)

        aug_loader = DataLoader(
            ConcatDataset([
                TensorDataset(anchor_bank, labels_all),
                TensorDataset(torch.cat(memory_adv_list, dim=0), torch.cat(memory_lbl_list, dim=0))
            ]),
            batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        size_aug = len(anchor_bank) + sum(m.size(0) for m in memory_adv_list)
        print(f"[ADA γ={gamma}] Minimization — {t} ERM steps on augmented set (size={size_aug}) …")
        sgd_steps(phi, head, aug_loader, optimizer, device, num_steps=t)

    return phi, head


def main():
    args = parse_args()
    device = get_device()
    print("Using device:", device)

    # Test loader shared
    _, testset = get_cifar10_datasets()
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Parse gammas
    gamma_list = []
    if args.gamma_list.strip():
        for s in args.gamma_list.split(','):
            s = s.strip()
            if s:
                gamma_list.append(float(s))
    if not gamma_list:
        gamma_list = [float(args.gamma)]
    print(f"Training γ-ensemble with gammas: {gamma_list}")

    # Train a model per gamma
    ensemble: List[Tuple[nn.Module, nn.Module]] = []
    for g in gamma_list:
        phi, head = train_single_gamma(
            gamma=g,
            base_ckpt_path=args.pretrained_path,
            cut_layer=args.cut_layer,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            K=args.K,
            tmin=args.tmin,
            t=args.t,
            ascent_steps=args.ascent_steps,
            ascent_step_size=args.ascent_step_size,
            device=device,
        )
        ensemble.append((phi, head))

    # ----- Evaluation -----
    print("=== Clean accuracy (per model) ===")
    for g, (phi, head) in zip(gamma_list, ensemble):
        loss, acc = evaluate_clean(phi, head, test_loader, device)
        print(f"  γ={g:g}: clean acc = {acc*100:.2f}% (loss {loss:.4f})")

    print("=== Clean accuracy (γ-ensemble, max-softmax selection) ===")
    loss_e, acc_e = evaluate_clean_ensemble(ensemble, test_loader, device)
    print(f"  Ensemble: clean acc = {acc_e*100:.2f}% (avg loss {loss_e:.4f})")

    # Pixel-space PGD evals
    if args.inp_steps > 0 and args.inp_eps > 0:
        p_input = 2 if str(args.inp_p) == "2" else float('inf')
        print("=== Input-space PGD (per model) ===")
        for g, (phi, head) in zip(gamma_list, ensemble):
            acc, info = evaluate_under_input_pgd(
                phi, head, test_loader, device,
                p=p_input, eps=args.inp_eps,
                steps=args.inp_steps, step_size=args.inp_step_size,
                restarts=args.inp_restarts,
            )
            print(f"  γ={g:g}: PGD acc = {acc*100:.2f}% | L2 {info['avg_l2']:.4f} | L∞ {info['avg_linf']:.4f}")

        print("=== Input-space PGD (γ-ensemble, attacks follow selection rule) ===")
        acc, info = evaluate_under_input_pgd_ensemble(
            ensemble, test_loader, device,
            p=p_input, eps=args.inp_eps,
            steps=args.inp_steps, step_size=args.inp_step_size,
            restarts=args.inp_restarts,
        )
        print(f"  Ensemble: PGD acc = {acc*100:.2f}% | L2 {info['avg_l2']:.4f} | L∞ {info['avg_linf']:.4f}")

    # Save ensemble
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save({
            "gammas": gamma_list,
            "models": [{
                "phi": phi.state_dict(),
                "head": head.state_dict(),
            } for (phi, head) in ensemble],
            "args": vars(args),
        }, args.save)
        print(f"Saved ensemble checkpoint to {args.save}")


if __name__ == "__main__":
    main()
