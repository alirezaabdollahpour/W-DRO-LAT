"""Common utilities shared across LAT training and evaluation scripts."""

from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise default CPU device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_deterministic(seed: int = 1) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for deterministic experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def per_sample_l2_normalize(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize each sample in a batch to unit L2 norm."""
    B = t.size(0)
    flat = t.view(B, -1)
    norms = flat.norm(p=2, dim=1).clamp(min=eps)
    reshape = (B,) + (1,) * (t.dim() - 1)
    return t / norms.view(reshape)


def to_pixel(x_norm: torch.Tensor) -> torch.Tensor:
    """Convert normalized inputs back to pixel space."""
    mean = torch.tensor(CIFAR10_MEAN, device=x_norm.device, dtype=x_norm.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=x_norm.device, dtype=x_norm.dtype).view(1, 3, 1, 1)
    return (x_norm * std) + mean


def to_normalized(x_pix: torch.Tensor) -> torch.Tensor:
    """Normalize pixel-space inputs using CIFAR-10 statistics."""
    mean = torch.tensor(CIFAR10_MEAN, device=x_pix.device, dtype=x_pix.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=x_pix.device, dtype=x_pix.dtype).view(1, 3, 1, 1)
    return (x_pix - mean) / std


def auto_pgd_step_size(p, eps: float, steps: int, user_step_size: Optional[float]) -> float:
    """Return user step size when provided, otherwise a heuristic default."""
    if user_step_size is not None and user_step_size > 0:
        return float(user_step_size)
    steps = max(1, int(steps))
    return float(2.0 * eps / steps)


def project_onto_lp_ball(delta: torch.Tensor, eps: float, p: Union[int, float, str]) -> torch.Tensor:
    """Project adversarial perturbations onto an L2 or Linf ball."""
    if p == 2:
        flat = delta.view(delta.size(0), -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        factors = (eps / norms).clamp(max=1.0)
        flat = flat * factors
        return flat.view_as(delta)
    if p in {float("inf"), "inf"}:
        return delta.clamp(min=-eps, max=eps)
    raise ValueError("Only L2 and Linf norms are supported for projection.")


def looks_like_state_dict(obj: Any) -> bool:
    """Heuristically determine whether an object resembles a state dict."""
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


def unwrap_state_dict(maybe_sd: Any) -> Dict[str, torch.Tensor]:
    """Strip common wrappers and prefixes from checkpoints to obtain a state dict."""
    if looks_like_state_dict(maybe_sd):
        sd = maybe_sd
    elif isinstance(maybe_sd, dict):
        priority = [
            "state_dict",
            "model",
            "net",
            "ema_state",
            "model_ema",
            "ema",
            "model_state",
            "model_state_dict",
            "weights",
            "params",
            "last",
            "best",
            "swa_last",
            "swa_best",
        ]
        sd = None
        for key in priority:
            if key in maybe_sd:
                candidate = maybe_sd[key]
                if looks_like_state_dict(candidate):
                    sd = candidate
                    break
                if isinstance(candidate, dict) and "state_dict" in candidate:
                    nested = candidate["state_dict"]
                    if looks_like_state_dict(nested):
                        sd = nested
                        break
        if sd is None:
            best = None
            best_len = -1
            for value in maybe_sd.values():
                if looks_like_state_dict(value) and len(value) > best_len:
                    best = value
                    best_len = len(value)
                elif isinstance(value, dict) and "state_dict" in value:
                    nested = value["state_dict"]
                    if looks_like_state_dict(nested) and len(nested) > best_len:
                        best = nested
                        best_len = len(nested)
            sd = best if sd is None else sd
        if sd is None:
            sd = maybe_sd
    else:
        sd = maybe_sd

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if not isinstance(key, str):
            continue
        if key.startswith("module."):
            key = key[len("module."):]
        elif key.startswith("model."):
            key = key[len("model."):]
        cleaned[key] = value

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in cleaned.items():
        if key.startswith("fc."):
            remapped["linear." + key[len("fc."):]] = value
        else:
            remapped[key] = value
    return remapped


def random_start_input_ball_pix(x0_pix: torch.Tensor, eps: float, p: Union[int, float, str]) -> torch.Tensor:
    """Sample a random starting point inside an input-space Lp ball (pixel units)."""
    if p == 2:
        z = torch.randn_like(x0_pix)
        z = per_sample_l2_normalize(z)
        batch = x0_pix.size(0)
        r = torch.rand(batch, device=x0_pix.device).view(batch, *([1] * (x0_pix.dim() - 1)))
        delta0 = z * (r * eps)
    else:
        delta0 = torch.empty_like(x0_pix).uniform_(-eps, eps)
    return (x0_pix + delta0).clamp(0.0, 1.0)


def evaluate_under_input_pgd(
    phi: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    p,
    eps: float,
    steps: int,
    step_size: Optional[float],
    restarts: int = 1,
):
    """Run input-space PGD in pixel units and report accuracy and perturbation stats."""
    phi.eval()
    head.eval()
    total_correct = 0
    total = 0
    avg_l2 = 0.0
    avg_linf = 0.0
    n_batches = 0

    step_size = auto_pgd_step_size(p, eps, steps, step_size)
    restarts = max(1, int(restarts))

    total_batches = len(loader) if hasattr(loader, "__len__") else None
    progress = tqdm(loader, desc="Input-PGD", leave=False, total=total_batches)
    for x_norm, y in progress:
        x_norm = x_norm.to(device)
        y = y.to(device)
        x0_pix = to_pixel(x_norm).detach()

        best_delta = torch.zeros_like(x0_pix)
        best_loss = torch.full((x0_pix.size(0),), -1e9, device=device)

        for _ in range(restarts):
            x_pix = random_start_input_ball_pix(x0_pix, eps, p).detach().requires_grad_(True)
            for _ in range(steps):
                logits = head(phi(to_normalized(x_pix)))
                loss_mean = F.cross_entropy(logits, y, reduction="mean")
                grad_pix = torch.autograd.grad(loss_mean, x_pix, retain_graph=False, create_graph=False)[0]
                with torch.no_grad():
                    step = step_size * (per_sample_l2_normalize(grad_pix) if p == 2 else torch.sign(grad_pix))
                    x_pix = x_pix + step
                    delta_pix = x_pix - x0_pix
                    delta_pix = project_onto_lp_ball(delta_pix, eps, 2 if p == 2 else float("inf"))
                    x_pix = (x0_pix + delta_pix).clamp(0.0, 1.0)
                    x_pix.requires_grad_(True)

            with torch.no_grad():
                logits = head(phi(to_normalized(x_pix)))
                losses = F.cross_entropy(logits, y, reduction="none")
                delta_pix = (x_pix - x0_pix).detach()
                mask = losses > best_loss
                if mask.any():
                    best_loss[mask] = losses[mask]
                    best_delta[mask] = delta_pix[mask]

        with torch.no_grad():
            x_adv_best_pix = (x0_pix + best_delta).clamp(0.0, 1.0)
            logits = head(phi(to_normalized(x_adv_best_pix)))
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x_norm.size(0)
            delta = best_delta
            avg_l2 += delta.view(delta.size(0), -1).norm(p=2, dim=1).mean().item()
            avg_linf += delta.abs().view(delta.size(0), -1).max(dim=1)[0].mean().item()
            n_batches += 1

    dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else total
    if hasattr(loader, "dataset") and total != dataset_size:
        raise RuntimeError(
            f"evaluate_under_input_pgd consumed {total} samples but dataset has {dataset_size}."
            " Ensure DataLoader is not dropping samples."
        )
    accuracy = total_correct / max(1, dataset_size)
    avg_l2 /= max(1, n_batches)
    avg_linf /= max(1, n_batches)
    progress.close()
    return accuracy, {"avg_l2": avg_l2, "avg_linf": avg_linf}


def dataloader_seed(base_seed: int, offset: int = 0) -> Tuple[torch.Generator, Callable[[int], None]]:
    """Return deterministic generator/worker init pair for DataLoader seeding."""
    seed = int((base_seed + offset) % (2 ** 32))
    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init_fn(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return generator, worker_init_fn


def _cifar10_transform(train: bool, augment_train: bool):
    import torchvision.transforms as T

    transforms = []
    if train and augment_train:
        transforms.extend([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    return T.Compose(transforms)


def get_cifar10_loader(
    split: str,
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str = "./data",
    augment_train: bool = True,
    shuffle_train: bool = True,
    pin_memory: bool = True,
    download: bool = True,
    seed: Optional[int] = None,
):
    """Return a CIFAR-10 dataloader with consistent normalization."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    import torchvision

    is_train = split == "train"
    transform = _cifar10_transform(is_train, augment_train)
    dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=is_train,
        download=download,
        transform=transform,
    )
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator, worker_init_fn = dataloader_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train and shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    return loader, len(dataset)


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str = "./data",
    augment_train: bool = True,
    shuffle_train: bool = True,
    pin_memory: bool = True,
    download: bool = True,
    seed: Optional[int] = None,
):
    """Return train/test CIFAR-10 loaders that share normalization."""
    train_seed = seed if seed is not None else None
    test_seed = (seed + 1) if seed is not None else None
    train_loader, _ = get_cifar10_loader(
        "train",
        batch_size=batch_size,
        num_workers=num_workers,
        data_root=data_root,
        augment_train=augment_train,
        shuffle_train=shuffle_train,
        pin_memory=pin_memory,
        download=download,
        seed=train_seed,
    )
    test_loader, _ = get_cifar10_loader(
        "test",
        batch_size=batch_size,
        num_workers=num_workers,
        data_root=data_root,
        augment_train=augment_train,
        shuffle_train=shuffle_train,
        pin_memory=pin_memory,
        download=download,
        seed=test_seed,
    )
    return train_loader, test_loader


__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "get_device",
    "set_deterministic",
    "per_sample_l2_normalize",
    "to_pixel",
    "to_normalized",
    "auto_pgd_step_size",
    "project_onto_lp_ball",
    "looks_like_state_dict",
    "unwrap_state_dict",
    "random_start_input_ball_pix",
    "evaluate_under_input_pgd",
    "dataloader_seed",
    "get_cifar10_loader",
    "get_cifar10_loaders",
]
