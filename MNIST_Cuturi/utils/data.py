"""MNIST loading, val split, and validation scoring."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets

from utils.eval import evaluate_clean, evaluate_pgd

if TYPE_CHECKING:
    from algorithms.base import TrainState
    from config import TrainConfig


def load_mnist():
    train_raw = datasets.MNIST(root="data", train=True, download=True)
    test_raw = datasets.MNIST(root="data", train=False, download=True)

    def to_tensor_dataset(ds):
        x = ds.data.unsqueeze(1).float().div(255.0)
        y = ds.targets
        return TensorDataset(x, y)

    return to_tensor_dataset(train_raw), to_tensor_dataset(test_raw)


def split_train_val(
    dataset: torch.utils.data.Dataset,
    val_frac: float,
    seed: int,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Deterministic train/val split using an independent generator."""
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")
    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=gen)


def compute_val_score(
    state: "TrainState",
    val_ds: torch.utils.data.Dataset,
    cfg: "TrainConfig",
    device: torch.device,
    is_clean_phase: bool,
) -> Tuple[float, float, float]:
    """Score a checkpoint on the held-out validation split.

    During clean warm-up returns val clean accuracy.
    During adversarial epochs returns
        score = α · val_clean_acc + (1 − α) · val_pgd_acc, α = cfg.es_clean_weight.
    """
    clean_m = evaluate_clean(state, val_ds, cfg.batch_size, device)
    val_clean_acc = clean_m["acc"]

    if is_clean_phase:
        return val_clean_acc, val_clean_acc, 0.0

    pgd_m = evaluate_pgd(
        state,
        val_ds,
        cfg.batch_size,
        eps=cfg.es_pgd_eps,
        num_steps=cfg.es_pgd_steps,
        restarts=cfg.es_pgd_restarts,
        device=device,
    )
    val_pgd_acc = pgd_m["acc"]
    alpha = cfg.es_clean_weight
    score = alpha * val_clean_acc + (1.0 - alpha) * val_pgd_acc
    return score, val_clean_acc, val_pgd_acc
