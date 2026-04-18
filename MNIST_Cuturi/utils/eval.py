"""Evaluation helpers: clean / PGD / PGD-sweep / MNIST-C."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.common import accuracy, cross_entropy_loss
from utils.pgd import pgd_l2_attack_restarts

if TYPE_CHECKING:
    from algorithms.base import TrainState
    from config import PGDEvalConfig


def evaluate_clean(
    state: "TrainState",
    dataset,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = state.model
    was_training = model.training
    model.eval()
    total_loss = total_acc = total_n = 0.0
    try:
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                if x.size(0) == 0:
                    continue
                logits = model(x)
                loss = cross_entropy_loss(logits, y).item()
                acc = accuracy(logits, y).item()
                n = x.size(0)
                total_loss += loss * n
                total_acc += acc * n
                total_n += n
    finally:
        model.train(was_training)
    return {"loss": total_loss / total_n, "acc": total_acc / total_n}


def evaluate_pgd(
    state: "TrainState",
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    eps: float = 0.3,
    step_size: Optional[float] = None,
    num_steps: int = 40,
    restarts: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Adversarial accuracy under PGD-L2 with random restarts."""
    device = device or next(state.model.parameters()).device
    _step = (
        step_size
        if (step_size is not None and step_size > 0)
        else 2.0 * eps / max(num_steps, 1)
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_correct = 0
    total_l2 = total_linf = 0.0
    total_n = 0

    was_training = state.model.training
    state.model.eval()
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if x.size(0) == 0:
                continue
            adv_x = pgd_l2_attack_restarts(
                state.model, x, y, eps, num_steps, _step, restarts
            )
            with torch.no_grad():
                logits = state.model(adv_x)
                total_correct += (logits.argmax(dim=1) == y).sum().item()
            n = x.size(0)
            total_n += n
            diff = (adv_x - x).detach().view(n, -1)
            total_l2 += diff.norm(p=2, dim=1).sum().item()
            total_linf += diff.abs().max(dim=1).values.sum().item()
    finally:
        state.model.train(was_training)

    return {
        "acc": total_correct / max(1, total_n),
        "avg_l2": total_l2 / max(1, total_n),
        "avg_linf": total_linf / max(1, total_n),
    }


def evaluate_pgd_l2_sweep(
    state: "TrainState",
    dataset: torch.utils.data.Dataset,
    pgd_cfg: "PGDEvalConfig",
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model robustness under PGD-L2 for each epsilon in pgd_cfg."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = state.model.training
    state.model.eval()
    results: Dict[str, Any] = {}

    try:
        for eps in pgd_cfg.epsilons:
            step_size = (
                pgd_cfg.step_size
                if pgd_cfg.step_size is not None and pgd_cfg.step_size > 0
                else 2.0 * eps / max(pgd_cfg.num_steps, 1)
            )

            total_correct = 0
            total_n = 0
            total_l2 = 0.0
            total_linf = 0.0

            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                if xb.size(0) == 0:
                    continue
                adv_x = pgd_l2_attack_restarts(
                    state.model,
                    xb,
                    yb,
                    eps,
                    pgd_cfg.num_steps,
                    step_size,
                    pgd_cfg.restarts,
                )
                with torch.no_grad():
                    logits = state.model(adv_x)
                    total_correct += (logits.argmax(dim=1) == yb).sum().item()
                    n = xb.size(0)
                    total_n += n
                    delta = (adv_x - xb).detach().view(n, -1)
                    total_l2 += delta.norm(p=2, dim=1).sum().item()
                    total_linf += delta.abs().max(dim=1).values.sum().item()

            acc = total_correct / max(1, total_n)
            avg_l2 = total_l2 / max(1, total_n)
            avg_linf = total_linf / max(1, total_n)

            eps_key = f"eps_{eps:.4g}"
            results[eps_key] = {
                "epsilon": eps,
                "step_size": step_size,
                "num_steps": pgd_cfg.num_steps,
                "restarts": pgd_cfg.restarts,
                "acc_pct": round(acc * 100, 2),
                "avg_l2": round(avg_l2, 4),
                "avg_linf": round(avg_linf, 4),
                "samples": total_n,
            }
            print(
                f"    eps={eps:.4g}: acc={acc * 100:.2f}%"
                f" avg_L2={avg_l2:.4f} avg_Linf={avg_linf:.4f}"
            )
    finally:
        state.model.train(was_training)

    return results


def evaluate_mnist_c(
    model: nn.Module,
    device: torch.device,
    root: str = "./data",
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Evaluate a model on every MNIST-C corruption."""
    from MNIST_C_utils import MNISTCDataset, CORRUPTIONS, ensure_mnist_c_downloaded

    ensure_mnist_c_downloaded(root)
    was_training = model.training
    model.eval()
    results: Dict[str, float] = {}

    try:
        with torch.no_grad():
            for corruption in CORRUPTIONS:
                dataset = MNISTCDataset(root=root, corruption=corruption, train=False)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                correct = 0
                total = 0
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    logits = model(data)
                    correct += (logits.argmax(dim=1) == target).sum().item()
                    total += target.size(0)
                results[corruption] = round(100.0 * correct / total, 2)
    finally:
        model.train(was_training)

    ood_accs = [acc for corr, acc in results.items() if corr != "identity"]
    results["avg_ood"] = round(sum(ood_accs) / len(ood_accs), 2)
    return results
