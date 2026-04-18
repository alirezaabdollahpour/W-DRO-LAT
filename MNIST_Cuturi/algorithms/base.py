"""Shared training primitives: TrainState, clean step, generic adv loop."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainConfig
from models.classifier import CarliniWagnerMNIST
from utils.common import (
    accuracy,
    cross_entropy_loss,
    seed_everything,
    set_requires_grad,
)
from utils.data import compute_val_score, load_mnist, split_train_val
from utils.eval import evaluate_clean
from utils.logging import CSVLogger, DEFAULT_LOG_PATH, LOG_FIELDNAMES


@dataclass
class TrainState:
    model: nn.Module
    opt: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None


def create_classifier_state(cfg: TrainConfig, device: torch.device) -> TrainState:
    model = CarliniWagnerMNIST().to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=cfg.lr_cls, momentum=0.9, nesterov=True,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=[cfg.lr_cls_drop_epoch],
        gamma=cfg.lr_cls_drop_factor,
    )
    return TrainState(model=model, opt=opt, scheduler=scheduler)


def train_step_clean(
    state: TrainState, batch, device: torch.device
) -> Tuple[TrainState, Dict[str, torch.Tensor]]:
    """Standard clean training step (no adversarial perturbation)."""
    model = state.model
    opt = state.opt
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    if x.size(0) == 0:
        zero = torch.tensor(0.0, device=device)
        return state, {"loss": zero, "acc_clean": zero}

    model.train()
    with torch.enable_grad():
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_post = model(x)
        acc = accuracy(logits_post, y)
    return state, {"loss": loss.detach(), "acc_clean": acc}


def max_steps_from_cfg(cfg: TrainConfig, algorithm_key: str) -> Optional[int]:
    attr = f"max_steps_{algorithm_key}"
    return getattr(cfg, attr, None)


def save_training_results_json(file_name: str, payload: Dict[str, Any]) -> None:
    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", file_name), "w") as f:
        json.dump(payload, f, indent=2)


def train_algorithm_generic_adv(
    algorithm_key: str,
    display_name: str,
    cfg: TrainConfig,
    device: torch.device,
    train_step_fn: Callable[
        [TrainState, Any, TrainConfig, torch.device],
        Tuple[TrainState, Dict[str, torch.Tensor]],
    ],
) -> Tuple[TrainState, Dict[str, Any]]:
    """Generic train/validate/test loop for adv-trained baselines (dual/wgf/wfr/svg/rgo)."""
    seed_everything(cfg.seed)
    train_raw_ds, test_ds = load_mnist()
    train_ds, val_ds = split_train_val(train_raw_ds, cfg.es_val_frac, cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    state = create_classifier_state(cfg, device)
    logger = CSVLogger(DEFAULT_LOG_PATH, LOG_FIELDNAMES)
    training_logs: List[Dict[str, Any]] = []

    best_score = -float("inf")
    best_epoch = -1
    best_model_sd = None
    best_opt_sd = None
    patience_count = 0
    in_adv_phase = False

    max_steps = max_steps_from_cfg(cfg, algorithm_key)

    for epoch in range(cfg.num_epochs):
        is_clean = epoch < cfg.epoch_clean
        num_steps = 0
        epoch_n = 0

        if not is_clean and not in_adv_phase:
            in_adv_phase = True
            patience_count = 0

        if is_clean:
            epoch_loss_sum = 0.0
            epoch_acc_clean_sum = 0.0
            for step, (x, y) in enumerate(train_loader):
                if max_steps is not None and step >= max_steps:
                    break
                n = x.size(0)
                state, metrics = train_step_clean(state, (x, y), device)
                epoch_loss_sum += float(metrics["loss"]) * n
                epoch_acc_clean_sum += float(metrics["acc_clean"]) * n
                epoch_n += n
                num_steps += 1
            if num_steps > 0:
                avg_loss = epoch_loss_sum / epoch_n
                avg_acc_clean = epoch_acc_clean_sum / epoch_n
                print(
                    f"[{display_name}] Epoch {epoch} (clean) "
                    f"loss={avg_loss:.4f} acc_clean={avg_acc_clean:.4f}"
                )
                logger.log(
                    algorithm=algorithm_key, phase="train_clean", epoch=epoch,
                    step=num_steps, loss_adv=avg_loss, adv_loss=None,
                    cls_loss=None, acc_clean=avg_acc_clean, acc_adv=None,
                    w2_proxy=None,
                )
                training_logs.append({
                    "epoch": epoch, "phase": "clean", "steps": num_steps,
                    "loss": avg_loss, "acc_clean": avg_acc_clean,
                })
                score, val_c, _ = compute_val_score(
                    state, val_ds, cfg, device, is_clean_phase=True
                )
                print(
                    f"[{display_name}] Epoch {epoch}  "
                    f"val_clean={val_c:.4f}  score={score:.4f}"
                )
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_model_sd = copy.deepcopy(state.model.state_dict())
                    best_opt_sd = copy.deepcopy(state.opt.state_dict())
            if state.scheduler is not None:
                state.scheduler.step()
        else:
            epoch_loss_adv_sum = 0.0
            epoch_acc_clean_sum = 0.0
            epoch_acc_adv_sum = 0.0
            epoch_w2_sum = 0.0
            for step, (x, y) in enumerate(train_loader):
                if max_steps is not None and step >= max_steps:
                    break
                n = x.size(0)
                state, metrics = train_step_fn(state, (x, y), cfg, device)
                epoch_loss_adv_sum += float(metrics["loss_adv"]) * n
                epoch_acc_clean_sum += float(metrics["acc_clean"]) * n
                epoch_acc_adv_sum += float(metrics["acc_adv"]) * n
                epoch_w2_sum += float(metrics["w2_proxy"]) * n
                epoch_n += n
                num_steps += 1
            if num_steps > 0:
                avg_loss_adv = epoch_loss_adv_sum / epoch_n
                avg_acc_clean = epoch_acc_clean_sum / epoch_n
                avg_acc_adv = epoch_acc_adv_sum / epoch_n
                avg_w2_proxy = epoch_w2_sum / epoch_n
                print(
                    f"[{display_name}] Epoch {epoch} (adv) "
                    f"loss_adv={avg_loss_adv:.4f} acc_clean={avg_acc_clean:.4f} "
                    f"acc_adv={avg_acc_adv:.4f} W2≈{avg_w2_proxy:.4f}"
                )
                logger.log(
                    algorithm=algorithm_key, phase="train_adv", epoch=epoch,
                    step=num_steps, loss_adv=avg_loss_adv, adv_loss=None,
                    cls_loss=None, acc_clean=avg_acc_clean,
                    acc_adv=avg_acc_adv, w2_proxy=avg_w2_proxy,
                )
                training_logs.append({
                    "epoch": epoch, "phase": "adv", "steps": num_steps,
                    "loss_adv": avg_loss_adv, "acc_clean": avg_acc_clean,
                    "acc_adv": avg_acc_adv, "w2_proxy": avg_w2_proxy,
                })

                score, val_c, val_p = compute_val_score(
                    state, val_ds, cfg, device, is_clean_phase=False
                )
                print(
                    f"[{display_name}] Epoch {epoch}  val_clean={val_c:.4f}  "
                    f"val_pgd={val_p:.4f}  score={score:.4f}"
                )
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_model_sd = copy.deepcopy(state.model.state_dict())
                    best_opt_sd = copy.deepcopy(state.opt.state_dict())
                    patience_count = 0
                else:
                    patience_count += 1

                if state.scheduler is not None:
                    state.scheduler.step()

                if cfg.es_patience > 0 and patience_count >= cfg.es_patience:
                    print(
                        f"[{display_name}] Early stopping at epoch {epoch} "
                        f"({patience_count} adv epochs without improvement)."
                    )
                    break

    if best_model_sd is not None:
        state.model.load_state_dict(best_model_sd)
        state.opt.load_state_dict(best_opt_sd)
        print(
            f"[{display_name}] Restored best epoch {best_epoch} "
            f"(score={best_score:.4f})"
        )

    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print(f"[{display_name}] Test: {test_metrics}")
    logger.log(
        algorithm=algorithm_key, phase="test", epoch=cfg.num_epochs, step=None,
        loss_adv=float(test_metrics["loss"]), adv_loss=None, cls_loss=None,
        acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None,
    )

    results = {
        "algorithm": algorithm_key,
        "display_name": display_name,
        "hyperparameters": asdict(cfg),
        "training_logs": training_logs,
        "test_metrics": test_metrics,
    }
    save_training_results_json(f"{algorithm_key}_results.json", results)
    return state, {"test": test_metrics}
