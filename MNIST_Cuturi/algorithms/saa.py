"""SAA: plain ERM (no adversary). Same data pipeline as the adv algorithms."""
from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from algorithms.base import (
    TrainState,
    create_classifier_state,
    save_training_results_json,
    train_step_clean,
)
from config import TrainConfig
from utils.common import seed_everything
from utils.data import compute_val_score, load_mnist, split_train_val
from utils.eval import evaluate_clean
from utils.logging import CSVLogger, DEFAULT_LOG_PATH, LOG_FIELDNAMES


def train_algorithm_saa(
    cfg: TrainConfig, device: torch.device
) -> Tuple[TrainState, Dict[str, Any]]:
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
    max_steps = cfg.max_steps_saa

    for epoch in range(cfg.num_epochs):
        epoch_loss_sum = 0.0
        epoch_acc_clean_sum = 0.0
        epoch_n = 0
        num_steps = 0
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
                f"[SAA] Epoch {epoch} loss={avg_loss:.4f} "
                f"acc_clean={avg_acc_clean:.4f}"
            )
            logger.log(
                algorithm="saa", phase="train_clean", epoch=epoch,
                step=num_steps, loss_adv=avg_loss, adv_loss=None,
                cls_loss=avg_loss, acc_clean=avg_acc_clean, acc_adv=None,
                w2_proxy=None,
            )
            training_logs.append({
                "epoch": epoch, "phase": "clean", "steps": num_steps,
                "loss": avg_loss, "acc_clean": avg_acc_clean,
            })
            score, val_c, _ = compute_val_score(
                state, val_ds, cfg, device, is_clean_phase=True
            )
            print(f"[SAA] Epoch {epoch}  val_clean={val_c:.4f}  score={score:.4f}")
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_model_sd = copy.deepcopy(state.model.state_dict())
                best_opt_sd = copy.deepcopy(state.opt.state_dict())
        if state.scheduler is not None:
            state.scheduler.step()

    if best_model_sd is not None:
        state.model.load_state_dict(best_model_sd)
        state.opt.load_state_dict(best_opt_sd)
        print(f"[SAA] Restored best epoch {best_epoch} (score={best_score:.4f})")

    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print(f"[SAA] Test: {test_metrics}")
    logger.log(
        algorithm="saa", phase="test", epoch=cfg.num_epochs, step=None,
        loss_adv=float(test_metrics["loss"]), adv_loss=None,
        cls_loss=float(test_metrics["loss"]),
        acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None,
    )
    results = {
        "algorithm": "saa",
        "display_name": "SAA",
        "hyperparameters": asdict(cfg),
        "training_logs": training_logs,
        "test_metrics": test_metrics,
    }
    save_training_results_json("saa_results.json", results)
    return state, {"test": test_metrics}
