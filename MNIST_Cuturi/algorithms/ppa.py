"""PPA: Projected Particle Ascent (WRM ascent + within-class Brenier projection)."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from algorithms.base import (
    TrainState,
    create_classifier_state,
    train_step_clean,
)
from config import TrainConfig
from utils.common import (
    accuracy,
    cross_entropy_loss,
    seed_everything,
    set_requires_grad,
)
from utils.data import compute_val_score, load_mnist, split_train_val
from utils.eval import evaluate_clean
from utils.logging import CSVLogger, DEFAULT_LOG_PATH, LOG_FIELDNAMES
from utils.projections import brenier_projection
from utils.wrm import wrm_ascent_x, wrm_ascent_x_anchored_const_lr


def train_step_ppa(
    state: TrainState, batch, cfg: TrainConfig, device: torch.device
) -> Tuple[TrainState, Dict[str, torch.Tensor]]:
    """Round 0 replicates Algo1 exactly; refinement rounds alternate Brenier + ascent."""
    model = state.model
    opt = state.opt
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    if x.size(0) == 0:
        zero = torch.tensor(0.0, device=device)
        return state, {
            "loss_adv": zero, "acc_clean": zero, "acc_adv": zero,
            "w2_proxy": zero, "delta_gap": zero,
        }

    set_requires_grad(model, False)
    model.eval()

    total_delta = 0.0

    z = wrm_ascent_x(
        x, model, y, cfg.lambda_reg,
        cfg.inner_steps_ppa_round0,
        lr=cfg.inner_lr_ppa_round0,
        clamp=(0.0, 1.0),
        step_offset=0,
    )

    for round_idx in range(1, cfg.ppa_num_rounds):
        z, _y_proj, delta, _C_id, _C_ot = brenier_projection(z, x, y)
        total_delta += delta

        if (
            round_idx >= cfg.ppa_min_rounds
            and delta < cfg.ppa_delta_rtol * max(_C_id, 1e-12)
        ):
            break

        z = wrm_ascent_x_anchored_const_lr(
            z, x, model, y, cfg.lambda_reg,
            num_steps=cfg.ppa_refine_steps,
            lr=cfg.ppa_refine_lr,
            clamp=(0.0, 1.0),
        )

    z, _y_proj, delta_final, _, _ = brenier_projection(z, x, y)
    total_delta += delta_final

    adv_x = z.detach()

    set_requires_grad(model, True)
    model.train()

    with torch.enable_grad():
        logits_adv = model(adv_x)
        loss = cross_entropy_loss(logits_adv, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_clean = model(x)
        logits_adv_post = model(adv_x)
        acc_clean = accuracy(logits_clean, y)
        acc_adv = accuracy(logits_adv_post, y)
        w2_proxy = ((adv_x - x) ** 2).sum(dim=(1, 2, 3)).mean()
    metrics = {
        "loss_adv": loss.detach(),
        "acc_clean": acc_clean,
        "acc_adv": acc_adv,
        "w2_proxy": w2_proxy,
        "delta_gap": torch.tensor(total_delta, device=device),
    }
    return state, metrics


def train_algorithm_ppa(
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
    patience_count = 0
    in_adv_phase = False

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
                if cfg.max_steps_ppa is not None and step >= cfg.max_steps_ppa:
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
                    f"[PPA] Epoch {epoch} (clean) loss={avg_loss:.4f} "
                    f"acc_clean={avg_acc_clean:.4f}"
                )
                logger.log(
                    algorithm="ppa", phase="train_clean", epoch=epoch,
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
                    f"[PPA]   Epoch {epoch}  val_clean={val_c:.4f}  "
                    f"score={score:.4f}"
                )
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_model_sd = copy.deepcopy(state.model.state_dict())
                    best_opt_sd = copy.deepcopy(state.opt.state_dict())
            state.scheduler.step()

        else:
            epoch_loss_adv_sum = 0.0
            epoch_acc_clean_sum = 0.0
            epoch_acc_adv_sum = 0.0
            epoch_w2_sum = 0.0
            epoch_delta_sum = 0.0
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_ppa is not None and step >= cfg.max_steps_ppa:
                    break
                n = x.size(0)
                state, metrics = train_step_ppa(state, (x, y), cfg, device)
                epoch_loss_adv_sum += float(metrics["loss_adv"]) * n
                epoch_acc_clean_sum += float(metrics["acc_clean"]) * n
                epoch_acc_adv_sum += float(metrics["acc_adv"]) * n
                epoch_w2_sum += float(metrics["w2_proxy"]) * n
                epoch_delta_sum += float(metrics["delta_gap"]) * n
                epoch_n += n
                num_steps += 1
            if num_steps > 0:
                avg_loss_adv = epoch_loss_adv_sum / epoch_n
                avg_acc_clean = epoch_acc_clean_sum / epoch_n
                avg_acc_adv = epoch_acc_adv_sum / epoch_n
                avg_w2_proxy = epoch_w2_sum / epoch_n
                avg_delta_gap = epoch_delta_sum / epoch_n
                print(
                    f"[PPA] Epoch {epoch} (adv) loss_adv={avg_loss_adv:.4f}"
                    f" acc_clean={avg_acc_clean:.4f} acc_adv={avg_acc_adv:.4f}"
                    f" W2≈{avg_w2_proxy:.4f} Δ={avg_delta_gap:.4f}"
                )
                logger.log(
                    algorithm="ppa", phase="train_adv", epoch=epoch,
                    step=num_steps, loss_adv=avg_loss_adv, adv_loss=None,
                    cls_loss=None, acc_clean=avg_acc_clean,
                    acc_adv=avg_acc_adv, w2_proxy=avg_w2_proxy,
                    delta_gap=avg_delta_gap,
                )
                training_logs.append({
                    "epoch": epoch, "phase": "adv", "steps": num_steps,
                    "loss_adv": avg_loss_adv, "acc_clean": avg_acc_clean,
                    "acc_adv": avg_acc_adv, "w2_proxy": avg_w2_proxy,
                    "delta_gap": avg_delta_gap,
                })

                score, val_c, val_p = compute_val_score(
                    state, val_ds, cfg, device, is_clean_phase=False
                )
                print(
                    f"[PPA]   Epoch {epoch}  val_clean={val_c:.4f}"
                    f"  val_pgd={val_p:.4f}  score={score:.4f}"
                )

                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_model_sd = copy.deepcopy(state.model.state_dict())
                    best_opt_sd = copy.deepcopy(state.opt.state_dict())
                    patience_count = 0
                else:
                    patience_count += 1

                state.scheduler.step()

                if cfg.es_patience > 0 and patience_count >= cfg.es_patience:
                    print(
                        f"[PPA]   Early stopping at epoch {epoch} "
                        f"({patience_count} adv epochs without improvement)."
                    )
                    break

    if best_model_sd is not None:
        state.model.load_state_dict(best_model_sd)
        state.opt.load_state_dict(best_opt_sd)
        print(f"[PPA] Restored best epoch {best_epoch} (score={best_score:.4f})")

    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print("[PPA] Test:", test_metrics)
    logger.log(
        algorithm="ppa", phase="test", epoch=cfg.num_epochs, step=None,
        loss_adv=float(test_metrics["loss"]), adv_loss=None, cls_loss=None,
        acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None,
    )
    results = {
        "algorithm": "ppa_projected_wrm",
        "hyperparameters": asdict(cfg),
        "training_logs": training_logs,
        "test_metrics": test_metrics,
    }
    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", "ppa_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return state, {"test": test_metrics}
