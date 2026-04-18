"""New_PPA: free-weight projected particle ascent."""
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
from utils.projections import free_weight_projection
from utils.wrm import wrm_ascent_x, wrm_ascent_x_anchored_const_lr


def train_step_new_ppa(
    state: TrainState, batch, cfg: TrainConfig, device: torch.device
) -> Tuple[TrainState, Dict[str, torch.Tensor]]:
    """Round 0 = Algo1; refinement rounds alternate WRM ascent + best-response projection."""
    model = state.model
    opt = state.opt
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    if x.size(0) == 0:
        zero = torch.tensor(0.0, device=device)
        return state, {
            "loss_adv": zero, "acc_clean": zero, "acc_adv": zero,
            "w2_proxy": zero, "projection_gain": zero,
            "active_support_frac": zero,
        }

    set_requires_grad(model, False)
    model.eval()

    total_projection_gain = 0.0
    active_support_frac = 1.0

    z = wrm_ascent_x(
        x, model, y, cfg.lambda_reg,
        cfg.inner_steps_new_ppa_round0,
        lr=cfg.inner_lr_new_ppa_round0,
        clamp=(0.0, 1.0),
        step_offset=0,
    )

    for round_idx in range(1, cfg.new_ppa_num_rounds):
        z, _y_proj, gain, obj_scale, active_support_frac = free_weight_projection(
            z, x, y, model, cfg.lambda_reg
        )
        total_projection_gain += gain

        if (
            round_idx >= cfg.new_ppa_min_rounds
            and gain <= cfg.new_ppa_gain_rtol * max(obj_scale, 1e-12)
        ):
            break

        z = wrm_ascent_x_anchored_const_lr(
            z, x, model, y, cfg.lambda_reg,
            num_steps=cfg.new_ppa_refine_steps,
            lr=cfg.new_ppa_refine_lr,
            clamp=(0.0, 1.0),
        )

    z, _y_proj, gain_final, _obj_scale_final, active_support_frac = free_weight_projection(
        z, x, y, model, cfg.lambda_reg
    )
    total_projection_gain += gain_final

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
        "projection_gain": torch.tensor(total_projection_gain, device=device),
        "active_support_frac": torch.tensor(active_support_frac, device=device),
    }
    return state, metrics


def train_algorithm_new_ppa(
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
                if cfg.max_steps_new_ppa is not None and step >= cfg.max_steps_new_ppa:
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
                    f"[New_PPA] Epoch {epoch} (clean) loss={avg_loss:.4f} "
                    f"acc_clean={avg_acc_clean:.4f}"
                )
                logger.log(
                    algorithm="new_ppa", phase="train_clean", epoch=epoch,
                    step=num_steps, loss_adv=avg_loss, adv_loss=None,
                    cls_loss=None, acc_clean=avg_acc_clean, acc_adv=None,
                    w2_proxy=None, projection_gain=None,
                    active_support_frac=None,
                )
                training_logs.append({
                    "epoch": epoch, "phase": "clean", "steps": num_steps,
                    "loss": avg_loss, "acc_clean": avg_acc_clean,
                })
                score, val_c, _ = compute_val_score(
                    state, val_ds, cfg, device, is_clean_phase=True
                )
                print(
                    f"[New_PPA] Epoch {epoch}  val_clean={val_c:.4f}  "
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
            epoch_projection_gain_sum = 0.0
            epoch_active_support_sum = 0.0
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_new_ppa is not None and step >= cfg.max_steps_new_ppa:
                    break
                n = x.size(0)
                state, metrics = train_step_new_ppa(state, (x, y), cfg, device)
                epoch_loss_adv_sum += float(metrics["loss_adv"]) * n
                epoch_acc_clean_sum += float(metrics["acc_clean"]) * n
                epoch_acc_adv_sum += float(metrics["acc_adv"]) * n
                epoch_w2_sum += float(metrics["w2_proxy"]) * n
                epoch_projection_gain_sum += float(metrics["projection_gain"]) * n
                epoch_active_support_sum += float(metrics["active_support_frac"]) * n
                epoch_n += n
                num_steps += 1
            if num_steps > 0:
                avg_loss_adv = epoch_loss_adv_sum / epoch_n
                avg_acc_clean = epoch_acc_clean_sum / epoch_n
                avg_acc_adv = epoch_acc_adv_sum / epoch_n
                avg_w2_proxy = epoch_w2_sum / epoch_n
                avg_projection_gain = epoch_projection_gain_sum / epoch_n
                avg_active_support = epoch_active_support_sum / epoch_n
                print(
                    f"[New_PPA] Epoch {epoch} (adv) loss_adv={avg_loss_adv:.4f}"
                    f" acc_clean={avg_acc_clean:.4f} acc_adv={avg_acc_adv:.4f}"
                    f" W2≈{avg_w2_proxy:.4f} gain={avg_projection_gain:.4f}"
                    f" active_support={avg_active_support:.4f}"
                )
                logger.log(
                    algorithm="new_ppa", phase="train_adv", epoch=epoch,
                    step=num_steps, loss_adv=avg_loss_adv, adv_loss=None,
                    cls_loss=None, acc_clean=avg_acc_clean,
                    acc_adv=avg_acc_adv, w2_proxy=avg_w2_proxy,
                    projection_gain=avg_projection_gain,
                    active_support_frac=avg_active_support,
                )
                training_logs.append({
                    "epoch": epoch, "phase": "adv", "steps": num_steps,
                    "loss_adv": avg_loss_adv, "acc_clean": avg_acc_clean,
                    "acc_adv": avg_acc_adv, "w2_proxy": avg_w2_proxy,
                    "projection_gain": avg_projection_gain,
                    "active_support_frac": avg_active_support,
                })

                score, val_c, val_p = compute_val_score(
                    state, val_ds, cfg, device, is_clean_phase=False
                )
                print(
                    f"[New_PPA] Epoch {epoch}  val_clean={val_c:.4f}  "
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

                state.scheduler.step()

                if cfg.es_patience > 0 and patience_count >= cfg.es_patience:
                    print(
                        f"[New_PPA] Early stopping at epoch {epoch} "
                        f"({patience_count} adv epochs without improvement)."
                    )
                    break

    if best_model_sd is not None:
        state.model.load_state_dict(best_model_sd)
        state.opt.load_state_dict(best_opt_sd)
        print(
            f"[New_PPA] Restored best epoch {best_epoch} "
            f"(score={best_score:.4f})"
        )

    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print("[New_PPA] Test:", test_metrics)
    logger.log(
        algorithm="new_ppa", phase="test", epoch=cfg.num_epochs, step=None,
        loss_adv=float(test_metrics["loss"]), adv_loss=None, cls_loss=None,
        acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None,
        projection_gain=None, active_support_frac=None,
    )
    results = {
        "algorithm": "new_ppa_free_weight_projected_wrm",
        "hyperparameters": asdict(cfg),
        "training_logs": training_logs,
        "test_metrics": test_metrics,
    }
    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", "new_ppa_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return state, {"test": test_metrics}
