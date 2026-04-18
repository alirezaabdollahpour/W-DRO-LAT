"""Algorithm 1: WRM adversarial training (Sinha et al.)."""
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
from utils.cm_diagnostics import aggregate_cm_results, check_cyclical_monotonicity
from utils.common import (
    accuracy,
    cross_entropy_loss,
    seed_everything,
    set_requires_grad,
)
from utils.data import compute_val_score, load_mnist, split_train_val
from utils.eval import evaluate_clean
from utils.logging import CSVLogger, DEFAULT_LOG_PATH, LOG_FIELDNAMES
from utils.wrm import wrm_ascent_x


def train_step_algo1(
    state: TrainState, batch, cfg: TrainConfig, device: torch.device
) -> Tuple[TrainState, Dict[str, torch.Tensor]]:
    model = state.model
    opt = state.opt
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    if x.size(0) == 0:
        zero = torch.tensor(0.0, device=device)
        return state, {
            "loss_adv": zero, "acc_clean": zero,
            "acc_adv": zero, "w2_proxy": zero,
        }

    set_requires_grad(model, False)
    model.eval()

    adv_x = wrm_ascent_x(
        x, model, y, cfg.lambda_reg,
        cfg.inner_steps_algo1,
        lr=cfg.inner_lr_algo1,
        clamp=(0.0, 1.0),
    )

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
    }

    if cfg.cm_diagnostics:
        with torch.no_grad():
            cm_gen = torch.Generator(device=x.device)
            cm_gen.manual_seed(0)
            cm = check_cyclical_monotonicity(x, adv_x, generator=cm_gen)
        metrics["cm_diagnostics"] = cm

    return state, metrics


def train_algorithm_1(
    cfg: TrainConfig, device: torch.device
) -> Tuple[TrainState, Dict[str, Any]]:
    """WRM adversarial training with PGD-scored checkpointing + early stopping."""
    seed_everything(cfg.seed)
    train_raw_ds, test_ds = load_mnist()
    train_ds, val_ds = split_train_val(train_raw_ds, cfg.es_val_frac, cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    state = create_classifier_state(cfg, device)
    logger = CSVLogger(DEFAULT_LOG_PATH, LOG_FIELDNAMES)
    training_logs: List[Dict[str, Any]] = []
    cm_epochs: List[Dict[str, Any]] = []

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
                if cfg.max_steps_algo1 is not None and step >= cfg.max_steps_algo1:
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
                    f"[Algo1] Epoch {epoch} (clean) loss={avg_loss:.4f} "
                    f"acc_clean={avg_acc_clean:.4f}"
                )
                logger.log(
                    algorithm="algo1", phase="train_clean", epoch=epoch,
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
                    f"[Algo1] Epoch {epoch}  val_clean={val_c:.4f}  "
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
            epoch_cm_batch_results: List[Dict[int, Dict[str, Any]]] = []
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_algo1 is not None and step >= cfg.max_steps_algo1:
                    break
                n = x.size(0)
                state, metrics = train_step_algo1(state, (x, y), cfg, device)
                epoch_loss_adv_sum += float(metrics["loss_adv"]) * n
                epoch_acc_clean_sum += float(metrics["acc_clean"]) * n
                epoch_acc_adv_sum += float(metrics["acc_adv"]) * n
                epoch_w2_sum += float(metrics["w2_proxy"]) * n
                if cfg.cm_diagnostics and "cm_diagnostics" in metrics:
                    epoch_cm_batch_results.append(metrics["cm_diagnostics"])
                epoch_n += n
                num_steps += 1
            if num_steps > 0:
                avg_loss_adv = epoch_loss_adv_sum / epoch_n
                avg_acc_clean = epoch_acc_clean_sum / epoch_n
                avg_acc_adv = epoch_acc_adv_sum / epoch_n
                avg_w2_proxy = epoch_w2_sum / epoch_n
                print(
                    f"[Algo1] Epoch {epoch} (adv) loss_adv={avg_loss_adv:.4f}"
                    f" acc_clean={avg_acc_clean:.4f} acc_adv={avg_acc_adv:.4f}"
                    f" W2≈{avg_w2_proxy:.4f}"
                )
                if cfg.cm_diagnostics and epoch_cm_batch_results:
                    epoch_cm_agg = aggregate_cm_results(epoch_cm_batch_results)
                    cm_epochs.append({
                        "epoch": epoch,
                        "cm": {str(k): v for k, v in epoch_cm_agg.items()},
                    })
                    cm_parts = [
                        f"k={k}: viol={epoch_cm_agg[k]['frac_violated']:.1%}"
                        f" pe={epoch_cm_agg[k]['mean_per_edge']:.4f}"
                        f" rel={epoch_cm_agg[k]['mean_relative']:.4f}"
                        for k in sorted(epoch_cm_agg.keys())
                    ]
                    print(f"        CM diagnostic: {' | '.join(cm_parts)}")
                logger.log(
                    algorithm="algo1", phase="train_adv", epoch=epoch,
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
                    f"[Algo1] Epoch {epoch}  val_clean={val_c:.4f}"
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
                        f"[Algo1] Early stopping at epoch {epoch} "
                        f"({patience_count} adv epochs without improvement)."
                    )
                    break

    if best_model_sd is not None:
        state.model.load_state_dict(best_model_sd)
        state.opt.load_state_dict(best_opt_sd)
        print(f"[Algo1] Restored best epoch {best_epoch} (score={best_score:.4f})")

    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print("[Algo1] Test:", test_metrics)
    logger.log(
        algorithm="algo1", phase="test", epoch=cfg.num_epochs, step=None,
        loss_adv=float(test_metrics["loss"]), adv_loss=None, cls_loss=None,
        acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None,
    )

    if cm_epochs:
        all_batch_cm = [e["cm"] for e in cm_epochs]
        all_k = sorted({int(k) for d in all_batch_cm for k in d})
        print("\n" + "=" * 72)
        print("[Algo1] Cyclical Monotonicity Summary (averaged over adv epochs)")
        print("=" * 72)
        print(
            f"  {'k':>4s}  {'Frac Violated':>14s}  {'Mean/Edge':>10s}"
            f"  {'Std/Edge':>10s}  {'Mean Rel':>10s}  {'Max Rel':>10s}"
        )
        print("-" * 72)
        for k in all_k:
            entries = [d[str(k)] for d in all_batch_cm if str(k) in d]
            n = len(entries)
            fv = sum(e["frac_violated"] for e in entries) / n
            mpe = sum(e["mean_per_edge"] for e in entries) / n
            spe = sum(e["std_per_edge"] for e in entries) / n
            mrl = sum(e["mean_relative"] for e in entries) / n
            xrl = max(e["max_relative"] for e in entries)
            print(
                f"  {k:4d}  {fv:14.2%}  {mpe:10.6f}  {spe:10.6f}"
                f"  {mrl:10.6f}  {xrl:10.6f}"
            )
        print("=" * 72 + "\n")

    results = {
        "algorithm": "algo1_wrm",
        "hyperparameters": asdict(cfg),
        "training_logs": training_logs,
        "test_metrics": test_metrics,
    }
    if cm_epochs:
        results["cyclical_monotonicity"] = cm_epochs
    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", "algo1_wrm_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    if cm_epochs:
        with open(os.path.join("MNIST", "algo1_cm_diagnostics.json"), "w") as f:
            json.dump({
                "algorithm": "algo1_wrm", "cm_per_epoch": cm_epochs,
            }, f, indent=2)
    return state, {"test": test_metrics}
