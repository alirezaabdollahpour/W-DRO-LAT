"""Algorithm 2: ICNN-transport adversarial training (BB+Armijo inner loop)."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from algorithms.base import (
    TrainState,
    create_classifier_state,
    train_step_clean,
)
from config import TrainConfig
from models.icnn import InputConvexPotential, icnn_gradient
from utils.bb_armijo import BBArmijoState, bb_armijo_step_params
from utils.common import (
    accuracy,
    adversary_loss,
    cross_entropy_loss,
    seed_everything,
    set_requires_grad,
)
from utils.data import compute_val_score, load_mnist, split_train_val
from utils.eval import evaluate_clean
from utils.flatten import flatten_params, unflatten_vector
from utils.logging import CSVLogger, DEFAULT_LOG_PATH, LOG_FIELDNAMES


@dataclass
class ICNNState:
    model: InputConvexPotential
    params_vec: torch.Tensor
    meta: Tuple[Tuple[str, Tuple[int, ...], int], ...]
    bb_state: BBArmijoState


def create_icnn_state(
    cfg: TrainConfig, input_dim: int, device: torch.device
) -> ICNNState:
    icnn_model = InputConvexPotential(
        input_dim=input_dim,
        hidden_sizes=cfg.icnn_hidden_sizes,
        activation="softplus",
        strong_convexity=1.0,
        nonneg_init="principled",
    ).to(device)
    icnn_model.init_as_identity()
    params_vec, meta = flatten_params(icnn_model)
    params_vec = params_vec.to(device)
    bb_state = BBArmijoState.create(alpha0=cfg.bb_alpha0_icnn)
    return ICNNState(
        model=icnn_model,
        params_vec=params_vec,
        meta=meta,
        bb_state=bb_state,
    )


def train_step_algo2(
    state: TrainState,
    icnn_state: ICNNState,
    batch,
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[TrainState, ICNNState, Dict[str, torch.Tensor]]:
    model = state.model
    opt = state.opt
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    if x.size(0) == 0:
        zero = torch.tensor(0.0, device=device)
        return state, icnn_state, {
            "adv_loss": zero, "cls_loss": zero, "acc_clean": zero,
            "acc_adv": zero, "w2_proxy": zero, "inner_grad_norm": zero,
        }

    x_flat = x.view(x.size(0), -1)
    icnn_model = icnn_state.model
    meta = icnn_state.meta
    bb_state = icnn_state.bb_state

    set_requires_grad(icnn_model, True)
    set_requires_grad(model, False)
    model.eval()

    def adv_obj_params(vec: torch.Tensor, create_graph: bool) -> torch.Tensor:
        params_dict = unflatten_vector(vec, meta)
        adv_flat = icnn_gradient(
            icnn_model, params_dict, x_flat, create_graph=create_graph
        )
        adv_x_inner = adv_flat.view_as(x)
        logits = model(adv_x_inner)
        adv_loss = adversary_loss(logits, y, cfg.use_margin_adv_algo2)
        w2 = ((adv_x_inner - x) ** 2).sum(dim=(1, 2, 3)).mean()
        return adv_loss - cfg.lambda_reg * w2

    current_vec = icnn_state.params_vec.to(device)
    adv_loss_val = torch.tensor(0.0, device=device)

    for _ in range(cfg.inner_steps_algo2):
        current_vec, bb_state, f_val_f = bb_armijo_step_params(
            current_vec, meta, adv_obj_params, bb_state
        )
        adv_loss_val = torch.tensor(f_val_f, device=device)

    # Final-iterate gradient norm for diagnostics.
    v_eval = current_vec.detach().requires_grad_(True)
    f_eval = adv_obj_params(v_eval, True)
    g_eval = torch.autograd.grad(f_eval, v_eval, create_graph=False)[0]
    inner_grad_norm = float(g_eval.norm().item())

    icnn_state = ICNNState(
        model=icnn_model,
        params_vec=current_vec,
        meta=meta,
        bb_state=bb_state,
    )

    set_requires_grad(model, True)
    model.train()

    params_dict_final = unflatten_vector(icnn_state.params_vec.to(device), meta)
    adv_flat = icnn_gradient(icnn_model, params_dict_final, x_flat).detach()
    adv_x = adv_flat.view_as(x).clamp(0.0, 1.0)

    with torch.enable_grad():
        logits_adv = model(adv_x)
        cls_loss = cross_entropy_loss(logits_adv, y)
        opt.zero_grad()
        cls_loss.backward()
        opt.step()

    with torch.no_grad():
        logits_clean = model(x)
        logits_adv_post = model(adv_x)
        acc_clean = accuracy(logits_clean, y)
        acc_adv = accuracy(logits_adv_post, y)
        w2_proxy = ((adv_x - x) ** 2).sum(dim=(1, 2, 3)).mean()
    metrics = {
        "adv_loss": adv_loss_val.detach(),
        "cls_loss": cls_loss.detach(),
        "acc_clean": acc_clean,
        "acc_adv": acc_adv,
        "w2_proxy": w2_proxy,
        "inner_grad_norm": torch.tensor(inner_grad_norm, device=device),
    }
    return state, icnn_state, metrics


def train_algorithm_2(
    cfg: TrainConfig, device: torch.device
) -> Tuple[TrainState, ICNNState, Dict[str, Any]]:
    """ICNN transport adversarial training with val-PGD checkpointing."""
    seed_everything(cfg.seed)
    train_raw_ds, test_ds = load_mnist()
    train_ds, val_ds = split_train_val(train_raw_ds, cfg.es_val_frac, cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    state = create_classifier_state(cfg, device)
    logger = CSVLogger(DEFAULT_LOG_PATH, LOG_FIELDNAMES)
    training_logs: List[Dict[str, Any]] = []
    input_dim = 28 * 28 * 1
    icnn_state = create_icnn_state(cfg, input_dim, device)

    best_score = -float("inf")
    best_epoch = -1
    best_model_sd = None
    best_opt_sd = None
    best_icnn_sd = None
    best_icnn_params_vec = None
    patience_count = 0
    in_adv_phase = False

    def _save_best():
        return (
            copy.deepcopy(state.model.state_dict()),
            copy.deepcopy(state.opt.state_dict()),
            copy.deepcopy(icnn_state.model.state_dict()),
            icnn_state.params_vec.detach().clone(),
        )

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
                if cfg.max_steps_algo2 is not None and step >= cfg.max_steps_algo2:
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
                    f"[Algo2] Epoch {epoch} (clean) loss={avg_loss:.4f} "
                    f"acc_clean={avg_acc_clean:.4f}"
                )
                logger.log(
                    algorithm="algo2", phase="train_clean", epoch=epoch,
                    step=num_steps, loss_adv=None, adv_loss=None,
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
                print(
                    f"[Algo2] Epoch {epoch}  val_clean={val_c:.4f}  "
                    f"score={score:.4f}"
                )
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    (best_model_sd, best_opt_sd, best_icnn_sd,
                     best_icnn_params_vec) = _save_best()
            state.scheduler.step()

        else:
            epoch_adv_loss_sum = 0.0
            epoch_cls_loss_sum = 0.0
            epoch_acc_clean_sum = 0.0
            epoch_acc_adv_sum = 0.0
            epoch_w2_sum = 0.0
            epoch_inner_grad_sum = 0.0
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_algo2 is not None and step >= cfg.max_steps_algo2:
                    break
                n = x.size(0)
                state, icnn_state, metrics = train_step_algo2(
                    state, icnn_state, (x, y), cfg, device
                )
                epoch_adv_loss_sum += float(metrics["adv_loss"]) * n
                epoch_cls_loss_sum += float(metrics["cls_loss"]) * n
                epoch_acc_clean_sum += float(metrics["acc_clean"]) * n
                epoch_acc_adv_sum += float(metrics["acc_adv"]) * n
                epoch_w2_sum += float(metrics["w2_proxy"]) * n
                epoch_inner_grad_sum += float(metrics["inner_grad_norm"]) * n
                epoch_n += n
                num_steps += 1
            if num_steps > 0:
                avg_adv_loss = epoch_adv_loss_sum / epoch_n
                avg_cls_loss = epoch_cls_loss_sum / epoch_n
                avg_acc_clean = epoch_acc_clean_sum / epoch_n
                avg_acc_adv = epoch_acc_adv_sum / epoch_n
                avg_w2_proxy = epoch_w2_sum / epoch_n
                avg_inner_grad_norm = epoch_inner_grad_sum / epoch_n
                print(
                    f"[Algo2] Epoch {epoch} (adv) adv_loss={avg_adv_loss:.4f}"
                    f" cls_loss={avg_cls_loss:.4f} acc_clean={avg_acc_clean:.4f}"
                    f" acc_adv={avg_acc_adv:.4f} W2≈{avg_w2_proxy:.4f}"
                    f" |∇θ|={avg_inner_grad_norm:.6f}"
                )
                logger.log(
                    algorithm="algo2", phase="train_adv", epoch=epoch,
                    step=num_steps, loss_adv=None, adv_loss=avg_adv_loss,
                    cls_loss=avg_cls_loss, acc_clean=avg_acc_clean,
                    acc_adv=avg_acc_adv, w2_proxy=avg_w2_proxy,
                    inner_grad_norm=avg_inner_grad_norm,
                )
                training_logs.append({
                    "epoch": epoch, "phase": "adv", "steps": num_steps,
                    "adv_loss": avg_adv_loss, "cls_loss": avg_cls_loss,
                    "acc_clean": avg_acc_clean, "acc_adv": avg_acc_adv,
                    "w2_proxy": avg_w2_proxy,
                    "inner_grad_norm": avg_inner_grad_norm,
                })

                score, val_c, val_p = compute_val_score(
                    state, val_ds, cfg, device, is_clean_phase=False
                )
                print(
                    f"[Algo2] Epoch {epoch}  val_clean={val_c:.4f}"
                    f"  val_pgd={val_p:.4f}  score={score:.4f}"
                )

                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    (best_model_sd, best_opt_sd, best_icnn_sd,
                     best_icnn_params_vec) = _save_best()
                    patience_count = 0
                else:
                    patience_count += 1

                state.scheduler.step()

                if cfg.es_patience > 0 and patience_count >= cfg.es_patience:
                    print(
                        f"[Algo2] Early stopping at epoch {epoch} "
                        f"({patience_count} adv epochs without improvement)."
                    )
                    break

    if best_model_sd is not None:
        state.model.load_state_dict(best_model_sd)
        state.opt.load_state_dict(best_opt_sd)
        icnn_state.model.load_state_dict(best_icnn_sd)
        icnn_state.params_vec = best_icnn_params_vec
        print(f"[Algo2] Restored best epoch {best_epoch} (score={best_score:.4f})")

    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print("[Algo2] Test:", test_metrics)
    logger.log(
        algorithm="algo2", phase="test", epoch=cfg.num_epochs, step=None,
        loss_adv=float(test_metrics["loss"]), adv_loss=None,
        cls_loss=float(test_metrics["loss"]),
        acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None,
    )
    results = {
        "algorithm": "algo2_icnn",
        "hyperparameters": asdict(cfg),
        "training_logs": training_logs,
        "test_metrics": test_metrics,
    }
    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", "algo2_icnn_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return state, icnn_state, {"test": test_metrics}
