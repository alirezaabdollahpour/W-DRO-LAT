"""Side-by-side ICNN vs NPF-ICNN adversary diagnostics.

The helpers in this file are intentionally notebook-friendly: they run the
old stable input ICNN and the newer NPF/LastQuad ICNN on the same CIFAR-10
batch, classifier, objective, BB+Armijo settings, and PGD reference attack.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrained_INPUT_icnn_stable_version import (
    InputConvexPotential as LegacyInputConvexPotential,
)

from .data import get_cifar10_loaders, get_cifar10_train_loader
from .models.classifier import load_pretrained_classifier
from .models.npf import NPFInputConvexPotential, npf_T_omega
from .utils import (
    BBArmijoState,
    adversary_loss_per_sample,
    bb_armijo_step_params,
    clamped_normalized_copy,
    frozen_module,
    input_pgd_best_adversary,
    normalized_mse,
    pixel_l2_squared,
    project_normalized_to_input_ball,
    set_deterministic,
    to_pixel,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANAGEABLE_SHARED_HIDDEN: Tuple[int, ...] = (256, 128, 64)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_hidden(value: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    return tuple(int(v) for v in value)


@dataclass
class DiagnosticConfig:
    """Configuration for one isolated ICNN-vs-NPF adversary comparison."""

    data_dir: str = str(REPO_ROOT / "data")
    pretrained_path: str = str(REPO_ROOT / "ResNet_checkpoints" / "R2.pth")
    pretrained_strict: bool = False
    device: str = "auto"
    seed: int = 1
    split: str = "train"
    batch_index: int = 0
    batch_size: int = 512
    max_samples: int = 64
    num_workers: int = 2
    augment_train: bool = True

    # DRO objective. Defaults mirror run_local_npf_lastquad.sh.
    omega_steps: int = 10
    penalty_lambda: float = 0.0976
    transport_cost: str = "pixel_l2_squared"
    use_margin_loss: bool = False
    attack_clean_correct_only: bool = True
    project_to_input_ball: bool = False

    # BB+Armijo, also mirroring run_local_npf_lastquad.sh.
    bb_alpha0: float = 5e-4
    bb_alpha_min: float = 1e-6
    bb_alpha_max: float = 1.0
    bb_ls_c: float = 0.1
    bb_ls_shrink: float = 0.5
    bb_ls_max_steps: int = 10
    parametric_bb_max_grad_norm: float = 1.0

    # Old stable ICNN.
    legacy_icnn_hidden: Tuple[int, ...] = (1024, 512, 512, 256, 128, 64)
    legacy_icnn_activation: str = "softplus"
    legacy_icnn_strong_convexity: float = 1.0
    legacy_icnn_init: str = "principled"

    # New NPF-LastQuad from run_local_npf_lastquad.sh.
    npf_lastquad_hidden: Tuple[int, ...] = (512, 256, 128, 64)
    npf_lastquad_activation: str = "softplus"
    npf_lastquad_elu_alpha: float = 1.0
    npf_lastquad_softplus_beta: float = 20.0
    npf_lastquad_output_rank: int = 2
    npf_lastquad_strong_convexity: float = 1.0
    npf_lastquad_identity_init: bool = True
    npf_lastquad_init_eps: float = 1e-2
    npf_lastquad_pos_weights: bool = True
    npf_lastquad_positive_weight_rectifier: str = "relu"

    # Optional full NPF ablation.
    npf_hidden: Tuple[int, ...] = (512, 256, 128, 64)
    npf_outer_rank: int = 8
    npf_inner_rank: int = 2
    npf_activation: str = "relu"
    npf_elu_alpha: float = 1.0
    npf_softplus_beta: float = 20.0
    npf_strong_convexity: float = 1.0
    npf_identity_init: bool = False
    npf_init_eps: float = 0.0
    npf_pos_weights: bool = True
    npf_positive_weight_rectifier: str = "relu"

    # PGD reference. Defaults follow run_local_npf_lastquad.sh before the
    # launcher converts normalized WRM radius rho to raw pixel L2 epsilon.
    run_pgd_reference: bool = True
    inp_p: str = "2"
    inp_eps: float = 0.05
    inp_steps: int = 20
    inp_step_size: float = 0.0
    inp_restarts: int = 5
    input_pgd_loss: str = "ce"
    input_pgd_geometry: str = "pixel_l2_squared"
    wrm_normalized_radius_protocol: bool = True
    wrm_pgd_eps_mode: str = "normalized_radius"
    wrm_cp: Optional[float] = None
    wrm_cp_batch_size: int = 1024
    wrm_cp_num_workers: int = 0

    num_visual_samples: int = 8
    output_dir: str = str(REPO_ROOT / "pretrained_input_icnn" / "debug_outputs")

    @classmethod
    def from_run_local_env(cls) -> "DiagnosticConfig":
        """Build config from LOCAL_* env vars used by run_local_npf_lastquad.sh."""
        cfg = cls()
        env = os.environ
        cfg.seed = int(env.get("LOCAL_SEED", cfg.seed))
        cfg.data_dir = env.get("LOCAL_DATA_DIR", cfg.data_dir)
        cfg.pretrained_path = env.get("LOCAL_PRETRAINED_PATH", cfg.pretrained_path)
        cfg.num_workers = int(env.get("LOCAL_NUM_WORKERS", cfg.num_workers))
        cfg.batch_size = int(env.get("LOCAL_COMMON_BATCH", cfg.batch_size))
        cfg.omega_steps = int(env.get("LOCAL_K", cfg.omega_steps))
        cfg.penalty_lambda = float(env.get("LOCAL_PENALTY_LAMBDA", cfg.penalty_lambda))
        cfg.transport_cost = env.get("LOCAL_TRANSPORT_COST", cfg.transport_cost)
        cfg.use_margin_loss = _parse_bool(env.get("LOCAL_USE_MARGIN_LOSS", cfg.use_margin_loss))
        cfg.attack_clean_correct_only = _parse_bool(
            env.get("LOCAL_ATTACK_CLEAN_CORRECT_ONLY", cfg.attack_clean_correct_only)
        )
        cfg.project_to_input_ball = _parse_bool(
            env.get("LOCAL_NPF_PROJECT_TO_INPUT_BALL", cfg.project_to_input_ball)
        )
        cfg.bb_alpha0 = float(env.get("LOCAL_BB_ALPHA0", cfg.bb_alpha0))
        cfg.bb_alpha_min = float(env.get("LOCAL_BB_ALPHA_MIN", cfg.bb_alpha_min))
        cfg.bb_alpha_max = float(env.get("LOCAL_BB_ALPHA_MAX", cfg.bb_alpha_max))
        cfg.bb_ls_c = float(env.get("LOCAL_BB_LS_C", cfg.bb_ls_c))
        cfg.bb_ls_shrink = float(env.get("LOCAL_BB_LS_SHRINK", cfg.bb_ls_shrink))
        cfg.bb_ls_max_steps = int(env.get("LOCAL_BB_LS_MAX_STEPS", cfg.bb_ls_max_steps))
        cfg.parametric_bb_max_grad_norm = float(
            env.get("LOCAL_PARAMETRIC_BB_MAX_GRAD_NORM", cfg.parametric_bb_max_grad_norm)
        )
        cfg.npf_lastquad_hidden = _parse_hidden(
            env.get("LOCAL_NPF_LASTQUAD_HIDDEN"), cfg.npf_lastquad_hidden
        )
        cfg.npf_lastquad_activation = env.get(
            "LOCAL_NPF_LASTQUAD_ACTIVATION", cfg.npf_lastquad_activation
        )
        cfg.npf_lastquad_output_rank = int(
            env.get("LOCAL_NPF_LASTQUAD_OUTPUT_RANK", cfg.npf_lastquad_output_rank)
        )
        cfg.npf_lastquad_identity_init = _parse_bool(
            env.get("LOCAL_NPF_LASTQUAD_IDENTITY_INIT", cfg.npf_lastquad_identity_init)
        )
        cfg.npf_lastquad_init_eps = float(
            env.get("LOCAL_NPF_LASTQUAD_INIT_EPS", cfg.npf_lastquad_init_eps)
        )
        cfg.npf_lastquad_strong_convexity = float(
            env.get("LOCAL_NPF_LASTQUAD_STRONG_CONVEXITY", cfg.npf_lastquad_strong_convexity)
        )
        cfg.npf_lastquad_positive_weight_rectifier = env.get(
            "LOCAL_NPF_LASTQUAD_POSITIVE_WEIGHT_RECTIFIER",
            cfg.npf_lastquad_positive_weight_rectifier,
        )
        cfg.npf_strong_convexity = float(
            env.get("LOCAL_NPF_STRONG_CONVEXITY", cfg.npf_strong_convexity)
        )
        cfg.inp_p = env.get("LOCAL_INP_P", cfg.inp_p)
        cfg.inp_eps = float(env.get("LOCAL_INP_EPS", cfg.inp_eps))
        cfg.inp_steps = int(env.get("LOCAL_INP_STEPS", cfg.inp_steps))
        cfg.inp_restarts = int(env.get("LOCAL_INP_RESTARTS", cfg.inp_restarts))
        cfg.input_pgd_loss = env.get("LOCAL_INPUT_PGD_LOSS", cfg.input_pgd_loss)
        cfg.input_pgd_geometry = env.get("LOCAL_INPUT_PGD_GEOMETRY", cfg.input_pgd_geometry)
        cfg.wrm_normalized_radius_protocol = _parse_bool(
            env.get("LOCAL_WRM_NORMALIZED_RADIUS_PROTOCOL", cfg.wrm_normalized_radius_protocol)
        )
        cfg.wrm_pgd_eps_mode = env.get("LOCAL_WRM_PGD_EPS_MODE", cfg.wrm_pgd_eps_mode)
        if env.get("LOCAL_WRM_CP"):
            cfg.wrm_cp = float(env["LOCAL_WRM_CP"])
        return cfg

    @classmethod
    def legacy_stable_parity(cls) -> "DiagnosticConfig":
        """Settings close to run_pretrained_input_icnn_stable_version.sh."""
        cfg = cls()
        cfg.penalty_lambda = 30.0
        cfg.transport_cost = "normalized_mse"
        cfg.use_margin_loss = True
        cfg.inp_eps = 0.5
        cfg.input_pgd_loss = "margin"
        cfg.wrm_normalized_radius_protocol = False
        cfg.legacy_icnn_hidden = (1024, 512, 512, 256, 128, 64)
        return cfg


def set_shared_hidden_sizes(
    cfg: DiagnosticConfig,
    hidden: Sequence[int] = MANAGEABLE_SHARED_HIDDEN,
) -> DiagnosticConfig:
    """Set old ICNN, NPF-LastQuad, and full NPF to the same hidden widths."""
    shared = tuple(int(h) for h in hidden)
    cfg.legacy_icnn_hidden = shared
    cfg.npf_lastquad_hidden = shared
    cfg.npf_hidden = shared
    return cfg


def _device_from_config(cfg: DiagnosticConfig) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _move_plain_tensor_attrs(module: nn.Module, device: torch.device) -> None:
    """Move legacy plain tensor attributes that are not registered buffers."""
    module.to(device)
    for child in module.modules():
        for attr in ("mu", "std"):
            value = getattr(child, attr, None)
            if torch.is_tensor(value):
                setattr(child, attr, value.to(device))


def _canonical_p(p: Any):
    label = str(p).lower()
    if label in {"2", "2.0"}:
        return 2
    if label in {"inf", "infinity"}:
        return float("inf")
    raise ValueError(f"Unsupported p norm: {p!r}")


def _transport_cost(mode: str, x_adv: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
    mode = str(mode).lower()
    if mode in {"normalized_mse", "normalized", "legacy"}:
        return normalized_mse(x_adv, x_clean)
    if mode in {"pixel_l2_squared", "pixel_l2", "pixel"}:
        return pixel_l2_squared(x_adv, x_clean)
    raise ValueError(f"Unsupported transport cost: {mode!r}")


def _delta_stats(x_adv: torch.Tensor, x_clean: torch.Tensor) -> Dict[str, torch.Tensor]:
    delta_pix = to_pixel(x_adv) - to_pixel(x_clean)
    flat = delta_pix.reshape(delta_pix.size(0), -1)
    return {
        "pixel_l2": flat.norm(p=2, dim=1),
        "pixel_linf": flat.abs().max(dim=1).values,
        "pixel_l2_squared": flat.pow(2).sum(dim=1),
        "normalized_mse": normalized_mse(x_adv, x_clean),
    }


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _legacy_transport(
    model: LegacyInputConvexPotential,
    x: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    x_leaf = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        potential = model(x_leaf)
        grad = torch.autograd.grad(
            potential.sum(),
            x_leaf,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        grad = clamped_normalized_copy(grad)
    return grad if create_graph else grad.detach()


def _npf_transport(
    model: NPFInputConvexPotential,
    x: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    with torch.enable_grad():
        x_adv = npf_T_omega(x, model, create_graph=create_graph)
        x_adv = clamped_normalized_copy(x_adv)
    return x_adv if create_graph else x_adv.detach()


def _maybe_project(
    cfg: DiagnosticConfig,
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
) -> torch.Tensor:
    if not cfg.project_to_input_ball:
        return x_adv
    return project_normalized_to_input_ball(
        x_adv,
        x_clean,
        eps=float(cfg.inp_eps),
        p=_canonical_p(cfg.inp_p),
        geometry=cfg.input_pgd_geometry,
    )


def _transport_for_variant(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    cfg: DiagnosticConfig,
    *,
    create_graph: bool,
) -> torch.Tensor:
    if name == "legacy_icnn":
        x_adv = _legacy_transport(model, x, create_graph=create_graph)  # type: ignore[arg-type]
    elif name in {"npf_lastquad", "npf"}:
        x_adv = _npf_transport(model, x, create_graph=create_graph)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown adversary variant: {name!r}")
    return _maybe_project(cfg, x_adv, x)


def _make_legacy_icnn(cfg: DiagnosticConfig, device: torch.device) -> LegacyInputConvexPotential:
    return LegacyInputConvexPotential(
        input_dim=3 * 32 * 32,
        hidden_sizes=cfg.legacy_icnn_hidden,
        activation=cfg.legacy_icnn_activation,
        strong_convexity=cfg.legacy_icnn_strong_convexity,
        nonneg_init=cfg.legacy_icnn_init,
    ).to(device)


def _make_npf_lastquad(cfg: DiagnosticConfig, device: torch.device) -> NPFInputConvexPotential:
    model = NPFInputConvexPotential(
        input_dim=3 * 32 * 32,
        hidden_sizes=cfg.npf_lastquad_hidden,
        outer_rank=0,
        inner_rank=0,
        output_rank=cfg.npf_lastquad_output_rank,
        quadratic_mode="last_layer_diagonal",
        trainable_outer_quadratic=False,
        activation=cfg.npf_lastquad_activation,
        elu_alpha=cfg.npf_lastquad_elu_alpha,
        softplus_beta=cfg.npf_lastquad_softplus_beta,
        init_eps=cfg.npf_lastquad_init_eps,
        strong_convexity=cfg.npf_lastquad_strong_convexity,
        pos_weights=cfg.npf_lastquad_pos_weights,
        positive_weight_rectifier=cfg.npf_lastquad_positive_weight_rectifier,
    ).to(device)
    if cfg.npf_lastquad_identity_init:
        model.init_as_identity(residual_eps=cfg.npf_lastquad_init_eps)
    return model


def _make_npf(cfg: DiagnosticConfig, device: torch.device) -> NPFInputConvexPotential:
    model = NPFInputConvexPotential(
        input_dim=3 * 32 * 32,
        hidden_sizes=cfg.npf_hidden,
        outer_rank=cfg.npf_outer_rank,
        inner_rank=cfg.npf_inner_rank,
        output_rank=None,
        quadratic_mode="all_layers",
        trainable_outer_quadratic=True,
        activation=cfg.npf_activation,
        elu_alpha=cfg.npf_elu_alpha,
        softplus_beta=cfg.npf_softplus_beta,
        init_eps=cfg.npf_init_eps,
        strong_convexity=cfg.npf_strong_convexity,
        pos_weights=cfg.npf_pos_weights,
        positive_weight_rectifier=cfg.npf_positive_weight_rectifier,
    ).to(device)
    if cfg.npf_identity_init:
        model.init_as_identity(residual_eps=cfg.npf_init_eps)
    return model


def _make_adversaries(
    cfg: DiagnosticConfig,
    device: torch.device,
    variants: Sequence[str],
) -> Dict[str, nn.Module]:
    makers = {
        "legacy_icnn": _make_legacy_icnn,
        "npf_lastquad": _make_npf_lastquad,
        "npf": _make_npf,
    }
    return {name: makers[name](cfg, device) for name in variants}


def _load_batch(
    cfg: DiagnosticConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    train_loader, test_loader, _ = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        data_root=cfg.data_dir,
        augment_train=cfg.augment_train,
        pin_memory=device.type == "cuda",
        download=True,
        seed=cfg.seed,
    )
    loader = train_loader if cfg.split.lower() == "train" else test_loader
    for idx, (x, y) in enumerate(loader):
        if idx == int(cfg.batch_index):
            if cfg.max_samples > 0:
                x = x[: cfg.max_samples]
                y = y[: cfg.max_samples]
            return x.to(device), y.to(device)
    raise IndexError(f"batch_index={cfg.batch_index} is out of range for split={cfg.split!r}.")


def compute_cifar10_cp(
    data_dir: str,
    *,
    p: Any = "2",
    batch_size: int = 1024,
    num_workers: int = 0,
) -> float:
    """Compute C_p = E_train ||X_pixel||_p used by the WRM radius protocol."""
    p_norm = _canonical_p(p)
    loader, _ = get_cifar10_train_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        data_root=data_dir,
        augment_train=False,
        pin_memory=False,
        download=True,
        seed=0,
        shuffle=False,
    )
    total = 0
    norm_sum = 0.0
    for x_norm, _ in loader:
        x_pix = to_pixel(x_norm).clamp(0.0, 1.0)
        flat = x_pix.flatten(1)
        norms = flat.norm(p=2, dim=1) if p_norm == 2 else flat.abs().max(dim=1).values
        norm_sum += float(norms.sum().item())
        total += int(x_norm.size(0))
    if total <= 0:
        raise RuntimeError("CIFAR-10 C_p computation saw no samples.")
    return norm_sum / total


def resolve_effective_pgd_eps(cfg: DiagnosticConfig) -> Tuple[float, Optional[float], Optional[float]]:
    """Return (effective_eps, rho, C_p) for the configured PGD reference."""
    if not cfg.wrm_normalized_radius_protocol:
        return float(cfg.inp_eps), None, cfg.wrm_cp

    cp = cfg.wrm_cp
    if cp is None:
        cp = compute_cifar10_cp(
            cfg.data_dir,
            p=cfg.inp_p,
            batch_size=cfg.wrm_cp_batch_size,
            num_workers=cfg.wrm_cp_num_workers,
        )

    mode = str(cfg.wrm_pgd_eps_mode).lower()
    if mode in {"normalized", "normalized_radius", "rho"}:
        rho = float(cfg.inp_eps)
        return rho * float(cp), rho, float(cp)
    if mode in {"raw", "epsilon", "epsilon_adv"}:
        eps = float(cfg.inp_eps)
        return eps, eps / float(cp), float(cp)
    raise ValueError(f"Unsupported wrm_pgd_eps_mode={cfg.wrm_pgd_eps_mode!r}.")


def _objective_components(
    classifier: nn.Module,
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    y: torch.Tensor,
    cfg: DiagnosticConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = classifier(x_adv)
    primary = adversary_loss_per_sample(logits, y, use_margin=cfg.use_margin_loss)
    cost = _transport_cost(cfg.transport_cost, x_adv, x_clean)
    objective = primary - float(cfg.penalty_lambda) * cost
    return objective, primary, cost


def _cosine_to_reference(
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    pgd_delta_pix: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if pgd_delta_pix is None:
        return None
    delta_pix = to_pixel(x_adv) - to_pixel(x_clean)
    flat = delta_pix.reshape(delta_pix.size(0), -1)
    ref = pgd_delta_pix.reshape(pgd_delta_pix.size(0), -1).to(flat.device)
    denom = (flat.norm(p=2, dim=1) * ref.norm(p=2, dim=1)).clamp_min(1e-12)
    return (flat * ref).sum(dim=1) / denom


@torch.no_grad()
def _applied_transport(
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if mask is None:
        return x_adv.detach()
    out = x_clean.detach().clone()
    out[mask] = x_adv.detach()[mask]
    return out


def _record_metrics(
    *,
    variant: str,
    step: int,
    classifier: nn.Module,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor],
    cfg: DiagnosticConfig,
    pgd_delta_pix: Optional[torch.Tensor],
    f_start: Optional[float] = None,
    grad_norm: Optional[float] = None,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    with torch.enable_grad():
        x_raw = _transport_for_variant(variant, model, x, cfg, create_graph=False)
    x_applied = _applied_transport(x_raw, x, mask)
    with torch.no_grad():
        raw_obj, raw_primary, raw_cost = _objective_components(classifier, x_raw, x, y, cfg)
        app_obj, app_primary, app_cost = _objective_components(classifier, x_applied, x, y, cfg)
        logits_raw = classifier(x_raw)
        logits_applied = classifier(x_applied)
        stats_raw = _delta_stats(x_raw, x)
        stats_app = _delta_stats(x_applied, x)
        cos = _cosine_to_reference(x_applied, x, pgd_delta_pix)
        selected = mask
        valid_cos = cos is not None and torch.isfinite(cos).any()
        row = {
            "variant": variant,
            "step": int(step),
            "lambda_param": float(cfg.penalty_lambda),
            "transport_cost": str(cfg.transport_cost),
            "use_margin_loss": bool(cfg.use_margin_loss),
            "bb_alpha0": float(cfg.bb_alpha0),
            "bb_alpha_min": float(cfg.bb_alpha_min),
            "bb_alpha_max": float(cfg.bb_alpha_max),
            "bb_ls_c": float(cfg.bb_ls_c),
            "bb_ls_shrink": float(cfg.bb_ls_shrink),
            "bb_ls_max_steps": int(cfg.bb_ls_max_steps),
            "parametric_bb_max_grad_norm": float(cfg.parametric_bb_max_grad_norm),
            "objective_raw_masked": float(_masked_mean(raw_obj, selected).item()),
            "primary_raw_masked": float(_masked_mean(raw_primary, selected).item()),
            "cost_raw_masked": float(_masked_mean(raw_cost, selected).item()),
            "raw_normalized_mse_masked": float(
                _masked_mean(stats_raw["normalized_mse"], selected).item()
            ),
            "raw_pixel_l2_squared_masked": float(
                _masked_mean(stats_raw["pixel_l2_squared"], selected).item()
            ),
            "raw_avg_pixel_l2": float(stats_raw["pixel_l2"].mean().item()),
            "objective_applied_all": float(app_obj.mean().item()),
            "primary_applied_all": float(app_primary.mean().item()),
            "cost_applied_all": float(app_cost.mean().item()),
            "weighted_penalty_applied_all": float(cfg.penalty_lambda * app_cost.mean().item()),
            "raw_acc": float(logits_raw.argmax(dim=1).eq(y).float().mean().item()),
            "applied_acc": float(logits_applied.argmax(dim=1).eq(y).float().mean().item()),
            "raw_avg_l2": float(stats_raw["pixel_l2"].mean().item()),
            "applied_avg_l2": float(stats_app["pixel_l2"].mean().item()),
            "applied_max_l2": float(stats_app["pixel_l2"].max().item()),
            "applied_avg_linf": float(stats_app["pixel_linf"].mean().item()),
            "applied_avg_normalized_mse": float(stats_app["normalized_mse"].mean().item()),
            "applied_avg_pixel_l2_squared": float(stats_app["pixel_l2_squared"].mean().item()),
            "cos_to_pgd_mean": float(cos.mean().item()) if valid_cos else None,
            "cos_to_pgd_median": float(cos.median().item()) if valid_cos else None,
            "f_start": f_start,
            "grad_norm": grad_norm,
            "bb_alpha": alpha,
        }
        if mask is not None:
            row["attack_samples"] = int(mask.sum().item())
            row["total_samples"] = int(mask.numel())
        else:
            row["attack_samples"] = int(x.size(0))
            row["total_samples"] = int(x.size(0))
    return row


def _protocol_record(
    variant: str,
    cfg: DiagnosticConfig,
    *,
    effective_pgd_eps: Optional[float],
    normalized_radius: Optional[float],
    cp: Optional[float],
) -> Dict[str, Any]:
    return {
        "variant": variant,
        "omega_steps": int(cfg.omega_steps),
        "lambda_param": float(cfg.penalty_lambda),
        "transport_cost": str(cfg.transport_cost),
        "use_margin_loss": bool(cfg.use_margin_loss),
        "attack_clean_correct_only": bool(cfg.attack_clean_correct_only),
        "project_to_input_ball": bool(cfg.project_to_input_ball),
        "bb_alpha0": float(cfg.bb_alpha0),
        "bb_alpha_min": float(cfg.bb_alpha_min),
        "bb_alpha_max": float(cfg.bb_alpha_max),
        "bb_ls_c": float(cfg.bb_ls_c),
        "bb_ls_shrink": float(cfg.bb_ls_shrink),
        "bb_ls_max_steps": int(cfg.bb_ls_max_steps),
        "parametric_bb_max_grad_norm": float(cfg.parametric_bb_max_grad_norm),
        "input_pgd_geometry": str(cfg.input_pgd_geometry),
        "input_pgd_loss": str(cfg.input_pgd_loss),
        "inp_p": str(cfg.inp_p),
        "inp_eps_config": float(cfg.inp_eps),
        "effective_pgd_eps": effective_pgd_eps,
        "wrm_normalized_radius": normalized_radius,
        "wrm_cp": cp,
    }


def _hidden_sizes_for_variant(variant: str, cfg: DiagnosticConfig) -> Tuple[int, ...]:
    if variant == "legacy_icnn":
        return tuple(cfg.legacy_icnn_hidden)
    if variant == "npf_lastquad":
        return tuple(cfg.npf_lastquad_hidden)
    if variant == "npf":
        return tuple(cfg.npf_hidden)
    raise ValueError(f"Unknown adversary variant: {variant!r}")


def _architecture_record(
    variant: str,
    cfg: DiagnosticConfig,
    model: nn.Module,
) -> Dict[str, Any]:
    hidden = _hidden_sizes_for_variant(variant, cfg)
    outer_trainable = None
    pos_def = getattr(model, "pos_def_potential", None)
    if pos_def is not None:
        outer_trainable = any(p.requires_grad for p in pos_def.parameters())
    return {
        "variant": variant,
        "hidden_sizes": hidden,
        "hidden_depth": len(hidden),
        "hidden_max_width": max(hidden) if hidden else 0,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "outer_quadratic_trainable": outer_trainable,
    }


def _run_variant_inner_ascent(
    *,
    variant: str,
    model: nn.Module,
    classifier: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor],
    cfg: DiagnosticConfig,
    pgd_delta_pix: Optional[torch.Tensor],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    params = list(model.parameters())
    bb_state = BBArmijoState.create(
        alpha0=cfg.bb_alpha0,
        alpha_min=cfg.bb_alpha_min,
        alpha_max=cfg.bb_alpha_max,
        ls_c=cfg.bb_ls_c,
        ls_shrink=cfg.bb_ls_shrink,
        ls_max_steps=cfg.bb_ls_max_steps,
        reject_on_armijo_failure=True,
    )

    rows.append(
        _record_metrics(
            variant=variant,
            step=0,
            classifier=classifier,
            model=model,
            x=x,
            y=y,
            mask=mask,
            cfg=cfg,
            pgd_delta_pix=pgd_delta_pix,
        )
    )

    def omega_objective(create_graph: bool) -> torch.Tensor:
        x_adv = _transport_for_variant(variant, model, x, cfg, create_graph=create_graph)
        objective, _, _ = _objective_components(classifier, x_adv, x, y, cfg)
        return torch.nan_to_num(
            _masked_mean(objective, mask),
            nan=-1e12,
            posinf=-1e12,
            neginf=-1e12,
        )

    for step in range(1, int(cfg.omega_steps) + 1):
        _, bb_state, f_start, grad_norm = bb_armijo_step_params(
            params,
            omega_objective,
            bb_state,
            max_grad_norm=cfg.parametric_bb_max_grad_norm,
        )
        rows.append(
            _record_metrics(
                variant=variant,
                step=step,
                classifier=classifier,
                model=model,
                x=x,
                y=y,
                mask=mask,
                cfg=cfg,
                pgd_delta_pix=pgd_delta_pix,
                f_start=float(f_start) if math.isfinite(float(f_start)) else None,
                grad_norm=float(grad_norm) if math.isfinite(float(grad_norm)) else None,
                alpha=float(bb_state.alpha_prev),
            )
        )
    return rows


def _final_variant_outputs(
    variants: Iterable[str],
    models: Dict[str, nn.Module],
    classifier: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor],
    cfg: DiagnosticConfig,
    pgd_delta_pix: Optional[torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    images: Dict[str, torch.Tensor] = {"clean": x.detach()}
    sample_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        logits_clean = classifier(x)
        pred_clean = logits_clean.argmax(dim=1)
        clean_loss = F.cross_entropy(logits_clean, y, reduction="none")
        for i in range(x.size(0)):
            sample_rows.append(
                {
                    "sample": int(i),
                    "variant": "clean",
                    "label": int(y[i].item()),
                    "pred": int(pred_clean[i].item()),
                    "correct": bool(pred_clean[i].eq(y[i]).item()),
                    "loss": float(clean_loss[i].item()),
                    "pixel_l2": 0.0,
                    "pixel_linf": 0.0,
                    "normalized_mse": 0.0,
                    "pixel_l2_squared": 0.0,
                    "cos_to_pgd": None,
                    "attacked": bool(mask[i].item()) if mask is not None else True,
                }
            )

    for name in variants:
        with torch.enable_grad():
            x_raw = _transport_for_variant(name, models[name], x, cfg, create_graph=False)
        x_app = _applied_transport(x_raw, x, mask)
        images[name] = x_app.detach()
        with torch.no_grad():
            logits = classifier(x_app)
            pred = logits.argmax(dim=1)
            loss = F.cross_entropy(logits, y, reduction="none")
            stats = _delta_stats(x_app, x)
            cos = _cosine_to_reference(x_app, x, pgd_delta_pix)
            for i in range(x.size(0)):
                sample_rows.append(
                    {
                        "sample": int(i),
                        "variant": name,
                        "label": int(y[i].item()),
                        "pred": int(pred[i].item()),
                        "correct": bool(pred[i].eq(y[i]).item()),
                        "loss": float(loss[i].item()),
                        "pixel_l2": float(stats["pixel_l2"][i].item()),
                        "pixel_linf": float(stats["pixel_linf"][i].item()),
                        "normalized_mse": float(stats["normalized_mse"][i].item()),
                        "pixel_l2_squared": float(stats["pixel_l2_squared"][i].item()),
                        "cos_to_pgd": float(cos[i].item()) if cos is not None else None,
                        "attacked": bool(mask[i].item()) if mask is not None else True,
                    }
                )
    return images, sample_rows


def run_icnn_vs_npficnn_diagnostic(
    cfg: Optional[DiagnosticConfig] = None,
    *,
    variants: Sequence[str] = ("legacy_icnn", "npf_lastquad"),
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the isolated adversary comparison and return notebook-ready data."""
    cfg = cfg or DiagnosticConfig.from_run_local_env()
    variants = tuple(variants)
    set_deterministic(cfg.seed)
    device = _device_from_config(cfg)

    if verbose:
        print(f"[diagnostic] device={device} variants={variants}")
        print(f"[diagnostic] loading classifier: {cfg.pretrained_path}")
    classifier = load_pretrained_classifier(
        cfg.pretrained_path,
        strict=cfg.pretrained_strict,
        device=device,
    )
    _move_plain_tensor_attrs(classifier, device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_(False)

    if verbose:
        print(f"[diagnostic] loading CIFAR-10 {cfg.split} batch {cfg.batch_index}")
    x, y = _load_batch(cfg, device)
    with torch.no_grad():
        clean_logits = classifier(x)
        clean_pred = clean_logits.argmax(dim=1)
        clean_correct = clean_pred.eq(y)
    mask = clean_correct if cfg.attack_clean_correct_only else None

    effective_eps = None
    normalized_radius = None
    cp = cfg.wrm_cp
    x_pgd = None
    pgd_delta_pix = None
    if cfg.run_pgd_reference or cfg.project_to_input_ball:
        effective_eps, normalized_radius, cp = resolve_effective_pgd_eps(cfg)
    objective_cfg = (
        replace(cfg, inp_eps=float(effective_eps))
        if cfg.project_to_input_ball and effective_eps is not None
        else cfg
    )

    if cfg.run_pgd_reference:
        if verbose:
            rho_msg = f", rho={normalized_radius:.6g}, C_p={cp:.6g}" if cp else ""
            print(f"[diagnostic] PGD eps={effective_eps:.6g}{rho_msg}")
        x_pgd, pgd_delta_pix = input_pgd_best_adversary(
            classifier,
            x,
            y,
            p=_canonical_p(cfg.inp_p),
            eps=float(effective_eps),
            steps=cfg.inp_steps,
            step_size=cfg.inp_step_size if cfg.inp_step_size > 0 else None,
            restarts=cfg.inp_restarts,
            loss=cfg.input_pgd_loss,
            geometry=cfg.input_pgd_geometry,
        )

    if verbose:
        attacked = int(mask.sum().item()) if mask is not None else int(x.size(0))
        print(
            f"[diagnostic] clean_acc={clean_correct.float().mean().item():.3f} "
            f"attacked={attacked}/{x.size(0)}"
        )

    models = _make_adversaries(cfg, device, variants)
    history: List[Dict[str, Any]] = []
    with frozen_module(classifier, eval_mode=True):
        for name in variants:
            if verbose:
                print(f"[diagnostic] {name}: BB+Armijo omega ascent")
            models[name].train()
            rows = _run_variant_inner_ascent(
                variant=name,
                model=models[name],
                classifier=classifier,
                x=x,
                y=y,
                mask=mask,
                cfg=objective_cfg,
                pgd_delta_pix=pgd_delta_pix,
            )
            history.extend(rows)
            models[name].eval()

    images, sample_rows = _final_variant_outputs(
        variants,
        models,
        classifier,
        x,
        y,
        mask,
        objective_cfg,
        pgd_delta_pix,
    )
    if x_pgd is not None:
        images["pgd"] = x_pgd.detach()
        with torch.no_grad():
            logits = classifier(x_pgd)
            pred = logits.argmax(dim=1)
            loss = F.cross_entropy(logits, y, reduction="none")
            stats = _delta_stats(x_pgd, x)
            for i in range(x.size(0)):
                sample_rows.append(
                    {
                        "sample": int(i),
                        "variant": "pgd_reference",
                        "label": int(y[i].item()),
                        "pred": int(pred[i].item()),
                        "correct": bool(pred[i].eq(y[i]).item()),
                        "loss": float(loss[i].item()),
                        "pixel_l2": float(stats["pixel_l2"][i].item()),
                        "pixel_linf": float(stats["pixel_linf"][i].item()),
                        "normalized_mse": float(stats["normalized_mse"][i].item()),
                        "pixel_l2_squared": float(stats["pixel_l2_squared"][i].item()),
                        "cos_to_pgd": 1.0,
                        "attacked": bool(mask[i].item()) if mask is not None else True,
                    }
                )

    result = {
        "config": asdict(cfg),
        "objective_config": asdict(objective_cfg),
        "protocol": [
            _protocol_record(
                name,
                objective_cfg,
                effective_pgd_eps=effective_eps,
                normalized_radius=normalized_radius,
                cp=cp,
            )
            for name in variants
        ],
        "architecture": [
            _architecture_record(name, objective_cfg, models[name])
            for name in variants
        ],
        "variants": list(variants),
        "history": history,
        "samples": sample_rows,
        "images": {k: v.detach().cpu() for k, v in images.items()},
        "labels": y.detach().cpu(),
        "clean_correct_mask": clean_correct.detach().cpu(),
        "attack_mask": (mask.detach().cpu() if mask is not None else torch.ones_like(y).bool().cpu()),
        "effective_pgd_eps": effective_eps,
        "wrm_normalized_radius": normalized_radius,
        "wrm_cp": cp,
        "pgd_delta_pix": pgd_delta_pix.detach().cpu() if pgd_delta_pix is not None else None,
    }
    return result


def history_dataframe(result: Dict[str, Any]):
    import pandas as pd

    return pd.DataFrame(result["history"])


def samples_dataframe(result: Dict[str, Any]):
    import pandas as pd

    return pd.DataFrame(result["samples"])


def protocol_dataframe(result: Dict[str, Any]):
    import pandas as pd

    return pd.DataFrame(result["protocol"])


def architecture_dataframe(result: Dict[str, Any]):
    import pandas as pd

    return pd.DataFrame(result["architecture"])


def assert_shared_protocol(result: Dict[str, Any]) -> None:
    """Raise if shared optimizer/objective settings differ across variants."""
    protocol = protocol_dataframe(result)
    shared_cols = [
        "omega_steps",
        "lambda_param",
        "transport_cost",
        "use_margin_loss",
        "attack_clean_correct_only",
        "project_to_input_ball",
        "bb_alpha0",
        "bb_alpha_min",
        "bb_alpha_max",
        "bb_ls_c",
        "bb_ls_shrink",
        "bb_ls_max_steps",
        "parametric_bb_max_grad_norm",
        "input_pgd_geometry",
        "input_pgd_loss",
        "inp_p",
        "inp_eps_config",
        "effective_pgd_eps",
        "wrm_normalized_radius",
        "wrm_cp",
    ]
    mismatched = [
        col
        for col in shared_cols
        if col in protocol.columns and protocol[col].nunique(dropna=False) > 1
    ]
    if mismatched:
        raise AssertionError(f"Shared protocol mismatch across variants: {mismatched}")


def assert_shared_hidden_sizes(result: Dict[str, Any]) -> None:
    """Raise if compared adversaries do not use the same hidden tuple."""
    arch = architecture_dataframe(result)
    if "hidden_sizes" not in arch.columns:
        raise AssertionError("Architecture table is missing hidden_sizes.")
    if arch["hidden_sizes"].nunique(dropna=False) > 1:
        raise AssertionError(
            "Hidden-size mismatch across variants:\n"
            + arch[["variant", "hidden_sizes"]].to_string(index=False)
        )


def _image_np(x_norm: torch.Tensor):
    x_pix = to_pixel(x_norm.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)
    return x_pix.permute(1, 2, 0).detach().cpu().numpy()


def _delta_np(x_adv: torch.Tensor, x_clean: torch.Tensor, scale: float):
    delta = to_pixel(x_adv.unsqueeze(0)).squeeze(0) - to_pixel(x_clean.unsqueeze(0)).squeeze(0)
    vis = (0.5 + delta / max(2.0 * scale, 1e-12)).clamp(0.0, 1.0)
    return vis.permute(1, 2, 0).detach().cpu().numpy()


def plot_comparison_grid(
    result: Dict[str, Any],
    *,
    n: Optional[int] = None,
    save_path: Optional[str] = None,
    dpi: int = 160,
):
    """Plot clean, PGD, learned adversarial images, and perturbation maps."""
    import matplotlib.pyplot as plt

    images = result["images"]
    clean = images["clean"]
    variants = ["pgd"] if "pgd" in images else []
    variants.extend(result["variants"])
    n = min(int(n or result["config"].get("num_visual_samples", 8)), clean.size(0))

    max_abs = 1e-6
    for name in variants:
        delta = to_pixel(images[name]) - to_pixel(clean)
        max_abs = max(max_abs, float(delta[:n].abs().max().item()))

    cols = 1 + 2 * len(variants)
    fig, axes = plt.subplots(n, cols, figsize=(2.2 * cols, 2.25 * n), squeeze=False)
    labels = result["labels"]
    samples = samples_dataframe(result)
    for row in range(n):
        ax = axes[row, 0]
        ax.imshow(_image_np(clean[row]))
        ax.set_title(f"clean\ny={int(labels[row])}", fontsize=8)
        ax.axis("off")
        col = 1
        for name in variants:
            x_adv = images[name][row]
            delta_stats = samples[(samples["sample"] == row) & (samples["variant"].isin([name, "pgd_reference"]))]
            if name == "pgd":
                delta_stats = samples[
                    (samples["sample"] == row) & (samples["variant"] == "pgd_reference")
                ]
            pred = int(delta_stats["pred"].iloc[0]) if len(delta_stats) else -1
            l2 = float(delta_stats["pixel_l2"].iloc[0]) if len(delta_stats) else float("nan")

            axes[row, col].imshow(_image_np(x_adv))
            axes[row, col].set_title(f"{name}\npred={pred} l2={l2:.3f}", fontsize=8)
            axes[row, col].axis("off")
            col += 1
            axes[row, col].imshow(_delta_np(x_adv, clean[row], max_abs))
            axes[row, col].set_title(f"{name} delta", fontsize=8)
            axes[row, col].axis("off")
            col += 1
    fig.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def save_diagnostic_tables(result: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Path]:
    """Save history/sample CSVs and return their paths."""
    out = Path(output_dir or result["config"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / "icnn_vs_npficnn_history.csv"
    samples_path = out / "icnn_vs_npficnn_samples.csv"
    history_dataframe(result).to_csv(history_path, index=False)
    samples_dataframe(result).to_csv(samples_path, index=False)
    return {"history": history_path, "samples": samples_path}


def summarize_final_step(result: Dict[str, Any]):
    """Compact final-step table for quick notebook inspection."""
    df = history_dataframe(result)
    if df.empty:
        return df
    final = df.sort_values("step").groupby("variant", as_index=False).tail(1)
    cols = [
        "variant",
        "step",
        "objective_raw_masked",
        "primary_raw_masked",
        "cost_raw_masked",
        "raw_normalized_mse_masked",
        "raw_pixel_l2_squared_masked",
        "applied_acc",
        "applied_avg_l2",
        "applied_max_l2",
        "applied_avg_normalized_mse",
        "applied_avg_pixel_l2_squared",
        "cos_to_pgd_mean",
        "grad_norm",
        "bb_alpha",
    ]
    return final[[c for c in cols if c in final.columns]].reset_index(drop=True)
