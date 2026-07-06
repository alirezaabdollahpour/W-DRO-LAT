"""Post-hoc lambda sensitivity and composition analysis for NPF maps.

This module evaluates the paper idea:

    Does applying T_lambda several times look like T_lambda' for another
    value lambda'?

It intentionally does not train. It loads a grid of NPF/NPF-LastQuad
checkpoints, applies their learned transport maps on a held-out CIFAR-10
split, and writes CSV tables for:

* one-step lambda sensitivity,
* composed-map vs one-step target-map equivalence,
* per-lambda transport/attack summaries,
* checkpoint metadata and classifier drift diagnostics.

For the cleanest interpretation, use checkpoints produced with the same
frozen classifier and different lambda values. End-to-end adversarial
training checkpoints are still supported, but then the learned maps are
confounded with classifier drift; the metadata table makes that visible.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import get_cifar10_loaders
from .models import NPFInputConvexPotential, load_pretrained_classifier, npf_T_omega
from .utils import (
    adversary_loss_per_sample,
    clamped_normalized_copy,
    get_device,
    pixel_l2_squared,
    set_deterministic,
    to_pixel,
)


@dataclass(frozen=True)
class LoadedMap:
    lam: float
    checkpoint_path: Path
    algorithm: str
    epoch: Optional[int]
    robust_input_pgd_acc: Optional[float]
    config: Dict[str, Any]
    potential: NPFInputConvexPotential
    classifier_state: Optional[Mapping[str, torch.Tensor]]


class RunningMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def add(self, value: torch.Tensor | float, n: Optional[int] = None) -> None:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return
            self.total += float(value.detach().sum().item())
            self.count += int(value.numel())
            return
        if n is None:
            n = 1
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def result(self) -> float:
        return self.total / max(1, self.count)


class RunningMax:
    def __init__(self) -> None:
        self.value = float("-inf")
        self.count = 0

    def add(self, value: torch.Tensor | float, n: Optional[int] = None) -> None:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return
            self.value = max(self.value, float(value.detach().max().item()))
            self.count += int(value.numel())
            return
        self.value = max(self.value, float(value))
        self.count += int(n or 1)

    @property
    def result(self) -> float:
        return self.value if self.count > 0 else float("nan")


class RowAccumulator:
    def __init__(self) -> None:
        self._stats: MutableMapping[Tuple[Any, ...], Dict[str, RunningMean | RunningMax]] = {}

    def add(self, key: Tuple[Any, ...], metric: str, value: torch.Tensor | float, n: Optional[int] = None) -> None:
        row = self._stats.setdefault(key, {})
        stat = row.setdefault(metric, RunningMean())
        stat.add(value, n=n)

    def add_max(self, key: Tuple[Any, ...], metric: str, value: torch.Tensor | float, n: Optional[int] = None) -> None:
        row = self._stats.setdefault(key, {})
        stat = row.setdefault(metric, RunningMax())
        stat.add(value, n=n)

    def rows(self) -> Iterable[Tuple[Tuple[Any, ...], Dict[str, float]]]:
        for key in sorted(self._stats.keys(), key=_sort_key):
            row = self._stats[key]
            metrics = {name: stat.result for name, stat in sorted(row.items())}
            metrics["n_samples"] = float(max((stat.count for stat in row.values()), default=0))
            yield key, metrics


def _sort_key(key: Tuple[Any, ...]) -> Tuple[Any, ...]:
    out: List[Any] = []
    for value in key:
        if isinstance(value, float):
            out.append(value)
        else:
            out.append(str(value))
    return tuple(out)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, allow_nan=False))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _ordered_fields(rows: Sequence[Mapping[str, Any]], leading: Sequence[str]) -> List[str]:
    keys = {key for row in rows for key in row.keys()}
    fields = [key for key in leading if key in keys]
    fields.extend(sorted(keys - set(fields)))
    return fields


def _parse_lambda_path(spec: str) -> Tuple[float, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Expected LAMBDA=PATH, got {spec!r}."
        )
    lhs, rhs = spec.split("=", 1)
    try:
        lam = float(lhs)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid lambda in {spec!r}.") from exc
    path = Path(rhs).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Checkpoint does not exist: {path}")
    return lam, path


def _discover_checkpoints(
    roots: Sequence[Path],
    *,
    checkpoint_kind: str,
    seed: Optional[int],
    required_substrings: Sequence[str],
    allow_last_fallback: bool,
) -> List[Tuple[Optional[float], Path]]:
    discovered: List[Tuple[Optional[float], Path]] = []
    if checkpoint_kind == "best_robust":
        primary = "*_best_robust.pth"
        fallback = "*_last.pth"
    else:
        primary = "*_last.pth"
        fallback = "*_best_robust.pth"

    for root in roots:
        root = root.expanduser()
        paths = sorted(root.rglob(primary))
        if not paths and allow_last_fallback:
            paths = sorted(root.rglob(fallback))
        for path in paths:
            text = str(path)
            if required_substrings and not all(s in text for s in required_substrings):
                continue
            if seed is not None and f"seed_{seed}" not in path.parts:
                continue
            discovered.append((None, path))
    return discovered


def _checkpoint_lambda(ckpt: Mapping[str, Any], explicit_lam: Optional[float], path: Path) -> float:
    if explicit_lam is not None:
        return float(explicit_lam)
    cfg = ckpt.get("config") or {}
    if "lambda_param" not in cfg:
        raise ValueError(
            f"Could not infer lambda_param from checkpoint config; pass --checkpoint LAMBDA={path}."
        )
    return float(cfg["lambda_param"])


def _as_tuple(value: Any, default: Sequence[int]) -> Tuple[int, ...]:
    if value is None:
        return tuple(int(v) for v in default)
    if isinstance(value, str):
        return tuple(int(v) for v in value.replace(",", " ").split())
    return tuple(int(v) for v in value)


def _build_potential_from_config(config: Mapping[str, Any], algorithm: str) -> NPFInputConvexPotential:
    algorithm = str(algorithm or config.get("algorithm", "npf_lastquad")).lower()
    input_dim = int(3 * 32 * 32)
    if algorithm == "npf_lastquad":
        return NPFInputConvexPotential(
            input_dim=input_dim,
            hidden_sizes=_as_tuple(config.get("npf_lastquad_hidden"), (512, 512, 256, 128, 64)),
            outer_rank=0,
            inner_rank=0,
            output_rank=int(config.get("npf_lastquad_output_rank", 0)),
            quadratic_mode="last_layer_diagonal",
            trainable_outer_quadratic=False,
            activation=str(config.get("npf_lastquad_activation", "softplus")),
            elu_alpha=float(config.get("npf_lastquad_elu_alpha", 1.0)),
            softplus_beta=float(config.get("npf_lastquad_softplus_beta", 10.0)),
            init_eps=float(config.get("npf_lastquad_init_eps", 1e-2)),
            strong_convexity=float(config.get("npf_lastquad_strong_convexity", 1.0)),
            pos_weights=bool(config.get("npf_lastquad_pos_weights", True)),
            positive_weight_rectifier=str(
                config.get("npf_lastquad_positive_weight_rectifier", "relu")
            ),
        )
    if algorithm == "npf":
        return NPFInputConvexPotential(
            input_dim=input_dim,
            hidden_sizes=_as_tuple(config.get("npf_hidden"), (512, 512, 256, 128, 64)),
            outer_rank=int(config.get("npf_outer_rank", 8)),
            inner_rank=int(config.get("npf_inner_rank", 2)),
            quadratic_mode="all_layers",
            trainable_outer_quadratic=True,
            activation=str(config.get("npf_activation", "softplus")),
            elu_alpha=float(config.get("npf_elu_alpha", 1.0)),
            softplus_beta=float(config.get("npf_softplus_beta", 10.0)),
            init_eps=float(config.get("npf_init_eps", 1e-4)),
            strong_convexity=float(config.get("npf_strong_convexity", 1.0)),
            pos_weights=bool(config.get("npf_pos_weights", True)),
            positive_weight_rectifier=str(
                config.get("npf_positive_weight_rectifier", "relu")
            ),
        )
    raise ValueError(f"Unsupported checkpoint algorithm {algorithm!r}; expected npf or npf_lastquad.")


def _load_maps(
    specs: Sequence[Tuple[Optional[float], Path]],
    *,
    device: torch.device,
) -> List[LoadedMap]:
    loaded: List[LoadedMap] = []
    seen: Dict[float, Path] = {}
    for explicit_lam, path in specs:
        ckpt = torch.load(path, map_location="cpu")
        lam = _checkpoint_lambda(ckpt, explicit_lam, path)
        if lam in seen:
            raise ValueError(
                f"Duplicate lambda {lam:g}: {seen[lam]} and {path}. "
                "Use separate analyzer invocations for repeated seeds or variants."
            )
        seen[lam] = path
        cfg = dict(ckpt.get("config") or {})
        algorithm = str(ckpt.get("algorithm") or cfg.get("algorithm") or "npf_lastquad")
        potential = _build_potential_from_config(cfg, algorithm)
        potential.load_state_dict(ckpt["psi_omega"], strict=True)
        potential.to(device)
        potential.eval()
        for param in potential.parameters():
            param.requires_grad_(False)
        classifier_state = ckpt.get("classifier")
        loaded.append(
            LoadedMap(
                lam=lam,
                checkpoint_path=path,
                algorithm=algorithm,
                epoch=ckpt.get("epoch"),
                robust_input_pgd_acc=ckpt.get("robust_input_pgd_acc"),
                config=cfg,
                potential=potential,
                classifier_state=classifier_state,
            )
        )
    loaded.sort(key=lambda item: item.lam)
    return loaded


def _load_reference_classifier(
    maps: Sequence[LoadedMap],
    *,
    source: str,
    checkpoint_path: Optional[Path],
    pretrained_path: Optional[str],
    device: torch.device,
) -> Tuple[nn.Module, Optional[Mapping[str, torch.Tensor]], str]:
    if not maps:
        raise ValueError("No maps were loaded.")

    if source == "checkpoint":
        if checkpoint_path is None:
            raise ValueError("--classifier-source=checkpoint requires --classifier-checkpoint.")
        ckpt = torch.load(checkpoint_path.expanduser(), map_location="cpu")
        cfg = ckpt.get("config") or maps[0].config
        state = ckpt.get("classifier")
        if state is None:
            raise ValueError(f"Checkpoint has no classifier state: {checkpoint_path}")
        label = str(checkpoint_path)
    elif source == "pretrained":
        cfg = maps[0].config
        state = None
        label = "pretrained"
    else:
        cfg = maps[0].config
        state = maps[0].classifier_state
        if state is None:
            raise ValueError(f"First checkpoint has no classifier state: {maps[0].checkpoint_path}")
        label = f"first_checkpoint_lambda_{maps[0].lam:g}"

    path = pretrained_path or str(cfg.get("pretrained_path") or "")
    if not path:
        raise ValueError(
            "Could not infer the classifier architecture checkpoint. Pass --pretrained-path."
        )
    classifier = load_pretrained_classifier(
        pretrained_path=path,
        num_classes=10,
        strict=bool(cfg.get("pretrained_strict", False)),
        device=device,
    ).to(device)
    if state is not None:
        classifier.load_state_dict(state, strict=True)
    classifier.to(device)
    _move_legacy_classifier_tensors(classifier, device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_(False)
    return classifier, state, label


def _move_legacy_classifier_tensors(classifier: nn.Module, device: torch.device) -> None:
    """Move non-buffer tensor attributes used by the legacy CIFAR models."""
    for module in classifier.modules():
        for name in ("mu", "std"):
            value = getattr(module, name, None)
            if torch.is_tensor(value):
                setattr(module, name, value.to(device))


def _state_dict_drift(
    state: Optional[Mapping[str, torch.Tensor]],
    ref_state: Optional[Mapping[str, torch.Tensor]],
) -> Dict[str, float]:
    if state is None or ref_state is None:
        return {
            "classifier_l2_to_reference": float("nan"),
            "classifier_rms_to_reference": float("nan"),
            "classifier_linf_to_reference": float("nan"),
        }
    total_sq = 0.0
    total_numel = 0
    max_abs = 0.0
    for key, value in state.items():
        ref = ref_state.get(key)
        if ref is None or not torch.is_tensor(value) or not torch.is_tensor(ref):
            continue
        if not torch.is_floating_point(value):
            continue
        diff = value.detach().cpu().float() - ref.detach().cpu().float()
        total_sq += float(diff.pow(2).sum().item())
        total_numel += int(diff.numel())
        max_abs = max(max_abs, float(diff.abs().max().item()))
    l2 = math.sqrt(total_sq)
    rms = math.sqrt(total_sq / max(1, total_numel))
    return {
        "classifier_l2_to_reference": l2,
        "classifier_rms_to_reference": rms,
        "classifier_linf_to_reference": max_abs,
    }


def _transport(model: NPFInputConvexPotential, x: torch.Tensor) -> torch.Tensor:
    x_adv = npf_T_omega(x, model, create_graph=False)
    return clamped_normalized_copy(x_adv)


def _pixel_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return pixel_l2_squared(x, y).clamp_min(0.0).sqrt()


def _delta_cosine(x_adv: torch.Tensor, target_adv: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
    delta_a = (to_pixel(x_adv) - to_pixel(x_clean)).reshape(x_clean.size(0), -1)
    delta_b = (to_pixel(target_adv) - to_pixel(x_clean)).reshape(x_clean.size(0), -1)
    denom = delta_a.norm(p=2, dim=1) * delta_b.norm(p=2, dim=1)
    cosine = (delta_a * delta_b).sum(dim=1) / denom.clamp_min(1e-12)
    zero_mask = denom <= 1e-12
    if zero_mask.any():
        cosine = cosine.masked_fill(zero_mask, 1.0)
    return cosine


def _margin_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return adversary_loss_per_sample(logits, y, use_margin=True)


def _batch_attack_metrics(
    classifier: nn.Module,
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    y: torch.Tensor,
    clean_pred: torch.Tensor,
    eps_reference: float,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        logits = classifier(x_adv)
        ce = F.cross_entropy(logits, y, reduction="none")
        margin = _margin_loss(logits, y)
        pred = logits.argmax(dim=1)
        cost = pixel_l2_squared(x_adv, x_clean)
        l2 = cost.clamp_min(0.0).sqrt()
        linf = (to_pixel(x_adv) - to_pixel(x_clean)).abs().reshape(x_clean.size(0), -1).max(dim=1)[0]
        correct = (pred == y).float()
        flip = (pred != clean_pred).float()
        within_eps = (l2 <= float(eps_reference)).float()
    return {
        "ce": ce,
        "margin": margin,
        "acc": correct,
        "flip_rate": flip,
        "transport_cost": cost,
        "transport_l2": l2,
        "transport_linf": linf,
        "within_eps_reference": within_eps,
    }


def _add_attack_metrics(
    acc: RowAccumulator,
    key: Tuple[Any, ...],
    metrics: Mapping[str, torch.Tensor],
    prefix: str,
    *,
    lambda_for_objective: Optional[float] = None,
) -> None:
    n = int(metrics["ce"].numel())
    for name, value in metrics.items():
        acc.add(key, f"{prefix}_{name}_mean", value)
        if name in {"transport_l2", "transport_linf"}:
            acc.add_max(key, f"{prefix}_{name}_max", value)
    if lambda_for_objective is not None:
        objective_ce = metrics["ce"] - float(lambda_for_objective) * metrics["transport_cost"]
        objective_margin = metrics["margin"] - float(lambda_for_objective) * metrics["transport_cost"]
        acc.add(key, f"{prefix}_objective_ce_mean", objective_ce)
        acc.add(key, f"{prefix}_objective_margin_mean", objective_margin)
    acc.add(key, "n_samples", float(n), n=1)


def _selected_loader(args: argparse.Namespace):
    train_loader, test_loader, _ = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_dir,
        augment_train=False,
        pin_memory=torch.cuda.is_available(),
        download=True,
        seed=args.seed,
        world_size=1,
        rank=0,
    )
    return train_loader if args.split == "train" else test_loader


def _format_lam(value: float) -> str:
    return f"{value:.12g}"


def _image_np(x_norm: torch.Tensor):
    x_pix = to_pixel(x_norm.unsqueeze(0)).squeeze(0).detach().cpu()
    return x_pix.clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def _perturbation_np(x_adv_norm: torch.Tensor, x_clean_norm: torch.Tensor):
    adv = to_pixel(x_adv_norm.unsqueeze(0)).squeeze(0).detach().cpu()
    clean = to_pixel(x_clean_norm.unsqueeze(0)).squeeze(0).detach().cpu()
    delta = adv - clean
    return delta.pow(2).sum(dim=0).clamp_min(0.0).sqrt().numpy()


def _sample_l2_linf(x_adv_norm: torch.Tensor, x_clean_norm: torch.Tensor) -> Tuple[float, float]:
    adv = to_pixel(x_adv_norm.unsqueeze(0)).squeeze(0).detach().cpu()
    clean = to_pixel(x_clean_norm.unsqueeze(0)).squeeze(0).detach().cpu()
    delta = adv - clean
    l2 = float(delta.reshape(-1).norm(p=2).item())
    linf = float(delta.abs().max().item())
    return l2, linf


def _write_plot_figures(
    output_dir: Path,
    records: Sequence[Dict[str, Any]],
    maps: Sequence[LoadedMap],
    steps: Sequence[int],
    *,
    dpi: int,
    image_format: str,
    cmap: str,
) -> List[str]:
    if not records:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    lambdas = [item.lam for item in maps]
    rows = len(lambdas)
    image_cols = 1 + len(steps)
    pert_cols = len(steps)

    for record in records:
        sample_id = int(record["sample_index"])
        y = int(record["label"])
        clean_pred = int(record["clean_pred"])
        clean = record["clean"]
        compositions: Mapping[Tuple[float, int], torch.Tensor] = record["compositions"]

        fig_w = max(6.0, 1.55 * image_cols)
        fig_h = max(2.4, 1.55 * rows)
        fig, axes = plt.subplots(
            rows,
            image_cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(
            f"NPF-LastQuad repeated transports | sample {sample_id} | y={y}, clean pred={clean_pred}",
            fontsize=10,
        )
        for r, lam in enumerate(lambdas):
            for c in range(image_cols):
                ax = axes[r][c]
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.imshow(_image_np(clean), interpolation="nearest")
                    ax.set_title("clean", fontsize=8)
                    ax.set_ylabel(f"lambda={_format_lam(lam)}", fontsize=8)
                    continue
                step = int(steps[c - 1])
                adv = compositions[(lam, step)]
                ax.imshow(_image_np(adv), interpolation="nearest")
                if r == 0:
                    title = r"$T_\lambda$" if step == 1 else rf"$T_\lambda^{step}$"
                    ax.set_title(title, fontsize=8)
        path = figures_dir / f"sample_{sample_id:04d}_adversarial_grid.{image_format}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

        perturbations = [
            _perturbation_np(compositions[(lam, step)], clean)
            for lam in lambdas
            for step in steps
        ]
        vmax = max(float(p.max()) for p in perturbations) if perturbations else 1.0
        vmax = max(vmax, 1e-8)
        fig_w = max(5.2, 1.65 * pert_cols + 0.5)
        fig_h = max(2.4, 1.55 * rows)
        fig, axes = plt.subplots(
            rows,
            pert_cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(
            f"Pixel-space perturbation magnitude | sample {sample_id}",
            fontsize=10,
        )
        last_im = None
        for r, lam in enumerate(lambdas):
            for c, step in enumerate(steps):
                ax = axes[r][c]
                ax.set_xticks([])
                ax.set_yticks([])
                adv = compositions[(lam, int(step))]
                pert = _perturbation_np(adv, clean)
                last_im = ax.imshow(
                    pert,
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=0.0,
                    vmax=vmax,
                )
                l2, linf = _sample_l2_linf(adv, clean)
                ax.text(
                    0.03,
                    0.96,
                    f"L2 {l2:.3f}\nLinf {linf:.3f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    color="white",
                    fontsize=6,
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"},
                )
                if r == 0:
                    ax.set_title(f"m={int(step)}", fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"lambda={_format_lam(lam)}", fontsize=8)
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.82, label="pixel L2 per location")
        path = figures_dir / f"sample_{sample_id:04d}_perturbation_grid.{image_format}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    return saved


def _write_metadata(
    output_dir: Path,
    maps: Sequence[LoadedMap],
    ref_state: Optional[Mapping[str, torch.Tensor]],
    classifier_label: str,
) -> None:
    rows = []
    for item in maps:
        drift = _state_dict_drift(item.classifier_state, ref_state)
        rows.append(
            {
                "lambda": _format_lam(item.lam),
                "checkpoint": str(item.checkpoint_path),
                "algorithm": item.algorithm,
                "epoch": item.epoch,
                "robust_input_pgd_acc": item.robust_input_pgd_acc,
                "use_margin_loss": item.config.get("use_margin_loss"),
                "npf_inner_optimizer": item.config.get("npf_inner_optimizer"),
                "omega_steps_per_batch": item.config.get("omega_steps_per_batch"),
                "npf_lastquad_hidden": " ".join(str(v) for v in _as_tuple(item.config.get("npf_lastquad_hidden"), ())),
                "npf_lastquad_activation": item.config.get("npf_lastquad_activation"),
                "classifier_reference": classifier_label,
                **drift,
            }
        )
    _write_csv(
        output_dir / "checkpoint_metadata.csv",
        [
            "lambda",
            "checkpoint",
            "algorithm",
            "epoch",
            "robust_input_pgd_acc",
            "use_margin_loss",
            "npf_inner_optimizer",
            "omega_steps_per_batch",
            "npf_lastquad_hidden",
            "npf_lastquad_activation",
            "classifier_reference",
            "classifier_l2_to_reference",
            "classifier_rms_to_reference",
            "classifier_linf_to_reference",
        ],
        rows,
    )


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    set_deterministic(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    explicit_specs = [_parse_lambda_path(spec) for spec in args.checkpoint]
    discovered_specs = _discover_checkpoints(
        [Path(root) for root in args.runs_root],
        checkpoint_kind=args.checkpoint_kind,
        seed=args.seed if args.seed_filter else None,
        required_substrings=args.run_name_contains,
        allow_last_fallback=args.allow_last_fallback,
    )
    specs: List[Tuple[Optional[float], Path]] = [(lam, path) for lam, path in explicit_specs]
    specs.extend(discovered_specs)
    if not specs:
        raise ValueError(
            "No checkpoints supplied. Use --checkpoint LAMBDA=PATH or --runs-root."
        )

    # De-duplicate exact paths before loading.
    deduped: Dict[Path, Optional[float]] = {}
    for lam, path in specs:
        deduped[path.resolve()] = lam
    maps = _load_maps([(lam, path) for path, lam in deduped.items()], device=device)
    if len(maps) < 2:
        raise ValueError("Need at least two lambda checkpoints for an equivalence ablation.")

    classifier, ref_state, classifier_label = _load_reference_classifier(
        maps,
        source=args.classifier_source,
        checkpoint_path=Path(args.classifier_checkpoint) if args.classifier_checkpoint else None,
        pretrained_path=args.pretrained_path,
        device=device,
    )
    _write_metadata(output_dir, maps, ref_state, classifier_label)

    loader = _selected_loader(args)
    steps = sorted(set(int(s) for s in args.composition_steps))
    if not steps or min(steps) < 1:
        raise ValueError("--composition-steps must contain positive integers.")
    max_step = max(steps)

    map_summary = RowAccumulator()
    pairwise = RowAccumulator()
    sensitivity = RowAccumulator()
    plot_records: List[Dict[str, Any]] = []

    total_seen = 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    progress = tqdm(loader, total=total_batches, desc="lambda-ablation", leave=True)
    for x_cpu, y_cpu in progress:
        if args.max_samples is not None and total_seen >= args.max_samples:
            break
        if args.max_samples is not None:
            remaining = int(args.max_samples) - total_seen
            if remaining <= 0:
                break
            x_cpu = x_cpu[:remaining]
            y_cpu = y_cpu[:remaining]
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        batch_n = int(x.size(0))
        total_seen += batch_n

        with torch.no_grad():
            clean_logits = classifier(x)
            clean_pred = clean_logits.argmax(dim=1)
            clean_ce = F.cross_entropy(clean_logits, y, reduction="none")
            clean_correct = (clean_pred == y).float()

        clean_key = ("clean", 0)
        map_summary.add(clean_key, "clean_ce_mean", clean_ce)
        map_summary.add(clean_key, "clean_acc_mean", clean_correct)
        map_summary.add(clean_key, "n_samples", float(batch_n), n=1)

        plot_batch_records: List[Tuple[int, Dict[str, Any]]] = []
        plots_remaining = max(0, int(args.plot_samples) - len(plot_records))
        if plots_remaining > 0:
            for local_idx in range(min(batch_n, plots_remaining)):
                record = {
                    "sample_index": total_seen - batch_n + local_idx,
                    "clean": x[local_idx].detach().cpu(),
                    "label": int(y[local_idx].item()),
                    "clean_pred": int(clean_pred[local_idx].item()),
                    "compositions": {},
                }
                plot_records.append(record)
                plot_batch_records.append((local_idx, record))

        one_step: Dict[float, torch.Tensor] = {}
        one_step_metrics: Dict[float, Dict[str, torch.Tensor]] = {}
        for target in maps:
            x_target = _transport(target.potential, x).detach()
            one_step[target.lam] = x_target
            metrics = _batch_attack_metrics(
                classifier,
                x_target,
                x,
                y,
                clean_pred,
                eps_reference=args.eps_reference,
            )
            one_step_metrics[target.lam] = metrics
            key = (_format_lam(target.lam), 1)
            _add_attack_metrics(
                map_summary,
                key,
                metrics,
                "map",
                lambda_for_objective=target.lam,
            )

        for source in maps:
            z = x
            prev_z = x
            for step in range(1, max_step + 1):
                z = _transport(source.potential, z).detach()
                if step not in steps:
                    prev_z = z
                    continue

                comp_metrics = _batch_attack_metrics(
                    classifier,
                    z,
                    x,
                    y,
                    clean_pred,
                    eps_reference=args.eps_reference,
                )
                comp_key = (_format_lam(source.lam), step)
                _add_attack_metrics(
                    map_summary,
                    comp_key,
                    comp_metrics,
                    "composition",
                    lambda_for_objective=source.lam,
                )
                map_summary.add(comp_key, "composition_step_drift_l2_mean", _pixel_l2(z, prev_z))
                map_summary.add(comp_key, "composition_step_drift_cost_mean", pixel_l2_squared(z, prev_z))
                for local_idx, record in plot_batch_records:
                    record["compositions"][(source.lam, step)] = z[local_idx].detach().cpu()

                for target in maps:
                    target_adv = one_step[target.lam]
                    target_metrics = one_step_metrics[target.lam]
                    key = (_format_lam(source.lam), step, _format_lam(target.lam))
                    map_dist_sq = pixel_l2_squared(z, target_adv)
                    map_dist_l2 = map_dist_sq.clamp_min(0.0).sqrt()
                    cost_gap = (comp_metrics["transport_cost"] - target_metrics["transport_cost"]).abs()
                    l2_gap = (comp_metrics["transport_l2"] - target_metrics["transport_l2"]).abs()
                    cosine = _delta_cosine(z, target_adv, x)

                    pairwise.add(key, "map_distance_l2_mean", map_dist_l2)
                    pairwise.add(key, "map_distance_l2sq_mean", map_dist_sq)
                    pairwise.add(key, "delta_cosine_mean", cosine)
                    pairwise.add(key, "transport_cost_abs_gap_mean", cost_gap)
                    pairwise.add(key, "transport_l2_abs_gap_mean", l2_gap)
                    pairwise.add(key, "composed_ce_mean", comp_metrics["ce"])
                    pairwise.add(key, "composed_margin_mean", comp_metrics["margin"])
                    pairwise.add(key, "composed_acc_mean", comp_metrics["acc"])
                    pairwise.add(key, "composed_transport_cost_mean", comp_metrics["transport_cost"])
                    pairwise.add(key, "target_ce_mean", target_metrics["ce"])
                    pairwise.add(key, "target_margin_mean", target_metrics["margin"])
                    pairwise.add(key, "target_acc_mean", target_metrics["acc"])
                    pairwise.add(key, "target_transport_cost_mean", target_metrics["transport_cost"])
                    pairwise.add(
                        key,
                        "composed_objective_ce_at_target_lambda_mean",
                        comp_metrics["ce"] - target.lam * comp_metrics["transport_cost"],
                    )
                    pairwise.add(
                        key,
                        "target_objective_ce_at_target_lambda_mean",
                        target_metrics["ce"] - target.lam * target_metrics["transport_cost"],
                    )
                    pairwise.add(
                        key,
                        "composed_objective_margin_at_target_lambda_mean",
                        comp_metrics["margin"] - target.lam * comp_metrics["transport_cost"],
                    )
                    pairwise.add(
                        key,
                        "target_objective_margin_at_target_lambda_mean",
                        target_metrics["margin"] - target.lam * target_metrics["transport_cost"],
                    )
                    pairwise.add(key, "n_samples", float(batch_n), n=1)

                    if step == 1:
                        sens_key = (_format_lam(source.lam), _format_lam(target.lam))
                        sensitivity.add(sens_key, "map_distance_l2_mean", map_dist_l2)
                        sensitivity.add(sens_key, "map_distance_l2sq_mean", map_dist_sq)
                        sensitivity.add(sens_key, "delta_cosine_mean", cosine)
                        sensitivity.add(sens_key, "transport_cost_abs_gap_mean", cost_gap)
                        sensitivity.add(sens_key, "transport_l2_abs_gap_mean", l2_gap)
                        sensitivity.add(sens_key, "n_samples", float(batch_n), n=1)
                prev_z = z
        progress.set_postfix({"samples": total_seen})
    progress.close()

    map_summary_rows = [
        {
            "lambda_or_clean": key[0],
            "composition_steps": key[1],
            **metrics,
        }
        for key, metrics in map_summary.rows()
    ]
    pairwise_rows = [
        {
            "source_lambda": key[0],
            "composition_steps": key[1],
            "target_lambda": key[2],
            **metrics,
        }
        for key, metrics in pairwise.rows()
    ]
    sensitivity_rows = [
        {
            "source_lambda": key[0],
            "target_lambda": key[1],
            **metrics,
        }
        for key, metrics in sensitivity.rows()
    ]

    best_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in pairwise_rows:
        grouped.setdefault((str(row["source_lambda"]), int(row["composition_steps"])), []).append(row)
    for (source_lam, step), rows in sorted(grouped.items(), key=lambda kv: (float(kv[0][0]), kv[0][1])):
        by_map = min(rows, key=lambda r: float(r["map_distance_l2sq_mean"]))
        by_cost = min(rows, key=lambda r: float(r["transport_cost_abs_gap_mean"]))
        by_objective_gap = min(
            rows,
            key=lambda r: abs(
                float(r["composed_objective_ce_at_target_lambda_mean"])
                - float(r["target_objective_ce_at_target_lambda_mean"])
            ),
        )
        best_rows.append(
            {
                "source_lambda": source_lam,
                "composition_steps": step,
                "best_target_lambda_by_map": by_map["target_lambda"],
                "best_map_distance_l2_mean": by_map["map_distance_l2_mean"],
                "best_map_distance_l2sq_mean": by_map["map_distance_l2sq_mean"],
                "best_delta_cosine_mean": by_map["delta_cosine_mean"],
                "best_target_lambda_by_transport_cost": by_cost["target_lambda"],
                "best_transport_cost_abs_gap_mean": by_cost["transport_cost_abs_gap_mean"],
                "best_target_lambda_by_ce_objective_gap": by_objective_gap["target_lambda"],
                "best_ce_objective_gap": abs(
                    float(by_objective_gap["composed_objective_ce_at_target_lambda_mean"])
                    - float(by_objective_gap["target_objective_ce_at_target_lambda_mean"])
                ),
                "composed_ce_mean": by_map["composed_ce_mean"],
                "composed_acc_mean": by_map["composed_acc_mean"],
                "composed_transport_cost_mean": by_map["composed_transport_cost_mean"],
                "n_samples": by_map["n_samples"],
            }
        )

    _write_csv(
        output_dir / "map_summary.csv",
        _ordered_fields(map_summary_rows, ["lambda_or_clean", "composition_steps", "n_samples"]),
        map_summary_rows,
    )
    _write_csv(
        output_dir / "composition_pairwise.csv",
        _ordered_fields(
            pairwise_rows,
            ["source_lambda", "composition_steps", "target_lambda", "n_samples"],
        ),
        pairwise_rows,
    )
    _write_csv(
        output_dir / "lambda_sensitivity.csv",
        _ordered_fields(sensitivity_rows, ["source_lambda", "target_lambda", "n_samples"]),
        sensitivity_rows,
    )
    _write_csv(
        output_dir / "composition_best_match.csv",
        [
            "source_lambda",
            "composition_steps",
            "best_target_lambda_by_map",
            "best_map_distance_l2_mean",
            "best_map_distance_l2sq_mean",
            "best_delta_cosine_mean",
            "best_target_lambda_by_transport_cost",
            "best_transport_cost_abs_gap_mean",
            "best_target_lambda_by_ce_objective_gap",
            "best_ce_objective_gap",
            "composed_ce_mean",
            "composed_acc_mean",
            "composed_transport_cost_mean",
            "n_samples",
        ],
        best_rows,
    )

    plot_files: List[str] = []
    if int(args.plot_samples) > 0:
        plot_files = _write_plot_figures(
            output_dir,
            plot_records,
            maps,
            steps,
            dpi=int(args.plot_dpi),
            image_format=str(args.plot_format),
            cmap=str(args.plot_cmap),
        )

    summary = {
        "output_dir": output_dir,
        "device": str(device),
        "split": args.split,
        "samples": total_seen,
        "composition_steps": steps,
        "eps_reference": args.eps_reference,
        "classifier_reference": classifier_label,
        "lambdas": [item.lam for item in maps],
        "checkpoints": [str(item.checkpoint_path) for item in maps],
        "files": {
            "checkpoint_metadata": str(output_dir / "checkpoint_metadata.csv"),
            "map_summary": str(output_dir / "map_summary.csv"),
            "lambda_sensitivity": str(output_dir / "lambda_sensitivity.csv"),
            "composition_pairwise": str(output_dir / "composition_pairwise.csv"),
            "composition_best_match": str(output_dir / "composition_best_match.csv"),
            "figures_dir": str(output_dir / "figures") if plot_files else None,
        },
        "plot_files": plot_files,
    }
    _write_json(output_dir / "summary.json", summary)

    print()
    print(f"[lambda-ablation] wrote {output_dir}")
    print(f"[lambda-ablation] samples={total_seen} lambdas={[ _format_lam(m.lam) for m in maps ]}")
    if plot_files:
        print(f"[lambda-ablation] figures={output_dir / 'figures'} ({len(plot_files)} files)")
    print("[lambda-ablation] best map matches:")
    for row in best_rows:
        print(
            "  "
            f"T_{row['source_lambda']}^{row['composition_steps']} -> "
            f"T_{row['best_target_lambda_by_map']} "
            f"(mean L2={float(row['best_map_distance_l2_mean']):.4f}, "
            f"cos={float(row['best_delta_cosine_mean']):.3f})"
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate lambda sensitivity and repeated-map equivalence for NPF transports."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LAMBDA=PATH",
        help="Explicit checkpoint. Repeat for each lambda.",
    )
    parser.add_argument(
        "--runs-root",
        action="append",
        default=[],
        metavar="PATH",
        help="Recursively discover npf_lastquad checkpoints below this root.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=["best_robust", "last"],
        default="best_robust",
        help="Which checkpoint name to discover when using --runs-root.",
    )
    parser.add_argument(
        "--allow-last-fallback",
        action="store_true",
        help="If discovery finds no requested checkpoint kind under a root, try the other kind.",
    )
    parser.add_argument(
        "--run-name-contains",
        action="append",
        default=[],
        help="Only use discovered checkpoint paths containing this substring. Repeat for an AND filter.",
    )
    parser.add_argument(
        "--seed-filter",
        action="store_true",
        help="When discovering, require paths to contain seed_<seed>.",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--pretrained-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--composition-steps",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Evaluate T_lambda^m for these m values.",
    )
    parser.add_argument(
        "--eps-reference",
        type=float,
        default=0.5,
        help="Reference L2 radius used only for reporting within_eps_reference.",
    )
    parser.add_argument(
        "--classifier-source",
        choices=["first_checkpoint", "checkpoint", "pretrained"],
        default="first_checkpoint",
        help=(
            "Classifier used for CE/margin/accuracy metrics. Map distances do not "
            "depend on this choice. Use pretrained or a fixed checkpoint for the "
            "least-confounded lambda analysis."
        ),
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default="",
        help="Checkpoint whose classifier state is used when --classifier-source=checkpoint.",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=0,
        help=(
            "Save visual grids for the first N evaluated samples. Each sample gets "
            "an adversarial-image grid and a perturbation-magnitude grid."
        ),
    )
    parser.add_argument(
        "--plot-format",
        choices=["png", "pdf"],
        default="png",
        help="Figure file format for --plot-samples.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=220,
        help="Figure DPI for --plot-samples.",
    )
    parser.add_argument(
        "--plot-cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap for perturbation heatmaps.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
