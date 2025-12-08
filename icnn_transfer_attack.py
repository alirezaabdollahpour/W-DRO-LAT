#!/usr/bin/env python3
"""
Evaluate transferability of ICNN-generated adversarial examples across pretrained models.

This script:
1) Loads an ICNN checkpoint produced by ``pretrained_INPUT_icnn.py``.
2) Reconstructs the ICNN architecture (including normalization/clamping behaviour).
3) Uses the ICNN transport map T(x) = ∇φ(x) to craft adversarial examples for a source
   classifier (e.g., R2.pth).
4) Evaluates how those adversarial examples transfer to other pretrained models (e.g., R3/R4).
5) Optionally saves clean/adv pairs and metadata for later reuse.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from pretrained_INPUT_icnn import InputConvexPotential, adversarial_pushforward
from pretrained_LAT import load_pretrained_resnet18
from utils import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    get_cifar10_loader,
    get_device,
    looks_like_state_dict,
    set_deterministic,
    to_pixel,
    unwrap_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attack pretrained models using an ICNN transport map and evaluate transfer."
    )
    parser.add_argument(
        "--icnn-checkpoint",
        required=True,
        help="Path to the ICNN checkpoint saved by pretrained_INPUT_icnn.py (can be *_icnn.pth).",
    )
    parser.add_argument(
        "--source-model",
        required=True,
        help="Checkpoint for the model the ICNN was trained against (e.g., ResNet_checkpoints/R2.pth).",
    )
    parser.add_argument(
        "--target-models",
        nargs="*",
        default=[],
        help="Additional model checkpoints to test transfer (e.g., R3.pth R4.pth).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Dataset split to attack.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Minibatch size for loading data and generating adversarial examples.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to attack (default: full split).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        help="Where to cache/download CIFAR-10.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/icnn_transfer",
        help="Directory to store outputs (adversarial tensors + metrics).",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional explicit path for saved adversarial set (.pt). Overrides --out-dir/run_id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for dataloader shuffling/augmentations.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name to tag outputs; defaults to timestamp.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing adversarial examples to disk (metrics still printed).",
    )
    return parser.parse_args()


def _infer_hidden_sizes(icnn_sd: Dict[str, torch.Tensor], fallback: List[int]) -> List[int]:
    """Recover ICNN hidden sizes from the state dict when args are missing."""
    if fallback:
        return [int(h) for h in fallback]
    widths: List[int] = []
    idx = 0
    while f"z_linears.{idx}.weight" in icnn_sd:
        weight = icnn_sd[f"z_linears.{idx}.weight"]
        widths.append(int(weight.shape[0]))
        idx += 1
    if not widths:
        raise RuntimeError("Could not infer ICNN hidden sizes from checkpoint.")
    return widths


def _infer_conv_flag(icnn_sd: Dict[str, torch.Tensor], args_dict: Dict) -> Tuple[bool, int]:
    """Determine whether the ICNN used conv layers and the kernel size."""
    if "icnn_conv" in args_dict:
        return bool(args_dict.get("icnn_conv")), int(args_dict.get("icnn_kernel_size", 3))
    first_weight = icnn_sd.get("z_linears.0.weight")
    if first_weight is None:
        return False, 3
    use_conv = first_weight.ndim == 4
    if use_conv:
        kernel_size = int(first_weight.shape[-1])
    else:
        kernel_size = int(args_dict.get("icnn_kernel_size", 3))
    return use_conv, kernel_size


def load_icnn_from_checkpoint(
    icnn_path: str,
    example_input: torch.Tensor,
    device: torch.device,
) -> Tuple[InputConvexPotential, Dict]:
    """Recreate InputConvexPotential with training-time hyperparameters and load weights."""
    checkpoint = torch.load(icnn_path, map_location=device)
    args_dict = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    if isinstance(checkpoint, dict) and "icnn" in checkpoint and looks_like_state_dict(checkpoint["icnn"]):
        icnn_sd = unwrap_state_dict(checkpoint["icnn"])
    elif looks_like_state_dict(checkpoint):
        icnn_sd = unwrap_state_dict(checkpoint)
    else:
        raise RuntimeError(f"ICNN state dict not found in checkpoint: {icnn_path}")

    hidden_sizes = _infer_hidden_sizes(icnn_sd, args_dict.get("icnn_hidden", []))
    activation = str(args_dict.get("icnn_activation", "relu"))
    strong_convexity = float(args_dict.get("icnn_strong_convexity", 1.0))
    nonneg_init = str(args_dict.get("icnn_init", "principled"))
    use_conv, kernel_size = _infer_conv_flag(icnn_sd, args_dict)

    latent_shape = tuple(example_input.shape[1:])
    icnn = InputConvexPotential(
        input_dim=example_input[0].numel(),
        hidden_sizes=hidden_sizes,
        activation=activation,
        strong_convexity=strong_convexity,
        nonneg_init=nonneg_init,
        input_shape=latent_shape,
        use_convs=use_conv,
        conv_kernel_size=kernel_size,
    ).to(device)
    icnn.load_state_dict(icnn_sd, strict=True)
    icnn.eval()
    return icnn, args_dict


def _checkpoint_looks_like_html(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            prefix = f.read(64).strip().lower()
        return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")
    except OSError:
        return False


def _infer_vit_model_name(path: Path) -> str:
    name = path.stem.lower()
    if "vitb16" in name:
        return "vit_base_patch16_224"
    if "vitb32" in name:
        return "vit_base_patch32_224"
    if "vit14" in name or "vitl14" in name:
        return "vit_large_patch14_224"
    return "vit_base_patch16_224"


def load_pretrained_vit(
    checkpoint_path: str,
    device: torch.device,
    num_classes: int = 10,
):
    """
    Load a ViT checkpoint (timm-style) for CIFAR10 classification.
    Tries to infer the correct architecture from the filename (vitb16/vitb32/vit14).
    """
    ckpt_path = Path(checkpoint_path)
    if _checkpoint_looks_like_html(ckpt_path):
        raise RuntimeError(
            f"Checkpoint at {ckpt_path} looks like an HTML page (download likely failed)."
        )

    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "timm is required to load ViT checkpoints. Install with `pip install timm`."
        ) from exc

    arch = _infer_vit_model_name(ckpt_path)
    model = None
    init_errors: List[str] = []
    for extra_kwargs in ({}, {"img_size": 32}):
        try:
            model = timm.create_model(
                arch,
                pretrained=False,
                num_classes=num_classes,
                **extra_kwargs,
            )
            break
        except Exception as exc:  # pragma: no cover - defensive
            init_errors.append(f"{arch} init failed with {extra_kwargs}: {exc}")
            model = None
    if model is None:
        raise RuntimeError(
            f"Could not instantiate ViT model for {ckpt_path}.\n" + "\n".join(init_errors)
        )

    raw = torch.load(ckpt_path, map_location=device)
    state = unwrap_state_dict(raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[ViT] Missing keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"[ViT] Unexpected keys ({len(unexpected)}): {unexpected}")
    print(f"[ViT] Loaded {arch} weights from: {ckpt_path}")
    return model.to(device)


def load_classifier(checkpoint_path: str, device: torch.device):
    """
    Try to load either a utils-style ResNet or a ViT checkpoint.
    Preference is given to ViT when the filename contains 'vit'.
    """
    path = Path(checkpoint_path)
    prefer_vit = "vit" in path.stem.lower()
    errors: List[str] = []

    if not prefer_vit:
        try:
            return load_pretrained_resnet18(
                pretrained_path=str(path),
                num_classes=10,
                strict=False,
                device=device,
            ).to(device)
        except Exception as exc:
            errors.append(f"ResNet loader failed: {exc}")

    try:
        return load_pretrained_vit(str(path), device=device, num_classes=10)
    except Exception as exc:
        errors.append(f"ViT loader failed: {exc}")

    if prefer_vit:
        try:
            return load_pretrained_resnet18(
                pretrained_path=str(path),
                num_classes=10,
                strict=False,
                device=device,
            ).to(device)
        except Exception as exc:
            errors.append(f"Secondary ResNet attempt failed: {exc}")

    error_msg = "\n".join(errors)
    raise RuntimeError(f"Unable to load checkpoint {checkpoint_path}:\n{error_msg}")


def build_models(
    source_path: str,
    target_paths: List[str],
    device: torch.device,
) -> Dict[str, torch.nn.Module]:
    models: Dict[str, torch.nn.Module] = {}
    src_name = Path(source_path).stem or "source"
    models[src_name] = load_classifier(source_path, device)
    models[src_name].eval()

    for path in target_paths:
        name = Path(path).stem or f"target_{len(models)}"
        if name in models:
            name = f"{name}_t{len(models)}"
        model = load_classifier(path, device)
        model.eval()
        models[name] = model

    return models


def clamp_normalized(x: torch.Tensor) -> torch.Tensor:
    """Clamp normalized CIFAR-10 inputs so pixel space stays within [0, 1]."""
    mean = torch.as_tensor(CIFAR10_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(CIFAR10_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return torch.clamp(x, min=lower, max=upper)


def run_attack(args: argparse.Namespace) -> None:
    device = get_device()
    set_deterministic(args.seed)

    loader, _ = get_cifar10_loader(
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        augment_train=True,
        shuffle_train=False,
        pin_memory=True,
        download=True,
        seed=args.seed,
    )
    example_batch = next(iter(loader))[0][:1].to(device)

    icnn, icnn_args = load_icnn_from_checkpoint(args.icnn_checkpoint, example_batch, device)
    models = build_models(args.source_model, args.target_models, device)
    source_name = next(iter(models.keys()))

    clean_correct = {name: 0 for name in models}
    adv_correct = {name: 0 for name in models}
    adv_overlap_counts = {name: 0 for name in models if name != source_name}
    adv_source_misclassified = 0
    total_seen = 0
    l2_sum = 0.0
    linf_sum = 0.0

    saved_clean: List[torch.Tensor] = []
    saved_adv: List[torch.Tensor] = []
    saved_labels: List[torch.Tensor] = []

    for batch_x, batch_y in loader:
        if args.max_samples is not None and total_seen >= args.max_samples:
            break
        if args.max_samples is not None:
            remaining = args.max_samples - total_seen
            if remaining < batch_x.size(0):
                batch_x = batch_x[:remaining]
                batch_y = batch_y[:remaining]

        x = batch_x.to(device)
        y = batch_y.to(device)

        with torch.no_grad():
            for name, model in models.items():
                logits = model(x)
                preds = logits.argmax(dim=1)
                clean_correct[name] += int((preds == y).sum().item())

        # Adversarial pushforward T(x) = ∇φ(x); clamp to stay in normalized pixel bounds.
        z_adv, _ = adversarial_pushforward(icnn, x, detach_for_model=True)
        z_adv = clamp_normalized(z_adv).detach()
        

        with torch.no_grad():
            adv_pix = to_pixel(z_adv)
            clean_pix = to_pixel(x)
            delta_pix = adv_pix - clean_pix
            flat = delta_pix.view(delta_pix.size(0), -1)
            l2_sum += flat.norm(p=2, dim=1).sum().item()
            linf_sum += flat.abs().max(dim=1).values.sum().item()

            batch_miscls = {}
            for name, model in models.items():
                logits = model(z_adv)
                preds = logits.argmax(dim=1)
                correct = preds == y
                adv_correct[name] += int(correct.sum().item())
                batch_miscls[name] = ~correct

            source_miscls_mask = batch_miscls[source_name]
            source_miscls_count = int(source_miscls_mask.sum().item())
            adv_source_misclassified += source_miscls_count
            if source_miscls_count > 0:
                for name, mask in batch_miscls.items():
                    if name == source_name:
                        continue
                    overlap = int((mask & source_miscls_mask).sum().item())
                    adv_overlap_counts[name] += overlap

        saved_clean.append(x.cpu())
        saved_adv.append(z_adv.cpu())
        saved_labels.append(batch_y)

        total_seen += x.size(0)

    if total_seen == 0:
        raise RuntimeError("No samples were processed. Check dataloader settings.")

    clean_acc = {k: v / total_seen for k, v in clean_correct.items()}
    adv_acc = {k: v / total_seen for k, v in adv_correct.items()}
    mean_l2 = l2_sum / total_seen
    mean_linf = linf_sum / total_seen

    run_id = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_path:
        save_path = Path(args.save_path)
    else:
        icnn_name = Path(args.icnn_checkpoint).stem
        src_name = Path(args.source_model).stem
        save_path = out_dir / f"{icnn_name}__{src_name}__{run_id}.pt"

    metrics = {
        "run_id": run_id,
        "split": args.split,
        "samples": total_seen,
        "icnn_checkpoint": str(Path(args.icnn_checkpoint).resolve()),
        "source_model": str(Path(args.source_model).resolve()),
        "target_models": [str(Path(p).resolve()) for p in args.target_models],
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "mean_l2_pix": mean_l2,
        "mean_linf_pix": mean_linf,
        "icnn_args": icnn_args,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "adv_source_misclassified": adv_source_misclassified,
        "adv_common_misclassified": adv_overlap_counts,
        "adv_common_misclassified_rate_given_source": {
            name: (overlap / adv_source_misclassified) if adv_source_misclassified > 0 else 0.0
            for name, overlap in adv_overlap_counts.items()
        },
        "adv_common_misclassified_rate_overall": {
            name: overlap / total_seen for name, overlap in adv_overlap_counts.items()
        },
    }

    if not args.no_save:
        payload = {
            "clean": torch.cat(saved_clean, dim=0),
            "adv": torch.cat(saved_adv, dim=0),
            "labels": torch.cat(saved_labels, dim=0),
            "metrics": metrics,
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, save_path)
        metrics_path = save_path.with_suffix(".json")
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved adversarial set to {save_path}")
        print(f"Saved metrics to {metrics_path}")

    print("\n=== Transfer Results ===")
    print(f"Samples processed: {total_seen}")
    print(f"Mean ||delta||_2 (pixel): {mean_l2:.4f} | Mean ||delta||_inf (pixel): {mean_linf:.4f}")
    for name in models:
        print(
            f"{name}: clean acc {clean_acc[name]*100:.2f}% -> adv acc {adv_acc[name]*100:.2f}% "
            f"(drop { (clean_acc[name]-adv_acc[name])*100:.2f}%)"
        )
    print(f"\nAdversarial samples misclassified by source ({source_name}): {adv_source_misclassified}")
    for name, overlap in adv_overlap_counts.items():
        cond_rate = (overlap / adv_source_misclassified) if adv_source_misclassified > 0 else 0.0
        overall_rate = overlap / total_seen
        print(
            f"{name}: overlap with source misclassifications {overlap} "
            f"({cond_rate*100:.2f}% of source errors, {overall_rate*100:.2f}% of all samples)"
        )


if __name__ == "__main__":
    run_attack(parse_args())
