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
import torch.nn as nn

from pretrained_INPUT_icnn import InputConvexPotential, adversarial_pushforward
from model import get_model
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
    Load a ViT checkpoint for CIFAR10 classification.
    Supports vit-pytorch SimpleViT checkpoints (e.g., ViT_checkpoints/ViT1.pth) and timm models.
    """
    ckpt_path = Path(checkpoint_path)
    if _checkpoint_looks_like_html(ckpt_path):
        raise RuntimeError(
            f"Checkpoint at {ckpt_path} looks like an HTML page (download likely failed)."
        )

    raw = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(raw)
    if state is None:
        raise RuntimeError(f"Could not find a state dict inside {ckpt_path}")

    if _looks_like_simple_vit_state_dict(state):
        model = _build_simple_vit_model(state, num_classes)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[ViT-simple] Missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"[ViT-simple] Unexpected keys ({len(unexpected)}): {unexpected}")
        width = state.get("transformer.layers.0.0.norm.weight", torch.tensor([])).shape[0]
        print(f"[ViT-simple] Loaded vit_exp-compatible weights (width={width}) from: {ckpt_path}")
        return model.to(device)

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

    # When the checkpoint name suggests a ViT, do not silently fall back to ResNet;
    # surface the ViT loading error instead so the correct model is evaluated.
    if prefer_vit:
        try:
            return load_pretrained_vit(str(path), device=device, num_classes=10)
        except Exception as exc:
            errors.append(f"ViT loader failed: {exc}")
            error_msg = "\n".join(errors)
            raise RuntimeError(f"Unable to load checkpoint {checkpoint_path} as ViT:\n{error_msg}")

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


def _looks_like_simple_vit_state_dict(state: Dict[str, torch.Tensor]) -> bool:
    """Heuristic: vit-pytorch SimpleViT checkpoints use these tensor names."""
    return all(
        key in state
        for key in (
            "to_patch_embedding.1.weight",
            "transformer.layers.0.0.norm.weight",
            "linear_head.1.weight",
        )
    )


def _build_simple_vit_model(
    state: Dict[str, torch.Tensor],
    num_classes: int,
) -> nn.Module:
    """Instantiate a SimpleViT variant that matches the checkpoint structure."""
    try:
        from vit_pytorch.simple_vit import (
            Attention,
            FeedForward,
            Rearrange,
            SimpleViT,
            Transformer,
            pair,
            posemb_sincos_2d,
        )
    except ImportError as exc:
        raise RuntimeError(
            "vit_pytorch is required to load vit_exp checkpoints. Install with `pip install vit-pytorch`."
        ) from exc

    class TransformerCompat(nn.Module):
        """Transformer without final LayerNorm (matches saved checkpoints)."""

        def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    nn.ModuleList([Attention(dim, heads=heads, dim_head=dim_head), FeedForward(dim, mlp_dim)])
                    for _ in range(depth)
                ]
            )

        def forward(self, x):
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return x

    class SimpleViTCompat(nn.Module):
        """Older SimpleViT variant without LayerNorms in patch embedding (matches saved checkpoints)."""

        def __init__(
            self,
            *,
            image_size,
            patch_size,
            num_classes,
            dim,
            depth,
            heads,
            mlp_dim,
            channels: int = 3,
            dim_head: int = 64,
        ):
            super().__init__()
            image_height, image_width = pair(image_size)
            patch_height, patch_width = pair(patch_size)
            patch_dim = channels * patch_height * patch_width

            self.to_patch_embedding = nn.Sequential(
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_height,
                    p2=patch_width,
                ),
                nn.Linear(patch_dim, dim),
            )
            self.pos_embedding = posemb_sincos_2d(
                h=image_height // patch_height,
                w=image_width // patch_width,
                dim=dim,
            )
            self.transformer = TransformerCompat(dim, depth, heads, dim_head, mlp_dim)
            self.pool = "mean"
            self.to_latent = nn.Identity()
            self.linear_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        def forward(self, img):
            x = self.to_patch_embedding(img)
            x = x + self.pos_embedding.to(device=img.device, dtype=x.dtype)
            x = self.transformer(x)
            x = x.mean(dim=1)
            x = self.to_latent(x)
            return self.linear_head(x)

    def _infer_depth() -> int:
        layers = [
            int(k.split(".")[2])
            for k in state
            if k.startswith("transformer.layers.") and k.split(".")[2].isdigit()
        ]
        return (max(layers) + 1) if layers else 6

    def _infer_heads_dim() -> Tuple[int, int]:
        qkv = state.get("transformer.layers.0.0.to_qkv.weight")
        to_out = state.get("transformer.layers.0.0.to_out.weight")
        if qkv is None or to_out is None:
            return 16, 64
        qkv_dim = int(qkv.shape[0])
        out_dim = int(to_out.shape[1])
        heads_dim_prod = out_dim if qkv_dim == 3 * out_dim else qkv_dim // 3
        if heads_dim_prod % 64 == 0:
            heads = heads_dim_prod // 64
            dim_head = 64
        else:
            heads = max(1, heads_dim_prod // 64)
            dim_head = heads_dim_prod // max(1, heads)
        return heads, dim_head

    linear_w = state.get("to_patch_embedding.2.weight") or state.get("to_patch_embedding.1.weight")
    if linear_w is None:
        raise RuntimeError("Could not find patch embedding weights in ViT checkpoint.")

    dim = int(state.get("transformer.layers.0.0.norm.weight", linear_w).shape[0])
    patch_dim = int(linear_w.shape[1] if linear_w.ndim == 2 else 48)
    patch_size = int((patch_dim // 3) ** 0.5) if patch_dim % 3 == 0 else 4
    depth = _infer_depth()
    heads, dim_head = _infer_heads_dim()
    mlp_dim = dim * 2

    use_compat = "to_patch_embedding.2.weight" not in state and linear_w.ndim == 2
    vit_cls = SimpleViTCompat if use_compat else SimpleViT
    model = vit_cls(
        image_size=32,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dim_head=dim_head,
    )
    return model


def _extract_state_dict(raw: object) -> Dict[str, torch.Tensor] | None:
    """Try common locations for a state dict inside a checkpoint."""
    try:
        sd = unwrap_state_dict(raw)
    except Exception:
        return None
    return sd if looks_like_state_dict(sd) else None


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
    total_seen_raw = 0
    total_used = 0
    l2_sum = 0.0
    linf_sum = 0.0

    saved_clean: List[torch.Tensor] = []
    saved_adv: List[torch.Tensor] = []
    saved_labels: List[torch.Tensor] = []

    for batch_x, batch_y in loader:
        if args.max_samples is not None and total_seen_raw >= args.max_samples:
            break
        if args.max_samples is not None:
            remaining = args.max_samples - total_seen_raw
            if remaining < batch_x.size(0):
                batch_x = batch_x[:remaining]
                batch_y = batch_y[:remaining]

        x = batch_x.to(device)
        y = batch_y.to(device)
        total_seen_raw += x.size(0)

        with torch.no_grad():
            clean_preds = {}
            for name, model in models.items():
                logits = model(x)
                preds = logits.argmax(dim=1)
                clean_preds[name] = preds

        source_clean_mask = clean_preds[source_name] == y
        source_clean_count = int(source_clean_mask.sum().item())
        if source_clean_count == 0:
            continue

        x = x[source_clean_mask]
        y = y[source_clean_mask]
        for name, preds in clean_preds.items():
            masked_preds = preds[source_clean_mask]
            clean_correct[name] += int((masked_preds == y).sum().item())

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
        saved_labels.append(y.cpu())

        total_used += x.size(0)

    if total_used == 0:
        raise RuntimeError(
            "No samples qualified for analysis (all were misclassified by the clean source model)."
        )

    clean_acc = {k: v / total_used for k, v in clean_correct.items()}
    adv_acc = {k: v / total_used for k, v in adv_correct.items()}
    mean_l2 = l2_sum / total_used
    mean_linf = linf_sum / total_used

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
        "samples_raw": total_seen_raw,
        "samples_analyzed": total_used,
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
            name: overlap / total_used for name, overlap in adv_overlap_counts.items()
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
    print(f"Samples processed (raw): {total_seen_raw} | analyzed after clean-source filter: {total_used}")
    print(f"Mean ||delta||_2 (pixel): {mean_l2:.4f} | Mean ||delta||_inf (pixel): {mean_linf:.4f}")
    for name in models:
        print(
            f"{name}: clean acc {clean_acc[name]*100:.2f}% -> adv acc {adv_acc[name]*100:.2f}% "
            f"(drop { (clean_acc[name]-adv_acc[name])*100:.2f}%)"
        )
    print(f"\nAdversarial samples misclassified by source ({source_name}): {adv_source_misclassified}")
    for name, overlap in adv_overlap_counts.items():
        cond_rate = (overlap / adv_source_misclassified) if adv_source_misclassified > 0 else 0.0
        overall_rate = overlap / total_used
        print(
            f"{name}: overlap with source misclassifications {overlap} "
            f"({cond_rate*100:.2f}% of source errors, {overall_rate*100:.2f}% of all samples)"
        )


if __name__ == "__main__":
    run_attack(parse_args())
