#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a saved WRM-LAT backbone checkpoint on CIFAR-10, CIFAR-10.1 (v6) and CIFAR-10.2.

What this script does
---------------------
1) Downloads CIFAR-10.1 (v6) and CIFAR-10.2 from the official GitHub repos.
2) Reconstructs your utils-style backbone (PreActResNet18 or ResNet18Plain)
   from a **future-proof checkpoint** you saved (with keys: arch, arch_init, state_dict, cut_layer, args, ...).
3) Applies the **same CIFAR-10 normalization** used in your WRM-LAT code and evaluates clean accuracy.
4) Optionally evaluates **input-space PGD** (pixel units) on CIFAR-10/10.1/10.2.
5) Optionally evaluates **AutoAttack** robustness (CIFAR-10) using the robustbench reference settings.

Notes
-----
* We evaluate using the **backbone forward** directly (no phi/head split needed for clean, PGD, or AutoAttack eval).
* Datasets are normalized with CIFAR10 mean/std. For PGD, we perturb in pixel space [0,1]
  and then re-normalize before feeding into the model (matching your training-eval style).
* The checkpoint loader matches the format you used in your saving block (arch/arch_init/state_dict/... on CPU).

Usage
-----
python evaluate_wrm_lat_cifar10_variants.py \
  --ckpt /path/to/your_saved_wrm_lat.ckpt \
  --batch-size 256 --num-workers 4 \
  --inp-eps 0.031372549 --inp-steps 20 --inp-restarts 5 \
  --autoattack --autoattack-bs 128 \
  --autoattack-version custom --autoattack-attacks apgd-ce apgd-dlr
"""

import os
import tarfile
import json
import argparse
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from typing import List, Optional, Tuple
from model import ResNet18 as ResNet18Plain  # alias kept consistent with your code
from model import PreActResNet18
from pretrained_LAT import build_split_resnet18

from utils import (
    auto_pgd_step_size,
    evaluate_under_input_pgd,
    dataloader_seed,
    get_cifar10_loader,
    get_device,
    parameterized_filename,
    set_deterministic,
    to_pixel,
    to_normalized,
    unwrap_state_dict,
)

DEFAULT_CIFAR10C_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]


class PhiHeadWrapper(nn.Module):
    def __init__(self, phi: nn.Module, head: nn.Module):
        super().__init__()
        self.phi = phi
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.phi(x))


class PixelModelWrapper(nn.Module):
    """Adapter so AutoAttack operates in pixel space while the backbone expects normalized inputs."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x_pix: torch.Tensor) -> torch.Tensor:
        return self.base(to_normalized(x_pix))

# -----------------------------
# Globals & utilities
# -----------------------------
def _infer_arch_from_state_dict(sd: dict) -> str:
    """
    Simple heuristic:
      - if top-level has 'bn1' (root BN at stem), assume ResNet18Plain
      - if top-level has 'bn' (tail BN as in PreAct), assume PreActResNet18
    """
    roots = {k.split(".", 1)[0] for k in sd.keys()}
    if "bn1" in roots:
        return "ResNet18Plain"
    if "bn" in roots:
        return "PreActResNet18"
    # default to PreActResNet18
    return "PreActResNet18"


# -----------------------------
# Checkpoint loader (matches your WRM-LAT save format)
# -----------------------------
def load_backbone_from_ckpt(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")

    # Handle phi/head checkpoints saved by pretrained_LAT.py
    if "phi" in ckpt and "head" in ckpt:
        args_ckpt = ckpt.get("args", {})
        num_classes = int(args_ckpt.get("num_classes", 10))
        cut_layer = args_ckpt.get("cut_layer", "layer4")
        phi, head = build_split_resnet18(num_classes=num_classes, cut_layer=cut_layer)

        phi_keys = phi.load_state_dict(ckpt["phi"], strict=False)
        head_keys = head.load_state_dict(ckpt["head"], strict=False)

        if phi_keys.missing_keys or phi_keys.unexpected_keys:
            print(f"[phi] load => missing={phi_keys.missing_keys}, unexpected={phi_keys.unexpected_keys}")
        if head_keys.missing_keys or head_keys.unexpected_keys:
            print(f"[head] load => missing={head_keys.missing_keys}, unexpected={head_keys.unexpected_keys}")

        model = PhiHeadWrapper(phi, head)
        model.to(device).eval()

        meta = {
            "arch": args_ckpt.get("arch", "PhiHeadWrapper"),
            "arch_init": {
                "num_classes": num_classes,
                "cut_layer": cut_layer,
            },
            "cut_layer": cut_layer,
            "args": args_ckpt,
            "epoch": ckpt.get("epoch", None),
            "date": ckpt.get("date", None),
            "format": "phi_head",
        }
        return model, meta

    arch = ckpt.get("arch", "PreActResNet18")
    arch_init = ckpt.get(
        "arch_init",
        {"n_cls": 10, "model_width": 64, "normalize_features": False, "normalize_logits": False},
    )

    # Recreate the backbone
    if arch == "PreActResNet18":
        base = PreActResNet18(**arch_init)
    elif arch == "ResNet18Plain":
        base = ResNet18Plain(**arch_init)
    else:
        # best-effort fallback
        print(f"[load] Unknown arch '{arch}', falling back to PreActResNet18 with arch_init={arch_init}")
        base = PreActResNet18(**arch_init)

    missing, unexpected = base.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[load] non-strict load => missing={missing}, unexpected={unexpected}")

    base.to(device).eval()

    meta = {
        "cut_layer": ckpt.get("cut_layer", "layer4"),
        "args": ckpt.get("args", {}),
        "epoch": ckpt.get("epoch", None),
        "date": ckpt.get("date", None),
        "pytorch": ckpt.get("pytorch", None),
        "arch": arch,
        "arch_init": arch_init,
    }
    return base, meta


# -----------------------------
# Download & load CIFAR-10.1 / CIFAR-10.2
# -----------------------------
def download_cifar10_variants(data_dir: str = "./cifar10_variants"):
    os.makedirs(data_dir, exist_ok=True)

    # CIFAR-10.1 (v6)
    c10_1 = {
        "cifar10.1_v6_data.npy": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy",
        "cifar10.1_v6_labels.npy": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy",
    }

    # CIFAR-10.2 (test set)
    c10_2 = {
        "cifar102_test.npy": "https://github.com/modestyachts/cifar-10.2/raw/61b0e3ac09809a2351379fb54331668cc9c975c4/cifar102_test.npy",
    }

    def _dl(url_map, desc):
        for fname, url in tqdm(url_map.items(), desc=desc, leave=False):
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                print(f"Downloading {fname} ...")
                urlretrieve(url, fpath)
                print(f"  ✓ {fname}")
            else:
                print(f"  ✓ {fname} already exists")

    print("[download] CIFAR-10.1 & CIFAR-10.2")
    _dl(c10_1, "CIFAR-10.1 files")
    _dl(c10_2, "CIFAR-10.2 files")

    return {
        "c10_1_data": os.path.join(data_dir, "cifar10.1_v6_data.npy"),
        "c10_1_labels": os.path.join(data_dir, "cifar10.1_v6_labels.npy"),
        "c10_2_test": os.path.join(data_dir, "cifar102_test.npy"),
    }


def _to_nchw_float01(arr: np.ndarray) -> np.ndarray:
    """Ensure array is (N, C, H, W) in [0,1]."""
    a = np.array(arr)
    if a.max() > 1.0:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32)
    if a.ndim == 4 and a.shape[-1] == 3:  # NHWC -> NCHW
        a = np.transpose(a, (0, 3, 1, 2))
    return a


def load_cifar10_1_as_dataset(data_path: str, labels_path: str) -> TensorDataset:
    data = np.load(data_path)
    labels = np.load(labels_path)
    x = torch.from_numpy(_to_nchw_float01(data)).float()
    y = torch.from_numpy(np.array(labels, dtype=np.int64)).long()
    x = to_normalized(x)
    return TensorDataset(x, y)


def load_cifar10_2_test_as_dataset(file_path: str) -> TensorDataset:
    """Robust loader for CIFAR-10.2 test file."""
    obj = np.load(file_path, allow_pickle=True)

    images = None
    labels = None

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        d = obj.item()
        # try common keys
        if "images" in d:
            images = d["images"]
        elif "data" in d:
            images = d["data"]
        elif "x" in d:
            images = d["x"]
        if "labels" in d:
            labels = d["labels"]
        elif "y" in d:
            labels = d["y"]
        elif "targets" in d:
            labels = d["targets"]
    elif isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] >= 3072:
        # flattened CIFAR-like
        images = obj[:, :3072].reshape(-1, 32, 32, 3)
        labels = obj[:, -1]
    else:
        raise RuntimeError(f"Unsupported cifar10.2 format: type={type(obj)}, shape={getattr(obj,'shape',None)}")

    x = torch.from_numpy(_to_nchw_float01(images)).float()
    y = torch.from_numpy(np.array(labels, dtype=np.int64)).long()
    x = to_normalized(x)
    return TensorDataset(x, y)

# === Download CIFAR-10-C dataset ===
def download_cifar10c():
    """Download CIFAR-10-C dataset."""
    data_dir = "./cifar10c_data"
    os.makedirs(data_dir, exist_ok=True)

    # CIFAR-10-C URL (from the official source)
    cifar10c_url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    cifar10c_path = os.path.join(data_dir, "CIFAR-10-C.tar")

    print("Downloading CIFAR-10-C dataset...")

    # Check if already downloaded and extracted
    extracted_dir = os.path.join(data_dir, "CIFAR-10-C")
    if os.path.exists(extracted_dir) and len(os.listdir(extracted_dir)) > 0:
        print(f"✓ CIFAR-10-C already exists at {extracted_dir}")
        return extracted_dir

    # Download if not exists
    if not os.path.exists(cifar10c_path):
        print(f"Downloading CIFAR-10-C from {cifar10c_url}...")
        try:
            urlretrieve(cifar10c_url, cifar10c_path)
            print(f"✓ Downloaded CIFAR-10-C.tar")
        except Exception as e:
            print(f"✗ Failed to download CIFAR-10-C: {e}")
            return None
    else:
        print(f"✓ CIFAR-10-C.tar already exists")

    # Extract the tar file
    print("Extracting CIFAR-10-C.tar...")
    try:
        with tarfile.open(cifar10c_path, 'r') as tar:
            tar.extractall(data_dir)
        print(f"✓ Extracted to {extracted_dir}")

        # Clean up tar file to save space
        os.remove(cifar10c_path)
        print("✓ Cleaned up tar file")

        return extracted_dir
    except Exception as e:
        print(f"✗ Failed to extract CIFAR-10-C: {e}")
        return None

def load_cifar10c(corruption, severity, n_examples=10000, data_dir=None, download=True):
    """
    Load CIFAR-10-C data for a specific corruption and severity.

    Args:
        corruption (str): Name of the corruption (e.g., 'gaussian_noise')
        severity (int): Severity level (1-5)
        n_examples (int): Number of examples to load (default: 10000, full test set)
        data_dir (str): Directory containing CIFAR-10-C data
        download (bool): Whether to download if not found

    Returns:
        tuple: (images, labels) as numpy arrays
    """
    if data_dir is None:
        if download:
            data_dir = download_cifar10c()
            if data_dir is None:
                raise ValueError("Failed to download CIFAR-10-C")
        else:
            raise ValueError("data_dir must be provided if download=False")

    # Path to corruption file
    corruption_file = os.path.join(data_dir, f"{corruption}.npy")
    labels_file = os.path.join(data_dir, "labels.npy")

    if not os.path.exists(corruption_file):
        raise FileNotFoundError(f"Corruption file not found: {corruption_file}")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    # Load data
    print(f"Loading {corruption} (severity {severity})...")

    # Load corruption data - shape is (50000, 32, 32, 3)
    # Each severity has 10k images: severity 1 = [0:10k], severity 2 = [10k:20k], etc.
    corruption_data = np.load(corruption_file)
    labels_data = np.load(labels_file)

    # Calculate start and end indices for the specific severity
    start_idx = (severity - 1) * 10000
    end_idx = start_idx + min(n_examples, 10000)

    # Extract the relevant slice
    images = corruption_data[start_idx:end_idx]
    labels = labels_data[start_idx:end_idx]

    print(f"✓ Loaded {len(images)} images for {corruption} (severity {severity})")
    print(f"  Images shape: {images.shape}, dtype: {images.dtype}")
    print(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"  Image range: [{images.min()}, {images.max()}]")

    return images, labels


def cifar10c_numpy_to_dataset(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
    """Convert CIFAR-10-C arrays to a normalized TensorDataset."""
    x = torch.from_numpy(_to_nchw_float01(images)).float()
    y = torch.from_numpy(np.array(labels, dtype=np.int64)).long()
    x = to_normalized(x)
    return TensorDataset(x, y)


@torch.no_grad()
def collect_pixels_and_labels(
    loader: DataLoader,
    *,
    max_examples: Optional[int] = None,
    desc: str = "Collect AutoAttack data",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize a dataloader into pixel-space tensors for AutoAttack."""
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    remaining = None if max_examples is None or max_examples < 0 else int(max_examples)
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    progress = tqdm(loader, desc=desc, leave=False, total=total_batches)
    for x_norm, y in progress:
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and x_norm.size(0) > remaining:
            x_norm = x_norm[:remaining]
            y = y[:remaining]
        xs.append(to_pixel(x_norm).cpu())
        ys.append(y.cpu())
        if remaining is not None:
            remaining -= x_norm.size(0)
    progress.close()
    x_pix = torch.cat(xs, dim=0) if xs else torch.empty(0, device="cpu")
    y_lab = torch.cat(ys, dim=0) if ys else torch.empty(0, dtype=torch.long, device="cpu")
    return x_pix.contiguous(), y_lab.contiguous()

# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def eval_clean(base_model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    base_model.eval()
    total, correct = 0, 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    progress = tqdm(loader, desc="Clean Eval", leave=False, total=total_batches)
    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = base_model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else total
    if hasattr(loader, "dataset") and total != dataset_size:
        raise RuntimeError(
            f"eval_clean consumed {total} samples but dataset has {dataset_size}."
            " Ensure DataLoader iterates the full set."
        )
    progress.close()
    return 100.0 * correct / max(1, dataset_size)


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate WRM-LAT checkpoint on CIFAR-10 / 10.1 / 10.2 (+ optional input-PGD)")
    p.add_argument("--ckpt", type=str, required=True, help="Path to saved WRM-LAT checkpoint (.ckpt/.pth)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-json", type=str, default="cifar10_variants_eval.json")

    # Optional: evaluate a clean/natural pretrained backbone instead of WRM-LAT ckpt
    p.add_argument("--natural-model", action="store_true",
                   help="If set, load a natural/clean pretrained backbone from a .pt/.pth file instead of the WRM ckpt")
    p.add_argument("--natural-model-path", type=str, default=".",
                   help="Path to the natural model .pt/.pth file (e.g., /mnt/.../R2.pt)")

    # CIFAR-10-C evaluation
    p.add_argument("--skip-cifar10c", action="store_true", help="Skip CIFAR-10-C evaluation.")
    p.add_argument("--cifar10c-data-dir", type=str, default="",
                   help="Directory containing CIFAR-10-C .npy files (default: auto download to ./cifar10c_data).")
    p.add_argument("--cifar10c-corruptions", type=str, nargs="+", default=None,
                   help="Which CIFAR-10-C corruptions to evaluate (default: all).")
    p.add_argument("--cifar10c-severities", type=int, nargs="+", default=[5],
                   help="Severities (1-5) to evaluate for each corruption (default: 5).")
    p.add_argument("--cifar10c-max-examples", type=int, default=10000,
                   help="Max examples per corruption/severity (<=10000 from CIFAR-10-C).")

    # AutoAttack evaluation
    p.add_argument("--autoattack", action="store_true", help="Enable AutoAttack evaluation.")
    p.add_argument("--autoattack-only", action="store_true",
                   help="Run only AutoAttack (skip clean/PGD/CIFAR-10 variants).")
    p.add_argument("--autoattack-norm", type=str, default="Linf", choices=["Linf", "L2", "L1"],
                   help="Norm ball for AutoAttack.")
    p.add_argument("--autoattack-eps", type=float, default=8/255,
                   help="Epsilon radius for AutoAttack (pixel units).")
    p.add_argument("--autoattack-bs", type=int, default=128,
                   help="Batch size used inside AutoAttack (bs argument).")
    p.add_argument("--autoattack-version", type=str, default="standard",
                   help="AutoAttack version (e.g., 'standard', 'plus', 'rand').")
    p.add_argument("--autoattack-attacks", type=str, nargs="+", default=None,
                   help="Specific attacks_to_run for AutoAttack (requires --autoattack-version custom).")
    p.add_argument("--autoattack-max-examples", type=int, default=-1,
                   help="Limit AutoAttack to the first N samples (<=0 means all).")
    p.add_argument("--autoattack-seed", type=int, default=None,
                   help="Optional seed for AutoAttack's internal RNG.")
    p.add_argument("--autoattack-iters", type=int, default=None,
                   help="Override total iterations for APGD-based AutoAttack components.")
    p.add_argument("--autoattack-restarts", type=int, default=None,
                   help="Override number of random restarts for APGD-based AutoAttack components.")

    # Input-space PGD config (pixel units)
    p.add_argument("--inp-p", type=str, default="inf", choices=["2", "inf"],
                   help="Norm for input-space PGD evaluation")
    p.add_argument("--inp-eps", type=float, default=8/255,
                   help="Epsilon (pixel units) for input-space PGD")
    p.add_argument("--inp-steps", type=int, default=20,
                   help="Steps for input-space PGD (0 disables PGD evaluation)")
    p.add_argument("--inp-step-size", type=float, default=0.0,
                   help="Step size for input-space PGD (pixel units). If <=0, auto = 2*eps/steps")
    p.add_argument("--inp-restarts", type=int, default=5,
                   help="Random restarts for input-space PGD")
    return p.parse_args()


def main():
    args = parse_args()
    set_deterministic(args.seed)
    device = get_device()
    print("Using device:", device)

    if args.autoattack_only:
        args.autoattack = True
    autoattack_requested = bool(args.autoattack)
    perform_standard_eval = not args.autoattack_only

    if autoattack_requested and args.autoattack_bs <= 0:
        raise ValueError("--autoattack-bs must be positive.")
    if autoattack_requested and args.autoattack_eps <= 0:
        raise ValueError("--autoattack-eps must be positive.")
    autoattack_max_examples = None
    if autoattack_requested and args.autoattack_max_examples > 0:
        autoattack_max_examples = int(args.autoattack_max_examples)
    autoattack_attacks = None
    if autoattack_requested:
        if args.autoattack_attacks is not None:
            autoattack_attacks = list(dict.fromkeys(args.autoattack_attacks))
        if args.autoattack_version.lower() == "custom":
            if autoattack_attacks is None or len(autoattack_attacks) == 0:
                autoattack_attacks = ["apgd-ce", "apgd-dlr"]
        elif args.autoattack_attacks is not None:
            print("[autoattack] Ignoring --autoattack-attacks because version is not 'custom'.")
            autoattack_attacks = None

    # 1) Load backbone from your WRM-LAT checkpoint
    base, meta = load_backbone_from_ckpt(args.ckpt, device)
    print(f"[ckpt] arch={meta['arch']}  cut_layer={meta['cut_layer']}  epoch={meta['epoch']}  date={meta['date']}")

    # ---- Robust NATURAL (clean) model block ----
    # If requested, REPLACE 'base' with a clean pretrained backbone loaded from a raw .pt/.pth
    if args.natural_model:
        nat_path = args.natural_model_path
        if not os.path.isfile(nat_path):
            raise FileNotFoundError(f"--natural-model was set, but file not found: {nat_path}")

        print(f"[natural] Loading clean pretrained backbone from: {nat_path}")
        ckpt_nat = torch.load(nat_path, map_location="cpu")
        sd_nat = unwrap_state_dict(ckpt_nat)
        nat_arch = _infer_arch_from_state_dict(sd_nat)
        nat_init = dict(n_cls=10, model_width=64, normalize_features=False, normalize_logits=False)

        if nat_arch == "PreActResNet18":
            base_nat = PreActResNet18(**nat_init)
        else:
            base_nat = ResNet18Plain(**nat_init)

        missing, unexpected = base_nat.load_state_dict(sd_nat, strict=False)
        if missing or unexpected:
            print(f"[natural] non-strict load => missing={missing}, unexpected={unexpected}")

        base = base_nat  # replace the model to be evaluated
        base.to(device).eval()
        print(f"[natural] Loaded as {nat_arch}. (Normalization remains as in this eval script.)")

    # 2) CIFAR-10 test loader
    c10_loader, c10_len = get_cifar10_loader(
        "test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=False,
        augment_train=False,
        seed=args.seed,
    )
    print(f"✓ CIFAR-10 test set: {c10_len} samples")

    c101_loader = None
    c102_loader = None
    if perform_standard_eval:
        # 3) Download + set up CIFAR-10.1 & 10.2
        paths = download_cifar10_variants()

        c101_ds = load_cifar10_1_as_dataset(paths["c10_1_data"], paths["c10_1_labels"]) if os.path.exists(paths["c10_1_data"]) else None
        c102_ds = load_cifar10_2_test_as_dataset(paths["c10_2_test"]) if os.path.exists(paths["c10_2_test"]) else None

        if c101_ds is not None:
            gen_c101, worker_c101 = dataloader_seed(args.seed, offset=2)
            c101_loader = DataLoader(
                c101_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=worker_c101,
                generator=gen_c101,
            )
        if c102_ds is not None:
            gen_c102, worker_c102 = dataloader_seed(args.seed, offset=3)
            c102_loader = DataLoader(
                c102_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=worker_c102,
                generator=gen_c102,
            )

    # 4) Evaluate
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "ckpt": os.path.abspath(args.ckpt),
        "arch": meta["arch"],
        "arch_init": meta["arch_init"],
        "cut_layer": meta["cut_layer"],
        "used_natural_model": bool(args.natural_model),
        "natural_model_path": os.path.abspath(args.natural_model_path) if args.natural_model else None,
        "scores": {},
    }

    if perform_standard_eval:
        print("\nEvaluating clean accuracy...")
        acc_c10 = eval_clean(base, c10_loader, device)
        print(f"  CIFAR-10   : {acc_c10:.2f}%")
        results["scores"]["cifar10"] = round(acc_c10, 2)

        if c101_loader is not None:
            acc_c101 = eval_clean(base, c101_loader, device)
            print(f"  CIFAR-10.1 : {acc_c101:.2f}%")
            results["scores"]["cifar10.1_v6"] = round(acc_c101, 2)
        else:
            print("  CIFAR-10.1 : (not available)")

        if c102_loader is not None:
            acc_c102 = eval_clean(base, c102_loader, device)
            print(f"  CIFAR-10.2 : {acc_c102:.2f}%")
            results["scores"]["cifar10.2_test"] = round(acc_c102, 2)
        else:
            print("  CIFAR-10.2 : (not available)")

        cifar10c_eval_summary = None
        if args.skip_cifar10c:
            print("  CIFAR-10-C : skipped (--skip-cifar10c)")
            results["scores"]["cifar10c"] = {"status": "skipped"}
        else:
            print("\nEvaluating CIFAR-10-C (clean only)...")
            data_dir = args.cifar10c_data_dir.strip()
            if data_dir:
                if not os.path.isdir(data_dir):
                    candidate = os.path.join(data_dir, "CIFAR-10-C")
                    if os.path.isdir(candidate):
                        data_dir = candidate
                    else:
                        raise FileNotFoundError(
                            f"CIFAR-10-C directory not found: '{data_dir}'. "
                            "Provide the directory containing the corruption .npy files."
                        )
            else:
                data_dir = download_cifar10c()

            if data_dir is None:
                print("  CIFAR-10-C : unavailable (download/extraction failed).")
                results["scores"]["cifar10c"] = {"status": "unavailable"}
            else:
                cifar10c_corruptions = (
                    list(DEFAULT_CIFAR10C_CORRUPTIONS)
                    if args.cifar10c_corruptions is None
                    else list(dict.fromkeys(args.cifar10c_corruptions))
                )
                cifar10c_severities = list(dict.fromkeys(args.cifar10c_severities))
                for sev in cifar10c_severities:
                    if sev < 1 or sev > 5:
                        raise ValueError(f"CIFAR-10-C severities must be between 1 and 5 (got {sev}).")
                if args.cifar10c_max_examples <= 0:
                    raise ValueError("--cifar10c-max-examples must be positive.")
                if args.cifar10c_max_examples > 10000:
                    print("[cifar10c] --cifar10c-max-examples truncated to 10000 (dataset size per severity).")
                cifar10c_max_examples = min(args.cifar10c_max_examples, 10000)

                print(f"[cifar10c] data dir: {data_dir}")
                corruption_summaries = {}
                all_acc = []
                for corr_idx, corruption in enumerate(cifar10c_corruptions):
                    severity_scores = {}
                    severity_vals = []
                    load_error = None
                    for severity in cifar10c_severities:
                        try:
                            images, labels = load_cifar10c(
                                corruption,
                                severity,
                                n_examples=cifar10c_max_examples,
                                data_dir=data_dir,
                                download=False,
                            )
                        except (FileNotFoundError, ValueError) as err:
                            load_error = str(err)
                            break
                        dataset = cifar10c_numpy_to_dataset(images, labels)
                        gen_c, worker_c = dataloader_seed(args.seed, offset=10 + corr_idx * 8 + severity)
                        loader = DataLoader(
                            dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            worker_init_fn=worker_c,
                            generator=gen_c,
                        )
                        acc = eval_clean(base, loader, device)
                        severity_scores[str(severity)] = round(acc, 2)
                        severity_vals.append(acc)

                    if load_error is not None:
                        print(f"  {corruption:<18} error: {load_error}")
                        corruption_summaries[corruption] = {"error": load_error}
                        continue

                    all_acc.extend(severity_vals)
                    if severity_vals:
                        corr_mean = float(np.mean(severity_vals))
                        severity_scores["mean"] = round(corr_mean, 2)
                    else:
                        corr_mean = None
                        severity_scores["mean"] = None
                    corruption_summaries[corruption] = severity_scores

                    sev_line = ", ".join(
                        f"s{sev}:{severity_scores[str(sev)]:.2f}%"
                        for sev in cifar10c_severities
                    )
                    if corr_mean is not None:
                        sev_line = f"{sev_line} | mean={corr_mean:.2f}%"
                    print(f"  {corruption:<18} {sev_line}")

                overall_mean = float(np.mean(all_acc)) if all_acc else None
                if overall_mean is not None:
                    print(f"  CIFAR-10-C mean : {overall_mean:.2f}%")
                else:
                    print("  CIFAR-10-C mean : (no corruptions evaluated)")
                has_error = any(
                    isinstance(summary, dict) and "error" in summary
                    for summary in corruption_summaries.values()
                )
                status = "ok"
                if overall_mean is None:
                    status = "no-data"
                elif has_error:
                    status = "partial"
                cifar10c_eval_summary = {
                    "corruptions": corruption_summaries,
                    "severities": cifar10c_severities,
                    "max_examples": cifar10c_max_examples,
                    "mean": round(overall_mean, 2) if overall_mean is not None else None,
                    "data_dir": os.path.abspath(data_dir),
                    "status": status,
                }
                results["scores"]["cifar10c"] = cifar10c_eval_summary

        # 4.b) Optional: Input-space PGD evaluation (pixel units)
        if args.inp_steps > 0 and args.inp_eps > 0:
            print("\nEvaluating input-space PGD (pixel units)...")
            p_value = 2 if args.inp_p == "2" else float("inf")
            step_size_used = auto_pgd_step_size(p_value, args.inp_eps, args.inp_steps, args.inp_step_size)
            results["scores"]["pgd_params"] = {
                "p": args.inp_p,
                "eps": args.inp_eps,
                "steps": args.inp_steps,
                "step_size": step_size_used,
                "restarts": args.inp_restarts,
            }

            phi_identity = nn.Identity().to(device)

            def run_input_pgd(loader):
                return evaluate_under_input_pgd(
                    phi_identity,
                    base,
                    loader,
                    device,
                    p=p_value,
                    eps=args.inp_eps,
                    steps=args.inp_steps,
                    step_size=step_size_used,
                    restarts=args.inp_restarts,
                )

            acc_pgd_c10, info_c10 = run_input_pgd(c10_loader)
            acc_pgd_c10_pct = acc_pgd_c10 * 100.0
            print(f"  CIFAR-10   PGD: {acc_pgd_c10_pct:.2f}% (avg L2 {info_c10['avg_l2']:.3f}, Linf {info_c10['avg_linf']:.3f})")
            results["scores"]["cifar10_pgd"] = {
                "acc": round(acc_pgd_c10_pct, 2),
                "avg_l2": round(info_c10["avg_l2"], 4),
                "avg_linf": round(info_c10["avg_linf"], 4),
            }

            if c101_loader is not None:
                acc_pgd_c101, info_c101 = run_input_pgd(c101_loader)
                acc_pgd_c101_pct = acc_pgd_c101 * 100.0
                print(f"  CIFAR-10.1 PGD: {acc_pgd_c101_pct:.2f}% (avg L2 {info_c101['avg_l2']:.3f}, Linf {info_c101['avg_linf']:.3f})")
                results["scores"]["cifar10.1_v6_pgd"] = {
                    "acc": round(acc_pgd_c101_pct, 2),
                    "avg_l2": round(info_c101["avg_l2"], 4),
                    "avg_linf": round(info_c101["avg_linf"], 4),
                }
            else:
                print("  CIFAR-10.1 PGD: (not available)")

            if c102_loader is not None:
                acc_pgd_c102, info_c102 = run_input_pgd(c102_loader)
                acc_pgd_c102_pct = acc_pgd_c102 * 100.0
                print(f"  CIFAR-10.2 PGD: {acc_pgd_c102_pct:.2f}% (avg L2 {info_c102['avg_l2']:.3f}, Linf {info_c102['avg_linf']:.3f})")
                results["scores"]["cifar10.2_test_pgd"] = {
                    "acc": round(acc_pgd_c102_pct, 2),
                    "avg_l2": round(info_c102["avg_l2"], 4),
                    "avg_linf": round(info_c102["avg_linf"], 4),
                }
            else:
                print("  CIFAR-10.2 PGD: (not available)")
        else:
            print("\nInput-space PGD: disabled (use --inp-steps > 0 to enable)")
    else:
        print("\nSkipping clean/PGD evaluations (--autoattack-only).")

    if autoattack_requested:
        print("\nEvaluating AutoAttack robustness...")
        try:
            from autoattack import AutoAttack
        except ImportError as err:
            print(f"  AutoAttack unavailable: {err}")
            results["scores"]["autoattack"] = {"status": "unavailable", "error": str(err)}
        else:
            pixel_model = PixelModelWrapper(base)
            pixel_model.eval()
            x_c10_pix, y_c10 = collect_pixels_and_labels(
                c10_loader,
                max_examples=autoattack_max_examples,
                desc="AutoAttack CIFAR-10",
            )
            num_samples = int(y_c10.numel())
            if num_samples == 0:
                print("  CIFAR-10 AutoAttack : no samples collected.")
                results["scores"]["autoattack"] = {"status": "empty"}
            else:
                x_c10_pix = x_c10_pix.to(torch.float32)
                y_c10_cpu = y_c10.clone()
                y_c10_device = y_c10_cpu.to(device)
                if autoattack_attacks is not None:
                    print(f"  AutoAttack attacks_to_run={autoattack_attacks}")
                aa_kwargs = {
                    "model": pixel_model,
                    "norm": args.autoattack_norm,
                    "eps": args.autoattack_eps,
                    "version": args.autoattack_version,
                }
                if autoattack_attacks is not None:
                    aa_kwargs["attacks_to_run"] = autoattack_attacks
                adversary = AutoAttack(**aa_kwargs)
                if args.autoattack_seed is not None:
                    adversary.seed = args.autoattack_seed
                adversary.device = device
                if args.autoattack_iters is not None:
                    if hasattr(adversary, "apgd"):
                        adversary.apgd.n_iter = args.autoattack_iters
                    if hasattr(adversary, "apgd_targeted"):
                        adversary.apgd_targeted.n_iter = args.autoattack_iters
                if args.autoattack_restarts is not None:
                    if hasattr(adversary, "apgd"):
                        adversary.apgd.n_restarts = args.autoattack_restarts
                    if hasattr(adversary, "apgd_targeted"):
                        adversary.apgd_targeted.n_restarts = args.autoattack_restarts
                x_adv = adversary.run_standard_evaluation(
                    x_c10_pix,
                    y_c10_cpu,
                    bs=args.autoattack_bs,
                )
                with torch.no_grad():
                    logits_adv = pixel_model(x_adv.to(device))
                    preds_adv = logits_adv.argmax(dim=1)
                    acc_adv = (preds_adv == y_c10_device).float().mean().item() * 100.0
                print(f"  CIFAR-10 AutoAttack: {acc_adv:.2f}% robust accuracy ({num_samples} samples)")
                results["scores"]["autoattack"] = {
                    "status": "ok",
                    "dataset": "cifar10",
                    "acc": round(acc_adv, 2),
                    "norm": args.autoattack_norm,
                    "eps": args.autoattack_eps,
                    "bs": args.autoattack_bs,
                    "version": args.autoattack_version,
                    "num_examples": num_samples,
                    "max_examples": autoattack_max_examples,
                    "seed": args.autoattack_seed,
                    "attacks": autoattack_attacks,
                    "iters": args.autoattack_iters,
                    "restarts": args.autoattack_restarts,
                }

    # 5) Save JSON
    save_path = parameterized_filename(
        args.save_json,
        {
            "ckpt": Path(args.ckpt).stem,
            "seed": args.seed,
            "inpP": args.inp_p,
            "eps": args.inp_eps,
            "steps": args.inp_steps,
            "restarts": args.inp_restarts,
            "natural": int(bool(args.natural_model)),
        },
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results["output_file"] = str(save_path)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {save_path}")


if __name__ == "__main__":
    main()
