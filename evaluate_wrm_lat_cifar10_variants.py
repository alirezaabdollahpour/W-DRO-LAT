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

Notes
-----
* We evaluate using the **backbone forward** directly (no phi/head split needed for clean eval).
* If you trained with CIFAR10 normalization in the dataloaders (as in your script), we do the same here.
* The checkpoint loader matches the format you used in your saving block (arch/arch_init/state_dict/... on CPU).

Usage
-----
python evaluate_wrm_lat_cifar10_variants.py \
  --ckpt /path/to/your_saved_wrm_lat.ckpt \
  --batch-size 256 --num-workers 4

"""

import os
import json
import argparse
from datetime import datetime
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T

# ---- import your utils-style models (same as in your WRM-LAT script) ----
from model import ResNet18 as ResNet18Plain  # alias kept consistent with your code
from model import PreActResNet18

# -----------------------------
# Globals & utilities
# -----------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_deterministic(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


# -----------------------------
# Checkpoint loader (matches your WRM-LAT save format)
# -----------------------------

def load_backbone_from_ckpt(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")

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

    def _dl(url_map):
        for fname, url in url_map.items():
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                print(f"Downloading {fname} ...")
                urlretrieve(url, fpath)
                print(f"  ✓ {fname}")
            else:
                print(f"  ✓ {fname} already exists")

    print("[download] CIFAR-10.1 & CIFAR-10.2")
    _dl(c10_1)
    _dl(c10_2)

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


def _normalize_in_place(t: torch.Tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD) -> torch.Tensor:
    # t: (N, C, H, W) in [0,1]
    m = torch.tensor(mean, dtype=t.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=t.dtype).view(1, 3, 1, 1)
    return (t - m) / s


def load_cifar10_1_as_dataset(data_path: str, labels_path: str) -> TensorDataset:
    data = np.load(data_path)
    labels = np.load(labels_path)
    x = torch.from_numpy(_to_nchw_float01(data)).float()
    y = torch.from_numpy(np.array(labels, dtype=np.int64)).long()
    x = _normalize_in_place(x)
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
    x = _normalize_in_place(x)
    return TensorDataset(x, y)


# -----------------------------
# Standard CIFAR-10 test loader (with the same normalization)
# -----------------------------

def get_cifar10_test_loader(batch_size=256, num_workers=2):
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True), len(testset)


# -----------------------------
# Evaluation
# -----------------------------

@torch.no_grad()
def eval_clean(base_model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    base_model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = base_model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)


# -----------------------------
# Main
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate WRM-LAT checkpoint on CIFAR-10 / 10.1 / 10.2")
    p.add_argument("--ckpt", type=str, required=True, help="Path to saved WRM-LAT checkpoint (.ckpt/.pth)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-json", type=str, default="cifar10_variants_eval.json")
    return p.parse_args()


def main():
    args = parse_args()
    set_deterministic(args.seed)
    device = get_device()
    print("Using device:", device)

    # 1) Load backbone from your WRM-LAT checkpoint
    base, meta = load_backbone_from_ckpt(args.ckpt, device)
    print(f"[ckpt] arch={meta['arch']}  cut_layer={meta['cut_layer']}  epoch={meta['epoch']}  date={meta['date']}")

    # 2) CIFAR-10 test loader
    c10_loader, c10_len = get_cifar10_test_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"✓ CIFAR-10 test set: {c10_len} samples")

    # 3) Download + set up CIFAR-10.1 & 10.2
    paths = download_cifar10_variants()

    c101_ds = load_cifar10_1_as_dataset(paths["c10_1_data"], paths["c10_1_labels"]) if os.path.exists(paths["c10_1_data"]) else None
    c102_ds = load_cifar10_2_test_as_dataset(paths["c10_2_test"]) if os.path.exists(paths["c10_2_test"]) else None

    c101_loader = DataLoader(c101_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if c101_ds is not None else None
    c102_loader = DataLoader(c102_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if c102_ds is not None else None

    # 4) Evaluate
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "ckpt": os.path.abspath(args.ckpt),
        "arch": meta["arch"],
        "arch_init": meta["arch_init"],
        "cut_layer": meta["cut_layer"],
        "scores": {},
    }

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

    # 5) Save JSON
    with open(args.save_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {args.save_json}")


if __name__ == "__main__":
    main()
