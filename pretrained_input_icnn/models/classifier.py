"""Load a pretrained CIFAR-10 classifier (ResNet18 or PreActResNet18).

We delegate to the existing :mod:`pretrained_LAT.load_pretrained_resnet18`
loader (which already understands the checkpoint formats produced by the
EPFL utils-style backbones) but wrap it so callers in this package don't
have to know about that legacy module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make the repository root importable so we can reuse the existing loaders
# without copying the entire ``model.py`` / ``pretrained_LAT.py`` files.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pretrained_LAT import load_pretrained_resnet18  # type: ignore  # noqa: E402


def load_pretrained_classifier(
    pretrained_path: str,
    num_classes: int = 10,
    strict: bool = False,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Return a ``nn.Module`` mapping normalized CIFAR inputs -> class logits."""
    base = load_pretrained_resnet18(
        pretrained_path=pretrained_path,
        num_classes=num_classes,
        strict=strict,
        device=device,
    )
    return base.to(device)
