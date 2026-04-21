"""CIFAR-10 ResNet-50 feature extraction with caching (512-dim)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import torchvision
    import torchvision.transforms as transforms
except Exception as _e:  # pragma: no cover
    torchvision = None  # type: ignore
    transforms = None  # type: ignore
    _TORCHVISION_IMPORT_ERROR: Exception = _e
else:
    _TORCHVISION_IMPORT_ERROR = None  # type: ignore


def extract_features(
    data_loader: DataLoader, model: nn.Module, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.to(device)
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="extracting features"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu())
            labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)


def get_cifar10_dataloaders(
    data_dir: str, batch_size: int = 128, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """CIFAR-10 dataloaders with ImageNet preprocessing (paper setup).

    CIFAR-10 images are resized to 224×224 and normalized with ImageNet mean/std.
    """
    if torchvision is None or transforms is None:
        raise RuntimeError(
            "torchvision is required for CIFAR-10 loading / transforms, but it failed to import."
        ) from _TORCHVISION_IMPORT_ERROR

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, test_loader


def build_resnet50_feature_extractor(output_dim: int = 512) -> nn.Module:
    """ResNet-50 with a randomly initialised 512-dim + ReLU head (Section 6.3)."""
    if torchvision is None:
        raise RuntimeError(
            "torchvision is required to construct the ResNet-50 feature extractor."
        ) from _TORCHVISION_IMPORT_ERROR
    try:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        resnet = torchvision.models.resnet50(weights=weights)
    except Exception:
        resnet = torchvision.models.resnet50(pretrained=True)
    resnet.fc = nn.Sequential(
        nn.Linear(resnet.fc.in_features, output_dim),
        nn.ReLU(inplace=True),
    )
    return resnet


def load_or_extract_cifar10_resnet50_features(
    feature_file: Path,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    force_reextract: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load cached CIFAR-10 ResNet-50 features or extract and cache them."""
    if feature_file.exists() and not force_reextract:
        print(f"Loading cached features from: {feature_file}")
        data = torch.load(feature_file, map_location="cpu")
        return (
            data["train_features"],
            data["train_labels"],
            data["test_features"],
            data["test_labels"],
        )

    print("Cached feature file not found (or forced re-extract). Extracting features...")
    if torchvision is None or transforms is None:
        raise RuntimeError(
            "torchvision is required to extract CIFAR-10 ResNet-50 features."
        ) from _TORCHVISION_IMPORT_ERROR
    feature_file.parent.mkdir(parents=True, exist_ok=True)

    resnet_feature_extractor = build_resnet50_feature_extractor(output_dim=512)
    cifar_train_loader, cifar_test_loader = get_cifar10_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )
    train_features, train_labels = extract_features(
        cifar_train_loader, resnet_feature_extractor, device
    )
    test_features, test_labels = extract_features(
        cifar_test_loader, resnet_feature_extractor, device
    )

    torch.save(
        {
            "train_features": train_features,
            "train_labels": train_labels,
            "test_features": test_features,
            "test_labels": test_labels,
        },
        feature_file,
    )
    print(f"Saved cached features to: {feature_file}")

    return train_features, train_labels, test_features, test_labels
