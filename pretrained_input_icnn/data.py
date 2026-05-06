"""CIFAR-10 dataloaders consistent with the original training pipeline."""
from __future__ import annotations

import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .utils.transforms import CIFAR10_MEAN, CIFAR10_STD


def _dataloader_seed(base_seed: int, offset: int = 0) -> Tuple[torch.Generator, Callable[[int], None]]:
    seed = int((base_seed + offset) % (2 ** 32))
    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init_fn(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return generator, worker_init_fn


def _cifar10_transform(train: bool, augment_train: bool):
    import torchvision.transforms as T
    from PIL import Image

    def pil_to_float_tensor(pic: Image.Image) -> torch.Tensor:
        if not isinstance(pic, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(pic)}")
        buf = torch.tensor(bytearray(pic.tobytes()), dtype=torch.uint8)
        nchannel = len(pic.getbands())
        img = buf.view(pic.height, pic.width, nchannel)
        img = img.permute(2, 0, 1).contiguous()
        return img.float().div(255.0)

    transforms = []
    if train and augment_train:
        transforms.extend([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])
    transforms.append(T.Lambda(pil_to_float_tensor))
    transforms.append(T.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    return T.Compose(transforms)


def get_cifar10_loaders(
    batch_size: int = 256,
    num_workers: int = 2,
    data_root: str = "./data",
    augment_train: bool = True,
    pin_memory: bool = True,
    download: bool = True,
    seed: Optional[int] = None,
    world_size: int = 1,
    rank: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    """Returns ``(train_loader, test_loader, train_sampler)``.

    When ``world_size > 1``, the train loader uses a
    :class:`DistributedSampler` so each rank sees a non-overlapping
    shard. The sampler is also returned so the caller can call
    ``set_epoch(epoch)`` between epochs to vary the shuffle. The test
    loader is NOT sharded — eval is rank-0-only in this pipeline, since
    PGD restarts are awkward to shard correctly.
    """
    import torchvision

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=download,
        transform=_cifar10_transform(True, augment_train),
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=download,
        transform=_cifar10_transform(False, augment_train),
    )

    train_gen = train_init = test_gen = test_init = None
    if seed is not None:
        train_gen, train_init = _dataloader_seed(seed, 0)
        test_gen, test_init = _dataloader_seed(seed, 1)

    train_sampler: Optional[DistributedSampler] = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(seed) if seed is not None else 0,
            drop_last=True,  # equal local batch sizes simplify gradient averaging
        )
        # Per-rank batch size = batch_size / world_size so the global
        # effective batch matches the single-GPU configuration.
        local_batch = max(1, batch_size // world_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=local_batch,
            shuffle=False,  # sampler handles shuffling
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=train_init,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=train_gen,
            worker_init_fn=train_init,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=test_gen,
        worker_init_fn=test_init,
    )
    return train_loader, test_loader, train_sampler
