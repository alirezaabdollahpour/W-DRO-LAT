"""Entry point for input-space adversarial training on CIFAR-10."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

# Cluster containers often ship with a tiny /dev/shm (Docker default 64 MB),
# which the default 'file_descriptor' sharing strategy exhausts via the
# DataLoader workers' SemLock + shared-tensor allocations once a few epochs
# have passed. 'file_system' uses regular files in the working tree
# instead — slightly slower but never hits "[Errno 28] No space left on
# device". Set BEFORE the first DataLoader spawns workers.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except (RuntimeError, AttributeError):
    pass

from . import distributed as dist_helpers
from .algorithms import ALGORITHMS
from .config import build_arg_parser, config_from_args
from .data import get_cifar10_loaders
from .models import load_pretrained_classifier
from .utils import (
    evaluate_clean,
    get_device,
    set_deterministic,
    set_seed_benchmark_mode,
)


_LOG_FIELDS = (
    "run_id",
    "algorithm",
    "epoch",
    "phase",
    "timestamp",
    "train_loss",
    "train_acc",
    "train_mse",
    "inner_loss",
    "epoch_seconds",
    "test_loss",
    "test_acc",
    "adv_loss",
    "adv_acc",
    "adv_penalty",
    "input_pgd_acc",
    "input_pgd_avg_l2",
    "input_pgd_avg_linf",
    "input_pgd_samples",
    "lambda_param",
    "lr_theta",
    "batch_size",
    "seed",
    "pretrained_path",
)


def _append_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_LOG_FIELDS))
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _LOG_FIELDS})


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)

    # Initialise DDP if torchrun set the env vars; safe no-op otherwise.
    dist_info = dist_helpers.init_distributed()
    is_main = dist_info.is_main

    if cfg.benchmark_mode:
        set_seed_benchmark_mode(cfg.seed)
        if is_main:
            print(
                f"[input-icnn] BENCHMARK MODE — cudnn.benchmark=True, "
                f"determinism off, seed={cfg.seed}"
            )
    else:
        set_deterministic(cfg.seed)

    if dist_info.is_distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{dist_info.local_rank}")
    else:
        device = get_device()

    if is_main:
        print(
            f"[input-icnn] device={device}, algorithm={cfg.algorithm}, "
            f"lambda={cfg.lambda_param}, world_size={dist_info.world_size}"
        )

    train_loader, test_loader, train_sampler = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        data_root=cfg.data_dir,
        augment_train=cfg.augment_train,
        seed=cfg.seed,
        world_size=dist_info.world_size,
        rank=dist_info.rank,
    )

    classifier = load_pretrained_classifier(
        pretrained_path=cfg.pretrained_path,
        num_classes=10,
        strict=cfg.pretrained_strict,
        device=device,
    ).to(device)

    if is_main:
        sanity_loss, sanity_acc = evaluate_clean(classifier, test_loader, device)
        print(f"[sanity] clean test  loss={sanity_loss:.4f}  acc={sanity_acc*100:.2f}%")
    dist_helpers.barrier()

    trainer_cls = ALGORITHMS[cfg.algorithm]
    trainer = trainer_cls(
        classifier=classifier,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        config=cfg,
        train_sampler=train_sampler,
    )

    history = trainer.fit()
    trainer.save_final()

    # Only rank 0 writes CSV / summary; other ranks finalise DDP and exit.
    if not is_main:
        dist_helpers.cleanup_distributed()
        return

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []
    for entry in history:
        rows.append(
            {
                "run_id": run_id,
                "algorithm": entry.get("algorithm", cfg.algorithm),
                "epoch": entry.get("epoch"),
                "phase": entry.get("phase", "adv"),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "train_loss": entry.get("train_loss"),
                "train_acc": entry.get("train_acc"),
                "train_mse": entry.get("train_mse"),
                "inner_loss": entry.get("inner_loss"),
                "epoch_seconds": entry.get("epoch_seconds"),
                "test_loss": entry.get("test_loss"),
                "test_acc": entry.get("test_acc"),
                "adv_loss": entry.get("adv_loss"),
                "adv_acc": entry.get("adv_acc"),
                "adv_penalty": entry.get("adv_penalty"),
                "input_pgd_acc": entry.get("input_pgd_acc"),
                "input_pgd_avg_l2": entry.get("input_pgd_avg_l2"),
                "input_pgd_avg_linf": entry.get("input_pgd_avg_linf"),
                "input_pgd_samples": entry.get("input_pgd_samples"),
                "lambda_param": cfg.lambda_param,
                "lr_theta": cfg.lr_theta,
                "batch_size": cfg.batch_size,
                "seed": cfg.seed,
                "pretrained_path": cfg.pretrained_path,
            }
        )
        total_epochs = cfg.epochs_icnn_pretrain + cfg.epochs_adv
        epoch_seconds = entry.get("epoch_seconds")
        msg_parts = [
            f"[{cfg.algorithm}|{entry.get('phase', 'adv')}]",
            f"epoch {entry['epoch']:02d}/{total_epochs}",
            f"train {entry['train_loss']:.4f}/{entry['train_acc']*100:.2f}%",
            f"mse {entry['train_mse']:.4f}",
            f"clean {entry['test_loss']:.4f}/{entry['test_acc']*100:.2f}%",
            f"adv {entry['adv_loss']:.4f}/{entry['adv_acc']*100:.2f}%",
        ]
        if epoch_seconds is not None:
            msg_parts.append(f"t={epoch_seconds:.2f}s")
        if entry.get("input_pgd_acc") is not None:
            msg_parts.append(
                f"pgd {entry['input_pgd_acc']*100:.2f}% (l2 {entry['input_pgd_avg_l2']:.3f})"
            )
        print(" | ".join(msg_parts))

    _append_csv(Path(cfg.log_csv), rows)
    summary = {
        "run_id": run_id,
        "algorithm": cfg.algorithm,
        "best_robust_acc": trainer.best_robust_acc,
        "best_robust_epoch": trainer.best_robust_epoch,
        "epochs_completed": trainer.last_completed_epoch,
        "config": cfg.to_dict(),
    }
    if cfg.save:
        Path(cfg.save).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[input-icnn] done. log={cfg.log_csv}  best_robust={trainer.best_robust_acc}")
    dist_helpers.cleanup_distributed()


if __name__ == "__main__":
    main()
