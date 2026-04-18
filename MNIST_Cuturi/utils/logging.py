"""CSV logger used across all algorithms."""
from __future__ import annotations

import csv
import os


DEFAULT_LOG_PATH = os.path.join("MNIST", "training_logs.csv")
LOG_FIELDNAMES = [
    "algorithm",
    "phase",
    "epoch",
    "step",
    "loss_adv",
    "adv_loss",
    "cls_loss",
    "acc_clean",
    "acc_adv",
    "w2_proxy",
    "inner_grad_norm",
    "delta_gap",
    "projection_gain",
    "active_support_frac",
]


class CSVLogger:
    def __init__(self, path: str, fieldnames):
        self.path = path
        self.fieldnames = list(fieldnames)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, **kwargs):
        row = {name: kwargs.get(name) for name in self.fieldnames}
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
