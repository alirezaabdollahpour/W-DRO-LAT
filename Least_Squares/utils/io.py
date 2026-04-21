"""JSON persistence for per-method delta-sweep results."""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Dict, List

import numpy as np

from config import ULSConfig


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def save_results_json(
    delta_values: np.ndarray,
    results: Dict[str, List[float]],
    save_dir: str,
    cfg: ULSConfig | None = None,
) -> None:
    """Save each method's (delta, test_loss) curve as its own JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    for method_name, test_losses in results.items():
        safe = _safe_name(method_name)
        payload = {
            "method": method_name,
            "delta_values": [float(d) for d in delta_values.tolist()],
            "test_losses": [float(v) for v in test_losses],
        }
        if cfg is not None:
            payload["hyperparameters"] = asdict(cfg)
        path = os.path.join(save_dir, f"{safe}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved: {path}")
