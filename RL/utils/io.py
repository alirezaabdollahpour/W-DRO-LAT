"""JSON/checkpoint I/O helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def choose_nonexisting_path(path: Path) -> Path:
    if not path.exists():
        return path
    k = 1
    while True:
        cand = path.with_name(f"{path.stem}_{k}{path.suffix}")
        if not cand.exists():
            return cand
        k += 1


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
