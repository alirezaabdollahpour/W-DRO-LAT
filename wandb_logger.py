from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional


def _make_serializable(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _make_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_serializable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: _make_serializable(v) for k, v in vars(value).items()}
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


@dataclass
class WandBLogger:
    enabled: bool
    project: str = ""
    run_name: Optional[str] = None
    entity: Optional[str] = None
    mode: str = "online"
    api_key: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.run = None
        self.wandb = None
        if not self.enabled:
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            warnings.warn("wandb requested but not installed; disabling logging.")
            self.enabled = False
            return

        if self.api_key:
            os.environ.setdefault("WANDB_API_KEY", str(self.api_key))
        if self.mode:
            os.environ.setdefault("WANDB_MODE", self.mode)

        run_name = self.run_name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
        init_kwargs: Dict[str, Any] = {
            "project": self.project or "ICNN",
            "name": run_name,
            "config": _make_serializable(self.config or {}),
        }
        if self.entity:
            init_kwargs["entity"] = self.entity
        settings = {}
        if self.mode == "offline":
            settings["mode"] = "offline"
        if settings:
            init_kwargs["settings"] = settings

        self.wandb = wandb
        self.run = wandb.init(**init_kwargs)

    def log(self, payload: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        if not self.enabled or self.run is None:
            return
        serializable = {k: _make_serializable(v) for k, v in payload.items()}
        self.wandb.log(serializable, step=step, commit=commit)

    def update_config(self, payload: Dict[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return
        serializable = {k: _make_serializable(v) for k, v in payload.items()}
        self.run.config.update(serializable)  # type: ignore[attr-defined]

    def log_summary(self, payload: Dict[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return
        serializable = {k: _make_serializable(v) for k, v in payload.items()}
        self.run.summary.update(serializable)  # type: ignore[attr-defined]

    def watch(self, modules: Iterable[Any]) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            for module in modules:
                if module is not None:
                    self.wandb.watch(module, log="all", log_freq=100)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"wandb watch failed: {exc}")

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.finish()
        self.run = None


__all__ = ["WandBLogger"]
