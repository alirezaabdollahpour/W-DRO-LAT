"""Basic utilities shared across the RL pipeline (logging, seeding, generators)."""
from __future__ import annotations

import argparse
import random
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm.auto import trange, tqdm
except Exception:
    trange = None
    tqdm = None


class IntListAction(argparse.Action):
    """argparse helper that parses a list of ints from flexible input formats."""

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, [])
            return

        raw_tokens = [values] if isinstance(values, str) else list(values)

        out: list[int] = []
        for raw in raw_tokens:
            s = str(raw).strip()
            if not s:
                continue
            s = s.strip("()[]{}")
            parts = [p.strip() for p in s.split(",")] if "," in s else s.split()
            for p in parts:
                if not p:
                    continue
                try:
                    out.append(int(p))
                except ValueError as e:
                    raise argparse.ArgumentTypeError(
                        f"Expected integer(s) for {option_string or self.dest}; got '{raw}'."
                    ) from e

        if not out:
            raise argparse.ArgumentTypeError(
                f"Expected at least one integer for {option_string or self.dest}."
            )
        setattr(namespace, self.dest, out)


@contextmanager
def freeze_params(*modules: Optional[nn.Module]):
    """Temporarily disable gradients for module parameters (alternating updates)."""
    prev: list[tuple[torch.nn.Parameter, bool]] = []
    for module in modules:
        if module is None:
            continue
        for p in module.parameters():
            prev.append((p, p.requires_grad))
            p.requires_grad_(False)
    try:
        yield
    finally:
        for p, req in prev:
            p.requires_grad_(req)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def log_line(msg: str) -> None:
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)
