"""DDP helpers: init/cleanup, gradient + scalar reducers, rank checks.

The package detects DDP by reading the env vars ``RANK`` / ``LOCAL_RANK``
/ ``WORLD_SIZE`` set by ``torchrun``. In single-GPU mode all helpers
degrade to no-ops, so the same algorithm code works in both regimes.

Design notes:

* The classifier is wrapped in :class:`torch.nn.parallel.DistributedDataParallel`.
  Its gradient sync runs automatically on the outer ``loss.backward()`` in
  ``classifier_update``.

* The NPF ω-parameters and NN-DRO MLP-parameters are NOT wrapped in DDP —
  their inner loops bypass ``loss.backward()`` (autograd.grad / a separate
  Adam optimiser). Each rank computes a local gradient; we manually
  all-reduce with ``op=mean`` so all ranks step identically and ω stays
  in sync.

* Inner-loop forward passes use :attr:`BaseAdvTrainer.classifier_module`
  (the unwrapped module). DDP's reducer must NOT fire during the inner
  loop — many forwards without a matching backward would corrupt its
  state. The DDP wrapper only fires on the outer classifier_update.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    """Snapshot of the current distributed setup. world_size==1 means single-GPU."""
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = ""

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


_DIST: DistInfo = DistInfo()


def init_distributed() -> DistInfo:
    """Initialize torch.distributed if torchrun env vars are set.

    Returns the cached :class:`DistInfo`. Subsequent calls return the same
    object (process-group init is idempotent). When ``WORLD_SIZE`` is unset
    or equals 1, this is a no-op and the returned info reflects single-GPU.
    """
    global _DIST
    if _DIST.is_distributed:
        return _DIST

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        _DIST = DistInfo()
        return _DIST

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = os.environ.get("PRETRAINED_INPUT_ICNN_DDP_BACKEND", "nccl")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    _DIST = DistInfo(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend=backend,
    )
    if _DIST.is_main:
        print(
            f"[ddp] init_process_group ok — world_size={world_size}, "
            f"backend={backend}, local_rank={local_rank}"
        )
    return _DIST


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def info() -> DistInfo:
    return _DIST


def is_main() -> bool:
    return _DIST.is_main


def barrier() -> None:
    if _DIST.is_distributed and dist.is_initialized():
        dist.barrier()


def all_reduce_grad_list(grads: List[torch.Tensor]) -> List[torch.Tensor]:
    """Average a list of gradient tensors across ranks (in-place safe).

    Used by NPF (BB+Armijo) and NN-DRO (Adam ascent) to make every rank's
    ω-gradient match before stepping. When not distributed, returns the
    list unchanged.
    """
    if not _DIST.is_distributed:
        return grads
    out: List[torch.Tensor] = []
    for g in grads:
        gg = g.contiguous()
        dist.all_reduce(gg, op=dist.ReduceOp.SUM)
        gg.div_(_DIST.world_size)
        out.append(gg)
    return out


def all_reduce_scalar(x: torch.Tensor) -> torch.Tensor:
    """Mean-reduce a 0-dim tensor across ranks. Returns a fresh tensor."""
    if not _DIST.is_distributed:
        return x.detach().clone()
    t = x.detach().clone().contiguous()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t.div_(_DIST.world_size)
    return t


def all_reduce_sum_scalar(x: torch.Tensor) -> torch.Tensor:
    """Sum-reduce a 0-dim tensor across ranks. Returns a fresh tensor."""
    if not _DIST.is_distributed:
        return x.detach().clone()
    t = x.detach().clone().contiguous()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_gather_concat(x: torch.Tensor) -> torch.Tensor:
    """All-gather equal-shaped per-rank tensors and concat along dim 0.

    Used by MPA (new_ppa) to build the GLOBAL reassignment candidate pool:
    every rank contributes its full-batch-shard particles/labels/losses so
    each sample best-responds over all B global candidates regardless of
    world size. Requires the local tensor shape to match across ranks —
    guaranteed per training batch because DistributedSampler hands every
    rank equally sized shards. Returns ``x`` unchanged when not distributed.
    """
    if not _DIST.is_distributed:
        return x
    x = x.detach().contiguous()
    parts = [torch.empty_like(x) for _ in range(_DIST.world_size)]
    dist.all_gather(parts, x)
    # Keep this rank's slot bitwise-identical to its local tensor (all_gather
    # round-trips through NCCL buffers; identity for the local slot keeps the
    # self-inclusion property of the reassignment exact).
    parts[_DIST.rank] = x
    return torch.cat(parts, dim=0)


def all_reduce_grads_(params) -> None:
    """In-place mean-reduce ``param.grad`` across ranks for an iterable of Parameters."""
    if not _DIST.is_distributed:
        return
    for p in params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
        p.grad.data.div_(_DIST.world_size)
