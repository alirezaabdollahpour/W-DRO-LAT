"""Shared training scaffolding for input-space adversarial training on CIFAR-10.

Each concrete algorithm subclasses :class:`BaseAdvTrainer` and supplies one
method, :meth:`step`, that returns the adversarial inputs for the current
batch (and optionally per-batch metrics). The base class drives the outer
loop: classifier optimisation, evaluation, checkpointing, and CSV logging.

The schedule is a two-phase one:

* ``epochs_icnn_pretrain`` warmup epochs run :meth:`step` only — the
  adversary's inner loop fires (e.g. NPF ω-ascent) but the classifier
  optimizer is *not* stepped, so θ is frozen. Stateless attacks (PGD /
  WRM / WFR / Dual / New_PPA) ignore this phase since they don't carry
  state across batches.
* ``epochs_adv`` standard adversarial epochs do the full minimax update.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from .. import distributed as dist_helpers
from ..utils import (
    cuda_sync,
    evaluate_clean,
    evaluate_under_input_pgd,
    evaluate_under_transport,
    set_requires_grad,
)


@dataclass
class EpochMetrics:
    train_loss: float = 0.0
    train_acc: float = 0.0
    train_mse: float = 0.0
    inner_loss: float = 0.0
    epoch_seconds: float = 0.0
    extras: Dict[str, float] = field(default_factory=dict)


class BaseAdvTrainer:
    """Common adversarial training driver.

    Subclasses must override :meth:`name`, :meth:`build_state`, and
    :meth:`step`. The base loop handles classifier updates, evaluation,
    early stopping by best robust accuracy, and persistent checkpoints.
    """

    name: str = "base"
    # Algorithms with persistent adversary state (e.g. parametric NPF /
    # NN-DRO modules) override this to True so the warmup phase
    # actually has somewhere to "warm up". Stateless attacks leave it
    # False; the base loop then warns and treats warmup as a no-op.
    has_parametric_adversary: bool = False

    def __init__(
        self,
        classifier: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        config,
        train_sampler: Optional[DistributedSampler] = None,
    ) -> None:
        self.config = config
        self.dist = dist_helpers.info()
        self.device = device

        # Wrap classifier in DDP when distributed. The unwrapped module is
        # kept as ``classifier_module`` for inner adversary forwards: those
        # forwards run many times without a matching backward, which would
        # corrupt DDP's reducer state if they went through the wrapper.
        # The DDP wrapper is only used on the outer classifier_update path
        # so its all-reduce fires exactly once per outer step.
        if self.dist.is_distributed:
            self._classifier_module = classifier
            self.classifier = DDP(
                classifier,
                device_ids=[self.dist.local_rank] if device.type == "cuda" else None,
                output_device=self.dist.local_rank if device.type == "cuda" else None,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
        else:
            self._classifier_module = classifier
            self.classifier = classifier

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_sampler = train_sampler

        self.optimizer = self._make_classifier_optimizer()
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        if config.epochs_adv > 0:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, config.epochs_adv)
            )

        self.best_robust_acc: Optional[float] = None
        self.best_robust_epoch: Optional[int] = None
        self.last_completed_epoch: int = 0

    # ------------------------------------------------------------------
    # Distributed helpers (no-op in single-GPU mode)
    # ------------------------------------------------------------------
    @property
    def classifier_module(self) -> nn.Module:
        """Unwrapped classifier — use this for inner-loop forwards.

        DDP's reducer must NOT see forward-without-backward; algorithms
        bypass the wrapper here and only hit the wrapper on the final
        ``classifier_update`` path where the all-reduce fires.
        """
        return self._classifier_module

    @property
    def is_main_rank(self) -> bool:
        return self.dist.is_main

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------
    def build_state(self) -> None:  # pragma: no cover - default no-op
        """Optional hook to construct adversary modules / optimizers."""
        return

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return adversarial inputs ``x_adv`` for a batch.

        Subclasses must update their adversary internally and return a
        DETACHED tensor of the same shape as ``x``.
        """
        raise NotImplementedError

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Map clean inputs to adversarial inputs for evaluation only.

        Default: identity (algorithms that only do PGD attacks during
        training, like Madry, can leave this as the default).
        """
        return x.detach()

    def adversary_state_dicts(self) -> Dict[str, Any]:
        """Optional extra checkpoint payload (adversary parameters)."""
        return {}

    # ------------------------------------------------------------------
    # Optimizer / classifier helpers
    # ------------------------------------------------------------------
    def _make_classifier_optimizer(self) -> optim.Optimizer:
        cfg = self.config
        # Optimise the unwrapped module's params — DDP only forwards
        # gradients to ``module.parameters()`` so this is the same set,
        # but avoids any wrapper-introduced parameter shuffling.
        params = [p for p in self._classifier_module.parameters() if p.requires_grad]
        return optim.SGD(
            params,
            lr=cfg.lr_theta,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )

    def classifier_update(self, x_adv: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        self.classifier.train()
        set_requires_grad(self.classifier, True)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.classifier(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.classifier.parameters() if p.requires_grad],
            max_norm=10.0,
        )
        self.optimizer.step()
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == y).float().mean().item()
        return float(loss.item()), float(acc)

    # ------------------------------------------------------------------
    # Outer loop
    # ------------------------------------------------------------------
    def fit(self) -> List[Dict[str, Any]]:
        cfg = self.config
        history: List[Dict[str, Any]] = []
        warmup_epochs = max(0, int(cfg.epochs_icnn_pretrain))
        adv_epochs = max(0, int(cfg.epochs_adv))
        total_epochs = warmup_epochs + adv_epochs
        if total_epochs <= 0:
            return history

        if warmup_epochs > 0 and not self.has_parametric_adversary:
            print(
                f"[{self.name}] --epochs-icnn-pretrain={warmup_epochs} requested but "
                f"{self.name} has no persistent adversary state; treating warmup as a no-op."
            )

        # ICNN warmup phase: train only the adversary, freeze the classifier.
        for warmup_idx in range(1, warmup_epochs + 1):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(warmup_idx)
            metrics = self._train_one_epoch(
                epoch=warmup_idx,
                total_epochs=total_epochs,
                phase="warmup",
            )
            evaluations = self._evaluate(warmup_idx)
            history.append(
                {
                    "algorithm": self.name,
                    "epoch": warmup_idx,
                    "phase": "warmup",
                    "train_loss": metrics.train_loss,
                    "train_acc": metrics.train_acc,
                    "train_mse": metrics.train_mse,
                    "inner_loss": metrics.inner_loss,
                    "epoch_seconds": metrics.epoch_seconds,
                    **metrics.extras,
                    **evaluations,
                }
            )
            # No checkpoint / scheduler step during warmup — the classifier
            # is frozen, so robust accuracy reflects θ_init only.
            self.last_completed_epoch = warmup_idx

        # Standard adversarial training phase.
        for adv_idx in range(1, adv_epochs + 1):
            epoch = warmup_epochs + adv_idx
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(warmup_epochs + adv_idx)
            metrics = self._train_one_epoch(
                epoch=epoch,
                total_epochs=total_epochs,
                phase="adv",
            )
            evaluations = self._evaluate(epoch)
            history.append(
                {
                    "algorithm": self.name,
                    "epoch": epoch,
                    "phase": "adv",
                    "train_loss": metrics.train_loss,
                    "train_acc": metrics.train_acc,
                    "train_mse": metrics.train_mse,
                    "inner_loss": metrics.inner_loss,
                    "epoch_seconds": metrics.epoch_seconds,
                    **metrics.extras,
                    **evaluations,
                }
            )
            self._maybe_checkpoint_best(epoch, evaluations.get("input_pgd_acc"))
            if self.scheduler is not None:
                self.scheduler.step()
            self.last_completed_epoch = epoch
        return history

    def _train_one_epoch(
        self,
        epoch: int,
        total_epochs: int,
        phase: str = "adv",
    ) -> EpochMetrics:
        cfg = self.config
        is_warmup = phase == "warmup"
        total_loss = 0.0
        total_acc = 0.0
        total_mse = 0.0
        total_inner = 0.0
        total_samples = 0
        desc_phase = "Warmup" if is_warmup else "Adv"
        # CUDA-synced wallclock around the training loop. Async kernel
        # launches mean reading the clock without sync only counts
        # dispatch time, not actual GPU work.
        cuda_sync()
        epoch_start = time.perf_counter()
        progress = tqdm(
            self.train_loader,
            desc=f"[{self.name}|{desc_phase}] Epoch {epoch:02d}/{total_epochs}",
            leave=False,
        )
        for x, y in progress:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            x_adv = self.step(x, y)
            if not torch.is_tensor(x_adv):
                raise TypeError("step() must return a tensor")
            x_adv = x_adv.detach()
            if not torch.isfinite(x_adv).all():
                continue
            if is_warmup:
                # Adversary's inner loop already ran inside step(); skip
                # the classifier update so θ stays frozen this epoch.
                loss = float(getattr(self, "_last_inner_loss", 0.0))
                was_training = self.classifier.training
                self.classifier.eval()
                try:
                    with torch.no_grad():
                        acc = (
                            self.classifier(x_adv).argmax(dim=1) == y
                        ).float().mean().item()
                finally:
                    self.classifier.train(was_training)
            else:
                loss, acc = self.classifier_update(x_adv, y)
            with torch.no_grad():
                mse = (x_adv - x).reshape(x.size(0), -1).pow(2).sum(dim=1).mean().item()
            bs = x.size(0)
            total_loss += loss * bs
            total_acc += acc * bs
            total_mse += float(mse) * bs
            total_samples += bs
            inner = float(getattr(self, "_last_inner_loss", 0.0))
            total_inner += inner * bs
            if total_samples > 0:
                progress.set_postfix(
                    loss=f"{total_loss/total_samples:.4f}",
                    acc=f"{total_acc/total_samples*100:.2f}%",
                    mse=f"{total_mse/total_samples:.4f}",
                )
        progress.close()
        cuda_sync()
        epoch_seconds = time.perf_counter() - epoch_start
        n = max(1, total_samples)
        return EpochMetrics(
            train_loss=total_loss / n,
            train_acc=total_acc / n,
            train_mse=total_mse / n,
            inner_loss=total_inner / n,
            epoch_seconds=epoch_seconds,
        )

    # ------------------------------------------------------------------
    # Evaluation + checkpointing
    # ------------------------------------------------------------------
    def _evaluate(self, epoch: int) -> Dict[str, Any]:
        cfg = self.config
        # Eval is rank-0-only — sharding PGD restarts cleanly across ranks
        # adds complexity for marginal speedup at this scale (~1k samples).
        # Other ranks still hit the dist barrier so timing measurements
        # stay aligned across ranks.
        if not self.is_main_rank:
            dist_helpers.barrier()
            return {}
        clean_loss, clean_acc = evaluate_clean(
            self.classifier_module, self.test_loader, self.device
        )

        adv_loss, adv_acc, adv_pen = evaluate_under_transport(
            self.classifier_module,
            lambda x: self.transport_for_eval(x),
            self.test_loader,
            self.device,
            penalty_lambda=getattr(cfg, "lambda_param", 0.0),
        )

        result: Dict[str, Any] = {
            "test_loss": clean_loss,
            "test_acc": clean_acc,
            "adv_loss": adv_loss,
            "adv_acc": adv_acc,
            "adv_penalty": adv_pen,
        }

        if (
            cfg.eval_input_pgd
            and not getattr(cfg, "skip_pgd_during_train", False)
            and cfg.inp_steps > 0
            and cfg.inp_eps > 0
        ):
            sample_limit = cfg.eval_input_pgd_samples
            max_batches: Optional[int] = None
            if (
                sample_limit is not None
                and sample_limit > 0
                and hasattr(self.test_loader, "dataset")
            ):
                samples_available = len(self.test_loader.dataset)
                sample_limit = min(sample_limit, samples_available)
                batch_size = getattr(self.test_loader, "batch_size", sample_limit) or sample_limit
                max_batches = max(1, math.ceil(sample_limit / batch_size))
            p_input = 2 if cfg.inp_p == "2" else float("inf")
            pgd_acc, pgd_info = evaluate_under_input_pgd(
                self.classifier_module,
                self.test_loader,
                self.device,
                p=p_input,
                eps=cfg.inp_eps,
                steps=cfg.inp_steps,
                step_size=cfg.inp_step_size,
                restarts=cfg.inp_restarts,
                max_batches=max_batches,
            )
            result["input_pgd_acc"] = pgd_acc
            result["input_pgd_avg_l2"] = pgd_info["avg_l2"]
            result["input_pgd_avg_linf"] = pgd_info["avg_linf"]
            result["input_pgd_samples"] = pgd_info["samples"]
        dist_helpers.barrier()
        return result

    def _maybe_checkpoint_best(self, epoch: int, robust_acc: Optional[float]) -> None:
        cfg = self.config
        if not self.is_main_rank:
            return
        if robust_acc is None or not cfg.save_best_robust:
            return
        if self.best_robust_acc is None or robust_acc > self.best_robust_acc:
            self.best_robust_acc = float(robust_acc)
            self.best_robust_epoch = epoch
            path = Path(cfg.save_best_robust)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "algorithm": self.name,
                "classifier": self._classifier_module.state_dict(),
                "epoch": epoch,
                "robust_input_pgd_acc": robust_acc,
                "config": cfg.to_dict() if hasattr(cfg, "to_dict") else None,
                **self.adversary_state_dicts(),
            }
            torch.save(payload, path)

    def save_final(self) -> None:
        cfg = self.config
        if not self.is_main_rank:
            return
        if not cfg.save:
            return
        path = Path(cfg.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": self.name,
            "classifier": self._classifier_module.state_dict(),
            "epoch": self.last_completed_epoch,
            "robust_input_pgd_acc": self.best_robust_acc,
            "config": cfg.to_dict() if hasattr(cfg, "to_dict") else None,
            **self.adversary_state_dicts(),
        }
        torch.save(payload, path)
