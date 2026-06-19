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
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from .. import distributed as dist_helpers
from ..utils import (
    adversary_loss_per_sample,
    cuda_sync,
    evaluate_clean,
    evaluate_under_input_pgd,
    evaluate_under_transport,
    pixel_l2_squared,
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

        # Wrap classifier in DDP when distributed. When BatchNorm is not frozen,
        # buffers must stay synchronized because the outer update runs the
        # classifier in train mode; otherwise each rank accumulates different
        # running statistics and rank 0's private buffers get checkpointed.
        # SyncBatchNorm makes the per-step BN statistics global, while
        # broadcast_buffers keeps all non-parameter buffers aligned at DDP
        # forward boundaries.
        #
        # The unwrapped module is kept as ``classifier_module`` for inner
        # adversary forwards: those forwards run many times without a matching
        # backward, which would corrupt DDP's reducer state if they went
        # through the wrapper. The DDP wrapper is only used on the outer
        # classifier_update path so its all-reduce fires exactly once per
        # outer step.
        if self.dist.is_distributed:
            if device.type == "cuda":
                classifier = nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
            self._classifier_module = classifier
            self.classifier = DDP(
                classifier,
                device_ids=[self.dist.local_rank] if device.type == "cuda" else None,
                output_device=self.dist.local_rank if device.type == "cuda" else None,
                find_unused_parameters=False,
                broadcast_buffers=True,
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
        self._profile_epoch_accum: Dict[str, float] = {}
        self._profile_epoch_batches: int = 0
        self._profile_current: Optional[Dict[str, float]] = None

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

    def load_adversary_state_dicts(
        self,
        payload: Dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:  # pragma: no cover - default no-op
        """Load optional adversary state from a checkpoint payload."""
        if self.has_parametric_adversary:
            raise KeyError(
                f"{self.name} has a parametric adversary but did not implement "
                "load_adversary_state_dicts()."
            )
        return

    def _global_epoch(self, epoch: int) -> int:
        offset = int(getattr(self.config, "checkpoint_epoch_offset", 0) or 0)
        return offset + int(epoch)

    def _lambda_for_adv_epoch(self, adv_idx: int) -> float:
        cfg = self.config
        schedule = tuple(getattr(cfg, "lambda_schedule", ()) or ())
        if not schedule:
            return float(getattr(cfg, "lambda_param", 0.0))
        stage_epochs = int(getattr(cfg, "lambda_stage_epochs", 0) or 0)
        if stage_epochs <= 0:
            raise ValueError("lambda_stage_epochs must be positive when lambda_schedule is set.")
        stage_idx = min((max(1, int(adv_idx)) - 1) // stage_epochs, len(schedule) - 1)
        return float(schedule[stage_idx])

    def _set_active_lambda(self, value: float) -> None:
        # TrainConfig is frozen because most fields should be immutable, but
        # lambda scheduling is a deliberate runtime control. Existing
        # algorithms read cfg.lambda_param, so update that one field centrally.
        object.__setattr__(self.config, "lambda_param", float(value))

    # ------------------------------------------------------------------
    # Inner-loop profiler helpers
    # ------------------------------------------------------------------
    @property
    def is_inner_profile_active(self) -> bool:
        return self._profile_current is not None

    def _reset_inner_profile_epoch(self) -> None:
        self._profile_epoch_accum = {}
        self._profile_epoch_batches = 0
        self._profile_current = None

    def _begin_inner_profile_batch(self) -> None:
        cfg = self.config
        if not bool(getattr(cfg, "profile_inner", False)):
            self._profile_current = None
            return
        max_batches = int(getattr(cfg, "profile_inner_batches", 0) or 0)
        if max_batches > 0 and self._profile_epoch_batches >= max_batches:
            self._profile_current = None
            return
        self._profile_current = {}

    def _finish_inner_profile_batch(self) -> None:
        if self._profile_current is None:
            return
        self.profile_add("batches", 1.0)
        for key, value in self._profile_current.items():
            self._profile_epoch_accum[key] = (
                self._profile_epoch_accum.get(key, 0.0) + float(value)
            )
        self._profile_epoch_batches += 1
        self._profile_current = None

    def profile_add(self, key: str, value: float = 1.0) -> None:
        if self._profile_current is None:
            return
        self._profile_current[key] = self._profile_current.get(key, 0.0) + float(value)

    @contextmanager
    def profile_time(self, key: str) -> Iterator[None]:
        if self._profile_current is None:
            with nullcontext():
                yield
            return
        cuda_sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            cuda_sync()
            self.profile_add(key, time.perf_counter() - start)

    def _inner_profile_epoch_extras(
        self,
        *,
        train_batches_total: int,
        train_samples_total: int,
    ) -> Dict[str, float]:
        accum = self._profile_epoch_accum
        batches = int(accum.get("batches", 0.0))
        if batches <= 0:
            return {}

        extras: Dict[str, float] = {
            "profile_batches": float(batches),
            "profile_train_batches_total": float(train_batches_total),
            "profile_train_samples_total": float(train_samples_total),
            "profile_batch_coverage": float(batches) / float(max(1, train_batches_total)),
            "profile_bb_steps": float(accum.get("bb_steps", 0.0)),
            "profile_bb_trials": float(accum.get("bb_trials", 0.0)),
            "profile_bb_rejections": float(accum.get("bb_rejections", 0.0)),
            "profile_muon_steps": float(accum.get("muon_steps", 0.0)),
            "profile_muon_matrix_params": float(accum.get("muon_matrix_params", 0.0)),
            "profile_muon_fallback_params": float(accum.get("muon_fallback_params", 0.0)),
        }
        inner_steps = max(
            1.0,
            float(accum.get("bb_steps", 0.0))
            + float(accum.get("muon_steps", 0.0)),
        )
        bb_steps = max(1.0, float(accum.get("bb_steps", 0.0)))
        bb_trials = max(1.0, float(accum.get("bb_trials", 0.0)))

        for key, value in sorted(accum.items()):
            profile_key = f"profile_{key}"
            extras[profile_key] = float(value)
            if key.endswith("_s"):
                extras[f"{profile_key}_per_batch"] = float(value) / float(batches)
                extras[f"{profile_key}_per_step"] = float(value) / inner_steps
                if key in {
                    "bb_trial_objective_s",
                    "bb_trial_write_s",
                    "bb_trial_make_s",
                }:
                    extras[f"{profile_key}_per_trial"] = float(value) / bb_trials

        extras["profile_bb_steps_per_batch"] = float(accum.get("bb_steps", 0.0)) / float(batches)
        extras["profile_bb_trials_per_step"] = float(accum.get("bb_trials", 0.0)) / bb_steps
        extras["profile_muon_steps_per_batch"] = float(accum.get("muon_steps", 0.0)) / float(batches)
        return extras

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

    def _freeze_batchnorm_layers(self) -> None:
        if not bool(getattr(self.config, "freeze_batchnorm", False)):
            return
        for module in self._classifier_module.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def classifier_update(self, x_adv: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        self.classifier.train()
        self._freeze_batchnorm_layers()
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
        if tuple(getattr(cfg, "lambda_schedule", ()) or ()):
            self._set_active_lambda(self._lambda_for_adv_epoch(1))

        # ICNN warmup phase: train only the adversary, freeze the classifier.
        for warmup_idx in range(1, warmup_epochs + 1):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self._global_epoch(warmup_idx))
            metrics = self._train_one_epoch(
                epoch=warmup_idx,
                total_epochs=total_epochs,
                phase="warmup",
            )
            evaluations = self._evaluate(warmup_idx)
            entry = {
                "algorithm": self.name,
                "epoch": warmup_idx,
                "phase": "warmup",
                "train_loss": metrics.train_loss,
                "train_acc": metrics.train_acc,
                "train_mse": metrics.train_mse,
                "inner_loss": metrics.inner_loss,
                "epoch_seconds": metrics.epoch_seconds,
                "global_epoch": self._global_epoch(warmup_idx),
                "lambda_param": float(getattr(cfg, "lambda_param", 0.0)),
                **metrics.extras,
                **evaluations,
            }
            history.append(entry)
            self._print_epoch_summary(entry, total_epochs)
            self._save_epoch_checkpoint(
                warmup_idx,
                phase="warmup",
                robust_acc=evaluations.get("input_pgd_acc"),
            )
            # No checkpoint / scheduler step during warmup — the classifier
            # is frozen, so robust accuracy reflects θ_init only.
            self.last_completed_epoch = warmup_idx

        # Standard adversarial training phase.
        for adv_idx in range(1, adv_epochs + 1):
            epoch = warmup_epochs + adv_idx
            self._set_active_lambda(self._lambda_for_adv_epoch(adv_idx))
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self._global_epoch(epoch))
            metrics = self._train_one_epoch(
                epoch=epoch,
                total_epochs=total_epochs,
                phase="adv",
            )
            evaluations = self._evaluate(epoch)
            entry = {
                "algorithm": self.name,
                "epoch": epoch,
                "phase": "adv",
                "train_loss": metrics.train_loss,
                "train_acc": metrics.train_acc,
                "train_mse": metrics.train_mse,
                "inner_loss": metrics.inner_loss,
                "epoch_seconds": metrics.epoch_seconds,
                "global_epoch": self._global_epoch(epoch),
                "lambda_param": float(getattr(cfg, "lambda_param", 0.0)),
                **metrics.extras,
                **evaluations,
            }
            history.append(entry)
            self._print_epoch_summary(entry, total_epochs)
            self._maybe_checkpoint_best(epoch, evaluations.get("input_pgd_acc"))
            self._save_epoch_checkpoint(
                epoch,
                phase="adv",
                robust_acc=evaluations.get("input_pgd_acc"),
            )
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
        self._reset_inner_profile_epoch()
        total_loss = 0.0
        total_acc = 0.0
        total_mse = 0.0
        total_inner = 0.0
        total_train_adv_loss = 0.0
        total_transport_cost = 0.0
        total_weighted_penalty = 0.0
        total_inner_objective = 0.0
        total_samples = 0
        total_batches = 0
        smoke_max_batches = int(os.environ.get("SMOKE_MAX_TRAIN_BATCHES", "0") or 0)
        desc_phase = "Warmup" if is_warmup else "Adv"
        primary_name = "margin" if bool(getattr(cfg, "use_margin_loss", False)) else "adv_ce"
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
            if smoke_max_batches > 0 and total_batches >= smoke_max_batches:
                break
            total_batches += 1
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self._begin_inner_profile_batch()
            try:
                x_adv = self.step(x, y)
            finally:
                self._finish_inner_profile_batch()
            if not torch.is_tensor(x_adv):
                raise TypeError("step() must return a tensor")
            x_adv = x_adv.detach()
            if not torch.isfinite(x_adv).all():
                continue
            diagnostics = self._batch_adversary_diagnostics(x, x_adv, y)
            if is_warmup:
                # Adversary's inner loop already ran inside step(); skip
                # the classifier update so θ stays frozen this epoch.
                loss = float(getattr(self, "_last_inner_loss", 0.0))
                was_training = self.classifier.training
                self.classifier.eval()
                try:
                    with torch.no_grad():
                        acc = (
                            self.classifier_module(x_adv).argmax(dim=1) == y
                        ).float().mean().item()
                finally:
                    self.classifier.train(was_training)
            else:
                loss, acc = self.classifier_update(x_adv, y)
            with torch.no_grad():
                mse = pixel_l2_squared(x_adv, x).mean().item()
            bs = x.size(0)
            total_loss += loss * bs
            total_acc += acc * bs
            total_mse += float(mse) * bs
            total_train_adv_loss += diagnostics["train_adv_loss"] * bs
            total_transport_cost += diagnostics["train_transport_cost"] * bs
            total_weighted_penalty += diagnostics["train_weighted_penalty"] * bs
            total_inner_objective += diagnostics["train_inner_objective"] * bs
            total_samples += bs
            inner = float(getattr(self, "_last_inner_loss", 0.0))
            total_inner += inner * bs
            if total_samples > 0:
                progress.set_postfix(
                    {
                        "outer_ce": f"{total_loss/total_samples:.4f}",
                        "acc": f"{total_acc/total_samples*100:.2f}%",
                        primary_name: f"{total_train_adv_loss/total_samples:.4f}",
                        "cost": f"{total_transport_cost/total_samples:.4f}",
                        "pen": f"{total_weighted_penalty/total_samples:.4f}",
                        "obj": f"{total_inner_objective/total_samples:.4f}",
                    }
                )
        progress.close()
        cuda_sync()
        epoch_seconds = time.perf_counter() - epoch_start
        n = max(1, total_samples)
        extras = {
            "train_adv_loss": total_train_adv_loss / n,
            "train_transport_cost": total_transport_cost / n,
            "train_weighted_penalty": total_weighted_penalty / n,
            "train_inner_objective": total_inner_objective / n,
        }
        if smoke_max_batches > 0:
            extras["smoke_max_train_batches"] = float(smoke_max_batches)
        extras.update(
            self._inner_profile_epoch_extras(
                train_batches_total=total_batches,
                train_samples_total=total_samples,
            )
        )
        return EpochMetrics(
            train_loss=total_loss / n,
            train_acc=total_acc / n,
            train_mse=total_mse / n,
            inner_loss=total_inner / n,
            epoch_seconds=epoch_seconds,
            extras=extras,
        )

    def _batch_adversary_diagnostics(
        self,
        x: torch.Tensor,
        x_adv: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        cfg = self.config
        lam = float(getattr(cfg, "lambda_param", 0.0))
        was_training = self.classifier_module.training
        self.classifier_module.eval()
        try:
            with torch.no_grad():
                logits = self.classifier_module(x_adv)
                primary = adversary_loss_per_sample(
                    logits, y, use_margin=bool(getattr(cfg, "use_margin_loss", False))
                )
                cost = pixel_l2_squared(x_adv, x)
                adv_loss = float(primary.mean().item())
                transport_cost = float(cost.mean().item())
        finally:
            self.classifier_module.train(was_training)
        weighted_penalty = lam * transport_cost
        return {
            "train_adv_loss": adv_loss,
            "train_transport_cost": transport_cost,
            "train_weighted_penalty": weighted_penalty,
            "train_inner_objective": adv_loss - weighted_penalty,
        }

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

        adv_loss, adv_acc, adv_cost, adv_pen = evaluate_under_transport(
            self.classifier_module,
            lambda x: self.transport_for_eval(x),
            self.test_loader,
            self.device,
            penalty_lambda=getattr(cfg, "lambda_param", 0.0),
        )

        result: Dict[str, Any] = {
            "test_loss": clean_loss,
            "test_acc": clean_acc,
            "clean_loss": clean_loss,
            "clean_acc": clean_acc,
            "adv_loss": adv_loss,
            "adv_acc": adv_acc,
            "adv_penalty": adv_pen,
            "eval_transport_cost": adv_cost,
            "eval_transport_weighted_penalty": adv_pen,
            "transport_adv_loss": adv_loss,
            "transport_adv_acc": adv_acc,
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
            else:
                sample_limit = None
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
                max_samples=sample_limit,
            )
            result["input_pgd_acc"] = pgd_acc
            result["input_pgd_clean_acc"] = pgd_info["clean_acc"]
            result["input_pgd_clean_correct"] = pgd_info["clean_correct"]
            result["input_pgd_robust_correct"] = pgd_info["robust_correct"]
            result["input_pgd_avg_l2"] = pgd_info["avg_l2"]
            result["input_pgd_avg_linf"] = pgd_info["avg_linf"]
            result["input_pgd_max_l2"] = pgd_info["max_l2"]
            result["input_pgd_max_linf"] = pgd_info["max_linf"]
            result["input_pgd_samples"] = pgd_info["samples"]
        dist_helpers.barrier()
        return result

    def _print_epoch_summary(self, entry: Dict[str, Any], total_epochs: int) -> None:
        if not self.is_main_rank:
            return
        epoch_seconds = entry.get("epoch_seconds")
        primary_label = (
            "inner_margin"
            if bool(getattr(self.config, "use_margin_loss", False))
            else "inner_adv_ce"
        )
        parts = [
            f"[{self.name}|{entry.get('phase', 'adv')}]",
            f"epoch {entry['epoch']:02d}/{total_epochs}",
            f"lambda {float(entry.get('lambda_param', getattr(self.config, 'lambda_param', 0.0))):g}",
            f"outer_ce {entry.get('train_loss', 0.0):.4f}/{entry.get('train_acc', 0.0)*100:.2f}%",
            f"{primary_label} {entry.get('train_adv_loss', 0.0):.4f}",
            f"cost {entry.get('train_transport_cost', entry.get('train_mse', 0.0)):.4f}",
            f"pen {entry.get('train_weighted_penalty', 0.0):.4f}",
            f"obj {entry.get('train_inner_objective', entry.get('inner_loss', 0.0)):.4f}",
        ]
        if "test_acc" in entry:
            parts.append(
                f"clean {entry.get('test_loss', 0.0):.4f}/{entry['test_acc']*100:.2f}%"
            )
        if "adv_acc" in entry:
            parts.append(
                f"transport {entry.get('adv_loss', 0.0):.4f}/{entry['adv_acc']*100:.2f}%"
            )
            parts.append(f"eval_pen {entry.get('adv_penalty', 0.0):.4f}")
        if entry.get("input_pgd_acc") is not None:
            parts.append(
                f"pgd_robust {entry['input_pgd_acc']*100:.2f}% "
                f"(avg_l2 {entry.get('input_pgd_avg_l2', 0.0):.3f}, "
                f"max_l2 {entry.get('input_pgd_max_l2', 0.0):.3f}, "
                f"n {entry.get('input_pgd_samples', 0)})"
            )
        else:
            parts.append("pgd_robust n/a")
        if epoch_seconds is not None:
            parts.append(f"t={epoch_seconds:.2f}s")
        if entry.get("profile_batches"):
            if str(getattr(self.config, "npf_inner_optimizer", "")).lower() == "muon":
                parts.append(
                    "profile "
                    f"step={entry.get('profile_step_wall_s_per_batch', 0.0):.3f}s/b "
                    f"inner={entry.get('profile_inner_loop_wall_s_per_batch', 0.0):.3f}s/b "
                    f"T={entry.get('profile_objective_transport_s_per_step', 0.0):.3f}s/step "
                    f"clf={entry.get('profile_objective_classifier_s_per_step', 0.0):.3f}s/step "
                    f"muon={entry.get('profile_muon_update_s_per_step', 0.0):.3f}s/step "
                    f"orth={entry.get('profile_muon_orthogonalize_s_per_step', 0.0):.3f}s/step"
                )
            else:
                parts.append(
                    "profile "
                    f"step={entry.get('profile_step_wall_s_per_batch', 0.0):.3f}s/b "
                    f"inner={entry.get('profile_inner_loop_wall_s_per_batch', 0.0):.3f}s/b "
                    f"T={entry.get('profile_objective_transport_s_per_step', 0.0):.3f}s/step "
                    f"clf={entry.get('profile_objective_classifier_s_per_step', 0.0):.3f}s/step "
                    f"ls={entry.get('profile_bb_linesearch_s_per_step', 0.0):.3f}s/step "
                    f"trials={entry.get('profile_bb_trials_per_step', 0.0):.2f}/step"
                )
        print(" | ".join(parts), flush=True)

    def _maybe_checkpoint_best(self, epoch: int, robust_acc: Optional[float]) -> None:
        cfg = self.config
        if not self.is_main_rank:
            return
        if robust_acc is None:
            return
        if self.best_robust_acc is not None and robust_acc <= self.best_robust_acc:
            return

        self.best_robust_acc = float(robust_acc)
        self.best_robust_epoch = epoch
        if not cfg.save_best_robust:
            return

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

    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        phase: str,
        robust_acc: Optional[float],
    ) -> Dict[str, Any]:
        cfg = self.config
        global_epoch = self._global_epoch(epoch)
        return {
            "algorithm": self.name,
            "classifier": self._classifier_module.state_dict(),
            "epoch": epoch,
            "global_epoch": global_epoch,
            "phase": phase,
            "robust_input_pgd_acc": robust_acc,
            "config": cfg.to_dict() if hasattr(cfg, "to_dict") else None,
            **self.adversary_state_dicts(),
        }

    def _save_epoch_checkpoint(
        self,
        epoch: int,
        *,
        phase: str,
        robust_acc: Optional[float],
    ) -> None:
        cfg = self.config
        if not self.is_main_rank:
            return
        out_dir = str(getattr(cfg, "save_every_epoch_dir", "") or "")
        if not out_dir:
            return
        offset = int(getattr(cfg, "checkpoint_epoch_offset", 0) or 0)
        global_epoch = offset + int(epoch)
        path = (
            Path(out_dir)
            / f"{self.name}_global_epoch{global_epoch:03d}_{phase}_local_epoch{int(epoch):03d}.pth"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self._checkpoint_payload(
                epoch=epoch,
                phase=phase,
                robust_acc=robust_acc,
            ),
            path,
        )

    def save_final(self) -> None:
        cfg = self.config
        if not self.is_main_rank:
            return
        if not cfg.save:
            return
        path = Path(cfg.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint_payload(
            epoch=self.last_completed_epoch,
            phase="final",
            robust_acc=self.best_robust_acc,
        )
        torch.save(payload, path)
