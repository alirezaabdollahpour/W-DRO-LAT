"""PPA (Projected Particle Ascent) DRO baseline.

Alternates WRM ascent with within-class Brenier projections.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from algorithms.base import BaseLinearDRO, make_linear_model
from utils.projections import brenier_projection_features


class PPA(BaseLinearDRO):
    """Projected Particle Ascent for distributionally robust optimisation.

    Round 0:   Plain WRM ascent (replicates WRM — dominance condition).
    Round r>=1: (a) Within-class Brenier projection  z <- Π(z)
                (b) Constant-lr WRM ascent from projected z
    Final:     One last Brenier projection to capture remaining wasted transport.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool = True,
        lambda_param: float = 1.0,
        max_itr: int = 30,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        inner_lr: float = 0.01,
        inner_itr: int = 200,
        ppa_num_rounds: int = 5,
        ppa_min_rounds: int = 2,
        ppa_refine_steps: int = 15,
        ppa_refine_lr: float = 5e-3,
        ppa_delta_rtol: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        super().__init__(input_dim, num_classes, fit_intercept)
        self.lambda_param = lambda_param
        self.inner_lr = inner_lr
        self.inner_itr = inner_itr
        self.max_itr = max_itr
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.ppa_num_rounds = ppa_num_rounds
        self.ppa_min_rounds = ppa_min_rounds
        self.ppa_refine_steps = ppa_refine_steps
        self.ppa_refine_lr = ppa_refine_lr
        self.ppa_delta_rtol = ppa_delta_rtol
        self.model = make_linear_model(
            self.input_dim, self.num_classes, self.fit_intercept, self.device
        )

    def _wrm_ascent(
        self,
        x_orig: torch.Tensor,
        y_orig: torch.Tensor,
        model: nn.Module,
        num_steps: int,
        lr: float,
    ) -> torch.Tensor:
        """WRM gradient ascent with diminishing step size (round 0)."""
        if num_steps == 0:
            return x_orig.detach()
        x_anc = x_orig.detach()
        z = x_anc.clone()
        for s in range(1, num_steps + 1):
            z.requires_grad_(True)
            per_sample_ce = F.cross_entropy(model(z), y_orig, reduction="none")
            grads = torch.autograd.grad(per_sample_ce.sum(), z, create_graph=False)[0]
            eta_s = lr / math.sqrt(s)
            z = z.detach() + eta_s * (
                grads - 2.0 * self.lambda_param * (z.detach() - x_anc)
            )
        return z.detach()

    def _wrm_ascent_const_lr(
        self,
        z0: torch.Tensor,
        x_anchor: torch.Tensor,
        y_orig: torch.Tensor,
        model: nn.Module,
        num_steps: int,
        lr: float,
    ) -> torch.Tensor:
        """Constant-lr WRM ascent for PPA refinement rounds."""
        if num_steps == 0:
            return z0.detach()
        x_anc = x_anchor.detach()
        z = z0.detach().clone()
        for _ in range(num_steps):
            z.requires_grad_(True)
            per_sample_ce = F.cross_entropy(model(z), y_orig, reduction="none")
            grads = torch.autograd.grad(per_sample_ce.sum(), z, create_graph=False)[0]
            z = z.detach() + lr * (
                grads - 2.0 * self.lambda_param * (z.detach() - x_anc)
            )
        return z.detach()

    def _ppa_sampler(
        self,
        x_orig: torch.Tensor,
        y_orig: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        """Full PPA inner loop: round-0 WRM + refinement rounds + final projection."""
        z = self._wrm_ascent(x_orig, y_orig, model, self.inner_itr, self.inner_lr)

        for round_idx in range(1, self.ppa_num_rounds):
            z, _y_proj, delta, _C_id, _C_ot = brenier_projection_features(
                z, x_orig, y_orig
            )

            if (
                round_idx >= self.ppa_min_rounds
                and delta < self.ppa_delta_rtol * max(_C_id, 1e-12)
            ):
                break

            z = self._wrm_ascent_const_lr(
                z,
                x_orig,
                y_orig,
                model,
                self.ppa_refine_steps,
                self.ppa_refine_lr,
            )

        z, _y_proj, _delta_final, _, _ = brenier_projection_features(z, x_orig, y_orig)
        return z.detach()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        checkpoint_dir: str = "checkpoints",
        run_id: int = 0,
    ) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        loss_history: List[float] = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(
                dataloader,
                desc=f"Run {run_id+1} Training PPA Epoch {epoch+1}/{self.max_itr}",
                leave=False,
            )
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev = x_original_batch.to(self.device)
                y_original_batch_dev = y_original_batch.to(self.device)
                self.model.eval()
                x_ppa_batch = self._ppa_sampler(
                    x_original_batch_dev, y_original_batch_dev, self.model
                )
                self.model.train()
                predictions_logits_batch = self.model(x_ppa_batch)
                optimization_loss = nn.CrossEntropyLoss()(
                    predictions_logits_batch, y_original_batch_dev
                )
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()
                with torch.no_grad():
                    comparable_loss = nn.CrossEntropyLoss()(
                        self.model(x_original_batch_dev), y_original_batch_dev
                    )
                epoch_loss_record += comparable_loss.item()
                pbar.set_postfix(
                    optim_loss=optimization_loss.item(),
                    record_loss=comparable_loss.item(),
                )
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            torch.save(
                self.model.state_dict(),
                f"{checkpoint_dir}/PPA_run{run_id}_epoch_{epoch+1}.pth",
            )
        return loss_history
