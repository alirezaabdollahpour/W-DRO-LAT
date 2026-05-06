"""Per-sample primary adversary loss: cross-entropy or log-sum-exp margin.

Mirrors the ``use_margin_adv`` branch of the legacy
``pretrained_INPUT_icnn.dro_inner_objective_per_sample``:

    margin_per_sample(x, y) = logsumexp_{j != y} ( logit_j - logit_y )

Strictly larger than CE for use as an adversarial objective (non-zero-sum
view). Toggled via ``cfg.use_margin_loss`` and a ``--use-margin-loss``
CLI flag.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def adversary_loss_per_sample(
    logits: torch.Tensor, y: torch.Tensor, use_margin: bool
) -> torch.Tensor:
    """Return a (B,) tensor of per-sample adversary losses."""
    if not use_margin:
        return F.cross_entropy(logits, y, reduction="none")
    logits_correct = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    margins = logits - logits_correct.unsqueeze(1)
    mask = F.one_hot(y, num_classes=logits.size(1)).bool()
    margins = margins.masked_fill(mask, float("-inf"))
    return torch.logsumexp(margins, dim=1)
