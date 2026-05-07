import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pretrained_input_icnn.algorithms import ALGORITHMS
from pretrained_input_icnn.config import TrainConfig
from pretrained_input_icnn.utils import to_pixel


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        return self.net(x)


def _loader():
    x = torch.zeros(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=2)


def _config(algorithm: str) -> TrainConfig:
    return TrainConfig(
        algorithm=algorithm,
        lambda_param=3.0,
        use_margin_loss=True,
        npf_hidden=(4,),
        nn_dro_hidden=(4,),
        omega_steps_per_batch=1,
        bb_alpha0=1e-4,
        bb_alpha_min=1e-7,
        bb_alpha_max=1e-3,
        bb_ls_max_steps=2,
        madry_epsilon=0.25,
        madry_pgd_steps=2,
        madry_pgd_step_size=0.1,
        madry_pgd_restarts=1,
        wrm_inner_steps=1,
        wfr_epsilon=0.01,
        wfr_num_samples=2,
        wfr_inner_steps=1,
        wfr_inner_lr=1e-3,
        dual_epsilon=0.01,
        dual_sample_level=1,
        dual_langevin_steps=1,
        dual_langevin_step_size=1e-4,
        dual_mala=True,
        ppa_num_rounds=1,
        ppa_min_rounds=1,
        ppa_round0_steps=1,
        ppa_refine_steps=1,
    )


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_inner_step_freezes_classifier_and_restores_state(algorithm):
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    cfg = _config(algorithm)
    loader = _loader()
    trainer = ALGORITHMS[algorithm](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    x = torch.zeros(2, 3, 32, 32, device=device)
    y = torch.tensor([0, 1], device=device)
    params = list(classifier.parameters())
    buffers = list(classifier.buffers())
    before = [p.detach().clone() for p in params]
    buffers_before = [b.detach().clone() for b in buffers]
    requires_grad_before = [p.requires_grad for p in params]
    training_before = classifier.training
    for p in params:
        p.grad = None

    x_adv = trainer.step(x, y)

    assert x_adv.shape == x.shape
    assert torch.isfinite(x_adv).all()
    assert classifier.training == training_before
    assert [p.requires_grad for p in params] == requires_grad_before
    for p, old in zip(params, before):
        assert torch.allclose(p.detach(), old)
        assert p.grad is None or torch.count_nonzero(p.grad).item() == 0
    for b, old in zip(buffers, buffers_before):
        assert torch.allclose(b.detach(), old)

    if algorithm == "madry":
        l2 = (to_pixel(x_adv) - to_pixel(x)).reshape(x.size(0), -1).norm(p=2, dim=1)
        assert torch.all(l2 <= cfg.madry_epsilon + 1e-5)
    if algorithm == "dual":
        assert trainer._dual_lam_reg == pytest.approx(
            2.0 * cfg.lambda_param * cfg.dual_epsilon
        )


def test_warmup_epoch_does_not_update_classifier_or_batchnorm_buffers():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    cfg = _config("nn_dro")
    loader = _loader()
    trainer = ALGORITHMS["nn_dro"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    params = list(classifier.parameters())
    buffers = list(classifier.buffers())
    params_before = [p.detach().clone() for p in params]
    buffers_before = [b.detach().clone() for b in buffers]
    for p in params:
        p.grad = None

    trainer._train_one_epoch(epoch=1, total_epochs=1, phase="warmup")

    assert classifier.training
    for p, old in zip(params, params_before):
        assert torch.allclose(p.detach(), old)
        assert p.grad is None or torch.count_nonzero(p.grad).item() == 0
    for b, old in zip(buffers, buffers_before):
        assert torch.allclose(b.detach(), old)
