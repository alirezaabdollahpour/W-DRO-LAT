import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pretrained_input_icnn.algorithms import ALGORITHMS
from pretrained_input_icnn.config import TrainConfig, build_arg_parser, config_from_args
from pretrained_input_icnn.models.classifier import SimpleViTCIFAR, load_pretrained_classifier
from pretrained_input_icnn.utils import normalized_mse, pixel_l2_squared, to_pixel
from pretrained_input_icnn.utils.eval import (
    _pgd_loss_per_sample,
    evaluate_clean,
    evaluate_under_input_pgd,
)


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


def _batchnorm_affine_params(module):
    params = []
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            if child.weight is not None:
                params.append(child.weight)
            if child.bias is not None:
                params.append(child.bias)
    return params


def _batchnorm_modules(module):
    return [
        child
        for child in module.modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    ]


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


def test_freeze_batchnorm_affine_keeps_bn_weight_bias_fixed():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    cfg = replace(
        _config("nn_dro"),
        freeze_batchnorm=True,
        freeze_batchnorm_affine=True,
    )
    loader = _loader()
    trainer = ALGORITHMS["nn_dro"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    bn_params = _batchnorm_affine_params(classifier)
    bn_param_ids = {id(p) for p in bn_params}
    assert bn_params
    assert all(not p.requires_grad for p in bn_params)
    optimizer_param_ids = {
        id(p)
        for group in trainer.optimizer.param_groups
        for p in group["params"]
    }
    assert all(id(p) not in optimizer_param_ids for p in bn_params)

    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    bn_before = [p.detach().clone() for p in bn_params]
    non_bn_before = [
        p.detach().clone()
        for p in classifier.parameters()
        if id(p) not in bn_param_ids
    ]
    non_bn_params = [p for p in classifier.parameters() if id(p) not in bn_param_ids]

    trainer.classifier_update(x, y)

    for p, old in zip(bn_params, bn_before):
        assert not p.requires_grad
        assert p.grad is None
        assert torch.allclose(p.detach(), old)
    assert any(
        not torch.allclose(p.detach(), old)
        for p, old in zip(non_bn_params, non_bn_before)
    )


def test_inner_step_preserves_frozen_batchnorm_eval_mode():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    cfg = replace(_config("nn_dro"), freeze_batchnorm=True)
    loader = _loader()
    trainer = ALGORITHMS["nn_dro"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    trainer._prepare_classifier_for_update()
    assert classifier.training
    assert all(not module.training for module in _batchnorm_modules(classifier))

    x, y = next(iter(loader))
    trainer.step(x.to(device), y.to(device))

    assert classifier.training
    assert all(not module.training for module in _batchnorm_modules(classifier))


def test_batch_adversary_diagnostics_preserves_frozen_batchnorm_eval_mode():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    cfg = replace(_config("nn_dro"), freeze_batchnorm=True)
    loader = _loader()
    trainer = ALGORITHMS["nn_dro"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    trainer._prepare_classifier_for_update()
    assert classifier.training
    assert all(not module.training for module in _batchnorm_modules(classifier))

    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    trainer._batch_adversary_diagnostics(x, x, y)

    assert classifier.training
    assert all(not module.training for module in _batchnorm_modules(classifier))


def test_freeze_batchnorm_affine_happens_before_ddp_wrap(monkeypatch):
    from pretrained_input_icnn.algorithms import base as base_module

    class FakeDistInfo:
        is_distributed = True
        local_rank = 0

    captured_bn_requires_grad = []

    class FakeDDP(nn.Module):
        def __init__(self, module, *args, **kwargs):
            super().__init__()
            self.module = module
            del args, kwargs
            captured_bn_requires_grad.extend(
                p.requires_grad for p in _batchnorm_affine_params(module)
            )

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    monkeypatch.setattr(base_module.dist_helpers, "info", lambda: FakeDistInfo())
    monkeypatch.setattr(base_module, "DDP", FakeDDP)

    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    cfg = replace(_config("nn_dro"), freeze_batchnorm_affine=True)
    loader = _loader()

    trainer = ALGORITHMS["nn_dro"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    assert captured_bn_requires_grad
    assert captured_bn_requires_grad == [False] * len(captured_bn_requires_grad)
    assert all(not p.requires_grad for p in _batchnorm_affine_params(trainer.classifier_module))


def test_online_batchnorm_refresh_updates_bn_buffers_when_batchnorm_is_frozen():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    with torch.no_grad():
        classifier.net[1].weight.zero_()
        classifier.net[1].bias.fill_(2.0)
    cfg = replace(
        _config("nn_dro"),
        freeze_batchnorm=True,
        freeze_batchnorm_affine=True,
        online_batchnorm_refresh=True,
        batchnorm_online_refresh_momentum=1.0,
    )
    loader = _loader()
    trainer = ALGORITHMS["nn_dro"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    bn = classifier.net[2]
    trainer._prepare_classifier_for_update()
    assert not bn.training
    params_before = [p.detach().clone() for p in classifier.parameters()]
    buffers_before = [b.detach().clone() for b in classifier.buffers()]
    for p in classifier.parameters():
        p.grad = None

    x = torch.ones(2, 3, 32, 32, device=device)
    stats = trainer._online_refresh_batchnorm(x)

    assert stats["bn_online_refresh_batches"] == 1.0
    assert stats["bn_online_refresh_samples"] == 2.0
    assert torch.allclose(bn.running_mean, torch.full_like(bn.running_mean, 2.0))
    assert not bn.training
    for p, old in zip(classifier.parameters(), params_before):
        assert torch.allclose(p.detach(), old)
        assert p.grad is None
    assert any(
        not torch.allclose(b.detach(), old)
        for b, old in zip(classifier.buffers(), buffers_before)
    )
    assert all(not p.requires_grad for p in _batchnorm_affine_params(classifier))


def test_online_batchnorm_refresh_updates_only_bn_buffers_when_bn_is_mutable():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    with torch.no_grad():
        classifier.net[1].weight.zero_()
        classifier.net[1].bias.fill_(2.0)
    cfg = replace(
        _config("nn_dro"),
        freeze_batchnorm=False,
        freeze_batchnorm_affine=True,
        online_batchnorm_refresh=True,
        batchnorm_online_refresh_momentum=1.0,
    )
    loader = _loader()
    trainer = ALGORITHMS["nn_dro"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    bn = classifier.net[2]
    trainer._prepare_classifier_for_update()
    assert bn.training
    params_before = [p.detach().clone() for p in classifier.parameters()]
    for p in classifier.parameters():
        p.grad = None

    x = torch.ones(2, 3, 32, 32, device=device)
    stats = trainer._online_refresh_batchnorm(x)

    assert stats["bn_online_refresh_batches"] == 1.0
    assert stats["bn_online_refresh_samples"] == 2.0
    assert torch.allclose(bn.running_mean, torch.full_like(bn.running_mean, 2.0))
    assert bn.training
    for p, old in zip(classifier.parameters(), params_before):
        assert torch.allclose(p.detach(), old)
        assert p.grad is None
    assert all(not p.requires_grad for p in _batchnorm_affine_params(classifier))


def test_online_batchnorm_refresh_is_noop_for_classifier_without_batchnorm():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 16),
        nn.LayerNorm(16),
        nn.ReLU(),
        nn.Linear(16, 10),
    ).to(device)
    cfg = replace(
        _config("nn_dro"),
        freeze_batchnorm=True,
        online_batchnorm_refresh=True,
        batchnorm_online_refresh_momentum=1.0,
    )
    loader = _loader()
    trainer = ALGORITHMS["nn_dro"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    params_before = [p.detach().clone() for p in classifier.parameters()]
    buffers_before = [b.detach().clone() for b in classifier.buffers()]
    stats = trainer._online_refresh_batchnorm(torch.ones(2, 3, 32, 32, device=device))

    assert stats == {}
    assert not _batchnorm_modules(classifier)
    for p, old in zip(classifier.parameters(), params_before):
        assert torch.allclose(p.detach(), old)
    for b, old in zip(classifier.buffers(), buffers_before):
        assert torch.allclose(b.detach(), old)


def test_online_batchnorm_refresh_cli_allows_frozen_batchnorm():
    parser = build_arg_parser()
    args = parser.parse_args(["--online-batchnorm-refresh"])

    cfg = config_from_args(args)

    assert cfg.freeze_batchnorm
    assert cfg.online_batchnorm_refresh


def test_simple_vit_checkpoint_loader_has_no_batchnorm(tmp_path):
    torch.manual_seed(123)
    device = torch.device("cpu")
    ckpt_path = tmp_path / "vit_tiny.pth"
    model = SimpleViTCIFAR(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=32,
        depth=1,
        heads=4,
        mlp_dim=64,
        dim_head=8,
    )
    torch.save({"best": model.state_dict()}, ckpt_path)

    loaded = load_pretrained_classifier(str(ckpt_path), device=device)
    with torch.no_grad():
        logits = loaded(torch.randn(2, 3, 32, 32, device=device))

    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    assert not _batchnorm_modules(loaded)
    assert any(isinstance(module, nn.LayerNorm) for module in loaded.modules())


def test_simple_vit_checkpoint_loader_prefers_swa_last(tmp_path):
    torch.manual_seed(123)
    device = torch.device("cpu")
    ckpt_path = tmp_path / "vit_tiny_branches.pth"
    model = SimpleViTCIFAR(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=32,
        depth=1,
        heads=4,
        mlp_dim=64,
        dim_head=8,
    )
    last_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    swa_last_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    last_state["linear_head.1.bias"].fill_(0.0)
    swa_last_state["linear_head.1.bias"].fill_(7.0)
    torch.save({"last": last_state, "swa_last": swa_last_state}, ckpt_path)

    loaded = load_pretrained_classifier(str(ckpt_path), device=device)

    assert torch.allclose(
        loaded.linear_head[1].bias.detach(),
        torch.full_like(loaded.linear_head[1].bias.detach(), 7.0),
    )


def test_online_batchnorm_refresh_runs_per_classifier_update_batch():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    cfg = replace(
        _config("nn_dro"),
        freeze_batchnorm=False,
        online_batchnorm_refresh=True,
    )
    loader = _loader()
    trainer = ALGORITHMS["nn_dro"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )
    calls = 0

    def fake_refresh(x):
        nonlocal calls
        calls += 1
        return {
            "bn_online_refresh_seconds": 0.25,
            "bn_online_refresh_batches": 1.0,
            "bn_online_refresh_samples": float(x.size(0)),
        }

    trainer._online_refresh_batchnorm = fake_refresh
    metrics = trainer._train_one_epoch(epoch=1, total_epochs=1, phase="adv")

    assert calls == len(loader)
    assert metrics.extras["bn_online_refresh_seconds"] == pytest.approx(0.25)
    assert metrics.extras["bn_online_refresh_batches"] == pytest.approx(float(len(loader)))
    assert metrics.extras["bn_online_refresh_samples"] == pytest.approx(2.0)


def test_frozen_adversary_epoch_reuses_map_without_updating_adversary():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    cfg = replace(_config("nn_dro"), frozen_adversary_map_steps=3)
    loader = _loader()
    trainer = ALGORITHMS["nn_dro"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    classifier_before = [p.detach().clone() for p in classifier.parameters()]
    adversary_before = [p.detach().clone() for p in trainer.adversary.parameters()]
    calls = 0

    def counting_transport(x):
        nonlocal calls
        calls += 1
        return x + 0.01

    def forbidden_step(x, y):
        raise AssertionError("frozen_adversary phase must not update the adversary")

    trainer.transport_for_eval = counting_transport
    trainer.step = forbidden_step

    trainer._train_one_epoch(epoch=1, total_epochs=1, phase="frozen_adversary")

    assert calls == len(loader) * cfg.frozen_adversary_map_steps
    assert any(
        not torch.allclose(p.detach(), old)
        for p, old in zip(classifier.parameters(), classifier_before)
    )
    for p, old in zip(trainer.adversary.parameters(), adversary_before):
        assert torch.allclose(p.detach(), old)


def test_transport_cost_defaults_to_legacy_normalized_mse_and_cli_override():
    parser = build_arg_parser()

    default_cfg = config_from_args(parser.parse_args([]))
    pixel_cfg = config_from_args(
        parser.parse_args(["--transport-cost", "pixel_l2_squared"])
    )

    assert default_cfg.transport_cost == "normalized_mse"
    assert pixel_cfg.transport_cost == "pixel_l2_squared"


def test_trainer_transport_cost_matches_legacy_normalized_mse_by_default():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    loader = _loader()
    cfg = replace(_config("madry"), transport_cost="normalized_mse")
    trainer = ALGORITHMS["madry"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    x = torch.randn(2, 3, 32, 32, device=device)
    x_adv = x + torch.randn_like(x) * 0.01

    assert torch.allclose(trainer._transport_cost(x_adv, x), normalized_mse(x_adv, x))

    pixel_cfg = replace(cfg, transport_cost="pixel_l2_squared")
    pixel_trainer = ALGORITHMS["madry"](
        classifier=TinyClassifier().to(device),
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=pixel_cfg,
    )
    assert torch.allclose(
        pixel_trainer._transport_cost(x_adv, x),
        pixel_l2_squared(x_adv, x),
    )


def test_input_pgd_loss_cli_defaults_to_margin_and_allows_ce():
    parser = build_arg_parser()

    default_cfg = config_from_args(parser.parse_args([]))
    ce_cfg = config_from_args(parser.parse_args(["--input-pgd-loss", "ce"]))

    assert default_cfg.input_pgd_loss == "margin"
    assert ce_cfg.input_pgd_loss == "ce"


def test_logsumexp_margin_pgd_loss_avoids_ce_saturation():
    y = torch.tensor([0])
    ce_logits = torch.tensor([[50.0, 0.0, -1.0]], requires_grad=True)
    margin_logits = ce_logits.detach().clone().requires_grad_(True)

    _pgd_loss_per_sample(ce_logits, y, "ce").sum().backward()
    _pgd_loss_per_sample(margin_logits, y, "margin").sum().backward()

    assert ce_logits.grad.abs().max().item() < 1e-6
    assert margin_logits.grad.abs().max().item() > 0.5


def test_eval_helpers_restore_classifier_training_modes():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    loader = _loader()

    classifier.train()
    for module in _batchnorm_modules(classifier):
        module.eval()
    assert classifier.training
    assert all(not module.training for module in _batchnorm_modules(classifier))

    evaluate_clean(classifier, loader, device)

    assert classifier.training
    assert all(not module.training for module in _batchnorm_modules(classifier))

    evaluate_under_input_pgd(
        classifier,
        loader,
        device,
        p=2,
        eps=0.1,
        steps=1,
        step_size=0.1,
        restarts=1,
        max_samples=2,
        loss="margin",
    )

    assert classifier.training
    assert all(not module.training for module in _batchnorm_modules(classifier))


def test_input_pgd_rejects_unknown_loss():
    with pytest.raises(ValueError, match="input PGD loss"):
        _pgd_loss_per_sample(torch.zeros(1, 3), torch.tensor([0]), "bad")
