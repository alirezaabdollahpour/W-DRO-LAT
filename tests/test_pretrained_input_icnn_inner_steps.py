import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pretrained_input_icnn import distributed as dist_helpers
from pretrained_input_icnn.algorithms import ALGORITHMS
from pretrained_input_icnn.algorithms.base import BaseAdvTrainer
from pretrained_input_icnn.config import TrainConfig, build_arg_parser, config_from_args
from pretrained_input_icnn.distributed import DistInfo
from pretrained_input_icnn.models.classifier import SimpleViTCIFAR, load_pretrained_classifier
from pretrained_input_icnn.models.npf import (
    NPFInputConvexPotential,
    NPFDense,
    NPFPosDefPotentials,
    convex_init_parameters,
    npf_T_omega,
)
from pretrained_input_icnn.utils import (
    BBArmijoState,
    bb_armijo_step_params,
    normalized_mse,
    pixel_l2_squared,
    to_pixel,
)
from pretrained_input_icnn.utils.eval import (
    _pgd_loss_per_sample,
    evaluate_clean,
    evaluate_transport_pgd_alignment,
    evaluate_under_input_pgd,
)




class AlwaysZeroClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(10))
        with torch.no_grad():
            self.bias[0] = 1.0

    def forward(self, x):
        return self.bias.view(1, -1).expand(x.size(0), -1)


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
        npf_lastquad_hidden=(4,),
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


def test_npf_positive_dense_uses_principled_lognormal_moments():
    torch.manual_seed(123)
    fan_in = 64
    layer = NPFDense(
        fan_in,
        4096,
        pos_weights=True,
        rectifier="relu",
    )
    projected = layer.projected_weight().detach()
    weight_mean_sq, weight_var, _bias_mean, _bias_var = convex_init_parameters(fan_in)

    assert torch.all(projected > 0.0)
    assert projected.mean().item() == pytest.approx(
        math.sqrt(weight_mean_sq),
        rel=0.08,
    )
    assert projected.var(unbiased=False).item() == pytest.approx(
        weight_var,
        rel=0.15,
    )


def test_npf_unprojected_dense_identity_helpers_zero_raw_weights():
    torch.manual_seed(123)
    layer = NPFDense(
        8,
        4,
        pos_weights=False,
        rectifier="softplus",
    )

    layer.zero_()
    assert torch.count_nonzero(layer.weight.detach()).item() == 0

    layer.near_zero_(1e-3)
    assert layer.weight.detach().abs().max().item() < 2e-3


def test_npf_principled_bias_shift_is_applied_to_preactivation_biases():
    torch.manual_seed(123)
    hidden_sizes = (16, 12)
    model = NPFInputConvexPotential(
        input_dim=8,
        hidden_sizes=hidden_sizes,
        outer_rank=0,
        inner_rank=1,
        output_rank=0,
        quadratic_mode="last_layer_diagonal",
        activation="relu",
        pos_weights=True,
        positive_weight_rectifier="relu",
    )
    first_hidden_skip = model.w_xs[0]
    second_hidden_skip = model.w_xs[1]
    output_skip = model.residual_output_potential
    # Legacy golden-run parity: the bias shift is +|bias_mean| (the
    # Hoedt-Klambauer derivation gives the negative value; the legacy 57%
    # run used its exact positive mirror, which keeps the deep stack
    # near-fully active and its sample-dependent gradients hot).
    _, _, second_bias_mean, _ = convex_init_parameters(hidden_sizes[0])
    _, _, output_bias_mean, _ = convex_init_parameters(hidden_sizes[1])
    second_bias_mean = abs(second_bias_mean)
    output_bias_mean = abs(output_bias_mean)

    assert isinstance(first_hidden_skip, nn.Linear)
    assert torch.count_nonzero(first_hidden_skip.bias.detach()).item() == 0
    assert isinstance(second_hidden_skip, nn.Linear)
    assert torch.allclose(
        second_hidden_skip.bias.detach(),
        torch.full_like(second_hidden_skip.bias.detach(), second_bias_mean),
    )
    assert torch.allclose(
        output_skip.bias.detach(),
        torch.full_like(output_skip.bias.detach(), output_bias_mean),
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


def test_parameter_l2_deltas_track_classifier_and_npf_adversary():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    cfg = replace(
        _config("npf_lastquad"),
        npf_lastquad_hidden=(4,),
    )
    loader = _loader()
    trainer = ALGORITHMS["npf_lastquad"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    trainer._ensure_parameter_delta_baseline()
    theta_param = next(trainer.classifier_module.parameters())
    omega_param = next(trainer.adversary_delta_parameters())
    theta_step = 0.25
    omega_step = 0.5

    with torch.no_grad():
        theta_param.add_(theta_step)
        omega_param.add_(omega_step)

    extras = trainer._finish_parameter_delta_epoch()

    assert extras["theta_l2_delta"] == pytest.approx(
        theta_step * math.sqrt(theta_param.numel())
    )
    assert extras["omega_l2_delta"] == pytest.approx(
        omega_step * math.sqrt(omega_param.numel())
    )
    unchanged = trainer._finish_parameter_delta_epoch()

    assert unchanged["theta_l2_delta"] == pytest.approx(0.0)
    assert unchanged["omega_l2_delta"] == pytest.approx(0.0)


def test_npf_lastquad_final_posdef_potential_is_ott_scaled_and_trainable():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    cfg = replace(
        _config("npf_lastquad"),
        npf_lastquad_hidden=(4,),
        npf_lastquad_activation="softplus",
    )
    loader = _loader()
    trainer = ALGORITHMS["npf_lastquad"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    q_out = trainer.psi_omega.residual_output_potential
    # Output block starts at effective diag 0.01 (near-identity transport at
    # init; the old softplus(-2)=0.1269 planted a measured 1.5-pixel-L2
    # global-rescale artifact) with the never-dead 'exp' diag rectifier
    # (relu had a permanent zero-gradient dead zone once the cost gradient
    # pushed raw entries negative).
    expected_diag = 0.01

    assert isinstance(q_out, NPFPosDefPotentials)
    assert q_out.diag_rectifier == "exp"
    assert q_out.diag.mean().item() == pytest.approx(expected_diag)

    x = torch.randn(2, 3, 32, 32, device=device)
    x_adv = npf_T_omega(x, trainer.psi_omega, create_graph=True)
    loss = x_adv.pow(2).mean()
    grad = torch.autograd.grad(loss, q_out.diag_kernel)[0]

    assert torch.isfinite(grad).all()
    assert grad.norm().item() > 1e-7


def test_npf_lastquad_can_use_learnable_low_rank_output_quadratic():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    cfg = replace(
        _config("npf_lastquad"),
        npf_lastquad_hidden=(4,),
        npf_lastquad_output_rank=2,
        npf_lastquad_activation="softplus",
    )
    loader = _loader()
    trainer = ALGORITHMS["npf_lastquad"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    assert not trainer.psi_omega.use_hidden_quadratics
    assert all(isinstance(module, nn.Linear) for module in trainer.psi_omega.w_xs[:-1])
    q_out = trainer.psi_omega.residual_output_potential
    assert q_out.quad_kernel is not None
    assert q_out.quad_kernel.shape == (1, trainer.input_dim, 2)

    x = torch.randn(2, 3, 32, 32, device=device)
    x_adv = npf_T_omega(x, trainer.psi_omega, create_graph=True)
    loss = x_adv.pow(2).mean()
    grad = torch.autograd.grad(loss, q_out.quad_kernel)[0]

    assert torch.isfinite(grad).all()
    assert grad.norm().item() > 0.0


def test_npf_lastquad_honors_frozen_outer_identity_potential():
    torch.manual_seed(123)
    model = NPFInputConvexPotential(
        input_dim=8,
        hidden_sizes=(4,),
        outer_rank=0,
        inner_rank=0,
        output_rank=2,
        quadratic_mode="last_layer_diagonal",
        trainable_outer_quadratic=False,
        activation="softplus",
    )

    assert all(not p.requires_grad for p in model.pos_def_potential.parameters())
    assert any(p.requires_grad for p in model.residual_output_potential.parameters())
    assert any(p.requires_grad for p in model.w_zs.parameters())

    x = torch.randn(3, 8)
    x_adv = npf_T_omega(x, model, create_graph=True)
    loss = x_adv.pow(2).mean()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, trainable_params, allow_unused=True)

    assert all(g is None or torch.isfinite(g).all() for g in grads)


def test_npf_lastquad_trainer_keeps_outer_identity_frozen_during_step():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    cfg = replace(
        _config("npf_lastquad"),
        npf_lastquad_hidden=(4,),
        npf_lastquad_output_rank=2,
        npf_lastquad_identity_init=True,
        attack_clean_correct_only=False,
    )
    loader = _loader()
    trainer = ALGORITHMS["npf_lastquad"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    assert all(not p.requires_grad for p in trainer.psi_omega.pos_def_potential.parameters())
    x, y = next(iter(loader))
    trainer.step(x.to(device), y.to(device))
    assert all(not p.requires_grad for p in trainer.psi_omega.pos_def_potential.parameters())


def test_full_npf_uses_ott_posdef_blocks_and_identity_base_potential():
    torch.manual_seed(123)
    model = NPFInputConvexPotential(
        input_dim=8,
        hidden_sizes=(4,),
        outer_rank=2,
        inner_rank=2,
        quadratic_mode="all_layers",
        trainable_outer_quadratic=True,
        activation="softplus",
    )

    assert all(isinstance(module, NPFPosDefPotentials) for module in model.w_xs)
    assert isinstance(model.residual_output_potential, NPFPosDefPotentials)
    assert model.residual_output_potential.quad_kernel.shape == (1, 8, 2)
    assert model.pos_def_potential.diag.mean().item() == pytest.approx(1.0)
    # Trainable outer quadratic must NOT start at exact zeros: grad of
    # 0.5*||A^T x||^2 w.r.t. A vanishes identically at A=0 (exact saddle), so
    # a zeros init would freeze it forever. The base potential stays
    # near-identity via a small draw instead.
    assert torch.count_nonzero(model.pos_def_potential.quad_kernel).item() > 0
    assert model.pos_def_potential.quad_kernel.abs().max().item() < 0.05
    assert torch.count_nonzero(model.pos_def_potential.lin_kernel).item() == 0

    frozen = NPFInputConvexPotential(
        input_dim=8,
        hidden_sizes=(4,),
        outer_rank=2,
        inner_rank=2,
        quadratic_mode="all_layers",
        trainable_outer_quadratic=False,
        activation="softplus",
    )
    assert torch.count_nonzero(frozen.pos_def_potential.quad_kernel).item() == 0

    x = torch.randn(3, 8)
    y = torch.randn(3, 8)
    t = torch.rand(3, 1)
    lhs = model(t * x + (1.0 - t) * y)
    rhs = t.squeeze(1) * model(x) + (1.0 - t.squeeze(1)) * model(y)

    assert torch.all(lhs <= rhs + 1e-5)


def test_bb_armijo_ignores_frozen_parameters():
    p_train = nn.Parameter(torch.tensor([1.0]))
    p_frozen = nn.Parameter(torch.tensor([7.0]), requires_grad=False)
    state = BBArmijoState.create(alpha0=0.1, alpha_min=1e-6, alpha_max=1.0)

    def objective(create_graph: bool):
        del create_graph
        return -((p_train - 2.0) ** 2).sum()

    _, _, f_val, grad_norm = bb_armijo_step_params(
        [p_train, p_frozen],
        objective,
        state,
    )

    assert math.isfinite(f_val)
    assert grad_norm > 0.0
    assert p_train.item() > 1.0
    assert p_frozen.item() == pytest.approx(7.0)


def test_npf_lastquad_output_rank_cli_is_forwarded_to_config():
    parser = build_arg_parser()

    cfg = config_from_args(
        parser.parse_args(["--npf-lastquad-output-rank", "4"])
    )

    assert cfg.npf_lastquad_output_rank == 4


def test_npf_identity_init_cli_switches_are_forwarded_to_config():
    parser = build_arg_parser()

    default_cfg = config_from_args(parser.parse_args([]))
    disabled_cfg = config_from_args(
        parser.parse_args(
            [
                "--no-npf-identity-init",
                "--no-npf-lastquad-identity-init",
            ]
        )
    )
    enabled_cfg = config_from_args(
        parser.parse_args(
            [
                "--npf-identity-init",
                "--npf-lastquad-identity-init",
            ]
        )
    )

    # Full-NPF default is the identity-adjacent start (the non-identity
    # all-layers init was measured 12.9 pixel-L2 off identity at objective
    # -40); LastQuad default stays random (its random init is benign and the
    # legacy golden run started random).
    assert default_cfg.npf_identity_init
    assert not default_cfg.npf_lastquad_identity_init
    assert not disabled_cfg.npf_identity_init
    assert not disabled_cfg.npf_lastquad_identity_init
    assert enabled_cfg.npf_identity_init
    assert enabled_cfg.npf_lastquad_identity_init


def test_npf_lastquad_identity_init_switch_controls_initial_map():
    torch.manual_seed(123)
    device = torch.device("cpu")
    loader = _loader()

    identity_cfg = replace(
        _config("npf_lastquad"),
        npf_lastquad_hidden=(4,),
        npf_lastquad_identity_init=True,
        npf_lastquad_init_eps=0.0,
    )
    identity_trainer = ALGORITHMS["npf_lastquad"](
        classifier=TinyClassifier().to(device),
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=identity_cfg,
    )

    hidden_skip = identity_trainer.psi_omega.w_xs[0]
    output_skip = identity_trainer.psi_omega.residual_output_potential
    assert isinstance(hidden_skip, nn.Linear)
    assert hidden_skip.weight.detach().abs().sum().item() == pytest.approx(0.0)
    assert output_skip.diag.detach().abs().sum().item() == pytest.approx(0.0)
    assert output_skip.lin_kernel.detach().abs().sum().item() == pytest.approx(0.0)

    x = torch.randn(2, 3, 32, 32, device=device)
    x_adv = npf_T_omega(x, identity_trainer.psi_omega, create_graph=False)
    relative_error = (x_adv - x).norm() / x.norm().clamp_min(1e-12)

    assert relative_error.item() < 1e-6

    torch.manual_seed(123)
    raw_cfg = replace(identity_cfg, npf_lastquad_identity_init=False)
    raw_trainer = ALGORITHMS["npf_lastquad"](
        classifier=TinyClassifier().to(device),
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=raw_cfg,
    )

    raw_hidden_skip = raw_trainer.psi_omega.w_xs[0]
    raw_output_skip = raw_trainer.psi_omega.residual_output_potential
    assert isinstance(raw_hidden_skip, nn.Linear)
    assert raw_hidden_skip.weight.detach().abs().sum().item() > 0.0
    assert raw_output_skip.lin_kernel.detach().abs().sum().item() > 0.0


def test_npf_lastquad_identity_init_default_is_not_dead_zero():
    torch.manual_seed(123)
    device = torch.device("cpu")
    loader = _loader()
    cfg = replace(
        _config("npf_lastquad"),
        npf_lastquad_hidden=(4,),
        npf_lastquad_identity_init=True,
    )
    trainer = ALGORITHMS["npf_lastquad"](
        classifier=TinyClassifier().to(device),
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    hidden_skip = trainer.psi_omega.w_xs[0]
    output_skip = trainer.psi_omega.residual_output_potential
    assert isinstance(hidden_skip, nn.Linear)
    assert hidden_skip.weight.detach().abs().sum().item() > 0.0
    assert output_skip.lin_kernel.detach().abs().sum().item() > 0.0


def test_bb_ascent_convex_region_proposes_legacy_positive_step():
    # Legacy golden-run BB semantics: in a locally CONVEX region along the
    # ascent path (<s, y> > 0) the proposal is +<s,s>/<s,y> (often large,
    # Armijo shrinks it), NOT a fallback to alpha_min — the old alpha_min
    # pin deadlocked the inner ascent out of flat starts.
    state = BBArmijoState.create(
        alpha0=0.1,
        alpha_min=1e-5,
        alpha_max=1.0,
    )
    state = state.update_history(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        alpha=0.1,
    )

    alpha = state.propose(torch.tensor([1.0]), torch.tensor([1.0]))

    # s = 1, y = 1 -> <s,s>/<s,y> = 1.0 (clamped to alpha_max).
    assert alpha == pytest.approx(1.0)


def test_bb_ascent_tiny_noisy_secant_keeps_previous_alpha():
    # Near-zero curvature within the scale-aware tolerance must not be
    # treated as real curvature; the proposal falls back to alpha_prev.
    state = BBArmijoState.create(alpha0=0.1, alpha_min=1e-5, alpha_max=1.0)
    state = state.update_history(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        alpha=0.1,
    )

    alpha = state.propose(torch.tensor([1e-9]), torch.tensor([1.0 + 1e-16]))

    assert alpha == pytest.approx(0.1)


def test_bb_armijo_clips_parametric_gradient_before_step():
    param = nn.Parameter(torch.tensor([0.0]))
    state = BBArmijoState.create(
        alpha0=1.0,
        alpha_min=1.0,
        alpha_max=1.0,
        ls_c=0.0,
        ls_max_steps=1,
    )

    def objective(create_graph: bool) -> torch.Tensor:
        del create_graph
        return 10.0 * param.sum()

    _, _, _, grad_norm = bb_armijo_step_params(
        [param],
        objective,
        state,
        max_grad_norm=1.0,
    )

    assert grad_norm == pytest.approx(1.0)
    assert param.detach().item() == pytest.approx(1.0)


def test_bb_armijo_rejection_refreshes_history_like_legacy_icnn():
    param = nn.Parameter(torch.tensor([0.0]))
    state = BBArmijoState.create(
        alpha0=1.0,
        alpha_min=1.0,
        alpha_max=1.0,
        ls_c=0.1,
        ls_max_steps=1,
        reject_on_armijo_failure=True,
    )

    def objective(create_graph: bool) -> torch.Tensor:
        del create_graph
        return -param.pow(2).sum() + param.sum()

    _, new_state, _, grad_norm = bb_armijo_step_params([param], objective, state)

    assert param.detach().item() == pytest.approx(0.0)
    assert grad_norm == pytest.approx(1.0)
    assert new_state.prev_params_vec is not None
    assert new_state.prev_grad_vec is not None
    assert torch.allclose(new_state.prev_params_vec, torch.tensor([0.0]))
    assert torch.allclose(new_state.prev_grad_vec, torch.tensor([1.0]))


def test_shared_adversary_masked_mean_uses_global_count_for_ddp(monkeypatch):
    trainer = BaseAdvTrainer.__new__(BaseAdvTrainer)
    trainer.dist = DistInfo(world_size=2, rank=0, local_rank=0, backend="test")

    values = torch.tensor([2.0, 5.0], requires_grad=True)
    mask = torch.tensor([True, False])

    def fake_global_count(x: torch.Tensor) -> torch.Tensor:
        assert x.item() == pytest.approx(1.0)
        return x.new_tensor(4.0)

    monkeypatch.setattr(dist_helpers, "all_reduce_sum_scalar", fake_global_count)

    objective = trainer._shared_adversary_masked_mean(values, mask)
    objective.backward()

    assert objective.item() == pytest.approx(1.0)
    assert torch.allclose(values.grad, torch.tensor([0.5, 0.0]))


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
    # freeze_batchnorm now defaults to False (legacy/flagship: BN
    # participates); the combination under test needs the explicit flag.
    args = parser.parse_args(["--freeze-batchnorm", "--online-batchnorm-refresh"])

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

    ablation_cfg = config_from_args(
        parser.parse_args(["--attack-all-samples", "--persistent-parametric-bb"])
    )

    assert default_cfg.transport_cost == "normalized_mse"
    assert default_cfg.attack_clean_correct_only
    assert default_cfg.reset_parametric_bb_each_batch
    assert default_cfg.parametric_bb_max_grad_norm == pytest.approx(1.0)
    assert pixel_cfg.transport_cost == "pixel_l2_squared"
    assert not ablation_cfg.attack_clean_correct_only
    assert not ablation_cfg.reset_parametric_bb_each_batch

    unclipped_cfg = config_from_args(
        parser.parse_args(["--parametric-bb-max-grad-norm", "0"])
    )
    assert unclipped_cfg.parametric_bb_max_grad_norm == pytest.approx(0.0)


def test_disabled_frozen_adversary_allows_zero_map_steps():
    parser = build_arg_parser()

    cfg = config_from_args(
        parser.parse_args(
            [
                "--frozen-adversary-epochs",
                "0",
                "--frozen-adversary-map-steps",
                "0",
            ]
        )
    )

    assert cfg.frozen_adversary_epochs == 0
    assert cfg.frozen_adversary_map_steps == 1

    with pytest.raises(ValueError, match="frozen-adversary-map-steps"):
        config_from_args(
            parser.parse_args(
                [
                    "--frozen-adversary-epochs",
                    "1",
                    "--frozen-adversary-map-steps",
                    "0",
                ]
            )
        )


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


def test_npf_legacy_attack_mask_keeps_clean_incorrect_samples_clean():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = AlwaysZeroClassifier().to(device)
    loader = _loader()
    cfg = replace(
        _config("npf_lastquad"),
        attack_clean_correct_only=True,
        reset_parametric_bb_each_batch=True,
        npf_lastquad_hidden=(4,),
        omega_steps_per_batch=1,
    )
    trainer = ALGORITHMS["npf_lastquad"](
        classifier=classifier,
        train_loader=loader,
        test_loader=loader,
        device=device,
        config=cfg,
    )

    x = torch.randn(2, 3, 32, 32, device=device)
    y = torch.tensor([1, 1], device=device)

    x_adv = trainer.step(x, y)

    assert torch.allclose(x_adv, x)


def test_input_pgd_loss_cli_defaults_to_ce_and_allows_margin():
    parser = build_arg_parser()

    default_cfg = config_from_args(parser.parse_args([]))
    margin_cfg = config_from_args(parser.parse_args(["--input-pgd-loss", "margin"]))
    ce_cfg = config_from_args(parser.parse_args(["--input-pgd-loss", "ce"]))
    normalized_cfg = config_from_args(
        parser.parse_args(["--input-pgd-geometry", "normalized_mse"])
    )
    align_cfg = config_from_args(
        parser.parse_args(
            [
                "--eval-transport-pgd-alignment",
                "--eval-transport-pgd-alignment-samples",
                "3",
            ]
        )
    )
    anchor_cfg = config_from_args(
        parser.parse_args(
            [
                "--npf-pgd-anchor-weight",
                "1000",
                "--npf-pgd-anchor-steps",
                "4",
                "--npf-pgd-anchor-restarts",
                "2",
                "--npf-pgd-anchor-loss",
                "ce",
            ]
        )
    )

    # ce is the legacy 57%-benchmark eval convention and every number of
    # record was measured with it; margin stays available for the stronger
    # local evaluator.
    assert default_cfg.input_pgd_loss == "ce"
    assert margin_cfg.input_pgd_loss == "margin"
    assert default_cfg.input_pgd_geometry == "pixel_l2_squared"
    assert default_cfg.eval_transport_pgd_alignment is False
    assert default_cfg.eval_transport_pgd_alignment_samples == 256
    assert default_cfg.npf_pgd_anchor_weight == 0.0
    assert default_cfg.npf_pgd_anchor_steps == 5
    assert default_cfg.npf_pgd_anchor_restarts == 1
    assert default_cfg.npf_pgd_anchor_loss == "margin"
    assert ce_cfg.input_pgd_loss == "ce"
    assert normalized_cfg.input_pgd_geometry == "normalized_mse"
    assert align_cfg.eval_transport_pgd_alignment is True
    assert align_cfg.eval_transport_pgd_alignment_samples == 3
    assert anchor_cfg.npf_pgd_anchor_weight == 1000
    assert anchor_cfg.npf_pgd_anchor_steps == 4
    assert anchor_cfg.npf_pgd_anchor_restarts == 2
    assert anchor_cfg.npf_pgd_anchor_loss == "ce"

    with pytest.raises(ValueError, match="normalized_mse"):
        config_from_args(
            parser.parse_args(["--input-pgd-geometry", "normalized_mse", "--inp-p", "inf"])
        )


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

    _, pixel_info = evaluate_under_input_pgd(
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
    assert pixel_info["geometry"] == "pixel_l2_squared"

    assert classifier.training
    assert all(not module.training for module in _batchnorm_modules(classifier))


def test_transport_pgd_alignment_reports_direction_stats():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    loader = _loader()

    classifier.train()
    for module in _batchnorm_modules(classifier):
        module.eval()

    def transport_fn(x):
        return x + 0.01

    info = evaluate_transport_pgd_alignment(
        classifier,
        transport_fn,
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

    assert info["samples"] == 2
    assert info["valid_cos_samples"] == 2
    assert -1.0 <= info["cos_mean"] <= 1.0
    assert "transport_projected_acc" in info
    assert "pgd_acc" in info
    assert classifier.training
    assert all(not module.training for module in _batchnorm_modules(classifier))


def test_input_pgd_normalized_mse_geometry_respects_budget():
    torch.manual_seed(123)
    device = torch.device("cpu")
    classifier = TinyClassifier().to(device)
    loader = _loader()

    _acc, info = evaluate_under_input_pgd(
        classifier,
        loader,
        device,
        p=2,
        eps=1e-4,
        steps=2,
        step_size=0.5,
        restarts=1,
        max_samples=2,
        loss="margin",
        geometry="normalized_mse",
    )

    assert info["geometry"] == "normalized_mse"
    assert info["max_normalized_mse"] <= 1e-4 + 1e-7


def test_input_pgd_rejects_unknown_loss():
    with pytest.raises(ValueError, match="input PGD loss"):
        _pgd_loss_per_sample(torch.zeros(1, 3), torch.tensor([0]), "bad")


def test_input_pgd_rejects_unknown_geometry():
    with pytest.raises(ValueError, match="input PGD geometry"):
        evaluate_under_input_pgd(
            TinyClassifier(),
            _loader(),
            torch.device("cpu"),
            p=2,
            eps=0.1,
            steps=1,
            step_size=0.1,
            restarts=1,
            max_samples=2,
            geometry="bad",
        )
