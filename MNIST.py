import os
import math
import csv
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
class CSVLogger:
    def __init__(self, path: str, fieldnames):
        self.path = path
        self.fieldnames = list(fieldnames)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, **kwargs):
        row = {name: kwargs.get(name) for name in self.fieldnames}
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


DEFAULT_LOG_PATH = os.path.join("MNIST", "training_logs.csv")
LOG_FIELDNAMES = [
    "algorithm",
    "phase",
    "epoch",
    "step",
    "loss_adv",
    "adv_loss",
    "cls_loss",
    "acc_clean",
    "acc_adv",
    "w2_proxy",
    "inner_grad_norm",
]

# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, reduction="mean")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean()


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def adversary_loss(logits: torch.Tensor, labels: torch.Tensor, use_margin_adv: bool) -> torch.Tensor:
    if use_margin_adv:
        logits_correct = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        margins = logits - logits_correct.unsqueeze(1)
        num_classes = logits.size(1)
        mask = F.one_hot(labels, num_classes=num_classes).bool()
        margins = torch.where(mask, torch.tensor(-float("inf"), device=logits.device), margins)
        return torch.logsumexp(margins, dim=1).mean()
    return cross_entropy_loss(logits, labels)

# ---------------------------------------------------------------------------
#  Layers and models
# ---------------------------------------------------------------------------

def icnn_principled_moments(fan_in: int):
    """Principled log-normal moments for positive weights (matches reference)."""
    denom_offset = 6.0 * (math.pi - 1.0)
    denom_slope = 3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0
    denom = denom_offset + (fan_in - 1.0) * denom_slope
    mu_w = math.sqrt((6.0 * math.pi) / (fan_in * denom))
    sigma_w2 = 1.0 / float(fan_in)
    mu_b = math.sqrt((3.0 * fan_in) / denom)
    mu_w_sq = mu_w * mu_w
    log_var_plus_mean_sq = math.log(sigma_w2 + mu_w_sq)
    log_mean_sq = math.log(mu_w_sq)
    tilde_mu = log_mean_sq - 0.5 * log_var_plus_mean_sq
    tilde_sigma2 = max(log_var_plus_mean_sq - log_mean_sq, 1e-12)
    tilde_sigma = math.sqrt(tilde_sigma2)
    return mu_w, sigma_w2, mu_b, tilde_mu, tilde_sigma


class NonNegativeDense(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True, init_mode: str = "principled"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.init_mode = init_mode.lower()
        self.weight_param = nn.Parameter(torch.empty(in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_mode == "principled":
            _mu_w, _sigma_w2, mu_b, tilde_mu, tilde_sigma = icnn_principled_moments(self.in_features)
            with torch.no_grad():
                if tilde_sigma == 0.0:
                    self.weight_param.fill_(tilde_mu)
                else:
                    self.weight_param.normal_(mean=tilde_mu, std=tilde_sigma)
                if self.bias is not None:
                    self.bias.fill_(mu_b)
        else:
            nn.init.xavier_uniform_(self.weight_param)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.exp(self.weight_param) if self.init_mode == "principled" else F.softplus(self.weight_param)
        y = x.matmul(weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class InputConvexPotential(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], activation: str = "relu", strong_convexity: float = 1.0, nonneg_init: str = "principled"):
        super().__init__()
        self.activation = activation.lower()
        self.strong_convexity = strong_convexity
        self.hidden_sizes = list(hidden_sizes)
        if len(self.hidden_sizes) == 0:
            raise ValueError("ICNN requires at least one hidden layer.")

        self.z_linears = nn.ModuleList()
        self.h_linears = nn.ModuleList()
        in_size = input_dim
        for i, width in enumerate(self.hidden_sizes):
            self.z_linears.append(nn.Linear(in_size, width))
            if i > 0:
                self.h_linears.append(NonNegativeDense(self.hidden_sizes[i - 1], width, use_bias=False, init_mode=nonneg_init))
        self.hidden_output = NonNegativeDense(self.hidden_sizes[-1], 1, init_mode=nonneg_init)
        self.input_skip = nn.Linear(in_size, 1)

    def init_as_identity(self):
        """Initialize so that nabla psi(x) = x (identity transport map).

        When z_linear weights are zero, h becomes a constant independent of x,
        so hidden_output(h) contributes no gradient w.r.t. x.  With input_skip
        weight also zeroed, the only gradient comes from the quadratic term
        0.5 * strong_convexity * ||x||^2, giving nabla psi(x) = x
        (requires strong_convexity = 1.0).
        """
        with torch.no_grad():
            for z_lin in self.z_linears:
                z_lin.weight.zero_()
                z_lin.bias.zero_()
            self.input_skip.weight.zero_()
            self.input_skip.bias.zero_()

    def act(self, u: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(u)
        if self.activation == "softplus":
            return F.softplus(20.0 * u) / 20.0
        raise ValueError(f"Unsupported activation {self.activation}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_flat = z.view(z.size(0), -1)
        h = None
        for i, z_lin in enumerate(self.z_linears):
            z_term = z_lin(z_flat)
            if i == 0:
                h = self.act(z_term)
            else:
                h_term = self.h_linears[i - 1](h)
                h = self.act(z_term + h_term)
        assert h is not None
        quadratic = 0.5 * self.strong_convexity * (z_flat ** 2).sum(dim=1, keepdim=True)
        out = quadratic + self.input_skip(z_flat) + self.hidden_output(h)
        return out.squeeze(1)


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---------------------------------------------------------------------------
#  ICNN gradient (functional)
# ---------------------------------------------------------------------------
try:
    from torch.func import functional_call
except ImportError:  # PyTorch < 2.0 fallback
    from torch.nn.utils._stateless import functional_call  # type: ignore


def icnn_gradient(model: InputConvexPotential, params: Dict[str, torch.Tensor], z_flat: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """Compute T(x) = nabla_x psi(x) via autograd.

    Uses torch.set_grad_enabled(True) so the inner gradient computation works
    correctly even when called inside a torch.no_grad() context (needed during
    Armijo line search).
    """
    z_flat_req = z_flat.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        out = functional_call(model, params, (z_flat_req,))
        phi_sum = out.sum()
        grad = torch.autograd.grad(phi_sum, z_flat_req, create_graph=create_graph)[0]
    return grad

# ---------------------------------------------------------------------------
#  BB + Armijo
# ---------------------------------------------------------------------------
@dataclass
class BBArmijoState:
    alpha_min: float
    alpha_max: float
    alpha_prev: float
    ls_c: float
    ls_shrink: float
    ls_max_steps: int
    prev_params_vec: Optional[torch.Tensor] = None
    prev_grad_vec: Optional[torch.Tensor] = None

    @classmethod
    def create(cls, alpha0: float = 5e-4, alpha_min: float = 1e-6, alpha_max: float = 1.0, ls_c: float = 0.1, ls_shrink: float = 0.5, ls_max_steps: int = 10) -> "BBArmijoState":
        alpha0 = float(max(alpha_min, min(alpha_max, alpha0)))
        return cls(alpha_min=max(alpha_min, 1e-12), alpha_max=max(alpha_max, alpha_min), alpha_prev=alpha0, ls_c=ls_c, ls_shrink=ls_shrink, ls_max_steps=int(max(ls_max_steps, 1)))

    def propose(self, params_vec: torch.Tensor, grad_vec: torch.Tensor) -> float:
        if self.prev_params_vec is None or self.prev_grad_vec is None or self.prev_params_vec.shape != params_vec.shape or self.prev_grad_vec.shape != grad_vec.shape:
            alpha = self.alpha_prev
        else:
            s = params_vec - self.prev_params_vec
            y = grad_vec - self.prev_grad_vec
            denom = torch.dot(s, y)
            num = torch.dot(s, s)
            cond = torch.isfinite(denom) & (torch.abs(denom) > 1e-12)
            alpha_bb = torch.where(cond, num / denom, torch.tensor(self.alpha_prev, device=denom.device))
            alpha = float(alpha_bb.clamp(self.alpha_min, self.alpha_max).item())
        if not math.isfinite(alpha):
            alpha = self.alpha_prev
        return max(self.alpha_min, min(self.alpha_max, float(alpha)))

    def update_history(self, params_vec: torch.Tensor, grad_vec: torch.Tensor, alpha: float) -> "BBArmijoState":
        alpha_clamped = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return BBArmijoState(alpha_min=self.alpha_min, alpha_max=self.alpha_max, alpha_prev=alpha_clamped, ls_c=self.ls_c, ls_shrink=self.ls_shrink, ls_max_steps=self.ls_max_steps, prev_params_vec=params_vec.detach().clone(), prev_grad_vec=grad_vec.detach().clone())

# ---------------------------------------------------------------------------
#  Parameter flattening helpers (for ICNN BB steps)
# ---------------------------------------------------------------------------

def flatten_params(module: nn.Module) -> Tuple[torch.Tensor, Tuple[Tuple[str, Tuple[int, ...], int], ...]]:
    names, shapes, tensors = [], [], []
    for name, param in module.named_parameters():
        names.append(name)
        shapes.append(param.shape)
        tensors.append(param.detach().reshape(-1))
    vec = torch.cat(tensors)
    meta = tuple((n, tuple(s), int(torch.prod(torch.tensor(s)).item())) for n, s in zip(names, shapes))
    return vec, meta


def unflatten_vector(vec: torch.Tensor, meta: Tuple[Tuple[str, Tuple[int, ...], int], ...]) -> Dict[str, torch.Tensor]:
    params = {}
    offset = 0
    for name, shape, size in meta:
        slice_view = vec[offset:offset + size].view(shape)
        params[name] = slice_view
        offset += size
    return params


def bb_armijo_step_params(vec: torch.Tensor, meta, f_params, bb_state: BBArmijoState):
    """Single BB+Armijo gradient-ascent step on ICNN parameter vector.

    f_params(vec, create_graph) -> scalar tensor.
    Uses torch.no_grad() during Armijo line search trials for efficiency.
    """
    vec_det = vec.detach().requires_grad_(True)
    f_val = f_params(vec_det, True)
    grad_vec = torch.autograd.grad(f_val, vec_det, create_graph=False)[0]
    alpha = bb_state.propose(vec_det.reshape(-1), grad_vec.reshape(-1))
    f_val_f = float(f_val.item())
    g_dot_g = float(torch.dot(grad_vec.reshape(-1), grad_vec.reshape(-1)).item())
    if g_dot_g == 0.0:
        return vec_det.detach(), bb_state, f_val_f
    grad_vec_det = grad_vec.detach()
    vec_det_val = vec_det.detach()
    alpha_k = alpha
    for _ in range(bb_state.ls_max_steps):
        v_trial = vec_det_val + alpha_k * grad_vec_det
        with torch.no_grad():
            f_trial = f_params(v_trial, False).item()
        if f_trial >= f_val_f + bb_state.ls_c * alpha_k * g_dot_g:
            break
        alpha_k *= bb_state.ls_shrink
    v_new = (vec_det_val + alpha_k * grad_vec_det).detach()
    v_new.requires_grad_(True)
    f_new = f_params(v_new, True)
    grad_vec_new = torch.autograd.grad(f_new, v_new, create_graph=False)[0]
    new_bb_state = bb_state.update_history(v_new.reshape(-1), grad_vec_new.reshape(-1), alpha_k)
    return v_new.detach(), new_bb_state, f_val_f

# ---------------------------------------------------------------------------
#  WRM inner maximization for adversary (Algorithm 1)
# ---------------------------------------------------------------------------

def wrm_ascent_x(
    x0: torch.Tensor,
    model: nn.Module,
    y: torch.Tensor,
    lambda_reg: float,
    num_steps: int,
    lr: float = 0.01,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> torch.Tensor:
    """WRM inner maximization (Sinha et al.).

    Gradient ascent on CE(z, y) - lambda ||z - x||^2 w.r.t. z.
    Update rule: z <- z + lr * (grad_z CE(z, y) - 2 * lambda * (z - x)).
    """
    if num_steps == 0:
        return x0.detach()
    x_orig = x0.detach()
    z = x_orig.clone()
    for _ in range(num_steps):
        z.requires_grad_(True)
        per_sample_ce = F.cross_entropy(model(z), y, reduction="none")
        grads = torch.autograd.grad(per_sample_ce.sum(), z, create_graph=False)[0]
        z = z.detach() + lr * (grads - 2.0 * lambda_reg * (z.detach() - x_orig))
        if clamp is not None:
            lo, hi = clamp
            z = z.clamp(lo, hi)
    return z.detach()

# ---------------------------------------------------------------------------
#  Config and states
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    num_epochs: int = 3
    lr_cls: float = 1e-3
    lambda_reg: float = 100
    epoch_clean: int = 0
    max_steps_algo1: Optional[int] = None
    max_steps_algo2: Optional[int] = None
    use_margin_adv_algo1: bool = False
    use_margin_adv_algo2: bool = False
    inner_steps_algo1: int = 5
    inner_steps_algo2: int = 5
    inner_lr_algo1: float = 1e-2
    bb_alpha0_icnn: float = 5e-4
    icnn_hidden_sizes: Sequence[int] = (64, 64, 64, 64)
    seed: int = 0


@dataclass(frozen=True)
class PGDEvalConfig:
    """PGD-L2 evaluation configuration with epsilon sweep."""
    epsilons: Sequence[float] = (1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
    num_steps: int = 40
    step_size: Optional[float] = None  # None => auto = 2*eps/steps
    restarts: int = 5


@dataclass
class TrainState:
    model: nn.Module
    opt: torch.optim.Optimizer


@dataclass
class ICNNState:
    model: InputConvexPotential
    params_vec: torch.Tensor
    meta: Tuple[Tuple[str, Tuple[int, ...], int], ...]
    bb_state: BBArmijoState

# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

def load_mnist():
    train_raw = datasets.MNIST(root="data", train=True, download=True)
    test_raw = datasets.MNIST(root="data", train=False, download=True)

    def to_tensor_dataset(ds):
        # Use the tensor version of the stored MNIST data to avoid numpy dtype inference issues
        x = ds.data.unsqueeze(1).float().div(255.0)
        y = ds.targets
        return TensorDataset(x, y)

    return to_tensor_dataset(train_raw), to_tensor_dataset(test_raw)

# ---------------------------------------------------------------------------
#  PGD
# ---------------------------------------------------------------------------

def project_l2(x: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    diff = x - x_orig
    flat = diff.view(diff.size(0), -1)
    norm = torch.norm(flat, dim=1, keepdim=True)
    factor = torch.clamp(eps / (norm + 1e-12), max=1.0)
    factor = factor.view(-1, *([1] * (x.dim() - 1)))
    return x_orig + diff * factor


def pgd_l2_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, num_steps: int, step_size: Optional[float] = None) -> torch.Tensor:
    if step_size is None:
        step_size = float(eps) / float(max(num_steps, 1))
    elif step_size <= 0:
        raise ValueError("pgd_l2_attack step_size must be positive.")
    model.eval()
    adv = x.detach()
    for _ in range(num_steps):
        adv_req = adv.clone().detach().requires_grad_(True)
        logits = model(adv_req)
        loss = cross_entropy_loss(logits, y)
        grad = torch.autograd.grad(loss, adv_req, create_graph=False)[0]
        grad_flat = grad.view(grad.size(0), -1)
        grad_norm = grad_flat.norm(dim=1, keepdim=True)
        scaled = grad / (grad_norm.view(-1, *([1] * (grad.dim() - 1))) + 1e-12)
        adv = adv_req + step_size * scaled
        adv = project_l2(adv, x, eps)
        adv = adv.clamp(0.0, 1.0).detach()
    return adv.detach()


def evaluate_pgd(state: TrainState, dataset, batch_size: int, eps: float = 0.3, step_size: Optional[float] = None, num_steps: int = 40, device: Optional[torch.device] = None) -> Dict[str, float]:
    device = device or next(state.model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_acc = total_n = 0
    total_l2 = total_linf = 0.0
    state.model.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if x.size(0) == 0:
            continue
        adv_x = pgd_l2_attack(state.model, x, y, eps, num_steps, step_size)
        with torch.no_grad():
            logits = state.model(adv_x)
            acc = accuracy(logits, y).item()
        n = x.size(0)
        total_acc += acc * n
        total_n += n
        diff = adv_x - x
        flat = diff.view(diff.size(0), -1)
        total_l2 += float(flat.norm(dim=1).mean().item()) * n
        total_linf += float(diff.abs().view(diff.size(0), -1).max(dim=1)[0].mean().item()) * n
    return {"acc": total_acc / total_n, "avg_l2": total_l2 / total_n, "avg_linf": total_linf / total_n}

# ---------------------------------------------------------------------------
#  Proper PGD-L2 attack (random restarts, correct step size & projection)
#  Follows the pattern from utils.evaluate_under_input_pgd
# ---------------------------------------------------------------------------

def _per_sample_l2_normalize(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize each sample in a batch to unit L2 norm."""
    B = t.size(0)
    flat = t.view(B, -1)
    norms = flat.norm(p=2, dim=1).clamp(min=eps)
    reshape = (B,) + (1,) * (t.dim() - 1)
    return t / norms.view(reshape)


def _random_start_l2_ball(x0: torch.Tensor, eps: float) -> torch.Tensor:
    """Sample a random starting point inside an L2 ball around x0."""
    z = torch.randn_like(x0)
    z = _per_sample_l2_normalize(z)
    batch = x0.size(0)
    r = torch.rand(batch, device=x0.device).view(batch, *([1] * (x0.dim() - 1)))
    delta = z * (r * eps)
    return (x0 + delta).clamp(0.0, 1.0)


def _project_onto_l2_ball(delta: torch.Tensor, eps: float) -> torch.Tensor:
    """Project perturbation onto an L2 ball of radius eps."""
    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    factors = (eps / norms).clamp(max=1.0)
    flat = flat * factors
    return flat.view_as(delta)


def pgd_l2_attack_restarts(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_steps: int,
    step_size: float,
    restarts: int,
) -> torch.Tensor:
    """PGD-L2 attack with random restarts and proper projection.

    - Step size default: 2*eps/steps (following utils.auto_pgd_step_size).
    - Random initialization inside L2 ball (following utils.random_start_input_ball_pix).
    - Per-restart best-loss tracking (following utils.evaluate_under_input_pgd).
    """
    model.eval()
    x0 = x.detach()
    best_delta = torch.zeros_like(x0)
    best_loss = torch.full((x0.size(0),), -1e9, device=x0.device)

    for _ in range(max(1, restarts)):
        x_adv = _random_start_l2_ball(x0, eps).detach().requires_grad_(True)
        for _ in range(num_steps):
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y, reduction="mean")
            grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
            with torch.no_grad():
                step = step_size * _per_sample_l2_normalize(grad)
                x_adv = x_adv + step
                delta = x_adv - x0
                delta = _project_onto_l2_ball(delta, eps)
                x_adv = (x0 + delta).clamp(0.0, 1.0)
                x_adv.requires_grad_(True)

        with torch.no_grad():
            logits = model(x_adv)
            losses = F.cross_entropy(logits, y, reduction="none")
            delta = (x_adv - x0).detach()
            mask = losses > best_loss
            if mask.any():
                best_loss[mask] = losses[mask]
                best_delta[mask] = delta[mask]

    return (x0 + best_delta).clamp(0.0, 1.0).detach()


def evaluate_pgd_l2_sweep(
    state: TrainState,
    dataset,
    pgd_cfg: PGDEvalConfig,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model under PGD-L2 attack for multiple epsilon values."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    state.model.eval()
    results: Dict[str, Any] = {}

    for eps in pgd_cfg.epsilons:
        step_size = (
            pgd_cfg.step_size
            if pgd_cfg.step_size is not None and pgd_cfg.step_size > 0
            else 2.0 * eps / max(pgd_cfg.num_steps, 1)
        )
        total_correct = 0
        total_n = 0
        total_l2 = 0.0
        total_linf = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if xb.size(0) == 0:
                continue
            adv_x = pgd_l2_attack_restarts(
                state.model, xb, yb, eps,
                pgd_cfg.num_steps, step_size, pgd_cfg.restarts,
            )
            with torch.no_grad():
                logits = state.model(adv_x)
                total_correct += (logits.argmax(dim=1) == yb).sum().item()
                total_n += xb.size(0)
                delta = adv_x - xb
                total_l2 += delta.view(delta.size(0), -1).norm(p=2, dim=1).mean().item()
                total_linf += delta.abs().view(delta.size(0), -1).max(dim=1)[0].mean().item()
                n_batches += 1

        acc = total_correct / max(1, total_n)
        avg_l2 = total_l2 / max(1, n_batches)
        avg_linf = total_linf / max(1, n_batches)

        eps_key = f"eps_{eps:.1f}"
        results[eps_key] = {
            "epsilon": eps,
            "step_size": step_size,
            "num_steps": pgd_cfg.num_steps,
            "restarts": pgd_cfg.restarts,
            "acc_pct": round(acc * 100, 2),
            "avg_l2": round(avg_l2, 4),
            "avg_linf": round(avg_linf, 4),
            "samples": total_n,
        }
        print(f"    eps={eps:.1f}: acc={acc*100:.2f}% avg_L2={avg_l2:.4f} avg_Linf={avg_linf:.4f}")

    return results

# ---------------------------------------------------------------------------
#  Training steps
# ---------------------------------------------------------------------------

def train_step_clean(state: TrainState, batch, device: torch.device) -> Tuple[TrainState, Dict[str, torch.Tensor]]:
    """Standard clean training step (no adversarial perturbation)."""
    model = state.model
    opt = state.opt
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    if x.size(0) == 0:
        zero = torch.tensor(0.0, device=device)
        return state, {"loss": zero, "acc_clean": zero}

    model.train()
    logits = model(x)
    loss = cross_entropy_loss(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        acc = accuracy(logits, y)
    return state, {"loss": loss.detach(), "acc_clean": acc}


def train_step_algo1(state: TrainState, batch, cfg: TrainConfig, device: torch.device) -> Tuple[TrainState, Dict[str, torch.Tensor]]:
    model = state.model
    opt = state.opt
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    if x.size(0) == 0:
        zero = torch.tensor(0.0, device=device)
        metrics = {"loss_adv": zero, "acc_clean": zero, "acc_adv": zero, "w2_proxy": zero}
        return state, metrics

    # Inner maximization: freeze classifier, WRM gradient ascent
    set_requires_grad(model, False)
    model.eval()

    adv_x = wrm_ascent_x(
        x, model, y, cfg.lambda_reg,
        cfg.inner_steps_algo1,
        lr=cfg.inner_lr_algo1,
        clamp=(0.0, 1.0),
    )

    # Outer minimization: unfreeze classifier weights
    set_requires_grad(model, True)
    model.train()

    logits_adv = model(adv_x)
    loss = cross_entropy_loss(logits_adv, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    # FIX Bug1: recompute BOTH logits from the SAME (post-update) model
    with torch.no_grad():
        logits_clean = model(x)
        logits_adv_post = model(adv_x)
        acc_clean = accuracy(logits_clean, y)
        acc_adv = accuracy(logits_adv_post, y)
        w2_proxy = ((adv_x - x) ** 2).sum(dim=(1, 2, 3)).mean()
    metrics = {"loss_adv": loss.detach(), "acc_clean": acc_clean, "acc_adv": acc_adv, "w2_proxy": w2_proxy}
    return state, metrics


def train_step_algo2(state: TrainState, icnn_state: ICNNState, batch, cfg: TrainConfig, device: torch.device) -> Tuple[TrainState, ICNNState, Dict[str, torch.Tensor]]:
    model = state.model
    opt = state.opt
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    if x.size(0) == 0:
        zero = torch.tensor(0.0, device=device)
        metrics = {"adv_loss": zero, "cls_loss": zero, "acc_clean": zero, "acc_adv": zero, "w2_proxy": zero, "inner_grad_norm": zero}
        return state, icnn_state, metrics

    x_flat = x.view(x.size(0), -1)
    icnn_model = icnn_state.model
    params_vec = icnn_state.params_vec.to(device)
    bb_state = icnn_state.bb_state
    meta = icnn_state.meta

    # FIX Bug3: ensure ICNN parameters have requires_grad=True before inner loop
    set_requires_grad(icnn_model, True)

    # Inner maximization over ICNN params: freeze classifier weights
    set_requires_grad(model, False)
    model.eval()

    def adv_obj_params(vec: torch.Tensor, create_graph: bool) -> torch.Tensor:
        params_dict = unflatten_vector(vec, meta)
        adv_flat = icnn_gradient(icnn_model, params_dict, x_flat, create_graph=create_graph)
        # FIX Bug4: clamp to [0,1] inside inner objective (match outer step)
        adv_x = adv_flat.view_as(x).clamp(0.0, 1.0)
        logits = model(adv_x)
        adv_loss = adversary_loss(logits, y, cfg.use_margin_adv_algo2)
        w2 = ((adv_x - x) ** 2).sum(dim=(1, 2, 3)).mean()
        return adv_loss - cfg.lambda_reg * w2

    # FIX Bug2: use Adam on ICNN parameters for robust inner optimization
    # BB+Armijo with tiny alpha0 on a near-flat landscape (identity init)
    # yields negligible updates.  Adam with momentum handles the small-gradient
    # regime far more reliably.
    inner_param = params_vec.detach().clone().requires_grad_(True)
    inner_opt = torch.optim.AdamW([inner_param], lr=cfg.bb_alpha0_icnn, weight_decay=1e-5)
    adv_loss_val = torch.tensor(0.0, device=device)
    inner_grad_norm = 0.0

    for _ in range(cfg.inner_steps_algo2):
        inner_opt.zero_grad()
        obj = adv_obj_params(inner_param, True)
        # Gradient ascent: negate for Adam (which minimizes)
        neg_obj = -obj
        neg_obj.backward()
        # FIX Bug5: diagnostic — track gradient norm
        if inner_param.grad is not None:
            inner_grad_norm = float(inner_param.grad.norm().item())
        inner_opt.step()
        adv_loss_val = obj.detach()

    params_vec = inner_param.detach()
    # BB state is kept for continuity but no longer drives the step size
    icnn_state = ICNNState(model=icnn_model, params_vec=params_vec, meta=meta, bb_state=bb_state)

    # Outer minimization: update classifier only
    set_requires_grad(model, True)
    model.train()
    # FIX Bug3: don't mutate icnn_model requires_grad during outer step —
    # we use functional_call with params_dict, so module params are irrelevant
    # (no set_requires_grad(icnn_model, False) here)

    params_dict_final = unflatten_vector(icnn_state.params_vec.to(device), meta)
    adv_flat = icnn_gradient(icnn_model, params_dict_final, x_flat).detach()
    adv_x = adv_flat.view_as(x).clamp(0.0, 1.0)

    logits_adv = model(adv_x)
    cls_loss = cross_entropy_loss(logits_adv, y)
    opt.zero_grad()
    cls_loss.backward()
    opt.step()

    # FIX Bug1: recompute BOTH logits from the SAME (post-update) model
    with torch.no_grad():
        logits_clean = model(x)
        logits_adv_post = model(adv_x)
        acc_clean = accuracy(logits_clean, y)
        acc_adv = accuracy(logits_adv_post, y)
        w2_proxy = ((adv_x - x) ** 2).sum(dim=(1, 2, 3)).mean()
    metrics = {
        "adv_loss": adv_loss_val.detach(),
        "cls_loss": cls_loss.detach(),
        "acc_clean": acc_clean,
        "acc_adv": acc_adv,
        "w2_proxy": w2_proxy,
        "inner_grad_norm": torch.tensor(inner_grad_norm, device=device),  # Bug5: diagnostic
    }
    return state, icnn_state, metrics

# ---------------------------------------------------------------------------
#  Training loops
# ---------------------------------------------------------------------------

def create_classifier_state(cfg: TrainConfig, device: torch.device) -> TrainState:
    model = LeNet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr_cls, momentum=0.9, nesterov=True, weight_decay=1e-4)
    return TrainState(model=model, opt=opt)


def create_icnn_state(cfg: TrainConfig, input_dim: int, device: torch.device) -> ICNNState:
    icnn_model = InputConvexPotential(input_dim=input_dim, hidden_sizes=cfg.icnn_hidden_sizes, activation="softplus", strong_convexity=1.0, nonneg_init="principled").to(device)
    icnn_model.init_as_identity()
    params_vec, meta = flatten_params(icnn_model)
    bb_state = BBArmijoState.create(alpha0=cfg.bb_alpha0_icnn)
    return ICNNState(model=icnn_model, params_vec=params_vec.to(device), meta=meta, bb_state=bb_state)


def evaluate_clean(state: TrainState, dataset, batch_size: int, device: torch.device) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = state.model
    model.eval()
    total_loss = total_acc = total_n = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if x.size(0) == 0:
                continue
            logits = model(x)
            loss = cross_entropy_loss(logits, y).item()
            acc = accuracy(logits, y).item()
            n = x.size(0)
            total_loss += loss * n
            total_acc += acc * n
            total_n += n
    return {"loss": total_loss / total_n, "acc": total_acc / total_n}


def train_algorithm_1(cfg: TrainConfig, device: torch.device) -> Tuple[TrainState, Dict[str, Any]]:
    torch.manual_seed(cfg.seed)
    train_ds, test_ds = load_mnist()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    state = create_classifier_state(cfg, device)
    logger = CSVLogger(DEFAULT_LOG_PATH, LOG_FIELDNAMES)
    training_logs = []
    for epoch in range(cfg.num_epochs):
        is_clean = epoch < cfg.epoch_clean
        num_steps = 0
        if is_clean:
            epoch_loss = 0.0
            epoch_acc_clean = 0.0
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_algo1 is not None and step >= cfg.max_steps_algo1:
                    break
                state, metrics = train_step_clean(state, (x, y), device)
                epoch_loss += float(metrics['loss'])
                epoch_acc_clean += float(metrics['acc_clean'])
                num_steps += 1
            if num_steps > 0:
                avg_loss = epoch_loss / num_steps
                avg_acc_clean = epoch_acc_clean / num_steps
                print(f"[Algo1] Epoch {epoch} (clean) loss={avg_loss:.4f} acc_clean={avg_acc_clean:.4f}")
                logger.log(algorithm="algo1", phase="train_clean", epoch=epoch, step=num_steps, loss_adv=avg_loss, adv_loss=None, cls_loss=None, acc_clean=avg_acc_clean, acc_adv=None, w2_proxy=None)
                training_logs.append({"epoch": epoch, "phase": "clean", "steps": num_steps, "loss": avg_loss, "acc_clean": avg_acc_clean})
        else:
            epoch_loss_adv = 0.0
            epoch_acc_clean = 0.0
            epoch_acc_adv = 0.0
            epoch_w2_proxy = 0.0
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_algo1 is not None and step >= cfg.max_steps_algo1:
                    break
                state, metrics = train_step_algo1(state, (x, y), cfg, device)
                epoch_loss_adv += float(metrics['loss_adv'])
                epoch_acc_clean += float(metrics['acc_clean'])
                epoch_acc_adv += float(metrics['acc_adv'])
                epoch_w2_proxy += float(metrics['w2_proxy'])
                num_steps += 1
            if num_steps > 0:
                avg_loss_adv = epoch_loss_adv / num_steps
                avg_acc_clean = epoch_acc_clean / num_steps
                avg_acc_adv = epoch_acc_adv / num_steps
                avg_w2_proxy = epoch_w2_proxy / num_steps
                print(f"[Algo1] Epoch {epoch} (adv) loss_adv={avg_loss_adv:.4f} acc_clean={avg_acc_clean:.4f} acc_adv={avg_acc_adv:.4f} W2≈{avg_w2_proxy:.4f}")
                logger.log(algorithm="algo1", phase="train_adv", epoch=epoch, step=num_steps, loss_adv=avg_loss_adv, adv_loss=None, cls_loss=None, acc_clean=avg_acc_clean, acc_adv=avg_acc_adv, w2_proxy=avg_w2_proxy)
                training_logs.append({"epoch": epoch, "phase": "adv", "steps": num_steps, "loss_adv": avg_loss_adv, "acc_clean": avg_acc_clean, "acc_adv": avg_acc_adv, "w2_proxy": avg_w2_proxy})
    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print("[Algo1] Test:", test_metrics)
    logger.log(algorithm="algo1", phase="test", epoch=cfg.num_epochs, step=None, loss_adv=float(test_metrics["loss"]), adv_loss=None, cls_loss=None, acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None)
    results = {"algorithm": "algo1_wrm", "hyperparameters": asdict(cfg), "training_logs": training_logs, "test_metrics": test_metrics}
    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", "algo1_wrm_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return state, {"test": test_metrics}


def train_algorithm_2(cfg: TrainConfig, device: torch.device) -> Tuple[TrainState, ICNNState, Dict[str, Any]]:
    torch.manual_seed(cfg.seed)
    train_ds, test_ds = load_mnist()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    state = create_classifier_state(cfg, device)
    logger = CSVLogger(DEFAULT_LOG_PATH, LOG_FIELDNAMES)
    training_logs = []
    input_dim = 28 * 28 * 1
    icnn_state = create_icnn_state(cfg, input_dim, device)
    for epoch in range(cfg.num_epochs):
        is_clean = epoch < cfg.epoch_clean
        num_steps = 0
        if is_clean:
            epoch_loss = 0.0
            epoch_acc_clean = 0.0
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_algo2 is not None and step >= cfg.max_steps_algo2:
                    break
                state, metrics = train_step_clean(state, (x, y), device)
                epoch_loss += float(metrics['loss'])
                epoch_acc_clean += float(metrics['acc_clean'])
                num_steps += 1
            if num_steps > 0:
                avg_loss = epoch_loss / num_steps
                avg_acc_clean = epoch_acc_clean / num_steps
                print(f"[Algo2] Epoch {epoch} (clean) loss={avg_loss:.4f} acc_clean={avg_acc_clean:.4f}")
                logger.log(algorithm="algo2", phase="train_clean", epoch=epoch, step=num_steps, loss_adv=None, adv_loss=None, cls_loss=avg_loss, acc_clean=avg_acc_clean, acc_adv=None, w2_proxy=None)
                training_logs.append({"epoch": epoch, "phase": "clean", "steps": num_steps, "loss": avg_loss, "acc_clean": avg_acc_clean})
        else:
            epoch_adv_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_acc_clean = 0.0
            epoch_acc_adv = 0.0
            epoch_w2_proxy = 0.0
            epoch_inner_grad_norm = 0.0
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_algo2 is not None and step >= cfg.max_steps_algo2:
                    break
                state, icnn_state, metrics = train_step_algo2(state, icnn_state, (x, y), cfg, device)
                epoch_adv_loss += float(metrics['adv_loss'])
                epoch_cls_loss += float(metrics['cls_loss'])
                epoch_acc_clean += float(metrics['acc_clean'])
                epoch_acc_adv += float(metrics['acc_adv'])
                epoch_w2_proxy += float(metrics['w2_proxy'])
                epoch_inner_grad_norm += float(metrics['inner_grad_norm'])
                num_steps += 1
            if num_steps > 0:
                avg_adv_loss = epoch_adv_loss / num_steps
                avg_cls_loss = epoch_cls_loss / num_steps
                avg_acc_clean = epoch_acc_clean / num_steps
                avg_acc_adv = epoch_acc_adv / num_steps
                avg_w2_proxy = epoch_w2_proxy / num_steps
                avg_inner_grad_norm = epoch_inner_grad_norm / num_steps
                print(f"[Algo2] Epoch {epoch} (adv) adv_loss={avg_adv_loss:.4f} cls_loss={avg_cls_loss:.4f} acc_clean={avg_acc_clean:.4f} acc_adv={avg_acc_adv:.4f} W2≈{avg_w2_proxy:.4f} |∇θ|={avg_inner_grad_norm:.6f}")
                logger.log(algorithm="algo2", phase="train_adv", epoch=epoch, step=num_steps, loss_adv=None, adv_loss=avg_adv_loss, cls_loss=avg_cls_loss, acc_clean=avg_acc_clean, acc_adv=avg_acc_adv, w2_proxy=avg_w2_proxy, inner_grad_norm=avg_inner_grad_norm)
                training_logs.append({"epoch": epoch, "phase": "adv", "steps": num_steps, "adv_loss": avg_adv_loss, "cls_loss": avg_cls_loss, "acc_clean": avg_acc_clean, "acc_adv": avg_acc_adv, "w2_proxy": avg_w2_proxy, "inner_grad_norm": avg_inner_grad_norm})
    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print("[Algo2] Test:", test_metrics)
    logger.log(algorithm="algo2", phase="test", epoch=cfg.num_epochs, step=None, loss_adv=float(test_metrics["loss"]), adv_loss=None, cls_loss=float(test_metrics["loss"]), acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None)
    results = {"algorithm": "algo2_icnn", "hyperparameters": asdict(cfg), "training_logs": training_logs, "test_metrics": test_metrics}
    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", "algo2_icnn_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return state, icnn_state, {"test": test_metrics}

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(
        batch_size=512,
        num_epochs=30,
        lr_cls=1e-2,
        lambda_reg=0.05,
        epoch_clean=10,
        max_steps_algo1=None,
        max_steps_algo2=None,
        use_margin_adv_algo1=False,
        use_margin_adv_algo2=False,
        inner_steps_algo1=20,
        inner_steps_algo2=20,    
        inner_lr_algo1=1e-2,
        bb_alpha0_icnn=1e-2,    
        icnn_hidden_sizes=(128, 64, 64),
        seed=0,
    )
    print("Training Algorithm 2 (ICNN transport with BB+Armijo)...")
    state_algo2, icnn_state_algo2, logs_algo2 = train_algorithm_2(cfg, device)
    
    print("Training Algorithm 1 (WRM adversarial training)...")
    state_algo1, logs_algo1 = train_algorithm_1(cfg, device)



    _, test_ds = load_mnist()
    pgd_kwargs = dict(batch_size=cfg.batch_size, eps=2.5, num_steps=20, device=device)

    pgd_algo1 = evaluate_pgd(state_algo1, test_ds, **pgd_kwargs)
    print(f"[Algo1] PGD acc={pgd_algo1['acc']*100:.2f}% L2={pgd_algo1['avg_l2']:.4f} Linf={pgd_algo1['avg_linf']:.4f}")

    pgd_algo2 = evaluate_pgd(state_algo2, test_ds, **pgd_kwargs)
    print(f"[Algo2] PGD acc={pgd_algo2['acc']*100:.2f}% L2={pgd_algo2['avg_l2']:.4f} Linf={pgd_algo2['avg_linf']:.4f}")

    pgd_results = {
        "hyperparameters": asdict(cfg),
        "pgd_config": {"eps": pgd_kwargs["eps"], "num_steps": pgd_kwargs["num_steps"]},
        "algo1_wrm_pgd": pgd_algo1,
        "algo2_icnn_pgd": pgd_algo2,
    }
    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", "pgd_evaluation_results.json"), "w") as f:
        json.dump(pgd_results, f, indent=2)

    # ------------------------------------------------------------------
    #  PGD-L2 sweep evaluation (eps 1.5 to 2.0, proper restarts & projection)
    # ------------------------------------------------------------------
    pgd_eval_cfg = PGDEvalConfig(
        epsilons=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0),
        num_steps=40,
        step_size=None,  # auto = 2*eps/steps
        restarts=5,
    )

    print("\n[Algo1] PGD-L2 sweep evaluation (eps 1.5 to 2.0, restarts=5)...")
    pgd_sweep_algo1 = evaluate_pgd_l2_sweep(state_algo1, test_ds, pgd_eval_cfg, cfg.batch_size, device)

    print("\n[Algo2] PGD-L2 sweep evaluation (eps 1.5 to 2.0, restarts=5)...")
    pgd_sweep_algo2 = evaluate_pgd_l2_sweep(state_algo2, test_ds, pgd_eval_cfg, cfg.batch_size, device)

    # Update algo1 JSON with PGD sweep results
    algo1_json_path = os.path.join("MNIST", "algo1_wrm_results.json")
    with open(algo1_json_path, "r") as f:
        algo1_data = json.load(f)
    algo1_data["pgd_l2_sweep"] = pgd_sweep_algo1
    algo1_data["pgd_eval_config"] = asdict(pgd_eval_cfg)
    with open(algo1_json_path, "w") as f:
        json.dump(algo1_data, f, indent=2)

    # Update algo2 JSON with PGD sweep results
    algo2_json_path = os.path.join("MNIST", "algo2_icnn_results.json")
    with open(algo2_json_path, "r") as f:
        algo2_data = json.load(f)
    algo2_data["pgd_l2_sweep"] = pgd_sweep_algo2
    algo2_data["pgd_eval_config"] = asdict(pgd_eval_cfg)
    with open(algo2_json_path, "w") as f:
        json.dump(algo2_data, f, indent=2)

    # Update PGD evaluation JSON with sweep results
    pgd_results["pgd_l2_sweep_config"] = asdict(pgd_eval_cfg)
    pgd_results["algo1_wrm_pgd_sweep"] = pgd_sweep_algo1
    pgd_results["algo2_icnn_pgd_sweep"] = pgd_sweep_algo2
    with open(os.path.join("MNIST", "pgd_evaluation_results.json"), "w") as f:
        json.dump(pgd_results, f, indent=2)