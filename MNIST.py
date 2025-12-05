import os
import math
import csv
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

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
]

# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, reduction="none")


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
    return cross_entropy_loss(logits, labels).mean()

# ---------------------------------------------------------------------------
#  Layers and models
# ---------------------------------------------------------------------------
class NonNegativeDense(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True, init_mode: str = "principled"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.init_mode = init_mode.lower()
        self.weight_param = nn.Parameter(torch.zeros(in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_mode == "principled":
            fan_in = self.in_features
            denom_offset = 6.0 * (math.pi - 1.0)
            denom_slope = 3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0
            denom = denom_offset + (fan_in - 1.0) * denom_slope
            mu_w = math.sqrt((6.0 * math.pi) / (fan_in * denom))
            mu_b = math.sqrt((3.0 * fan_in) / denom)
            with torch.no_grad():
                self.weight_param.fill_(math.log(max(mu_w, 1e-8)))
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
                self.h_linears.append(NonNegativeDense(self.hidden_sizes[i - 1], width, init_mode=nonneg_init))
        self.hidden_output = NonNegativeDense(self.hidden_sizes[-1], 1, init_mode=nonneg_init)
        self.input_skip = nn.Linear(in_size, 1)

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
    z_flat_req = z_flat.detach().requires_grad_(True)
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
    def create(cls, alpha0: float = 1e-1, alpha_min: float = 1e-6, alpha_max: float = 10.0, ls_c: float = 1e-4, ls_shrink: float = 0.5, ls_max_steps: int = 10) -> "BBArmijoState":
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
            if torch.isfinite(denom) and torch.abs(denom) > 1e-12:
                alpha_bb = num / denom
            else:
                alpha_bb = torch.tensor(self.alpha_prev, device=params_vec.device)
            alpha = float(torch.clamp(alpha_bb, self.alpha_min, self.alpha_max).item())
        if not math.isfinite(alpha):
            alpha = self.alpha_prev
        return max(self.alpha_min, min(self.alpha_max, float(alpha)))

    def update_history(self, params_vec: torch.Tensor, grad_vec: torch.Tensor, alpha: float) -> "BBArmijoState":
        alpha_clamped = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return BBArmijoState(alpha_min=self.alpha_min, alpha_max=self.alpha_max, alpha_prev=alpha_clamped, ls_c=self.ls_c, ls_shrink=self.ls_shrink, ls_max_steps=self.ls_max_steps, prev_params_vec=params_vec.detach(), prev_grad_vec=grad_vec.detach())


def bb_armijo_ascent_x(x0: torch.Tensor, f, num_steps: int, bb_state: Optional[BBArmijoState] = None) -> torch.Tensor:
    if x0.numel() == 0 or num_steps == 0:
        return x0
    if bb_state is None:
        bb_state = BBArmijoState.create()
    x = x0.detach()
    state = bb_state
    for _ in range(num_steps):
        x_req = x.detach().requires_grad_(True)
        fx = f(x_req)
        g = torch.autograd.grad(fx, x_req, create_graph=False)[0]
        g_vec = g.reshape(g.size(0), -1)
        g_flat = g_vec.reshape(-1)
        x_vec = x_req.reshape(-1)
        alpha = state.propose(x_vec, g_flat)
        g_dot_g = float(torch.dot(g_flat, g_flat).item())
        if g_dot_g == 0.0:
            break
        alpha_k = alpha
        for _ in range(state.ls_max_steps):
            x_trial = x_req + alpha_k * g
            f_trial = f(x_trial).item()
            if f_trial >= fx.item() + state.ls_c * alpha_k * g_dot_g:
                break
            alpha_k *= state.ls_shrink
        x_new = (x_req + alpha_k * g).detach()
        g_new = torch.autograd.grad(f(x_new.requires_grad_(True)), x_new, create_graph=False)[0]
        state = state.update_history(x_new.reshape(-1), g_new.reshape(-1), alpha_k)
        x = x_new
    return x.detach()

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
    vec_det = vec.detach()
    vec_det.requires_grad_(True)
    f_val = f_params(vec_det)
    grad_vec = torch.autograd.grad(f_val, vec_det, create_graph=False)[0]
    alpha = bb_state.propose(vec_det.reshape(-1), grad_vec.reshape(-1))
    f_val_f = float(f_val.item())
    g_dot_g = float(torch.dot(grad_vec.reshape(-1), grad_vec.reshape(-1)).item())
    if g_dot_g == 0.0:
        return vec_det.detach(), bb_state, f_val_f
    alpha_k = alpha
    for _ in range(bb_state.ls_max_steps):
        v_trial = vec_det + alpha_k * grad_vec
        f_trial = f_params(v_trial).item()
        if f_trial >= f_val_f + bb_state.ls_c * alpha_k * g_dot_g:
            break
        alpha_k *= bb_state.ls_shrink
    v_new = (vec_det + alpha_k * grad_vec).detach()
    grad_vec_new = torch.autograd.grad(f_params(v_new.requires_grad_(True)), v_new, create_graph=False)[0]
    new_bb_state = bb_state.update_history(v_new.reshape(-1), grad_vec_new.reshape(-1), alpha_k)
    return v_new.detach(), new_bb_state, f_val_f

# ---------------------------------------------------------------------------
#  Config and states
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    num_epochs: int = 3
    lr_cls: float = 1e-3
    lambda_reg: float = 0.5
    log_every: int = 100
    max_steps_algo1: Optional[int] = None
    max_steps_algo2: Optional[int] = None
    use_margin_adv_algo1: bool = False
    use_margin_adv_algo2: bool = False
    inner_steps_algo1: int = 5
    inner_steps_algo2: int = 5
    bb_alpha0_x: float = 1e-1
    bb_alpha0_icnn: float = 1e-2
    icnn_hidden_sizes: Sequence[int] = (64, 64, 64, 64)
    seed: int = 0


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
        loss = cross_entropy_loss(logits, y).mean()
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
#  Training steps
# ---------------------------------------------------------------------------

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

    # Inner maximization: freeze classifier weights
    set_requires_grad(model, False)
    model.eval()

    def adv_obj(z: torch.Tensor) -> torch.Tensor:
        logits = model(z)
        adv_loss = adversary_loss(logits, y, cfg.use_margin_adv_algo1)
        sq_dist = ((z - x) ** 2).sum(dim=(1, 2, 3)).mean()
        return adv_loss - cfg.lambda_reg * sq_dist

    adv_x = bb_armijo_ascent_x(x, adv_obj, cfg.inner_steps_algo1, BBArmijoState.create(alpha0=cfg.bb_alpha0_x)).detach()
    adv_x_stop = adv_x.detach()

    # Outer minimization: unfreeze classifier weights
    set_requires_grad(model, True)
    model.train()

    logits_adv = model(adv_x_stop)
    loss = cross_entropy_loss(logits_adv, y).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        logits_clean = model(x)
        acc_clean = accuracy(logits_clean, y)
        acc_adv = accuracy(logits_adv, y)
        w2_proxy = ((adv_x_stop - x) ** 2).sum(dim=(1, 2, 3)).mean()
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
        metrics = {"adv_loss": zero, "cls_loss": zero, "acc_clean": zero, "acc_adv": zero, "w2_proxy": zero}
        return state, icnn_state, metrics

    x_flat = x.view(x.size(0), -1)
    icnn_model = icnn_state.model
    params_vec = icnn_state.params_vec.to(device)
    bb_state = icnn_state.bb_state
    meta = icnn_state.meta

    # Inner maximization over ICNN params: freeze classifier weights
    set_requires_grad(model, False)
    model.eval()

    def adv_obj_params(vec: torch.Tensor) -> torch.Tensor:
        params_dict = unflatten_vector(vec, meta)
        adv_flat = icnn_gradient(icnn_model, params_dict, x_flat, create_graph=True)
        adv_x = adv_flat.view_as(x)
        logits = model(adv_x)
        adv_loss = adversary_loss(logits, y, cfg.use_margin_adv_algo2)
        w2 = ((adv_flat - x_flat) ** 2).sum(dim=1).mean()
        return adv_loss - cfg.lambda_reg * w2

    adv_loss_val = torch.tensor(0.0, device=device)
    for _ in range(cfg.inner_steps_algo2):
        params_vec, bb_state, adv_loss_scalar = bb_armijo_step_params(params_vec, meta, adv_obj_params, bb_state)
        adv_loss_val = torch.tensor(adv_loss_scalar, device=device)

    icnn_state = ICNNState(model=icnn_model, params_vec=params_vec.detach(), meta=meta, bb_state=bb_state)

    # Outer minimization: update classifier only
    set_requires_grad(model, True)
    model.train()
    set_requires_grad(icnn_model, False)

    params_dict_final = unflatten_vector(icnn_state.params_vec.to(device), meta)
    adv_flat = icnn_gradient(icnn_model, params_dict_final, x_flat).detach()
    adv_x = adv_flat.view_as(x)

    logits_adv = model(adv_x)
    cls_loss = cross_entropy_loss(logits_adv, y).mean()
    opt.zero_grad()
    cls_loss.backward()
    opt.step()

    with torch.no_grad():
        logits_clean = model(x)
        acc_clean = accuracy(logits_clean, y)
        acc_adv = accuracy(logits_adv, y)
        w2_proxy = ((adv_flat - x_flat) ** 2).sum(dim=1).mean()
    metrics = {"adv_loss": adv_loss_val.detach(), "cls_loss": cls_loss.detach(), "acc_clean": acc_clean, "acc_adv": acc_adv, "w2_proxy": w2_proxy}
    return state, icnn_state, metrics

# ---------------------------------------------------------------------------
#  Training loops
# ---------------------------------------------------------------------------

def create_classifier_state(cfg: TrainConfig, device: torch.device) -> TrainState:
    model = LeNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_cls)
    return TrainState(model=model, opt=opt)


def create_icnn_state(cfg: TrainConfig, input_dim: int, device: torch.device) -> ICNNState:
    icnn_model = InputConvexPotential(input_dim=input_dim, hidden_sizes=cfg.icnn_hidden_sizes, activation="softplus", strong_convexity=1.0, nonneg_init="principled").to(device)
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
            loss = cross_entropy_loss(logits, y).mean().item()
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
    for epoch in range(cfg.num_epochs):
        for step, (x, y) in enumerate(train_loader):
            if cfg.max_steps_algo1 is not None and step >= cfg.max_steps_algo1:
                break
            state, metrics = train_step_algo1(state, (x, y), cfg, device)
            if cfg.log_every and step % cfg.log_every == 0:
                print(f"[Algo1] Epoch {epoch} step {step} loss_adv={float(metrics['loss_adv']):.4f} acc_clean={float(metrics['acc_clean']):.4f} acc_adv={float(metrics['acc_adv']):.4f} W2≈{float(metrics['w2_proxy']):.4f}")
                logger.log(algorithm="algo1", phase="train", epoch=epoch, step=step, loss_adv=float(metrics['loss_adv']), adv_loss=None, cls_loss=None, acc_clean=float(metrics['acc_clean']), acc_adv=float(metrics['acc_adv']), w2_proxy=float(metrics['w2_proxy']))
    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print("[Algo1] Test:", test_metrics)
    logger.log(algorithm="algo1", phase="test", epoch=cfg.num_epochs, step=None, loss_adv=float(test_metrics["loss"]), adv_loss=None, cls_loss=None, acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None)
    return state, {"test": test_metrics}


def train_algorithm_2(cfg: TrainConfig, device: torch.device) -> Tuple[TrainState, ICNNState, Dict[str, Any]]:
    torch.manual_seed(cfg.seed)
    train_ds, test_ds = load_mnist()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    state = create_classifier_state(cfg, device)
    logger = CSVLogger(DEFAULT_LOG_PATH, LOG_FIELDNAMES)
    input_dim = 28 * 28 * 1
    icnn_state = create_icnn_state(cfg, input_dim, device)
    for epoch in range(cfg.num_epochs):
        for step, (x, y) in enumerate(train_loader):
            if cfg.max_steps_algo2 is not None and step >= cfg.max_steps_algo2:
                break
            state, icnn_state, metrics = train_step_algo2(state, icnn_state, (x, y), cfg, device)
            if cfg.log_every and step % cfg.log_every == 0:
                print(f"[Algo2] Epoch {epoch} step {step} adv_loss={float(metrics['adv_loss']):.4f} cls_loss={float(metrics['cls_loss']):.4f} acc_clean={float(metrics['acc_clean']):.4f} acc_adv={float(metrics['acc_adv']):.4f} W2≈{float(metrics['w2_proxy']):.4f}")
                logger.log(algorithm="algo2", phase="train", epoch=epoch, step=step, loss_adv=None, adv_loss=float(metrics['adv_loss']), cls_loss=float(metrics['cls_loss']), acc_clean=float(metrics['acc_clean']), acc_adv=float(metrics['acc_adv']), w2_proxy=float(metrics['w2_proxy']))
    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print("[Algo2] Test:", test_metrics)
    logger.log(algorithm="algo2", phase="test", epoch=cfg.num_epochs, step=None, loss_adv=float(test_metrics["loss"]), adv_loss=None, cls_loss=float(test_metrics["loss"]), acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None)
    return state, icnn_state, {"test": test_metrics}

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(
        batch_size=512,
        num_epochs=1,
        lr_cls=1e-3,
        lambda_reg=0.5,
        log_every=100,
        max_steps_algo1=None,
        max_steps_algo2=None,
        use_margin_adv_algo1=False,
        use_margin_adv_algo2=False,
        inner_steps_algo1=5,
        inner_steps_algo2=5,
        bb_alpha0_x=1e-1,
        bb_alpha0_icnn=1e-2,
        icnn_hidden_sizes=(512, 512, 256, 128),
        seed=0,
    )

    print("Training Algorithm 1 (particle ascent with BB+Armijo)...")
    state_algo1, logs_algo1 = train_algorithm_1(cfg, device)

    print("Training Algorithm 2 (ICNN transport with BB+Armijo)...")
    state_algo2, icnn_state_algo2, logs_algo2 = train_algorithm_2(cfg, device)

    _, test_ds = load_mnist()
    pgd_kwargs = dict(batch_size=cfg.batch_size, eps=2.5, num_steps=20, device=device)

    pgd_algo1 = evaluate_pgd(state_algo1, test_ds, **pgd_kwargs)
    print(f"[Algo1] PGD acc={pgd_algo1['acc']*100:.2f}% L2={pgd_algo1['avg_l2']:.4f} Linf={pgd_algo1['avg_linf']:.4f}")

    pgd_algo2 = evaluate_pgd(state_algo2, test_ds, **pgd_kwargs)
    print(f"[Algo2] PGD acc={pgd_algo2['acc']*100:.2f}% L2={pgd_algo2['avg_l2']:.4f} Linf={pgd_algo2['avg_linf']:.4f}")
