import os
# Must be set before any CUDA kernel is launched.
# :4096:8  → 32 KB workspace (safe for all matmul sizes)
# :16:8    → 128 B workspace (may fail on large matmuls)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import copy
import math
import csv
import json
import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from scipy.optimize import linear_sum_assignment

from MNIST_C_utils import MNISTCDataset, CORRUPTIONS, ensure_mnist_c_downloaded


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# warn_only=True avoids hard errors on ops that have no deterministic kernel;
# set to False if you need strict guarantees and are willing to accept NotImplementedError.
torch.use_deterministic_algorithms(True, warn_only=True)


def seed_everything(seed: int) -> None:
    """Seed all RNG sources for full reproducibility.

    Covers: Python hash seed (not set here — pass PYTHONHASHSEED=<seed>),
    PyTorch CPU RNG, PyTorch CUDA RNG on all devices, and NumPy.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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
    "delta_gap",
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


class CarliniWagnerMNIST(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        # 28->26->24->MaxPool(12)->10->8->MaxPool(4) => 64 * 4 * 4 = 1024
        self.fc1 = nn.Linear(64 * 4 * 4, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
    step_offset: int = 0,
) -> torch.Tensor:
    """WRM inner maximization (Sinha et al.).

    Gradient ascent on CE(z, y) - lambda ||z - x||^2 w.r.t. z.
    Update rule: z <- z + (lr/sqrt(s)) * (grad_z CE(z,y) - 2*lambda*(z-x)).

    Parameters
    ----------
    step_offset : global step index of the first internal step.
                  Bug 9 fix: PPA calls this function across multiple rounds;
                  passing step_offset = round_idx * inner_steps_ppa ensures
                  the diminishing schedule lr/sqrt(s) is monotonically
                  decreasing across rounds rather than resetting to lr each
                  round.  Default 0 preserves the original single-round
                  behaviour.
    """
    if num_steps == 0:
        return x0.detach()
    x_orig = x0.detach()
    z = x_orig.clone()
    for s in range(1 + step_offset, num_steps + 1 + step_offset):
        z.requires_grad_(True)
        per_sample_ce = F.cross_entropy(model(z), y, reduction="none")
        grads = torch.autograd.grad(per_sample_ce.sum(), z, create_graph=False)[0]
        eta_s = lr / math.sqrt(s)
        z = z.detach() + eta_s * (grads - 2.0 * lambda_reg * (z.detach() - x_orig))
        if clamp is not None:
            lo, hi = clamp
            z = z.clamp(lo, hi)
    return z.detach()


def wrm_ascent_x_anchored(
    z0: torch.Tensor,
    x_anchor: torch.Tensor,
    model: nn.Module,
    y: torch.Tensor,
    lambda_reg: float,
    num_steps: int,
    lr: float = 0.01,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
    step_offset: int = 0,
) -> torch.Tensor:
    """WRM gradient ascent starting from z0 with penalty anchored at x_anchor.

    Unlike wrm_ascent_x where start == anchor, here the starting point z0
    can differ from the anchor x_anchor.  This is needed after Brenier
    projection: the projected point z_proj is the new start, but the
    quadratic penalty is still measured from the original nominal x.

    Maximises  CE(z, y) - lambda ||z - x_anchor||^2  w.r.t. z,
    starting from z = z0.

    Parameters
    ----------
    step_offset : see wrm_ascent_x.  In PPA, pass round_idx * inner_steps
                  so the step-size schedule continues across rounds.
    """
    if num_steps == 0:
        return z0.detach()
    x_anc = x_anchor.detach()
    z = z0.detach().clone()
    for s in range(1 + step_offset, num_steps + 1 + step_offset):
        z.requires_grad_(True)
        per_sample_ce = F.cross_entropy(model(z), y, reduction="none")
        grads = torch.autograd.grad(per_sample_ce.sum(), z, create_graph=False)[0]
        eta_s = lr / math.sqrt(s)
        z = z.detach() + eta_s * (grads - 2.0 * lambda_reg * (z.detach() - x_anc))
        if clamp is not None:
            lo, hi = clamp
            z = z.clamp(lo, hi)
    return z.detach()


def wrm_ascent_x_anchored_const_lr(
    z0: torch.Tensor,
    x_anchor: torch.Tensor,
    model: nn.Module,
    y: torch.Tensor,
    lambda_reg: float,
    num_steps: int,
    lr: float = 0.01,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> torch.Tensor:
    r"""WRM gradient ascent with constant step size for PPA refinement rounds.

    After Brenier projection, the particles sit at an OT-optimal coupling.
    The subsequent ascent is a refinement from a new starting point in a
    potentially different basin of attraction.  A constant step size is
    more appropriate than a diminishing schedule because:

    (a) The 1/sqrt(s) schedule is designed for cold-start convergence.
        In refinement rounds we need meaningful progress in few steps.

    (b) Constant-lr projected gradient ascent on a bounded domain converges
        to an eps-approximate stationary point in O(1/eps^2) steps.
        With K steps and step size eta, eps ~ 1/(eta*sqrt(K)).

    Maximises  CE(z, y) - lambda ||z - x_anchor||^2  w.r.t. z.
    """
    if num_steps == 0:
        return z0.detach()
    x_anc = x_anchor.detach()
    z = z0.detach().clone()
    for _ in range(num_steps):
        z.requires_grad_(True)
        per_sample_ce = F.cross_entropy(model(z), y, reduction="none")
        grads = torch.autograd.grad(per_sample_ce.sum(), z, create_graph=False)[0]
        z = z.detach() + lr * (grads - 2.0 * lambda_reg * (z.detach() - x_anc))
        if clamp is not None:
            lo, hi = clamp
            z = z.clamp(lo, hi)
    return z.detach()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def brenier_projection(
    z: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    r"""Within-class Brenier projection: optimally reassign adversarial
    points to nominals *within each class*.

    Bug 10 fix (critical for PPA):
    ---------------------------------------------------------------
    The paper (Section 2, "Supervised / label-preserving variants")
    explicitly requires that permutations be restricted to within each
    class:  G = prod_y S_{I_y}  where I_y = {i : y_i = y}.

    The previous implementation solved a single N x N LAP over ALL
    samples, allowing cross-class swaps.  This is incorrect for two
    reasons:

    (a) Semantic incoherence: after a cross-class swap, adversarial
        sample z_proj[i] carries label y_proj[i] != y_original[i].
        The subsequent WRM ascent round pushes z_i toward fooling the
        classifier on this WRONG label (relative to the nominal x_i),
        wasting adversarial capacity and producing incoherent gradients.

    (b) Violated theory: all propositions and lemmas in the paper assume
        within-class permutations.  The gain identity
          L(theta, Pi(z)) = L(theta, z) + lambda * Delta(z)
        still holds formally (loss sum is permutation-invariant), but the
        subsequent ascent step does NOT benefit because cross-class
        anchor reassignment creates a Lagrangian landscape unrelated to
        the original adversarial task.

    Fix: decompose into per-class LAPs.  For each class c, extract the
    indices I_c, solve the |I_c| x |I_c| assignment problem, and combine.
    Total cost is sum_c O(|I_c|^3) << O(N^3) for balanced classes.

    Parameters
    ----------
    z : Tensor [N, C, H, W]  -- adversarial points
    x : Tensor [N, C, H, W]  -- nominal points
    y : Tensor [N]            -- labels (permuted alongside z within class)

    Returns
    -------
    z_proj  : Tensor [N, C, H, W]  -- optimally reassigned within each class
    y_proj  : Tensor [N]            -- same as y (labels unchanged within class)
    delta   : float                 -- wasted-transport gap
    C_id    : float                 -- identity transport cost
    C_ot    : float                 -- optimal transport cost
    """
    N = z.size(0)
    if N <= 1:
        C_id = float(((z - x) ** 2).sum().item()) / max(N, 1)
        return z.clone(), y.clone(), 0.0, C_id, C_id

    z_flat = z.detach().view(N, -1)   # [N, d]
    x_flat = x.detach().view(N, -1)   # [N, d]

    # Build the global permutation by solving per-class LAPs
    perm_np = np.arange(N, dtype=np.int64)  # identity by default
    y_np = y.cpu().numpy()
    unique_classes = np.unique(y_np)

    for c in unique_classes:
        idx_c = np.where(y_np == c)[0]
        n_c = len(idx_c)
        if n_c <= 1:
            continue

        # Extract class-c subsets (all on CPU for determinism)
        x_c = x_flat[idx_c].cpu().float()   # [n_c, d]
        z_c = z_flat[idx_c].cpu().float()   # [n_c, d]

        # Cost matrix for class c:  cost[i, j] = ||x_{I_c[i]} - z_{I_c[j]}||^2
        x_sq = (x_c ** 2).sum(dim=1, keepdim=True)
        z_sq = (z_c ** 2).sum(dim=1, keepdim=True)
        cross = x_c.mm(z_c.t())
        cost_c = (x_sq + z_sq.t() - 2.0 * cross).clamp(min=0.0)
        cost_c_np = cost_c.numpy()

        # Solve LAP within class c
        row_ind, col_ind = linear_sum_assignment(cost_c_np)

        # Map back to global indices
        local_perm = np.empty(n_c, dtype=np.int64)
        local_perm[row_ind] = col_ind
        perm_np[idx_c] = idx_c[local_perm]

    perm = torch.tensor(perm_np, device=z.device, dtype=torch.long)

    # Apply permutation
    z_proj = z[perm]
    y_proj = y[perm]  # within-class: y_proj == y (labels are preserved)

    # Compute metrics
    C_id = float(((z - x) ** 2).view(N, -1).sum(dim=1).mean().item())
    C_ot = float(((z_proj - x) ** 2).view(N, -1).sum(dim=1).mean().item())
    delta = max(C_id - C_ot, 0.0)

    return z_proj, y_proj, delta, C_id, C_ot


# ---------------------------------------------------------------------------
#  Cyclical Monotonicity Diagnostic (for Algorithm 1 / WRM)
# ---------------------------------------------------------------------------

def check_cyclical_monotonicity(
    x: torch.Tensor,
    z: torch.Tensor,
    cycle_lengths: Sequence[int] = (2, 3, 4, 5, 6, 8, 10),
    num_samples: int = 500,
    generator: Optional[torch.Generator] = None,
) -> Dict[int, Dict[str, Any]]:
    r"""Check cyclical monotonicity (CM) violations of the identity coupling.

    Background (Rockafellar 1966; Villani 2009, Thm 5.10):
    -------------------------------------------------------
    A coupling (x_i, z_i) is induced by the gradient of a convex potential
    psi (i.e. z_i = nabla psi(x_i)) **if and only if** it is cyclically
    monotone: for every k >= 2 and every k-tuple of indices (i_1, ..., i_k),

        sum_{j=1}^{k}  <z_{i_j},  x_{i_j} - x_{i_{j+1}}>  >=  0
                                                   (indices mod k)

    Equivalently, for the squared-Euclidean cost c(a,b) = ||a-b||^2, the
    identity assignment must be at least as cheap as any cyclic reassignment:

        C_id  :=  sum_j  ||z_{i_j} - x_{i_j}||^2
                  <=  sum_j  ||z_{sigma(i_j)} - x_{i_j}||^2  =: C_cyc

    for every cyclic permutation sigma of the selected indices.

    The *violation* V_k = C_id - C_cyc > 0 when the identity coupling is
    sub-optimal for that cycle, meaning transport "crosses" and the coupling
    cannot be the gradient of any convex function.

    Checking only k=2 (pairwise swaps) tests ordinary monotonicity but
    misses violations that only appear at longer cycle lengths.  A coupling
    can be 2-monotone yet fail k-monotonicity for k >= 3.

    Normalization across cycle lengths:
    ------------------------------------
    Raw violations scale linearly with k (more edges in the cycle), making
    direct comparison misleading.  We report two normalizations:

    1. **Per-edge**:  V_k / k
       Converts the extensive violation sum into an intensive (per-edge)
       quantity.  Comparable across cycle lengths under the null hypothesis
       that violations are i.i.d. along edges.

    2. **Relative**:  V_k / C_id
       Dimensionless ratio giving the fraction of identity-coupling cost
       that is "wasted" (could be saved by cyclic reassignment).  Scale-
       invariant: unaffected by the magnitude of perturbations.

    Parameters
    ----------
    x : [N, C, H, W]  nominal samples
    z : [N, C, H, W]  adversarial samples (WRM output)
    cycle_lengths : tuple of k values to probe
    num_samples   : number of random k-cycles to sample per k
    generator     : optional torch.Generator for reproducible sampling.
                    Bug 7 fix: without an explicit generator, torch.randperm
                    consumes the global RNG, making CM statistics dependent on
                    the exact sequence of preceding RNG calls (batch index,
                    epoch, etc.).  Pass a seeded generator to make the CM
                    diagnostic fully reproducible independently of context.

    Returns
    -------
    dict  keyed by cycle length k, each value a dict of statistics.
    """
    N = x.size(0)
    x_flat = x.detach().view(N, -1)
    z_flat = z.detach().view(N, -1)

    # Pre-compute per-sample identity costs  ||z_i - x_i||^2
    id_costs = (z_flat - x_flat).pow(2).sum(dim=1)          # [N]

    results: Dict[int, Dict[str, Any]] = {}

    for k in cycle_lengths:
        if k > N:
            continue

        violations_raw = []
        violations_per_edge = []
        violations_relative = []

        for _ in range(num_samples):
            # Bug 7 fix: pass generator so sampling is reproducible.
            idx = torch.randperm(N, device=x.device, generator=generator)[:k]

            # Identity cost for this cycle
            c_id = id_costs[idx].sum().item()

            # Cyclic-permutation cost: pair x_{i_j} with z_{i_{j+1 mod k}}
            idx_shifted = idx.roll(-1)
            c_cyc = (z_flat[idx_shifted] - x_flat[idx]).pow(2).sum().item()

            raw = c_id - c_cyc                             # > 0 ⟹ CM violated
            violations_raw.append(raw)
            violations_per_edge.append(raw / k)
            violations_relative.append(raw / max(c_id, 1e-12))

        raw_t = torch.tensor(violations_raw)
        pe_t  = torch.tensor(violations_per_edge)
        rel_t = torch.tensor(violations_relative)

        results[k] = {
            "cycle_length":      k,
            "mean_raw":          float(raw_t.mean()),
            "std_raw":           float(raw_t.std()),
            "mean_per_edge":     float(pe_t.mean()),
            "std_per_edge":      float(pe_t.std()),
            "mean_relative":     float(rel_t.mean()),
            "std_relative":      float(rel_t.std()),
            "frac_violated":     float((raw_t > 0).float().mean()),
            "max_raw":           float(raw_t.max()),
            "max_per_edge":      float(pe_t.max()),
            "max_relative":      float(rel_t.max()),
            "num_samples":       num_samples,
        }

    return results


def aggregate_cm_results(
    batch_results: Sequence[Dict[int, Dict[str, Any]]],
) -> Dict[int, Dict[str, Any]]:
    """Average CM diagnostics collected across multiple mini-batches.

    For quantities that are means (mean_raw, mean_per_edge, mean_relative,
    frac_violated), we take the mean-of-means.  For maxima (max_raw, …)
    we take the max-of-maxes.  For standard deviations, we average them
    (providing a rough pooled estimate without needing the raw samples).
    """
    if len(batch_results) == 0:
        return {}

    # Collect all cycle lengths seen
    all_k = sorted({k for res in batch_results for k in res})
    agg: Dict[int, Dict[str, Any]] = {}

    for k in all_k:
        entries = [res[k] for res in batch_results if k in res]
        if len(entries) == 0:
            continue
        n = len(entries)
        agg[k] = {
            "cycle_length":   k,
            "mean_raw":       sum(e["mean_raw"]      for e in entries) / n,
            "std_raw":        sum(e["std_raw"]        for e in entries) / n,
            "mean_per_edge":  sum(e["mean_per_edge"]  for e in entries) / n,
            "std_per_edge":   sum(e["std_per_edge"]   for e in entries) / n,
            "mean_relative":  sum(e["mean_relative"]  for e in entries) / n,
            "std_relative":   sum(e["std_relative"]   for e in entries) / n,
            "frac_violated":  sum(e["frac_violated"]  for e in entries) / n,
            "max_raw":        max(e["max_raw"]        for e in entries),
            "max_per_edge":   max(e["max_per_edge"]   for e in entries),
            "max_relative":   max(e["max_relative"]   for e in entries),
            "num_batches":    n,
        }
    return agg


# ---------------------------------------------------------------------------
#  Config and states
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    num_epochs: int = 3
    lr_cls: float = 1e-3
    lr_cls_drop_epoch: int = 30       # epoch at which lr drops by 10x
    lr_cls_drop_factor: float = 0.1   # multiplicative factor applied at drop epoch
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
    # PPA (Projected Particle Ascent) parameters — enhanced
    # Round 0 replicates Algo1 exactly (the dominance condition):
    inner_steps_ppa_round0: int = 75     # MUST equal inner_steps_algo1
    inner_lr_ppa_round0: float = 1e-2    # MUST equal inner_lr_algo1
    # Refinement rounds (additional computation beyond Algo1):
    ppa_num_rounds: int = 5        # total rounds including round 0
    ppa_min_rounds: int = 2        # minimum before early stopping kicks in
    ppa_refine_steps: int = 15     # constant-lr ascent steps per refinement
    ppa_refine_lr: float = 5e-3    # constant lr for refinement rounds
    ppa_delta_rtol: float = 1e-4   # relative Δ threshold for stopping
    max_steps_ppa: Optional[int] = None
    # Diagnostics
    cm_diagnostics: bool = False   # cyclical monotonicity diagnostic for Algo1
    seed: int = 0
    # -----------------------------------------------------------------------
    # Checkpoint-selection / early-stopping parameters
    # -----------------------------------------------------------------------
    # es_val_frac  — fraction of the 60 000-sample MNIST training set held out
    #   as a validation split.  The split is created once per training run with
    #   a fixed generator seeded by cfg.seed and is never exposed to gradient
    #   updates.  0.0833 ≈ 5 000 samples.
    es_val_frac: float = 0.0833
    # es_patience  — number of consecutive adversarial epochs with no
    #   improvement in the validation score before training terminates early.
    #   Setting this to 0 disables early termination; the best checkpoint is
    #   still restored at the end.
    es_patience: int = 0
    # es_pgd_eps / es_pgd_steps / es_pgd_restarts — PGD-L2 attack parameters
    #   used to score each checkpoint on the validation split.  Intentionally
    #   cheaper than the final evaluation (20 steps / 3 restarts vs 40 / 5)
    #   to keep per-epoch overhead manageable.
    es_pgd_eps: float = 1.3
    es_pgd_steps: int = 20
    es_pgd_restarts: int = 3
    # es_clean_weight — α in  score = α·val_clean_acc + (1−α)·val_pgd_acc.
    #
    #   α = 0.0  (default): select purely on adversarial robustness.  This is
    #     the theoretically motivated choice: the WRM / ICNN / PPA training
    #     objectives minimise worst-case risk; the checkpoint criterion should
    #     be consistent with that objective.  Tsipras et al. (2019) and the
    #     TRADES bound show that clean and robust accuracy are in tension, so
    #     their unweighted sum (α = 0.5) has no theoretical grounding and
    #     systematically over-selects models near the clean-accuracy ceiling.
    #   α ∈ (0, 1): blended criterion for applications where clean accuracy
    #     matters; set α to reflect the application's clean/robust trade-off.
    es_clean_weight: float = 0.0


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
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None


@dataclass
class ICNNState:
    model: InputConvexPotential
    params_vec: torch.Tensor
    meta: Tuple[Tuple[str, Tuple[int, ...], int], ...]
    bb_state: BBArmijoState
    # FIX Bug 4: persistent leaf tensor and AdamW instance so that the
    # first-moment (m̂_t) and second-moment (v̂_t) accumulators survive across
    # mini-batches, giving Adam its full momentum benefit.  Recreating the
    # optimizer every step discards these accumulators, degrading Adam to a
    # bias-corrected SGD step with constant scaling.
    inner_param: torch.Tensor                # persistent leaf, updated in-place
    inner_opt: torch.optim.Optimizer         # AdamW bound to inner_param

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


def split_train_val(
    dataset: torch.utils.data.Dataset,
    val_frac: float,
    seed: int,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Deterministically split *dataset* into a training and validation subset.

    The split uses a generator seeded independently of the global RNG so it
    is stable across different training configurations.  MNIST classes are
    balanced (6 000 samples per class), so unstratified random splitting
    produces a validation set with class proportions within 1–2% of uniform.

    Parameters
    ----------
    val_frac : fraction of samples assigned to the validation set.
    seed     : seed for the splitting generator.

    Returns
    -------
    train_subset, val_subset  — two non-overlapping Subsets of *dataset*.
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")
    n       = len(dataset)
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=gen)


def compute_val_score(
    state: "TrainState",
    val_ds: torch.utils.data.Dataset,
    cfg: "TrainConfig",
    device: torch.device,
    is_clean_phase: bool,
) -> Tuple[float, float, float]:
    r"""Score a checkpoint on the held-out validation split.

    During clean warm-up (*is_clean_phase=True*) the model has never been
    trained adversarially, so PGD accuracy is undefined as a robustness
    signal.  In that phase the score equals the clean validation accuracy,
    which is the natural criterion for warm-up checkpointing.

    During adversarial epochs the score is:

        score = α · val_clean_acc + (1 − α) · val_pgd_acc

    where α = cfg.es_clean_weight.

    Choosing α = 0 (the default) selects purely on robustness.  This is
    consistent with the WRM / ICNN / PPA training objectives which minimise
    worst-case risk.  The original code used

        score = train_acc_clean + train_adv_acc   (α = 0.5 on *training* data)

    which is wrong for three independent reasons:
      (a) Evaluated on training data  → optimistic bias; not a generalisation
          signal.
      (b) "train_adv_acc" is accuracy against the *training* adversary (WRM
          gradient ascent / ICNN transport), not against PGD.  The two metrics
          can diverge substantially, so the selected checkpoint may be the
          worst under PGD evaluation.
      (c) Equal weight 1:1 has no theoretical grounding (Tsipras et al. 2019).

    Parameters
    ----------
    is_clean_phase : True if the current epoch is a clean warm-up epoch.

    Returns
    -------
    (score, val_clean_acc, val_pgd_acc)
        val_pgd_acc is 0.0 during clean phase (PGD not run).
    """
    clean_m = evaluate_clean(state, val_ds, cfg.batch_size, device)
    val_clean_acc = clean_m["acc"]

    if is_clean_phase:
        return val_clean_acc, val_clean_acc, 0.0

    pgd_m = evaluate_pgd(
        state, val_ds, cfg.batch_size,
        eps=cfg.es_pgd_eps,
        num_steps=cfg.es_pgd_steps,
        restarts=cfg.es_pgd_restarts,
        device=device,
    )
    val_pgd_acc = pgd_m["acc"]
    alpha = cfg.es_clean_weight
    score = alpha * val_clean_acc + (1.0 - alpha) * val_pgd_acc
    return score, val_clean_acc, val_pgd_acc

# ---------------------------------------------------------------------------
#  PGD
# ---------------------------------------------------------------------------

def project_l2(x: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    """Project x onto the L2 ball of radius eps centred at x_orig.

    For each sample i:
        x_proj[i] = x_orig[i] + (x[i] - x_orig[i]) * min(1, eps / ||x[i]-x_orig[i]||_2)

    The +1e-12 guard prevents division by zero when x == x_orig without
    introducing meaningful error (||diff|| > 1e-12 for any non-trivial step).
    """
    diff = x - x_orig
    flat = diff.view(diff.size(0), -1)
    norm = flat.norm(p=2, dim=1, keepdim=True)
    factor = (eps / (norm + 1e-12)).clamp(max=1.0)
    factor = factor.view(-1, *([1] * (x.dim() - 1)))
    return x_orig + diff * factor


def _per_sample_l2_normalize(t: torch.Tensor, floor: float = 1e-12) -> torch.Tensor:
    """Normalise each sample in a batch to unit L2 norm.

    Returns t / ||t[i]||_2 for each sample i, with a floor to avoid NaN when
    the gradient is exactly zero (degenerate case; zero-gradient samples take
    a zero step, which is the correct behaviour).
    """
    B = t.size(0)
    flat = t.view(B, -1)
    norms = flat.norm(p=2, dim=1).clamp(min=floor)
    return t / norms.view(B, *([1] * (t.dim() - 1)))


def _sphere_start_l2(x0: torch.Tensor, eps: float) -> torch.Tensor:
    """Sample a random starting point for PGD on the L2 sphere of radius eps,
    respecting the box constraint [0, 1]^d.

    Bug 6 fix — why the naive clamp(x0 + z*eps, 0, 1) is wrong
    ------------------------------------------------------------
    A random unit direction z has roughly half its components pointing into the
    box boundary (negative for pixels near 0, positive for pixels near 1).
    After clamping, the effective perturbation is

        delta_eff = clamp(x0 + z*eps, 0, 1) - x0

    whose L2 norm satisfies  E[||delta_eff||_2] ≈ eps/sqrt(2) ≈ 0.707*eps
    for a typical MNIST image (half-white, half-black background).  PGD does
    NOT re-expand the iterate: _project_onto_l2_ball clips *within* the eps-
    ball but does not expand toward the boundary.  So the clamp permanently
    wastes ~30% of the perturbation budget on the very first restart.

    Correct approach: project-then-normalise
    ----------------------------------------
    1. Draw z ~ N(0, I_d) and normalise to the unit sphere.
    2. Form the unconstrained proposal  p = x0 + eps * z.
    3. Project p onto the box: p_box = clamp(p, 0, 1).
    4. Compute the actual delta: d = p_box - x0.
    5. If ||d||_2 > 0, re-normalise d to exactly eps so the start sits on the
       sphere of the *feasible* perturbation set (intersection of L2 ball and
       box).  If ||d||_2 == 0 (degenerate, extremely rare), fall back to x0.

    After step 5 the iterate satisfies ||start - x0||_2 = eps exactly *and*
    start in [0,1]^d, so no budget is wasted.  The distribution over feasible
    directions is not perfectly uniform, but it is unbiased with respect to the
    eps boundary of the feasible set, which is what matters for PGD restarts.
    """
    z = torch.randn_like(x0)
    z = _per_sample_l2_normalize(z)           # unit direction on R^d sphere
    p = x0 + eps * z                          # unconstrained sphere point
    p_box = p.clamp(0.0, 1.0)                 # project onto box
    d = p_box - x0                            # feasible delta (may be < eps in norm)
    B = x0.size(0)
    d_flat = d.view(B, -1)
    norms = d_flat.norm(p=2, dim=1)           # [B]
    # Re-normalise to eps; fall back to x0 for the degenerate zero-norm case.
    safe_norms = norms.clamp(min=1e-12)
    scale = (eps / safe_norms)                # [B]
    scale = scale.view(B, *([1] * (x0.dim() - 1)))
    d_scaled = d * scale                      # ||d_scaled[i]||_2 = eps for all i
    # For any sample where the original norm was effectively 0, use zero delta.
    zero_mask = (norms < 1e-12).view(B, *([1] * (x0.dim() - 1)))
    d_final = torch.where(zero_mask, torch.zeros_like(d_scaled), d_scaled)
    return (x0 + d_final).clamp(0.0, 1.0)


def _project_onto_l2_ball(delta: torch.Tensor, eps: float) -> torch.Tensor:
    """Project a perturbation tensor onto the L2 ball of radius eps.

    Operates per-sample:  delta[i] <- delta[i] * min(1, eps / ||delta[i]||_2).
    Returns a tensor of the same shape as delta.
    """
    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    factors = (eps / norms).clamp(max=1.0)
    return (flat * factors).view_as(delta)


def pgd_l2_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_steps: int,
    step_size: Optional[float] = None,
) -> torch.Tensor:
    """Single-start PGD-L2 from x (no random restart).

    Intended for training-time adversarial example generation where a cheap,
    deterministic attack suffices.  For evaluation use
    ``pgd_l2_attack_restarts`` which includes random restarts and is
    therefore a strictly stronger attack.

    Step size convention: 2*eps/T (standard; allows the iterate to traverse
    the full ball diameter 2*eps in T perfectly-aligned steps).
    """
    if step_size is None:
        step_size = 2.0 * float(eps) / float(max(num_steps, 1))
    elif step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size!r}.")

    was_training = model.training
    model.eval()
    try:
        adv = x.detach().clone()
        for _ in range(num_steps):
            adv = adv.detach().requires_grad_(True)
            logits = model(adv)
            # Use sum reduction so grad[i] = ∂L_i/∂x_i exactly (no 1/B factor).
            loss = F.cross_entropy(logits, y, reduction="sum")
            grad = torch.autograd.grad(loss, adv, create_graph=False)[0]
            with torch.no_grad():
                adv = adv + step_size * _per_sample_l2_normalize(grad)
                adv = project_l2(adv, x, eps)
                adv = adv.clamp(0.0, 1.0)
        return adv.detach()
    finally:
        model.train(was_training)


def pgd_l2_attack_restarts(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_steps: int,
    step_size: float,
    restarts: int,
) -> torch.Tensor:
    r"""PGD-L2 attack with a deterministic first restart and random subsequent restarts.

    Algorithm
    ---------
    For r = 0, 1, ..., restarts-1:
        x_adv^(r,0) = x                              if r == 0   (deterministic)
                    = sphere_start(x, eps)           if r >= 1   (random)
        for t = 1, ..., T:
            g_t = nabla_{x} CE(f(x_adv^(r,t-1)), y)
            x_adv^(r,t) = Pi_{B(x,eps)} ( x_adv^(r,t-1) + alpha * g_t / ||g_t||_2 )
            x_adv^(r,t) = clamp(x_adv^(r,t), 0, 1)
        keep x_adv^(r,T) if CE(f(x_adv^(r,T)), y) > current best (per sample)

    The deterministic r=0 restart ensures the attack is always at least as
    strong as single-start PGD from x.  Random restarts r≥1 escape local
    maxima.

    Gradient convention
    -------------------
    Using ``reduction="sum"`` means grad[i] = ∂L_i/∂x_i without any 1/B
    scaling factor.  After ``_per_sample_l2_normalize`` the direction is
    identical to the reduction="mean" case, but the computation graph is
    cleaner and per-sample gradients are never implicitly divided by B.

    Parameters
    ----------
    restarts : total number of restarts (including the deterministic one).
               Must be >= 1; values < 1 are treated as 1.
    """
    was_training = model.training
    model.eval()
    try:
        x0 = x.detach()
        best_delta = torch.zeros_like(x0)
        best_loss  = torch.full((x0.size(0),), -float("inf"), device=x0.device)

        for r in range(max(1, restarts)):
            # Restart 0: deterministic start from x (clean PGD baseline).
            # Restarts ≥ 1: random start on the sphere boundary.
            if r == 0:
                x_adv = x0.clone().requires_grad_(True)
            else:
                x_adv = _sphere_start_l2(x0, eps).detach().requires_grad_(True)

            for _ in range(num_steps):
                logits = model(x_adv)
                # sum reduction: grad[i] = ∂L_i/∂x_i, no implicit 1/B scaling.
                loss = F.cross_entropy(logits, y, reduction="sum")
                grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
                with torch.no_grad():
                    x_adv = x_adv + step_size * _per_sample_l2_normalize(grad)
                    delta  = _project_onto_l2_ball(x_adv - x0, eps)
                    x_adv  = (x0 + delta).clamp(0.0, 1.0)
                x_adv = x_adv.detach().requires_grad_(True)

            # Track the best adversarial example per sample.
            with torch.no_grad():
                logits = model(x_adv)
                per_sample_loss = F.cross_entropy(logits, y, reduction="none")
                delta = (x_adv - x0).detach()
                improved = per_sample_loss > best_loss
                best_loss[improved]  = per_sample_loss[improved]
                best_delta[improved] = delta[improved]

        return (x0 + best_delta).clamp(0.0, 1.0).detach()
    finally:
        model.train(was_training)


def evaluate_pgd(
    state: TrainState,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    eps: float = 0.3,
    step_size: Optional[float] = None,
    num_steps: int = 40,
    restarts: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate adversarial accuracy under PGD-L2 with random restarts.

    Uses the same attack (``pgd_l2_attack_restarts``) and the same default
    step size (``2*eps/num_steps``) as ``evaluate_pgd_l2_sweep`` so the two
    functions are directly comparable at the same epsilon.

    Metrics are sample-weighted means (not batch-count means), so the
    result is unbiased when the last batch is smaller than batch_size.
    """
    device = device or next(state.model.parameters()).device
    _step = step_size if (step_size is not None and step_size > 0) \
            else 2.0 * eps / max(num_steps, 1)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_correct = 0
    total_l2 = total_linf = 0.0
    total_n = 0

    was_training = state.model.training
    state.model.eval()
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if x.size(0) == 0:
                continue
            adv_x = pgd_l2_attack_restarts(
                state.model, x, y, eps, num_steps, _step, restarts
            )
            with torch.no_grad():
                logits = state.model(adv_x)
                total_correct += (logits.argmax(dim=1) == y).sum().item()
            n = x.size(0)
            total_n += n
            diff = (adv_x - x).detach().view(n, -1)
            total_l2   += diff.norm(p=2, dim=1).sum().item()
            total_linf += diff.abs().max(dim=1).values.sum().item()
    finally:
        state.model.train(was_training)

    return {
        "acc":      total_correct / max(1, total_n),
        "avg_l2":   total_l2     / max(1, total_n),
        "avg_linf": total_linf   / max(1, total_n),
    }


def evaluate_pgd_l2_sweep(
    state: TrainState,
    dataset: torch.utils.data.Dataset,
    pgd_cfg: PGDEvalConfig,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model robustness under PGD-L2 for each epsilon in pgd_cfg.

    Uses ``pgd_l2_attack_restarts`` with random restarts and step size
    ``2*eps/num_steps`` (unless overridden in pgd_cfg).  Both accuracy and
    distortion metrics are true sample-means (not batch-count means).

    Notes
    -----
    The DataLoader is created once and reused across epsilon values; since
    ``shuffle=False`` this yields identical batches on every pass.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    was_training = state.model.training
    state.model.eval()
    results: Dict[str, Any] = {}

    try:
        for eps in pgd_cfg.epsilons:
            step_size = (
                pgd_cfg.step_size
                if pgd_cfg.step_size is not None and pgd_cfg.step_size > 0
                else 2.0 * eps / max(pgd_cfg.num_steps, 1)
            )

            total_correct = 0
            total_n       = 0
            total_l2      = 0.0
            total_linf    = 0.0

            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                if xb.size(0) == 0:
                    continue
                adv_x = pgd_l2_attack_restarts(
                    state.model, xb, yb, eps,
                    pgd_cfg.num_steps, step_size, pgd_cfg.restarts,
                )
                with torch.no_grad():
                    logits = state.model(adv_x)
                    total_correct += (logits.argmax(dim=1) == yb).sum().item()
                    n = xb.size(0)
                    total_n += n
                    delta = (adv_x - xb).detach().view(n, -1)
                    # Sample-weighted accumulation (Bug 5 fix: .sum() not .mean())
                    total_l2   += delta.norm(p=2, dim=1).sum().item()
                    total_linf += delta.abs().max(dim=1).values.sum().item()

            acc       = total_correct / max(1, total_n)
            avg_l2    = total_l2      / max(1, total_n)
            avg_linf  = total_linf    / max(1, total_n)

            # Use :.4g so 1.6, 1.60, 2.0, 0.1 all produce exact, collision-free keys.
            eps_key = f"eps_{eps:.4g}"
            results[eps_key] = {
                "epsilon":   eps,
                "step_size": step_size,
                "num_steps": pgd_cfg.num_steps,
                "restarts":  pgd_cfg.restarts,
                "acc_pct":   round(acc * 100, 2),
                "avg_l2":    round(avg_l2,    4),
                "avg_linf":  round(avg_linf,  4),
                "samples":   total_n,
            }
            print(
                f"    eps={eps:.4g}: acc={acc*100:.2f}%"
                f" avg_L2={avg_l2:.4f} avg_Linf={avg_linf:.4f}"
            )
    finally:
        state.model.train(was_training)

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

    # Bug 1 fix: recompute logits from the post-update model so that
    # acc_clean reflects the weights that will be used going forward,
    # consistent with train_step_algo1 and train_step_algo2.
    with torch.no_grad():
        logits_post = model(x)
        acc = accuracy(logits_post, y)
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

    # --- Cyclical monotonicity diagnostic (no grad, pure analysis) ---
    if cfg.cm_diagnostics:
        with torch.no_grad():
            # Bug 7 fix: use a seeded generator so CM statistics are
            # reproducible regardless of global RNG state at call time.
            cm_gen = torch.Generator(device=x.device)
            cm_gen.manual_seed(0)
            cm = check_cyclical_monotonicity(x, adv_x, generator=cm_gen)
        metrics["cm_diagnostics"] = cm

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
        # Bug 4 fix: do NOT clamp adv_flat inside the inner objective.
        #
        # Two problems with clamping here:
        # (4a) Gradient truncation: clamp has zero derivative outside [0,1], so
        #      any pixel where the transport map overshoots the box receives zero
        #      gradient signal back to the ICNN parameters theta.  As the map
        #      grows stronger and more pixels saturate, the effective gradient
        #      vanishes — causing the inner loss to plateau prematurely.
        #
        # (4b) Biased W2 proxy: the WRM objective penalises
        #          ||nabla_theta psi(x) - x||^2
        #      Replacing nabla_theta psi(x) with clamp(nabla_theta psi(x))
        #      understates the transport cost for out-of-range components,
        #      letting the optimizer push pixels outside [0,1] for free.
        #
        # The correct place for the clamp is at the outer minimization step
        # (already present at line ~1221), which is a post-processing step on
        # the final adversarial point and does not participate in ICNN training.
        adv_x_inner = adv_flat.view_as(x)
        logits = model(adv_x_inner)
        adv_loss = adversary_loss(logits, y, cfg.use_margin_adv_algo2)
        w2 = ((adv_x_inner - x) ** 2).sum(dim=(1, 2, 3)).mean()
        return adv_loss - cfg.lambda_reg * w2

    # FIX Bug 4: reuse the persistent inner_param / inner_opt stored in
    # icnn_state so that Adam's first- and second-moment buffers accumulate
    # across mini-batches.  We sync inner_param.data to the current params_vec
    # (they are the same tensor after the first step, but this is safe).
    inner_param = icnn_state.inner_param
    inner_opt = icnn_state.inner_opt
    with torch.no_grad():
        inner_param.data.copy_(icnn_state.params_vec.to(device))
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

    params_vec = inner_param.detach().clone()
    # BB state is kept for continuity but no longer drives the step size
    icnn_state = ICNNState(model=icnn_model, params_vec=params_vec, meta=meta,
                           bb_state=bb_state, inner_param=inner_param, inner_opt=inner_opt)

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


def train_step_ppa(state: TrainState, batch, cfg: TrainConfig, device: torch.device) -> Tuple[TrainState, Dict[str, torch.Tensor]]:
    """Enhanced PPA (Projected Particle Ascent) training step.

    Corrected algorithm (per Section 3.5.4 of the paper):
    =====================================================
    The paper's theoretical comparison requires Algorithm B (PPA) to be a
    strict *refinement* of Algorithm A (plain WRM ascent):

        "We first run plain particle ascent (Algorithm A) to convergence
         to obtain z_A. Algorithm B then takes this exact z_A, applies the
         Brenier projection, and continues with additional projection/ascent
         cycles to obtain z_B."

    This ensures L(theta, z_PPA) >= L(theta, z_Algo1) and hence
    eps_PPA <= eps_Algo1 (inner oracle error dominance, Equation 11).

    Round 0:  Replicate Algo1 exactly — same steps, same lr.
    Round r (r >= 1):
        (a) Brenier projection: z <- Pi(z)        [+lambda*Delta gain]
        (b) Constant-lr ascent from projected z    [further refinement]
    Final: one last projection to capture remaining wasted transport.

    Key fixes vs. previous implementation:
    --------------------------------------
    1. Round 0 uses inner_steps_ppa_round0 = inner_steps_algo1 (was 10).
       This is the dominance condition; without it, PPA can be *weaker*.

    2. Refinement rounds use constant step size (was diminishing with
       geometric base-lr decay).  After projection the particles are in
       a new basin; constant-lr makes meaningful progress in few steps.

    3. Adaptive stopping uses a relative threshold (delta / C_id) with
       a minimum-rounds guarantee (was absolute 1e-8, no minimum).
       Prevents premature stopping in early training.

    4. A final projection captures any remaining wasted transport from
       the last ascent phase.  Free improvement (Lemma proj_gain).

    Invariant (Lemma proj_gain):
        L(theta, Pi(z)) = L(theta, z) + lambda * Delta(z),  Delta >= 0.
    """
    model = state.model
    opt = state.opt
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    if x.size(0) == 0:
        zero = torch.tensor(0.0, device=device)
        metrics = {"loss_adv": zero, "acc_clean": zero, "acc_adv": zero,
                   "w2_proxy": zero, "delta_gap": zero}
        return state, metrics

    # Inner maximization: freeze classifier
    set_requires_grad(model, False)
    model.eval()

    total_delta = 0.0

    # ===== Round 0: Replicate Algo1 exactly =====
    # This is the dominance condition.  By using the same number of ascent
    # steps and the same learning rate as Algo1, the inner objective after
    # round 0 is identical to what Algo1 would produce.  Everything after
    # this is strictly additional refinement.
    z = wrm_ascent_x(
        x, model, y, cfg.lambda_reg,
        cfg.inner_steps_ppa_round0,       # same as inner_steps_algo1
        lr=cfg.inner_lr_ppa_round0,       # same as inner_lr_algo1
        clamp=(0.0, 1.0),
        step_offset=0,
    )

    # ===== Refinement rounds 1..R-1: project then ascend =====
    for round_idx in range(1, cfg.ppa_num_rounds):

        # (a) Within-class Brenier projection (Bug 10 fix preserved)
        z, _y_proj, delta, _C_id, _C_ot = brenier_projection(z, x, y)
        total_delta += delta

        # (b) Adaptive stopping (with minimum-rounds guarantee).
        #     In early training, perturbations are small and Delta ~ 0
        #     even though the coupling may be suboptimal.  The minimum-
        #     rounds guarantee forces at least ppa_min_rounds of
        #     refinement before we consider stopping.
        if (round_idx >= cfg.ppa_min_rounds
                and delta < cfg.ppa_delta_rtol * max(_C_id, 1e-12)):
            break

        # (c) Constant-lr ascent from projected position.
        #     Anchor is always the original x (Lagrangian penalty).
        #     Constant lr is appropriate because this is a refinement
        #     from a new OT-optimal starting point, not a cold start.
        z = wrm_ascent_x_anchored_const_lr(
            z, x, model, y, cfg.lambda_reg,
            num_steps=cfg.ppa_refine_steps,
            lr=cfg.ppa_refine_lr,
            clamp=(0.0, 1.0),
        )

    # Final projection: captures any remaining wasted transport from
    # the last ascent phase.  Gain is lambda * Delta >= 0 (Lemma 1).
    z, _y_proj, delta_final, _, _ = brenier_projection(z, x, y)
    total_delta += delta_final

    adv_x = z.detach()

    # Outer minimization: unfreeze classifier, train
    set_requires_grad(model, True)
    model.train()

    logits_adv = model(adv_x)
    loss = cross_entropy_loss(logits_adv, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Post-update metrics (both from same model)
    with torch.no_grad():
        logits_clean = model(x)
        logits_adv_post = model(adv_x)
        acc_clean = accuracy(logits_clean, y)
        acc_adv = accuracy(logits_adv_post, y)
        w2_proxy = ((adv_x - x) ** 2).sum(dim=(1, 2, 3)).mean()
    metrics = {
        "loss_adv": loss.detach(),
        "acc_clean": acc_clean,
        "acc_adv": acc_adv,
        "w2_proxy": w2_proxy,
        "delta_gap": torch.tensor(total_delta, device=device),
    }
    return state, metrics


# ---------------------------------------------------------------------------
#  Training loops
# ---------------------------------------------------------------------------

def create_classifier_state(cfg: TrainConfig, device: torch.device) -> TrainState:
    model = CarliniWagnerMNIST().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr_cls, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[cfg.lr_cls_drop_epoch], gamma=cfg.lr_cls_drop_factor
    )
    return TrainState(model=model, opt=opt, scheduler=scheduler)


def create_icnn_state(cfg: TrainConfig, input_dim: int, device: torch.device) -> ICNNState:
    icnn_model = InputConvexPotential(input_dim=input_dim, hidden_sizes=cfg.icnn_hidden_sizes, activation="softplus", strong_convexity=1.0, nonneg_init="principled").to(device)
    icnn_model.init_as_identity()
    params_vec, meta = flatten_params(icnn_model)
    params_vec = params_vec.to(device)
    bb_state = BBArmijoState.create(alpha0=cfg.bb_alpha0_icnn)
    # FIX Bug 4: create the persistent leaf tensor and bind Adam to it once.
    # train_step_algo2 will copy the latest params_vec into inner_param.data
    # before each inner loop, then read the result back — preserving Adam state.
    inner_param = params_vec.detach().clone().requires_grad_(True)
    inner_opt = torch.optim.Adam([inner_param], lr=cfg.bb_alpha0_icnn)
    return ICNNState(model=icnn_model, params_vec=params_vec, meta=meta,
                     bb_state=bb_state, inner_param=inner_param, inner_opt=inner_opt)


def evaluate_clean(state: TrainState, dataset, batch_size: int, device: torch.device) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = state.model
    # Bug 2 fix: save and restore training mode so that a mid-epoch call does
    # not silently leave the model in eval mode for the rest of training.
    was_training = model.training
    model.eval()
    total_loss = total_acc = total_n = 0.0
    try:
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
    finally:
        model.train(was_training)
    return {"loss": total_loss / total_n, "acc": total_acc / total_n}


def evaluate_mnist_c(model: nn.Module, device: torch.device, root: str = "./data", batch_size: int = 256) -> Dict[str, Any]:
    """Evaluate a model on every MNIST-C corruption and return per-corruption accuracy."""
    ensure_mnist_c_downloaded(root)
    # Bug 2 fix: save and restore training mode.
    was_training = model.training
    model.eval()
    results: Dict[str, float] = {}

    try:
        with torch.no_grad():
            for corruption in CORRUPTIONS:
                dataset = MNISTCDataset(root=root, corruption=corruption, train=False)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                correct = 0
                total = 0
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    logits = model(data)
                    correct += (logits.argmax(dim=1) == target).sum().item()
                    total += target.size(0)
                results[corruption] = round(100.0 * correct / total, 2)
    finally:
        model.train(was_training)

    ood_accs = [acc for corr, acc in results.items() if corr != 'identity']
    results["avg_ood"] = round(sum(ood_accs) / len(ood_accs), 2)
    return results


def train_algorithm_1(cfg: TrainConfig, device: torch.device) -> Tuple[TrainState, Dict[str, Any]]:
    """WRM adversarial training (Algorithm 1).

    Checkpoint-selection and early-stopping changes vs. the original:
      - Fix 2/3: validation split held out; never used for gradient updates.
      - Fix 3:   checkpoints scored by PGD-L2 on val split, not by the
                 training adversary on training data.
      - Fix 4:   score = α·val_clean + (1−α)·val_pgd  (α = cfg.es_clean_weight,
                 default 0); no longer an ad-hoc unweighted sum.
      - Fix 5:   clean warm-up checkpoints are also saved.
      - Fix 6:   per-epoch metrics are sample-weighted (not batch-count means).
      - Fix 1:   genuine early stopping: training breaks when patience is
                 exhausted (cfg.es_patience > 0).
    """
    seed_everything(cfg.seed)
    train_raw_ds, test_ds = load_mnist()
    train_ds, val_ds = split_train_val(train_raw_ds, cfg.es_val_frac, cfg.seed)   # Fix 2
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    state  = create_classifier_state(cfg, device)
    logger = CSVLogger(DEFAULT_LOG_PATH, LOG_FIELDNAMES)
    training_logs = []
    cm_epochs     = []

    best_score    = -float("inf")
    best_epoch    = -1
    best_model_sd = None
    best_opt_sd   = None
    patience_count = 0          # Fix 1: patience counter
    in_adv_phase   = False      # resets patience at the clean→adv transition

    for epoch in range(cfg.num_epochs):
        is_clean  = epoch < cfg.epoch_clean
        num_steps = 0
        epoch_n   = 0

        # Detect first adversarial epoch; reset patience cleanly.
        if not is_clean and not in_adv_phase:
            in_adv_phase   = True
            patience_count = 0

        if is_clean:
            # Fix 6: accumulate sample-weighted sums, not batch-count sums.
            epoch_loss_sum      = 0.0
            epoch_acc_clean_sum = 0.0
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_algo1 is not None and step >= cfg.max_steps_algo1:
                    break
                n = x.size(0)
                state, metrics = train_step_clean(state, (x, y), device)
                epoch_loss_sum      += float(metrics['loss'])      * n
                epoch_acc_clean_sum += float(metrics['acc_clean']) * n
                epoch_n   += n
                num_steps += 1
            if num_steps > 0:
                avg_loss      = epoch_loss_sum      / epoch_n
                avg_acc_clean = epoch_acc_clean_sum / epoch_n
                print(f"[Algo1] Epoch {epoch} (clean) loss={avg_loss:.4f} acc_clean={avg_acc_clean:.4f}")
                logger.log(algorithm="algo1", phase="train_clean", epoch=epoch, step=num_steps,
                           loss_adv=avg_loss, adv_loss=None, cls_loss=None,
                           acc_clean=avg_acc_clean, acc_adv=None, w2_proxy=None)
                training_logs.append({"epoch": epoch, "phase": "clean", "steps": num_steps,
                                       "loss": avg_loss, "acc_clean": avg_acc_clean})
                # Fix 5: score clean-phase checkpoints on val clean accuracy.
                score, val_c, _ = compute_val_score(state, val_ds, cfg, device, is_clean_phase=True)
                print(f"[Algo1] Epoch {epoch}  val_clean={val_c:.4f}  score={score:.4f}")
                if score > best_score:
                    best_score    = score
                    best_epoch    = epoch
                    best_model_sd = copy.deepcopy(state.model.state_dict())
                    best_opt_sd   = copy.deepcopy(state.opt.state_dict())
            state.scheduler.step()

        else:
            # Fix 6: accumulate sample-weighted sums.
            epoch_loss_adv_sum  = 0.0
            epoch_acc_clean_sum = 0.0
            epoch_acc_adv_sum   = 0.0
            epoch_w2_sum        = 0.0
            epoch_cm_batch_results = []
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_algo1 is not None and step >= cfg.max_steps_algo1:
                    break
                n = x.size(0)
                state, metrics = train_step_algo1(state, (x, y), cfg, device)
                epoch_loss_adv_sum  += float(metrics['loss_adv'])  * n
                epoch_acc_clean_sum += float(metrics['acc_clean']) * n
                epoch_acc_adv_sum   += float(metrics['acc_adv'])   * n
                epoch_w2_sum        += float(metrics['w2_proxy'])  * n
                if cfg.cm_diagnostics and 'cm_diagnostics' in metrics:
                    epoch_cm_batch_results.append(metrics['cm_diagnostics'])
                epoch_n   += n
                num_steps += 1
            if num_steps > 0:
                avg_loss_adv  = epoch_loss_adv_sum  / epoch_n
                avg_acc_clean = epoch_acc_clean_sum / epoch_n
                avg_acc_adv   = epoch_acc_adv_sum   / epoch_n
                avg_w2_proxy  = epoch_w2_sum        / epoch_n
                print(f"[Algo1] Epoch {epoch} (adv) loss_adv={avg_loss_adv:.4f}"
                      f" acc_clean={avg_acc_clean:.4f} acc_adv={avg_acc_adv:.4f}"
                      f" W2≈{avg_w2_proxy:.4f}")
                if cfg.cm_diagnostics and epoch_cm_batch_results:
                    epoch_cm_agg = aggregate_cm_results(epoch_cm_batch_results)
                    cm_epochs.append({"epoch": epoch, "cm": {str(k): v for k, v in epoch_cm_agg.items()}})
                    cm_parts = [
                        f"k={k}: viol={epoch_cm_agg[k]['frac_violated']:.1%}"
                        f" pe={epoch_cm_agg[k]['mean_per_edge']:.4f}"
                        f" rel={epoch_cm_agg[k]['mean_relative']:.4f}"
                        for k in sorted(epoch_cm_agg.keys())
                    ]
                    print(f"        CM diagnostic: {' | '.join(cm_parts)}")
                logger.log(algorithm="algo1", phase="train_adv", epoch=epoch, step=num_steps,
                           loss_adv=avg_loss_adv, adv_loss=None, cls_loss=None,
                           acc_clean=avg_acc_clean, acc_adv=avg_acc_adv, w2_proxy=avg_w2_proxy)
                training_logs.append({"epoch": epoch, "phase": "adv", "steps": num_steps,
                                       "loss_adv": avg_loss_adv, "acc_clean": avg_acc_clean,
                                       "acc_adv": avg_acc_adv, "w2_proxy": avg_w2_proxy})

                # Fix 2/3/4: score on held-out val split using PGD.
                score, val_c, val_p = compute_val_score(
                    state, val_ds, cfg, device, is_clean_phase=False)
                print(f"[Algo1] Epoch {epoch}  val_clean={val_c:.4f}"
                      f"  val_pgd={val_p:.4f}  score={score:.4f}")

                if score > best_score:
                    best_score    = score
                    best_epoch    = epoch
                    best_model_sd = copy.deepcopy(state.model.state_dict())
                    best_opt_sd   = copy.deepcopy(state.opt.state_dict())
                    patience_count = 0
                else:
                    patience_count += 1

                state.scheduler.step()

                # Fix 1: break when patience is exhausted.
                if cfg.es_patience > 0 and patience_count >= cfg.es_patience:
                    print(f"[Algo1] Early stopping at epoch {epoch}"
                          f" ({patience_count} adversarial epochs without improvement).")
                    break

    if best_model_sd is not None:
        state.model.load_state_dict(best_model_sd)
        state.opt.load_state_dict(best_opt_sd)
        print(f"[Algo1] Restored best epoch {best_epoch} (score={best_score:.4f})")

    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print("[Algo1] Test:", test_metrics)
    logger.log(algorithm="algo1", phase="test", epoch=cfg.num_epochs, step=None,
               loss_adv=float(test_metrics["loss"]), adv_loss=None, cls_loss=None,
               acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None)

    if cm_epochs:
        all_batch_cm = [e["cm"] for e in cm_epochs]
        all_k = sorted({int(k) for d in all_batch_cm for k in d})
        print("\n" + "=" * 72)
        print("[Algo1] Cyclical Monotonicity Summary (averaged over adv epochs)")
        print("=" * 72)
        print(f"  {'k':>4s}  {'Frac Violated':>14s}  {'Mean/Edge':>10s}"
              f"  {'Std/Edge':>10s}  {'Mean Rel':>10s}  {'Max Rel':>10s}")
        print("-" * 72)
        for k in all_k:
            entries = [d[str(k)] for d in all_batch_cm if str(k) in d]
            n   = len(entries)
            fv  = sum(e["frac_violated"]  for e in entries) / n
            mpe = sum(e["mean_per_edge"]  for e in entries) / n
            spe = sum(e["std_per_edge"]   for e in entries) / n
            mrl = sum(e["mean_relative"]  for e in entries) / n
            xrl = max(e["max_relative"]   for e in entries)
            print(f"  {k:4d}  {fv:14.2%}  {mpe:10.6f}  {spe:10.6f}"
                  f"  {mrl:10.6f}  {xrl:10.6f}")
        print("=" * 72 + "\n")

    results = {"algorithm": "algo1_wrm", "hyperparameters": asdict(cfg),
               "training_logs": training_logs, "test_metrics": test_metrics}
    if cm_epochs:
        results["cyclical_monotonicity"] = cm_epochs
    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", "algo1_wrm_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    if cm_epochs:
        with open(os.path.join("MNIST", "algo1_cm_diagnostics.json"), "w") as f:
            json.dump({"algorithm": "algo1_wrm", "cm_per_epoch": cm_epochs}, f, indent=2)
    return state, {"test": test_metrics}


def train_algorithm_2(cfg: TrainConfig, device: torch.device) -> Tuple[TrainState, ICNNState, Dict[str, Any]]:
    """ICNN transport adversarial training (Algorithm 2).

    Checkpoint-selection / early-stopping uses a held-out val split scored
    by PGD-L2.  See train_algorithm_1 docstring and TrainConfig.es_* fields.
    """
    seed_everything(cfg.seed)
    train_raw_ds, test_ds = load_mnist()
    train_ds, val_ds = split_train_val(train_raw_ds, cfg.es_val_frac, cfg.seed)   # Fix 2
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    state     = create_classifier_state(cfg, device)
    logger    = CSVLogger(DEFAULT_LOG_PATH, LOG_FIELDNAMES)
    training_logs = []
    input_dim = 28 * 28 * 1
    icnn_state = create_icnn_state(cfg, input_dim, device)

    best_score             = -float("inf")
    best_epoch             = -1
    best_model_sd          = None
    best_opt_sd            = None
    best_icnn_sd           = None
    best_icnn_params_vec   = None
    best_icnn_inner_param  = None
    best_icnn_inner_opt_sd = None
    patience_count = 0          # Fix 1
    in_adv_phase   = False

    def _save_best():
        return (
            copy.deepcopy(state.model.state_dict()),
            copy.deepcopy(state.opt.state_dict()),
            copy.deepcopy(icnn_state.model.state_dict()),
            icnn_state.params_vec.detach().clone(),
            icnn_state.inner_param.detach().clone(),
            copy.deepcopy(icnn_state.inner_opt.state_dict()),
        )

    for epoch in range(cfg.num_epochs):
        is_clean  = epoch < cfg.epoch_clean
        num_steps = 0
        epoch_n   = 0

        if not is_clean and not in_adv_phase:
            in_adv_phase   = True
            patience_count = 0

        if is_clean:
            # Fix 6: sample-weighted sums.
            epoch_loss_sum      = 0.0
            epoch_acc_clean_sum = 0.0
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_algo2 is not None and step >= cfg.max_steps_algo2:
                    break
                n = x.size(0)
                state, metrics = train_step_clean(state, (x, y), device)
                epoch_loss_sum      += float(metrics['loss'])      * n
                epoch_acc_clean_sum += float(metrics['acc_clean']) * n
                epoch_n   += n
                num_steps += 1
            if num_steps > 0:
                avg_loss      = epoch_loss_sum      / epoch_n
                avg_acc_clean = epoch_acc_clean_sum / epoch_n
                print(f"[Algo2] Epoch {epoch} (clean) loss={avg_loss:.4f} acc_clean={avg_acc_clean:.4f}")
                logger.log(algorithm="algo2", phase="train_clean", epoch=epoch, step=num_steps,
                           loss_adv=None, adv_loss=None, cls_loss=avg_loss,
                           acc_clean=avg_acc_clean, acc_adv=None, w2_proxy=None)
                training_logs.append({"epoch": epoch, "phase": "clean", "steps": num_steps,
                                       "loss": avg_loss, "acc_clean": avg_acc_clean})
                # Fix 5: checkpoint clean-phase models.
                score, val_c, _ = compute_val_score(state, val_ds, cfg, device, is_clean_phase=True)
                print(f"[Algo2] Epoch {epoch}  val_clean={val_c:.4f}  score={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    (best_model_sd, best_opt_sd, best_icnn_sd,
                     best_icnn_params_vec, best_icnn_inner_param,
                     best_icnn_inner_opt_sd) = _save_best()
            state.scheduler.step()

        else:
            # Fix 6: sample-weighted sums.
            epoch_adv_loss_sum   = 0.0
            epoch_cls_loss_sum   = 0.0
            epoch_acc_clean_sum  = 0.0
            epoch_acc_adv_sum    = 0.0
            epoch_w2_sum         = 0.0
            epoch_inner_grad_sum = 0.0
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_algo2 is not None and step >= cfg.max_steps_algo2:
                    break
                n = x.size(0)
                state, icnn_state, metrics = train_step_algo2(state, icnn_state, (x, y), cfg, device)
                epoch_adv_loss_sum   += float(metrics['adv_loss'])        * n
                epoch_cls_loss_sum   += float(metrics['cls_loss'])        * n
                epoch_acc_clean_sum  += float(metrics['acc_clean'])       * n
                epoch_acc_adv_sum    += float(metrics['acc_adv'])         * n
                epoch_w2_sum         += float(metrics['w2_proxy'])        * n
                epoch_inner_grad_sum += float(metrics['inner_grad_norm']) * n
                epoch_n   += n
                num_steps += 1
            if num_steps > 0:
                avg_adv_loss        = epoch_adv_loss_sum   / epoch_n
                avg_cls_loss        = epoch_cls_loss_sum   / epoch_n
                avg_acc_clean       = epoch_acc_clean_sum  / epoch_n
                avg_acc_adv         = epoch_acc_adv_sum    / epoch_n
                avg_w2_proxy        = epoch_w2_sum         / epoch_n
                avg_inner_grad_norm = epoch_inner_grad_sum / epoch_n
                print(f"[Algo2] Epoch {epoch} (adv) adv_loss={avg_adv_loss:.4f}"
                      f" cls_loss={avg_cls_loss:.4f} acc_clean={avg_acc_clean:.4f}"
                      f" acc_adv={avg_acc_adv:.4f} W2≈{avg_w2_proxy:.4f}"
                      f" |∇θ|={avg_inner_grad_norm:.6f}")
                logger.log(algorithm="algo2", phase="train_adv", epoch=epoch, step=num_steps,
                           loss_adv=None, adv_loss=avg_adv_loss, cls_loss=avg_cls_loss,
                           acc_clean=avg_acc_clean, acc_adv=avg_acc_adv,
                           w2_proxy=avg_w2_proxy, inner_grad_norm=avg_inner_grad_norm)
                training_logs.append({"epoch": epoch, "phase": "adv", "steps": num_steps,
                                       "adv_loss": avg_adv_loss, "cls_loss": avg_cls_loss,
                                       "acc_clean": avg_acc_clean, "acc_adv": avg_acc_adv,
                                       "w2_proxy": avg_w2_proxy,
                                       "inner_grad_norm": avg_inner_grad_norm})

                # Fix 2/3/4: val PGD score.
                score, val_c, val_p = compute_val_score(
                    state, val_ds, cfg, device, is_clean_phase=False)
                print(f"[Algo2] Epoch {epoch}  val_clean={val_c:.4f}"
                      f"  val_pgd={val_p:.4f}  score={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    (best_model_sd, best_opt_sd, best_icnn_sd,
                     best_icnn_params_vec, best_icnn_inner_param,
                     best_icnn_inner_opt_sd) = _save_best()
                    patience_count = 0
                else:
                    patience_count += 1

                state.scheduler.step()

                # Fix 1: actual early stopping.
                if cfg.es_patience > 0 and patience_count >= cfg.es_patience:
                    print(f"[Algo2] Early stopping at epoch {epoch}"
                          f" ({patience_count} adversarial epochs without improvement).")
                    break

    if best_model_sd is not None:
        state.model.load_state_dict(best_model_sd)
        state.opt.load_state_dict(best_opt_sd)
        icnn_state.model.load_state_dict(best_icnn_sd)
        icnn_state.params_vec = best_icnn_params_vec
        icnn_state.inner_param.data.copy_(best_icnn_inner_param)
        icnn_state.inner_opt.load_state_dict(best_icnn_inner_opt_sd)
        print(f"[Algo2] Restored best epoch {best_epoch} (score={best_score:.4f})")

    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print("[Algo2] Test:", test_metrics)
    logger.log(algorithm="algo2", phase="test", epoch=cfg.num_epochs, step=None,
               loss_adv=float(test_metrics["loss"]), adv_loss=None,
               cls_loss=float(test_metrics["loss"]),
               acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None)
    results = {"algorithm": "algo2_icnn", "hyperparameters": asdict(cfg),
               "training_logs": training_logs, "test_metrics": test_metrics}
    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", "algo2_icnn_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return state, icnn_state, {"test": test_metrics}


def train_algorithm_ppa(cfg: TrainConfig, device: torch.device) -> Tuple[TrainState, Dict[str, Any]]:
    """Algorithm 3: Projected Particle Ascent (PPA).

    Same outer structure as Algo1, but the inner adversary uses iterative
    (WRM ascent → Brenier projection) cycles.  Each projection eliminates
    wasted transport (Delta >= 0), yielding a provably stronger adversary
    per Lemma (proj_gain) in the paper.

    Checkpoint-selection / early-stopping uses a held-out val split scored
    by PGD-L2.  See train_algorithm_1 docstring and TrainConfig.es_* fields.
    """
    seed_everything(cfg.seed)
    train_raw_ds, test_ds = load_mnist()
    train_ds, val_ds = split_train_val(train_raw_ds, cfg.es_val_frac, cfg.seed)   # Fix 2
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    state  = create_classifier_state(cfg, device)
    logger = CSVLogger(DEFAULT_LOG_PATH, LOG_FIELDNAMES)
    training_logs = []

    best_score    = -float("inf")
    best_epoch    = -1
    best_model_sd = None
    best_opt_sd   = None
    patience_count = 0          # Fix 1
    in_adv_phase   = False

    for epoch in range(cfg.num_epochs):
        is_clean  = epoch < cfg.epoch_clean
        num_steps = 0
        epoch_n   = 0

        if not is_clean and not in_adv_phase:
            in_adv_phase   = True
            patience_count = 0

        if is_clean:
            # Fix 6: sample-weighted sums.
            epoch_loss_sum      = 0.0
            epoch_acc_clean_sum = 0.0
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_ppa is not None and step >= cfg.max_steps_ppa:
                    break
                n = x.size(0)
                state, metrics = train_step_clean(state, (x, y), device)
                epoch_loss_sum      += float(metrics['loss'])      * n
                epoch_acc_clean_sum += float(metrics['acc_clean']) * n
                epoch_n   += n
                num_steps += 1
            if num_steps > 0:
                avg_loss      = epoch_loss_sum      / epoch_n
                avg_acc_clean = epoch_acc_clean_sum / epoch_n
                print(f"[PPA] Epoch {epoch} (clean) loss={avg_loss:.4f} acc_clean={avg_acc_clean:.4f}")
                logger.log(algorithm="ppa", phase="train_clean", epoch=epoch, step=num_steps,
                           loss_adv=avg_loss, adv_loss=None, cls_loss=None,
                           acc_clean=avg_acc_clean, acc_adv=None, w2_proxy=None)
                training_logs.append({"epoch": epoch, "phase": "clean", "steps": num_steps,
                                       "loss": avg_loss, "acc_clean": avg_acc_clean})
                # Fix 5: checkpoint clean-phase models.
                score, val_c, _ = compute_val_score(state, val_ds, cfg, device, is_clean_phase=True)
                print(f"[PPA]   Epoch {epoch}  val_clean={val_c:.4f}  score={score:.4f}")
                if score > best_score:
                    best_score    = score
                    best_epoch    = epoch
                    best_model_sd = copy.deepcopy(state.model.state_dict())
                    best_opt_sd   = copy.deepcopy(state.opt.state_dict())
            state.scheduler.step()

        else:
            # Fix 6: sample-weighted sums.
            epoch_loss_adv_sum  = 0.0
            epoch_acc_clean_sum = 0.0
            epoch_acc_adv_sum   = 0.0
            epoch_w2_sum        = 0.0
            epoch_delta_sum     = 0.0
            for step, (x, y) in enumerate(train_loader):
                if cfg.max_steps_ppa is not None and step >= cfg.max_steps_ppa:
                    break
                n = x.size(0)
                state, metrics = train_step_ppa(state, (x, y), cfg, device)
                epoch_loss_adv_sum  += float(metrics['loss_adv'])  * n
                epoch_acc_clean_sum += float(metrics['acc_clean']) * n
                epoch_acc_adv_sum   += float(metrics['acc_adv'])   * n
                epoch_w2_sum        += float(metrics['w2_proxy'])  * n
                epoch_delta_sum     += float(metrics['delta_gap']) * n
                epoch_n   += n
                num_steps += 1
            if num_steps > 0:
                avg_loss_adv  = epoch_loss_adv_sum  / epoch_n
                avg_acc_clean = epoch_acc_clean_sum / epoch_n
                avg_acc_adv   = epoch_acc_adv_sum   / epoch_n
                avg_w2_proxy  = epoch_w2_sum        / epoch_n
                avg_delta_gap = epoch_delta_sum     / epoch_n
                print(f"[PPA] Epoch {epoch} (adv) loss_adv={avg_loss_adv:.4f}"
                      f" acc_clean={avg_acc_clean:.4f} acc_adv={avg_acc_adv:.4f}"
                      f" W2≈{avg_w2_proxy:.4f} Δ={avg_delta_gap:.4f}")
                logger.log(algorithm="ppa", phase="train_adv", epoch=epoch, step=num_steps,
                           loss_adv=avg_loss_adv, adv_loss=None, cls_loss=None,
                           acc_clean=avg_acc_clean, acc_adv=avg_acc_adv,
                           w2_proxy=avg_w2_proxy, delta_gap=avg_delta_gap)
                training_logs.append({"epoch": epoch, "phase": "adv", "steps": num_steps,
                                       "loss_adv": avg_loss_adv, "acc_clean": avg_acc_clean,
                                       "acc_adv": avg_acc_adv, "w2_proxy": avg_w2_proxy,
                                       "delta_gap": avg_delta_gap})

                # Fix 2/3/4: val PGD score.
                score, val_c, val_p = compute_val_score(
                    state, val_ds, cfg, device, is_clean_phase=False)
                print(f"[PPA]   Epoch {epoch}  val_clean={val_c:.4f}"
                      f"  val_pgd={val_p:.4f}  score={score:.4f}")

                if score > best_score:
                    best_score    = score
                    best_epoch    = epoch
                    best_model_sd = copy.deepcopy(state.model.state_dict())
                    best_opt_sd   = copy.deepcopy(state.opt.state_dict())
                    patience_count = 0
                else:
                    patience_count += 1

                state.scheduler.step()

                # Fix 1: actual early stopping.
                if cfg.es_patience > 0 and patience_count >= cfg.es_patience:
                    print(f"[PPA]   Early stopping at epoch {epoch}"
                          f" ({patience_count} adversarial epochs without improvement).")
                    break

    if best_model_sd is not None:
        state.model.load_state_dict(best_model_sd)
        state.opt.load_state_dict(best_opt_sd)
        print(f"[PPA] Restored best epoch {best_epoch} (score={best_score:.4f})")

    test_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
    print("[PPA] Test:", test_metrics)
    logger.log(algorithm="ppa", phase="test", epoch=cfg.num_epochs, step=None,
               loss_adv=float(test_metrics["loss"]), adv_loss=None, cls_loss=None,
               acc_clean=float(test_metrics["acc"]), acc_adv=None, w2_proxy=None)
    results = {"algorithm": "ppa_projected_wrm", "hyperparameters": asdict(cfg),
               "training_logs": training_logs, "test_metrics": test_metrics}
    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", "ppa_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return state, {"test": test_metrics}


# ---------------------------------------------------------------------------
#  Checkpoint saving
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = os.path.join("MNIST_checkpoint")


def _lambda_str(lam: float) -> str:
    """Format lambda for filenames: 5.0 -> '5.0', 0.01 -> '0.01'."""
    return f"{lam:g}"


def save_checkpoint_algo1(state: TrainState, cfg: TrainConfig) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    fname = f"algo1_wrm_lambda{_lambda_str(cfg.lambda_reg)}.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)
    torch.save({
        "algorithm": "algo1_wrm",
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.opt.state_dict(),
        "lambda_reg": cfg.lambda_reg,
        "hyperparameters": asdict(cfg),
    }, path)
    print(f"[Algo1] Checkpoint saved to {path}")
    return path


def save_checkpoint_algo2(state: TrainState, icnn_state: ICNNState, cfg: TrainConfig) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    hidden_str = "_".join(str(h) for h in cfg.icnn_hidden_sizes)
    fname = f"algo2_icnn_lambda{_lambda_str(cfg.lambda_reg)}_hidden{hidden_str}.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)
    torch.save({
        "algorithm": "algo2_icnn",
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.opt.state_dict(),
        "icnn_model_state_dict": icnn_state.model.state_dict(),
        "icnn_params_vec": icnn_state.params_vec,
        # Bug 3 fix: persist Adam's moment buffers (m̂_t, v̂_t) and the
        # persistent leaf tensor so that resumption continues from the
        # correct optimizer state rather than resetting to t=0.
        "icnn_inner_param": icnn_state.inner_param.detach().clone(),
        "icnn_inner_opt_state_dict": icnn_state.inner_opt.state_dict(),
        "lambda_reg": cfg.lambda_reg,
        "icnn_hidden_sizes": list(cfg.icnn_hidden_sizes),
        "hyperparameters": asdict(cfg),
    }, path)
    print(f"[Algo2] Checkpoint saved to {path}")
    return path


def save_checkpoint_ppa(state: TrainState, cfg: TrainConfig) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    fname = f"ppa_lambda{_lambda_str(cfg.lambda_reg)}.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)
    torch.save({
        "algorithm": "ppa_projected_wrm",
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.opt.state_dict(),
        "lambda_reg": cfg.lambda_reg,
        "hyperparameters": asdict(cfg),
    }, path)
    print(f"[PPA]   Checkpoint saved to {path}")
    return path


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST WRM/ICNN/PPA adversarial training")
    parser.add_argument("--cm-diagnostics", action="store_true",
                        help="Enable cyclical monotonicity diagnostics for Algorithm 1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(
        batch_size=512,
        num_epochs=50,
        lr_cls=1e-2,
        lambda_reg=0.05,
        epoch_clean=0,
        max_steps_algo1=None,
        max_steps_algo2=None,
        use_margin_adv_algo1=False,
        use_margin_adv_algo2=False,
        inner_steps_algo1=75,
        inner_steps_algo2=20,    
        inner_lr_algo1=1e-2,
        bb_alpha0_icnn=1e-2,    
        icnn_hidden_sizes=(512, 512, 256, 256, 128),
        # PPA parameters (enhanced — provably dominates Algo1)
        inner_steps_ppa_round0=75,   # == inner_steps_algo1 (dominance condition)
        inner_lr_ppa_round0=1e-2,    # == inner_lr_algo1
        ppa_num_rounds=5,            # 1 base round + 4 refinement rounds
        ppa_min_rounds=2,            # don't stop before 2 refinement rounds
        ppa_refine_steps=15,         # constant-lr steps per refinement
        ppa_refine_lr=5e-3,          # constant lr for refinement
        ppa_delta_rtol=1e-4,         # relative stopping threshold
        max_steps_ppa=None,
        cm_diagnostics=args.cm_diagnostics,
        seed=0,
    )
    # print("Training Algorithm 2 (ICNN transport with BB+Armijo)...")
    # state_algo2, icnn_state_algo2, logs_algo2 = train_algorithm_2(cfg, device)
    # save_checkpoint_algo2(state_algo2, icnn_state_algo2, cfg)

    print("Training Algorithm 1 (WRM adversarial training)...")
    state_algo1, logs_algo1 = train_algorithm_1(cfg, device)
    save_checkpoint_algo1(state_algo1, cfg)

    print("\nTraining Algorithm 3 (PPA — Projected Particle Ascent)...")
    state_ppa, logs_ppa = train_algorithm_ppa(cfg, device)
    save_checkpoint_ppa(state_ppa, cfg)


    _, test_ds = load_mnist()

    # FIX Bug 2: reseed all RNGs before evaluation so that the random starts
    # inside pgd_l2_attack_restarts (torch.randn_like, torch.rand) are identical
    # across runs.  Without this, the RNG state at eval time is the cumulative
    # result of all stochastic operations during the three training runs, which
    # varies with hardware, batch ordering, and any GPU non-determinism.
    seed_everything(cfg.seed)

    pgd_kwargs = dict(batch_size=cfg.batch_size, eps=2.0, num_steps=40, restarts=5, device=device)
    # Bug 5 fix: eps=2.0 and num_steps=40 now match the top of the sweep range
    # (max sweep eps = 2.0, sweep num_steps = 40), making the single-epsilon
    # report directly comparable to the corresponding sweep entry.
    # The previous eps=2.5 / num_steps=20 was outside the sweep range and used
    # a different step size (2*2.5/20=0.25 vs 2*2.0/40=0.1), producing numbers
    # that could not be compared to any sweep result.

    pgd_algo1 = evaluate_pgd(state_algo1, test_ds, **pgd_kwargs)
    print(f"[Algo1] PGD acc={pgd_algo1['acc']*100:.2f}% L2={pgd_algo1['avg_l2']:.4f} Linf={pgd_algo1['avg_linf']:.4f}")

    # pgd_algo2 = evaluate_pgd(state_algo2, test_ds, **pgd_kwargs)
    # print(f"[Algo2] PGD acc={pgd_algo2['acc']*100:.2f}% L2={pgd_algo2['avg_l2']:.4f} Linf={pgd_algo2['avg_linf']:.4f}")

    pgd_ppa = evaluate_pgd(state_ppa, test_ds, **pgd_kwargs)
    print(f"[PPA]   PGD acc={pgd_ppa['acc']*100:.2f}% L2={pgd_ppa['avg_l2']:.4f} Linf={pgd_ppa['avg_linf']:.4f}")

    pgd_results = {
        "hyperparameters": asdict(cfg),
        "pgd_config": {"eps": pgd_kwargs["eps"], "num_steps": pgd_kwargs["num_steps"]},
        "algo1_wrm_pgd": pgd_algo1,
        # "algo2_icnn_pgd": pgd_algo2,
        "ppa_pgd": pgd_ppa,
    }
    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", "pgd_evaluation_results.json"), "w") as f:
        json.dump(pgd_results, f, indent=2)

    # ------------------------------------------------------------------
    #  PGD-L2 sweep evaluation (proper restarts & projection)
    # ------------------------------------------------------------------
    pgd_eval_cfg = PGDEvalConfig(
        epsilons=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0),
        num_steps=40,
        step_size=None,  # auto = 2*eps/steps
        restarts=5,
    )

    print("\n[Algo1] PGD-L2 sweep evaluation...")
    pgd_sweep_algo1 = evaluate_pgd_l2_sweep(state_algo1, test_ds, pgd_eval_cfg, cfg.batch_size, device)

    # print("\n[Algo2] PGD-L2 sweep evaluation...")
    # pgd_sweep_algo2 = evaluate_pgd_l2_sweep(state_algo2, test_ds, pgd_eval_cfg, cfg.batch_size, device)

    print("\n[PPA] PGD-L2 sweep evaluation...")
    pgd_sweep_ppa = evaluate_pgd_l2_sweep(state_ppa, test_ds, pgd_eval_cfg, cfg.batch_size, device)

    # Update algo1 JSON with PGD sweep results
    algo1_json_path = os.path.join("MNIST", "algo1_wrm_results.json")
    with open(algo1_json_path, "r") as f:
        algo1_data = json.load(f)
    algo1_data["pgd_l2_sweep"] = pgd_sweep_algo1
    algo1_data["pgd_eval_config"] = asdict(pgd_eval_cfg)
    with open(algo1_json_path, "w") as f:
        json.dump(algo1_data, f, indent=2)

    # Update algo2 JSON with PGD sweep results
    # algo2_json_path = os.path.join("MNIST", "algo2_icnn_results.json")
    # with open(algo2_json_path, "r") as f:
    #     algo2_data = json.load(f)
    # algo2_data["pgd_l2_sweep"] = pgd_sweep_algo2
    # algo2_data["pgd_eval_config"] = asdict(pgd_eval_cfg)
    # with open(algo2_json_path, "w") as f:
    #     json.dump(algo2_data, f, indent=2)

    # Update PPA JSON with PGD sweep results
    ppa_json_path = os.path.join("MNIST", "ppa_results.json")
    with open(ppa_json_path, "r") as f:
        ppa_data = json.load(f)
    ppa_data["pgd_l2_sweep"] = pgd_sweep_ppa
    ppa_data["pgd_eval_config"] = asdict(pgd_eval_cfg)
    with open(ppa_json_path, "w") as f:
        json.dump(ppa_data, f, indent=2)

    # Update PGD evaluation JSON with sweep results
    pgd_results["pgd_l2_sweep_config"] = asdict(pgd_eval_cfg)
    pgd_results["algo1_wrm_pgd_sweep"] = pgd_sweep_algo1
    # pgd_results["algo2_icnn_pgd_sweep"] = pgd_sweep_algo2
    pgd_results["ppa_pgd_sweep"] = pgd_sweep_ppa
    with open(os.path.join("MNIST", "pgd_evaluation_results.json"), "w") as f:
        json.dump(pgd_results, f, indent=2)

    # ------------------------------------------------------------------
    #  MNIST-C corruption robustness evaluation
    # ------------------------------------------------------------------
    # print("\n" + "=" * 72)
    # print("  MNIST-C Corruption Robustness Evaluation")
    # print("=" * 72)

    # mnist_c_results = {}
    # for name, st in [("Algo1 (WRM)", state_algo1), ("Algo2 (ICNN)", state_algo2), ("PPA", state_ppa)]:
    #     print(f"\n[{name}] Evaluating on MNIST-C corruptions...")
    #     res = evaluate_mnist_c(st.model, device, root="data", batch_size=cfg.batch_size)
    #     mnist_c_results[name] = res

    # Print comparison table
    # header_corruptions = CORRUPTIONS + ["avg_ood"]
    # algo_names = list(mnist_c_results.keys())
    # print("\n" + "-" * 72)
    # print(f"  {'Corruption':<20}", end="")
    # for aname in algo_names:
    #     print(f" | {aname:>14}", end="")
    # print("\n" + "-" * 72)
    # for corr in header_corruptions:
    #     label = corr if corr != "avg_ood" else "Avg (OOD only)"
    #     print(f"  {label:<20}", end="")
    #     for aname in algo_names:
    #         val = mnist_c_results[aname].get(corr, 0.0)
    #         print(f" | {val:>13.2f}%", end="")
    #     print()
    # print("-" * 72)

    # # Save MNIST-C results to JSON
    # with open(os.path.join("MNIST", "mnist_c_results.json"), "w") as f:
    #     json.dump(mnist_c_results, f, indent=2)