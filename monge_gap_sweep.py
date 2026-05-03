"""Compute the Monge gap of every trained adversary on every dataset and
record the result in a tidy CSV.

Usage:
    python monge_gap_sweep.py \\
        --dataset least_squares \\
        --methods erm particle_ascent ppa npf wfr dual nn_dro \\
        --seeds 219 220 221 222 223 \\
        --lams 0.01 0.1 1.0 10.0 \\
        --eval_size 4096 \\
        --out monge_gap_ls.csv

For ``least_squares`` the script trains each (method, lam, seed) cell
inline using the existing ``_solve_*_with_zstar`` solvers (the inner
loop is too cheap for an offline checkpoint round-trip to be worth it),
then evaluates the held-out adversary push and computes the Monge gap.

For ``lr_cifar10`` and ``rl_cartpole`` the script loads pre-trained
checkpoints from ``--checkpoint_dir``. The dataset-specific glue lives in
the ``DATASET_BACKENDS`` registry below; see the ``LeastSquaresBackend``
class for the reference implementation.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

# Make sure the per-dataset packages are importable regardless of cwd.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from monge_gap_utils.monge_gap import (
    monge_gap_hungarian,
    monge_gap_sinkhorn,
    monge_gap_subsample_hungarian,
)


# ---------------------------------------------------------------------------
# Dataset backend interface.
#
# A backend implements:
#   - supported_methods() -> tuple[str, ...]
#   - sample_held_out_anchors(eval_size, seed, device) -> z_hat (B, d)
#   - train_or_load(method, lam, seed, eval_size, device, **kwargs) -> state
#   - push(method, state, z_hat) -> Tz (B, d)
#   - compute_test_loss(state, seed, device, delta=...) -> float
#
# Each backend owns its push dispatch. We do not use a single global
# REGISTRY because LR-CIFAR10 and RL operate on different state schemas
# (e.g. RL adversary classes need ``env_eval`` and a trained policy; LS
# operates on plain (theta, A0, A1, b)).
# ---------------------------------------------------------------------------


class LeastSquaresBackend:
    """Inline-training backend for the 1D uncertain-least-squares problem.

    No checkpoints needed: the inner loop is cheap. We replay the existing
    pareto_sweep solvers, then evaluate the held-out push.
    """

    name = "least_squares"

    def __init__(self, args=None):
        # The existing LS code expects to be importable as flat modules
        # (``from config import ...`` etc.) from inside Least_Squares/.
        # Prepend that directory to sys.path so the imports resolve.
        ls_root = ROOT / "Least_Squares"
        if str(ls_root) not in sys.path:
            sys.path.insert(0, str(ls_root))

        from config import ULSConfig
        from utils.common import seed_everything, clip_grad
        from utils.data import generate_training_data, make_problem
        from utils.pareto_metrics import evaluate_test_loss
        from utils.loss import loss_function, loss_grad_theta
        from algorithms.dual import _solve_dual_with_zstar
        from algorithms.erm import solve_erm_closed_form
        from algorithms.npf import _solve_npf_icnn_map_with_zstar
        from algorithms.particle_ascent import _solve_Particle_Ascent_with_zstar
        from algorithms.ppa import _solve_ppa_with_zstar
        from algorithms.wfr import _solve_wfr_with_zstar
        from models.nn_dro import MLPAdversary

        self.ULSConfig = ULSConfig
        self.seed_everything = seed_everything
        self.generate_training_data = generate_training_data
        self.make_problem = make_problem
        self.evaluate_test_loss = evaluate_test_loss
        self._solvers = {
            "particle_ascent": _solve_Particle_Ascent_with_zstar,
            "ppa":             _solve_ppa_with_zstar,
            "npf":             _solve_npf_icnn_map_with_zstar,
            "wfr":             _solve_wfr_with_zstar,
            "dual":            _solve_dual_with_zstar,
        }
        self._solve_erm = solve_erm_closed_form
        # NN-DRO inline trainer that *also* returns the trained adversary.
        self._MLPAdversary = MLPAdversary
        self._loss_function = loss_function
        self._loss_grad_theta = loss_grad_theta
        self._clip_grad = clip_grad

    def supported_methods(self):
        return ("erm", "particle_ascent", "ppa", "npf", "wfr", "dual", "nn_dro")

    def push(self, method: str, state: Dict[str, Any], z_hat: torch.Tensor) -> torch.Tensor:
        from monge_gap_utils.eval_adversary import REGISTRY as LS_REGISTRY
        push_fn = LS_REGISTRY[method]
        Tz = push_fn(state, z_hat)
        if Tz.shape != z_hat.shape:
            Tz = Tz.view_as(z_hat)
        return Tz

    # --- dataset-specific helpers ---

    def sample_held_out_anchors(
        self, eval_size: int, seed: int, device: torch.device
    ) -> torch.Tensor:
        """Held-out reference batch: Unif([-0.5, 0.5]), per-seed RNG."""
        g = torch.Generator(device=device).manual_seed(int(seed))
        return (torch.rand(eval_size, generator=g, device=device,
                           dtype=torch.float64) - 0.5)

    def _train_nn_dro(self, xi_train, cfg, A0, A1, b):
        """NN-DRO inline trainer that also returns the trained adversary
        (the upstream ``solve_nn_dro`` discards it; we need it here)."""
        device = xi_train.device
        theta = torch.zeros(cfg.dim_n, device=device)
        adversary = self._MLPAdversary(
            input_dim=1,
            hidden_sizes=tuple(cfg.nn_dro_hidden),
            activation=cfg.nn_dro_activation,
            softplus_beta=cfg.nn_dro_softplus_beta,
            init_scale=cfg.nn_dro_init_scale,
        ).to(device).to(xi_train.dtype)
        inner_opt = torch.optim.Adam(adversary.parameters(), lr=cfg.nn_dro_inner_lr)
        xi_2d = xi_train.view(-1, 1)
        for _ in range(cfg.epochs):
            for _ in range(cfg.nn_dro_omega_steps_per_epoch):
                inner_opt.zero_grad()
                z_adv = adversary(xi_2d).view(-1)
                f = self._loss_function(theta, z_adv, A0, A1, b, cfg.dim_m)
                reg = cfg.lam * (z_adv - xi_train) ** 2
                obj = (f - reg).mean()
                (-obj).backward()
                inner_opt.step()
            with torch.no_grad():
                z_adv = adversary(xi_2d).view(-1).clamp(-1.0, 1.0)
                grad_theta = self._loss_grad_theta(theta, z_adv, A0, A1, b).mean(dim=0)
                grad_theta = self._clip_grad(grad_theta, cfg.grad_clip)
                theta = theta - cfg.lr_theta * grad_theta
        return {"theta": theta, "adversary": adversary}

    def train_or_load(
        self,
        method: str,
        lam: float,
        seed: int,
        eval_size: int,
        device: torch.device,
        checkpoint_dir: Path | None,
    ) -> Dict[str, Any]:
        cfg = replace(self.ULSConfig(), lam=lam, seed=seed)
        self.seed_everything(cfg.seed)
        A0, A1, b = self.make_problem(cfg, device)
        xi_train = self.generate_training_data(cfg, device)

        common = {
            "method": method,
            "cfg":    cfg,
            "A0":     A0,
            "A1":     A1,
            "b":      b,
        }

        if method == "erm":
            theta = self._solve_erm(xi_train, cfg, A0, A1, b)
            return {**common, "theta": theta}
        if method == "nn_dro":
            out = self._train_nn_dro(xi_train, cfg, A0, A1, b)
            return {**common, "theta": out["theta"], "adversary": out["adversary"]}
        if method in self._solvers:
            out = self._solvers[method](xi_train, cfg, A0, A1, b)
            state = {**common, "theta": out["theta"]}
            if "psi" in out:
                state["psi"] = out["psi"]
            return state
        raise ValueError(f"[least_squares] Unknown method {method!r}")

    def compute_test_loss(
        self,
        state: Dict[str, Any],
        seed: int,
        device: torch.device,
        delta: float = 1.0,
    ) -> float:
        cfg = state["cfg"]
        return float(self.evaluate_test_loss(
            state["theta"], state["A0"], state["A1"], state["b"],
            delta=delta, dim_m=cfg.dim_m,
            n_test=cfg.n_test, seed=int(seed),
        ))


def _import_lr_cifar10_modules():
    """Push Logistic_Regression_CIFAR10/ onto sys.path, importing in flat form."""
    lr_root = ROOT / "Logistic_Regression_CIFAR10"
    if str(lr_root) not in sys.path:
        sys.path.insert(0, str(lr_root))


def _import_rl_modules():
    """Push RL/ onto sys.path, importing in flat form."""
    rl_root = ROOT / "RL"
    if str(rl_root) not in sys.path:
        sys.path.insert(0, str(rl_root))


# ---------------------------------------------------------------------------
# LR-CIFAR10 backend.
# ---------------------------------------------------------------------------


class LRCifar10Backend:
    """Loads existing LR-CIFAR10 checkpoints and evaluates the Monge gap on
    a held-out subset of the cached ResNet-50 features.

    The LR-CIFAR10 pipeline does NOT encode (lam, seed) into the checkpoint
    filename. Instead, each ``--out_dir`` corresponds to a single (tau, seed)
    training run, with checkpoints named ``<METHOD>_run0_epoch_{epoch}.pth``
    inside ``<out_dir>/checkpoints/``. Pass ``--checkpoint_dir`` pointing at
    that ``checkpoints/`` directory. Optionally use ``{lam}``/``{seed}``
    placeholders in ``--checkpoint_dir`` to map sweep cells to per-cell
    training runs.
    """

    name = "lr_cifar10"

    # Maps the sweep's codebase keys to the file prefix used by the
    # LR-CIFAR10 pipeline (CIFAR10_LogReg.py / algorithms/registry.py).
    _CKPT_PREFIX = {
        "erm":             "SAA",
        "particle_ascent": "WRM",      # WRM = Sinha-style ascent ≈ PA
        "ppa":             "PPA",
        "npf":             "NPF",
        "wfr":             "SDRO_WFR",
        "dual":            "SDRO_Dual",
        "nn_dro":          "NN_DRO",
    }

    def __init__(self, args):
        _import_lr_cifar10_modules()
        from config import TrainConfig  # type: ignore

        self.TrainConfig = TrainConfig
        self.input_dim = 512
        self.num_classes = 10

        self._feature_cache_path = Path(
            args.feature_cache or
            (ROOT / "Logistic_Regression_CIFAR10/data/cifar10_resnet50_features.pth")
        )
        cache = torch.load(self._feature_cache_path, map_location="cpu", weights_only=False)
        # Cache is a dict with train/test feature/label tensors.
        self._test_features = cache["test_features"].float()
        self._test_labels = cache["test_labels"].long()
        # Monge-gap reference pool = train + test (60K for CIFAR-10). The PGD
        # test-loss eval below still uses test-only — the union pool is only
        # the source distribution P̂ for the Monge gap M(T) estimator, where
        # a larger sample tightens the W₂(P̂, T#P̂) approximation.
        pool_pieces, label_pieces = [], []
        if "train_features" in cache:
            pool_pieces.append(cache["train_features"].float())
            label_pieces.append(cache["train_labels"].long())
        pool_pieces.append(self._test_features)
        label_pieces.append(self._test_labels)
        self._pool_features = torch.cat(pool_pieces, dim=0).contiguous()
        self._pool_labels = torch.cat(label_pieces, dim=0).contiguous()
        self._eval_size_max = int(self._pool_features.shape[0])

        # Push chunk size. PPA's Brenier projection runs per-class scipy LAPs
        # whose cost scales as O((N/10)^3); at N=60K an unchunked call takes
        # tens of minutes, while at N≤8K each call is sub-second. Chunking
        # also matches PPA's training-time per-minibatch projection (the
        # CIFAR-10 trainer uses batch_size=128–256). Output is concatenated
        # without re-projection across chunks, so the global Monge-gap
        # estimator is still computed on the full B push.
        self.push_chunk = int(getattr(args, "lr_cifar10_push_chunk", 0) or 0)

        # Inner-loop hyperparameters used by PPA / WFR / Dual reruns.
        # These default to the values in ``Logistic_Regression_CIFAR10/config.py``;
        # the user can override via --lr_cifar10_inner_*.
        cfg_default = TrainConfig()

        def _val(arg_name, default):
            v = getattr(args, arg_name, None)
            return default if v is None else v

        self.inner_lr = float(_val("lr_cifar10_inner_lr", cfg_default.inner_lr))
        self.inner_steps = int(_val("lr_cifar10_inner_steps", cfg_default.inner_steps))
        # ``cfg_default.eps_ent`` may be a tuple (the canonical type annotation)
        # or a single float (e.g. ``= (0.2)`` was meant to be a 1-tuple but the
        # trailing comma was dropped). Handle both — the failure mode is
        # silent and confusing otherwise.
        eps_default = cfg_default.eps_ent
        if isinstance(eps_default, (tuple, list)):
            eps_default = eps_default[0] if eps_default else 0.02
        self.dual_eps = float(_val("lr_cifar10_dual_eps", eps_default))
        self.wfr_num_samples = int(_val("lr_cifar10_wfr_num_samples", cfg_default.num_samples))
        self.dual_sample_level = int(_val("lr_cifar10_sample_level", cfg_default.sample_level))
        self.ppa_num_rounds = int(cfg_default.ppa_num_rounds)
        self.ppa_min_rounds = int(cfg_default.ppa_min_rounds)
        self.ppa_refine_steps = int(cfg_default.ppa_refine_steps)
        self.ppa_refine_lr = float(cfg_default.ppa_refine_lr)
        self.ppa_delta_rtol = float(cfg_default.ppa_delta_rtol)
        # NPF and NN-DRO architectures must match the training-time defaults
        # (since the saved state dicts encode shape).
        self.npf_hidden = tuple(cfg_default.npf_hidden)
        self.npf_outer_rank = int(cfg_default.npf_outer_rank)
        self.npf_inner_rank = int(cfg_default.npf_inner_rank)
        self.npf_activation = str(cfg_default.npf_activation)
        self.npf_softplus_beta = float(cfg_default.npf_softplus_beta)
        self.npf_init_eps = float(cfg_default.npf_init_eps)
        self.nn_dro_hidden = tuple(cfg_default.nn_dro_hidden)
        self.nn_dro_activation = str(cfg_default.nn_dro_activation)
        self.nn_dro_softplus_beta = float(cfg_default.nn_dro_softplus_beta)
        self.nn_dro_init_scale = float(cfg_default.nn_dro_init_scale)

        self.epoch_to_load = getattr(args, "lr_cifar10_epoch", None)  # None -> highest

        self._checkpoint_dir_arg = args.checkpoint_dir

        # PGD eval shift parameter
        self._shift_levels = (0.008, 0.016, 0.032, 0.064)

    # --- backend interface ---

    def supported_methods(self):
        return tuple(self._CKPT_PREFIX.keys())

    def sample_held_out_anchors(self, eval_size: int, seed: int, device: torch.device) -> torch.Tensor:
        if eval_size > self._eval_size_max:
            raise ValueError(
                f"Requested eval_size={eval_size} exceeds pool size {self._eval_size_max}."
            )
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        idx = torch.randperm(self._eval_size_max, generator=g)[:eval_size]
        anchors = self._pool_features[idx].to(device)
        # Stash labels for the same indices so _held_out_labels(z_hat) below
        # is a direct lookup rather than an O(B * N * d) NN search.
        self._last_anchor_labels = self._pool_labels[idx].to(device)
        return anchors

    def _resolve_checkpoint_dir(self, lam: float, seed: int) -> Path:
        """Resolve placeholders in --checkpoint_dir.

        Supports {lam}, {seed}, and {tau} (with tau = 1 / (2 lam), since the
        LR-CIFAR10 trainer takes --tau). Use ``:g`` format spec to strip
        trailing zeros, e.g. ``outputs_tau_{tau:g}_seed_{seed}``.

        Auto-descends into a ``checkpoints/`` subdir when the resolved path
        contains one but no checkpoint files directly. This makes both
        ``--checkpoint_dir <out_dir>`` and ``--checkpoint_dir <out_dir>/checkpoints``
        work, since the LR-CIFAR10 trainer writes per-epoch checkpoints into
        ``<out_dir>/checkpoints/`` (CIFAR10_LogReg.py:67).
        """
        template = str(self._checkpoint_dir_arg)
        tau = 1.0 / (2.0 * float(lam))
        resolved = Path(template.format(lam=lam, seed=seed, tau=tau))
        if resolved.is_dir():
            sub = resolved / "checkpoints"
            if sub.is_dir() and not any(resolved.glob("*_run*_epoch_*.pth")):
                return sub
        return resolved

    @staticmethod
    def _infer_mlp_arch(state_dict):
        """Recover (hidden_sizes, input_dim) from a saved MLPAdversary state.

        MLPAdversary layout: ``hidden.{i}.weight`` shape (out_i, in_i), then
        ``out.weight`` shape (input_dim, last_hidden). Returns (None, None)
        if the state-dict doesn't look like an MLPAdversary.
        """
        hidden_keys = sorted(
            (k for k in state_dict if k.startswith("hidden.") and k.endswith(".weight")),
            key=lambda k: int(k.split(".")[1]),
        )
        if not hidden_keys:
            return None, None
        first = state_dict[hidden_keys[0]]
        input_dim = int(first.shape[1])
        hidden_sizes = tuple(int(state_dict[k].shape[0]) for k in hidden_keys)
        return hidden_sizes, input_dim

    @staticmethod
    def _infer_npf_arch(state_dict):
        """Recover (hidden_sizes, input_dim) from a saved NPFInputConvexPotential
        state. q_blocks.<l>.delta_raw has shape (num_forms_l, input_dim), and
        num_forms_l == hidden_sizes[l]."""
        q_keys = sorted(
            (k for k in state_dict
             if k.startswith("q_blocks.") and k.endswith(".delta_raw")),
            key=lambda k: int(k.split(".")[1]),
        )
        if not q_keys:
            return None, None
        first = state_dict[q_keys[0]]
        input_dim = int(first.shape[1])
        hidden_sizes = tuple(int(state_dict[k].shape[0]) for k in q_keys)
        return hidden_sizes, input_dim

    def _find_latest_epoch(self, ckpt_dir: Path, prefix: str, run_id: int = 0) -> Path:
        candidates = sorted(
            ckpt_dir.glob(f"{prefix}_run{run_id}_epoch_*.pth"),
            key=lambda p: int(p.stem.split("_epoch_")[-1]),
        )
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found under {ckpt_dir} for prefix {prefix!r}."
            )
        if self.epoch_to_load is None:
            return candidates[-1]
        target = ckpt_dir / f"{prefix}_run{run_id}_epoch_{int(self.epoch_to_load)}.pth"
        if not target.exists():
            raise FileNotFoundError(f"Requested epoch checkpoint missing: {target}")
        return target

    def train_or_load(self, method, lam, seed, eval_size, device, **_):
        from models.classifier import LinearModel  # type: ignore

        ckpt_dir = self._resolve_checkpoint_dir(lam, seed)
        prefix = self._CKPT_PREFIX[method]
        ckpt_path = self._find_latest_epoch(ckpt_dir, prefix)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        classifier = LinearModel(self.input_dim, self.num_classes, bias=True).to(device)

        state = {
            "method":    method,
            "lam":       float(lam),
            "ckpt_path": str(ckpt_path),
            "device":    device,
        }

        if method == "erm":  # SAA: bare LinearModel state dict
            classifier.load_state_dict(ckpt)
            state["classifier"] = classifier
            return state

        if method == "npf":
            from models.npf import NPFInputConvexPotential  # type: ignore
            classifier.load_state_dict(ckpt["B_state_dict"])
            sd = ckpt["psi_omega_state_dict"]
            # Auto-detect hidden sizes from q_blocks.<l>.delta_raw shape (W, d).
            inferred_hidden, inferred_in_dim = self._infer_npf_arch(sd)
            psi = NPFInputConvexPotential(
                input_dim=inferred_in_dim or self.input_dim,
                hidden_sizes=inferred_hidden or self.npf_hidden,
                outer_rank=self.npf_outer_rank,
                inner_rank=self.npf_inner_rank,
                activation=self.npf_activation,
                elu_alpha=1.0,
                softplus_beta=self.npf_softplus_beta,
                init_eps=self.npf_init_eps,
            ).to(device)
            psi.load_state_dict(sd)
            psi.eval()
            state.update({"classifier": classifier, "psi": psi})
            return state

        if method == "nn_dro":
            from models.nn_dro import MLPAdversary  # type: ignore
            classifier.load_state_dict(ckpt["B_state_dict"])
            sd = ckpt["adversary_state_dict"]
            # Auto-detect from hidden.<i>.weight shapes.
            inferred_hidden, inferred_in_dim = self._infer_mlp_arch(sd)
            mlp = MLPAdversary(
                input_dim=inferred_in_dim or self.input_dim,
                hidden_sizes=inferred_hidden or self.nn_dro_hidden,
                activation=self.nn_dro_activation,
                softplus_beta=self.nn_dro_softplus_beta,
                init_scale=self.nn_dro_init_scale,
            ).to(device)
            mlp.load_state_dict(sd)
            mlp.eval()
            state.update({"classifier": classifier, "adversary": mlp})
            return state

        # Implicit-adversary methods: only the LinearModel is saved.
        classifier.load_state_dict(ckpt)
        state["classifier"] = classifier
        return state

    # --- per-method push wrappers (LR-CIFAR10 flavor) ---

    def push(self, method, state, z_hat):
        # Held-out labels: needed by PPA/WFR/Dual (cross-entropy gradient).
        # ``sample_held_out_anchors`` stashes them; we slice through chunks
        # so the per-anchor label alignment is preserved.
        chunk = self.push_chunk
        B = z_hat.shape[0]
        if chunk and chunk > 0 and B > chunk:
            full_labels = getattr(self, "_last_anchor_labels", None)
            outs = []
            try:
                for i in range(0, B, chunk):
                    sub = z_hat[i:i + chunk]
                    if full_labels is not None:
                        self._last_anchor_labels = full_labels[i:i + chunk]
                    outs.append(self._push_one(method, state, sub))
            finally:
                if full_labels is not None:
                    self._last_anchor_labels = full_labels
            return torch.cat(outs, dim=0)
        return self._push_one(method, state, z_hat)

    def _push_one(self, method, state, z_hat):
        if method == "erm":
            return z_hat.detach().clone()
        if method == "npf":
            return self._push_npf(state, z_hat)
        if method == "nn_dro":
            return self._push_nn_dro(state, z_hat)
        if method in ("particle_ascent", "ppa"):
            return self._push_inner_ascent(state, z_hat, projected=(method == "ppa"))
        if method == "wfr":
            return self._push_wfr(state, z_hat)
        if method == "dual":
            return self._push_dual(state, z_hat)
        raise ValueError(f"[lr_cifar10] unknown method {method!r}")

    def _held_out_labels(self, z_hat: torch.Tensor) -> torch.Tensor:
        """Return labels for the most-recently-sampled anchor batch.

        ``sample_held_out_anchors`` stashes ``self._last_anchor_labels`` so
        this is a direct lookup. Falls back to a chunked nearest-neighbour
        search only if the cache is missing or its shape doesn't match (e.g.,
        a custom caller bypassed ``sample_held_out_anchors``).
        """
        cached = getattr(self, "_last_anchor_labels", None)
        if cached is not None and cached.shape[0] == z_hat.shape[0]:
            return cached.to(z_hat.device)

        # Fallback: chunked NN search to avoid materializing a B x N x d tensor.
        pool = self._pool_features.to(z_hat.device)
        pool_lbls = self._pool_labels.to(z_hat.device)
        out = torch.empty(z_hat.shape[0], dtype=torch.long, device=z_hat.device)
        chunk = 256
        pool_sq = (pool * pool).sum(-1)  # (N,)
        for i in range(0, z_hat.shape[0], chunk):
            zb = z_hat[i:i + chunk]
            zb_sq = (zb * zb).sum(-1, keepdim=True)
            d2 = zb_sq + pool_sq.unsqueeze(0) - 2.0 * (zb @ pool.t())
            out[i:i + chunk] = d2.argmin(dim=1)
        return pool_lbls[out]

    def _push_npf(self, state, z_hat):
        from models.npf import npf_transport_map  # type: ignore
        psi = state["psi"]
        # NPF expects a parameter dict for ``functional_call``; build it from
        # the loaded state_dict.
        params = dict(psi.named_parameters())
        for k, v in psi.named_buffers():
            params[k] = v
        Tz = npf_transport_map(psi, params, z_hat.detach(), create_graph=False)
        return Tz.detach()

    def _push_nn_dro(self, state, z_hat):
        adversary = state["adversary"]
        with torch.no_grad():
            return adversary(z_hat).detach()

    def _push_inner_ascent(self, state, z_hat, projected: bool):
        """Replay PPA / particle-ascent inner loop with classifier frozen.

        Mirrors :meth:`Logistic_Regression_CIFAR10.algorithms.ppa.PPA._ppa_sampler`
        (and ``WRM._sinha_sampler`` for ``projected=False``).
        """
        import torch.nn.functional as F

        classifier = state["classifier"]
        for p in classifier.parameters():
            p.requires_grad_(False)
        classifier.eval()

        labels = self._held_out_labels(z_hat)
        x_anc = z_hat.detach()
        z = x_anc.clone()
        lam = float(state["lam"])

        # Round 0: WRM-style ascent with diminishing step size.
        for s in range(1, self.inner_steps + 1):
            z.requires_grad_(True)
            ce = F.cross_entropy(classifier(z), labels, reduction="none")
            grads = torch.autograd.grad(ce.sum(), z, create_graph=False)[0]
            eta_s = self.inner_lr / (s ** 0.5)
            z = z.detach() + eta_s * (grads - 2.0 * lam * (z.detach() - x_anc))

        if not projected:
            return z.detach()

        # PPA: alternate Brenier projection + constant-lr refinement.
        from utils.projections import brenier_projection_features  # type: ignore
        for round_idx in range(1, self.ppa_num_rounds):
            z, _, delta, c_id, _ = brenier_projection_features(z.detach(), x_anc, labels)
            if round_idx >= self.ppa_min_rounds and delta < self.ppa_delta_rtol * max(c_id, 1e-12):
                break
            for _ in range(self.ppa_refine_steps):
                z.requires_grad_(True)
                ce = F.cross_entropy(classifier(z), labels, reduction="none")
                grads = torch.autograd.grad(ce.sum(), z, create_graph=False)[0]
                z = z.detach() + self.ppa_refine_lr * (
                    grads - 2.0 * lam * (z.detach() - x_anc)
                )
        z, _, _, _, _ = brenier_projection_features(z.detach(), x_anc, labels)
        return z.detach()

    def _push_wfr(self, state, z_hat):
        """Mirror ``algorithms/wfr.py:_WFR_sampler`` to produce a particle cloud,
        then reduce to its weighted barycenter as the deterministic Monge map."""
        import torch.nn as nn

        classifier = state["classifier"]
        for p in classifier.parameters():
            p.requires_grad_(False)
        classifier.eval()

        labels = self._held_out_labels(z_hat)
        B = z_hat.shape[0]
        S = self.wfr_num_samples
        d = z_hat.shape[1]
        eps = float(self.dual_eps)
        lam = float(state["lam"])

        x_anchor = z_hat.detach().unsqueeze(1).expand(-1, S, -1).reshape(B * S, d).contiguous()
        x = x_anchor.clone()
        y = labels.repeat_interleave(S, dim=0)
        weights = torch.full((B, S), 1.0 / S, device=z_hat.device, dtype=z_hat.dtype)
        crit = nn.CrossEntropyLoss(reduction="none")
        wt_lr = self.inner_lr
        weight_exp = 1.0 - lam * eps * wt_lr
        std = (2.0 * self.inner_lr * lam * eps) ** 0.5

        for _ in range(self.inner_steps):
            x.requires_grad_(True)
            loss = crit(classifier(x), y)
            grads = torch.autograd.grad(loss.sum(), x, create_graph=False)[0]
            x = x.detach()
            with torch.no_grad():
                mean = x + self.inner_lr * (grads - 2.0 * lam * (x - x_anchor))
                x = mean + torch.randn_like(mean) * std
                cur = crit(classifier(x), y).view(B, S)
                dist = ((x - x_anchor) ** 2).sum(dim=1).view(B, S)
                energy = cur - 2.0 * lam * dist
                weights = (weights.clamp_min(1e-12) ** weight_exp) * torch.exp(
                    torch.clamp(energy * wt_lr, min=-40.0, max=40.0)
                )
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)

        # Weighted barycenter -> deterministic Monge map per anchor.
        x_grid = x.view(B, S, d)
        return (weights.unsqueeze(-1) * x_grid).sum(dim=1).detach()

    def _push_dual(self, state, z_hat):
        """SDRO Dual: Gaussian noise around z_hat, softmax reweighting by loss."""
        import torch.nn as nn

        classifier = state["classifier"]
        classifier.eval()
        labels = self._held_out_labels(z_hat)
        B, d = z_hat.shape
        m = 2 ** int(self.dual_sample_level)
        eps = float(self.dual_eps)
        lam = float(state["lam"])

        z_anc = z_hat.detach().unsqueeze(1).expand(-1, m, -1).reshape(B * m, d).contiguous()
        noise = torch.randn_like(z_anc) * (eps ** 0.5)
        z_samples = z_anc + noise
        y_rep = labels.repeat_interleave(m, dim=0)
        with torch.no_grad():
            ce = nn.CrossEntropyLoss(reduction="none")(classifier(z_samples), y_rep)
            v = ce / max(lam * eps, 1e-8)
            v = v.view(B, m)
            w = torch.softmax(v, dim=1)
            return (w.unsqueeze(-1) * z_samples.view(B, m, d)).sum(dim=1).detach()

    # --- test-loss eval (PGD accuracy at a single shift level) ---

    def compute_test_loss(self, state, seed, device, delta: float = 1.0):
        """Evaluate clean + PGD accuracy at one shift level.

        ``delta`` selects the shift index into ``self._shift_levels``; the
        resulting epsilon is delta * average test feature norm * shift_levels[idx].
        We expose a single number: 100 - PGD_accuracy at the chosen level
        (so ``test_loss`` is monotonically increasing in lack-of-robustness).
        """
        import numpy as np
        from utils.pgd import pgd_attack  # type: ignore

        classifier = state["classifier"]
        classifier.eval()

        # Use the entire test set for the eval metric (independent of held-out
        # subset used for the Monge gap, which only takes the first eval_size).
        x = self._test_features.to(device)
        y = self._test_labels.to(device)
        avg_norm = float(torch.linalg.norm(x, dim=1).mean().item())
        # Pick a shift level by index = floor(delta) (default 1 -> first level).
        idx = max(0, min(len(self._shift_levels) - 1, int(delta)))
        epsilon = float(self._shift_levels[idx]) * avg_norm

        # Streaming PGD eval to keep memory low.
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(x, y), batch_size=256)
        correct, total = 0, 0
        for xb, yb in loader:
            x_adv = pgd_attack(
                model=classifier, features=xb, labels=yb,
                epsilon=epsilon, alpha=epsilon / 8, num_iter=10, device=device,
            )
            with torch.no_grad():
                pred = classifier(x_adv).argmax(dim=-1)
                correct += int((pred == yb.to(device)).sum().item())
                total += int(yb.numel())
        pgd_acc = 100.0 * correct / total
        return 100.0 - pgd_acc  # "loss" = 1 - accuracy in pct


# ---------------------------------------------------------------------------
# RL CartPole backend.
# ---------------------------------------------------------------------------


class RLCartpoleBackend:
    """Loads existing RL CartPole policy checkpoints and evaluates the Monge
    gap on a held-out batch of xi anchors (drawn uniformly from the
    [xi_low, xi_high] box).

    Behavior depends on whether the checkpoint was produced by the *current*
    RL_minimal.py (which persists the trained parametric adversary state
    alongside the policy under the keys ``adv_kind`` / ``adv_state_dict`` /
    ``adv_arch``) or an older version (policy-only):

      * **Parametric adversaries** (``icnn``, ``npf``, ``nn_dro``).  When the
        checkpoint contains ``adv_kind in {"icnn", "npf", "nn_dro"}``, we
        rebuild the trained module from ``adv_arch`` + ``adv_state_dict`` and
        apply its transport map ``T(z_hat)`` *directly* (one map evaluation;
        no inner re-optimization). This matches what the LR-CIFAR10 backend
        does and is what the paper table reports.
      * **Non-parametric adversaries** (``particle``, ``ppa``, ``dual``,
        ``wfr``, ``wgf``, ``svg``, ``rgo``, ``algo1``, ``new_ppa``).  These
        carry no trained state of their own; we replay the inner
        ascent/sampling against the trained policy via
        ``adversary.adversarial_xi(...)`` — semantically identical to a
        per-minibatch call at training time.
      * **Nominal** (= ERM). Identity push by definition (the row used as
        ``T = id`` reference in the LR-CIFAR10 table).

    Older policy-only checkpoints are still accepted: parametric methods
    fall back to a freshly initialized ``adversarial_xi`` run, with a
    one-time warning printed to stderr.

    Checkpoint filenames are
        ``RL_minimal_{env}_{tag}_seed{seed}_lam_{lam}_softplusbeta_{beta}_{ts}_{method}_policy.pt``
    where ``ts`` is a timestamp (so filenames cannot be deterministically
    reconstructed from (lam, seed)). Pass ``--checkpoint_dir`` pointing at
    the directory containing those .pt files; this backend globs to find
    the right one for each (method, lam, seed) cell.
    """

    name = "rl_cartpole"

    def __init__(self, args):
        _import_rl_modules()
        from config import InnerConfig  # type: ignore
        from envs import ENV_SPECS, make_vec_env  # type: ignore

        self.InnerConfig = InnerConfig
        self.ENV_SPECS = ENV_SPECS
        self._make_vec_env = make_vec_env

        self.env_name = getattr(args, "rl_env", "cartpole")
        self.env_max_steps = int(getattr(args, "rl_env_max_steps", 500))
        self.fd_episodes = int(getattr(args, "rl_fd_episodes", 1))
        self.fd_horizon = int(getattr(args, "rl_fd_horizon", 200))
        self.policy_hidden = int(getattr(args, "rl_policy_hidden", 64))
        # Test-loss shift parameters (CartPole physics).
        self.eval_n_trials = int(getattr(args, "rl_eval_n_trials", 200))

        spec = ENV_SPECS[self.env_name]
        self._xi_low = torch.tensor(spec.xi_low, dtype=torch.float32)
        self._xi_high = torch.tensor(spec.xi_high, dtype=torch.float32)
        self._xi_dim = int(spec.xi_dim)

        self._checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
        if self._checkpoint_dir is None:
            raise ValueError("--checkpoint_dir is required for rl_cartpole.")

    # --- backend interface ---

    def supported_methods(self):
        # Mirrors the registry in RL/algorithms/registry.py. The paper's table
        # rows map as follows:
        #   ERM      -> nominal      (identity push)
        #   PA       -> particle     (or algo1 / WRM)
        #   WFR      -> wfr
        #   SDRO     -> dual
        #   NN-DRO   -> nn_dro
        #   MPA      -> ppa          (Brenier-projection-based)
        #   ICNN-DRO -> icnn or npf  (gradient of ICNN potential; both Brenier)
        return (
            "nominal", "particle", "icnn", "algo1", "npf",
            "ppa", "new_ppa", "dual", "wgf", "wfr", "svg", "rgo", "nn_dro",
        )

    def sample_held_out_anchors(self, eval_size, seed, device):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        u = torch.rand(eval_size, self._xi_dim, generator=g, dtype=torch.float32)
        return (self._xi_low + (self._xi_high - self._xi_low) * u).to(device)

    def _find_policy(self, method, lam, seed) -> Path:
        # Filename: RL_minimal_{env}_{tag}_seed{seed}_lam_{lam}_..._{method}_policy.pt
        # Strict primary match on (seed, lam, method). The fallback to a less
        # specific glob is only allowed when it is *unambiguous* (1 candidate);
        # otherwise we fail loudly rather than silently evaluating the wrong
        # checkpoint and reporting it as if it were the requested cell. The
        # legacy fallback masked missing (seed, lam) combinations and produced
        # constant rows across lambdas/seeds in the CSV.
        pattern = (
            f"RL_minimal_*_seed{int(seed)}_lam_{lam}_*_{method}_policy.pt"
        )
        candidates = sorted(self._checkpoint_dir.glob(pattern))
        if not candidates:
            fallback = sorted(self._checkpoint_dir.glob(f"*_{method}_policy.pt"))
            if len(fallback) == 1:
                # Single training run with a custom filename — preserve the
                # original "user supplied --json-path" use case.
                return fallback[0]
            if len(fallback) > 1:
                names = "\n  ".join(p.name for p in fallback)
                raise FileNotFoundError(
                    f"No RL policy found matching seed={seed} lam={lam} method={method!r} "
                    f"under {self._checkpoint_dir} (pattern: {pattern}). "
                    f"Refusing to fall back: {len(fallback)} ambiguous "
                    f"{method!r} checkpoints exist:\n  {names}\n"
                    f"Train the missing cell, or restrict --seeds / --lams to "
                    f"combinations that have a checkpoint."
                )
            raise FileNotFoundError(
                f"No RL policy found for method={method!r} lam={lam} seed={seed} "
                f"under {self._checkpoint_dir}."
            )
        # Most-recent timestamp wins (sort lexicographic on the YYYYMMDDTHHMMSSZ tag).
        return candidates[-1]

    def train_or_load(self, method, lam, seed, eval_size, device, **_):
        from models.policy import PolicyNet  # type: ignore

        ckpt_path = self._find_policy(method, lam, seed)
        # Use weights_only=False so the optional ``adv_state_dict`` (which
        # contains tensors but is wrapped in a plain dict) loads cleanly.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy = PolicyNet(
            obs_dim=int(ckpt["obs_dim"]),
            hidden=int(ckpt["hidden"]),
            act_dim=int(ckpt["act_dim"]),
        ).to(device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()

        # If the training run wrote xi-box explicitly, prefer it (covers the
        # case where the user trained with overridden --xi-low / --xi-high).
        xi_low = ckpt.get("xi_low", self._xi_low.tolist())
        xi_high = ckpt.get("xi_high", self._xi_high.tolist())
        cfg_inner = self.InnerConfig(
            lam=float(lam),
            xi_low=tuple(float(v) for v in xi_low),
            xi_high=tuple(float(v) for v in xi_high),
            M_diag=tuple(1.0 / (h - l) ** 2 for l, h in zip(xi_low, xi_high)),
            fd_episodes=self.fd_episodes,
            fd_horizon=self.fd_horizon,
        )
        env_eval = self._make_vec_env(self.env_name, max_steps=self.env_max_steps, device=device)

        adv_module = self._maybe_build_trained_adversary(ckpt, cfg_inner, device)

        return {
            "method":     method,
            "lam":        float(lam),
            "policy":     policy,
            "env_eval":   env_eval,
            "cfg_inner":  cfg_inner,
            "device":     device,
            "ckpt_path":  str(ckpt_path),
            "adv_kind":   ckpt.get("adv_kind", None),
            "adv_module": adv_module,
            "xi_low":     torch.tensor(xi_low, device=device, dtype=torch.float32),
            "xi_high":    torch.tensor(xi_high, device=device, dtype=torch.float32),
        }

    @staticmethod
    def _maybe_build_trained_adversary(ckpt: Dict[str, Any], cfg_inner, device):
        """Reconstruct the trained parametric adversary module if persisted.

        Returns one of:
          * an ``ICNN`` module with state loaded (adv_kind == "icnn"),
          * an ``NPFInputConvexPotential`` module (adv_kind == "npf"),
          * an ``MLPAdversary`` module (adv_kind == "nn_dro"),
          * ``None`` (no parametric state in checkpoint, or adv_kind == "none").
        """
        kind = ckpt.get("adv_kind", None)
        if kind in (None, "none"):
            return None
        sd = ckpt.get("adv_state_dict", None)
        arch = ckpt.get("adv_arch", None)
        if sd is None or arch is None:
            return None

        if kind == "icnn":
            from models.icnn import ICNN  # type: ignore
            mod = ICNN(
                input_dim=int(arch["input_dim"]),
                hidden_sizes=tuple(int(h) for h in arch["hidden_sizes"]),
                activation=str(arch["activation"]),
                strong_convexity=float(arch["strong_convexity"]),
                nonneg_init=str(arch["nonneg_init"]),
                softplus_beta=float(arch["softplus_beta"]),
            ).to(device)
            mod.load_state_dict(sd)
            mod.eval()
            return mod
        if kind == "npf":
            from models.npf import NPFInputConvexPotential  # type: ignore
            mod = NPFInputConvexPotential(
                input_dim=int(arch["input_dim"]),
                hidden_sizes=tuple(int(h) for h in arch["hidden_sizes"]),
                outer_rank=int(arch["outer_rank"]),
                inner_rank=int(arch["inner_rank"]),
                activation=str(arch["activation"]),
                elu_alpha=float(arch.get("elu_alpha", 1.0)),
                softplus_beta=float(arch["softplus_beta"]),
                init_eps=float(arch.get("init_eps", 1e-3)),
            ).to(device)
            mod.load_state_dict(sd)
            mod.eval()
            return mod
        if kind == "nn_dro":
            from models.nn_dro import MLPAdversary  # type: ignore
            mod = MLPAdversary(
                input_dim=int(arch["input_dim"]),
                hidden_sizes=tuple(int(h) for h in arch["hidden_sizes"]),
                activation=str(arch["activation"]),
                softplus_beta=float(arch["softplus_beta"]),
                init_scale=float(arch.get("init_scale", 1e-3)),
            ).to(device)
            mod.load_state_dict(sd)
            mod.eval()
            return mod
        return None

    def push(self, method, state, z_hat):
        # ERM / nominal: identity push by definition (matches the LR-CIFAR10
        # paper's reference row T_ERM := id with M=0 by definition).
        if method == "nominal":
            return z_hat.detach().clone()

        adv_module = state.get("adv_module", None)
        adv_kind = state.get("adv_kind", None)

        # Parametric adversaries (icnn / npf / nn_dro) — apply the *trained*
        # transport map directly. This is the single map evaluation T_psi(z_hat),
        # *not* a freshly-initialized inner re-optimization. The Brenier
        # methods (icnn, npf) satisfy M(T) = 0 by construction *in u-space*
        # (the unconstrained latent where the convex potential lives). After
        # the sigmoid box-bijection back to xi, T is no longer OT for ||.||^2
        # in xi-space — the gap is small but non-zero by construction. See
        # gap_inputs() below: for icnn/npf we evaluate the gap in u-space so
        # the reported number reflects what the method actually optimizes.
        if adv_module is not None and adv_kind in ("icnn", "npf", "nn_dro"):
            xi_low = state["xi_low"]
            xi_high = state["xi_high"]
            with torch.no_grad():
                if adv_kind in ("icnn", "npf"):
                    return self._apply_brenier_potential(adv_module, z_hat, xi_low, xi_high)
                return self._apply_mlp_displacement(adv_module, z_hat, xi_low, xi_high)

        # Parametric method requested but no trained-adversary state in
        # checkpoint — emit a one-time warning and fall back to the legacy
        # behavior (re-run inner ascent from a fresh adversary).
        if method in ("icnn", "npf", "nn_dro") and not getattr(self, "_warned_legacy", False):
            print(
                f"[rl_cartpole] WARNING: checkpoint for method={method!r} has no "
                f"adv_state_dict. Falling back to a freshly-initialized adversary "
                f"inner-loop run; results will not reflect training. Re-train "
                f"with the current RL_minimal.py to persist adv state.",
                file=sys.stderr,
            )
            self._warned_legacy = True

        # Non-parametric adversaries (particle, ppa, dual, wfr, wgf, svg,
        # rgo, algo1, new_ppa) — these carry no learned state, so re-running
        # adversarial_xi against the trained policy is exactly what training
        # does on each minibatch.
        from algorithms.registry import build_registry  # type: ignore

        reg = build_registry()
        if method not in reg:
            raise ValueError(f"[rl_cartpole] unknown method {method!r}")
        _, factory = reg[method]
        adv = factory(state["cfg_inner"], state["device"])
        seed0 = int(z_hat.sum().abs().item()) % 1_000_003  # stable per-batch seed
        Tz = adv.adversarial_xi(state["env_eval"], state["policy"], z_hat, seed0=seed0)
        Tz = Tz.detach()

        # Cloud adversaries (WFR, WGF, RGO, ...) return shape (B*S, p), not (B, p).
        # Reduce to the barycenter per anchor — same convention used by the
        # LR-CIFAR10 backend, and the spec's WFR/SDRO Monge-gap reduction.
        if Tz.shape[0] != z_hat.shape[0] and Tz.shape[0] % z_hat.shape[0] == 0:
            S = Tz.shape[0] // z_hat.shape[0]
            Tz = Tz.view(z_hat.shape[0], S, -1).mean(dim=1)

        return Tz

    def gap_inputs(self, method, state, z_hat, Tz):
        """Return (z, Tz) in the space where the Monge gap should be measured.

        For the Brenier-potential methods (icnn, npf) the convex potential
        lives in unconstrained latent u-space, with T(xi) = decode(grad_u
        psi(encode(xi))). Brenier guarantees grad_u psi is OT in u-space
        under ||.||^2; the sigmoid decode bijection breaks that in xi-space.
        We recompute u_adv = grad_u psi(u_hat) directly (rather than
        round-tripping logit(Tz), which clamps at the box bijection's eps
        and loses points whose pushed |u_adv| > logit(1 - eps)) so the gap
        is exactly 0 at convergence — matching the LR-CIFAR10 backend.

        For all other methods the native space *is* xi (no bijection in
        push), so we leave inputs unchanged.
        """
        if method not in ("icnn", "npf"):
            return z_hat, Tz
        adv_module = state.get("adv_module", None)
        if adv_module is None:
            # No persisted adversary state — push fell back to a freshly-
            # initialized inner loop, so u-space recovery is not meaningful.
            return z_hat, Tz
        xi_low = state["xi_low"]
        xi_high = state["xi_high"]
        eps = 1e-6
        rng = (xi_high - xi_low).clamp_min(1e-12)
        p = ((z_hat - xi_low) / rng).clamp(eps, 1.0 - eps)
        u_hat = (torch.log(p) - torch.log1p(-p)).detach().requires_grad_(True)
        with torch.enable_grad():
            phi = adv_module(u_hat)
            u_adv = torch.autograd.grad(phi.sum(), u_hat, create_graph=False)[0]
        u_adv = torch.nan_to_num(u_adv, nan=0.0, posinf=0.0, neginf=0.0)
        return u_hat.detach(), u_adv.detach()

    @staticmethod
    def _apply_brenier_potential(potential, hat_xi, xi_low, xi_high):
        """T(hat_xi) = decode(grad_u phi(encode(hat_xi))) — same box bijection
        used at training time by ICNN/NPF adversaries (see
        RL/algorithms/icnn.py and RL/algorithms/npf.py)."""
        eps = 1e-6
        rng = (xi_high - xi_low).clamp_min(1e-12)
        p = ((hat_xi - xi_low) / rng).clamp(eps, 1.0 - eps)
        u_in = (torch.log(p) - torch.log1p(-p)).detach().requires_grad_(True)
        with torch.enable_grad():
            phi = potential(u_in)
            u_adv = torch.autograd.grad(phi.sum(), u_in, create_graph=False)[0]
        u_adv = torch.nan_to_num(u_adv, nan=0.0, posinf=0.0, neginf=0.0)
        xi_adv = xi_low + rng * torch.sigmoid(u_adv)
        xi_adv = torch.where(torch.isfinite(xi_adv), xi_adv, hat_xi)
        return xi_adv.detach()

    @staticmethod
    def _apply_mlp_displacement(mlp, hat_xi, xi_low, xi_high):
        """T(hat_xi) = decode(encode(hat_xi) + mlp(encode(hat_xi))) — matches
        RL/algorithms/nn_dro.py. mlp is the trained MLPAdversary."""
        eps = 1e-6
        rng = (xi_high - xi_low).clamp_min(1e-12)
        p = ((hat_xi - xi_low) / rng).clamp(eps, 1.0 - eps)
        hat_u = (torch.log(p) - torch.log1p(-p)).detach()
        delta = mlp(hat_u)
        u_adv = hat_u + delta
        u_adv = torch.nan_to_num(u_adv, nan=0.0, posinf=0.0, neginf=0.0)
        xi_adv = xi_low + rng * torch.sigmoid(u_adv)
        xi_adv = torch.where(torch.isfinite(xi_adv), xi_adv, hat_xi)
        return xi_adv.detach()

    def compute_test_loss(self, state, seed, device, delta: float = 1.0):
        """Average episode length over a grid of perturbed (masspole, length)
        physics. Returns a "loss" = max_steps - avg_length so the metric is
        monotonically increasing in lack-of-robustness."""
        # Re-implement evaluate_episode_lengths inline against env_eval, so the
        # test loss eval doesn't depend on RL_eval_table.py.
        policy = state["policy"]
        policy.eval()
        env_eval = state["env_eval"]
        # Sample a worst-case grid and compute mean return.
        from utils.eval import evaluate_worst_case_grid  # type: ignore

        # delta is informational here; we always compute the worst-case-grid
        # return on the original xi-box.
        worst = evaluate_worst_case_grid(
            env_eval, policy, state["cfg_inner"],
            grid_n=11, n_episodes=1, max_steps=self.env_max_steps,
            seed0=int(seed),
        )
        # "loss" = max_steps - worst_return. Higher means less robust.
        return float(self.env_max_steps - worst)


DATASET_BACKENDS = {
    "least_squares": LeastSquaresBackend,
    "lr_cifar10":    LRCifar10Backend,
    "rl_cartpole":   RLCartpoleBackend,
}


# ---------------------------------------------------------------------------
# Sweep driver.
# ---------------------------------------------------------------------------


def _gap_for_state(
    z_hat: torch.Tensor,
    Tz: torch.Tensor,
    eval_size: int,
    mode: str = "auto",
    sub_M: int = 4096,
    sub_K: int = 10,
    sub_seed: int = 0,
    sinkhorn_eps_frac: float = 0.01,
    sinkhorn_n_iter: int = 1000,
) -> Dict[str, float]:
    if mode == "subsample_hungarian":
        return monge_gap_subsample_hungarian(
            z_hat, Tz, M=sub_M, K=sub_K, seed=sub_seed,
        )
    if mode == "hungarian":
        return monge_gap_hungarian(z_hat, Tz)
    if mode == "sinkhorn":
        from monge_gap_utils.monge_gap import median_pairwise_distance_sq
        eps = sinkhorn_eps_frac * median_pairwise_distance_sq(
            torch.cat([z_hat, Tz], dim=0)
        )
        if eps <= 0.0:
            eps = 1e-6
        return monge_gap_sinkhorn(z_hat, Tz, eps=eps, n_iter=sinkhorn_n_iter)
    # auto: Hungarian for small N, Sinkhorn otherwise.
    if eval_size <= 4096:
        return monge_gap_hungarian(z_hat, Tz)
    from monge_gap_utils.monge_gap import median_pairwise_distance_sq
    eps = sinkhorn_eps_frac * median_pairwise_distance_sq(
        torch.cat([z_hat, Tz], dim=0)
    )
    if eps <= 0.0:
        eps = 1e-6
    return monge_gap_sinkhorn(z_hat, Tz, eps=eps, n_iter=sinkhorn_n_iter)


def _flatten_lams(lams: List[float]) -> List[float]:
    """Round to a stable float so CSV round-trips don't cause re-runs."""
    return [round(float(l), 8) for l in lams]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASET_BACKENDS.keys()))
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--lams", nargs="+", type=float, required=True)
    parser.add_argument("--eval_size", type=int, default=4096,
                        help="Size of held-out anchor batch.")
    parser.add_argument("--checkpoint_dir", type=Path, default=None,
                        help="Directory of trained adversaries (for "
                             "checkpoint-based backends). Ignored for "
                             "least_squares which trains inline.")
    parser.add_argument("--out", type=Path, default=Path("monge_gap.csv"))
    parser.add_argument("--device", type=str, default=None,
                        help="Defaults to cuda if available, else cpu.")
    parser.add_argument("--shift_delta", type=float, default=1.0,
                        help="Distribution shift delta passed to the test-loss eval.")
    parser.add_argument("--gap_mode", type=str, default="auto",
                        choices=["auto", "hungarian", "sinkhorn",
                                 "subsample_hungarian"],
                        help="Estimator for the Monge gap. ``auto`` uses "
                             "Hungarian when eval_size <= 4096 and Sinkhorn "
                             "otherwise. ``subsample_hungarian`` runs K "
                             "exact Hungarian solves on random subsamples of "
                             "size M and reports median + IQR — robust to "
                             "the adaptive-ε artifacts that show up when "
                             "the push collapses or expands the cloud.")
    parser.add_argument("--gap_subsample_M", type=int, default=4096,
                        help="Subsample size for --gap_mode subsample_hungarian.")
    parser.add_argument("--gap_subsample_K", type=int, default=10,
                        help="Number of subsamples for --gap_mode "
                             "subsample_hungarian.")
    parser.add_argument("--sinkhorn_eps_frac", type=float, default=0.01,
                        help="Sinkhorn eps as a fraction of the median "
                             "pairwise sq-distance on the union of (z, Tz). "
                             "0.01 = paper default; raise to 0.05 for "
                             "tighter convergence at the cost of more bias "
                             "(canceled in the debiased Sinkhorn divergence).")
    parser.add_argument("--sinkhorn_n_iter", type=int, default=1000,
                        help="Sinkhorn iteration cap (log-domain). 1000 is "
                             "enough at eps_frac >= 0.01; raise to 5000 if "
                             "you push eps_frac below 0.005.")

    # LR-CIFAR10 backend
    lrc = parser.add_argument_group("lr_cifar10 backend")
    lrc.add_argument("--feature_cache", type=str, default=None,
                     help="Path to cifar10_resnet50_features.pth.")
    lrc.add_argument("--lr_cifar10_epoch", type=int, default=None,
                     help="Specific checkpoint epoch (default: highest available).")
    lrc.add_argument("--lr_cifar10_inner_lr", type=float, default=None)
    lrc.add_argument("--lr_cifar10_inner_steps", type=int, default=None)
    lrc.add_argument("--lr_cifar10_dual_eps", type=float, default=None,
                     help="eps_ent value used at training time (default: 0.02).")
    lrc.add_argument("--lr_cifar10_wfr_num_samples", type=int, default=None)
    lrc.add_argument("--lr_cifar10_sample_level", type=int, default=None)
    lrc.add_argument("--lr_cifar10_push_chunk", type=int, default=4096,
                     help="Anchor-batch chunk size for the LR-CIFAR10 push. "
                          "Bounds the per-class Brenier LAP cost in PPA and "
                          "caps activation memory in WFR/Dual. 0 disables. "
                          "Defaults to 4096; concatenation across chunks does "
                          "not change the final Monge-gap estimator.")

    # RL CartPole backend
    rl = parser.add_argument_group("rl_cartpole backend")
    rl.add_argument("--rl_env", type=str, default="cartpole",
                    choices=["cartpole", "swingup_pendulum"])
    rl.add_argument("--rl_env_max_steps", type=int, default=500)
    rl.add_argument("--rl_fd_episodes", type=int, default=1)
    rl.add_argument("--rl_fd_horizon", type=int, default=200)
    rl.add_argument("--rl_policy_hidden", type=int, default=64)
    rl.add_argument("--rl_eval_n_trials", type=int, default=200)

    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # The LS solvers and metrics expect float64.
    if args.dataset == "least_squares":
        torch.set_default_dtype(torch.float64)

    backend = DATASET_BACKENDS[args.dataset](args)
    supported = set(backend.supported_methods())
    for m in args.methods:
        if supported and m not in supported:
            raise ValueError(
                f"Method {m!r} not supported by {args.dataset!r} backend. "
                f"Supported: {sorted(supported)}."
            )

    rows: List[Dict[str, Any]] = []
    if args.out.exists():
        rows = pd.read_csv(args.out).to_dict("records")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lams = _flatten_lams(args.lams)
    n_total = len(args.methods) * len(lams) * len(args.seeds)
    counter = 0

    for method in args.methods:
        for lam in lams:
            for seed in args.seeds:
                counter += 1
                # Skip if already present.
                if any(
                    r["dataset"] == args.dataset
                    and r["method"] == method
                    and round(float(r["lam"]), 8) == lam
                    and int(r["seed"]) == int(seed)
                    and int(r.get("eval_size", -1)) == args.eval_size
                    for r in rows
                ):
                    print(f"[{counter}/{n_total}] skip {method} lam={lam} seed={seed} (already in CSV)")
                    continue

                print(f"[{counter}/{n_total}] {method} lam={lam} seed={seed}", flush=True)

                state = backend.train_or_load(
                    method=method, lam=lam, seed=seed,
                    eval_size=args.eval_size, device=device,
                    checkpoint_dir=args.checkpoint_dir,
                )

                z_hat = backend.sample_held_out_anchors(
                    args.eval_size, seed + 100_000, device,
                )
                Tz = backend.push(method, state, z_hat)

                # Optional per-backend hook: transform (z, Tz) into the space
                # where M(T) is well-defined for the method. Default identity.
                # rl_cartpole overrides this for npf/icnn (u-space encoding).
                gap_z, gap_Tz = z_hat, Tz
                if hasattr(backend, "gap_inputs"):
                    gap_z, gap_Tz = backend.gap_inputs(method, state, z_hat, Tz)

                gap_result = _gap_for_state(
                    gap_z, gap_Tz, args.eval_size,
                    mode=args.gap_mode,
                    sub_M=args.gap_subsample_M,
                    sub_K=args.gap_subsample_K,
                    sub_seed=int(seed) + 300_000,
                    sinkhorn_eps_frac=args.sinkhorn_eps_frac,
                    sinkhorn_n_iter=args.sinkhorn_n_iter,
                )

                test_loss = backend.compute_test_loss(
                    state, seed + 200_000, device, delta=args.shift_delta,
                )

                row = {
                    "dataset":   args.dataset,
                    "method":    method,
                    "lam":       lam,
                    "seed":      seed,
                    "eval_size": args.eval_size,
                    "test_loss": test_loss,
                    **gap_result,
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f"Done. Wrote {len(rows)} rows to {args.out}.")


if __name__ == "__main__":
    main()
