"""WDRO-RL training driver (refactored, registry-driven).

Runs one or more adversary algorithms against PPO on vectorized CartPole or
swing-up pendulum. Supported methods (see `algorithms/registry.py`):
    nominal, ro, particle, icnn, algo1, npf, ppa, new_ppa,
    dual, wgf, wfr, svg, rgo, nn_dro.

Convenience groups:
    --method both     # particle + icnn
    --method all      # every method in the registry

Usage:
    python RL_minimal.py --help
    python RL_minimal.py --method icnn --no-plot
    python RL_minimal.py --method nominal particle icnn --env cartpole --iters 100
    python RL_minimal.py --env swingup_pendulum --method all --no-plot
"""
from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms.registry import build_registry, training_title
from config import (
    InnerConfig,
    PPOConfig,
    build_arg_parser,
    build_inner_config_from_args,
    build_ppo_config_from_args,
    resolve_methods,
)
from envs import ENV_SPECS, canonical_env_name, make_vec_env
from models.policy import PolicyNet
from models.value import ValueNet
from utils.common import (
    freeze_params,
    log_line,
    resolve_device,
    set_seed,
    to_np,
    trange,
)
from utils.eval import evaluate_worst_case_grid, sample_hat_xi
from utils.io import choose_nonexisting_path, write_json
from utils.rollouts import evaluate_return_batch, ppo_update, rollout_batch
from utils.summaries import icnn_config_summary_lines, icnn_summary_lines

try:
    from render_gif import render_comparison_gif as _render_comparison_gif
except ImportError:
    _render_comparison_gif = None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    method: str = "particle",
    env_name: str = "cartpole",
    seed: int = 0,
    iters: int = 200,
    n_hat: int = 64,
    steps_per_xi: int = 64,
    device: str = "auto",
    env_max_steps: int = 500,
    pendulum_dt: float = 0.1,
    pendulum_u_max: float = 8.0,
    pendulum_max_speed: float = 8.0,
    pendulum_theta_tol: float = 0.2,
    pendulum_vel_tol: float = 1.0,
    pendulum_actions: int = 3,
    policy_hidden: int = 64,
    value_hidden: int = 64,
    log_every: int = 10,
    plot_samples: int = 512,
    eval_nom_episodes: int = 3,
    eval_nom_horizon: int = 500,
    eval_seed_nom: int = 999,
    worst_grid_n: int = 21,
    eval_worst_episodes: int = 1,
    eval_worst_horizon: int = 300,
    eval_seed_worst: int = 123,
    cfg_inner: Optional[InnerConfig] = None,
    cfg_ppo: Optional[PPOConfig] = None,
    save_gif: bool = False,
    gif_every: int = 10,
    gif_max_steps: int = 500,
    gif_fps: int = 20,
):
    set_seed(seed)
    env_name = canonical_env_name(env_name)
    device = resolve_device(device)
    log_line(
        f"[train] env={env_name} method={method} device={device.type} seed={seed} "
        f"iters={iters} n_hat={n_hat} steps_per_xi={steps_per_xi}"
    )

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if cfg_inner is None:
        cfg_inner = InnerConfig()
    if cfg_ppo is None:
        cfg_ppo = PPOConfig()

    spec = ENV_SPECS[env_name]
    log_line(
        f"[env] xi_names={spec.xi_names} xi_low={tuple(cfg_inner.xi_low)} "
        f"xi_high={tuple(cfg_inner.xi_high)}"
    )
    if env_name == "swingup_pendulum":
        log_line(
            "[env] pendulum "
            f"dt={float(pendulum_dt)} u_max={float(pendulum_u_max)} "
            f"max_speed={float(pendulum_max_speed)} theta_tol={float(pendulum_theta_tol)} "
            f"vel_tol={float(pendulum_vel_tol)} actions={int(pendulum_actions)}"
        )

    env_roll = make_vec_env(
        env_name, max_steps=int(env_max_steps), device=device,
        pendulum_dt=pendulum_dt, pendulum_u_max=pendulum_u_max,
        pendulum_max_speed=pendulum_max_speed,
        pendulum_theta_tol=pendulum_theta_tol,
        pendulum_vel_tol=pendulum_vel_tol,
        pendulum_actions=pendulum_actions,
    )
    env_eval = make_vec_env(
        env_name, max_steps=int(env_max_steps), device=device,
        pendulum_dt=pendulum_dt, pendulum_u_max=pendulum_u_max,
        pendulum_max_speed=pendulum_max_speed,
        pendulum_theta_tol=pendulum_theta_tol,
        pendulum_vel_tol=pendulum_vel_tol,
        pendulum_actions=pendulum_actions,
    )

    policy = PolicyNet(obs_dim=int(env_roll.obs_dim), hidden=policy_hidden, act_dim=int(env_roll.act_dim)).to(device)
    value = ValueNet(obs_dim=int(env_roll.obs_dim), hidden=value_hidden).to(device)

    pi_opt = torch.optim.SGD(
        policy.parameters(),
        lr=cfg_ppo.pi_lr, momentum=cfg_ppo.sgd_momentum,
        nesterov=True, weight_decay=cfg_ppo.sgd_weight_decay,
    )
    vf_opt = torch.optim.SGD(
        value.parameters(),
        lr=cfg_ppo.vf_lr, momentum=cfg_ppo.sgd_momentum,
        nesterov=True, weight_decay=cfg_ppo.sgd_weight_decay,
    )

    registry = build_registry()
    if method not in registry:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(registry)}.")
    display_name, factory = registry[method]
    adv = factory(cfg_inner, device)
    log_line(f"[train] adversary = {display_name}")

    icnn_mod = getattr(adv, "icnn", None)
    mlp_mod = getattr(adv, "mlp", None)
    if icnn_mod is not None:
        for line in icnn_summary_lines(icnn_mod):
            log_line(line)
    elif mlp_mod is not None:
        n_params = int(sum(p.numel() for p in mlp_mod.parameters()))
        log_line(
            f"[nn_dro] arch input_dim={getattr(mlp_mod, 'input_dim', '?')} "
            f"hidden_sizes={tuple(getattr(mlp_mod, 'hidden_sizes', ()))} "
            f"activation={getattr(mlp_mod, 'activation', '?')} "
            f"params={n_params:,}"
        )
    else:
        for line in icnn_config_summary_lines(
            hidden_sizes=cfg_inner.icnn_hidden_sizes,
            activation=cfg_inner.icnn_activation,
            strong_convexity=cfg_inner.icnn_strong_convexity,
            softplus_beta=cfg_inner.icnn_softplus_beta,
            nonneg_init=cfg_inner.icnn_nonneg_init,
            icnn_init=cfg_inner.icnn_init,
            device=device,
            input_dim=len(cfg_inner.xi_low),
            note=f"not instantiated (method={method})",
        ):
            log_line(line)

    logs = {"iter": [], "J_nominal": [], "J_worst_grid": [], "adv_cov_logdet": []}

    pbar = trange(iters, desc=f"train[{env_name}:{method}]", dynamic_ncols=True) if trange is not None else range(iters)
    for t in pbar:
        hat_xi = sample_hat_xi(cfg_inner, n_hat, seed=seed + 10_000 + t, device=device)

        with freeze_params(policy, value):
            xi_adv = adv.adversarial_xi(env_eval, policy, hat_xi, seed0=seed + 20_000 + t)

        if bool(getattr(adv, "outer_uses_adv_only", False)):
            xi_mixed = xi_adv
        else:
            xi_mixed = torch.cat([hat_xi, xi_adv], dim=0)
        data = rollout_batch(
            env_roll, policy, value, xi_mixed, steps_per_xi,
            seed0=seed + 30_000 + t, gamma=cfg_ppo.gamma, gae_lambda=cfg_ppo.gae_lambda,
        )

        # Freeze any parametric adversary module during the PPO update so its
        # weights can't pick up stale grads or be touched by the outer optim.
        adv_icnn = getattr(adv, "icnn", None)
        adv_mlp = getattr(adv, "mlp", None)
        with freeze_params(adv_icnn, adv_mlp):
            ppo_update(policy, value, data, cfg_ppo, pi_opt, vf_opt)

        if log_every > 0 and (t % int(log_every) == 0):
            low = torch.tensor(cfg_inner.xi_low, device=device, dtype=torch.float32)
            high = torch.tensor(cfg_inner.xi_high, device=device, dtype=torch.float32)
            xi0 = ((low + high) * 0.5).unsqueeze(0)
            J_nom = evaluate_return_batch(
                env_eval, policy, xi0,
                n_episodes=int(eval_nom_episodes), max_steps=int(eval_nom_horizon),
                seed0=int(eval_seed_nom), deterministic=True,
            ).item()
            J_worst = evaluate_worst_case_grid(
                env_eval, policy, cfg_inner,
                grid_n=int(worst_grid_n),
                n_episodes=int(eval_worst_episodes),
                max_steps=int(eval_worst_horizon),
                seed0=int(eval_seed_worst),
            )

            x = xi_adv - xi_adv.mean(dim=0, keepdim=True)
            cov = (x.T @ x) / max(1, (x.shape[0] - 1))
            cov = cov + 1e-6 * torch.eye(int(x.shape[1]), device=device)
            _sign, logdet = torch.linalg.slogdet(cov)
            logdet = logdet.item()

            logs["iter"].append(t)
            logs["J_nominal"].append(J_nom)
            logs["J_worst_grid"].append(J_worst)
            logs["adv_cov_logdet"].append(logdet)

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(J_nom=f"{J_nom:.1f}", J_worst=f"{J_worst:.1f}", logdet=f"{logdet:.2f}")
            log_line(f"[{method}] iter={t:04d}  J_nom={J_nom:.1f}  J_worst≈{J_worst:.1f}  logdetCov={logdet:.2f}")

        if save_gif and _render_comparison_gif is not None:
            is_last = (t == iters - 1)
            should_save = (t % max(1, gif_every) == 0) or is_last
            if should_save:
                gif_dir = Path(__file__).resolve().parent / "gifs" / f"{env_name}_{method}_seed{seed}"
                gif_path = gif_dir / f"iter_{t:04d}.gif"

                low_t = torch.tensor(cfg_inner.xi_low, device=device, dtype=torch.float32)
                high_t = torch.tensor(cfg_inner.xi_high, device=device, dtype=torch.float32)
                xi_nom_gif = (low_t + high_t) * 0.5

                with torch.no_grad():
                    J_adv_eval = evaluate_return_batch(
                        env_eval, policy, xi_adv,
                        n_episodes=1, max_steps=int(gif_max_steps),
                        seed0=seed + 60_000 + t, deterministic=True,
                    )
                    worst_idx = int(torch.argmin(J_adv_eval).item())
                    xi_worst_gif = xi_adv[worst_idx]

                def _make_render_env():
                    return make_vec_env(
                        env_name, max_steps=int(gif_max_steps), device=device,
                        pendulum_dt=pendulum_dt, pendulum_u_max=pendulum_u_max,
                        pendulum_max_speed=pendulum_max_speed,
                        pendulum_theta_tol=pendulum_theta_tol,
                        pendulum_vel_tol=pendulum_vel_tol,
                        pendulum_actions=pendulum_actions,
                    )

                with torch.no_grad():
                    _render_comparison_gif(
                        env_factory=_make_render_env,
                        policy=policy,
                        xi_nom=xi_nom_gif, xi_adv=xi_worst_gif,
                        out_path=gif_path, max_steps=int(gif_max_steps),
                        seed=seed + 50_000 + t, deterministic=True,
                        fps=gif_fps, iteration=t, method=method,
                        env_name=env_name, xi_names=spec.xi_names,
                    )
                log_line(
                    f"[gif] saved {gif_path}  "
                    f"(worst xi idx={worst_idx}, J={J_adv_eval[worst_idx]:.1f})"
                )

    with torch.no_grad():
        hat_xi_plot = sample_hat_xi(cfg_inner, int(plot_samples), seed=seed + 99_999, device=device)

    # Plot samples should not silently continue training a persistent
    # parametric adversary after the last PPO update and before checkpointing.
    adv_icnn = getattr(adv, "icnn", None)
    adv_mlp = getattr(adv, "mlp", None)
    adv_snapshot: Dict[str, Any] = {}
    if adv_icnn is not None:
        adv_snapshot["icnn"] = deepcopy(adv_icnn.state_dict())
    if adv_mlp is not None:
        adv_snapshot["mlp"] = deepcopy(adv_mlp.state_dict())
    if hasattr(adv, "bb_state"):
        adv_snapshot["bb_state"] = deepcopy(getattr(adv, "bb_state"))
    if hasattr(adv, "opt"):
        adv_snapshot["opt"] = deepcopy(getattr(adv, "opt").state_dict())

    try:
        with freeze_params(policy, value):
            xi_adv_plot = adv.adversarial_xi(env_eval, policy, hat_xi_plot, seed0=seed + 88_888)
    finally:
        if adv_icnn is not None and "icnn" in adv_snapshot:
            adv_icnn.load_state_dict(adv_snapshot["icnn"])
        if adv_mlp is not None and "mlp" in adv_snapshot:
            adv_mlp.load_state_dict(adv_snapshot["mlp"])
        if "bb_state" in adv_snapshot:
            setattr(adv, "bb_state", adv_snapshot["bb_state"])
        if hasattr(adv, "opt") and "opt" in adv_snapshot:
            getattr(adv, "opt").load_state_dict(adv_snapshot["opt"])

    return logs, to_np(hat_xi_plot), to_np(xi_adv_plot), policy, adv


def plot_results(logs, hat_xi, xi_adv, title: str, *, xi_names: Optional[Tuple[str, ...]] = None):
    it = logs["iter"]

    plt.figure()
    plt.plot(it, logs["J_nominal"], label="J nominal")
    plt.plot(it, logs["J_worst_grid"], label="J worst (grid approx)")
    plt.xlabel("iteration")
    plt.ylabel("return")
    plt.legend()
    plt.title(title)

    plt.figure()
    plt.plot(it, logs["adv_cov_logdet"])
    plt.xlabel("iteration")
    plt.ylabel("log det Cov(adv xi)")
    plt.title(title + " — diversity proxy")

    d = int(hat_xi.shape[1]) if getattr(hat_xi, "ndim", 0) == 2 and hat_xi.size else 0
    if xi_names is None or len(xi_names) != d:
        xi_names = tuple(f"xi{i}" for i in range(d))

    if d == 2:
        plt.figure()
        plt.scatter(hat_xi[:, 0], hat_xi[:, 1], s=8, alpha=0.5, label="hat_xi ~ P_hat")
        plt.scatter(xi_adv[:, 0], xi_adv[:, 1], s=8, alpha=0.5, label="adv xi")
        plt.xlabel(xi_names[0]); plt.ylabel(xi_names[1])
        plt.legend()
        plt.title(title + " — domain distribution")
    elif d == 3:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False, sharey=False)
        pairs = [(0, 1), (0, 2), (1, 2)]
        for ax, (i, j) in zip(axes, pairs):
            ax.scatter(hat_xi[:, i], hat_xi[:, j], s=8, alpha=0.35, label="hat_xi ~ P_hat")
            ax.scatter(xi_adv[:, i], xi_adv[:, j], s=8, alpha=0.6, label="adv xi")
            ax.set_xlabel(xi_names[i]); ax.set_ylabel(xi_names[j])
        axes[0].legend()
        fig.suptitle(title + " — domain distribution (pairwise)")
    else:
        plt.figure()
        plt.scatter(hat_xi[:, 0], hat_xi[:, 1], s=8, alpha=0.5, label="hat_xi ~ P_hat")
        plt.scatter(xi_adv[:, 0], xi_adv[:, 1], s=8, alpha=0.5, label="adv xi")
        plt.xlabel(xi_names[0] if d > 0 else "xi0")
        plt.ylabel(xi_names[1] if d > 1 else "xi1")
        plt.legend()
        plt.title(title + " — domain distribution (first two dims)")

    plt.show()


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.env = canonical_env_name(args.env)
    cfg_inner = build_inner_config_from_args(args, env_name=str(args.env))
    cfg_ppo = build_ppo_config_from_args(args)

    methods = resolve_methods(list(args.method))
    method_tag = methods[0] if len(methods) == 1 else (
        "both" if set(methods) == {"particle", "icnn"} else "custom"
    )

    methods_payload: Dict[str, Any] = {}
    for method in methods:
        logs, hat_xi, xi_adv, trained_policy, trained_adv = train(
            method=method,
            env_name=str(args.env),
            seed=int(args.seed),
            iters=int(args.iters),
            n_hat=int(args.n_hat),
            steps_per_xi=int(args.steps_per_xi),
            device=str(args.device),
            env_max_steps=int(args.env_max_steps),
            pendulum_dt=float(args.pendulum_dt),
            pendulum_u_max=float(args.pendulum_u_max),
            pendulum_max_speed=float(args.pendulum_max_speed),
            pendulum_theta_tol=float(args.pendulum_theta_tol),
            pendulum_vel_tol=float(args.pendulum_vel_tol),
            pendulum_actions=int(args.pendulum_actions),
            policy_hidden=int(args.policy_hidden),
            value_hidden=int(args.value_hidden),
            log_every=int(args.log_every),
            plot_samples=int(args.plot_samples),
            eval_nom_episodes=int(args.eval_nom_episodes),
            eval_nom_horizon=int(args.eval_nom_horizon),
            eval_seed_nom=int(args.eval_seed_nom),
            worst_grid_n=int(args.worst_grid_n),
            eval_worst_episodes=int(args.eval_worst_episodes),
            eval_worst_horizon=int(args.eval_worst_horizon),
            eval_seed_worst=int(args.eval_seed_worst),
            cfg_inner=cfg_inner,
            cfg_ppo=cfg_ppo,
            save_gif=bool(getattr(args, "save_gif", False)),
            gif_every=int(getattr(args, "gif_every", 10)),
            gif_max_steps=int(getattr(args, "gif_max_steps", 500)),
            gif_fps=int(getattr(args, "gif_fps", 20)),
        )

        methods_payload[str(method)] = {
            "logs": logs,
            "hat_xi_plot": hat_xi.tolist(),
            "xi_adv_plot": xi_adv.tolist(),
            "_policy": trained_policy,
            "_adv": trained_adv,
        }

        if not args.no_plot:
            env_spec = ENV_SPECS[str(args.env)]
            plot_results(logs, hat_xi, xi_adv, title=training_title(method, args.env), xi_names=env_spec.xi_names)

    if bool(getattr(args, "save_json", True)):
        created_utc = datetime.now(timezone.utc)
        meta_argv = [Path(sys.argv[0]).name] + (list(argv) if argv is not None else sys.argv[1:])
        out_dir = Path(__file__).resolve().parent
        ts_tag = created_utc.strftime("%Y%m%dT%H%M%SZ")

        if args.json_path is None:
            lam_tag = str(args.lam)
            beta_tag = str(args.icnn_softplus_beta)
            out_path = out_dir / (
                f"RL_minimal_{args.env}_{method_tag}_seed{int(args.seed)}"
                f"_lam_{lam_tag}_softplusbeta_{beta_tag}_{ts_tag}.json"
            )
        else:
            cand = Path(str(args.json_path))
            out_path = cand if cand.is_absolute() else (out_dir / cand)
        out_path = choose_nonexisting_path(out_path)

        payload: Dict[str, Any] = {
            "args": vars(args),
            "cfg_inner": asdict(cfg_inner),
            "cfg_ppo": asdict(cfg_ppo),
            "meta": {
                "argv": meta_argv,
                "created_utc": created_utc.isoformat(),
                "numpy": getattr(np, "__version__", "unknown"),
                "python": sys.version,
                "script": Path(__file__).name,
                "torch": getattr(torch, "__version__", "unknown"),
            },
            "methods": list(methods_payload.keys()),
            "runs": methods_payload,
        }

        for method_name, md in payload["runs"].items():
            policy_obj = md.pop("_policy", None)
            adv_obj = md.pop("_adv", None)
            if policy_obj is not None:
                ckpt_path = out_path.parent / f"{out_path.stem}_{method_name}_policy.pt"
                ckpt_payload: Dict[str, Any] = {
                    "policy_state_dict": policy_obj.state_dict(),
                    "obs_dim": policy_obj.net[0].in_features,
                    "act_dim": policy_obj.net[-1].out_features,
                    "hidden": policy_obj.net[0].out_features,
                    "env": str(args.env),
                    "method": method_name,
                    "seed": int(args.seed),
                    "lam": float(args.lam),
                    "iters": int(args.iters),
                    "cfg_inner": asdict(cfg_inner),
                }
                # Persist the trained parametric-adversary state so the Monge-gap
                # eval (monge_gap_sweep.py rl_cartpole backend) can reload the
                # exact trained transport map T_psi rather than re-running an
                # inner ascent from a freshly initialized adversary.
                # Non-parametric adversaries (particle, ppa, dual, wfr, ...)
                # carry no learned state, so we record adv_kind="none".
                if adv_obj is not None and getattr(adv_obj, "icnn", None) is not None:
                    icnn_mod = adv_obj.icnn
                    is_npf = type(icnn_mod).__name__ == "NPFInputConvexPotential"
                    ckpt_payload["adv_kind"] = "npf" if is_npf else "icnn"
                    ckpt_payload["adv_state_dict"] = icnn_mod.state_dict()
                    if is_npf:
                        ckpt_payload["adv_arch"] = {
                            "input_dim": int(len(cfg_inner.xi_low)),
                            "hidden_sizes": list(cfg_inner.npf_hidden_sizes),
                            "outer_rank": int(cfg_inner.npf_outer_rank),
                            "inner_rank": int(cfg_inner.npf_inner_rank),
                            "activation": str(cfg_inner.npf_activation),
                            "elu_alpha": float(cfg_inner.npf_elu_alpha),
                            "softplus_beta": float(cfg_inner.npf_softplus_beta),
                            "init_eps": float(cfg_inner.npf_init_eps),
                            "strong_convexity": float(cfg_inner.npf_strong_convexity),
                        }
                    else:
                        ckpt_payload["adv_arch"] = {
                            "input_dim": int(len(cfg_inner.xi_low)),
                            "hidden_sizes": list(cfg_inner.icnn_hidden_sizes),
                            "activation": str(cfg_inner.icnn_activation),
                            "strong_convexity": float(cfg_inner.icnn_strong_convexity),
                            "nonneg_init": str(cfg_inner.icnn_nonneg_init),
                            "softplus_beta": float(cfg_inner.icnn_softplus_beta),
                            "icnn_init": str(cfg_inner.icnn_init),
                        }
                elif adv_obj is not None and getattr(adv_obj, "mlp", None) is not None:
                    mlp_mod = adv_obj.mlp
                    ckpt_payload["adv_kind"] = "nn_dro"
                    ckpt_payload["adv_state_dict"] = mlp_mod.state_dict()
                    ckpt_payload["adv_arch"] = {
                        "input_dim": int(len(cfg_inner.xi_low)),
                        "hidden_sizes": list(cfg_inner.nn_dro_hidden_sizes),
                        "activation": str(cfg_inner.nn_dro_activation),
                        "softplus_beta": float(cfg_inner.nn_dro_softplus_beta),
                        "init_scale": float(cfg_inner.nn_dro_init_scale),
                    }
                else:
                    ckpt_payload["adv_kind"] = "none"
                # The xi-box defines the box bijection used by parametric
                # adversaries; persist alongside arch so the eval backend can
                # reconstruct without re-reading the JSON.
                ckpt_payload["xi_low"] = list(cfg_inner.xi_low)
                ckpt_payload["xi_high"] = list(cfg_inner.xi_high)
                torch.save(ckpt_payload, ckpt_path)
                log_line(f"[save] wrote checkpoint: {ckpt_path}")

        write_json(out_path, payload)
        log_line(f"[save] wrote JSON: {out_path}")
    else:
        for md in methods_payload.values():
            md.pop("_policy", None)
            md.pop("_adv", None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
