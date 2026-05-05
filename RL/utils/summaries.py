"""Pretty-printing helpers for ICNN architecture / sizing."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _format_num(x: float) -> str:
    if abs(x) >= 1e6:
        return f"{x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.0f}"


def icnn_summary_lines(icnn: nn.Module) -> list[str]:
    params = list(icnn.parameters())
    device = params[0].device if params else torch.device("cpu")
    dtype = params[0].dtype if params else torch.float32
    total_params = int(sum(p.numel() for p in params))
    trainable_params = int(sum(p.numel() for p in params if p.requires_grad))
    total_bytes = int(sum(p.numel() * p.element_size() for p in params))
    mib = total_bytes / (1024.0 ** 2)

    lines: list[str] = []
    is_npf = hasattr(icnn, "q_blocks") and hasattr(icnn, "b_linears")
    tag = "npf" if is_npf else "icnn"

    if is_npf:
        lines.append(
            "[npf] arch "
            f"input_dim={getattr(icnn, 'input_dim')} "
            f"hidden_sizes={tuple(getattr(icnn, 'hidden_sizes'))} "
            f"activation={getattr(icnn, 'activation', 'unknown')} "
            f"outer_rank={getattr(icnn, 'outer_rank', 'unknown')} "
            f"inner_rank={getattr(icnn, 'inner_rank', 'unknown')} "
            f"strong_convexity={getattr(icnn, 'strong_convexity', 'unknown')} "
            f"init_eps={getattr(icnn, 'init_eps', 'unknown')} "
            f"softplus_beta={getattr(icnn, 'softplus_beta', 'unknown')}"
        )
    elif hasattr(icnn, "input_dim") and hasattr(icnn, "hidden_sizes"):
        lines.append(
            "[icnn] arch "
            f"input_dim={getattr(icnn, 'input_dim')} "
            f"hidden_sizes={tuple(getattr(icnn, 'hidden_sizes'))} "
            f"activation={getattr(icnn, 'activation', 'unknown')} "
            f"strong_convexity={getattr(icnn, 'strong_convexity', 'unknown')} "
            f"softplus_beta={getattr(icnn, 'softplus_beta', 'unknown')}"
        )
    else:
        lines.append("[icnn] arch (unknown module type)")

    lines.append(
        f"[{tag}] size "
        f"params={total_params:,} ({_format_num(total_params)}) "
        f"trainable={trainable_params:,} "
        f"bytes={total_bytes:,} (~{mib:.2f} MiB) "
        f"device={device.type} dtype={str(dtype).replace('torch.', '')}"
    )

    lines.append(f"[{tag}] layers:")
    if is_npf:
        q_blocks = getattr(icnn, "q_blocks")
        b_linears = getattr(icnn, "b_linears")
        for i, (q, lin) in enumerate(zip(q_blocks, b_linears)):
            lines.append(
                f"  q{i}: QuadraticForms(num_forms={getattr(q, 'num_forms', '?')}, "
                f"rank={getattr(q, 'rank', '?')}) + "
                f"b{i}: Linear({lin.in_features} -> {lin.out_features}, bias={lin.bias is not None})"
            )
        if hasattr(icnn, "w_linears"):
            for i, lin in enumerate(getattr(icnn, "w_linears")):
                if lin is None:
                    continue
                if hasattr(lin, "in_features") and hasattr(lin, "out_features"):
                    lines.append(
                        f"  w{i}: NPFNonNegativeDense({lin.in_features} -> {lin.out_features}, bias=False, param=exp)"
                    )
        if hasattr(icnn, "w_out"):
            lin = getattr(icnn, "w_out")
            lines.append(
                f"  w_out: NPFNonNegativeDense({lin.in_features} -> {lin.out_features}, bias=False, param=exp)"
            )
        if hasattr(icnn, "q_out"):
            q = getattr(icnn, "q_out")
            lines.append(
                f"  q_out: QuadraticForms(num_forms={getattr(q, 'num_forms', '?')}, rank={getattr(q, 'rank', '?')})"
            )
        if hasattr(icnn, "b_out"):
            lin = getattr(icnn, "b_out")
            lines.append(f"  b_out: Linear({lin.in_features} -> {lin.out_features}, bias={lin.bias is not None})")
        lines.append("  outer: 0.5*mu*||z||^2 + 0.5*z^T(diag(delta^2)+A^T A)z + a^Tz")
    elif hasattr(icnn, "z_linears"):
        for i, lin in enumerate(getattr(icnn, "z_linears")):
            if hasattr(lin, "in_features") and hasattr(lin, "out_features"):
                lines.append(f"  z{i}: Linear({lin.in_features} -> {lin.out_features}, bias={lin.bias is not None})")
    if hasattr(icnn, "h_linears"):
        for i, lin in enumerate(getattr(icnn, "h_linears")):
            if lin is None:
                continue
            if hasattr(lin, "in_features") and hasattr(lin, "out_features"):
                bias = getattr(lin, "use_bias", getattr(lin, "bias", None) is not None)
                param = getattr(lin, "parametrization", "unknown")
                lines.append(f"  h{i}: NonNegativeLinear({lin.in_features} -> {lin.out_features}, bias={bias}, param={param})")
    if hasattr(icnn, "hidden_output"):
        lin = getattr(icnn, "hidden_output")
        if hasattr(lin, "in_features") and hasattr(lin, "out_features"):
            bias = getattr(lin, "use_bias", getattr(lin, "bias", None) is not None)
            param = getattr(lin, "parametrization", "unknown")
            lines.append(f"  out: NonNegativeLinear({lin.in_features} -> {lin.out_features}, bias={bias}, param={param})")
    if hasattr(icnn, "input_skip"):
        lin = getattr(icnn, "input_skip")
        if hasattr(lin, "in_features") and hasattr(lin, "out_features"):
            lines.append(f"  skip: Linear({lin.in_features} -> {lin.out_features}, bias={lin.bias is not None})")

    return lines


def icnn_config_summary_lines(
    *,
    hidden_sizes: Tuple[int, ...],
    activation: str,
    strong_convexity: float,
    softplus_beta: float,
    nonneg_init: str,
    icnn_init: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    input_dim: int = 2,
    note: str = "",
) -> list[str]:
    hs = tuple(int(v) for v in hidden_sizes)
    if len(hs) == 0:
        raise ValueError("icnn_config_summary_lines requires at least one hidden layer.")

    z_params = sum(w * input_dim + w for w in hs)
    h_params = sum(prev * cur for prev, cur in zip(hs[:-1], hs[1:]))
    out_params = hs[-1] * 1 + 1
    skip_params = input_dim * 1 + 1
    total_params = int(z_params + h_params + out_params + skip_params)

    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = int(total_params * element_size)
    mib = total_bytes / (1024.0 ** 2)
    parametrization = "exp" if str(nonneg_init).lower() == "principled" else "softplus"

    lines: list[str] = []
    suffix = f" ({note})" if note else ""
    lines.append(
        "[icnn] arch "
        f"input_dim={int(input_dim)} "
        f"hidden_sizes={hs} "
        f"activation={activation} "
        f"strong_convexity={float(strong_convexity)} "
        f"softplus_beta={float(softplus_beta)} "
        f"nonneg_init={nonneg_init} "
        f"init={icnn_init}{suffix}"
    )
    lines.append(
        "[icnn] size "
        f"params={total_params:,} ({_format_num(total_params)}) "
        f"bytes≈{total_bytes:,} (~{mib:.2f} MiB; assuming {str(dtype).replace('torch.', '')}) "
        f"device={device.type}"
    )
    lines.append("[icnn] layers:")
    for i, w in enumerate(hs):
        lines.append(f"  z{i}: Linear({input_dim} -> {w}, bias=True)")
    for i, (prev, cur) in enumerate(zip(hs[:-1], hs[1:]), start=1):
        lines.append(f"  h{i}: NonNegativeLinear({prev} -> {cur}, bias=False, param={parametrization})")
    lines.append(f"  out: NonNegativeLinear({hs[-1]} -> 1, bias=True, param={parametrization})")
    lines.append(f"  skip: Linear({input_dim} -> 1, bias=True)")
    lines.append(f"  + quadratic term: 0.5*{float(strong_convexity)}*||z||^2")
    return lines
