# latent_viz_decision_path.py
# --------------------------------------------------------
# Visualization helpers for "decision path" along u(t) = u0 + t * delta
# --------------------------------------------------------
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


@torch.no_grad()
def compute_decision_path(
    head: torch.nn.Module,
    u0: torch.Tensor,
    delta: torch.Tensor,
    y: torch.Tensor,
    T: int = 21,
    return_logits: bool = False,
) -> Dict[str, Any]:
    """
    Compute logits along the latent line u(t) = u0 + t*delta for t in [0,1].
    Returns statistics per t and per sample, plus the first flip time.

    Shapes:
        u0:    (B, D) or (B, C, H, W)  -- whatever your cut-layer latent is
        delta: same shape as u0
        y:     (B,) Long

    Output dict keys:
        'ts'            : (T,) float32 numpy array in [0,1]
        'true_logit'    : (T, B) numpy array (logit of true class)
        'max_logit'     : (T, B) numpy array (max class logit)
        'margin'        : (T, B) numpy array (true_logit - max_other_logit)
        'pred'          : (T, B) numpy array (argmax class id)
        'flip_t_idx'    : (B,) int array (first t index where pred != y, or -1 if never flips)
        'flip_t'        : (B,) float array in [0,1] (nan if no flip)
        (optional) 'logits': (T, B, C) numpy array (if return_logits=True)
    """
    device = u0.device
    B = u0.size(0)

    ts = torch.linspace(0.0, 1.0, T, device=device)
    true_logit_list = []
    max_logit_list = []
    margin_list = []
    pred_list = []
    logits_list = [] if return_logits else None

    for t in ts:
        u_t = u0 + t * delta
        logits = head(u_t)  # (B, C)

        # store logits if requested
        if return_logits:
            logits_list.append(logits.detach().cpu())

        # true-class logits
        tl = logits.gather(1, y.view(-1, 1)).squeeze(1)  # (B,)
        true_logit_list.append(tl.detach().cpu())

        # predicted / max logits
        maxl, _ = logits.max(dim=1)  # (B,)
        max_logit_list.append(maxl.detach().cpu())

        # margin = true_logit - max_other_logit
        # (avoid choosing the true class in max-other: set its logit to -inf)
        C = logits.size(1)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, y.view(-1, 1), True)
        logits_other = logits.masked_fill(mask, float("-inf"))
        max_other, _ = logits_other.max(dim=1)
        margin = tl - max_other
        margin_list.append(margin.detach().cpu())

        # predicted labels
        pred = logits.argmax(dim=1)
        pred_list.append(pred.detach().cpu())

    # stack over T
    true_logit = torch.stack(true_logit_list, dim=0).numpy()  # (T,B)
    max_logit = torch.stack(max_logit_list, dim=0).numpy()    # (T,B)
    margin = torch.stack(margin_list, dim=0).numpy()          # (T,B)
    pred = torch.stack(pred_list, dim=0).numpy()              # (T,B)
    ts_np = ts.detach().cpu().numpy()                         # (T,)

    # first flip time index per sample
    y_np = y.detach().cpu().numpy()
    flip_t_idx = np.full((B,), -1, dtype=np.int64)
    for b in range(B):
        diffs = (pred[:, b] != y_np[b])
        idx = np.argmax(diffs) if diffs.any() else -1
        flip_t_idx[b] = idx if idx != 0 or diffs[0] else -1  # handle argmax quirk

    # convert idx to continuous t (nan if never flips)
    flip_t = np.where(flip_t_idx >= 0, ts_np[flip_t_idx], np.nan)

    out = {
        "ts": ts_np,
        "true_logit": true_logit,
        "max_logit": max_logit,
        "margin": margin,
        "pred": pred,
        "flip_t_idx": flip_t_idx,
        "flip_t": flip_t,
    }
    if return_logits:
        out["logits"] = torch.stack(logits_list, dim=0).numpy()  # (T,B,C)
    return out


def _nice_ax(ax, xlabel: str, ylabel: str, title: Optional[str] = None, grid: bool = True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)


def plot_decision_path_mean(
    stats: Dict[str, Any],
    class_names: Optional[list] = None,
    title: str = "Decision path (batch mean)",
    savepath: Optional[str] = './fig/test.png',
    show: bool = False,
):
    """
    Plot batch-mean curves:
      - mean true-class logit
      - mean max logit (argmax)
      - mean margin (true - max_other)
    Also draws a vertical line at the mean flip t across samples that flip.
    """
    ts = stats["ts"]                  # (T,)
    tl = stats["true_logit"].mean(axis=1)  # (T,)
    ml = stats["max_logit"].mean(axis=1)   # (T,)
    mg = stats["margin"].mean(axis=1)      # (T,)
    flip_t = stats["flip_t"]               # (B,)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ts, tl, label="True-class logit (mean)")
    ax.plot(ts, ml, label="Max logit (mean)")
    ax.plot(ts, mg, label="Margin true−max_other (mean)")
    # mean flip t among samples that flip
    if np.isfinite(flip_t).any():
        mean_flip = np.nanmean(flip_t)
        ax.axvline(mean_flip, linestyle="--", linewidth=1.2, label=f"Mean flip t ≈ {mean_flip:.2f}")

    _nice_ax(ax, xlabel="t in [0,1] along u0 + t·δ*", ylabel="logit / margin", title=title)
    ax.legend(loc="best")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_decision_path_sample(
    stats: Dict[str, Any],
    sample_idx: int,
    title: Optional[str] = None,
    savepath: Optional[str] = './fig/test_sample.png',
    show: bool = False,
    also_plot_competitor: bool = True,
):
    """
    Plot the curves for a specific sample:
      - true-class logit
      - max logit
      - margin
      - first flip t (vertical line), if any
    Optionally also plot the *final* competitor class logit (the argmax at t=1 that differs from y).
    """
    ts = stats["ts"]                      # (T,)
    tl_b = stats["true_logit"][:, sample_idx]  # (T,)
    ml_b = stats["max_logit"][:, sample_idx]   # (T,)
    mg_b = stats["margin"][:, sample_idx]      # (T,)
    pred_b = stats["pred"][:, sample_idx]      # (T,)
    flip_t_b = stats["flip_t"][sample_idx]     # float or nan

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ts, tl_b, label="True-class logit")
    ax.plot(ts, ml_b, label="Max logit")
    ax.plot(ts, mg_b, label="Margin true−max_other")

    if also_plot_competitor and "logits" in stats:
        logits_tb = stats["logits"][:, sample_idx, :]  # (T,C)
        # competitor class = argmax at t=1 excluding true class
        C = logits_tb.shape[1]
        # identify true class from the margin: not directly stored; recover from margin logic
        # Instead, infer as class with highest logit at t=0 (usually equals true pred). If not robust, this is still fine for viz.
        y0_hat = int(np.argmax(logits_tb[0]))
        y1_hat = int(np.argmax(logits_tb[-1]))
        # If it already predicts the true class at t=0, try to choose the *most competitive* non-true class at t=1
        # Build competitor by masking the argmax at t=1 with the most competitive non-y0_hat class
        comp_mask = np.ones((C,), dtype=bool)
        comp_mask[y0_hat] = False
        comp_logits_t1 = logits_tb[-1, comp_mask]
        comp_ids = np.arange(C)[comp_mask]
        comp_id = int(comp_ids[np.argmax(comp_logits_t1)])
        comp_curve = logits_tb[:, comp_id]
        ax.plot(ts, comp_curve, linestyle="--", label=f"Competitor logit (class {comp_id})")

    if np.isfinite(flip_t_b):
        ax.axvline(flip_t_b, linestyle="--", linewidth=1.2, label=f"Flip t ≈ {flip_t_b:.2f}", color="k")

    _nice_ax(ax, xlabel="t in [0,1] along u0 + t·δ*", ylabel="logit / margin",
             title=title or f"Decision path (sample {sample_idx})")
    ax.legend(loc="best")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
