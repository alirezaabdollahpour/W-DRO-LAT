"""
Professional GIF renderer for RL agent behavior visualization.

Renders side-by-side comparison GIFs showing agent performance under
nominal vs adversarial (worst-case) physics parameters.

Usage (standalone):
    python render_gif.py --help

Programmatic (from RL_minimal.py):
    from render_gif import render_comparison_gif
    render_comparison_gif(env_factory, policy, xi_nom, xi_adv, out_path, ...)
"""

import math
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None


# ---------------------------------------------------------------------------
# Layout & color constants
# ---------------------------------------------------------------------------

PANEL_W = 520
PANEL_H = 360
PENDULUM_PANEL_W = 420
PENDULUM_PANEL_H = 420
HUD_H = 76
DIVIDER_W = 4
LABEL_H = 24

# Colors
COL_BG            = (245, 245, 248)
COL_RAIL          = (170, 170, 170)
COL_CART          = (50, 100, 200)
COL_CART_OUTLINE  = (30, 60, 140)
COL_POLE          = (200, 50, 50)
COL_POLE_TIP      = (220, 80, 80)
COL_WHEEL         = (60, 60, 60)
COL_DANGER_LINE   = (200, 50, 50)
COL_DANGER_ZONE   = (255, 210, 210)
COL_SAFE_GREEN    = (60, 180, 80)
COL_WARN_YELLOW   = (220, 180, 40)
COL_WARN_RED      = (220, 60, 50)
COL_FORCE_ARROW   = (0, 120, 200)
COL_HUD_BG        = (35, 35, 45)
COL_HUD_TEXT      = (220, 220, 225)
COL_HUD_DIM       = (140, 140, 150)
COL_LABEL_NOM     = (55, 130, 210)
COL_LABEL_ADV     = (210, 70, 55)
COL_DIVIDER       = (60, 60, 70)
COL_REWARD_BAR_BG = (70, 70, 80)
COL_REWARD_BAR_FG = (80, 195, 115)
COL_STAMP_OK      = (40, 170, 80)
COL_STAMP_FAIL    = (210, 50, 50)
COL_HIGHLIGHT     = (255, 150, 60)
COL_GAUGE_BG      = (210, 210, 215)
COL_GAUGE_THRESH  = (200, 60, 60)
COL_PENDULUM_PIVOT = (100, 100, 100)
COL_PENDULUM_ROD   = (60, 60, 60)
COL_PENDULUM_BOB   = (200, 50, 50)
COL_PENDULUM_TARGET = (100, 200, 100)
COL_TORQUE_ARROW   = (0, 120, 200)


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------

_FONT_CACHE: Dict = {}


def _get_font(size: int = 13, bold: bool = False):
    key = (size, bold)
    if key not in _FONT_CACHE:
        candidates = []
        if bold:
            candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
            candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
        candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        font = None
        for p in candidates:
            try:
                font = ImageFont.truetype(p, size)
                break
            except Exception:
                continue
        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
        _FONT_CACHE[key] = font
    return _FONT_CACHE[key]


def _text_size(draw, text, font):
    """Get (width, height) of text."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _proximity_color(
    x: float, theta: float,
    x_thresh: float = 2.4,
    theta_thresh: float = 12 * math.pi / 180,
) -> Tuple[int, int, int]:
    """Green→yellow→red based on proximity to CartPole failure boundaries."""
    prox = max(abs(x) / max(x_thresh, 1e-9), abs(theta) / max(theta_thresh, 1e-9))
    prox = min(prox, 1.0)
    if prox < 0.5:
        t = prox / 0.5
        c0, c1 = COL_SAFE_GREEN, COL_WARN_YELLOW
    else:
        t = (prox - 0.5) / 0.5
        c0, c1 = COL_WARN_YELLOW, COL_WARN_RED
    return tuple(int(c0[i] + t * (c1[i] - c0[i])) for i in range(3))


def _lerp_color(c0, c1, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c0[i] + t * (c1[i] - c0[i])) for i in range(min(len(c0), len(c1))))


# ---------------------------------------------------------------------------
# Dashed-line helper
# ---------------------------------------------------------------------------

def _draw_dashed_vline(draw, x, y0, y1, color, width=1, dash=8, gap=5):
    y = y0
    while y < y1:
        ye = min(y + dash, y1)
        draw.line([(x, y), (x, ye)], fill=color, width=width)
        y = ye + gap


# ---------------------------------------------------------------------------
# CartPole frame renderer
# ---------------------------------------------------------------------------

def _render_cartpole_frame(
    state: np.ndarray,
    *,
    width: int = PANEL_W,
    height: int = PANEL_H,
    cart_w: int = 60,
    cart_h: int = 30,
    pole_len_px: int = 120,
    pole_w: int = 6,
    action: Optional[int] = None,
    x_viewport: float = 4.0,
) -> np.ndarray:
    """Render a single CartPole frame with danger zones, force arrow, theta gauge."""
    if Image is None:
        raise ImportError("Pillow is required: pip install Pillow")

    img = Image.new("RGB", (width, height), COL_BG)
    draw = ImageDraw.Draw(img)

    x = float(state[0])
    theta = float(state[2])

    rail_y = int(height * 0.62)

    # --- danger zones (shaded regions beyond x=±2.4) ---
    x_thresh = 2.4
    left_bnd_px = int((-x_thresh + x_viewport) / (2 * x_viewport) * width)
    right_bnd_px = int((x_thresh + x_viewport) / (2 * x_viewport) * width)
    zone_top = rail_y - pole_len_px - 30
    zone_bot = rail_y + cart_h // 2 + 30
    draw.rectangle([0, zone_top, left_bnd_px, zone_bot], fill=COL_DANGER_ZONE)
    draw.rectangle([right_bnd_px, zone_top, width, zone_bot], fill=COL_DANGER_ZONE)

    # dashed boundary lines
    _draw_dashed_vline(draw, left_bnd_px, zone_top, zone_bot, COL_DANGER_LINE, width=2)
    _draw_dashed_vline(draw, right_bnd_px, zone_top, zone_bot, COL_DANGER_LINE, width=2)

    # --- rail ---
    draw.line([(0, rail_y), (width, rail_y)], fill=COL_RAIL, width=2)

    # --- cart position ---
    x_frac = (x + x_viewport) / (2 * x_viewport)
    cx = int(x_frac * width)
    cy = rail_y

    # proximity color for cart outline
    prox_col = _proximity_color(x, theta)

    # cart body
    draw.rectangle(
        [cx - cart_w // 2, cy - cart_h // 2, cx + cart_w // 2, cy + cart_h // 2],
        fill=COL_CART,
        outline=prox_col,
        width=3,
    )

    # wheels
    wh_r = 6
    for dx in [-cart_w // 3, cart_w // 3]:
        wx, wy = cx + dx, cy + cart_h // 2 + wh_r
        draw.ellipse([wx - wh_r, wy - wh_r, wx + wh_r, wy + wh_r], fill=COL_WHEEL)

    # --- pole ---
    px = cx + int(pole_len_px * math.sin(theta))
    py = cy - int(pole_len_px * math.cos(theta))
    draw.line([(cx, cy - cart_h // 4), (px, py)], fill=COL_POLE, width=pole_w)
    draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=COL_POLE_TIP)

    # --- force arrow ---
    if action is not None:
        arrow_y = cy + cart_h // 2 + 18
        arrow_len = 35
        if action == 0:  # left
            ax_s, ax_e = cx + 5, cx - arrow_len
        else:  # right
            ax_s, ax_e = cx - 5, cx + arrow_len
        draw.line([(ax_s, arrow_y), (ax_e, arrow_y)], fill=COL_FORCE_ARROW, width=3)
        sign = -1 if action == 0 else 1
        for dy in [-5, 5]:
            draw.line([(ax_e, arrow_y), (ax_e - sign * 8, arrow_y + dy)],
                      fill=COL_FORCE_ARROW, width=2)

    # --- theta gauge (top-right corner) ---
    gauge_cx = width - 38
    gauge_cy = 32
    gauge_r = 22
    theta_max_deg = 12.0
    draw_range_deg = 20.0

    # background arc
    draw.arc(
        [gauge_cx - gauge_r, gauge_cy - gauge_r, gauge_cx + gauge_r, gauge_cy + gauge_r],
        start=-90 - draw_range_deg, end=-90 + draw_range_deg,
        fill=COL_GAUGE_BG, width=3,
    )

    # threshold tick marks at ±12 deg
    for sign in [-1, 1]:
        tick_angle = math.radians(-90 + sign * theta_max_deg)
        tx0 = gauge_cx + int((gauge_r - 4) * math.cos(tick_angle))
        ty0 = gauge_cy + int((gauge_r - 4) * math.sin(tick_angle))
        tx1 = gauge_cx + int((gauge_r + 4) * math.cos(tick_angle))
        ty1 = gauge_cy + int((gauge_r + 4) * math.sin(tick_angle))
        draw.line([(tx0, ty0), (tx1, ty1)], fill=COL_GAUGE_THRESH, width=2)

    # needle (current theta in degrees)
    theta_deg = math.degrees(theta)
    theta_deg_clamped = max(-draw_range_deg, min(draw_range_deg, theta_deg))
    needle_angle = math.radians(-90 + theta_deg_clamped)
    nx = gauge_cx + int((gauge_r - 2) * math.cos(needle_angle))
    ny = gauge_cy + int((gauge_r - 2) * math.sin(needle_angle))
    needle_col = _proximity_color(0, theta)  # only theta component
    draw.line([(gauge_cx, gauge_cy), (nx, ny)], fill=needle_col, width=2)
    draw.ellipse([gauge_cx - 3, gauge_cy - 3, gauge_cx + 3, gauge_cy + 3], fill=needle_col)

    # gauge label
    font_tiny = _get_font(9)
    if font_tiny:
        draw.text((gauge_cx - gauge_r - 2, gauge_cy + gauge_r + 2), "theta", fill=COL_HUD_DIM, font=font_tiny)

    return np.array(img)


# ---------------------------------------------------------------------------
# Pendulum frame renderer
# ---------------------------------------------------------------------------

def _render_pendulum_frame(
    state: np.ndarray,
    *,
    width: int = PENDULUM_PANEL_W,
    height: int = PENDULUM_PANEL_H,
    rod_len_px: int = 140,
    rod_w: int = 5,
    torque: Optional[float] = None,
    u_max: float = 8.0,
) -> np.ndarray:
    """Render a single Pendulum frame. theta=0 hanging down, theta=pi upright."""
    if Image is None:
        raise ImportError("Pillow is required: pip install Pillow")

    img = Image.new("RGB", (width, height), COL_BG)
    draw = ImageDraw.Draw(img)

    theta = float(state[0])
    cx, cy = width // 2, height // 2

    # range-of-motion guide circle
    draw.arc(
        [cx - rod_len_px - 5, cy - rod_len_px - 5,
         cx + rod_len_px + 5, cy + rod_len_px + 5],
        start=0, end=360, fill=(220, 220, 225), width=1,
    )

    # pivot
    draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=COL_PENDULUM_PIVOT)

    # rod + bob (theta=0 → hanging down → bob at bottom)
    bx = cx + int(rod_len_px * math.sin(theta))
    by = cy + int(rod_len_px * math.cos(theta))
    draw.line([(cx, cy), (bx, by)], fill=COL_PENDULUM_ROD, width=rod_w)

    # bob
    bob_r = 14
    draw.ellipse([bx - bob_r, by - bob_r, bx + bob_r, by + bob_r], fill=COL_PENDULUM_BOB)

    # upright target (top)
    target_bx = cx
    target_by = cy - rod_len_px
    tr = 16
    draw.ellipse(
        [target_bx - tr, target_by - tr, target_bx + tr, target_by + tr],
        outline=COL_PENDULUM_TARGET, width=2,
    )

    # torque arrow
    if torque is not None:
        arrow_scale = 50.0
        arrow_len = int(arrow_scale * abs(torque) / max(u_max, 1e-6))
        arrow_len = min(arrow_len, 80)
        if arrow_len > 3:
            sign = 1 if torque > 0 else -1
            ax_start = cx - sign * 15
            ax_end = cx - sign * (15 + arrow_len)
            ay = cy + rod_len_px + 30
            draw.line([(ax_start, ay), (ax_end, ay)], fill=COL_TORQUE_ARROW, width=3)
            for dy in [-5, 5]:
                draw.line([(ax_end, ay), (ax_end + sign * 8, ay + dy)],
                          fill=COL_TORQUE_ARROW, width=2)

    return np.array(img)


# ---------------------------------------------------------------------------
# HUD renderer
# ---------------------------------------------------------------------------

def _draw_hud(
    draw: "ImageDraw.Draw",
    *,
    total_width: int,
    hud_height: int,
    step: int,
    max_steps: int,
    reward_nom: float,
    reward_adv: float,
    max_reward: float,
    xi_nom_dict: Dict[str, float],
    xi_adv_dict: Dict[str, float],
    iteration: Optional[int],
    method: Optional[str],
    env_name: str,
) -> None:
    """Draw the HUD bar across the top of the composite frame."""
    font = _get_font(12)
    font_bold = _get_font(12, bold=True)
    font_sm = _get_font(10)

    # background
    draw.rectangle([0, 0, total_width, hud_height], fill=COL_HUD_BG)

    pad_x = 10
    # --- Row 1: header info ---
    y1 = 5
    header_parts = []
    if iteration is not None:
        header_parts.append(f"iter {iteration}")
    if method:
        header_parts.append(method)
    header_parts.append(env_name)
    header_left = "  |  ".join(header_parts)
    draw.text((pad_x, y1), header_left, fill=COL_HUD_TEXT, font=font_bold)

    step_text = f"Step {step}/{max_steps}"
    tw, _ = _text_size(draw, step_text, font_bold)
    draw.text((total_width - tw - pad_x, y1), step_text, fill=COL_HUD_TEXT, font=font_bold)

    # --- Row 2: Nominal xi + reward ---
    y2 = y1 + 20
    draw.text((pad_x, y2), "NOM", fill=COL_LABEL_NOM, font=font_bold)
    xi_x = pad_x + 38
    for name, val in xi_nom_dict.items():
        txt = f"{name}={val:.3f}"
        draw.text((xi_x, y2), txt, fill=COL_HUD_TEXT, font=font_sm)
        tw, _ = _text_size(draw, txt, font_sm)
        xi_x += tw + 10

    # reward bar
    _draw_reward_bar(draw, total_width - 170, y2 + 1, 120, 12,
                     reward_nom, max_reward, COL_REWARD_BAR_FG, font_sm)

    # --- Row 3: Adversarial xi + reward ---
    y3 = y2 + 20
    draw.text((pad_x, y3), "ADV", fill=COL_LABEL_ADV, font=font_bold)
    xi_x = pad_x + 38
    for name in xi_adv_dict:
        adv_val = xi_adv_dict[name]
        nom_val = xi_nom_dict.get(name, adv_val)
        diff_frac = abs(adv_val - nom_val) / max(abs(nom_val), 1e-6)
        txt = f"{name}={adv_val:.3f}"
        col = COL_HIGHLIGHT if diff_frac > 0.1 else COL_HUD_TEXT
        draw.text((xi_x, y3), txt, fill=col, font=font_sm)
        tw, _ = _text_size(draw, txt, font_sm)
        xi_x += tw + 10

    _draw_reward_bar(draw, total_width - 170, y3 + 1, 120, 12,
                     reward_adv, max_reward, COL_REWARD_BAR_FG, font_sm)


def _draw_reward_bar(draw, x, y, w, h, reward, max_reward, color, font):
    """Draw a reward progress bar."""
    frac = min(1.0, max(0.0, reward / max(max_reward, 1e-6)))
    draw.rectangle([x, y, x + w, y + h], fill=COL_REWARD_BAR_BG)
    if frac > 0:
        fill_col = _lerp_color(COL_WARN_RED, color, frac)
        draw.rectangle([x, y, x + int(w * frac), y + h], fill=fill_col)
    draw.rectangle([x, y, x + w, y + h], outline=(90, 90, 100), width=1)
    pct_text = f"R={reward:.0f}"
    draw.text((x + w + 5, y - 1), pct_text, fill=COL_HUD_TEXT, font=font)


# ---------------------------------------------------------------------------
# Panel label
# ---------------------------------------------------------------------------

def _draw_panel_label(
    draw: "ImageDraw.Draw",
    text: str,
    color: Tuple[int, int, int],
    x_offset: int,
    y_offset: int,
    panel_width: int,
    label_height: int = LABEL_H,
) -> None:
    """Draw a colored label banner above a panel."""
    draw.rectangle(
        [x_offset, y_offset, x_offset + panel_width, y_offset + label_height],
        fill=color,
    )
    font = _get_font(13, bold=True)
    tw, th = _text_size(draw, text, font)
    tx = x_offset + (panel_width - tw) // 2
    ty = y_offset + (label_height - th) // 2
    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)


# ---------------------------------------------------------------------------
# Outcome stamp
# ---------------------------------------------------------------------------

def _draw_outcome_stamp(
    draw: "ImageDraw.Draw",
    outcome: str,
    survived: bool,
    x_offset: int,
    y_offset: int,
    panel_width: int,
    panel_height: int,
) -> None:
    """Draw a large outcome stamp on a panel."""
    stamp_w = int(panel_width * 0.8)
    stamp_h = 44
    sx = x_offset + (panel_width - stamp_w) // 2
    sy = y_offset + (panel_height - stamp_h) // 2
    bg_col = COL_STAMP_OK if survived else COL_STAMP_FAIL

    # semi-transparent effect: draw filled rect, then slightly lighter inner
    draw.rectangle([sx, sy, sx + stamp_w, sy + stamp_h], fill=bg_col, outline=(255, 255, 255), width=2)
    inner_col = _lerp_color(bg_col, (255, 255, 255), 0.15)
    draw.rectangle([sx + 3, sy + 3, sx + stamp_w - 3, sy + stamp_h - 3], fill=inner_col)

    font = _get_font(16, bold=True)
    tw, th = _text_size(draw, outcome, font)
    tx = sx + (stamp_w - tw) // 2
    ty = sy + (stamp_h - th) // 2
    draw.text((tx, ty), outcome, fill=(255, 255, 255), font=font)


# ---------------------------------------------------------------------------
# Episode rollout helper (no rendering)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rollout_episode(
    env,
    policy,
    xi: torch.Tensor,
    *,
    max_steps: int = 500,
    seed: int = 0,
    deterministic: bool = True,
) -> Dict:
    """
    Roll out one episode, return trajectory data.

    Returns dict with keys:
        states:             list of np.ndarray
        actions:            list of int
        rewards:            list of float
        cumulative_rewards: list of float
        torques:            list of Optional[float]  (for pendulum)
        done_step:          int or None
        survived:           bool
        total_reward:       float
    """
    device = next(policy.parameters()).device
    xi = xi.to(device)
    if xi.dim() == 1:
        xi = xi.unsqueeze(0)

    is_pendulum = "Pendulum" in type(env).__name__ or "SwingUp" in type(env).__name__

    env.reset(1, seed=seed)
    states, actions, rewards, cum_rewards, torques = [], [], [], [], []
    total_reward = 0.0
    done_step = None

    for t in range(max_steps):
        obs = env.state
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        env.state = obs

        logits = policy(obs)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        if deterministic:
            a = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            a = torch.multinomial(probs, num_samples=1).squeeze(-1)

        a_int = int(a[0].item())
        torque_val = None
        if is_pendulum and hasattr(env, "action_values"):
            torque_val = float(env.action_values[a[0]].item())

        states.append(obs[0].detach().cpu().numpy())
        actions.append(a_int)
        torques.append(torque_val)

        _, r, d, info = env.step(a, xi)
        rew = float(r[0].item())
        total_reward += rew
        rewards.append(rew)
        cum_rewards.append(total_reward)

        terminated = info.get("terminated", torch.zeros(1, dtype=torch.bool))
        if d[0].item():
            done_step = t + 1
            if terminated[0].item():
                # true failure, not just truncation
                break
            else:
                # truncated (survived to time limit)
                break

    survived = done_step is None or (not info.get("terminated", torch.zeros(1, dtype=torch.bool))[0].item())

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "cumulative_rewards": cum_rewards,
        "torques": torques,
        "done_step": done_step,
        "survived": survived,
        "total_reward": total_reward,
    }


# ---------------------------------------------------------------------------
# Side-by-side comparison GIF (main entry point)
# ---------------------------------------------------------------------------

def render_comparison_gif(
    env_factory: Callable,
    policy,
    xi_nom: torch.Tensor,
    xi_adv: torch.Tensor,
    out_path,
    *,
    max_steps: int = 500,
    seed: int = 0,
    deterministic: bool = True,
    fps: int = 20,
    iteration: Optional[int] = None,
    method: Optional[str] = None,
    env_name: str = "cartpole",
    xi_names: Optional[Tuple[str, ...]] = None,
    stamp_frames: int = 25,
) -> Path:
    """
    Render a side-by-side comparison GIF: nominal (left) vs adversarial (right).
    """
    if Image is None:
        raise ImportError("Pillow is required: pip install Pillow")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ensure 1D
    if xi_nom.dim() > 1:
        xi_nom = xi_nom.squeeze(0)
    if xi_adv.dim() > 1:
        xi_adv = xi_adv.squeeze(0)

    # detect env type
    is_pendulum = "pendulum" in env_name.lower() or "swing" in env_name.lower()

    # roll out both episodes
    env_nom = env_factory()
    env_adv = env_factory()
    traj_nom = _rollout_episode(env_nom, policy, xi_nom, max_steps=max_steps,
                                seed=seed, deterministic=deterministic)
    traj_adv = _rollout_episode(env_adv, policy, xi_adv, max_steps=max_steps,
                                seed=seed, deterministic=deterministic)

    # panel dimensions
    if is_pendulum:
        pw, ph = PENDULUM_PANEL_W, PENDULUM_PANEL_H
    else:
        pw, ph = PANEL_W, PANEL_H

    total_w = pw * 2 + DIVIDER_W
    total_h = HUD_H + LABEL_H + ph

    # xi dicts for HUD
    xi_dim = xi_nom.shape[0]
    if xi_names is None or len(xi_names) != xi_dim:
        xi_names = tuple(f"xi{i}" for i in range(xi_dim))
    xi_nom_dict = {n: float(xi_nom[i].item()) for i, n in enumerate(xi_names)}
    xi_adv_dict = {n: float(xi_adv[i].item()) for i, n in enumerate(xi_names)}

    max_reward = float(max_steps)

    # outcome strings
    nom_outcome = (f"SURVIVED ({len(traj_nom['states'])} steps)"
                   if traj_nom["survived"]
                   else f"FAILED at step {traj_nom['done_step']}")
    adv_outcome = (f"SURVIVED ({len(traj_adv['states'])} steps)"
                   if traj_adv["survived"]
                   else f"FAILED at step {traj_adv['done_step']}")

    # u_max for pendulum torque display
    u_max_val = 8.0
    if is_pendulum and hasattr(env_nom, "u_max"):
        u_max_val = float(env_nom.u_max.item())

    n_nom = len(traj_nom["states"])
    n_adv = len(traj_adv["states"])
    n_frames = max(n_nom, n_adv)

    frames: List[Image.Image] = []

    for t in range(n_frames):
        composite = Image.new("RGB", (total_w, total_h), COL_BG)
        draw = ImageDraw.Draw(composite)

        # current rewards
        r_nom = traj_nom["cumulative_rewards"][min(t, n_nom - 1)] if n_nom > 0 else 0.0
        r_adv = traj_adv["cumulative_rewards"][min(t, n_adv - 1)] if n_adv > 0 else 0.0

        # HUD
        _draw_hud(
            draw, total_width=total_w, hud_height=HUD_H,
            step=t, max_steps=max_steps,
            reward_nom=r_nom, reward_adv=r_adv, max_reward=max_reward,
            xi_nom_dict=xi_nom_dict, xi_adv_dict=xi_adv_dict,
            iteration=iteration, method=method, env_name=env_name,
        )

        # panel labels
        label_y = HUD_H
        _draw_panel_label(draw, "NOMINAL", COL_LABEL_NOM, 0, label_y, pw)
        _draw_panel_label(draw, "ADVERSARIAL", COL_LABEL_ADV, pw + DIVIDER_W, label_y, pw)

        # divider
        draw.rectangle([pw, HUD_H, pw + DIVIDER_W, total_h], fill=COL_DIVIDER)

        panel_y = HUD_H + LABEL_H

        # --- render nominal panel (left) ---
        t_nom = min(t, n_nom - 1) if n_nom > 0 else 0
        if is_pendulum:
            frame_nom = _render_pendulum_frame(
                traj_nom["states"][t_nom], width=pw, height=ph,
                torque=traj_nom["torques"][t_nom], u_max=u_max_val,
            )
        else:
            frame_nom = _render_cartpole_frame(
                traj_nom["states"][t_nom], width=pw, height=ph,
                action=traj_nom["actions"][t_nom],
            )
        composite.paste(Image.fromarray(frame_nom), (0, panel_y))

        # --- render adversarial panel (right) ---
        t_adv = min(t, n_adv - 1) if n_adv > 0 else 0
        if is_pendulum:
            frame_adv = _render_pendulum_frame(
                traj_adv["states"][t_adv], width=pw, height=ph,
                torque=traj_adv["torques"][t_adv], u_max=u_max_val,
            )
        else:
            frame_adv = _render_cartpole_frame(
                traj_adv["states"][t_adv], width=pw, height=ph,
                action=traj_adv["actions"][t_adv],
            )
        composite.paste(Image.fromarray(frame_adv), (pw + DIVIDER_W, panel_y))

        # --- outcome stamps ---
        # nominal stamp
        nom_ended = t >= n_nom - 1
        nom_near_end = t >= n_nom - stamp_frames
        if nom_ended and (not traj_nom["survived"] or nom_near_end):
            _draw_outcome_stamp(draw, nom_outcome, traj_nom["survived"],
                                0, panel_y, pw, ph)

        # adversarial stamp
        adv_ended = t >= n_adv - 1
        adv_near_end = t >= n_adv - stamp_frames
        if adv_ended and (not traj_adv["survived"] or adv_near_end):
            _draw_outcome_stamp(draw, adv_outcome, traj_adv["survived"],
                                pw + DIVIDER_W, panel_y, pw, ph)

        frames.append(composite)

    # save GIF
    if frames:
        duration_ms = max(1, int(1000 / fps))
        frames[0].save(
            str(out_path),
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
    return out_path


# ---------------------------------------------------------------------------
# Single-episode GIF (backward-compatible)
# ---------------------------------------------------------------------------

def render_episode_gif(
    env,
    policy,
    xi: torch.Tensor,
    out_path,
    *,
    max_steps: int = 500,
    seed: int = 0,
    deterministic: bool = True,
    fps: int = 20,
    iteration: Optional[int] = None,
    method: Optional[str] = None,
    extra_info: Optional[str] = None,
    width: int = 0,
    height: int = 0,
) -> Path:
    """
    Render a single-episode GIF (backward-compatible interface).
    """
    if Image is None:
        raise ImportError("Pillow is required for GIF rendering: pip install Pillow")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_type = type(env).__name__
    is_pendulum = "Pendulum" in env_type or "SwingUp" in env_type

    traj = _rollout_episode(env, policy, xi, max_steps=max_steps,
                            seed=seed, deterministic=deterministic)

    u_max_val = 8.0
    if is_pendulum and hasattr(env, "u_max"):
        u_max_val = float(env.u_max.item())

    frames = []
    for t, state_np in enumerate(traj["states"]):
        if is_pendulum:
            w = width if width > 0 else PENDULUM_PANEL_W
            h = height if height > 0 else PENDULUM_PANEL_H
            frame = _render_pendulum_frame(
                state_np, width=w, height=h,
                torque=traj["torques"][t], u_max=u_max_val,
            )
        else:
            w = width if width > 0 else PANEL_W
            h = height if height > 0 else PANEL_H
            frame = _render_cartpole_frame(
                state_np, width=w, height=h,
                action=traj["actions"][t],
            )

        # simple text overlay for single-episode mode
        frame = _add_text_overlay(frame, iteration=iteration, method=method,
                                  step=t, reward=traj["cumulative_rewards"][t],
                                  state_np=state_np, is_pendulum=is_pendulum,
                                  torque=traj["torques"][t], extra_info=extra_info)
        frames.append(Image.fromarray(frame))

    if frames:
        duration_ms = max(1, int(1000 / fps))
        frames[0].save(
            str(out_path),
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
    return out_path


def _add_text_overlay(
    frame: np.ndarray,
    *,
    iteration: Optional[int] = None,
    method: Optional[str] = None,
    step: int = 0,
    reward: float = 0.0,
    state_np: Optional[np.ndarray] = None,
    is_pendulum: bool = False,
    torque: Optional[float] = None,
    extra_info: Optional[str] = None,
) -> np.ndarray:
    """Add text overlay for single-episode GIF mode."""
    if Image is None:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _get_font(12)

    lines = []
    if iteration is not None:
        lines.append(f"iter {iteration}")
    if method:
        lines.append(f"method: {method}")
    lines.append(f"step {step}  R={reward:.1f}")
    if state_np is not None:
        if is_pendulum:
            theta_deg = float(state_np[0]) * 180.0 / math.pi
            lines.append(f"theta={theta_deg:.1f} deg  dtheta={float(state_np[1]):.2f}")
            if torque is not None:
                lines.append(f"torque={torque:.2f}")
        else:
            lines.append(f"x={float(state_np[0]):.3f}  theta={float(state_np[2]) * 180 / math.pi:.1f} deg")
    if extra_info:
        lines.append(extra_info)

    y = 6
    for line in lines:
        tw, th = _text_size(draw, line, font)
        draw.rectangle([4, y - 1, 8 + tw, y + th + 1], fill=(255, 255, 255))
        draw.text((6, y), line, fill=(30, 30, 30), font=font)
        y += th + 4
    return np.array(img)
