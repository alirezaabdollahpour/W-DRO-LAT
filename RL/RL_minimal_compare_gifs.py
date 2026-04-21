"""
Create publication-quality filmstrip figures comparing ICNN vs Particle Ascent
training progression from saved GIF directories.

Extracts key frames from each iteration's GIF and arranges them in a grid:
  rows = training iterations (e.g., iter 0, 10, 20, ..., 99)
  columns = [ICNN Nominal, ICNN Adversarial, Particle Nominal, Particle Adversarial]

Example:
  python RL_minimal_compare_gifs.py \
    --icnn-dir  gifs/cartpole_icnn_seed0 \
    --particle-dir gifs/cartpole_particle_seed0 \
    --out comparison_filmstrip.pdf

  # Auto-discover from a base directory:
  python RL_minimal_compare_gifs.py \
    --gif-base gifs --env cartpole --seed 0 \
    --out comparison_filmstrip.pdf

  # Select specific iterations:
  python RL_minimal_compare_gifs.py \
    --gif-base gifs --env cartpole --seed 0 \
    --select-iters 0 20 50 99 \
    --out comparison_filmstrip.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


def _discover_gif_dir(base: Path, env: str, method: str, seed: int) -> Path:
    """Construct the expected gif directory path."""
    return base / f"{env}_{method}_seed{seed}"


def _list_iter_gifs(gif_dir: Path) -> dict[int, Path]:
    """Map iteration number -> gif path from a directory of iter_XXXX.gif files."""
    result: dict[int, Path] = {}
    if not gif_dir.is_dir():
        return result
    for p in sorted(gif_dir.glob("iter_*.gif")):
        m = re.match(r"iter_(\d+)\.gif$", p.name)
        if m:
            result[int(m.group(1))] = p
    return result


def _extract_frame(gif_path: Path, frame_index: int = 0) -> np.ndarray:
    """Extract a specific frame from a GIF as an RGB numpy array."""
    if Image is None:
        raise ImportError("Pillow is required: pip install Pillow")
    img = Image.open(gif_path)
    try:
        img.seek(frame_index)
    except EOFError:
        img.seek(0)
    return np.array(img.convert("RGB"))


def _extract_side_panels(
    gif_path: Path, frame_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract nominal (left) and adversarial (right) panels from a side-by-side
    comparison GIF. The GIF format from render_gif.py has:
      [HUD | LABEL | NOM_PANEL | DIVIDER | ADV_PANEL]
    We split the frame vertically at the midpoint.
    """
    frame = _extract_frame(gif_path, frame_index)
    h, w, _ = frame.shape
    mid = w // 2
    return frame[:, :mid, :], frame[:, mid:, :]


def _count_gif_frames(gif_path: Path) -> int:
    """Count total frames in a GIF."""
    if Image is None:
        raise ImportError("Pillow is required: pip install Pillow")
    img = Image.open(gif_path)
    n = 0
    try:
        while True:
            n += 1
            img.seek(n)
    except EOFError:
        pass
    return n


def create_filmstrip(
    icnn_dir: Path,
    particle_dir: Path,
    *,
    out_path: Path,
    select_iters: list[int] | None = None,
    frame_index: int | str = "mid",
    dpi: int = 150,
    env: str = "cartpole",
    seed: int = 0,
    show: bool = False,
) -> Path:
    """
    Create a filmstrip figure comparing ICNN vs Particle training progression.

    Layout (4 columns):
      ICNN Nominal | ICNN Adversarial | Particle Nominal | Particle Adversarial
    One row per selected iteration.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    icnn_gifs = _list_iter_gifs(icnn_dir)
    particle_gifs = _list_iter_gifs(particle_dir)

    if not icnn_gifs and not particle_gifs:
        raise FileNotFoundError(
            f"No GIFs found in {icnn_dir} or {particle_dir}. "
            "Check --icnn-dir / --particle-dir paths."
        )

    # Determine which iterations to show
    all_iters = sorted(set(icnn_gifs.keys()) | set(particle_gifs.keys()))
    if select_iters is not None:
        iters_to_show = [i for i in select_iters if i in all_iters]
        if not iters_to_show:
            raise ValueError(
                f"None of --select-iters {select_iters} found. Available: {all_iters}"
            )
    else:
        # Auto-select: show ~6 evenly spaced iterations including first and last
        if len(all_iters) <= 6:
            iters_to_show = all_iters
        else:
            indices = np.linspace(0, len(all_iters) - 1, 6, dtype=int)
            iters_to_show = [all_iters[i] for i in indices]

    n_rows = len(iters_to_show)

    # Extract one frame to determine panel dimensions
    sample_gif = next(iter(icnn_gifs.values()), None) or next(iter(particle_gifs.values()))
    sample_frame = _extract_frame(sample_gif, 0)
    frame_h, frame_w = sample_frame.shape[:2]
    panel_w = frame_w // 2  # each side of the comparison GIF

    # Figure sizing
    col_inch = 3.0
    row_inch = col_inch * (frame_h / max(panel_w, 1))
    fig_w = col_inch * 4 + 1.5  # 4 columns + space for row labels
    fig_h = row_inch * n_rows + 1.8  # rows + suptitle/header space

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_rows + 1, 5,  # +1 for header row, 5 = label_col + 4 image cols
        figure=fig,
        width_ratios=[0.3, 1, 1, 1, 1],
        height_ratios=[0.15] + [1] * n_rows,
        hspace=0.05, wspace=0.03,
    )

    # Column headers
    col_headers = [
        "ICNN\nNominal", "ICNN\nAdversarial",
        "Particle\nNominal", "Particle\nAdversarial",
    ]
    header_colors = ["#1f77b4", "#c44e52", "#ff7f0e", "#c44e52"]
    for j, (header, hcolor) in enumerate(zip(col_headers, header_colors)):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.set_facecolor(hcolor)
        ax.text(
            0.5, 0.5, header, transform=ax.transAxes,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide the top-left corner cell
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_corner.axis("off")

    for row_idx, it in enumerate(iters_to_show):
        grid_row = row_idx + 1  # offset by header row

        # Row label (iteration number)
        ax_label = fig.add_subplot(gs[grid_row, 0])
        ax_label.text(
            0.9, 0.5, f"iter\n{it}", transform=ax_label.transAxes,
            ha="right", va="center", fontsize=10, fontweight="bold",
            color="#333333",
        )
        ax_label.axis("off")

        # Determine frame index to extract
        fi: int
        if isinstance(frame_index, str) and frame_index == "mid":
            # Use the middle frame of each GIF
            ref_gif = icnn_gifs.get(it) or particle_gifs.get(it)
            n_frames = _count_gif_frames(ref_gif) if ref_gif else 1
            fi = n_frames // 2
        else:
            fi = int(frame_index)

        # Extract panels: [icnn_nom, icnn_adv, particle_nom, particle_adv]
        panels: list[np.ndarray | None] = [None, None, None, None]

        if it in icnn_gifs:
            nom, adv = _extract_side_panels(icnn_gifs[it], fi)
            panels[0] = nom
            panels[1] = adv
        if it in particle_gifs:
            nom, adv = _extract_side_panels(particle_gifs[it], fi)
            panels[2] = nom
            panels[3] = adv

        for col_idx, panel in enumerate(panels):
            ax = fig.add_subplot(gs[grid_row, col_idx + 1])
            if panel is not None:
                ax.imshow(panel, aspect="auto")
            else:
                ax.set_facecolor("#f0f0f0")
                ax.text(
                    0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#999999",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
                spine.set_linewidth(0.5)

    fig.suptitle(
        f"Training Progression: ICNN vs Particle Ascent ({env}, seed={seed})",
        fontsize=13, fontweight="bold", y=0.99,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved filmstrip: {out_path}")

    if show:
        plt.show()
    plt.close(fig)
    return out_path


def create_side_by_side_gif(
    icnn_dir: Path,
    particle_dir: Path,
    *,
    iteration: int,
    out_path: Path,
    env: str = "cartpole",
) -> Path:
    """
    Stitch the ICNN and Particle GIFs for the same iteration into a single
    4-panel GIF: [ICNN_nom | ICNN_adv || Particle_nom | Particle_adv].
    """
    if Image is None:
        raise ImportError("Pillow is required: pip install Pillow")

    icnn_gif = icnn_dir / f"iter_{iteration:04d}.gif"
    particle_gif = particle_dir / f"iter_{iteration:04d}.gif"

    if not icnn_gif.exists():
        raise FileNotFoundError(f"ICNN GIF not found: {icnn_gif}")
    if not particle_gif.exists():
        raise FileNotFoundError(f"Particle GIF not found: {particle_gif}")

    icnn_img = Image.open(icnn_gif)
    particle_img = Image.open(particle_gif)

    # Count frames
    n_icnn = _count_gif_frames(icnn_gif)
    n_particle = _count_gif_frames(particle_gif)
    n_frames = max(n_icnn, n_particle)

    # Get dimensions
    iw, ih = icnn_img.size
    pw, ph = particle_img.size
    divider_w = 6
    total_w = iw + divider_w + pw
    total_h = max(ih, ph)

    frames: list[Image.Image] = []
    for t in range(n_frames):
        # Get ICNN frame
        try:
            icnn_img.seek(min(t, n_icnn - 1))
        except EOFError:
            icnn_img.seek(n_icnn - 1)
        icnn_frame = icnn_img.convert("RGB")

        # Get Particle frame
        try:
            particle_img.seek(min(t, n_particle - 1))
        except EOFError:
            particle_img.seek(n_particle - 1)
        particle_frame = particle_img.convert("RGB")

        # Compose
        composite = Image.new("RGB", (total_w, total_h), (40, 40, 50))
        composite.paste(icnn_frame, (0, 0))
        composite.paste(particle_frame, (iw + divider_w, 0))

        # Draw method labels on the divider
        from PIL import ImageDraw
        draw = ImageDraw.Draw(composite)
        draw.rectangle([iw, 0, iw + divider_w, total_h], fill=(40, 40, 50))

        frames.append(composite)

    if frames:
        # Get duration from original
        duration = icnn_img.info.get("duration", 50)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            str(out_path),
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        print(f"Saved combined GIF: {out_path} ({n_frames} frames)")

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare ICNN vs Particle Ascent training GIFs in publication-quality figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Directory specification
    parser.add_argument("--icnn-dir", type=Path, default=None,
                        help="Path to ICNN gif directory (e.g., gifs/cartpole_icnn_seed0).")
    parser.add_argument("--particle-dir", type=Path, default=None,
                        help="Path to Particle gif directory (e.g., gifs/cartpole_particle_seed0).")
    parser.add_argument("--gif-base", type=Path, default=None,
                        help="Base gif directory; auto-discovers method dirs using --env and --seed.")
    parser.add_argument("--env", type=str, default="cartpole",
                        help="Environment name (used with --gif-base for auto-discovery).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed (used with --gif-base for auto-discovery).")

    # Output control
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path for the filmstrip figure (png/pdf/svg).")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true", help="Show interactive figure.")

    # Iteration selection
    parser.add_argument("--select-iters", type=int, nargs="+", default=None,
                        help="Specific iterations to show (default: ~6 evenly spaced).")

    # Frame selection
    parser.add_argument("--frame", type=str, default="mid",
                        help="Which frame to extract from each GIF: 'mid' (default) or integer index.")

    # Combined GIF mode
    parser.add_argument("--combined-gif", type=int, default=None, metavar="ITER",
                        help="Instead of filmstrip, create a combined GIF for the given iteration.")
    parser.add_argument("--combined-gif-out", type=Path, default=None,
                        help="Output path for combined GIF.")

    args = parser.parse_args(argv)

    # Resolve directories
    script_dir = Path(__file__).resolve().parent
    if args.icnn_dir is None and args.gif_base is not None:
        args.icnn_dir = _discover_gif_dir(args.gif_base, args.env, "icnn", args.seed)
    elif args.icnn_dir is None:
        args.icnn_dir = _discover_gif_dir(script_dir / "gifs", args.env, "icnn", args.seed)

    if args.particle_dir is None and args.gif_base is not None:
        args.particle_dir = _discover_gif_dir(args.gif_base, args.env, "particle", args.seed)
    elif args.particle_dir is None:
        args.particle_dir = _discover_gif_dir(script_dir / "gifs", args.env, "particle", args.seed)

    # Combined GIF mode
    if args.combined_gif is not None:
        out = args.combined_gif_out or (
            script_dir / f"combined_{args.env}_iter{args.combined_gif:04d}_seed{args.seed}.gif"
        )
        create_side_by_side_gif(
            args.icnn_dir, args.particle_dir,
            iteration=args.combined_gif, out_path=out, env=args.env,
        )
        return 0

    # Filmstrip mode (default)
    out_path = args.out
    if out_path is None:
        out_path = script_dir / f"filmstrip_{args.env}_seed{args.seed}.pdf"

    frame_index: int | str
    try:
        frame_index = int(args.frame)
    except ValueError:
        frame_index = args.frame

    create_filmstrip(
        args.icnn_dir, args.particle_dir,
        out_path=out_path,
        select_iters=args.select_iters,
        frame_index=frame_index,
        dpi=args.dpi,
        env=args.env,
        seed=args.seed,
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
