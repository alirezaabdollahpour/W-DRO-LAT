

from typing import Dict, Iterable, List, Optional, Tuple
import math

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE: Dict[str, str] = {
    "signum": "#7BDDE2",     # bright cyan
    "mars":   "#7A4E2A",     # warm brown
    "lion":   "#F5C300",     # golden yellow
    "sf-adamw": "#F1511B",   # orange-red
    # utility accents
    "gridblue": "#3A6EA5",   # desaturated blue for dashed y-grid
    "ink": "#000000",
    "bg":  "#F5F0D7",        # light cream/beige background
    "bg_frame": "#F5F0D7",
    "spine": "#000000",
    "legend_edge": "#D8D2B8",
}

def set_paper_style(
    base_fontsize: int = 14,
    figure_dpi: int = 300,
    use_tex: bool = True,
) -> None:
    """
    Configure global rcParams for a clean serif, TeX-like appearance.
    The defaults avoid requiring a LaTeX installation.
    """
    mpl.rcParams.update({
        # --- Fonts ---
        "text.usetex": use_tex,
        "font.family": "DejaVu Serif",
        # "font.serif": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif", "Times"],
        "mathtext.fontset": "stix",
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize + 6,
        "xtick.labelsize": base_fontsize - 2,
        "ytick.labelsize": base_fontsize - 2,
        "legend.fontsize": base_fontsize - 0,
        # --- Figure / Axes ---
        "figure.dpi": figure_dpi,
        "savefig.dpi": figure_dpi,
        "figure.facecolor": "white",
        "axes.facecolor": PALETTE["bg"],
        "axes.edgecolor": PALETTE["spine"],
        "axes.linewidth": 1.6,
        # --- Ticks ---
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.major.size": 6,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "xtick.minor.width": 1.2,
        "ytick.minor.width": 1.2,
        # --- Lines ---
        "lines.linewidth": 3.0,
        "lines.solid_capstyle": "round",
        # --- Legend ---
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": True,
        "legend.borderpad": 0.6,
        "legend.borderaxespad": 1.0,
    })


def styled_axes(
    figsize: Tuple[float, float] = (5, 3.5),
    left: float = 0.14, right: float = 0.98, bottom: float = 0.20, top: float = 0.98,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a single axes figure with the paper style, beige background, bold spines.
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    # Strong black spines, keep top/right hidden for a clean look
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_color(PALETTE["spine"])

    # Horizontal dashed grid on y only, blue, semi-transparent
    ax.grid(axis="y", which="major", linestyle=(0, (4, 4)), linewidth=0.5,
            color=PALETTE["gridblue"], alpha=0.45)
    ax.grid(False, axis="x")

    return fig, ax


def add_horizontal_guides(ax: plt.Axes, levels: Iterable[float]) -> None:
    """
    Optional: Add custom dashed horizontal guides at specified y values.
    """
    for y in levels:
        ax.axhline(y, color=PALETTE["gridblue"], lw=1.0, ls=(0, (5, 5)), alpha=0.55, zorder=0)


def add_bottom_secondary_ticks(
    ax: plt.Axes,
    tick_positions: Iterable[float],
    tick_labels: Iterable[str],
    offset_axes: float = -0.12,
    tick_labelsize: Optional[int] = None,
) -> plt.Axes:
    """
    Add a *second* bottom axis (twiny) with custom tick positions/labels.
    Useful to emulate the small '64k 128k ...' labels that ride along the x-axis baseline.
    This axis shares the same data coordinates (no transform).
    """
    ax2 = ax.twiny()
    # Move the new axis to below the main one
    ax2.xaxis.set_ticks_position("bottom")
    ax2.spines["bottom"].set_position(("axes", offset_axes))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Mirror limits
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(list(tick_positions))
    ax2.set_xticklabels(list(tick_labels))
    for spine in ax2.spines.values():
        spine.set_visible(False)  # just ticks/labels
    ax2.tick_params(axis="x", which="both", length=0)

    if tick_labelsize is not None:
        for label in ax2.get_xticklabels():
            label.set_fontsize(tick_labelsize)

    return ax2


def add_legend(ax: plt.Axes, loc: str = "best") -> None:
    """
    Legend styling to match the look in the figure.
    """
    leg = ax.legend(loc=loc, frameon=True, facecolor=PALETTE["bg_frame"],
                    edgecolor=PALETTE["legend_edge"], handlelength=2.6, labelspacing=0.6)
    leg.get_frame().set_linewidth(1.0)


def plot_series(ax: plt.Axes, x, y, label: str, color_key: str, **kwargs) -> None:
    """
    Helper to plot a series with the template defaults.
    """
    kwargs.setdefault("lw", mpl.rcParams["lines.linewidth"])
    kwargs.setdefault("alpha", 0.98)
    ax.plot(x, y, label=label, color=PALETTE[color_key], **kwargs)



