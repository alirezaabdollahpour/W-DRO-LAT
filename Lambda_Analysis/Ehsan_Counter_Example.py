import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ==========================================
# 1. Configuration
# ==========================================

K_STEEP = 10.0
LAM_DEFAULT = 0.08
ETA = 0.005
STEPS = 2000
PLOT_XLIM = (-12.0, 12.0)
PLOT_YLIM = (-6.0, 6.0)
BASIN_XLIM = (-18.0, 18.0)
BASIN_YLIM = (-8.0, 8.0)

V1 = np.array([-4.0, 0.5])   # Starts Left, in Top Lane
V2 = np.array([4.0, -0.5])   # Starts Right, in Bottom Lane
STARTS = {"v1": V1, "v2": V2}

# Lambda sweep for GIFs (similar spirit to Ehsan_Basin_Analysis)
LAMBDAS = np.linspace(20.0, 4.0, 100, endpoint=True)  # Nonlinear spacing for more detail at low lambda

# Basin colors: right basin, left basin, center/merged basin
CMAP = ListedColormap(["#9999FF", "#DDAAEE", "#FF9999"])


# ==========================================
# 2. Stable Loss & Gradients
# ==========================================

def _ab_terms(x, y):
    """Compute exponent terms for ridge branches."""
    a = x - K_STEEP * (y - 1.0) ** 2
    b = -x - K_STEEP * (y + 1.0) ** 2
    return a, b


def loss_xy(x, y):
    """Stable log(exp(A) + exp(B)) for scalar or array x,y."""
    a, b = _ab_terms(x, y)
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def grad_loss_xy(x, y):
    """Gradient of loss wrt x,y for scalar or array x,y."""
    a, b = _ab_terms(x, y)
    m = np.maximum(a, b)
    w_a = np.exp(a - m)
    w_b = np.exp(b - m)
    denom = w_a + w_b
    sigma_a = w_a / denom
    sigma_b = w_b / denom

    dloss_dx = sigma_a - sigma_b
    dloss_dy = sigma_a * (-2.0 * K_STEEP * (y - 1.0)) + sigma_b * (
        -2.0 * K_STEEP * (y + 1.0)
    )
    return dloss_dx, dloss_dy


def loss(z):
    return float(loss_xy(z[0], z[1]))


def grad_ell(z):
    gx, gy = grad_loss_xy(z[0], z[1])
    return np.array([gx, gy], dtype=float)


def f_lambda(x, y, lam, v_start):
    return loss_xy(x, y) - lam * ((x - v_start[0]) ** 2 + (y - v_start[1]) ** 2)


def grad_f_lambda(x, y, lam, v_start):
    gx, gy = grad_loss_xy(x, y)
    gx = gx - 2.0 * lam * (x - v_start[0])
    gy = gy - 2.0 * lam * (y - v_start[1])
    return gx, gy


def gradient_check(num_points=25, eps=1e-6, tol=5e-5):
    """
    Finite-difference gradient check for grad_ell at random points.
    Raises AssertionError if the max error exceeds tol.
    """
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(num_points):
        z = np.array([rng.uniform(-5.0, 5.0), rng.uniform(-2.5, 2.5)])
        g = grad_ell(z)
        gx_fd = (loss(z + np.array([eps, 0.0])) - loss(z - np.array([eps, 0.0]))) / (
            2.0 * eps
        )
        gy_fd = (loss(z + np.array([0.0, eps])) - loss(z - np.array([0.0, eps]))) / (
            2.0 * eps
        )
        err = max(abs(g[0] - gx_fd), abs(g[1] - gy_fd))
        max_err = max(max_err, err)
    if max_err > tol:
        raise AssertionError(
            f"Gradient check failed: max error {max_err:.3e} > tol {tol:.3e}"
        )
    print(f"Gradient check passed. max abs error = {max_err:.3e}")


# ==========================================
# 3. Ascent Dynamics
# ==========================================

def run_duchi(v_start, lam=LAM_DEFAULT, eta=ETA, steps=STEPS):
    x = v_start.astype(float).copy()
    traj = [x.copy()]
    for _ in range(steps):
        gx, gy = grad_f_lambda(x[0], x[1], lam, v_start)
        x[0] += eta * gx
        x[1] += eta * gy
        traj.append(x.copy())
    return x, np.array(traj)


def run_many_ascent(points_xy, lam, v_start, steps=300, eta=0.02):
    px = points_xy[:, 0].copy()
    py = points_xy[:, 1].copy()
    for _ in range(steps):
        gx, gy = grad_f_lambda(px, py, lam, v_start)
        px += eta * gx
        py += eta * gy
    return np.stack([px, py], axis=1)


def get_exact_attractors(lam, v_start, steps=3000, eta=0.01):
    """
    Track attractors from structural seeds.
    We include both lanes and offsets around the anchor point.
    """
    vx, vy = v_start
    seeds = np.array(
        [
            [vx, vy],
            [vx - 4.0, 1.0],
            [vx + 4.0, 1.0],
            [vx - 4.0, -1.0],
            [vx + 4.0, -1.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [vx, 1.0],
            [vx, -1.0],
        ],
        dtype=float,
    )
    return run_many_ascent(seeds, lam=lam, v_start=v_start, steps=steps, eta=eta)


def get_unique_maxima(points, threshold=1.0):
    unique = []
    for pt in points:
        if not any(np.linalg.norm(pt - ref) < threshold for ref in unique):
            unique.append(pt)
    return np.array(unique)


def labels_for_sorted_maxima(num_max):
    # Keep stable semantics with Ehsan_Basin_Analysis coloring:
    # 0=right(blue), 1=left(purple), 2=middle/merged(pink)
    if num_max <= 0:
        return []
    if num_max == 1:
        return [2]
    if num_max == 2:
        return [1, 0]
    if num_max == 3:
        return [1, 2, 0]
    # For >3, cycle labels; uncommon for this objective but keeps code safe.
    base = [1, 2, 0]
    out = []
    for i in range(num_max):
        out.append(base[i % 3])
    return out


# ==========================================
# 4. Drawing Logic
# ==========================================

def draw_surface(ax, lam, v_start, xlim=PLOT_XLIM, ylim=PLOT_YLIM, resolution=220):
    ax.clear()
    x_val = np.linspace(xlim[0], xlim[1], resolution)
    y_val = np.linspace(ylim[0], ylim[1], resolution)
    xg, yg = np.meshgrid(x_val, y_val)
    zg = f_lambda(xg, yg, lam, v_start)
    ax.plot_surface(xg, yg, zg, cmap="viridis", edgecolor="none", alpha=0.92)

    attractors = get_unique_maxima(get_exact_attractors(lam, v_start))
    for mx, my in attractors:
        mz = f_lambda(mx, my, lam, v_start)
        ax.plot([mx], [my], [mz], "w*", markersize=10, markeredgecolor="black")

    ax.set_title(
        f"Surface: $f_\\lambda(z; v_0)$, $\\lambda$={lam:.3f}, $v_0$=({v_start[0]:.1f},{v_start[1]:.1f})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.view_init(elev=28, azim=45)


def draw_basin(
    ax,
    lam,
    v_start,
    xlim=BASIN_XLIM,
    ylim=BASIN_YLIM,
    resolution=280,
    steps=240,
    eta=0.02,
):
    ax.clear()

    attractors = get_exact_attractors(lam, v_start)
    unique_maxs = get_unique_maxima(attractors)
    if unique_maxs.size == 0:
        unique_maxs = np.array([v_start], dtype=float)

    unique_maxs = np.array(sorted(unique_maxs, key=lambda t: t[0]))
    labels = labels_for_sorted_maxima(len(unique_maxs))

    x_val = np.linspace(xlim[0], xlim[1], resolution)
    y_val = np.linspace(ylim[0], ylim[1], resolution)
    xg, yg = np.meshgrid(x_val, y_val)
    px, py = xg.copy(), yg.copy()

    for _ in range(steps):
        gx, gy = grad_f_lambda(px, py, lam, v_start)
        px += eta * gx
        py += eta * gy

    distances = []
    for mx, my in unique_maxs:
        distances.append((px - mx) ** 2 + (py - my) ** 2)
    dist = np.stack(distances, axis=-1)
    nearest_idx = np.argmin(dist, axis=-1)

    basins = np.zeros_like(nearest_idx)
    for idx, label in enumerate(labels):
        basins[nearest_idx == idx] = label

    ax.pcolormesh(xg, yg, basins, cmap=CMAP, vmin=0, vmax=2, shading="auto", alpha=0.6)
    z = f_lambda(xg, yg, lam, v_start)
    ax.contour(xg, yg, z, levels=24, colors="black", alpha=0.38, linewidths=0.8)

    for mx, my in unique_maxs:
        ax.plot(mx, my, "k*", markersize=11, markeredgecolor="white")

    ax.plot(v_start[0], v_start[1], "wo", markersize=7, markeredgecolor="black")

    ax.set_title(
        f"Basins: $f_\\lambda(z; v_0)$, $\\lambda$={lam:.3f}, $v_0$=({v_start[0]:.1f},{v_start[1]:.1f})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


# ==========================================
# 5. Standalone Plotters
# ==========================================

def plot_specific_frame_surface(lam, v_start, tag, out_dir="."):
    print(f"Plotting surface frame: lambda={lam:.3f}, start={tag}")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    draw_surface(ax, lam, v_start, resolution=360)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"surface_{tag}_lambda_{lam:.2f}.png"))
    plt.close(fig)


def plot_specific_frame_basin(lam, v_start, tag, out_dir="."):
    print(f"Plotting basin frame: lambda={lam:.3f}, start={tag}")
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_basin(ax, lam, v_start, resolution=520, steps=320)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"basin_{tag}_lambda_{lam:.2f}.png"))
    plt.close(fig)


# ==========================================
# 6. GIF Generators
# ==========================================

def generate_surface_gif(v_start, tag, lambdas=LAMBDAS, out_dir="."):
    print(f"Generating surface GIF for {tag}...")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        draw_surface(ax, lambdas[frame], v_start, resolution=180)

    ani = animation.FuncAnimation(fig, update, frames=len(lambdas), interval=180)
    os.makedirs(out_dir, exist_ok=True)
    ani.save(os.path.join(out_dir, f"surface_evolution_{tag}.gif"), writer="pillow", fps=6)
    plt.close(fig)


def generate_basins_gif(v_start, tag, lambdas=LAMBDAS, out_dir="."):
    print(f"Generating basin GIF for {tag}...")
    fig, ax = plt.subplots(figsize=(12, 6))

    def update(frame):
        draw_basin(ax, lambdas[frame], v_start, resolution=250, steps=180)

    ani = animation.FuncAnimation(fig, update, frames=len(lambdas), interval=180)
    os.makedirs(out_dir, exist_ok=True)
    ani.save(os.path.join(out_dir, f"basins_evolution_{tag}.gif"), writer="pillow", fps=6)
    plt.close(fig)


# ==========================================
# 7. Original Counterexample Trajectories
# ==========================================

def plot_counterexample_trajectories(lam=LAM_DEFAULT, out_dir="."):
    t_v1, traj1 = run_duchi(V1, lam=lam)
    t_v2, traj2 = run_duchi(V2, lam=lam)

    dv = V2 - V1
    d_t = t_v2 - t_v1
    inner_prod = float(np.dot(d_t, dv))

    print(f"Start v1: {V1} -> End T(v1): {t_v1.round(3)}")
    print(f"Start v2: {V2} -> End T(v2): {t_v2.round(3)}")
    print(f"Input dV: {dv}")
    print(f"Output dT: {d_t.round(3)}")
    print(f"Inner Product: {inner_prod:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_grid = np.linspace(PLOT_XLIM[0], PLOT_XLIM[1], 220)
    y_grid = np.linspace(PLOT_YLIM[0], PLOT_YLIM[1], 180)
    xg, yg = np.meshgrid(x_grid, y_grid)
    z = loss_xy(xg, yg)
    cs = ax.contourf(xg, yg, z, levels=36, cmap="viridis", alpha=0.6)
    fig.colorbar(cs, ax=ax, shrink=0.9, label="loss(x, y)")

    ax.plot(traj1[:, 0], traj1[:, 1], "r-", lw=2.5, label="v1 trajectory")
    ax.plot(traj2[:, 0], traj2[:, 1], "b-", lw=2.5, label="v2 trajectory")
    ax.plot(V1[0], V1[1], "ro", markersize=7)
    ax.plot(V2[0], V2[1], "bo", markersize=7)
    ax.plot(t_v1[0], t_v1[1], "r*", markersize=12, markeredgecolor="white")
    ax.plot(t_v2[0], t_v2[1], "b*", markersize=12, markeredgecolor="white")

    ax.set_title(f"Analytic Softmax Counterexample (lambda={lam:.3f})\nInner Product: {inner_prod:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(*PLOT_XLIM)
    ax.set_ylim(*PLOT_YLIM)
    ax.legend(loc="best")

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"counterexample_trajectories_lambda_{lam:.2f}.png"))
    plt.close(fig)

    return inner_prod, t_v1, t_v2


# ==========================================
# 8. Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Counterexample basin/surface analysis with GIFs and trajectory plots."
    )
    parser.add_argument("--out-dir", default="counterexample_analysis_outputs")
    parser.add_argument("--lam", type=float, default=LAM_DEFAULT)
    parser.add_argument(
        "--mode",
        choices=["all", "plots", "gifs", "surface-gif", "basin-gif", "counterexample"],
        default="all",
    )
    parser.add_argument("--single-frame-lam", type=float, default=0.08)
    parser.add_argument(
        "--skip-grad-check",
        action="store_true",
        help="Skip finite-difference gradient validation.",
    )
    args = parser.parse_args()

    if not args.skip_grad_check:
        gradient_check()

    if args.mode in ("all", "counterexample"):
        plot_counterexample_trajectories(lam=args.lam, out_dir=args.out_dir)

    if args.mode in ("all", "plots"):
        for tag, v_start in STARTS.items():
            plot_specific_frame_surface(args.single_frame_lam, v_start, tag, out_dir=args.out_dir)
            plot_specific_frame_basin(args.single_frame_lam, v_start, tag, out_dir=args.out_dir)

    if args.mode in ("all", "gifs", "surface-gif"):
        for tag, v_start in STARTS.items():
            generate_surface_gif(v_start, tag, out_dir=args.out_dir)

    if args.mode in ("all", "gifs", "basin-gif"):
        for tag, v_start in STARTS.items():
            generate_basins_gif(v_start, tag, out_dir=args.out_dir)

    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
