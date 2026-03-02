import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# 1. Enable high-resolution rendering for Colab/Jupyter display
try:
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
except Exception:
    pass

# 2. Update rcParams for professional aesthetics
plt.rcParams.update({
    # Figure Size, Background, and Resolution (DPI)
    # 'figure.figsize': (10, 6),        # Standard wide aspect ratio
    'figure.dpi': 120,                # Baseline DPI for in-notebook display
    'savefig.dpi': 300,               # Publication-quality DPI for saving files
    'figure.facecolor': 'white',      # Clean white background
    'figure.autolayout': True,        # Prevents labels from getting cut off

    # Fonts & Text
    'font.family': 'sans-serif',
    'font.size': 12,                  # Base font size

    # Titles
    'axes.titlesize': 16,             # Larger, bold title
    'axes.titleweight': 'bold',
    'axes.titlepad': 15,              # Space between title and plot

    # Axes Names (Labels)
    'axes.labelsize': 18,             # Font size for X and Y axis names
    'axes.labelweight': 'normal',     # Can change to 'bold' if preferred
    'axes.labelpad': 8,              # Space between axis and label

    # Tick Labels (The numbers on the axes)
    'xtick.labelsize': 12,            # Font size for X-axis numbers
    'ytick.labelsize': 12,            # Font size for Y-axis numbers

    # Legend
    'legend.fontsize': 16,
    'legend.frameon': False,          # Clean legend without a box

    # Axes & Borders (Spines)
    'axes.spines.top': False,         # Despine top
    'axes.spines.right': False,       # Despine right
    'axes.linewidth': 1.2,            # Slightly bolder borders
    'axes.edgecolor': '#333333',      # Dark grey rather than harsh black

    # Gridlines
    'axes.grid': True,                # Turn on the grid
    'grid.alpha': 0.3,                # Very transparent
    'grid.linestyle': '--',           # Dashed lines
    'grid.color': '#cccccc',          # Light grey so it doesn't distract

    # Line & Marker Styles
    'lines.linewidth': 2.5,           # Thicker, bolder lines
    'lines.markersize': 8,            # Larger markers

    # Professional Color Palette (Tableau 10)
    'axes.prop_cycle': plt.cycler('color', [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])
})

np.random.seed(42)
K_STEEP = 10.0

# ======================================================================
# Core functions
# ======================================================================
def loss_fn(z):
    x, y = z[0], z[1]
    A = x - K_STEEP*(y-1)**2; B = -x - K_STEEP*(y+1)**2; M = max(A, B)
    return M + np.log(np.exp(A-M) + np.exp(B-M))

def loss_grad(z):
    x, y = z[0], z[1]
    A = x - K_STEEP*(y-1)**2; B = -x - K_STEEP*(y+1)**2; M = max(A, B)
    wA, wB = np.exp(A-M), np.exp(B-M); s = wA+wB; sA, sB = wA/s, wB/s
    return sA*np.array([1.0, -2*K_STEEP*(y-1)]) + sB*np.array([-1.0, -2*K_STEEP*(y+1)])

def inner_obj(z, z_hat, lam):
    return loss_fn(z) - lam*np.sum((z-z_hat)**2)

def inner_grad(z, z_hat, lam):
    return loss_grad(z) - 2.0*lam*(z-z_hat)

def global_max(z_hat, lam, xr=10.0, yr=3.5, res=250):
    xlin = np.linspace(-xr, xr, res)
    ylin = np.linspace(-yr, yr, int(res*yr/xr))
    best_val, best_pt = -np.inf, z_hat.copy()
    for xi in xlin:
        for yi in ylin:
            pt = np.array([xi, yi]); v = inner_obj(pt, z_hat, lam)
            if v > best_val: best_val, best_pt = v, pt.copy()
    r = minimize(lambda z: -inner_obj(z, z_hat, lam), best_pt,
                 jac=lambda z: -inner_grad(z, z_hat, lam), method="L-BFGS-B")
    return r.x

def particle_ascent(z_hat, lam, K=2000, eta=0.005):
    z = z_hat.copy()
    for _ in range(K): z = z + eta*inner_grad(z, z_hat, lam)
    return z

def project_onto_pairwise_cm_cone(z_all, x_hats):
    """Euclidean projection onto the pairwise (2-cycle) CM constraints.

    We enforce the monotonicity inequalities
        ⟨z_i - z_j, x̂_i - x̂_j⟩ ≥ 0  for all i<j.
    This guarantees 2-cyclic monotonicity (monotonicity), which is necessary but
    not sufficient for full cyclic monotonicity in dimensions d>1.

    For the toy example (small N), we solve the projection via SLSQP.
    """
    N, d = z_all.shape; z0 = z_all.flatten().copy()
    constraints = []
    for i, j in combinations(range(N), 2):
        dhat = x_hats[i]-x_hats[j]
        def make_con(ii, jj, dh):
            def con_fn(zf): return np.dot(zf[ii*d:(ii+1)*d]-zf[jj*d:(jj+1)*d], dh)
            def con_jac(zf):
                g = np.zeros_like(zf); g[ii*d:(ii+1)*d]=dh; g[jj*d:(jj+1)*d]=-dh; return g
            return {"type":"ineq","fun":con_fn,"jac":con_jac}
        constraints.append(make_con(i,j,dhat))
    res = minimize(fun=lambda zf: 0.5*np.sum((zf-z0)**2), x0=z0,
                   jac=lambda zf: zf-z0, constraints=constraints,
                   method="SLSQP", options={"maxiter":500,"ftol":1e-12})
    return res.x.reshape(N, d)

def projected_pa_hard(x_hats, lam, K=5000, eta=0.0005, proj_every=1):
    N = x_hats.shape[0]; z_all = x_hats.copy()
    for k in range(1, K+1):
        for i in range(N): z_all[i] = z_all[i] + eta*inner_grad(z_all[i], x_hats[i], lam)
        if k % proj_every == 0: z_all = project_onto_pairwise_cm_cone(z_all, x_hats)
    z_all = project_onto_pairwise_cm_cone(z_all, x_hats)
    return z_all

def cm_penalty_grad(z_all, x_hats, mu):
    N, d = z_all.shape; grad = np.zeros_like(z_all)
    for i, j in combinations(range(N), 2):
        dhat = x_hats[i]-x_hats[j]; cm_ij = np.dot(z_all[i]-z_all[j], dhat)
        if cm_ij < 0: grad[i] -= mu*2*cm_ij*dhat; grad[j] += mu*2*cm_ij*dhat
    return grad

def projected_pa_soft(x_hats, lam, K=2000, eta=0.005, mu=0.5):
    N = x_hats.shape[0]; z_all = x_hats.copy()
    for k in range(1, K+1):
        obj_grad = np.array([inner_grad(z_all[i], x_hats[i], lam) for i in range(N)])
        pen_grad = cm_penalty_grad(z_all, x_hats, mu)
        z_all = z_all + eta*(obj_grad + pen_grad)
    return z_all

def cm_2(zi, zj, xhi, xhj): return np.dot(zi-zj, xhi-xhj)

def all_pairs(z_sel, xh, zs, lam):
    N_l = z_sel.shape[0]; cms, gaps, idxs = [], [], []
    for i, j in combinations(range(N_l), 2):
        cm = cm_2(z_sel[i], z_sel[j], xh[i], xh[j])
        di = inner_obj(zs[i], xh[i], lam) - inner_obj(z_sel[i], xh[i], lam)
        dj = inner_obj(zs[j], xh[j], lam) - inner_obj(z_sel[j], xh[j], lam)
        cms.append(cm); gaps.append(max(0,di)+max(0,dj)); idxs.append((i,j))
    return np.array(cms), np.array(gaps), idxs

# ======================================================================
# CM violation scores + suboptimality certificates (aligned with theory)
# ======================================================================
def pairwise_violation(cm_vals):
    """Elementwise kernel h_{ij} = [-CM_{(i,j)}]^+ for each pair."""
    return np.maximum(-cm_vals, 0.0)

def u_stat_global(cm_vals):
    """Global empirical average pairwise violation U_N = average_{i<j} h_{ij}."""
    h = pairwise_violation(cm_vals)
    return float(h.mean()) if h.size else 0.0

def cert_gap_avg_pair(cm_vals, lam):
    r"""Aggregate pairwise certificate:
    Φ* - Φ(\tilde z) ≥ λ · U_N,
    where U_N is the global average pairwise violation.
    """
    return lam * u_stat_global(cm_vals)

def cert_gap_worst_pair(cm_vals, lam, N):
    r"""Baseline (weaker) worst-pair certificate:
    Φ* - Φ(\tilde z) ≥ (2λ/N) · max_{i<j} h_{ij}.
    Included for comparison with the aggregate certificate.
    """
    h = pairwise_violation(cm_vals)
    return (2.0 * lam / N) * (float(h.max()) if h.size else 0.0)

def cycle_cover_deficit(z_sel, x_hats):
    r"""Maximum-weight directed cycle cover deficit.

    Viol_cover(z) = max_{π∈S_N} Σ_i ⟨ z_i, x̂_{π(i)} - x̂_i ⟩.

    This is a maximum-weight assignment problem and can be solved via the
    Hungarian algorithm (scipy.optimize.linear_sum_assignment).

    Returns:
        viol_cover (float): optimal cover weight.
        pi (ndarray): permutation with pi[i] = π(i).
    """
    # W[i,j] = ⟨z_i, x̂_j - x̂_i⟩ = ⟨z_i, x̂_j⟩ - ⟨z_i, x̂_i⟩
    W = (z_sel @ x_hats.T) - np.sum(z_sel * x_hats, axis=1)[:, None]
    row_ind, col_ind = linear_sum_assignment(-W)  # maximize W
    return float(W[row_ind, col_ind].sum()), col_ind

def cert_gap_cycle_cover(z_sel, x_hats, lam):
    r"""Cycle-cover certificate (aggregate disjoint-cycle bound):

    Φ* - Φ(\tilde z) ≥ (2λ/N) · Viol_cover(\tilde z).
    """
    N = x_hats.shape[0]
    viol_cover, pi = cycle_cover_deficit(z_sel, x_hats)
    return (2.0 * lam / N) * viol_cover, viol_cover, pi

def empirical_diameter(points):
    """Empirical diameter max_{a,b} ||p_a - p_b||_2 (O(M^2), fine for small M)."""
    diffs = points[:, None, :] - points[None, :, :]
    return float(np.max(np.linalg.norm(diffs, axis=-1)))

def minibatch_u_stat(z_sel, x_hats, batch_idx):
    """Mini-batch average pairwise violation U_B over indices in batch_idx."""
    batch_idx = list(batch_idx)
    if len(batch_idx) < 2:
        return 0.0
    h_vals = []
    for i, j in combinations(batch_idx, 2):
        cm = np.dot(z_sel[i] - z_sel[j], x_hats[i] - x_hats[j])
        h_vals.append(max(0.0, -cm))
    return float(np.mean(h_vals))

def cert_gap_minibatch(z_sel, x_hats, lam, B, delta=0.05, rng=None, D=None):
    r"""High-probability mini-batch certificate (Appendix C.5):

    With probability ≥ 1-δ over a uniform B-subset S_B (without replacement),

        Φ* - Φ(\tilde z) ≥ λ · [ U_B - D^2 · sqrt( 2 ln(1/δ) / B ) ]_+,

    where U_B is the mini-batch average pairwise violation and D bounds the
    diameter of the compact domain Z containing both {\tilde z_i} and {x̂_i}.
    If D is not provided, we use an empirical diameter over the observed points.

    Returns:
        cert_gap (float), U_B (float), eps (float), batch_idx (ndarray), D_used (float).
    """
    rng = np.random.default_rng() if rng is None else rng
    N = x_hats.shape[0]
    if B > N:
        raise ValueError(f"Mini-batch size B={B} cannot exceed N={N}.")
    batch_idx = rng.choice(N, size=B, replace=False)
    U_B = minibatch_u_stat(z_sel, x_hats, batch_idx)

    if D is None:
        D = empirical_diameter(np.vstack([x_hats, z_sel]))

    eps = (D ** 2) * np.sqrt(2.0 * np.log(1.0 / delta) / B)
    cert_gap = lam * max(0.0, U_B - eps)
    return cert_gap, U_B, eps, batch_idx, D



# ======================================================================
# Run all methods
# ======================================================================
LAM = 0.3
x_hats = np.array([
    [-4.0,  0.5], [-4.0, -0.5],
    [-3.0,  0.3], [-3.0, -0.3],
    [-2.0,  0.4], [-2.0, -0.4],
    [ 4.0, -0.5], [ 4.0,  0.5],
    [ 3.0, -0.3], [ 3.0,  0.3],
    [ 2.0, -0.4], [ 2.0,  0.4],
    [ 0.0,  0.6], [ 0.0, -0.6],
    [-1.0,  0.2], [ 1.0, -0.2],
])
N = x_hats.shape[0]
print(f"N={N}, lambda={LAM}")

print("Global maxima ...");  z_stars = np.array([global_max(x_hats[i], LAM) for i in range(N)])
print("Vanilla PA ...");     z_pa    = np.array([particle_ascent(x_hats[i], LAM) for i in range(N)])
print("Hard proj PA ...");   z_hppa  = projected_pa_hard(x_hats, LAM)
print("Soft proj PA ...");   z_sppa  = projected_pa_soft(x_hats, LAM)

def avg_obj(zs):
    return np.mean([inner_obj(zs[i], x_hats[i], LAM) for i in range(N)])

obj_gl   = avg_obj(z_stars)
obj_pa   = avg_obj(z_pa)
obj_hppa = avg_obj(z_hppa)
obj_sppa = avg_obj(z_sppa)

# CM analysis for each method
cm_pa,   gaps_pa,   pairs = all_pairs(z_pa,   x_hats, z_stars, LAM)
cm_hppa, gaps_hppa, _     = all_pairs(z_hppa, x_hats, z_stars, LAM)
cm_sppa, gaps_sppa, _     = all_pairs(z_sppa, x_hats, z_stars, LAM)

n_viol_pa   = (cm_pa   < -1e-10).sum()
n_viol_hppa = (cm_hppa < -1e-10).sum()
n_viol_sppa = (cm_sppa < -1e-10).sum()

thm_pa   = (gaps_pa   >= 2*LAM*np.maximum(-cm_pa,   0) - 1e-8).all()
thm_hppa = (gaps_hppa >= 2*LAM*np.maximum(-cm_hppa, 0) - 1e-8).all()
thm_sppa = (gaps_sppa >= 2*LAM*np.maximum(-cm_sppa, 0) - 1e-8).all()

worst_idx_pa = np.argmin(cm_pa)
wi_pa, wj_pa = pairs[worst_idx_pa]
# ----------------------------------------------------------------------
# Certificates (worst-pair vs aggregate-pair, plus optional cycle-cover)
# ----------------------------------------------------------------------
U_pa   = u_stat_global(cm_pa)
U_hppa = u_stat_global(cm_hppa)
U_sppa = u_stat_global(cm_sppa)

# New aggregate pairwise certificate:  Φ* - Φ(tilde z) ≥ λ · U_N
cert_pa_avg   = cert_gap_avg_pair(cm_pa,   LAM)
cert_hppa_avg = cert_gap_avg_pair(cm_hppa, LAM)
cert_sppa_avg = cert_gap_avg_pair(cm_sppa, LAM)

# Baseline worst-pair certificate (kept for comparison): (2λ/N) · max h_{ij}
cert_pa_worst   = cert_gap_worst_pair(cm_pa,   LAM, N)
cert_hppa_worst = cert_gap_worst_pair(cm_hppa, LAM, N)
cert_sppa_worst = cert_gap_worst_pair(cm_sppa, LAM, N)

# Aggregate disjoint-cycle (cycle-cover) certificate: (2λ/N) · Viol_cover
cert_pa_cover,   viol_cover_pa,   pi_pa   = cert_gap_cycle_cover(z_pa,   x_hats, LAM)
cert_hppa_cover, viol_cover_hppa, pi_hppa = cert_gap_cycle_cover(z_hppa, x_hats, LAM)
cert_sppa_cover, viol_cover_sppa, pi_sppa = cert_gap_cycle_cover(z_sppa, x_hats, LAM)

# Mini-batch certificate (same batch for all methods; set B_MB=None to skip)
B_MB     = 8
DELTA_MB = 0.05
D_emp = empirical_diameter(np.vstack([x_hats, z_stars, z_pa, z_hppa, z_sppa]))
rng_mb = np.random.default_rng(123)
batch_idx = rng_mb.choice(N, size=B_MB, replace=False)
eps_mb = (D_emp ** 2) * np.sqrt(2.0 * np.log(1.0 / DELTA_MB) / B_MB)

UB_pa   = minibatch_u_stat(z_pa,   x_hats, batch_idx)
UB_hppa = minibatch_u_stat(z_hppa, x_hats, batch_idx)
UB_sppa = minibatch_u_stat(z_sppa, x_hats, batch_idx)

cert_pa_mb   = LAM * max(0.0, UB_pa   - eps_mb)
cert_hppa_mb = LAM * max(0.0, UB_hppa - eps_mb)
cert_sppa_mb = LAM * max(0.0, UB_sppa - eps_mb)


# Find canonical (0,6) pair index
canon_idx = None
for k, (pi, pj) in enumerate(pairs):
    if pi == 0 and pj == 6: canon_idx = k; break

print(f"Objectives: Global={obj_gl:.3f}, PA={obj_pa:.3f}, "
      f"Hard={obj_hppa:.3f}, Soft={obj_sppa:.3f}")
print(f"CM violations: PA={n_viol_pa}, Hard={n_viol_hppa}, Soft={n_viol_sppa}")

# Precompute contour grid
xg = np.linspace(-10, 10, 400); yg = np.linspace(-3.5, 3.5, 200)
XG, YG = np.meshgrid(xg, yg)
ZG = np.log(np.exp(XG - K_STEEP*(YG-1)**2) + np.exp(-XG - K_STEEP*(YG+1)**2))


# ======================================================================
# Plotting helpers
# ======================================================================
C_TOP    = "#d62728"
C_BOT    = "#1f77b4"
C_GLOBAL = "#2ca02c"
C_BOUND  = "#2c3e50"

# Method-specific accent colors
METHOD_COLORS = {"Vanilla PA": "#d62728", "Hard Proj-PA": "#984ea3", "Soft Proj-PA": "#ff7f0e"}


def plot_transport(ax, z_dest, title, n_viol, obj_val, method_color, u_pair=None, cert_pair=None):
    """
    Clean transport map: one arrow per sample, colored by destination ridge.
    """
    ridge = ["top" if z_dest[i,1] > 0 else "bot" for i in range(N)]

    # Background
    ax.contourf(XG, YG, ZG, levels=35, cmap="viridis", alpha=0.28)
    ax.contour(XG, YG, ZG, levels=10, colors="#555", linewidths=0.15, alpha=0.2)
    ax.fill_between([-10,10], 0.85, 1.15, color=C_TOP, alpha=0.06)
    ax.fill_between([-10,10], -1.15, -0.85, color=C_BOT, alpha=0.06)
    ax.axhline(0, color="#888", ls=":", lw=0.7, alpha=0.3)
    ax.text(9.5, 1.35, "Top ridge", fontsize=7, color=C_TOP,
            fontweight="bold", ha="right", va="bottom")
    ax.text(9.5, -1.35, "Bottom ridge", fontsize=7, color=C_BOT,
            fontweight="bold", ha="right", va="top")

    # Arrows + source markers
    for i in range(N):
        c = C_TOP if ridge[i] == "top" else C_BOT
        ax.annotate("", xy=z_dest[i], xytext=x_hats[i],
            arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2,
                            alpha=0.7, mutation_scale=13))
        ax.scatter(x_hats[i,0], x_hats[i,1], c=c, s=70, zorder=7,
                   edgecolors="white", linewidths=1.1)
        ax.text(x_hats[i,0], x_hats[i,1]-0.28, str(i),
                fontsize=5.5, fontweight="bold", color="#333",
                ha="center", va="top", zorder=8)

    # Gold rings on canonical pair (0, 6)
    for idx in [0, 6]:
        ax.scatter(x_hats[idx,0], x_hats[idx,1], s=250, facecolors="none",
                   edgecolors="gold", linewidths=2.3, zorder=9)
    ax.text(x_hats[0,0]-0.3, x_hats[0,1]+0.5, r"$v_1$",
            fontsize=9, fontweight="bold",
            color=C_TOP if ridge[0]=="top" else C_BOT, ha="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gold", alpha=0.9))
    ax.text(x_hats[6,0]+0.3, x_hats[6,1]-0.5, r"$v_2$",
            fontsize=9, fontweight="bold",
            color=C_BOT if ridge[6]=="bot" else C_TOP, ha="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gold", alpha=0.9))

    # Global max stars
    ax.scatter(*z_stars.T, c=C_GLOBAL, s=35, zorder=4, marker="*", alpha=0.4)

    # Info box
    gap = obj_gl - obj_val
    info_line1 = f"CM violations: {n_viol}/120     " + r"$\Phi$" + f" = {obj_val:.3f}     Gap = {gap:.3f}"
    if (u_pair is not None) and (cert_pair is not None):
        info_line2 = f"Avg pair viol  U_N = {u_pair:.3f}     Cert  λ·U_N = {cert_pair:.3f}"
        info_text = info_line1 + "\n" + info_line2
    else:
        info_text = info_line1

    ax.text(0.5, 0.02,
        info_text,
        transform=ax.transAxes, fontsize=8, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.92,
                  edgecolor=method_color, linewidth=1.5))

    ax.set_xlim(-10, 10); ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel(r"$z_1$", fontsize=15); ax.set_ylabel(r"$z_2$", fontsize=15)
    ax.set_title(title, fontsize=11)


def plot_theorem(ax, cm_vals, gap_vals, z_dest, n_viol, thm_verified,
                 method_name, method_color):
    """
    Theorem scatter: |CM_{(i,j)}| vs delta_i + delta_j for one method.
    """
    viol_mask = cm_vals < -1e-10
    conf_mask = ~viol_mask

    # Non-violating (grey, faded)
    ax.scatter(np.abs(cm_vals[conf_mask]), gap_vals[conf_mask],
               c="#ccc", s=14, alpha=0.25, zorder=2,
               label=rf"CM $\geq 0$ ({conf_mask.sum()} pairs)")

    # Violating (method color, highlighted)
    if viol_mask.any():
        ax.scatter(np.abs(cm_vals[viol_mask]), gap_vals[viol_mask],
                   c=method_color, s=48, alpha=0.85, zorder=5,
                   edgecolors="#333", linewidths=0.5,
                   label=rf"CM $< 0$ ({viol_mask.sum()} crossings)")

    # Theorem bound line
    xmax_plot = max(np.abs(cm_vals).max(), 1.0) * 1.15
    xl = np.linspace(0, xmax_plot, 300)
    yl = 2*LAM*xl
    ax.plot(xl, yl, "--", color=C_BOUND, lw=2.0, alpha=0.8,
            label=r"Bound $2\lambda|$CM$|$")
    ymax_p = max(gap_vals.max(), yl.max()) * 1.3
    ax.fill_between(xl, yl, ymax_p, color=C_GLOBAL, alpha=0.04)

    # Highlight canonical pair (0, 6) if it exists
    if canon_idx is not None:
        cx = np.abs(cm_vals[canon_idx])
        cy = gap_vals[canon_idx]
        ax.scatter([cx], [cy], c="gold", s=130, zorder=8,
                   edgecolors="#8B0000", linewidths=1.8, marker="D")
        ax.annotate(f"$(v_1,v_2)$\n|CM|={cx:.1f}",
            xy=(cx, cy),
            xytext=(cx - xmax_plot*0.15, cy + max(3, gap_vals.max()*0.15)),
            fontsize=10, ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="gold", lw=1.3),
            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="gold", alpha=0.95))

    # Worst pair annotation
    worst_k = np.argmin(cm_vals)
    if cm_vals[worst_k] < -1e-10:
        wpi, wpj = pairs[worst_k]
        wx, wy = np.abs(cm_vals[worst_k]), gap_vals[worst_k]
        ax.annotate(f"Worst ({wpi},{wpj})\n|CM|={wx:.1f}",
            xy=(wx, wy), xytext=(wx + xmax_plot*0.06, wy*0.55),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="->", color=method_color, lw=1),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9))

    # Verification tag
    ax.text(0.97, 0.03,
        f"Theorem verified: {thm_verified}",
        transform=ax.transAxes, fontsize=7.5,
        ha="right", va="bottom", fontstyle="italic",
        color=C_GLOBAL if thm_verified else "red",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

    ax.set_xlabel(r"$|\mathrm{CM}_{(i,j)}|$", fontsize=10)
    ax.set_ylabel(r"$\delta_i + \delta_j$", fontsize=10)
    ax.set_title(f"{method_name}: theorem scatter  ({n_viol} violations)",
                 fontsize=11)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.92)
    ax.set_ylim(bottom=-0.1 * max(1, gap_vals.max()))


# ======================================================================
# Figure: 3 rows × 2 columns
# ======================================================================
print("Plotting ...")

fig = plt.figure(figsize=(17, 20))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25,
                      left=0.055, right=0.955, top=0.92, bottom=0.04)

fig.suptitle(
    "Projected Particle Ascent: Method-by-Method Comparison\n"
    "Transport Maps and Theorem Verification",
    fontsize=15, fontweight="bold", y=0.98)
fig.text(0.5, 0.94,
    r"Red arrow = top ridge ($y\!\approx\!+1$)"
    r"          Blue arrow = bottom ridge ($y\!\approx\!-1$)"
    r"          Green $\star$ = global max"
    r"          Gold $\diamondsuit$ = canonical pair $(v_1, v_2)$",
    fontsize=9.5, ha="center", fontstyle="italic", color="#555")

# ── Row 1: Vanilla PA ────────────────────────────────────────────────
ax_t1 = fig.add_subplot(gs[0, 0])
ax_s1 = fig.add_subplot(gs[0, 1])

plot_transport(ax_t1, z_pa,
    rf"(A1)  Vanilla PA — transport map  ($\lambda\!=\!{LAM}$)",
    n_viol_pa, obj_pa, METHOD_COLORS["Vanilla PA"],
    u_pair=U_pa, cert_pair=cert_pa_avg)

plot_theorem(ax_s1, cm_pa, gaps_pa, z_pa, n_viol_pa, thm_pa,
    "(A2)  Vanilla PA", METHOD_COLORS["Vanilla PA"])

# Row label
fig.text(0.01, 0.82, "Vanilla\nPA", fontsize=12, fontweight="bold",
         color=METHOD_COLORS["Vanilla PA"], rotation=90, va="center", ha="center")

# ── Row 2: Hard-Projected PA ─────────────────────────────────────────
ax_t2 = fig.add_subplot(gs[1, 0])
ax_s2 = fig.add_subplot(gs[1, 1])

plot_transport(ax_t2, z_hppa,
    "(B1)  Hard-Projected PA — transport map  (QP every 100 steps)",
    n_viol_hppa, obj_hppa, METHOD_COLORS["Hard Proj-PA"],
    u_pair=U_hppa, cert_pair=cert_hppa_avg)

plot_theorem(ax_s2, cm_hppa, gaps_hppa, z_hppa, n_viol_hppa, thm_hppa,
    "(B2)  Hard Proj-PA", METHOD_COLORS["Hard Proj-PA"])

fig.text(0.01, 0.52, "Hard\nProj-PA", fontsize=12, fontweight="bold",
         color=METHOD_COLORS["Hard Proj-PA"], rotation=90, va="center", ha="center")

# ── Row 3: Soft-Projected PA ─────────────────────────────────────────
ax_t3 = fig.add_subplot(gs[2, 0])
ax_s3 = fig.add_subplot(gs[2, 1])

plot_transport(ax_t3, z_sppa,
    rf"(C1)  Soft-Projected PA — transport map  ($\mu\!=\!0.5$)",
    n_viol_sppa, obj_sppa, METHOD_COLORS["Soft Proj-PA"],
    u_pair=U_sppa, cert_pair=cert_sppa_avg)

plot_theorem(ax_s3, cm_sppa, gaps_sppa, z_sppa, n_viol_sppa, thm_sppa,
    "(C2)  Soft Proj-PA", METHOD_COLORS["Soft Proj-PA"])

fig.text(0.01, 0.20, "Soft\nProj-PA", fontsize=12, fontweight="bold",
         color=METHOD_COLORS["Soft Proj-PA"], rotation=90, va="center", ha="center")

# ── Shared legend at bottom ──────────────────────────────────────────
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_TOP,
           markersize=9, markeredgecolor="white", markeredgewidth=1,
           label=r"Source → top ridge ($y_0\!>\!0$)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_BOT,
           markersize=9, markeredgecolor="white", markeredgewidth=1,
           label=r"Source → bottom ridge ($y_0\!<\!0$)"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor=C_GLOBAL,
           markersize=11, label=r"Global max $z_i^\star$"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="gold",
           markersize=8, markeredgecolor="#8B0000", markeredgewidth=1,
           label=r"Canonical pair $(v_1,v_2)$"),
    Line2D([0],[0], color=C_BOUND, ls="--", lw=2,
           label=r"Theorem bound $2\lambda|$CM$|$"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=5,
           fontsize=15, framealpha=0.92, bbox_to_anchor=(0.5, -0.009))


# plt.savefig("/home/method_comparison_rows.png", dpi=200, bbox_inches="tight")
# plt.savefig("/home/method_comparison_rows.pdf", dpi=200, bbox_inches="tight")
# print("Figures saved.")
plt.show()

# Summary table
print(f"\n{'='*65}")
print(f"  SUMMARY   (lambda = {LAM},  N = {N})")
print(f"{'='*65}")
print(f"  {'Method':<18s} {'Viol':>6s} {'Obj':>9s} {'Gap':>9s} {'Thm':>6s}")
print(f"  {'─'*48}")
print(f"  {'Global optimum':<18s} {'0':>6s} {obj_gl:>9.3f} {'0.000':>9s} {'—':>6s}")
print(f"  {'Vanilla PA':<18s} {n_viol_pa:>6d} {obj_pa:>9.3f} {obj_gl-obj_pa:>9.3f} {str(thm_pa):>6s}")
print(f"  {'Hard Proj-PA':<18s} {n_viol_hppa:>6d} {obj_hppa:>9.3f} {obj_gl-obj_hppa:>9.3f} {str(thm_hppa):>6s}")
print(f"  {'Soft Proj-PA':<18s} {n_viol_sppa:>6d} {obj_sppa:>9.3f} {obj_gl-obj_sppa:>9.3f} {str(thm_sppa):>6s}")
print(f"  {'─'*48}")
print(f"  Certificates (global empirical):")
print(f"    Vanilla PA:   gap >= λ·U_N = {cert_pa_avg:.4f}   (U_N={U_pa:.4f})")
print(f"                 gap >= (2λ/N)·max h_ij = {cert_pa_worst:.4f}")
print(f"                 gap >= (2λ/N)·Viol_cover = {cert_pa_cover:.4f}   (Viol_cover={viol_cover_pa:.3f})")
print(f"    Hard Proj-PA: gap >= λ·U_N = {cert_hppa_avg:.4f}   (U_N={U_hppa:.4f})")
print(f"                 gap >= (2λ/N)·max h_ij = {cert_hppa_worst:.4f}")
print(f"                 gap >= (2λ/N)·Viol_cover = {cert_hppa_cover:.4f}   (Viol_cover={viol_cover_hppa:.3f})")
print(f"    Soft Proj-PA: gap >= λ·U_N = {cert_sppa_avg:.4f}   (U_N={U_sppa:.4f})")
print(f"                 gap >= (2λ/N)·max h_ij = {cert_sppa_worst:.4f}")
print(f"                 gap >= (2λ/N)·Viol_cover = {cert_sppa_cover:.4f}   (Viol_cover={viol_cover_sppa:.3f})")

print(f"\n  Mini-batch certificate (B={B_MB}, δ={DELTA_MB}, D_emp={D_emp:.2f}):")
print(f"    Batch indices: {sorted(batch_idx.tolist())}")
print(f"    Vanilla PA:   gap >= {cert_pa_mb:.4f}   (U_B={UB_pa:.4f}, eps={eps_mb:.4f})")
print(f"    Hard Proj-PA: gap >= {cert_hppa_mb:.4f}   (U_B={UB_hppa:.4f}, eps={eps_mb:.4f})")
print(f"    Soft Proj-PA: gap >= {cert_sppa_mb:.4f}   (U_B={UB_sppa:.4f}, eps={eps_mb:.4f})")

print(f"{'='*65}")
