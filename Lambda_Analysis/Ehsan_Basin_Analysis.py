import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# ==========================================
# 1. Configuration & Mathematical Definitions
# ==========================================

a_1 = 2
a_2 = 5
c_1 = np.array([2.0, 2.0])
c_2 = np.array([-2.0, -2.0])
starting_point = np.array([0.2, 0.2])

def f_0(x, y):
    term1 = np.exp(-((x - c_1[0])**2 + (y - c_1[1])**2))
    term2 = np.exp(-((x - c_2[0])**2 + (y - c_2[1])**2))
    return a_1 * term1 + a_2 * term2

def grad_f_lambda(x, y, lam):
    term1 = np.exp(-((x - c_1[0])**2 + (y - c_1[1])**2))
    term2 = np.exp(-((x - c_2[0])**2 + (y - c_2[1])**2))
    
    df0_dx = -2 * a_1 * (x - c_1[0]) * term1 - 2 * a_2 * (x - c_2[0]) * term2
    df0_dy = -2 * a_1 * (y - c_1[1]) * term1 - 2 * a_2 * (y - c_2[1]) * term2
    
    dx = df0_dx - 2 * lam * (x - starting_point[0])
    dy = df0_dy - 2 * lam * (y - starting_point[1])
    
    return dx, dy

def f_lambda(x, y, lam):
    return f_0(x, y) - lam * ((x - starting_point[0])**2 + (y - starting_point[1])**2)

# ==========================================
# 2. Setup & Color Mapping
# ==========================================

lambdas = np.linspace(1.5, 0.0, 100)

# Colors mapped to: 0 (Blue, right side), 1 (Purple, left side), 2 (Pink, merged center)
cmap = ListedColormap(['#9999FF', '#DDAAEE', '#FF9999']) 

# ==========================================
# 3. Precise Local Maxima Finder
# ==========================================

def get_exact_attractors(lam, steps=2000, lr=0.02):
    """Tracks the exact final positions of the 3 fundamental structural points."""
    pts = np.array([c_1, c_2, starting_point], dtype=float)
    
    for _ in range(steps):
        dx, dy = grad_f_lambda(pts[:, 0], pts[:, 1], lam)
        pts[:, 0] += lr * dx
        pts[:, 1] += lr * dy
        
    return pts

def get_unique_maxima(attractors, threshold=1e-1):
    """Filters out duplicate peaks if points converged to the same max."""
    unique_maxima = []
    for pt in attractors:
        if not any(np.linalg.norm(pt - u) < threshold for u in unique_maxima):
            unique_maxima.append(pt)
    return np.array(unique_maxima)

# ==========================================
# 4. Core Drawing Logic
# ==========================================

def draw_surface(ax, lam, resolution=200):
    ax.clear()
    x_val = np.linspace(-4, 4, resolution)
    y_val = np.linspace(-4, 4, resolution)
    X, Y = np.meshgrid(x_val, y_val)
    
    Z = f_lambda(X, Y, lam)
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_title(f"Function Surface ($\lambda$ = {lam:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_zlim(-10, 7)
    ax.view_init(elev=30, azim=45)

def draw_basin(ax, lam, resolution=250, steps=200, lr=0.02):
    ax.clear()
    
    # 1. Find the TRUE unique maxima
    attractors = get_exact_attractors(lam)
    unique_maxs = get_unique_maxima(attractors)
    
    # 2. Sort maxima by X-coordinate to assign stable colors across GIF frames
    unique_maxs = np.array(sorted(unique_maxs, key=lambda val: val[0]))
    labels = []
    if len(unique_maxs) == 1:
        labels = [2] 
    elif len(unique_maxs) == 2:
        labels = [1, 0] 
    elif len(unique_maxs) == 3:
        labels = [1, 2, 0] 
    
    # 3. Create the grid and run gradient ascent
    x_val = np.linspace(-4, 4, resolution)
    y_val = np.linspace(-4, 4, resolution)
    X, Y = np.meshgrid(x_val, y_val)
    Px, Py = X.copy(), Y.copy()
    
    for _ in range(steps):
        dx, dy = grad_f_lambda(Px, Py, lam)
        Px += lr * dx
        Py += lr * dy

    # 4. Calculate distance ONLY to the actual unique maxima
    distances = []
    for um in unique_maxs:
        dist = (Px - um[0])**2 + (Py - um[1])**2
        distances.append(dist)
    
    distances = np.stack(distances, axis=-1)
    
    # closest_idx tells us which unique max (0, 1, or 2) each point is closest to
    closest_idx = np.argmin(distances, axis=-1)
    
    # Map that index to our stable color labels
    basins = np.zeros_like(closest_idx)
    for i, label in enumerate(labels):
        basins[closest_idx == i] = label

    # 5. Draw the colored basins
    # vmin=0, vmax=2 guarantees the colors map correctly to the ListedColormap
    ax.pcolormesh(X, Y, basins, cmap=cmap, vmin=0, vmax=2, shading='auto', alpha=0.6)
    
    # 6. ADD LEVEL SETS
    Z = f_lambda(X, Y, lam)
    ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.4, linewidths=0.8)
    
    # 7. Plot the unique maxima
    for mx, my in unique_maxs:
        ax.plot(mx, my, 'k*', markersize=12, markeredgecolor='white')

    ax.set_title(f"Basins of Attraction ($\lambda$ = {lam:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.0, 4.0)

# ==========================================
# 5. Standalone Frame Generators
# ==========================================

def plot_specific_frame_surface(lam):
    print(f"Plotting surface frame for lambda = {lam}...")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    draw_surface(ax, lam, resolution=500) 
    plt.savefig(f"surface_lambda_{lam:.2f}.png")
    plt.close(fig)

def plot_specific_frame_basin(lam):
    print(f"Plotting basin frame for lambda = {lam}...")
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_basin(ax, lam, resolution=800, steps=400) 
    plt.savefig(f"basins_lambda_{lam:.2f}.png")
    plt.close(fig)

# ==========================================
# 6. Animation Callbacks & GIF Generation
# ==========================================

def update_surface_frame(frame, ax):
    draw_surface(ax, lambdas[frame], resolution=150)

def update_basin_frame(frame, ax):
    draw_basin(ax, lambdas[frame], resolution=250, steps=150)

def generate_surface_gif():
    print("Generating 3D Surface GIF...")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ani = animation.FuncAnimation(fig, update_surface_frame, frames=len(lambdas), fargs=(ax,), interval=200)
    ani.save("surface_evolution.gif", writer="pillow", fps=5)
    plt.close(fig)

def generate_basins_gif():
    print("Generating Basins of Attraction GIF...")
    fig, ax = plt.subplots(figsize=(7, 6))
    ani = animation.FuncAnimation(fig, update_basin_frame, frames=len(lambdas), fargs=(ax,), interval=200)
    ani.save("basins_evolution.gif", writer="pillow", fps=5)
    plt.close(fig)

# ==========================================
# Run the Code
# ==========================================

if __name__ == "__main__":
    # Generate the GIFs
    generate_surface_gif()
    generate_basins_gif()
    
    # Test your exact problematic frame here:
    plot_specific_frame_surface(0.2)
    plot_specific_frame_basin(0.2)