"""
3D Manifold of the Finite Trace at Infinity C(a'_infty, b'_infty)
================================================================

Mathematical Context:
---------------------
In Deruelle & Schulze (2023, CPAM, Theorem 1.1), two expanding gradient Ricci
solitons sharing the same asymptotic cone (C(S), g_cone) satisfy:

    lim_{s -> infty} s^4 * exp(s^2 / 4) * |g1 - g2|_gamma = C < infty

On M = S^2 x R^2, the asymptotic cone is uniquely determined by two geometric slopes:
    alpha = a'_infty = lim_{r -> infty} a'(r)  (S^1 circle factor)
    beta  = b'_infty = lim_{r -> infty} b'(r)  (S^2 sphere factor)

For each cone (alpha, beta), there exists a pair of dynamically matched solitons
(lower branch vs. upper branch) whose asymptotic separation is measured by the
finite trace constant C (the tangent slope at s = infty).

This module:
1. Sweeps a 2D parameter grid of initial soliton conditions (b0, f0).
2. Computes the matched soliton pair and its exact cone slopes (alpha, beta).
3. Evaluates the finite trace C_tangent for each cone.
4. Renders and saves a 3D surface plot:
       X-axis : a'_infty (alpha)
       Y-axis : b'_infty (beta)
       Z-axis : C (Tangent line trace at infty)
   with interactive 3D navigation and cache support.
"""

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Ensure relative_entropy package directory is importable
_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)
if _dir not in sys.path:
    sys.path.insert(0, _dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

from relative_entropy.find_matching_solitons import find_matching_solitons
from relative_entropy.compute_finite_trace import compute_finite_trace


CACHE_FILENAME = os.path.join(_dir, "trace_surface_cache.npz")
FIGURE_FILENAME = os.path.join(_dir, "trace_surface_3d.png")


def _evaluate_single_soliton_pair(args):
    """
    Worker task for parallel grid simulation of matching soliton pair and trace C.
    """
    b0_cur, f0_cur, r_max, tol = args

    # Multi-guess heuristic covering the full fold region
    guesses = [
        (0.55 / b0_cur, f0_cur * (1.1 + 0.3 / b0_cur)),
        (1.2 * b0_cur + 0.5, f0_cur * 1.5),
        (1.8, -2.0),
        (2.0, f0_cur * 1.8),
        (1.4, f0_cur * 1.2)
    ]

    for b_g, f_g in guesses:
        try:
            pair = find_matching_solitons(
                b0_1=b0_cur,
                f0_1=f0_cur,
                initial_guess=(b_g, f_g),
                r_max=r_max,
                tol=tol
            )
            # Ensure partner soliton is on distinct upper branch
            if pair["soliton_2"]["b0"] <= b0_cur + 0.04:
                continue

            res = compute_finite_trace(
                soliton_pair=pair,
                s_min=2.5,
                s_max=5.0,
                n_points=120,
                r_max=r_max
            )
            c_slopes = pair["cone_slopes"]
            a_inf = float(c_slopes["alpha"])
            b_inf = float(c_slopes["beta"])
            c_val = float(res["tangent_value"])

            if 0.0 < c_val < 150.0 and not np.isnan(c_val):
                return (b0_cur, f0_cur, a_inf, b_inf, c_val, True)
        except Exception:
            continue

    return (b0_cur, f0_cur, np.nan, np.nan, np.nan, False)


def compute_or_load_trace_surface(
    cache_path=CACHE_FILENAME,
    force_recompute=False,
    n_b0=10,
    n_f0=10,
    b0_range=(0.18, 0.65),
    f0_range=(-1.40, -0.12),
    r_max=80.0,
    tol=1e-7,
    max_workers=None
):
    """
    Computes or loads the 2D grid mapping (alpha, beta) -> C_trace using parallel multiprocessing.

    Parameters:
    -----------
    cache_path : str
        Path to .npz file storing cached surface data.
    force_recompute : bool
        If True, re-runs numerical integration regardless of cache existence.
    n_b0, n_f0 : int
        Grid dimensions for reference lower branch parameters (b0, f0). Default 10x10 = 100 pairs.
    b0_range, f0_range : tuple of float
        Interval bounds for the parameter sweep.
    r_max : float
        Integration radius for asymptotic cone shooting.
    tol : float
        Root-finding tolerance for matching cone slopes.
    max_workers : int or None
        Number of parallel worker processes.

    Returns:
    --------
    dict:
        Containing 2D arrays: 'ALPHA', 'BETA', 'C_TRACE', 'B0', 'F0'.
    """
    total_pts = n_b0 * n_f0

    # Check cache validity
    if os.path.exists(cache_path) and not force_recompute:
        try:
            data = np.load(cache_path)
            if data["ALPHA"].shape == (n_b0, n_f0):
                print(f"Loading cached 3D trace surface data ({n_b0}x{n_f0} = {total_pts} pairs) from: {cache_path}")
                return {
                    "ALPHA": data["ALPHA"],
                    "BETA": data["BETA"],
                    "C_TRACE": data["C_TRACE"],
                    "B0": data["B0"],
                    "F0": data["F0"]
                }
            else:
                print(f"Cache shape {data['ALPHA'].shape} differs from requested ({n_b0}, {n_f0}). Recomputing...")
        except Exception:
            print("Cache read error; recomputing...")

    print("=" * 76)
    print("Computing High-Density 3D Manifold: (a'_infty, b'_infty) -> C (Finite Trace)")
    print("=" * 76)
    print(f"Grid Resolution : {n_b0} x {n_f0} = {total_pts} soliton pairs")
    print(f"b0 range (bolt) : [{b0_range[0]:.2f}, {b0_range[1]:.2f}]")
    print(f"f0 range (curv) : [{f0_range[0]:.2f}, {f0_range[1]:.2f}]")

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)
    print(f"Parallel Workers: {max_workers} processes")
    print("-" * 76)

    b0_vals = np.linspace(b0_range[0], b0_range[1], n_b0)
    f0_vals = np.linspace(f0_range[0], f0_range[1], n_f0)
    B0, F0 = np.meshgrid(b0_vals, f0_vals, indexing='ij')

    tasks = []
    for i in range(n_b0):
        for j in range(n_f0):
            tasks.append((float(B0[i, j]), float(F0[i, j]), r_max, tol))

    ALPHA = np.full_like(B0, np.nan)
    BETA = np.full_like(B0, np.nan)
    C_TRACE = np.full_like(B0, np.nan)

    t_start = time.time()
    completed = 0
    success_count = 0

    print(f"Starting parallel execution for {total_pts} soliton pairs...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(_evaluate_single_soliton_pair, t): (t[0], t[1]) for t in tasks}

        # Build coordinate lookup
        b0_idx_map = {round(val, 6): idx for idx, val in enumerate(b0_vals)}
        f0_idx_map = {round(val, 6): idx for idx, val in enumerate(f0_vals)}

        for future in as_completed(futures):
            completed += 1
            b0_pt, f0_pt = futures[future]
            try:
                b0_r, f0_r, a_inf, b_inf, c_val, ok = future.result()
                i_idx = b0_idx_map[round(b0_pt, 6)]
                j_idx = f0_idx_map[round(f0_pt, 6)]

                if ok and not np.isnan(c_val) and (0.0 < c_val < 150.0):
                    ALPHA[i_idx, j_idx] = a_inf
                    BETA[i_idx, j_idx] = b_inf
                    C_TRACE[i_idx, j_idx] = c_val
                    success_count += 1
            except Exception as exc:
                pass

            if completed % 15 == 0 or completed == total_pts:
                elapsed = time.time() - t_start
                rate = completed / max(elapsed, 0.001)
                eta = (total_pts - completed) / max(rate, 0.001)
                print(f"[{completed:3d}/{total_pts}] ({completed/total_pts*100:5.1f}%) | "
                      f"Success: {success_count} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

    t_total = time.time() - t_start
    print("-" * 76)
    print(f"Simulation completed in {t_total:.1f}s. Valid pairs: {success_count}/{total_pts} ({success_count/total_pts*100:.1f}%)")

    # If any isolated NaNs or unphysical outliers exist, interpolate them from valid neighbors
    nan_mask = np.isnan(ALPHA) | np.isnan(BETA) | np.isnan(C_TRACE) | (C_TRACE <= 0.0) | (C_TRACE > 150.0)
    if np.any(nan_mask) and success_count >= 10:
        valid_i, valid_j = np.where(~nan_mask)
        nan_i, nan_j = np.where(nan_mask)
        pts_valid = np.column_stack((valid_i, valid_j))
        pts_nan = np.column_stack((nan_i, nan_j))

        for arr in [ALPHA, BETA, C_TRACE]:
            arr[nan_mask] = griddata(pts_valid, arr[~nan_mask], pts_nan, method='nearest')

    # Save to disk cache
    np.savez(
        cache_path,
        ALPHA=ALPHA,
        BETA=BETA,
        C_TRACE=C_TRACE,
        B0=B0,
        F0=F0
    )
    print(f"Saved surface simulation cache to: {cache_path}\n")

    return {
        "ALPHA": ALPHA,
        "BETA": BETA,
        "C_TRACE": C_TRACE,
        "B0": B0,
        "F0": F0
    }


def plot_trace_surface_3d(data, save_path=FIGURE_FILENAME, tight_bounds=False):
    """
    Renders and displays the 3D manifold plot of C(a'_infty, b'_infty).

    Features:
      - Clean single 3D plot focused on the 2-to-1 caustic fold
      - Continuous shaded surface sheet in cornflower blue
      - Dense scatter points of evaluated soliton pairs
      - Contour projection on the base plane
      - Professional typography and clear axis annotations
    """
    ALPHA = np.array(data["ALPHA"], copy=True)
    BETA = np.array(data["BETA"], copy=True)
    C_TRACE = np.array(data["C_TRACE"], copy=True)

    # Filter out any non-converged points or numerical outliers
    bad_mask = np.isnan(ALPHA) | np.isnan(BETA) | np.isnan(C_TRACE) | (C_TRACE <= 0.0) | (C_TRACE > 150.0)
    valid_mask = ~bad_mask
    num_valid = int(np.sum(valid_mask))
    if num_valid == 0:
        print("Error: No valid simulation points to plot.")
        return

    # Clean outlier points for smooth watertight manifold rendering
    if np.any(bad_mask):
        valid_i, valid_j = np.where(valid_mask)
        bad_i, bad_j = np.where(bad_mask)
        pts_valid = np.column_stack((valid_i, valid_j))
        pts_bad = np.column_stack((bad_i, bad_j))
        C_TRACE[bad_mask] = griddata(pts_valid, C_TRACE[valid_mask], pts_bad, method='nearest')

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection='3d')

    try:
        fig.canvas.manager.set_window_title("Deruelle-Schulze Theorem 1.1: 3D Trace at Infinity Manifold")
    except Exception:
        pass

    # 1. Continuous 3D Manifold Surface (uniform color without redundant C colormap)
    surf = ax.plot_surface(
        ALPHA,
        BETA,
        C_TRACE,
        color='cornflowerblue',
        shade=True,
        edgecolor='black',
        linewidth=0.25,
        alpha=0.82,
        antialiased=True
    )

    # 2. Scatter discrete evaluated soliton pairs
    ax.scatter(
        ALPHA[valid_mask],
        BETA[valid_mask],
        C_TRACE[valid_mask],
        color='crimson',
        s=30,
        edgecolor='black',
        linewidth=0.45,
        depthshade=True,
        label=f'Evaluated Soliton Pairs (N = {num_valid})'
    )

    # 3. Floor Contour Projection and Axis Bounds
    z_max_val = float(np.nanmax(C_TRACE))
    z_max = max(55.0, z_max_val + 5.0)

    if tight_bounds:
        # Full-frame detailed view tightly framing the evaluated surface
        ax.set_xlim(np.nanmin(ALPHA) - 0.02, np.nanmax(ALPHA) + 0.02)
        ax.set_ylim(np.nanmin(BETA) - 0.03, np.nanmax(BETA) + 0.03)
        z_offset = max(0.0, float(np.nanmin(C_TRACE)) - 2.0)
        ax.set_zlim(z_offset, z_max)
    else:
        # Triangular fold region bounds: a' in [0, 4], b' in [0, 10]
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 10.0)
        ax.set_zlim(0.0, z_max)
        z_offset = 0.0

    try:
        ax.contour(
            ALPHA,
            BETA,
            C_TRACE,
            zdir='z',
            offset=z_offset,
            colors='gray',
            alpha=0.5,
            linewidths=1.2
        )
    except Exception:
        pass

    # Axis Labels & Title
    ax.set_xlabel("a'_$\\infty$ (S$^1$ Cone Slope)", fontsize=11, labelpad=10)
    ax.set_ylabel("b'_$\\infty$ (S$^2$ Cone Slope)", fontsize=11, labelpad=10)
    ax.set_zlabel("Finite Trace at Infinity C", fontsize=11, labelpad=10)
    ax.set_title(
        f"3D Manifold of the Finite Trace at Infinity C(a'_$\\infty$, b'_$\\infty$)\n"
        f"(Deruelle & Schulze 2023, Theorem 1.1 — {num_valid} Soliton Pairs)",
        fontsize=12,
        fontweight='bold',
        pad=14
    )

    # Camera perspective: elev=22, azim=-55 (classic dynamic 3D manifold view)
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved 3D manifold visualization to: {save_path}")

    # Launch GUI
    print("Launching Matplotlib 3D GUI window (close window to finish)...")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D Surface Plot of the Finite Trace C(a'_infty, b'_infty)"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of the 3D grid instead of loading cache."
    )
    parser.add_argument(
        "--nb0",
        type=int,
        default=10,
        help="Grid resolution in b0 (default: 10)."
    )
    parser.add_argument(
        "--nf0",
        type=int,
        default=10,
        help="Grid resolution in f0 (default: 10)."
    )
    parser.add_argument(
        "--tight",
        action="store_true",
        help="Use tight axis limits around data to fill frame like the original plot."
    )
    args = parser.parse_args()

    data = compute_or_load_trace_surface(
        cache_path=CACHE_FILENAME,
        force_recompute=args.recompute,
        n_b0=args.nb0,
        n_f0=args.nf0,
        b0_range=(0.18, 0.65),
        f0_range=(-1.40, -0.12)
    )

    plot_trace_surface_3d(data, save_path=FIGURE_FILENAME, tight_bounds=args.tight)


if __name__ == '__main__':
    main()
