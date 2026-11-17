"""
3D Manifold of the Cross-Topological Finite Trace at Infinity C_AC(a'_infty, b'_infty)
====================================================================================

Mathematical Context:
---------------------
In Deruelle & Schulze (2023, CPAM, Theorem 1.1), two expanding gradient Ricci
solitons sharing the same asymptotic cone (C(S), g_cone) satisfy:

    lim_{s -> infty} s^4 * exp(s^2 / 4) * |g1 - g2|_gamma = C < infty

On 4-dimensional cohomogeneity-one manifolds with SO(3) x SO(2) symmetry, two
distinct complete non-singular topologies share the link S = S^1 x S^2:
1. M_1 = D^2 x S^2 (S^2 bolt at r = 0, circle S^1 collapses)
2. M_2 = D^3 x S^1 (S^1 circle core at r = 0, sphere S^2 collapses)

For any cone slopes (alpha*, beta*) in the physical expander region:
- Soliton A (S_A): Soliton on D^2 x S^2 (lower branch)
- Soliton C (S_C): Soliton on D^3 x S^1 (circle core)

Both solitons share the EXACT SAME asymptotic cone (alpha*, beta*).
Their asymptotic separation is measured by the cross-topological finite trace C_AC.

This module:
1. Sweeps a 2D parameter grid of initial conditions (b0, f0) on D^2 x S^2.
2. For each cone (alpha, beta), computes the matching soliton on D^3 x S^1.
3. Evaluates the Deruelle-Schulze finite trace C_AC.
4. Renders and saves an interactive 3D surface plot:
       X-axis : a'_infty (alpha, S^1 cone slope)
       Y-axis : b'_infty (beta,  S^2 cone slope)
       Z-axis : C_AC (Cross-topological finite trace at infinity)
"""

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata, CubicSpline
from scipy.optimize import root_scalar

# Ensure relative_entropy and root are importable
_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_dir, '..'))
if _dir not in sys.path:
    sys.path.insert(0, _dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.ricci_odes import ricci_soliton
from relative_entropy.compute_finite_trace import compute_pair_finite_trace
from relative_entropy.triple_solitons_d3s1 import (
    get_initial_conditions_f2,
    get_initial_conditions_f1,
    get_f2_slopes,
    get_f1_slopes
)

CACHE_FILENAME = os.path.join(_dir, "trace_surface_d3s1_cache.npz")
FIGURE_FILENAME = os.path.join(_dir, "trace_surface_d3s1_3d.png")


def _evaluate_single_d3s1_pair(args):
    """
    Worker task: computes target cone from Soliton A (D^2 x S^2),
    matches Soliton C (D^3 x S^1), and computes C_AC.
    """
    b0_cur, f0_cur, r_max, tol = args

    try:
        # Step 1: Cone slopes from Soliton A (D^2 x S^2)
        alpha_star, beta_star = get_f2_slopes(b0_cur, f0_cur, r_max=r_max)
        if np.isnan(alpha_star) or np.isnan(beta_star) or beta_star <= 0.05:
            return (b0_cur, f0_cur, np.nan, np.nan, np.nan, False)

        # Step 2: Solve Soliton C (D^3 x S^1) matching beta_star
        # For D^3 x S^1, beta depends strictly and monotonically on f0
        res_C = root_scalar(
            lambda f: get_f1_slopes(f, r_max=r_max)[1] - beta_star,
            bracket=[-4.0, -0.1],
            method='brentq',
            xtol=1e-7
        )
        if not res_C.converged:
            return (b0_cur, f0_cur, alpha_star, beta_star, np.nan, False)

        f0_C = res_C.root
        alpha_0_C, beta_C = get_f1_slopes(f0_C, r_max=r_max)
        if alpha_0_C <= 1e-6:
            return (b0_cur, f0_cur, alpha_star, beta_star, np.nan, False)

        a0_C = alpha_star / alpha_0_C

        # Step 3: Trajectory integration
        solA = solve_ivp(
            ricci_soliton, (1e-4, r_max),
            get_initial_conditions_f2(b0_cur, f0_cur),
            method='Radau', rtol=1e-8, atol=1e-10
        )
        solC = solve_ivp(
            ricci_soliton, (1e-4, r_max),
            get_initial_conditions_f1(a0_C, f0_C),
            method='Radau', rtol=1e-8, atol=1e-10
        )

        if solA.status != 0 or solC.status != 0:
            return (b0_cur, f0_cur, alpha_star, beta_star, np.nan, False)

        # Step 4: Compute Deruelle-Schulze finite trace C_AC
        res_AC = compute_pair_finite_trace(solA, solC, s_min=2.5, s_max=5.0)
        c_val = float(res_AC["C_trace"])
        r2 = float(res_AC.get("r2_score", 0.0))

        if 0.0 < c_val < 200.0 and not np.isnan(c_val) and r2 > 0.90:
            return (b0_cur, f0_cur, float(alpha_star), float(beta_star), c_val, True)
        else:
            return (b0_cur, f0_cur, float(alpha_star), float(beta_star), np.nan, False)

    except Exception:
        return (b0_cur, f0_cur, np.nan, np.nan, np.nan, False)


def compute_trace_surface_d3s1(
    n_b0=10,
    n_f0=10,
    b0_range=(0.20, 0.60),
    f0_range=(-0.70, -2.00),
    r_max=60.0,
    tol=1e-7,
    max_workers=None,
    force_recompute=False,
    cache_path=CACHE_FILENAME
):
    """
    Computes or loads the 2D grid simulation for the D^3 x S^1 trace surface C_AC.
    """
    total_pts = n_b0 * n_f0

    if not force_recompute and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            if data["ALPHA"].shape == (n_b0, n_f0):
                print(f"[Cache] Successfully loaded precomputed D3S1 surface from: {cache_path}")
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
    print("Computing 3D Cross-Topological Trace Manifold: D^2 x S^2 vs D^3 x S^1")
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

    print(f"Starting parallel execution for {total_pts} cross-topological pairs...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_evaluate_single_d3s1_pair, t): (t[0], t[1]) for t in tasks}

        b0_idx_map = {round(val, 6): idx for idx, val in enumerate(b0_vals)}
        f0_idx_map = {round(val, 6): idx for idx, val in enumerate(f0_vals)}

        for future in as_completed(futures):
            completed += 1
            b0_pt, f0_pt = futures[future]
            try:
                b0_r, f0_r, a_inf, b_inf, c_val, ok = future.result()
                i_idx = b0_idx_map[round(b0_pt, 6)]
                j_idx = f0_idx_map[round(f0_pt, 6)]

                if ok and not np.isnan(c_val) and (0.0 < c_val < 200.0):
                    ALPHA[i_idx, j_idx] = a_inf
                    BETA[i_idx, j_idx] = b_inf
                    C_TRACE[i_idx, j_idx] = c_val
                    success_count += 1
            except Exception:
                pass

            if completed % 10 == 0 or completed == total_pts:
                elapsed = time.time() - t_start
                rate = completed / max(elapsed, 0.001)
                eta = (total_pts - completed) / max(rate, 0.001)
                print(f"[{completed:3d}/{total_pts}] ({completed/total_pts*100:5.1f}%) | "
                      f"Success: {success_count} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

    t_total = time.time() - t_start
    print("-" * 76)
    print(f"Simulation completed in {t_total:.1f}s. Valid pairs: {success_count}/{total_pts} ({success_count/total_pts*100:.1f}%)")

    # Infill any missing border/interior points via nearest neighbor interpolation
    nan_mask = np.isnan(ALPHA) | np.isnan(BETA) | np.isnan(C_TRACE)
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
    print(f"Saved D3S1 surface cache to: {cache_path}\n")

    return {
        "ALPHA": ALPHA,
        "BETA": BETA,
        "C_TRACE": C_TRACE,
        "B0": B0,
        "F0": F0
    }


def plot_trace_surface_d3s1_3d(data, save_path=FIGURE_FILENAME, show_gui=True):
    """
    Renders an interactive 3D manifold surface for the cross-topological trace C_AC.
    """
    ALPHA = np.array(data["ALPHA"], copy=True)
    BETA = np.array(data["BETA"], copy=True)
    C_TRACE = np.array(data["C_TRACE"], copy=True)

    valid_mask = ~np.isnan(ALPHA) & ~np.isnan(BETA) & ~np.isnan(C_TRACE) & (C_TRACE > 0.0)
    num_valid = int(np.sum(valid_mask))
    if num_valid == 0:
        print("Error: No valid simulation points to plot.")
        return

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection='3d')

    try:
        fig.canvas.manager.set_window_title("D^3 x S^1 Cross-Topology Trace Manifold C_AC(a'_infty, b'_infty)")
    except Exception:
        pass

    # 1. Continuous 3D Manifold Surface in orchid/mediumpurple
    surf = ax.plot_surface(
        ALPHA,
        BETA,
        C_TRACE,
        color='mediumorchid',
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
        color='darkviolet',
        s=32,
        edgecolor='black',
        linewidth=0.5,
        depthshade=True,
        label=f'Cross-Topological Pairs ($D^2 \\times S^2$ vs $D^3 \\times S^1$, N={num_valid})'
    )

    # 3. Floor Contour Projection and Axis Bounds
    z_min_val = float(np.nanmin(C_TRACE))
    z_max_val = float(np.nanmax(C_TRACE))
    z_offset = max(0.0, z_min_val - 0.5)
    z_max = z_max_val * 1.15

    ax.set_xlim(np.nanmin(ALPHA) - 0.02, np.nanmax(ALPHA) + 0.02)
    ax.set_ylim(np.nanmin(BETA) - 0.03, np.nanmax(BETA) + 0.03)
    ax.set_zlim(z_offset, z_max)

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
    ax.set_xlabel(r"$a'_\infty$ ($S^1$ Cone Slope)", fontsize=11, labelpad=10)
    ax.set_ylabel(r"$b'_\infty$ ($S^2$ Cone Slope)", fontsize=11, labelpad=10)
    ax.set_zlabel(r"Cross-Topological Trace $C_{AC}$", fontsize=11, labelpad=10)
    ax.set_title(
        f"3D Manifold of Cross-Topology Finite Trace at Infinity $C_{{AC}}(a'_\\infty, b'_\\infty)$\n"
        f"(Deruelle & Schulze 2023, Theorem 1.1 — $D^2 \\times S^2$ vs. $D^3 \\times S^1$)",
        fontsize=12,
        fontweight='bold',
        pad=14
    )

    # Camera perspective: classic dynamic 3D angle
    ax.view_init(elev=24, azim=-50)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved 3D cross-topology trace manifold visualization to: {save_path}")

    # Launch GUI
    if '--no-gui' in sys.argv or not show_gui:
        print("Skipping GUI window display (--no-gui flag detected).")
    else:
        print("Launching Matplotlib 3D GUI window (close window to finish)...")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D Surface Plot of the Cross-Topological Trace C_AC(a'_infty, b'_infty) for D^3 x S^1"
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=10,
        help="Resolution of grid per dimension (default: 10, total 100 pairs)"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of the 2D grid instead of reading cache."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without displaying the Matplotlib GUI window and save figure directly."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes."
    )

    args = parser.parse_args()

    data = compute_trace_surface_d3s1(
        n_b0=args.grid_size,
        n_f0=args.grid_size,
        b0_range=(0.20, 0.60),
        f0_range=(-0.70, -2.00),
        r_max=60.0,
        tol=1e-7,
        max_workers=args.workers,
        force_recompute=args.recompute
    )

    plot_trace_surface_d3s1_3d(
        data,
        save_path=FIGURE_FILENAME,
        show_gui=(not args.no_gui)
    )


if __name__ == "__main__":
    main()
