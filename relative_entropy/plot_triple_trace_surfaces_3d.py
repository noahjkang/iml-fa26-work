"""
Triple Trace Comparison: All 3 Surfaces (C_AB, C_AC, C_BC) in 3D
=================================================================

Mathematical Context:
---------------------
In Deruelle & Schulze (2023, CPAM, Theorem 1.1), the relative trace at infinity
between two expanding gradient Ricci solitons emerging from the same asymptotic cone
(C(S), g_cone) is finite and non-zero:
    lim_{s -> infty} s^4 * exp(s^2 / 4) * |g1 - g2|_gamma = C < infty.

On 4-dimensional cohomogeneity-one manifolds with SO(3) x SO(2) symmetry, two
distinct complete non-singular topologies share the link S = S^1 x S^2:
1. M = D^2 x S^2 (degree 0, 2-to-1 inside caustic fold: Soliton A lower, Soliton B upper)
2. M = D^3 x S^1 (degree 1, globally 1-to-1: Soliton C unique)

For any cone (alpha*, beta*) in the fold region, there exists a geometric triple
of solitons (S_A, S_B, S_C) sharing the EXACT SAME asymptotic cone, generating
all 3 pairwise finite traces:
- C_AB: Soliton A vs Soliton B (both on D^2 x S^2)
- C_AC: Soliton A (D^2 x S^2 lower) vs Soliton C (D^3 x S^1)
- C_BC: Soliton B (D^2 x S^2 upper) vs Soliton C (D^3 x S^1)

This module computes and plots all three trace surfaces (C_AB, C_AC, C_BC)
simultaneously on the same 3D coordinate chart over the (a'_infty, b'_infty) cone plane.
"""

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata
from scipy.optimize import root_scalar

# Ensure relative_entropy and root are importable
_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_dir, '..'))
if _dir not in sys.path:
    sys.path.insert(0, _dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.ricci_odes import ricci_soliton
from relative_entropy.find_matching_solitons import find_matching_solitons
from relative_entropy.compute_finite_trace import compute_pair_finite_trace
from relative_entropy.triple_solitons_d3s1 import (
    get_initial_conditions_f2,
    get_initial_conditions_f1,
    get_f2_slopes,
    get_f1_slopes
)

CACHE_FILENAME = os.path.join(_dir, "triple_trace_surfaces_cache.npz")
FIGURE_FILENAME = os.path.join(_dir, "triple_trace_surfaces_3d.png")


def _evaluate_single_triple(args):
    """
    Worker task: computes target cone from Soliton A (D^2 x S^2, lower),
    matches Soliton B (D^2 x S^2, upper) and Soliton C (D^3 x S^1),
    integrates all three trajectories, and computes C_AB, C_AC, C_BC.
    """
    b0_cur, f0_cur, r_max, tol = args

    try:
        # Step 1: Cone slopes from Soliton A
        alpha_star, beta_star = get_f2_slopes(b0_cur, f0_cur, r_max=r_max)
        if np.isnan(alpha_star) or np.isnan(beta_star) or beta_star <= 0.05:
            return (b0_cur, f0_cur, np.nan, np.nan, np.nan, np.nan, np.nan, False)

        # Step 2: Solve Soliton B (D^2 x S^2, upper branch)
        guesses = [
            (0.55 / b0_cur, f0_cur * (1.1 + 0.3 / b0_cur)),
            (1.2 * b0_cur + 0.5, f0_cur * 1.5),
            (1.8, -2.0),
            (2.0, f0_cur * 1.8),
            (1.4, f0_cur * 1.2)
        ]
        b0_B, f0_B = None, None
        for b_g, f_g in guesses:
            try:
                pair = find_matching_solitons(
                    b0_1=b0_cur,
                    f0_1=f0_cur,
                    initial_guess=(b_g, f_g),
                    r_max=r_max,
                    tol=tol
                )
                if pair["soliton_2"]["b0"] > b0_cur + 0.04:
                    b0_B = pair["soliton_2"]["b0"]
                    f0_B = pair["soliton_2"]["f0"]
                    break
            except Exception:
                continue

        if b0_B is None:
            return (b0_cur, f0_cur, alpha_star, beta_star, np.nan, np.nan, np.nan, False)

        # Step 3: Solve Soliton C (D^3 x S^1)
        res_C = root_scalar(
            lambda f: get_f1_slopes(f, r_max=r_max)[1] - beta_star,
            bracket=[-4.0, -0.1],
            method='brentq',
            xtol=1e-7
        )
        if not res_C.converged:
            return (b0_cur, f0_cur, alpha_star, beta_star, np.nan, np.nan, np.nan, False)

        f0_C = res_C.root
        alpha_0_C, beta_C = get_f1_slopes(f0_C, r_max=r_max)
        if alpha_0_C <= 1e-6:
            return (b0_cur, f0_cur, alpha_star, beta_star, np.nan, np.nan, np.nan, False)

        a0_C = alpha_star / alpha_0_C

        # Step 4: Integrate all three trajectories
        solA = solve_ivp(
            ricci_soliton, (1e-4, r_max),
            get_initial_conditions_f2(b0_cur, f0_cur),
            method='Radau', rtol=1e-8, atol=1e-10
        )
        solB = solve_ivp(
            ricci_soliton, (1e-4, r_max),
            get_initial_conditions_f2(b0_B, f0_B),
            method='Radau', rtol=1e-8, atol=1e-10
        )
        solC = solve_ivp(
            ricci_soliton, (1e-4, r_max),
            get_initial_conditions_f1(a0_C, f0_C),
            method='Radau', rtol=1e-8, atol=1e-10
        )

        if solA.status != 0 or solB.status != 0 or solC.status != 0:
            return (b0_cur, f0_cur, alpha_star, beta_star, np.nan, np.nan, np.nan, False)

        # Step 5: Compute all 3 pairwise finite traces
        res_AB = compute_pair_finite_trace(solA, solB, s_min=2.5, s_max=5.0)
        res_AC = compute_pair_finite_trace(solA, solC, s_min=2.5, s_max=5.0)
        res_BC = compute_pair_finite_trace(solB, solC, s_min=2.5, s_max=5.0)

        c_ab = float(res_AB["C_trace"])
        c_ac = float(res_AC["C_trace"])
        c_bc = float(res_BC["C_trace"])

        if (0.0 < c_ab < 250.0 and 0.0 < c_ac < 250.0 and 0.0 < c_bc < 250.0):
            return (b0_cur, f0_cur, float(alpha_star), float(beta_star), c_ab, c_ac, c_bc, True)
        else:
            return (b0_cur, f0_cur, float(alpha_star), float(beta_star), np.nan, np.nan, np.nan, False)

    except Exception:
        return (b0_cur, f0_cur, np.nan, np.nan, np.nan, np.nan, np.nan, False)


def compute_triple_trace_surfaces(
    n_b0=8,
    n_f0=8,
    b0_range=(0.20, 0.55),
    f0_range=(-0.80, -1.80),
    r_max=60.0,
    tol=1e-6,
    max_workers=None,
    force_recompute=False,
    cache_path=CACHE_FILENAME
):
    """
    Computes or loads the 2D grid simulation for all three surfaces: C_AB, C_AC, C_BC.
    """
    total_pts = n_b0 * n_f0

    if not force_recompute and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            if data["ALPHA"].shape == (n_b0, n_f0):
                print(f"[Cache] Successfully loaded precomputed triple surfaces from: {cache_path}")
                return {
                    "ALPHA": data["ALPHA"],
                    "BETA": data["BETA"],
                    "C_AB": data["C_AB"],
                    "C_AC": data["C_AC"],
                    "C_BC": data["C_BC"],
                    "B0": data["B0"],
                    "F0": data["F0"]
                }
            else:
                print(f"Cache shape {data['ALPHA'].shape} differs from requested ({n_b0}, {n_f0}). Recomputing...")
        except Exception:
            print("Cache read error; recomputing...")

    print("=" * 76)
    print("Computing Triple Trace Surfaces: C_AB, C_AC, C_BC across Asymptotic Cones")
    print("=" * 76)
    print(f"Grid Resolution : {n_b0} x {n_f0} = {total_pts} soliton triples")
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
    C_AB = np.full_like(B0, np.nan)
    C_AC = np.full_like(B0, np.nan)
    C_BC = np.full_like(B0, np.nan)

    t_start = time.time()
    completed = 0
    success_count = 0

    print(f"Starting parallel execution for {total_pts} triples...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_evaluate_single_triple, t): (t[0], t[1]) for t in tasks}

        b0_idx_map = {round(val, 6): idx for idx, val in enumerate(b0_vals)}
        f0_idx_map = {round(val, 6): idx for idx, val in enumerate(f0_vals)}

        for future in as_completed(futures):
            completed += 1
            b0_pt, f0_pt = futures[future]
            try:
                b0_r, f0_r, a_inf, b_inf, c_ab, c_ac, c_bc, ok = future.result()
                i_idx = b0_idx_map[round(b0_pt, 6)]
                j_idx = f0_idx_map[round(f0_pt, 6)]

                if ok and not np.isnan(c_ab) and not np.isnan(c_ac) and not np.isnan(c_bc):
                    ALPHA[i_idx, j_idx] = a_inf
                    BETA[i_idx, j_idx] = b_inf
                    C_AB[i_idx, j_idx] = c_ab
                    C_AC[i_idx, j_idx] = c_ac
                    C_BC[i_idx, j_idx] = c_bc
                    success_count += 1
            except Exception:
                pass

            if completed % 8 == 0 or completed == total_pts:
                elapsed = time.time() - t_start
                rate = completed / max(elapsed, 0.001)
                eta = (total_pts - completed) / max(rate, 0.001)
                print(f"[{completed:3d}/{total_pts}] ({completed/total_pts*100:5.1f}%) | "
                      f"Success: {success_count} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

    t_total = time.time() - t_start
    print("-" * 76)
    print(f"Simulation completed in {t_total:.1f}s. Valid triples: {success_count}/{total_pts} ({success_count/total_pts*100:.1f}%)")

    # Infill any missing points via nearest neighbor interpolation
    nan_mask = np.isnan(ALPHA) | np.isnan(BETA) | np.isnan(C_AB) | np.isnan(C_AC) | np.isnan(C_BC)
    if np.any(nan_mask) and success_count >= 8:
        valid_i, valid_j = np.where(~nan_mask)
        nan_i, nan_j = np.where(nan_mask)
        pts_valid = np.column_stack((valid_i, valid_j))
        pts_nan = np.column_stack((nan_i, nan_j))

        for arr in [ALPHA, BETA, C_AB, C_AC, C_BC]:
            arr[nan_mask] = griddata(pts_valid, arr[~nan_mask], pts_nan, method='nearest')

    # Save to disk cache
    np.savez(
        cache_path,
        ALPHA=ALPHA,
        BETA=BETA,
        C_AB=C_AB,
        C_AC=C_AC,
        C_BC=C_BC,
        B0=B0,
        F0=F0
    )
    print(f"Saved triple trace surface cache to: {cache_path}\n")

    return {
        "ALPHA": ALPHA,
        "BETA": BETA,
        "C_AB": C_AB,
        "C_AC": C_AC,
        "C_BC": C_BC,
        "B0": B0,
        "F0": F0
    }


def plot_triple_trace_surfaces_3d(data, save_path=FIGURE_FILENAME, show_gui=True):
    """
    Renders all 3 surfaces (C_AB, C_AC, C_BC) together on the same 3D coordinate chart.
    """
    ALPHA = np.array(data["ALPHA"], copy=True)
    BETA = np.array(data["BETA"], copy=True)
    C_AB = np.array(data["C_AB"], copy=True)
    C_AC = np.array(data["C_AC"], copy=True)
    C_BC = np.array(data["C_BC"], copy=True)

    valid_mask = ~np.isnan(ALPHA) & ~np.isnan(BETA) & ~np.isnan(C_AB) & ~np.isnan(C_AC) & ~np.isnan(C_BC)
    num_valid = int(np.sum(valid_mask))
    if num_valid == 0:
        print("Error: No valid simulation points to plot.")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    try:
        fig.canvas.manager.set_window_title("Triple Trace Comparison: C_AB, C_AC, C_BC")
    except Exception:
        pass

    # Surface 1: C_AB (D^2 x S^2 Lower vs Upper) in DodgerBlue
    surf_AB = ax.plot_surface(
        ALPHA, BETA, C_AB,
        color='dodgerblue',
        shade=True,
        edgecolor='navy',
        linewidth=0.2,
        alpha=0.65,
        antialiased=True
    )

    # Surface 2: C_AC (D^2 x S^2 Lower vs D^3 x S^1) in MediumOrchid
    surf_AC = ax.plot_surface(
        ALPHA, BETA, C_AC,
        color='mediumorchid',
        shade=True,
        edgecolor='purple',
        linewidth=0.2,
        alpha=0.65,
        antialiased=True
    )

    # Surface 3: C_BC (D^2 x S^2 Upper vs D^3 x S^1) in Coral
    surf_BC = ax.plot_surface(
        ALPHA, BETA, C_BC,
        color='coral',
        shade=True,
        edgecolor='darkred',
        linewidth=0.2,
        alpha=0.65,
        antialiased=True
    )

    # Axis bounds
    z_min = min(float(np.nanmin(C_AB)), float(np.nanmin(C_AC)), float(np.nanmin(C_BC)))
    z_max = max(float(np.nanmax(C_AB)), float(np.nanmax(C_AC)), float(np.nanmax(C_BC)))
    ax.set_xlim(np.nanmin(ALPHA) - 0.02, np.nanmax(ALPHA) + 0.02)
    ax.set_ylim(np.nanmin(BETA) - 0.03, np.nanmax(BETA) + 0.03)
    ax.set_zlim(max(0.0, z_min - 0.5), z_max * 1.10)

    # Axis labels & title
    ax.set_xlabel(r"$a'_\infty$ ($S^1$ Cone Slope)", fontsize=11, labelpad=10)
    ax.set_ylabel(r"$b'_\infty$ ($S^2$ Cone Slope)", fontsize=11, labelpad=10)
    ax.set_zlabel("Finite Trace at Infinity $C$", fontsize=11, labelpad=10)
    ax.set_title(
        f"Triple Trace Comparison: $C_{{AB}}, C_{{AC}}, C_{{BC}}$ over Asymptotic Cones\n"
        f"(Deruelle & Schulze 2023, Theorem 1.1 — $D^2 \\times S^2$ and $D^3 \\times S^1$)",
        fontsize=12,
        fontweight='bold',
        pad=14
    )

    # Legend with proxy patches
    legend_elements = [
        Patch(facecolor='dodgerblue', edgecolor='navy', alpha=0.8,
              label=r'$C_{AB}$: $D^2 \times S^2$ Lower vs Upper'),
        Patch(facecolor='mediumorchid', edgecolor='purple', alpha=0.8,
              label=r'$C_{AC}$: $D^2 \times S^2$ Lower vs $D^3 \times S^1$'),
        Patch(facecolor='coral', edgecolor='darkred', alpha=0.8,
              label=r'$C_{BC}$: $D^2 \times S^2$ Upper vs $D^3 \times S^1$')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

    ax.view_init(elev=24, azim=-45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved triple trace comparison 3D visualization to: {save_path}")

    if '--no-gui' in sys.argv or not show_gui:
        print("Skipping GUI window display (--no-gui flag detected).")
    else:
        print("Launching Matplotlib 3D GUI window (close window to finish)...")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot all 3 surfaces (C_AB, C_AC, C_BC) in 3D over the cone plane"
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=8,
        help="Grid resolution per dimension (default: 8, total 64 triples)"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of the 2D grid instead of reading cache."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without displaying GUI window and save figure directly."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes."
    )

    args = parser.parse_args()

    data = compute_triple_trace_surfaces(
        n_b0=args.grid_size,
        n_f0=args.grid_size,
        b0_range=(0.20, 0.55),
        f0_range=(-0.80, -1.80),
        r_max=60.0,
        tol=1e-6,
        max_workers=args.workers,
        force_recompute=args.recompute
    )

    plot_triple_trace_surfaces_3d(
        data,
        save_path=FIGURE_FILENAME,
        show_gui=(not args.no_gui)
    )


if __name__ == "__main__":
    main()
