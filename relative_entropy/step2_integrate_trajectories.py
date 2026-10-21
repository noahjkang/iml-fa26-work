"""
Step 2: High-Precision Trajectory Integration & Dense Profiling

Mathematical Context:
---------------------
In Step 1, we identified two expanding Ricci solitons (Soliton 1 and Soliton 2) 
originating from distinct initial geometric data (b_0^{(1)}, f_0^{(1)}) and (b_0^{(2)}, f_0^{(2)}),
yet possessing the exact same asymptotic cone (a'_infty, b'_infty).

Before we can compute the Deruelle–Schulze relative trace at infinity:
    tr_infty( r^n e^{r^2/4} (g_1 - g_2) )
we must gauge both metrics so that their soliton vector fields coincide outside a compact set.
This gauge condition requires constructing the reparametrization:
    psi(r) = f_1^{-1}(f_2(r))

To perform this reparametrization accurately in Step 3, we require high-resolution, 
continuous trajectories of all metric components and potential profiles:
    y_i(r) = [ a_i(r), a_i'(r), b_i(r), b_i'(r), f_i(r), f_i'(r) ]
across the entire radial domain r in [1e-4, 25.0].

This script accomplishes:
1. Loads the verified matching parameter set from `matching_solitons.json`.
2. Integrates both nonlinear ODE systems using scipy's implicit Radau solver with 
   strict error tolerances (rtol=1e-8, atol=1e-10) and dense output enabled.
3. Samples the continuous solutions onto a dense 2,500-point radial grid.
4. Exports all trajectory data into a compressed archive (`trajectories.npz`).
5. Generates a multi-panel diagnostic figure visualizing the warp factor evolution, 
   derivative slope convergence, potential function behavior, and raw coordinate decay.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import json
import os
import sys

# Ensure project root is in python path to import core ODE models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ricci_odes import ricci_soliton


def integrate_soliton_trajectory(b0, f0, r_max=25.0):
    """
    Integrates the complete 6D soliton ODE system on S^2 x R^2 and returns a dense solution.

    Governing Differential Equations:
    ---------------------------------
    The system of ordinary differential equations for cohomogeneity-one expanding 
    solitons with metric g = dr^2 + a(r)^2 g_{S^1} + b(r)^2 g_{S^2} is:
        a'' = -2 (a' b' / b) + a' f' + a
        b'' = (1 - (b')^2) / b - (a' b' / a) + b' f' + b
        f'' = (a'' / a) + 2 (b'' / b) - 1

    Singular Orbit Initialization:
    ------------------------------
    The system is singular at r = 0 due to divisions by a and b. We bypass the singularity
    by initializing at eps = 1e-4 via L'Hôpital expansions:
        b''(0)   = (1/b_0 + b_0) / 2
        a'''(0)  = 1 + f_0 - 2 * b''(0) / b_0
        
        a(eps)   = eps + (a'''(0) / 6) * eps^3
        a'(eps)  = 1.0 + (a'''(0) / 2) * eps^2
        b(eps)   = b_0 + (b''(0) / 2) * eps^2
        b'(eps)  = b''(0) * eps
        f(eps)   = (f_0 / 2) * eps^2
        f'(eps)  = f_0 * eps

    Numerical Method:
    -----------------
    The ODE system is stiff (governed by both exponential expander growth and curvature decay).
    We employ the Radau IIA implicit Runge-Kutta method (order 5) with dense_output=True,
    which provides a continuous local polynomial interpolation for arbitrary sampling.

    Parameters:
    -----------
    b0 : float
        Initial radius of the S^2 core at r = 0.
    f0 : float
        Initial potential second derivative at r = 0.
    r_max : float
        Upper integration boundary (default 25.0).

    Returns:
    --------
    scipy.integrate.OdeResult:
        The integration result object containing the continuous interpolator `.sol(r)`.
    """
    eps = 1e-4
    
    # Evaluate Taylor initial conditions at offset eps
    b_double_prime_0 = (1.0 / b0 + b0) / 2.0
    a_triple_prime_0 = 1.0 + f0 - 2.0 * b_double_prime_0 / b0
    
    a_eps = eps + (a_triple_prime_0 / 6.0) * eps**3
    a_prime_eps = 1.0 + (a_triple_prime_0 / 2.0) * eps**2
    b_eps = b0 + (b_double_prime_0 / 2.0) * eps**2
    b_prime_eps = b_double_prime_0 * eps
    f_eps = (f0 / 2.0) * eps**2
    f_prime_eps = f0 * eps
    
    y0 = [a_eps, a_prime_eps, b_eps, b_prime_eps, f_eps, f_prime_eps]
    
    sol = solve_ivp(
        ricci_soliton,
        (eps, r_max),
        y0,
        method='Radau',
        rtol=1e-8,
        atol=1e-10,
        dense_output=True
    )
    
    if sol.status != 0:
        raise RuntimeError(f"Trajectory integration failed with status code {sol.status}")
        
    return sol


def main():
    """
    Main driver for Step 2:
    - Loads the verified matching parameter set from `matching_solitons.json`.
    - Solves both trajectories continuously out to r_max = 25.
    - Evaluates and exports dense arrays to `trajectories.npz`.
    - Creates a 4-subplot diagnostic visualization (`step2_trajectories_comparison.png`).
    """
    print("======================================================================")
    print("Step 2: High-Precision Trajectory Integration & Dense Profiling")
    print("======================================================================")
    
    # -------------------------------------------------------------------------
    # Step 2.1: Load verified parameters from Step 1
    # -------------------------------------------------------------------------
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matching_solitons.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing matching parameters file: {json_path}. Run step1 first.")
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    b0_1 = data["soliton_1"]["b0"]
    f0_1 = data["soliton_1"]["f0"]
    b0_2 = data["soliton_2"]["b0"]
    f0_2 = data["soliton_2"]["f0"]
    r_max = data["integration_r_max"]
    a_target = data["cone_target"]["a_prime_infinity"]
    b_target = data["cone_target"]["b_prime_infinity"]
    
    print(f"Loaded Soliton 1: b0_1 = {b0_1:.6f}, f0_1 = {f0_1:.6f}")
    print(f"Loaded Soliton 2: b0_2 = {b0_2:.6f}, f0_2 = {f0_2:.6f}")
    print(f"Target Cone     : a'_inf = {a_target:.8f}, b'_inf = {b_target:.8f}")
    print(f"Radial Domain   : r in [1e-4, {r_max:.1f}]")
    
    # -------------------------------------------------------------------------
    # Step 2.2: Perform high-precision continuous ODE integration
    # -------------------------------------------------------------------------
    print("\nIntegrating Soliton 1 with implicit Radau solver...")
    sol1 = integrate_soliton_trajectory(b0_1, f0_1, r_max=r_max)
    
    print("Integrating Soliton 2 with implicit Radau solver...")
    sol2 = integrate_soliton_trajectory(b0_2, f0_2, r_max=r_max)
    
    # -------------------------------------------------------------------------
    # Step 2.3: Sample continuous solutions onto a dense evaluation grid
    # -------------------------------------------------------------------------
    num_samples = 2500
    r_grid = np.linspace(1e-4, r_max, num_samples)
    
    # Evaluate 6D state vectors: [a, a', b, b', f, f']
    y1_grid = sol1.sol(r_grid)
    y2_grid = sol2.sol(r_grid)
    
    a1, a1_p, b1, b1_p, f1, f1_p = y1_grid
    a2, a2_p, b2, b2_p, f2, f2_p = y2_grid
    
    # -------------------------------------------------------------------------
    # Step 2.4: Export trajectory data to compressed archive
    # -------------------------------------------------------------------------
    npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories.npz")
    np.savez_compressed(
        npz_path,
        r=r_grid,
        a1=a1, a1_p=a1_p, b1=b1, b1_p=b1_p, f1=f1, f1_p=f1_p,
        a2=a2, a2_p=a2_p, b2=b2, b2_p=b2_p, f2=f2, f2_p=f2_p,
        b0_1=b0_1, f0_1=f0_1, b0_2=b0_2, f0_2=f0_2,
        a_target=a_target,
        b_target=b_target
    )
    print(f"\nDense trajectory data successfully saved to: {npz_path}")
    
    # -------------------------------------------------------------------------
    # Step 2.5: Generate 4-panel diagnostic figure
    # -------------------------------------------------------------------------
    print("Generating trajectory diagnostic visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot (0, 0): Metric warp factors a(r) and b(r)
    ax = axes[0, 0]
    ax.plot(r_grid, a1, 'b-', label=rf'$a_1(r)$ ($b_0={b0_1:.2f}$)')
    ax.plot(r_grid, a2, 'b--', label=rf'$a_2(r)$ ($b_0={b0_2:.2f}$)')
    ax.plot(r_grid, b1, 'g-', label=rf'$b_1(r)$ ($b_0={b0_1:.2f}$)')
    ax.plot(r_grid, b2, 'g--', label=rf'$b_2(r)$ ($b_0={b0_2:.2f}$)')
    ax.set_title(r"Metric Warp Factors $a(r)$ and $b(r)$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Radial coordinate $r$")
    ax.set_ylabel("Warp factor size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot (0, 1): Slopes a'(r) and b'(r) converging to the common cone
    ax = axes[0, 1]
    ax.plot(r_grid, a1_p, 'b-', lw=1.8, label=r"$a'_1(r)$")
    ax.plot(r_grid, a2_p, 'b--', lw=1.8, label=r"$a'_2(r)$")
    ax.plot(r_grid, b1_p, 'g-', lw=1.8, label=r"$b'_1(r)$")
    ax.plot(r_grid, b2_p, 'g--', lw=1.8, label=r"$b'_2(r)$")
    ax.axhline(a_target, color='blue', linestyle=':', label=rf"$a'_\infty \approx {a_target:.4f}$")
    ax.axhline(b_target, color='green', linestyle=':', label=rf"$b'_\infty \approx {b_target:.4f}$")
    ax.set_title(r"Derivative Slopes $a'(r)$ and $b'(r)$ Converging to Common Cone", fontsize=12, fontweight='bold')
    ax.set_xlabel("Radial coordinate $r$")
    ax.set_ylabel("Derivative value")
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    
    # Subplot (1, 0): Potential functions f(r) and f'(r)
    ax = axes[1, 0]
    ax.plot(r_grid, f1, 'purple', linestyle='-', label=r'$f_1(r)$')
    ax.plot(r_grid, f2, 'purple', linestyle='--', label=r'$f_2(r)$')
    ax.plot(r_grid, f1_p, 'orange', linestyle='-', label=r"$f'_1(r)$")
    ax.plot(r_grid, f2_p, 'orange', linestyle='--', label=r"$f'_2(r)$")
    ax.set_title(r"Soliton Potential Profiles $f(r)$ and $f'(r)$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Radial coordinate $r$")
    ax.set_ylabel("Potential value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot (1, 1): Convergence rate in raw un-gauged coordinates
    ax = axes[1, 1]
    mask = r_grid > 1.0
    ax.semilogy(r_grid[mask], np.abs(a1_p - a2_p)[mask], 'b-', label=r"$|a'_1(r) - a'_2(r)|$ (raw un-gauged)")
    ax.semilogy(r_grid[mask], np.abs(b1_p - b2_p)[mask], 'g-', label=r"$|b'_1(r) - b'_2(r)|$ (raw un-gauged)")
    ax.set_title("Raw Coordinate Convergence (Prior to Vector Field Alignment)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Radial coordinate $r$")
    ax.set_ylabel("Log absolute difference")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step2_trajectories_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Diagnostic plot successfully saved to: {plot_path}")
    print("======================================================================\n")


if __name__ == '__main__':
    main()
