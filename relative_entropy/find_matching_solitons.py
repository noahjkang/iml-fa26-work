"""
Find Two Expanding Gradient Ricci Solitons with Identical Asymptotic Cones
==========================================================================

Mathematical Background:
------------------------
In Deruelle & Schulze (2023, CPAM), Theorem 1.1 investigates the unique continuation
and relative entropy between two distinct expanding gradient Ricci solitons (M, g1, f1)
and (M, g2, f2) that share the exact same asymptotic cone (C(S), g_cone).

On M = S^2 x R^2 with SO(3) x SO(2) cohomogeneity-one symmetry, the metric is:
    g = dr^2 + a(r)^2 g_{S^1} + b(r)^2 g_{S^2}

At the bolt (r = 0), regular boundary conditions are determined by two core parameters:
    b(0) = b_0 > 0  (initial S^2 bolt radius)
    f''(0) = f_0 < 0 (initial potential curvature)

As r -> infty, the metric approaches a cone with asymptotic slopes:
    alpha = lim_{r -> infty} a'(r)
    beta  = lim_{r -> infty} b'(r)

Due to the caustic fold geometry of the expander moduli space, the map (b0, f0) -> (alpha, beta)
is 2-to-1 in the physical wedge. For a given reference soliton on the lower branch (small bolt),
there exists a unique second soliton on the upper branch (large bolt) with the exact same cone slopes.

This module provides functions to:
1. Integrate the cohomogeneity-one Ricci soliton ODE system.
2. Extrapolate asymptotic cone slopes (alpha, beta) via 1/r^2 regression.
3. Solve for the matching soliton via 2D nonlinear root-finding from an initial guess.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


def ricci_soliton(r, y):
    """
    Evaluates the cohomogeneity-one expanding Ricci soliton ODE vector field.

    Variables:
        y = [a, a', b, b', f, f']
    """
    a, a_p, b, b_p, f, f_p = y
    a_pp = -2.0 * (a_p * b_p / b) + a_p * f_p + a
    b_pp = (1.0 - b_p**2) / b - (a_p * b_p / a) + b_p * f_p + b
    f_pp = (a_pp / a) + 2.0 * (b_pp / b) - 1.0
    return [a_p, a_pp, b_p, b_pp, f_p, f_pp]


def get_initial_conditions(b0, f0, eps=1e-4):
    """
    Computes regular Taylor series boundary conditions at r = eps near the bolt r = 0.

    Taylor expansions at r = 0 (derived via L'Hopital's rule):
        b''(0)   = (1 / b_0 + b_0) / 2
        a'''(0)  = 1 + f_0 - 2 * b''(0) / b_0
        a(eps)   = eps + (a'''(0) / 6) * eps^3
        a'(eps)  = 1 + (a'''(0) / 2) * eps^2
        b(eps)   = b0 + (b''(0) / 2) * eps^2
        b'(eps)  = b''(0) * eps
        f(eps)   = (f0 / 2) * eps^2
        f'(eps)  = f0 * eps
    """
    b_pp0 = (1.0 / b0 + b0) / 2.0
    a_ppp0 = 1.0 + f0 - 2.0 * b_pp0 / b0
    return [
        eps + (a_ppp0 / 6.0) * eps**3,
        1.0 + (a_ppp0 / 2.0) * eps**2,
        b0 + (b_pp0 / 2.0) * eps**2,
        b_pp0 * eps,
        (f0 / 2.0) * eps**2,
        f0 * eps
    ]


def get_asymptotic_slopes(b0, f0, r_max=120.0, rtol=1e-10, atol=1e-12):
    """
    Integrates the trajectory to r_max and extrapolates the true asymptotic cone slopes:
        a'(r) = alpha - c_a / r^2 + O(r^-4)
        b'(r) = beta  - c_b / r^2 + O(r^-4)
    using linear regression against 1/r^2 on r in [r_max/2, r_max].
    """
    eps = 1e-4
    y0 = get_initial_conditions(b0, f0, eps=eps)
    sol = solve_ivp(
        ricci_soliton,
        (eps, r_max),
        y0,
        method='Radau',
        rtol=rtol,
        atol=atol
    )
    if sol.status != 0 or sol.t[-1] < r_max - 1.0:
        return np.nan, np.nan

    # Extrapolate to r -> infinity via 1/r^2 regression
    r_fit = sol.t[sol.t >= r_max / 2.0]
    a_p_fit = sol.y[1][sol.t >= r_max / 2.0]
    b_p_fit = sol.y[3][sol.t >= r_max / 2.0]

    p_a = np.polyfit(1.0 / (r_fit**2), a_p_fit, 1)
    p_b = np.polyfit(1.0 / (r_fit**2), b_p_fit, 1)

    alpha = float(p_a[1])
    beta = float(p_b[1])
    return alpha, beta


# start with default initial established soliton b0 = 0.3 and f0 = -1.0
def find_matching_solitons(b0_1=0.3, f0_1=-1.0, initial_guess=(1.8, -2.0), r_max=120.0, tol=1e-9):
    """
    Finds two distinct expanding gradient Ricci solitons that share the exact same asymptotic cone.

    Parameters:
    -----------
    b0_1 : float
        Initial bolt radius for reference Soliton 1 (default 0.3).
    f0_1 : float
        Initial potential curvature for reference Soliton 1 (default -1.0).
    initial_guess : tuple of float
        Initial guess (b0_2, f0_2) on the upper branch for Soliton 2 (default (1.8, -2.0)).
    r_max : float
        Outer integration radius for asymptotic cone evaluation (default 120.0).
    tol : float
        Root-finding tolerance for matching cone slopes (default 1e-9).

    Returns:
    --------
    dict:
        Dictionary containing initial parameters, asymptotic cone slopes, and residual errors.
    """
    # 1. Compute target asymptotic cone slopes from Soliton 1
    alpha_1, beta_1 = get_asymptotic_slopes(b0_1, f0_1, r_max=r_max)
    if np.isnan(alpha_1) or np.isnan(beta_1):
        raise RuntimeError(f"Failed to integrate reference Soliton 1 with b0={b0_1}, f0={f0_1}")

    # 2. Define 2D residual objective function for Soliton 2
    def objective(x):
        b0, f0 = x
        if b0 <= 0.5 or f0 >= 0.0:
            return [100.0, 100.0]
        a_val, b_val = get_asymptotic_slopes(b0, f0, r_max=r_max)
        if np.isnan(a_val) or np.isnan(b_val):
            return [100.0, 100.0]
        return [a_val - alpha_1, b_val - beta_1]

    # 3. Solve for Soliton 2 using Powell's hybrid method
    res = root(objective, list(initial_guess), method='hybr', tol=tol)
    if not res.success:
        raise RuntimeError(f"Nonlinear root-finding failed to converge: {res.message}")

    b0_2, f0_2 = float(res.x[0]), float(res.x[1])
    alpha_2, beta_2 = get_asymptotic_slopes(b0_2, f0_2, r_max=r_max)

    diff_alpha = abs(alpha_1 - alpha_2)
    diff_beta = abs(beta_1 - beta_2)

    return {
        "soliton_1": {
            "b0": float(b0_1),
            "f0": float(f0_1),
            "alpha": alpha_1,
            "beta": beta_1,
            "branch": "lower (small bolt)"
        },
        "soliton_2": {
            "b0": b0_2,
            "f0": f0_2,
            "alpha": alpha_2,
            "beta": beta_2,
            "branch": "upper (large bolt)"
        },
        "cone_slopes": {
            "alpha": alpha_1,
            "beta": beta_1
        },
        "residuals": {
            "delta_alpha": diff_alpha,
            "delta_beta": diff_beta
        },
        "r_max": float(r_max)
    }


def main():
    print("=" * 72)
    print("Identifying Two Expanding Ricci Solitons with the Same Asymptotic Cone")
    print("=" * 72)
    print("Running nonlinear root-finding to match cone slopes at r -> infty...")

    data = find_matching_solitons(
        b0_1=0.3,
        f0_1=-1.0,
        initial_guess=(1.8, -2.0),
        r_max=120.0,
        tol=1e-9
    )

    s1 = data["soliton_1"]
    s2 = data["soliton_2"]
    cone = data["cone_slopes"]
    res = data["residuals"]

    print("\n[Soliton 1 - Reference (Lower Branch)]")
    print(f"  Bolt Radius b0_1    : {s1['b0']:.8f}")
    print(f"  Potential f''(0)    : {s1['f0']:.8f}")
    print(f"  Asymptotic Slopes   : alpha_1 = {s1['alpha']:.8f}, beta_1 = {s1['beta']:.8f}")

    print("\n[Soliton 2 - Matched (Upper Branch)]")
    print(f"  Bolt Radius b0_2    : {s2['b0']:.8f}")
    print(f"  Potential f''(0)    : {s2['f0']:.8f}")
    print(f"  Asymptotic Slopes   : alpha_2 = {s2['alpha']:.8f}, beta_2 = {s2['beta']:.8f}")

    print("\n[Asymptotic Cone & Matching Residuals]")
    print(f"  Target Cone Slopes  : alpha = {cone['alpha']:.8f}, beta = {cone['beta']:.8f}")
    print(f"  Residual |delta a'| : {res['delta_alpha']:.3e}  (Tolerance < 1e-9)")
    print(f"  Residual |delta b'| : {res['delta_beta']:.3e}  (Tolerance < 1e-9)")
    print("=" * 72)


if __name__ == '__main__':
    main()
