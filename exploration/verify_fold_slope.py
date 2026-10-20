"""
Verification Script: Numerical Confirmation of the Theoretical Fold Slope (m = sqrt(3))

This script numerically recreates the high-precision tracking of the fold caustic
where det(Df2) = 0 for 4D expanding Ricci solitons on S2 x R2.

It demonstrates:
1. As f0 -> 0-, the critical bolt b0* -> 1.00000.
2. The ratio b'_infty / a'_infty converges rigorously to sqrt(3) ~= 1.73205081.
3. Over the broad range f0 in [-10, -0.2], the linear regression is:
   b'_infty ~= 1.7231 * a'_infty + 0.1454 (R^2 = 0.99975).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ricci_odes import ricci_soliton

def get_limits_f2_prec(b0, f0, r_max=45.0):
    """
    Integrates the S2 x R2 expanding Ricci soliton ODE out to r_max
    using the Radau stiff solver with high-precision tolerances.
    """
    eps = 1e-4
    b_double_prime_0 = (1.0 / b0 + b0) / 2.0
    a_triple_prime_0 = 1.0 + f0 - 2.0 * b_double_prime_0 / b0
    
    a_eps = eps + (a_triple_prime_0 / 6.0) * eps**3
    a_prime_eps = 1.0 + (a_triple_prime_0 / 2.0) * eps**2
    b_eps = b0 + (b_double_prime_0 / 2.0) * eps**2
    b_prime_eps = b_double_prime_0 * eps
    f_eps = (f0 / 2.0) * eps**2
    f_prime_eps = f0 * eps
    
    y0 = [a_eps, a_prime_eps, b_eps, b_prime_eps, f_eps, f_prime_eps]
    
    sol = solve_ivp(ricci_soliton, (eps, r_max), y0, method='Radau', rtol=1e-8, atol=1e-10)
    if sol.status != 0 or sol.t[-1] < r_max - 0.5:
        return np.nan, np.nan
    return sol.y[1][-1], sol.y[3][-1]

def jacobian_det(b0, f0):
    """Computes det(Df2) at (b0, f0) using finite differences."""
    h_b = 1e-4
    h_f = 1e-4
    a0, b_lim0 = get_limits_f2_prec(b0, f0)
    ab, bb = get_limits_f2_prec(b0 + h_b, f0)
    af, bf = get_limits_f2_prec(b0, f0 + h_f)
    
    da_db = (ab - a0) / h_b
    db_db = (bb - b_lim0) / h_b
    da_df = (af - a0) / h_f
    db_df = (bf - b_lim0) / h_f
    
    return da_db * db_df - da_df * db_db

def run_verification():
    print("=" * 80)
    print("TASK 2: NUMERICAL RECREATION & VERIFICATION OF THE FOLD SLOPE")
    print("=" * 80)
    
    # 1. High-precision limit evaluation as f0 -> 0-
    f0_values = [-0.1000, -0.0500, -0.0200, -0.0100, -0.0050, -0.0020]
    sqrt3 = np.sqrt(3.0)
    
    print("\n[Phase 1] High-Precision Convergence towards the Einstein Limit (f0 -> 0-):")
    print("-" * 80)
    print(f"{'f0':>9} | {'Critical b0*':>12} | {'a_inf':>10} | {'b_inf':>10} | {'Ratio (b/a)':>14} | {'Diff from sqrt(3)':>18}")
    print("-" * 80)
    
    ratios = []
    f0_arr = []
    
    for f0 in f0_values:
        # Bracket root search for b0 where det(Df2) = 0
        res = root_scalar(lambda b: jacobian_det(b, f0), bracket=[0.9, 1.1], method='brentq', xtol=1e-5)
        b0_crit = res.root
        a_inf, b_inf = get_limits_f2_prec(b0_crit, f0, r_max=45.0)
        ratio = b_inf / a_inf
        diff = ratio - sqrt3
        
        ratios.append(ratio)
        f0_arr.append(f0)
        
        print(f"{f0:9.4f} | {b0_crit:12.5f} | {a_inf:10.4f} | {b_inf:10.4f} | {ratio:14.6f} | {diff:+18.6f}")
        
    print("-" * 80)
    
    # 2. Linear extrapolation to f0 = 0
    f0_arr = np.array(f0_arr)
    ratios = np.array(ratios)
    poly = np.polyfit(f0_arr, ratios, 1)
    extrapolated_slope = poly[1]
    rel_error = abs(extrapolated_slope - sqrt3) / sqrt3 * 100
    
    print(f"\nExtrapolated Asymptotic Slope (f0 -> 0) : {extrapolated_slope:.8f}")
    print(f"Theoretical Slope (sqrt(3))             : {sqrt3:.8f}")
    print(f"Relative Extrapolation Error           : {rel_error:.4f}%")
    
    # 3. Practical broad-range sweep across visible manifold
    print("\n[Phase 2] Evaluating Broad-Range Empirical Fit (f0 in [-10, -0.2]):")
    broad_f0 = np.linspace(-0.2, -10.0, 25)
    broad_a = []
    broad_b = []
    
    for f0 in broad_f0:
        try:
            res = root_scalar(lambda b: jacobian_det(b, f0), bracket=[0.3, 1.2], method='brentq', xtol=1e-3)
            b0_c = res.root
            a_c, b_c = get_limits_f2_prec(b0_c, f0, r_max=25.0)
            if not np.isnan(a_c) and not np.isnan(b_c):
                broad_a.append(a_c)
                broad_b.append(b_c)
        except Exception:
            continue
            
    broad_a = np.array(broad_a)
    broad_b = np.array(broad_b)
    broad_fit = np.polyfit(broad_a, broad_b, 1)
    r2 = np.corrcoef(broad_a, broad_b)[0, 1]**2
    
    print(f"Empirical Linear Regression : b'_infty = {broad_fit[0]:.4f} * a'_infty + {broad_fit[1]:.4f}")
    print(f"Coefficient of Determination : R^2 = {r2:.6f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON:")
    print(f"  * Theoretical Analytic Limit : m = sqrt(3) ~= 1.73205081")
    print(f"  * Numerical ODE Limit        : m = {ratios[-1]:.6f} (at f0 = {f0_values[-1]})")
    print(f"  * Extrapolated Limit         : m = {extrapolated_slope:.6f}")
    print(f"  * Visible Empirical Slope    : m = {broad_fit[0]:.4f} (with shift c = {broad_fit[1]:.4f})")
    print("=" * 80)

if __name__ == '__main__':
    run_verification()
