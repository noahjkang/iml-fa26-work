import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
import os
import sys

# Add project root to path
# Ensure project root is in sys.path
_cur = os.path.abspath(__file__)
while _cur != os.path.dirname(_cur):
    if os.path.exists(os.path.join(_cur, 'src', 'ricci_odes.py')):
        if _cur not in sys.path:
            sys.path.insert(0, _cur)
        break
    _cur = os.path.dirname(_cur)
from src.ricci_odes import ricci_soliton

def get_limits_f2(b0, f0, r_max=20.0):
    """Integrates f2 ODE to r_max using Radau stiff solver."""
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
    
    sol = solve_ivp(ricci_soliton, (eps, r_max), y0, method='Radau', rtol=1e-5, atol=1e-7)
    if sol.status != 0 or sol.t[-1] < r_max - 0.1:
        return np.nan, np.nan
    return sol.y[1][-1], sol.y[3][-1]

def jacobian_det(b0, f0):
    """Computes det(Df2) at (b0, f0) using finite differences."""
    h_b = 1e-3
    h_f = 1e-3
    a0, b_lim0 = get_limits_f2(b0, f0)
    ab, bb = get_limits_f2(b0 + h_b, f0)
    af, bf = get_limits_f2(b0, f0 + h_f)
    
    da_db = (ab - a0) / h_b
    db_db = (bb - b_lim0) / h_b
    da_df = (af - a0) / h_f
    db_df = (bf - b_lim0) / h_f
    
    return da_db * db_df - da_df * db_db

def main():
    print("Step 1: Generating background grid for f2 manifold...")
    b0_vals = np.linspace(0.1, 10.0, 35)
    f0_vals = np.linspace(-0.2, -10.0, 35)
    B0, F0 = np.meshgrid(b0_vals, f0_vals)
    
    A_inf = np.zeros_like(B0)
    B_inf = np.zeros_like(B0)
    
    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            A_inf[i, j], B_inf[i, j] = get_limits_f2(B0[i, j], F0[i, j])
            
    print("Step 2: Tracking the fold line (det Df2 = 0)...")
    f0_scan = np.linspace(-0.2, -10.0, 40)
    fold_a = []
    fold_b = []
    fold_b0 = []
    fold_f0 = []
    
    for f0 in f0_scan:
        try:
            # Bracket for b0 where det_J crosses 0 (typically between 0.3 and 1.2)
            res = root_scalar(lambda b: jacobian_det(b, f0), bracket=[0.3, 1.2], method='brentq', xtol=1e-3)
            b0_crit = res.root
            a_c, b_c = get_limits_f2(b0_crit, f0)
            if not np.isnan(a_c) and not np.isnan(b_c):
                fold_a.append(a_c)
                fold_b.append(b_c)
                fold_b0.append(b0_crit)
                fold_f0.append(f0)
        except Exception:
            continue
            
    fold_a = np.array(fold_a)
    fold_b = np.array(fold_b)
    fold_b0 = np.array(fold_b0)
    fold_f0 = np.array(fold_f0)
    
    print(f"Found {len(fold_a)} points on the fold caustic!")
    
    # Linear fit of the fold line: b'_infty = m * a'_infty + c
    poly_fold = np.polyfit(fold_a, fold_b, 1)
    slope_fold = poly_fold[0]
    intercept_fold = poly_fold[1]
    
    # Quadratic fit for higher precision curvature
    poly2_fold = np.polyfit(fold_a, fold_b, 2)
    
    corr_fold = np.corrcoef(fold_a, fold_b)[0, 1]**2
    print(f"\n--- FOLD LINE RESULTS ---")
    print(f"Linear Fit Equation : b'_infty = {slope_fold:.4f} * a'_infty + {intercept_fold:.4f}")
    print(f"R-squared           : {corr_fold:.6f}")
    print(f"Quadratic Equation  : b'_infty = {poly2_fold[0]:.4f} * (a'_infty)^2 + {poly2_fold[1]:.4f} * a'_infty + {poly2_fold[2]:.4f}")
    
    # Also check the upper boundary (b0 = 10.0)
    upper_a = A_inf[:, -1]
    upper_b = B_inf[:, -1]
    valid_up = ~np.isnan(upper_a) & ~np.isnan(upper_b)
    poly_upper = np.polyfit(upper_a[valid_up], upper_b[valid_up], 1)
    print(f"\n--- UPPER BOUNDARY (b0 = 10.0) ---")
    print(f"Upper Edge Linear Fit : b'_infty = {poly_upper[0]:.4f} * a'_infty + {poly_upper[1]:.4f}")
    
    # Step 3: Plotting
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    # Background mesh lines
    norm = plt.Normalize(F0.min(), F0.max())
    cmap = cm.viridis
    
    segments = []
    colors_f0 = []
    
    # Constant b0 lines
    for j in range(B0.shape[1]):
        for i in range(B0.shape[0] - 1):
            if np.isnan(A_inf[i, j]) or np.isnan(A_inf[i+1, j]):
                continue
            seg = [(A_inf[i, j], B_inf[i, j]), (A_inf[i+1, j], B_inf[i+1, j])]
            segments.append(seg)
            colors_f0.append(0.5 * (F0[i, j] + F0[i+1, j]))
            
    # Constant f0 lines
    for i in range(B0.shape[0]):
        for j in range(B0.shape[1] - 1):
            if np.isnan(A_inf[i, j]) or np.isnan(A_inf[i, j+1]):
                continue
            seg = [(A_inf[i, j], B_inf[i, j]), (A_inf[i, j+1], B_inf[i, j+1])]
            segments.append(seg)
            colors_f0.append(0.5 * (F0[i, j] + F0[i, j+1]))
            
    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.45, linewidths=0.9)
    lc.set_array(np.array(colors_f0))
    ax.add_collection(lc)
    
    # Plot the fold caustic points and fit line
    ax.scatter(fold_a, fold_b, color='red', s=35, zorder=5, label='Fold Points (det Df2 = 0)')
    
    a_fit = np.linspace(0, max(fold_a)*1.1, 100)
    b_fit_linear = slope_fold * a_fit + intercept_fold
    ax.plot(a_fit, b_fit_linear, 'r--', lw=2.2, 
            label=f'Fold Line Fit: $b\'_\\infty = {slope_fold:.2f} a\'_\\infty {intercept_fold:+.2f}$ ($R^2={corr_fold:.4f}$)')
    
    # Plot sqrt(3) reference
    ax.plot(a_fit, np.sqrt(3)*a_fit, color='orange', linestyle=':', lw=1.8, label='$b\'_\\infty = \\sqrt{3} a\'_\\infty$ (Hypothesis ref)')
    
    # Plot upper edge fit
    ax.plot(upper_a[valid_up], poly_upper[0]*upper_a[valid_up] + poly_upper[1], 
            color='magenta', linestyle='-.', lw=1.8, 
            label=f'Upper Edge ($b_0=10$): $b\'_\\infty = {poly_upper[0]:.2f} a\'_\\infty {poly_upper[1]:+.2f}$')

    ax.set_xlim(-0.1, np.nanmax(A_inf) + 0.2)
    ax.set_ylim(-0.5, np.nanmax(B_inf) + 1.0)
    ax.set_xlabel("a'_infty (S1 Limit)", fontsize=12)
    ax.set_ylabel("b'_infty (S2 Limit)", fontsize=12)
    ax.set_title("2D Projection of f2 Map with Exact Fold Caustic (det Df2 = 0)", fontsize=14, fontweight='bold')
    
    cb = fig.colorbar(lc, ax=ax)
    cb.set_label("f''(0) value", fontsize=11)
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'f2_fold_analysis.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"\nSaved analysis plot to: {output_path}")

if __name__ == '__main__':
    main()
