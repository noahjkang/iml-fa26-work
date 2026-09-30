"""
Part 3: The Asymptotic Trapping Region (Empirical Analysis)

This script analyzes the numerical flow up to r = R to evaluate the behavior
of the Lyapunov ratio variables (X, Y, W) defined in the LaTeX proof framework.
This helps establish the correct boundaries (\delta_X, \delta_Y, \delta_W) 
needed for the final rigorous trapping region.
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ricci_odes import ricci_soliton

def analyze_trapping_region(a0=1.0, f2=-1.0, R=50.0):
    print(f"--- Trapping Region Empirical Scan (up to r={R}) ---")
    
    eps = 1e-4
    
    # 2nd order Taylor init (sufficient for floating point survey)
    a_eps = a0 + (a0 / 6.0) * eps**2
    a_prime_eps = (a0 / 3.0) * eps
    b_eps = eps
    b_prime_eps = 1.0
    f_eps = f2 * eps**2
    f_prime_eps = 2 * f2 * eps
    
    y0 = [a_eps, a_prime_eps, b_eps, b_prime_eps, f_eps, f_prime_eps]
    
    # Solve using a stiff solver since we integrate out far
    sol = solve_ivp(ricci_soliton, (eps, R), y0, method='Radau', dense_output=True)
    
    # Extract end state
    r_end = sol.t[-1]
    a, a_prime, b, b_prime, f, f_prime = sol.y[:, -1]
    
    # Calculate Lyapunov Ratio Variables
    X = (r_end * a_prime) / a
    Y = (r_end * b_prime) / b
    W = f_prime / r_end
    
    print(f"\nFinal state at r = {r_end:.2f}:")
    print(f"a  = {a:.5e}, a' = {a_prime:.5e}")
    print(f"b  = {b:.5e}, b' = {b_prime:.5e}")
    print(f"f' = {f_prime:.5e}")
    
    print("\n--- Lyapunov Ratio Variables ---")
    print(f"X (r*a'/a) = {X:.5f}  (Expected limit: 1.0 for strictly conical)")
    
    # Because there is no log divergence, Y = (r*b')/b should also strictly converge to 1.
    print(f"Y (r*b'/b) = {Y:.5f}  (Expected limit: 1.0 for strictly conical)")
    
    # f' ~ -r + c1, so W = f'/r should limit to -1.0
    print(f"W (f'/r)   = {W:.5f}  (Expected limit: -1.0)")
    
    print("\nTo rigorously prove Part 3, we must construct inequalities around these")
    print("X, Y, W values such that the vector field points inward on the boundaries.")

if __name__ == "__main__":
    analyze_trapping_region(a0=1.0, f2=-1.0, R=100.0)
