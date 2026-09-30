"""
Part 1: The Majorant Bound (Analytic Remainder)

This script verifies the empirical growth rate of the Taylor coefficients
to help guide the rigorous pencil-and-paper proof of the majorant constants K and C.

We expect: |coefficient_k| <= K * C^k
"""

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.taylor_expansion import compute_taylor_series

def evaluate_empirical_majorant(a0_val=1.0, f2_val=-1.0, order=10):
    print(f"--- Empirical Majorant Scan (up to order {order}) ---")
    print(f"Parameters: a0={a0_val}, f2={f2_val}")
    
    solutions, a_coeffs, b_coeffs, f_coeffs = compute_taylor_series(order=order)
    
    # Evaluate coefficients numerically
    eval_a = {}
    eval_b = {}
    eval_f = {}
    
    for k, expr in solutions.items():
        val = abs(float(expr.subs({'a0': a0_val, 'f2': f2_val})))
        # Map variables back
        var_str = str(k)
        if var_str.startswith('a'):
            eval_a[int(var_str[1:])] = val
        elif var_str.startswith('b'):
            eval_b[int(var_str[1:])] = val
        elif var_str.startswith('f'):
            eval_f[int(var_str[1:])] = val

    # Estimate C based on the ratio of consecutive terms
    max_C = 0.0
    for d in [eval_a, eval_b, eval_f]:
        keys = sorted(d.keys())
        for i in range(len(keys)-1):
            k1, k2 = keys[i], keys[i+1]
            if d[k1] > 1e-10:
                ratio = (d[k2] / d[k1])**(1.0 / (k2 - k1))
                if ratio > max_C:
                    max_C = ratio
                    
    print(f"\nEmpirical estimate for geometric growth rate C: {max_C:.4f}")
    
    # Calculate required K to bound all terms
    K_candidates = []
    for d in [eval_a, eval_b, eval_f]:
        for k, val in d.items():
            if max_C > 0:
                K_candidates.append(val / (max_C**k))
                
    if K_candidates:
        max_K = max(K_candidates)
        print(f"Empirical estimate for multiplier K: {max_K:.4f}")
    
    print("\nNote: These are EMPIRICAL bounds. A strict mathematical proof")
    print("must still be written to rigorously certify K and C for all k to infinity.")

if __name__ == "__main__":
    evaluate_empirical_majorant(a0_val=1.0, f2_val=-1.0, order=14)
