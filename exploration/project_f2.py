import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ricci_odes import ricci_soliton

def get_limits_f2(b0, f0, r_max=20.0):
    """Evaluates the f2 map for the S2 x R2 topology."""
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
    
    def event_a(r, y): return y[0] - 1e-5
    event_a.terminal = True
    def event_b(r, y): return y[2] - 1e-5
    event_b.terminal = True
    
    sol = solve_ivp(ricci_soliton, (eps, r_max), y0, method='RK45', events=[event_a, event_b])
    
    if sol.status != 0 or sol.t[-1] < r_max - 0.1:
        return np.nan, np.nan
    return sol.y[1][-1], sol.y[3][-1]

b0_vals = np.linspace(0.1, 10.0, 35)
f0_vals = np.linspace(-0.1, -10.0, 35)
B0, F0 = np.meshgrid(b0_vals, f0_vals)

A_inf_f2 = np.zeros_like(B0)
B_inf_f2 = np.zeros_like(B0)

print("Simulating f2 map limit grid...")
for i in range(B0.shape[0]):
    for j in range(B0.shape[1]):
        A_inf_f2[i,j], B_inf_f2[i,j] = get_limits_f2(B0[i,j], F0[i,j])

from matplotlib.collections import LineCollection

fig, ax = plt.subplots(figsize=(10, 8))

# Map F0 to colors
valid_mask = ~np.isnan(F0) & ~np.isnan(A_inf_f2) & ~np.isnan(B_inf_f2)
norm = plt.Normalize(F0[valid_mask].min(), F0[valid_mask].max())
cmap = cm.viridis

segments = []
colors_f0 = []

# Constant b0 lines (varying f0)
for j in range(B0.shape[1]):
    for i in range(B0.shape[0] - 1):
        # Skip if any nan
        if np.isnan(A_inf_f2[i, j]) or np.isnan(A_inf_f2[i+1, j]):
            continue
        seg = [(A_inf_f2[i, j], B_inf_f2[i, j]), (A_inf_f2[i+1, j], B_inf_f2[i+1, j])]
        segments.append(seg)
        colors_f0.append(0.5 * (F0[i, j] + F0[i+1, j]))

# Constant f0 lines (varying b0)
for i in range(B0.shape[0]):
    for j in range(B0.shape[1] - 1):
        if np.isnan(A_inf_f2[i, j]) or np.isnan(A_inf_f2[i, j+1]):
            continue
        seg = [(A_inf_f2[i, j], B_inf_f2[i, j]), (A_inf_f2[i, j+1], B_inf_f2[i, j+1])]
        segments.append(seg)
        colors_f0.append(0.5 * (F0[i, j] + F0[i, j+1]))

lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.7, linewidths=1.0)
lc.set_array(np.array(colors_f0))
ax.add_collection(lc)

if len(segments) > 0:
    ax.set_xlim(np.nanmin(A_inf_f2) - 0.1, np.nanmax(A_inf_f2) + 0.1)
    ax.set_ylim(np.nanmin(B_inf_f2) - 0.1, np.nanmax(B_inf_f2) + 0.1)

ax.set_xlabel("a'_infty (S1 Limit)")
ax.set_ylabel("b'_infty (S2 Limit)")
ax.set_title("2D Projection of f2 Map Manifold (Colored by f''(0))")
fig.colorbar(lc, ax=ax, label="f''(0) value")

ax.grid(True)
plt.show()

