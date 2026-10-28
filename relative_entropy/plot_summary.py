"""
Generates a concise 2-panel summary figure showing the final trace value and invariants.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
diff = np.load(os.path.join(base_dir, "gauged_difference.npz"))
trace = np.load(os.path.join(base_dir, "trace_at_infinity.npz"))

s = diff['s']
norm_h = diff['norm_h']
tr_h = diff['tr_h']
hat_h_ss = diff['hat_h_ss']
l2_s2 = trace['l2_s2']

s2_norm = (s**2) * norm_h
s2_tr = (s**2) * tr_h

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Scaled Difference Tensor and Scalar Trace Invariant
mask = s >= 5.0
ax1.plot(s[mask], s2_norm[mask], 'b-', lw=2.5, label=r'Riemannian norm: $s^2 |g_1 - g_2|_{g_C} \to 8.42$')
ax1.plot(s[mask], s2_tr[mask], 'r--', lw=2.0, label=r'Scalar trace: $s^2 \mathrm{tr}_{g_C}(g_1 - g_2) \to -8.04$')
ax1.axhline(8.4208, color='blue', linestyle=':', alpha=0.6)
ax1.axhline(-8.0426, color='red', linestyle=':', alpha=0.6)
ax1.set_title("Finite Curvature Difference Invariant", fontsize=12, fontweight='bold')
ax1.set_xlabel("Canonical Cone Coordinate $s$", fontsize=11)
ax1.set_ylabel("Scale-Invariant Value", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=10)

# Panel 2: Link L2 Norm Plateau
ax2.plot(s[mask], l2_s2[mask], 'teal', lw=2.5, label=r'Link $L^2(S)$ norm: $\|s^2 h\|_{L^2(S)} \to 54.30$')
ax2.axhline(54.3037, color='teal', linestyle=':', alpha=0.6)
ax2.set_title(r"Link $L^2(S^1 \times S^2)$ Norm ($Vol = 41.59$)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Canonical Cone Coordinate $s$", fontsize=11)
ax2.set_ylabel(r"Norm $\|s^2 h\|_{L^2(S)}$", fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=10)

plt.tight_layout()
out_path = os.path.join(base_dir, "trace_summary.png")
plt.savefig(out_path, dpi=200)
plt.close()
print(f"Summary figure saved to: {out_path}")
