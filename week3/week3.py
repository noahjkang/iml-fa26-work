import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm  # Required for surface color mapping
from mpl_toolkits.mplot3d import Axes3D

# --- Core Soliton ODE System ---
def ricci_soliton(r, y):
    a, a_prime, b, b_prime, f, f_prime = y
    
    # Floor values to prevent division by zero near the origin
    a = max(a, 1e-10)
    b = max(b, 1e-10)

    a_double_prime = -2 * (a_prime * b_prime) / b + a_prime * f_prime + a
    b_double_prime = (1 - b_prime**2) / b - (a_prime * b_prime) / a + b_prime * f_prime + b
    f_double_prime = a_double_prime / a + 2 * (b_double_prime / b) - 1
    
    return [a_prime, a_double_prime, b_prime, b_double_prime, f_prime, f_double_prime]

# --- Simulators for Asymptotic Limits ---
def get_limits_f1(a0, f0, r_max=20.0):
    """Evaluates the f1 map for the S1 x R3 topology."""
    eps = 1e-4
    a_eps = a0 + (a0 / 6.0) * eps**2
    a_prime_eps = (a0 / 3.0) * eps
    b_eps = eps
    b_prime_eps = 1.0
    f_eps = (f0 / 2.0) * eps**2
    f_prime_eps = f0 * eps
    
    y0 = [a_eps, a_prime_eps, b_eps, b_prime_eps, f_eps, f_prime_eps]
    sol = solve_ivp(ricci_soliton, (eps, r_max), y0, method='RK45')
    return sol.y[1][-1], sol.y[3][-1]

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
    sol = solve_ivp(ricci_soliton, (eps, r_max), y0, method='RK45')
    return sol.y[1][-1], sol.y[3][-1]

# --- Goal 1: 3D Manifold Plot for f2 ---
# Increased resolution slightly (15 -> 25) for a smoother surface
b0_vals = np.linspace(0.5, 3.0, 25)
f0_vals = np.linspace(-0.5, -4.0, 25)
B0, F0 = np.meshgrid(b0_vals, f0_vals)

A_inf_f2 = np.zeros_like(B0)
B_inf_f2 = np.zeros_like(B0)

print("Simulating f2 map limit grid...")
for i in range(B0.shape[0]):
    for j in range(B0.shape[1]):
        A_inf_f2[i,j], B_inf_f2[i,j] = get_limits_f2(B0[i,j], F0[i,j])

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')

# Map the F0 parameter to colors for the surface
norm = plt.Normalize(F0.min(), F0.max())
colors = cm.viridis(norm(F0))

# Plot the continuous manifold
surf = ax1.plot_surface(A_inf_f2, B_inf_f2, B0, facecolors=colors, shade=True, edgecolor='k', linewidth=0.2)

ax1.set_xlabel("a'_infty (S1 Limit)")
ax1.set_ylabel("b'_infty (S2 Limit)")
ax1.set_zlabel("b0 (Initial S2 Size)")
ax1.set_title("Goal 1: f2 Map Asymptotic Manifold")

# Add the colorbar using a ScalarMappable
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax1, label="f''(0) value")

# --- Goal 2: Graphing the a'_infty / b'_infty Ratio ---
f0_line = np.linspace(-0.5, -5.0, 30)
ratio_f1 = []
ratio_f2 = []

print("Simulating limit ratios...")
for f in f0_line:
    a_f1, b_f1 = get_limits_f1(1.0, f)
    a_f2, b_f2 = get_limits_f2(1.0, f)
    ratio_f1.append(a_f1 / b_f1)
    ratio_f2.append(a_f2 / b_f2)

ax2 = fig.add_subplot(132)
ax2.plot(f0_line, ratio_f1, label="Map f1 (S1 x R3, a0=1)", color='blue', lw=2)
ax2.set_xlabel("f''(0) (Soliton Potential Initial Condition)")
ax2.set_ylabel("Ratio (a'_infty / b'_infty)")
ax2.set_title("Goal 2: Map f1 Limit Ratio")
ax2.legend()
ax2.grid(True)

ax4 = fig.add_subplot(133)
ax4.plot(f0_line, ratio_f2, label="Map f2 (S2 x R2, b0=1)", color='orange', lw=2)
ax4.set_xlabel("f''(0) (Soliton Potential Initial Condition)")
ax4.set_ylabel("Ratio (a'_infty / b'_infty)")
ax4.set_title("Goal 2: Map f2 Limit Ratio")
ax4.legend()
ax4.grid(True)

# --- Goal 3: 2D Scatter Plot to Find the Missing Region of f2 ---
fig2 = plt.figure(figsize=(8, 6))
ax3 = fig2.add_subplot(111)

# Flatten the arrays to plot them as a scatter
a_flat = A_inf_f2.flatten()
b_flat = B_inf_f2.flatten()
f0_flat = F0.flatten()

# Scatter plot of the image of f2 in the (a'_infty, b'_infty) plane
sc = ax3.scatter(a_flat, b_flat, c=f0_flat, cmap='viridis', alpha=0.7)
x_vals = np.array([0, np.max(a_flat)])
ax3.plot(x_vals, np.sqrt(3)*x_vals, color='red', linestyle='--', label='b\'_infty = sqrt(3) a\'_infty bound')
ax3.set_xlabel("a'_infty")
ax3.set_ylabel("b'_infty")
ax3.set_title("Image of f2 in the Limit Plane")
ax3.legend()
fig2.colorbar(sc, label="f''(0) value")

plt.tight_layout()
plt.savefig('week3_plots.png')
fig2.savefig('week3_image_f2.png')

# Display the interactive plot windows
plt.show()


