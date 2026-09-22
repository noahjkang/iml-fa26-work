import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def ricci_soliton(r, y):
    a, a_prime, b, b_prime, f, f_prime = y
    a = max(a, 1e-10)
    b = max(b, 1e-10)
    a_double_prime = -2 * (a_prime * b_prime) / b + a_prime * f_prime + a
    b_double_prime = (1 - b_prime**2) / b - (a_prime * b_prime) / a + b_prime * f_prime + b
    f_double_prime = a_double_prime / a + 2 * (b_double_prime / b) - 1
    return [a_prime, a_double_prime, b_prime, b_double_prime, f_prime, f_double_prime]

def get_limits_f2(b0, f0, r_max=30.0):
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
    # Use max_step to prevent solver from taking too large steps and missing features
    sol = solve_ivp(ricci_soliton, (eps, r_max), y0, method='RK45', max_step=0.5)
    return sol.y[1][-1], sol.y[3][-1]

print("Generating random samples for f2...")
# Generate random samples to quickly cover a large space
N_samples = 3000
np.random.seed(42)
# Sample logarithmically to cover different scales
b0_samples = 10**np.random.uniform(-1, 1, N_samples) # 0.1 to 10
f0_samples = -10**np.random.uniform(-1, 1, N_samples) # -10 to -0.1

a_inf = []
b_inf = []

for i in range(N_samples):
    try:
        a_val, b_val = get_limits_f2(b0_samples[i], f0_samples[i])
        a_inf.append(a_val)
        b_inf.append(b_val)
    except:
        pass
    if (i+1) % 300 == 0:
        print(f"Processed {i+1}/{N_samples} samples")

a_inf = np.array(a_inf)
b_inf = np.array(b_inf)

# Plot the scatter
plt.figure(figsize=(10, 8))
plt.scatter(a_inf, b_inf, s=5, alpha=0.5, color='blue', label='Mapped points (a\'_infty, b\'_infty)')

# Attempt to find the boundary (upper envelope)
bins = np.linspace(0, max(a_inf), 50)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
max_b_in_bin = []

for i in range(len(bins)-1):
    mask = (a_inf >= bins[i]) & (a_inf < bins[i+1])
    if np.any(mask):
        max_b_in_bin.append(np.max(b_inf[mask]))
    else:
        max_b_in_bin.append(np.nan)

plt.plot(bin_centers, max_b_in_bin, color='red', linewidth=2, label='Empirical Upper Boundary')
plt.plot(bins, np.sqrt(3)*bins, color='green', linestyle='--', label='b\'_infty = sqrt(3) a\'_infty bound')

plt.xlabel("a'_infty")
plt.ylabel("b'_infty")
plt.title("Dense Sampling of f2 Image and Boundary Estimation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('week3_boundary_estimation.png')
print("Saved week3_boundary_estimation.png")
