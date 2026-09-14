import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ricci_odes(r, y):
    a, a_prime, b, b_prime, f, f_prime = y
    
    # CRITICAL CODE REQUIREMENTS: Floor guards to protect Radau solver from division-by-zero
    a = max(a, 1e-10)
    b = max(b, 1e-10)
    
    a_prime_prime = -2 * a_prime * b_prime / b + a_prime * f_prime + a
    b_prime_prime = (1 - b_prime**2) / b - a_prime * b_prime / a + b_prime * f_prime + b
    f_prime_prime = a_prime_prime / a + 2 * b_prime_prime / b - 1
    
    return [a_prime, a_prime_prime, b_prime, b_prime_prime, f_prime, f_prime_prime]

def initial_conditions(a0, f0, epsilon=1e-4):
    a = a0 + (a0 / 6) * epsilon**2
    a_prime = (a0 / 3) * epsilon
    b = epsilon
    b_prime = 1
    f = (f0 / 2) * epsilon**2
    f_prime = f0 * epsilon
    return [a, a_prime, b, b_prime, f, f_prime]

def task1_trivial_scaling():
    print("--- Task 1: Trivial Scaling Proof ---")
    epsilon = 1e-4
    r_span = (epsilon, 5.0)
    
    a0_1 = 1.0
    f0 = -1.0
    y0_1 = initial_conditions(a0_1, f0, epsilon)
    sol_1 = solve_ivp(ricci_odes, r_span, y0_1, method='Radau', dense_output=True, rtol=1e-10, atol=1e-10)
    
    c = 2.5
    y0_2 = initial_conditions(c * a0_1, f0, epsilon)
    sol_2 = solve_ivp(ricci_odes, r_span, y0_2, method='Radau', dense_output=True, rtol=1e-10, atol=1e-10)
    
    r_eval = np.linspace(epsilon, 5.0, 100)
    y1 = sol_1.sol(r_eval)
    y2 = sol_2.sol(r_eval)
    
    a_diff = np.max(np.abs(y2[0] - c * y1[0]))
    b_diff = np.max(np.abs(y2[2] - y1[2]))
    f_diff = np.max(np.abs(y2[4] - y1[4]))
    
    print(f"Max difference for a(r) scaled by {c}: {a_diff:.2e}")
    print(f"Max difference for b(r) unchanged: {b_diff:.2e}")
    print(f"Max difference for f(r) unchanged: {f_diff:.2e}")
    if a_diff < 1e-6 and b_diff < 1e-6 and f_diff < 1e-6:
        print("Trivial scaling property proven numerically!")
    else:
        print("Trivial scaling property verification failed.")

def task2_limit_mapping():
    print("\n--- Task 2: Limit Mapping (Expander Degree) ---")
    epsilon = 1e-4
    r_span = (epsilon, 20.0)
    a0 = 1.0
    
    f0_vals = np.linspace(-0.1, -5.0, 50)
    
    a_prime_infty = []
    b_prime_infty = []
    
    max_a_double_prime = 0.0
    max_b_double_prime = 0.0
    
    f0_plot = []
    for f0 in f0_vals:
        y0 = initial_conditions(a0, f0, epsilon)
        sol = solve_ivp(ricci_odes, r_span, y0, method='Radau')
        
        if sol.status == 0:
            y_end = sol.y[:, -1]
            a_prime_infty.append(y_end[1])
            b_prime_infty.append(y_end[3])
            f0_plot.append(f0)
            
            derivatives = ricci_odes(20.0, y_end)
            a_prime_prime_end = abs(derivatives[1])
            b_prime_prime_end = abs(derivatives[3])
            
            max_a_double_prime = max(max_a_double_prime, a_prime_prime_end)
            max_b_double_prime = max(max_b_double_prime, b_prime_prime_end)
        
    print(f"Maximum absolute value of a''(20): {max_a_double_prime:.2e}")
    print(f"Maximum absolute value of b''(20): {max_b_double_prime:.2e}")
    
    fig = plt.figure(figsize=(14, 6))
    
    # Plot A: 3D scatter plot (x=a'_infty, y=f_0, z=b'_infty) colored by f_0
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(a_prime_infty, f0_plot, b_prime_infty, c=f0_plot, cmap='viridis')
    ax1.set_xlabel(r"$\lim_{r \to \infty} a'(r)$")
    ax1.set_ylabel(r"$f_0$")
    ax1.set_zlabel(r"$\lim_{r \to \infty} b'(r)$")
    ax1.set_title("Plot A: 3D Limit Mapping")
    fig.colorbar(scatter1, ax=ax1, label=r"$f_0$")
    
    # Plot B: 2D parametric scatter plot (x=a'_infty, y=b'_infty) colored by f_0
    ax2 = fig.add_subplot(122)
    scatter2 = ax2.scatter(a_prime_infty, b_prime_infty, c=f0_plot, cmap='viridis')
    ax2.set_xlabel(r"$\lim_{r \to \infty} a'(r)$")
    ax2.set_ylabel(r"$\lim_{r \to \infty} b'(r)$")
    ax2.set_title("Plot B: 2D Parametric Mapping")
    fig.colorbar(scatter2, ax=ax2, label=r"$f_0$")
    
    plt.tight_layout()
    plt.savefig('limit_mapping.png')
    print("Saved plot to limit_mapping.png")

if __name__ == "__main__":
    task1_trivial_scaling()
    task2_limit_mapping()
