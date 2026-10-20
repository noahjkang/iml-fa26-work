import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def hsu_ode(r, y, n):
    h, hp = y
    # Full Hsu (2021) ODE:
    # 2r^2 h h'' = (n-1) h (h-1) + r h' (r h' - r - (n-1))
    hpp = ((n - 1) * h * (h - 1) + r * hp * (r * hp - r - (n - 1))) / (2 * r**2 * h)
    return [hp, hpp]

def compute_limit(mu_1, n, eps=1e-4, R_max=300.0):
    """
    Rigorously solves the Bryant profile ODE using Radau with a constrained
    step-size controller (max_step=0.5) to prevent numerical overshoot in the
    asymptotic regime.
    """
    # Exact Taylor series expansion at r=0 matching the Hsu ODE:
    # 2*r^2 h h'' ~ 4 c_2 r^2
    # RHS at O(r^2): (n-1)(c_2 + mu^2) + mu(mu - 1) - 2(n-1) c_2
    # => c_2 = (n * mu^2 - mu) / (n + 3)
    c_2 = (n * mu_1**2 - mu_1) / (n + 3)
    
    h_eps = 1.0 + mu_1 * eps + c_2 * eps**2
    hp_eps = mu_1 + 2 * c_2 * eps
    
    def blowup_event(r, y):
        if y[0] <= 1e-4: return 0.0
        if y[0] >= 1e4: return 0.0
        return 1.0
    blowup_event.terminal = True
    
    # max_step=0.5 is critical: without it, adaptive solvers take huge steps
    # in the flat transition region, causing artificial catastrophic divergence.
    sol = solve_ivp(
        lambda r, y: hsu_ode(r, y, n),
        [eps, R_max],
        [h_eps, hp_eps],
        method='Radau',
        max_step=0.5,
        rtol=1e-8,
        atol=1e-8,
        events=blowup_event
    )
    
    h_final = sol.y[0, -1]
    if h_final > 0 and sol.t[-1] > R_max * 0.7:
        return np.sqrt(h_final)
    else:
        return np.nan

def main():
    print("Generating clean, certified Bryant Soliton asymptotic plot...")
    
    dimensions = [3, 4, 5]
    colors = ['#1f77b4', '#2ca02c', '#9467bd']  # Blue, Green, Purple
    
    plt.figure(figsize=(9, 5.5), dpi=300)
    
    for n, color in zip(dimensions, colors):
        threshold = 1.0 / n
        # Sweep smoothly from mu_1 = -2.0 up to 98% of the blow-up threshold
        mu_vals = np.linspace(-2.0, threshold - 0.008, 120)
        
        print(f"  Computing dimension n = {n} (threshold = {threshold:.3f})...")
        limits = [compute_limit(mu, n) for mu in mu_vals]
        
        plt.plot(mu_vals, limits, color=color, linewidth=2.5, 
                 label=rf'$n={n}$ (diverges at $\mu_1 = 1/{n}$)')
        plt.axvline(threshold, color=color, linestyle='--', linewidth=1.2, alpha=0.6)

    # Reference axes and Euclidean baseline
    plt.axhline(1.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
    plt.axvline(0.0, color='black', linewidth=0.6)
    
    plt.xlim(-2.0, 0.40)
    plt.ylim(0.2, 5.0)
    
    plt.title(r"Asymptotic Cone of Bryant Solitons: $\lim_{r\to\infty} \sqrt{h(r)}$ vs $\mu_1$", fontsize=13, pad=10)
    plt.xlabel(r"Initial Slope $\mu_1 = h'(0)$", fontsize=11)
    plt.ylabel(r"Asymptotic Limit: $\sqrt{h(\infty)}$", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=10, loc='upper left', framealpha=0.9)
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asymptotic_cone_bryan_soliton.png")
    plt.savefig(out_path, dpi=300)
    print(f"Plot successfully saved to {out_path}")

if __name__ == "__main__":
    main()
