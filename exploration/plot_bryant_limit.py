import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def bryant_ode(r, y):
    h, hp = y
    # From Eq (2.4): 2r^2 h(r) h''(r) = h(r)(h(r) - 1) + r h'(r)(r h'(r) - r - 1)
    # => h'' = [h(h-1) + r*hp*(r*hp - r - 1)] / (2 * r^2 * h)
    hpp = (h * (h - 1) + r * hp * (r * hp - r - 1)) / (2 * r**2 * h)
    return [hp, hpp]

def compute_limit(mu_1, eps=1e-4, R_max=500.0):
    """
    Integrates the Bryant Soliton ODE from r=eps to r=R_max.
    Uses a Taylor expansion at r=0 to jump over the r=0 singularity.
    """
    # Taylor expansion coefficients for h(r) = 1 + mu_1*r + c_2*r^2
    c_2 = (2 * mu_1**2 - mu_1) / 5.0
    
    h_eps = 1.0 + mu_1 * eps + c_2 * eps**2
    hp_eps = mu_1 + 2 * c_2 * eps
    
    # Event to stop integration if h goes negative or blows up 
    # (limit undefined / outside valid domain)
    def stop_event(r, y):
        if y[0] <= 1e-3: return 0  # hit zero
        if y[0] >= 1e6: return 0   # exploded
        return 1
    stop_event.terminal = True
    
    # Use Radau or BDF because the ODE can become stiff as r gets large
    sol = solve_ivp(
        bryant_ode, 
        [eps, R_max], 
        [h_eps, hp_eps], 
        method='Radau',
        events=stop_event,
        rtol=1e-8, 
        atol=1e-8
    )
    
    h_final = sol.y[0, -1]
    
    # Return sqrt(h) if it successfully integrated to the end or stabilized
    if h_final > 0 and sol.t[-1] > R_max * 0.9:
        return np.sqrt(h_final)
    else:
        return np.nan

def main():
    print("Sweeping mu_1 values for Bryant Soliton asymptotic limit...")
    
    # The limit is theoretically finite only for mu_1 < 1/n.
    # We'll scan a range from -1.5 up to 0.5 to see this threshold in action.
    mu_1_vals = np.linspace(-1.5, 0.5, 100)
    limits = []
    
    for mu_1 in mu_1_vals:
        limit = compute_limit(mu_1)
        limits.append(limit)
        
    limits = np.array(limits)
    
    plt.figure(figsize=(8, 5))
    plt.plot(mu_1_vals, limits, 'b-', linewidth=2, label=r'$\lim_{r\to\infty} \sqrt{h(r)}$')
    
    # Draw reference axes
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Theorem bounds (examples for different dimensions n)
    plt.axvline(1/3, color='red', linewidth=1, linestyle='--', label=r'$\mu_1 = 1/3$ (Threshold for n=3)')
    plt.axvline(1/4, color='orange', linewidth=1, linestyle='--', label=r'$\mu_1 = 1/4$ (Threshold for n=4)')
    
    plt.title(r"Asymptotic Cone of Bryant Soliton: $\lim_{r\to\infty} \sqrt{h(r)}$ vs $\mu_1$")
    plt.xlabel(r"Initial Slope $\mu_1 = h'(0)$")
    plt.ylabel(r"Asymptotic Limit: $\sqrt{h(\infty)}$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Display the plot in a GUI window instead of saving to a PNG
    plt.show()

if __name__ == "__main__":
    main()
