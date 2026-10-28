r"""
Step 4: Construction of the Gauged Metric Difference Tensor in Canonical Cone Chart

Mathematical Context & Theoretical Foundation:
----------------------------------------------
In Deruelle & Schulze (2023, arXiv:2101.02638v1, CPAM) and Bamler & Chen (2025, CPAM), 
once both Ricci expanders are brought into the common canonical cone coordinate chart s 
such that their potential functions satisfy f_1(s) = f_2(s) = -s^2/4, the geometric 
difference between the two metrics is captured by the difference 2-tensor:
    h := g_1 - g_2

In the shared coordinate chart (s, \theta, S^2), the two metric tensors are:
    g_1(s) = (r_1'(s))^2 ds^2 + a_1(r_1(s))^2 g_{S^1} + b_1(r_1(s))^2 g_{S^2}
    g_2(s) = (r_2'(s))^2 ds^2 + a_2(r_2(s))^2 g_{S^1} + b_2(r_2(s))^2 g_{S^2}

where r_i'(s) = -s / (2 f_i'(r_i(s))) is the canonical coordinate velocity established in Step 3.
The difference tensor h is diagonal in the symmetry coordinates:
    h = h_{ss}(s) ds^2 + h_{\theta\theta}(s) d\theta^2 + h_{S^2}(s) g_{S^2}
with components:
    h_{ss}(s)         = (r_1'(s))^2 - (r_2'(s))^2
    h_{\theta\theta}(s) = a_1(r_1(s))^2 - a_2(r_2(s))^2
    h_{S^2}(s)         = b_1(r_1(s))^2 - b_2(r_2(s))^2

Canonical Cone Metric Normalization:
-------------------------------------
In the canonical coordinate s, the Euclidean cone metric g_C is:
    g_C = \frac{1}{2} ds^2 + \frac{1}{2} s^2 (a'_\infty)^2 d\theta^2 + \frac{1}{2} s^2 (b'_\infty)^2 g_{S^2}

The dimensionless relative perturbations are:
    \hat{h}_{ss}(s) = \frac{h_{ss}(s)}{(g_C)_{ss}}           = 2 \Big( (r_1'(s))^2 - (r_2'(s))^2 \Big)
    \hat{h}_{11}(s) = \frac{h_{\theta\theta}(s)}{(g_C)_{\theta\theta}} = \frac{a_1(r_1(s))^2 - a_2(r_2(s))^2}{\frac{1}{2} s^2 (a'_\infty)^2}
    \hat{h}_{22}(s) = \frac{h_{S^2}(s)}{(g_C)_{S^2}}         = \frac{b_1(r_1(s))^2 - b_2(r_2(s))^2}{\frac{1}{2} s^2 (b'_\infty)^2}

The pointwise Riemannian norm of h with respect to g_C is:
    |h|_{g_C}(s) = \sqrt{ \hat{h}_{ss}(s)^2 + \hat{h}_{11}(s)^2 + 2 \hat{h}_{22}(s)^2 }
and its Riemannian trace is:
    tr_{g_C}(h)(s) = \hat{h}_{ss}(s) + \hat{h}_{11}(s) + 2 \hat{h}_{22}(s)

Asymptotic Scaling:
-------------------
Because both solitons emerge from the exact same asymptotic cone with slopes (a'_\infty, b'_\infty), 
their difference h decays as O(s^{-2}) as s -> \infty. The scaled quantity s^2 |h|_{g_C}(s) 
approaches a finite, non-zero constant reflecting the distinct sub-leading curvature profiles 
of the two solitons.

Pipeline Role:
--------------
- Inputs: Loads `aligned_coordinates.npz` (Step 3) and `matching_solitons.json` (Step 1).
- Computations: Evaluates raw and cone-normalized difference tensor components, computes 
  the Riemannian tensor norm and trace, and tracks asymptotic decay.
- Outputs: Serializes difference tensor data to `relative_entropy/gauged_difference.npz`.
  Generates 4-panel publication-grade diagnostic figure `relative_entropy/step4_gauged_difference.png`.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_difference_tensor_in_canonical_gauge(aligned_data, a_inf, b_inf):
    r"""
    Computes raw and cone-normalized components of the difference tensor in canonical cone chart s.

    Parameters:
    -----------
    aligned_data : dict
        Dictionary of aligned coordinate arrays from Step 3.
    a_inf : float
        Asymptotic cone slope a'_\infty > 0.
    b_inf : float
        Asymptotic cone slope b'_\infty > 0.

    Returns:
    --------
    dict:
        Comprehensive dictionary of raw, normalized, norm, trace, and radial deviation arrays.
    """
    s = aligned_data['s']
    dr1_ds = aligned_data['dr1_ds']
    dr2_ds = aligned_data['dr2_ds']
    a1 = aligned_data['a1']
    b1 = aligned_data['b1']
    a2 = aligned_data['a2']
    b2 = aligned_data['b2']

    # Step 4.1: Raw metric difference components
    h_ss = dr1_ds**2 - dr2_ds**2
    h_thth = a1**2 - a2**2
    h_s2 = b1**2 - b2**2

    # Step 4.2: Cone metric components in coordinate s:
    # (g_C)_{ss} = 1/2, (g_C)_{11} = 1/2 * s^2 * a_inf^2, (g_C)_{22} = 1/2 * s^2 * b_inf^2
    gC_ss = 0.5
    gC_11 = 0.5 * (s**2) * (a_inf**2)
    gC_22 = 0.5 * (s**2) * (b_inf**2)

    # Step 4.3: Dimensionless cone-normalized components
    hat_h_ss = h_ss / gC_ss
    hat_h_11 = h_thth / gC_11
    hat_h_22 = h_s2 / gC_22

    # Step 4.4: Riemannian norm and trace with respect to g_C
    norm_h = np.sqrt(hat_h_ss**2 + hat_h_11**2 + 2.0 * hat_h_22**2)
    tr_h = hat_h_ss + hat_h_11 + 2.0 * hat_h_22

    # Step 4.5: Deruelle-Schulze Proposition 7.6 radial identity: |hat_h_ss - (1/2) tr h|
    radial_trace_dev = np.abs(hat_h_ss - 0.5 * tr_h)

    return {
        's': s,
        'h_ss': h_ss,
        'h_thth': h_thth,
        'h_s2': h_s2,
        'hat_h_ss': hat_h_ss,
        'hat_h_11': hat_h_11,
        'hat_h_22': hat_h_22,
        'norm_h': norm_h,
        'tr_h': tr_h,
        'radial_trace_dev': radial_trace_dev,
        'a_inf': a_inf,
        'b_inf': b_inf
    }


def plot_difference_tensor_diagnostics(diff_data, output_path):
    """
    Generates a 4-panel publication-grade diagnostic plot of the difference tensor.
    """
    s = diff_data['s']
    h_ss = diff_data['h_ss']
    h_thth = diff_data['h_thth']
    h_s2 = diff_data['h_s2']
    hat_h_ss = diff_data['hat_h_ss']
    hat_h_11 = diff_data['hat_h_11']
    hat_h_22 = diff_data['hat_h_22']
    norm_h = diff_data['norm_h']
    tr_h = diff_data['tr_h']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (0, 0): Raw difference components h_ij(s)
    ax = axes[0, 0]
    ax.plot(s, h_ss, 'r-', lw=2, label=r'$h_{ss} = (r_1^\prime)^2 - (r_2^\prime)^2$')
    ax.plot(s, h_thth, 'b--', lw=2, label=r'$h_{\theta\theta} = a_1^2 - a_2^2$')
    ax.plot(s, h_s2, 'g-.', lw=2, label=r'$h_{S^2} = b_1^2 - b_2^2$')
    ax.set_title(r"Raw Metric Difference Tensor Components $h_{ij}(s)$", fontsize=11, fontweight='bold')
    ax.set_xlabel("Canonical cone coordinate $s$")
    ax.set_ylabel("Raw metric difference component")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Panel (0, 1): Normalized components hat_h_ij(s) showing O(s^{-2}) decay
    ax = axes[0, 1]
    ax.plot(s, hat_h_ss, 'r-', lw=2, label=r'$\hat{h}_{ss} = h_{ss} / (g_C)_{ss}$')
    ax.plot(s, hat_h_11, 'b--', lw=2, label=r'$\hat{h}_{11} = h_{\theta\theta} / (g_C)_{\theta\theta}$')
    ax.plot(s, hat_h_22, 'g-.', lw=2, label=r'$\hat{h}_{22} = h_{S^2} / (g_C)_{S^2}$')
    ax.plot(s, tr_h, 'm:', lw=2, label=r'$\mathrm{tr}_{g_C}(h) = \hat{h}_{ss} + \hat{h}_{11} + 2\hat{h}_{22}$')
    ax.set_title(r"Cone-Normalized Components $\hat{h}_{ij}(s)$", fontsize=11, fontweight='bold')
    ax.set_xlabel("Canonical cone coordinate $s$")
    ax.set_ylabel("Dimensionless normalized perturbation")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Panel (1, 0): Log-Log plot showing clean s^{-2} power-law decay
    ax = axes[1, 0]
    mask = s >= 5.0
    ax.loglog(s[mask], norm_h[mask], 'k-', lw=2.5, label=r'$|h|_{g_C}(s)$ (Riemannian norm)')
    # Reference s^-2 line scaled to match at s=20
    idx_ref = np.argmin(np.abs(s - 20.0))
    ref_line = norm_h[idx_ref] * (20.0 / s[mask])**2
    ax.loglog(s[mask], ref_line, 'r--', lw=1.5, label=r'Reference $O(s^{-2})$ power law')
    ax.set_title(r"Power-Law Asymptotic Decay: $|h|_{g_C}(s) \sim O(s^{-2})$", fontsize=11, fontweight='bold')
    ax.set_xlabel("Canonical cone coordinate $s$ (log scale)")
    ax.set_ylabel(r"$|h|_{g_C}(s)$ (log scale)")
    ax.legend(loc='upper right')
    ax.grid(True, which='both', alpha=0.3)

    # Panel (1, 1): Scaled product s^2 |h|_{g_C}(s) -> Constant curvature coefficient
    ax = axes[1, 1]
    s2_norm = (s**2) * norm_h
    ax.plot(s[mask], s2_norm[mask], 'b-', lw=2, label=r'$s^2 |h|_{g_C}(s)$')
    asymp_c = s2_norm[-1]
    ax.axhline(asymp_c, color='red', linestyle='--', label=rf'Asymptotic limit $\approx {asymp_c:.2f}$')
    ax.set_title(r"Curvature Scale Invariant: $s^2 |h|_{g_C}(s) \to \mathrm{const}$", fontsize=11, fontweight='bold')
    ax.set_xlabel("Canonical cone coordinate $s$")
    ax.set_ylabel(r"$s^2 |h|_{g_C}(s)$")
    ax.set_ylim([0, max(s2_norm[mask]) * 1.2])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Diagnostic plot saved to: {output_path}")


def main():
    print("======================================================================")
    print("Step 4: Construction of the Gauged Metric Difference Tensor h = g1 - g2")
    print("======================================================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load Step 3 aligned coordinates
    aligned_path = os.path.join(base_dir, "aligned_coordinates.npz")
    if not os.path.exists(aligned_path):
        raise FileNotFoundError(f"Missing aligned coordinates archive: {aligned_path}. Run step3 first.")
    aligned_raw = np.load(aligned_path)
    aligned_data = {
        's': aligned_raw['s'],
        'dr1_ds': aligned_raw['dr1_ds'],
        'dr2_ds': aligned_raw['dr2_ds'],
        'a1': aligned_raw['a1'],
        'b1': aligned_raw['b1'],
        'a2': aligned_raw['a2'],
        'b2': aligned_raw['b2']
    }
    print("Loaded Step 3 canonical aligned coordinates.")

    # Load Step 1 cone parameters
    match_path = os.path.join(base_dir, "matching_solitons.json")
    if not os.path.exists(match_path):
        raise FileNotFoundError(f"Missing matching solitons metadata: {match_path}. Run step1 first.")
    with open(match_path, 'r') as f:
        match_data = json.load(f)
    a_inf = match_data['soliton_1']['a_prime_inf']
    b_inf = match_data['soliton_1']['b_prime_inf']
    print(f"Loaded cone slopes: a'_inf = {a_inf:.8f}, b'_inf = {b_inf:.8f}")

    # Compute difference tensor
    diff_data = compute_difference_tensor_in_canonical_gauge(aligned_data, a_inf, b_inf)

    # Checkpoint print table
    s = diff_data['s']
    hat_ss = diff_data['hat_h_ss']
    hat_11 = diff_data['hat_h_11']
    hat_22 = diff_data['hat_h_22']
    norm_h = diff_data['norm_h']
    tr_h = diff_data['tr_h']
    s2_norm = (s**2) * norm_h

    print("\nDifference Tensor Component Checkpoints:")
    print("-----------------------------------------------------------------------------------------")
    print("   s   |   hat_h_ss   |   hat_h_11   |   hat_h_22   |   |h|_{gC}   |   tr_{gC}(h) | s^2|h|_{gC}")
    print("-----------------------------------------------------------------------------------------")
    for s_val in [2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]:
        idx = int(np.argmin(np.abs(s - s_val)))
        print(f" {s[idx]:5.1f} |  {hat_ss[idx]:+10.4e} |  {hat_11[idx]:+10.4e} |  {hat_22[idx]:+10.4e} |  {norm_h[idx]:10.4e} |  {tr_h[idx]:+10.4e} |   {s2_norm[idx]:7.4f}")
    print("-----------------------------------------------------------------------------------------")

    print(f"\nAsymptotic Behavior:")
    print(f"  |h|_gC at s=10.0: {norm_h[np.argmin(np.abs(s - 10.0))]:.4e}")
    print(f"  |h|_gC at s=40.0: {norm_h[-1]:.4e}")
    print(f"  Decay ratio (40/10)^(-2) = 1/16 = 0.0625; Actual ratio = {norm_h[-1]/norm_h[np.argmin(np.abs(s - 10.0))]:.4f}")
    print(f"  Curvature scale coefficient s^2 |h|_gC -> {s2_norm[-1]:.4f}")

    # Save to gauged_difference.npz
    npz_out = os.path.join(base_dir, "gauged_difference.npz")
    np.savez_compressed(
        npz_out,
        s=s,
        h_ss=diff_data['h_ss'],
        h_thth=diff_data['h_thth'],
        h_s2=diff_data['h_s2'],
        hat_h_ss=hat_ss,
        hat_h_11=hat_11,
        hat_h_22=hat_22,
        norm_h=norm_h,
        tr_h=tr_h,
        radial_trace_dev=diff_data['radial_trace_dev'],
        a_inf=a_inf,
        b_inf=b_inf
    )
    print(f"\nDifference tensor data saved to: {npz_out}")

    # Save diagnostic visualization
    plot_path = os.path.join(base_dir, "step4_gauged_difference.png")
    plot_difference_tensor_diagnostics(diff_data, plot_path)
    print("======================================================================\n")


if __name__ == '__main__':
    main()
