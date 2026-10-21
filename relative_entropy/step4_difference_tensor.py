r"""
Step 4: Construction of the Gauged Metric Difference Tensor

Mathematical Context:
---------------------
In Deruelle & Schulze (2023, arXiv:2101.02638v1), once the coordinate gauge is fixed 
such that the soliton drift vector fields coincide identically outside a compact set:
    V_1 = V_2 = f_2'(r) \partial_r  for all r >= R_0
the geometric discrepancy between the two solitons is captured by the difference 2-tensor:
    h := \tilde{g}_1 - g_2

In the shared radial coordinate chart (r, \theta, S^2), the two metric tensors are:
    \tilde{g}_1 = (\psi'(r))^2 dr^2 + a_1(\psi(r))^2 g_{S^1} + b_1(\psi(r))^2 g_{S^2}
    g_2         = dr^2 + a_2(r)^2 g_{S^1} + b_2(r)^2 g_{S^2}

where r_1 = \psi(r) is the coordinate reparametrization established in Step 3.
The difference tensor h is diagonal in the symmetry coordinates:
    h = h_{rr}(r) dr^2 + h_{\theta\theta}(r) d\theta^2 + h_{S^2}(r) g_{S^2}
with components:
    h_{rr}(r)         = (\psi'(r))^2 - 1
    h_{\theta\theta}(r) = a_1(\psi(r))^2 - a_2(r)^2
    h_{S^2}(r)         = b_1(\psi(r))^2 - b_2(r)^2

Normalization Relative to Background Cone Metric:
--------------------------------------------------
On an asymptotically conical manifold, tensor norms and convergence are measured 
relative to the Euclidean cone metric g_C:
    g_C = dr^2 + r^2 (a'_\infty)^2 d\theta^2 + r^2 (b'_\infty)^2 g_{S^2}

The dimensionless relative components are:
    \hat{h}_{rr}(r) = \frac{h_{rr}(r)}{(g_C)_{rr}}           = (\psi'(r))^2 - 1
    \hat{h}_{11}(r) = \frac{h_{\theta\theta}(r)}{(g_C)_{\theta\theta}} = \frac{a_1(\psi(r))^2 - a_2(r)^2}{(a'_\infty r)^2}
    \hat{h}_{22}(r) = \frac{h_{S^2}(r)}{(g_C)_{S^2}}         = \frac{b_1(\psi(r))^2 - b_2(r)^2}{(b'_\infty r)^2}

The pointwise Riemannian norm of h with respect to g_C is:
    |h|_{g_C}(r) = \sqrt{ \hat{h}_{rr}(r)^2 + \hat{h}_{11}(r)^2 + 2 \hat{h}_{22}(r)^2 }
and its Riemannian trace is:
    tr_{g_C}(h)(r) = \hat{h}_{rr}(r) + \hat{h}_{11}(r) + 2 \hat{h}_{22}(r)

Pipeline Role:
--------------
- Inputs: Loads `aligned_coordinates.npz` (Step 3) and `matching_solitons.json` (Step 1).
- Computations: Evaluates the raw and normalized difference tensor components, 
  computes the Riemannian tensor norm and trace, and tracks asymptotic decay.
- Outputs: Serializes difference tensor data to `relative_entropy/gauged_difference.npz`.
  Generates 4-panel diagnostic figure `relative_entropy/step4_gauged_difference.png`.

Numerical Methods:
------------------
- Component-wise tensor algebraic evaluations on the 2,600-point uniform radial grid.
- Vectorized norm evaluations preserving Riemannian signature and multiplicities.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Ensure project root is in python path to import core models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_difference_tensor_components(aligned_data, a_inf, b_inf):
    r"""
    Computes raw and cone-normalized components of the gauged metric difference tensor.

    Mathematical Formulation:
    -------------------------
    Given \psi(r), \psi'(r), a_1(\psi(r)), b_1(\psi(r)), a_2(r), b_2(r):
        h_{rr}         = (\psi')^2 - 1
        h_{\theta\theta} = a_1(\psi)^2 - a_2(r)^2
        h_{S^2}         = b_1(\psi)^2 - b_2(r)^2

    Normalized relative to cone metric g_C:
        \hat{h}_{rr} = h_{rr}
        \hat{h}_{11} = h_{\theta\theta} / (a'_\infty r)^2
        \hat{h}_{22} = h_{S^2} / (b'_\infty r)^2

    Tensor norm and trace:
        |h|_{g_C}    = \sqrt{\hat{h}_{rr}^2 + \hat{h}_{11}^2 + 2 \hat{h}_{22}^2}
        tr_{g_C}(h)  = \hat{h}_{rr} + \hat{h}_{11} + 2 \hat{h}_{22}

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
        Comprehensive dictionary of raw, normalized, norm, and trace arrays.
    """
    r = aligned_data['r']
    psi_p = aligned_data['psi_prime']
    a1_g = aligned_data['a1_gauged']
    b1_g = aligned_data['b1_gauged']
    a2 = aligned_data['a2']
    b2 = aligned_data['b2']

    # Step 4.1: Compute raw difference tensor components
    h_rr = psi_p**2 - 1.0
    h_thth = a1_g**2 - a2**2
    h_s2 = b1_g**2 - b2**2

    # Step 4.2: Compute cone-normalized relative perturbations
    hat_h_rr = h_rr
    hat_h_11 = h_thth / (a_inf**2 * r**2)
    hat_h_22 = h_s2 / (b_inf**2 * r**2)

    # Step 4.3: Compute Riemannian tensor norm and trace with respect to g_C
    norm_h = np.sqrt(hat_h_rr**2 + hat_h_11**2 + 2.0 * hat_h_22**2)
    tr_h = hat_h_rr + hat_h_11 + 2.0 * hat_h_22

    # Step 4.4: Compute radial trace alignment quantity (Deruelle-Schulze Proposition 7.6)
    # Checks the quantity |h(d_r) - (tr h / 2) d_r|^2
    radial_trace_dev = np.abs(hat_h_rr - 0.5 * tr_h)

    return {
        'r': r,
        'h_rr': h_rr,
        'h_thth': h_thth,
        'h_s2': h_s2,
        'hat_h_rr': hat_h_rr,
        'hat_h_11': hat_h_11,
        'hat_h_22': hat_h_22,
        'norm_h': norm_h,
        'tr_h': tr_h,
        'radial_trace_dev': radial_trace_dev,
        'a_inf': a_inf,
        'b_inf': b_inf
    }


def main():
    """
    Main driver for Step 4:
    - Loads Step 3 aligned coordinate data (`aligned_coordinates.npz`).
    - Loads Step 1 cone parameters (`matching_solitons.json`).
    - Evaluates all components of the gauged difference tensor h = g_1 - g_2.
    - Exports computed tensor data to `gauged_difference.npz`.
    - Generates 4-panel diagnostic plot `step4_gauged_difference.png`.
    """
    print("======================================================================")
    print("Step 4: Construction of the Gauged Metric Difference Tensor h = g1 - g2")
    print("======================================================================")

    # -------------------------------------------------------------------------
    # Phase 1: Load aligned data and cone metadata
    # -------------------------------------------------------------------------
    aligned_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aligned_coordinates.npz")
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matching_solitons.json")

    if not os.path.exists(aligned_path):
        raise FileNotFoundError(f"Missing aligned coordinates file: {aligned_path}. Run step3 first.")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing metadata file: {json_path}. Run step1 first.")

    aligned_data = np.load(aligned_path)
    with open(json_path, "r") as f:
        meta = json.load(f)

    a_inf = meta["cone_target"]["a_prime_infinity"]
    b_inf = meta["cone_target"]["b_prime_infinity"]

    print(f"Target Asymptotic Cone Slopes: a'_inf = {a_inf:.8f}, b'_inf = {b_inf:.8f}")
    print(f"Evaluating difference tensor across radial grid r in [{aligned_data['r'][0]:.1f}, {aligned_data['r'][-1]:.1f}]...")

    # -------------------------------------------------------------------------
    # Phase 2: Compute difference tensor components
    # -------------------------------------------------------------------------
    res = compute_difference_tensor_components(aligned_data, a_inf, b_inf)

    r_grid = res['r']
    h_rr = res['h_rr']
    h_thth = res['h_thth']
    h_s2 = res['h_s2']
    hat_11 = res['hat_h_11']
    hat_22 = res['hat_h_22']
    norm_h = res['norm_h']
    tr_h = res['tr_h']

    # -------------------------------------------------------------------------
    # Phase 3: Print diagnostic checkpoints
    # -------------------------------------------------------------------------
    print("\nDifference Tensor Component Checkpoints:")
    print("--------------------------------------------------------------------------------------------")
    print("   r   |    h_rr     |   hat_h_11   |   hat_h_22   |    |h|_gC    |   tr_gC(h)   |  log10(|h|)")
    print("--------------------------------------------------------------------------------------------")
    for r_chk in [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 27.0]:
        idx = int(np.argmin(np.abs(r_grid - r_chk)))
        print(f" {r_grid[idx]:5.1f} | {h_rr[idx]:+10.3e} | {hat_11[idx]:+11.4e} | {hat_22[idx]:+11.4e} | {norm_h[idx]:11.4e} | {tr_h[idx]:+11.4e} | {np.log10(norm_h[idx]):8.2f}")
    print("--------------------------------------------------------------------------------------------")

    # -------------------------------------------------------------------------
    # Phase 4: Save difference tensor arrays to compressed archive
    # -------------------------------------------------------------------------
    npz_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gauged_difference.npz")
    np.savez_compressed(
        npz_out,
        r=r_grid,
        h_rr=h_rr,
        h_thth=h_thth,
        h_s2=h_s2,
        hat_h_rr=res['hat_h_rr'],
        hat_h_11=hat_11,
        hat_h_22=hat_22,
        norm_h=norm_h,
        tr_h=tr_h,
        radial_trace_dev=res['radial_trace_dev'],
        a_inf=a_inf,
        b_inf=b_inf
    )
    print(f"\nGauged difference tensor data saved to: {npz_out}")

    # -------------------------------------------------------------------------
    # Phase 5: Generate 4-panel diagnostic visualization
    # -------------------------------------------------------------------------
    print("Generating difference tensor diagnostic visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot (0, 0): Raw difference components h_rr, h_theta_theta, h_S2
    ax = axes[0, 0]
    ax.plot(r_grid, h_rr, 'r-', lw=2, label=r'$h_{rr}(r) = (\psi^\prime(r))^2 - 1$')
    ax.plot(r_grid, h_thth, 'b-', lw=2, label=r'$h_{\theta\theta}(r) = a_1(\psi(r))^2 - a_2(r)^2$')
    ax.plot(r_grid, h_s2, 'g-', lw=2, label=r'$h_{S^2}(r) = b_1(\psi(r))^2 - b_2(r)^2$')
    ax.set_title(r"Raw Gauged Difference Components $h_{ij} = (\tilde{g}_1 - g_2)_{ij}$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Master radial coordinate $r$")
    ax.set_ylabel("Raw metric difference")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot (0, 1): Relative perturbations hat_h_11, hat_h_22, hat_h_rr
    ax = axes[0, 1]
    ax.plot(r_grid, res['hat_h_rr'], 'r-', lw=2, label=r'$\hat{h}_{rr} = h_{rr}$')
    ax.plot(r_grid, hat_11, 'b-', lw=2, label=r'$\hat{h}_{11} = h_{\theta\theta} / (a^\prime_\infty r)^2$')
    ax.plot(r_grid, hat_22, 'g-', lw=2, label=r'$\hat{h}_{22} = h_{S^2} / (b^\prime_\infty r)^2$')
    ax.set_title(r"Cone-Normalized Relative Perturbations $\hat{h}_{ij}$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Master radial coordinate $r$")
    ax.set_ylabel("Normalized relative perturbation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot (1, 0): Pointwise Riemannian norm |h|_gC and trace tr_gC(h)
    ax = axes[1, 0]
    ax.semilogy(r_grid, norm_h, 'k-', lw=2.2, label=r'Riemannian norm $|h|_{g_C}(r)$')
    ax.semilogy(r_grid, np.abs(tr_h), 'purple', linestyle='--', lw=1.8, label=r'Absolute trace $|\mathrm{tr}_{g_C}(h)|(r)$')
    ax.set_title(r"Riemannian Norm $|h|_{g_C}$ and Trace on Cone Metric $g_C$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Master radial coordinate $r$")
    ax.set_ylabel("Magnitude (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot (1, 1): Anisotropic perturbation ratio hat_h_11 / hat_h_22
    ax = axes[1, 1]
    ratio_angular = hat_11 / (hat_22 + 1e-15)
    ax.plot(r_grid, ratio_angular, 'teal', lw=2, label=r'$\hat{h}_{11}(r) / \hat{h}_{22}(r)$')
    ax.axhline(1.6, color='gray', linestyle=':', label='Asymptotic ratio ~ 1.60')
    ax.set_title(r"Anisotropic Angular Perturbation Ratio $\hat{h}_{11} / \hat{h}_{22}$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Master radial coordinate $r$")
    ax.set_ylabel(r"Ratio $\hat{h}_{11} / \hat{h}_{22}$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step4_gauged_difference.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Diagnostic plot successfully saved to: {plot_path}")
    print("======================================================================\n")


if __name__ == '__main__':
    main()
