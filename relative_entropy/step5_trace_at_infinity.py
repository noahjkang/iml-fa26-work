r"""
Step 5: Evaluation of the Deruelle–Schulze Trace at Infinity and Relative Trace Quantity

Mathematical Context:
---------------------
In Theorem 1.1 of Deruelle & Schulze (2023, arXiv:2101.02638v1, CPAM), the central unique 
continuation and relative entropy result states:
Let (M^n, g_1, \nabla^{g_1} f_1) and (M^n, g_2, \nabla^{g_2} f_2) be two expanding gradient 
Ricci solitons emerging out of the exact same asymptotic cone (C(S), g_C, (r/2) \partial_r) 
over a smooth link (S, g_S). Once gauged so that their soliton drift vector fields coincide 
identically outside a compact set (V_1 = V_2 for r >= R_0), the trace at infinity:

    \text{tr}_\infty \Big( r^n e^{r^2/4} (g_1 - g_2) \Big) := \lim_{r \to \infty} r^n e^{r^2/4} (g_1 - g_2)

exists in the L^2(S)-topology on the link S. Furthermore:
1. It preserves the radial vector field \partial_r:
       \hat{h}(\partial_r, \partial_r) = \frac{1}{2} \text{tr}_{g_C}(\hat{h})
   as established in Proposition 7.6.
2. Its tangential component on the link (S, g_S) is weakly divergence-free.
3. The two solitons coincide pointwise outside a compact set if and only if the trace 
   at infinity vanishes identically:
       \text{tr}_\infty \Big( r^n e^{r^2/4} (g_1 - g_2) \Big) \equiv 0

Geometric Setup on M = S^2 x R^2:
---------------------------------
For 4D cohomogeneity-one expanding solitons with doubly warped product metric:
    g = dr^2 + a(r)^2 g_{S^1} + b(r)^2 g_{S^2}
the asymptotic cone is C(S^1 x S^2) with link metric:
    g_S = (a'_\infty)^2 d\theta^2 + (b'_\infty)^2 g_{S^2}
The volume of the link manifold S = S^1 x S^2 is given exactly by:
    \text{Vol}(S, g_S) = (2\pi a'_\infty) \times (4\pi (b'_\infty)^2) = 8\pi^2 a'_\infty (b'_\infty)^2

Because the solitons possess SO(3) x SO(2) symmetry, the metric difference tensor 
h = \tilde{g}_1 - g_2 is homogeneous across each link orbit. Thus, the L^2(S) integral 
norm of any tensor T on the level set S_r is given algebraically by:
    \|T\|_{L^2(S)}(r) = \sqrt{\int_S |T|_{g_C}^2 \, d\mu_{g_S}} = |T|_{g_C}(r) \sqrt{\text{Vol}(S, g_S)}

Conventions & Weighting Schemes:
--------------------------------
- Deruelle–Schulze CPAM Convention:
      \text{Ric} + \frac{1}{2} g = \nabla^2 f \implies f \sim +\frac{r^2}{4}
  yielding the Gaussian weight w_{DS}(r) = r^n e^{r^2/4} (with n = 4).
- Project / IML Convention:
      \text{Ric} + g + \nabla^2 f = 0 \implies -f \sim +\frac{r^2}{2}
  yielding the natural potential weight w_{pot}(r) = r^n e^{-f_2(r)}.
- Conical Polynomial Regularizations:
  In finite-domain numerical integrations, curvature terms scale as O(r^{-2}). 
  We evaluate polynomial weights r^2 |h|_{g_C} and r^4 |h|_{g_C} alongside the exponential 
  weights to track convergence and characterize the transition between the non-perturbative 
  decay regime and the numerical floor.

Pipeline Role:
--------------
- Inputs: Loads `gauged_difference.npz` (Step 4), `trajectories.npz` (Step 2), and 
  `matching_solitons.json` (Step 1).
- Computations: Evaluates the weighted trace tensor components, calculates the link L^2 
  norms, checks the radial preservation identity, and identifies the optimal trace value.
- Outputs: Serializes all trace data to `relative_entropy/trace_at_infinity.npz`.
  Generates 4-panel publication-grade diagnostic plot `relative_entropy/step5_trace_at_infinity.png`.

Numerical Methods:
------------------
- Shape-preserving PchipInterpolator to transfer potential trajectories smoothly onto the aligned grid.
- Vectorized tensor algebraic evaluations of the L^2(S) integral norm.
- Exact closed-form integration over the compact homogeneous link S^1 x S^2.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import json
import os
import sys

# Ensure project root is in python path to import core models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_link_geometry(a_inf, b_inf):
    r"""
    Computes exact geometric invariants of the asymptotic cone link S = S^1 x S^2.

    Mathematical Formulation:
    -------------------------
    The link manifold S = S^1 x S^2 is endowed with the Riemannian metric:
        g_S = (a'_\infty)^2 d\theta^2 + (b'_\infty)^2 g_{S^2}
    where:
        \theta \in [0, 2\pi) parametrizes the S^1 circle of radius a'_\infty,
        g_{S^2} is the standard round metric on the 2-sphere with radius b'_\infty.

    The total Riemannian volume measure is:
        \text{Vol}(S, g_S) = \int_0^{2\pi} a'_\infty d\theta \times \int_{S^2} (b'_\infty)^2 d\Omega
                           = (2\pi a'_\infty) \times (4\pi (b'_\infty)^2)
                           = 8 \pi^2 a'_\infty (b'_\infty)^2

    Parameters:
    -----------
    a_inf : float
        Asymptotic cone slope a'_\infty > 0 along the S^1 fiber.
    b_inf : float
        Asymptotic cone slope b'_\infty > 0 along the S^2 base.

    Returns:
    --------
    dict:
        Dictionary containing:
        - 'vol_link': Total Riemannian volume Vol(S, g_S).
        - 'vol_s1': Circumference of S^1 (2\pi a'_\infty).
        - 'vol_s2': Area of S^2 (4\pi (b'_\infty)^2).
        - 'sqrt_vol': \sqrt{Vol(S, g_S)} scaling factor for L^2 norms.
    """
    if a_inf <= 0.0 or b_inf <= 0.0:
        raise ValueError(f"Asymptotic cone slopes must be strictly positive: a_inf={a_inf}, b_inf={b_inf}")

    vol_s1 = 2.0 * np.pi * a_inf
    vol_s2 = 4.0 * np.pi * (b_inf**2)
    vol_link = vol_s1 * vol_s2
    sqrt_vol = np.sqrt(vol_link)

    return {
        'vol_link': vol_link,
        'vol_s1': vol_s1,
        'vol_s2': vol_s2,
        'sqrt_vol': sqrt_vol
    }


def compute_weighted_trace_tensor(gauged_diff, f2_grid, dim_n=4):
    r"""
    Computes the Deruelle–Schulze weighted trace tensor and its L^2(S) norms.

    Mathematical Formulation:
    -------------------------
    Given the gauged difference tensor components:
        \hat{h}_{rr}(r) = (\psi'(r))^2 - 1
        \hat{h}_{11}(r) = \frac{a_1(\psi(r))^2 - a_2(r)^2}{(a'_\infty r)^2}
        \hat{h}_{22}(r) = \frac{b_1(\psi(r))^2 - b_2(r)^2}{(b'_\infty r)^2}
    and Riemannian norm |h|_{g_C}(r) = \sqrt{\hat{h}_{rr}^2 + \hat{h}_{11}^2 + 2\hat{h}_{22}^2}.

    We construct the weighted trace tensor:
        \hat{\mathbf{H}}_{\text{weight}}(r) = w(r) (\tilde{g}_1 - g_2)
    under four complementary weighting schemes:
    1. Deruelle–Schulze Theorem 1.1 weight:
           w_{DS}(r) = r^n e^{r^2/4}
    2. Soliton potential weight (IML convention):
           w_{pot}(r) = r^n e^{-f_2(r)}
    3. Quadratic conical weight:
           w_{r2}(r) = r^2
    4. Quartic conical weight:
           w_{r4}(r) = r^4

    The L^2(S) norm of the weighted tensor across the link level set is:
        \|\hat{\mathbf{H}}\|_{L^2(S)}(r) = w(r) |h|_{g_C}(r) \sqrt{\text{Vol}(S, g_S)}

    Radial Preservation Check (Proposition 7.6):
        \text{residual}_{\text{rad}}(r) = |\hat{h}_{rr}(r) - \frac{1}{2} \text{tr}_{g_C}(h)(r)|

    Parameters:
    -----------
    gauged_diff : dict
        Loaded dictionary from `gauged_difference.npz`.
    f2_grid : np.ndarray
        Potential function f_2(r) interpolated onto the radial grid.
    dim_n : int
        Dimension of the manifold (n = 4 for S^2 x R^2).

    Returns:
    --------
    dict:
        Comprehensive dictionary of weighted components, norms, and limits.
    """
    r = gauged_diff['r']
    h_rr = gauged_diff['hat_h_rr']
    h_11 = gauged_diff['hat_h_11']
    h_22 = gauged_diff['hat_h_22']
    norm_h = gauged_diff['norm_h']
    tr_h = gauged_diff['tr_h']
    a_inf = float(gauged_diff['a_inf'])
    b_inf = float(gauged_diff['b_inf'])

    # Step 5.1: Compute link geometric volume
    geom = compute_link_geometry(a_inf, b_inf)
    sqrt_vol = geom['sqrt_vol']

    # Step 5.2: Evaluate weight functions across the radial grid
    # To prevent floating-point overflow for very large r, compute in log-space where needed
    w_r2 = r**2
    w_r4 = r**4
    
    # Deruelle-Schulze weight: r^n * exp(r^2 / 4)
    log_w_ds = dim_n * np.log(r) + 0.25 * (r**2)
    # Natural soliton potential weight: r^n * exp(-f_2)
    log_w_pot = dim_n * np.log(r) - f2_grid

    # Direct evaluation for stable radial range
    w_ds = np.exp(np.clip(log_w_ds, -100.0, 700.0))
    w_pot = np.exp(np.clip(log_w_pot, -100.0, 700.0))

    # Step 5.3: Compute weighted tensor components under Deruelle-Schulze weight
    H_rr_ds = w_ds * h_rr
    H_11_ds = w_ds * h_11
    H_22_ds = w_ds * h_22
    tr_H_ds = w_ds * tr_h
    norm_H_ds = w_ds * norm_h

    # Step 5.4: Compute L^2(S) integral norms on the link
    l2_unweighted = norm_h * sqrt_vol
    l2_r2 = w_r2 * norm_h * sqrt_vol
    l2_r4 = w_r4 * norm_h * sqrt_vol
    l2_ds = norm_H_ds * sqrt_vol

    # Step 5.5: Compute log-space L^2 norm for the full domain to avoid clipping artifacts
    log_l2_ds = log_w_ds + np.log(norm_h) + np.log(sqrt_vol)
    log_l2_pot = log_w_pot + np.log(norm_h) + np.log(sqrt_vol)

    # Step 5.6: Radial vector field preservation identity check (Proposition 7.6)
    # Identifies deviation |hat_h_rr - 0.5 * tr_h|
    radial_dev = np.abs(h_rr - 0.5 * tr_h)
    radial_dev_weighted = w_ds * radial_dev

    # Step 5.7: Identify transitional minimum / plateau of the weighted trace
    # In numerical solitons, the exponential weight reaches a transitional minimum 
    # before numerical truncation in matching takes over
    min_idx_ds = int(np.argmin(log_l2_ds[:int(len(r) * 0.4)]))
    r_plat = float(r[min_idx_ds])
    l2_plat = float(l2_ds[min_idx_ds])

    return {
        'r': r,
        'a_inf': a_inf,
        'b_inf': b_inf,
        'geom': geom,
        'w_r2': w_r2,
        'w_r4': w_r4,
        'w_ds': w_ds,
        'log_w_ds': log_w_ds,
        'log_w_pot': log_w_pot,
        'H_rr_ds': H_rr_ds,
        'H_11_ds': H_11_ds,
        'H_22_ds': H_22_ds,
        'tr_H_ds': tr_H_ds,
        'norm_H_ds': norm_H_ds,
        'l2_unweighted': l2_unweighted,
        'l2_r2': l2_r2,
        'l2_r4': l2_r4,
        'l2_ds': l2_ds,
        'log_l2_ds': log_l2_ds,
        'log_l2_pot': log_l2_pot,
        'radial_dev': radial_dev,
        'radial_dev_weighted': radial_dev_weighted,
        'r_plateau': r_plat,
        'l2_plateau': l2_plat
    }


def plot_trace_at_infinity(results, output_path):
    r"""
    Generates a 4-panel publication-grade diagnostic plot of the trace at infinity.

    Diagnostic Panels:
    ------------------
    - Subplot (0, 0): Deruelle–Schulze Weighted Components H_11, H_22, H_rr vs r.
    - Subplot (0, 1): Link L^2(S) Norms under Polynomial and Exponential Weightings.
    - Subplot (1, 0): Pointwise Tensor Trace tr_gC(H) and Radial Identity Comparison.
    - Subplot (1, 1): Logarithmic L^2 Trajectory and Extraction of the Intermediate Transition.

    Parameters:
    -----------
    results : dict
        Dictionary of computed trace arrays from `compute_weighted_trace_tensor`.
    output_path : str
        Path where the generated figure will be saved.
    """
    r = results['r']
    geom = results['geom']

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # -------------------------------------------------------------------------
    # Panel 1: Weighted Tensor Components on Intermediate Domain r in [1, 5]
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    mask_near = (r >= 1.0) & (r <= 5.0)
    r_n = r[mask_near]
    ax.plot(r_n, results['H_11_ds'][mask_near], 'b-', lw=2.2, label=r'$\hat{\mathbf{H}}_{11} = r^4 e^{r^2/4} \hat{h}_{11}$')
    ax.plot(r_n, results['H_22_ds'][mask_near], 'g-', lw=2.2, label=r'$\hat{\mathbf{H}}_{22} = r^4 e^{r^2/4} \hat{h}_{22}$')
    ax.plot(r_n, results['H_rr_ds'][mask_near], 'r--', lw=1.8, label=r'$\hat{\mathbf{H}}_{rr} = r^4 e^{r^2/4} \hat{h}_{rr}$')
    ax.set_title(r"Deruelle–Schulze Weighted Tensor Components $\hat{\mathbf{H}}_{ij}$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Master radial coordinate $r$")
    ax.set_ylabel(r"Weighted component magnitude $r^4 e^{r^2/4} \hat{h}_{ij}$")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 2: Link L^2(S) Norms across Full Radial Range
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.semilogy(r, results['l2_unweighted'], 'k-', lw=2.0, label=r'Unweighted $\|h\|_{L^2(S)}$')
    ax.semilogy(r, results['l2_r2'], 'teal', linestyle='--', lw=2.0, label=r'Quadratic $\|r^2 h\|_{L^2(S)}$')
    ax.semilogy(r, results['l2_r4'], 'purple', linestyle='-.', lw=2.0, label=r'Quartic $\|r^4 h\|_{L^2(S)}$')
    ax.set_title(r"Link $L^2(S^1 \times S^2)$ Norms: Regularized Convergence", fontsize=12, fontweight='bold')
    ax.set_xlabel("Master radial coordinate $r$")
    ax.set_ylabel(r"Link $L^2(S)$ norm (log scale)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 3: Radial Vector Field Preservation (Proposition 7.6)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.semilogy(r, np.abs(results['H_rr_ds']), 'r-', lw=2.0, label=r'$|\hat{\mathbf{H}}_{rr}| = |\hat{\mathbf{H}}(\partial_r, \partial_r)|$')
    ax.semilogy(r, 0.5 * np.abs(results['tr_H_ds']), 'b--', lw=2.0, label=r'$\frac{1}{2} |\mathrm{tr}_{g_C} \hat{\mathbf{H}}|$')
    ax.semilogy(r, results['radial_dev'], 'k:', lw=1.8, label=r'Raw deviation $|\hat{h}_{rr} - \frac{1}{2}\mathrm{tr}_{g_C} h|$')
    ax.set_title(r"Radial Field Preservation (Proposition 7.6)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Master radial coordinate $r$")
    ax.set_ylabel(r"Magnitude (log scale)")
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 4: Full Log-Scale L^2(S) Trajectory & Transitional Minimum
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(r, results['log_l2_ds'], 'crimson', lw=2.2, label=r'$\ln \|\hat{\mathbf{H}}_{\mathrm{DS}}\|_{L^2(S)} = \ln(r^4 e^{r^2/4} |h|_{g_C} \sqrt{\mathrm{Vol}})$')
    ax.plot(r, results['log_l2_pot'], 'navy', linestyle='--', lw=2.0, label=r'$\ln \|\hat{\mathbf{H}}_{\mathrm{pot}}\|_{L^2(S)} = \ln(r^4 e^{-f} |h|_{g_C} \sqrt{\mathrm{Vol}})$')
    ax.axvline(results['r_plateau'], color='darkgreen', linestyle=':', lw=2.0, 
               label=f"Transition point $r^* = {results['r_plateau']:.2f}$")
    ax.scatter([results['r_plateau']], [np.log(results['l2_plateau'])], color='darkgreen', s=70, zorder=5)
    ax.set_title(r"Logarithmic Relative Trace Norm $\ln \|\hat{\mathbf{H}}\|_{L^2(S)}(r)$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Master radial coordinate $r$")
    ax.set_ylabel(r"Logarithmic L2 Norm $\ln \|\hat{\mathbf{H}}\|_{L^2(S)}$")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    r"""
    Main driver for Step 5:
    - Loads `gauged_difference.npz`, `trajectories.npz`, and `matching_solitons.json`.
    - Interpolates potential function f_2(r) onto the shared master radial grid.
    - Evaluates the Deruelle–Schulze weighted trace at infinity and link L^2 norms.
    - Verifies the Proposition 7.6 radial vector field preservation identity.
    - Serializes complete trace analysis data to `trace_at_infinity.npz`.
    - Generates 4-panel diagnostic plot `step5_trace_at_infinity.png`.
    """
    print("======================================================================")
    print("Step 5: Evaluation of Deruelle-Schulze Trace at Infinity (Theorem 1.1)")
    print("======================================================================")

    # -------------------------------------------------------------------------
    # Phase 1: Load input data archives from preceding steps
    # -------------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gauged_path = os.path.join(base_dir, "gauged_difference.npz")
    trajs_path = os.path.join(base_dir, "trajectories.npz")
    meta_path = os.path.join(base_dir, "matching_solitons.json")

    for path, name in [(gauged_path, "Gauged difference"), (trajs_path, "Trajectories"), (meta_path, "Metadata")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path} ({name}). Complete preceding steps first.")

    gauged_data = np.load(gauged_path)
    trajs_data = np.load(trajs_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    r_grid = gauged_data['r']
    a_inf = meta["cone_target"]["a_prime_infinity"]
    b_inf = meta["cone_target"]["b_prime_infinity"]

    print(f"Loaded {len(r_grid)} grid points across r in [{r_grid[0]:.1f}, {r_grid[-1]:.1f}].")
    print(f"Cone slopes: a'_inf = {a_inf:.8f}, b'_inf = {b_inf:.8f}")

    # -------------------------------------------------------------------------
    # Phase 2: Interpolate potential function f_2(r) onto the master radial grid
    # -------------------------------------------------------------------------
    print("Interpolating potential function f_2(r) using shape-preserving PCHIP...")
    interp_f2 = PchipInterpolator(trajs_data['r'], trajs_data['f2'])
    f2_grid = interp_f2(r_grid)

    # -------------------------------------------------------------------------
    # Phase 3: Evaluate weighted trace tensor, link geometry, and norms
    # -------------------------------------------------------------------------
    print("Evaluating weighted trace tensor components and link L^2(S) norms...")
    results = compute_weighted_trace_tensor(gauged_data, f2_grid, dim_n=4)

    geom = results['geom']
    print(f"\nLink Geometry S = S^1 x S^2:")
    print(f"  Vol(S^1) = {geom['vol_s1']:.6f}")
    print(f"  Vol(S^2) = {geom['vol_s2']:.6f}")
    print(f"  Vol(S, g_S) = {geom['vol_link']:.6f}")
    print(f"  L^2 scaling factor sqrt(Vol) = {geom['sqrt_vol']:.6f}")

    # -------------------------------------------------------------------------
    # Phase 4: Print quantitative checkpoints for diagnostic verification
    # -------------------------------------------------------------------------
    print("\nQuantitative Trace at Infinity Checkpoints:")
    print("-------------------------------------------------------------------------------------------------------------")
    print("   r   |    |h|_gC    |  ||h||_L2(S)  | ||r^2 h||_L2  | ||r^4 h||_L2  | ||w_DS h||_L2 |  ln ||w_DS h||_L2 ")
    print("-------------------------------------------------------------------------------------------------------------")
    check_radii = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 27.0]
    for r_chk in check_radii:
        idx = int(np.argmin(np.abs(r_grid - r_chk)))
        rv = r_grid[idx]
        nh = results['geom']['sqrt_vol'] * gauged_data['norm_h'][idx]
        r2_norm = results['l2_r2'][idx]
        r4_norm = results['l2_r4'][idx]
        ds_norm = results['l2_ds'][idx]
        log_ds = results['log_l2_ds'][idx]
        print(f" {rv:5.1f} | {gauged_data['norm_h'][idx]:11.4e} | {nh:12.4e} | {r2_norm:12.4e} | {r4_norm:12.4e} | {ds_norm:13.4e} | {log_ds:15.3f}")
    print("-------------------------------------------------------------------------------------------------------------")

    print(f"\nTransitional Minimum / Intermediate Plateau:")
    print(f"  Transition Radius r* = {results['r_plateau']:.2f}")
    print(f"  Link L^2 Norm at Transition: ||w_DS h||_L2(S)(r*) = {results['l2_plateau']:.6f}")
    print(f"  Log L^2 Norm at Transition: ln ||w_DS h||_L2(S)(r*) = {np.log(results['l2_plateau']):.6f}")

    # -------------------------------------------------------------------------
    # Phase 5: Serialize computed trace data to compressed archive
    # -------------------------------------------------------------------------
    npz_out = os.path.join(base_dir, "trace_at_infinity.npz")
    np.savez_compressed(
        npz_out,
        r=r_grid,
        a_inf=a_inf,
        b_inf=b_inf,
        vol_link=geom['vol_link'],
        sqrt_vol=geom['sqrt_vol'],
        norm_h=gauged_data['norm_h'],
        tr_h=gauged_data['tr_h'],
        h_rr=gauged_data['hat_h_rr'],
        h_11=gauged_data['hat_h_11'],
        h_22=gauged_data['hat_h_22'],
        H_rr_ds=results['H_rr_ds'],
        H_11_ds=results['H_11_ds'],
        H_22_ds=results['H_22_ds'],
        tr_H_ds=results['tr_H_ds'],
        norm_H_ds=results['norm_H_ds'],
        l2_unweighted=results['l2_unweighted'],
        l2_r2=results['l2_r2'],
        l2_r4=results['l2_r4'],
        l2_ds=results['l2_ds'],
        log_l2_ds=results['log_l2_ds'],
        log_l2_pot=results['log_l2_pot'],
        radial_dev=results['radial_dev'],
        radial_dev_weighted=results['radial_dev_weighted'],
        r_plateau=results['r_plateau'],
        l2_plateau=results['l2_plateau']
    )
    print(f"\nTrace at infinity data saved to: {npz_out}")

    # -------------------------------------------------------------------------
    # Phase 6: Generate 4-panel diagnostic plot
    # -------------------------------------------------------------------------
    plot_out = os.path.join(base_dir, "step5_trace_at_infinity.png")
    print(f"Generating diagnostic figure: {plot_out}...")
    plot_trace_at_infinity(results, plot_out)
    print(f"Diagnostic figure successfully saved to: {plot_out}")
    print("======================================================================\n")


if __name__ == '__main__':
    main()
