# Symmetric Fixed Points of the Ricci Flow
**Illinois Mathematics Lab (IML) — Fall 2026**

## Project Overview

This research project investigates Ricci solitons with $SO(3) \times SO(2)$ symmetry on the topologies $S^1 \times \mathbb{R}^3$ and $S^2 \times \mathbb{R}^2$, as well as Bryant solitons on $\mathbb{R}^n$.

## Directory Organization

The codebase is organized into modular directories corresponding to each major milestone of the project:

```
iml-fa26-work/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python environment dependencies
│
├── src/                               # Core mathematical models & ODE solver engine
│   ├── __init__.py
│   ├── ricci_odes.py                  # Core 6D vector field with numerical floor guards
│   ├── taylor_expansion.py            # SymPy Taylor expansion engine around the r=0 singularity
│   ├── generate_julia_box.py          # Bridge converting Taylor polynomials into Julia interval sets
│   └── initial_box.jl                 # Auto-generated rigorous interval arithmetic initial condition
│
├── cap_verification/                  # Goal 1: 3-Phase Computer-Assisted Proof (CAP) Pipeline
│   ├── docs/                          # Proof LaTeX writeups
│   │   ├── proof_framework.tex        # Master proof blueprint (Phases 1-3)
│   │   ├── majorant_proof.tex         # Phase 1 Cauchy Majorant inductive error bounds
│   │   └── trapping_region.tex        # Phase 3 Lyapunov funnel boundary inequalities
│   ├── part1_majorant.py              # Phase 1: High-order geometric coefficient decay verification
│   ├── part2_integrator.jl            # Phase 2: Rigorous interval flow via Taylor Models (TMJets)
│   ├── part3_trapping.py              # Phase 3: Empirical Lyapunov ratio variable tracking
│   ├── part3_rigorous_trapping.jl     # Phase 3: Certified interval arithmetic trapping proof
│   └── tests/                         # CAP test suite
│       ├── debug_ricci.jl
│       ├── test_interval.jl
│       ├── test_step3.jl
│       ├── test_tmjets.jl
│       └── test_tm_eval.jl
│
├── asymptotic_mapping/                # Goal 2: Geometric Parameter Mapping & Asymptotic Cones
│   ├── bryant_solitons/               # Bryant soliton parameter scan & threshold analysis
│   │   ├── plot_bryant_limit.py       # Stiff Radau solver mapping lim sqrt(h(r)) vs initial slope mu_1
│   │   └── asymptotic_cone_bryan_soliton.png # Divergence plot confirming mu_1 = 1/n threshold
│   └── f2_projections/                # S2 x R2 (f2 map) asymptotic cone space exploration
│       ├── project_f2.py              # 2D projection grid (a'_infty vs b'_infty) colored by f''(0)
│       ├── initial_profiles.py        # Multi-variable 3D surface generator (a'_infty, b'_infty, b0)
│       ├── boundary_scan.py           # Monte Carlo point cloud boundary estimator
│       ├── f2_map_asymptotic_manifold.png # 3D manifold visualization
│       └── f2_projection.png          # 2D achievable wedge projection
│
├── fold_analysis/                     # Goal 3: Fold Caustic & sqrt(3) Theoretical Derivation
│   ├── fold_slope_derivation.tex      # Full derivation proving slope m = sqrt(3)
│   ├── trace_fold_line.py             # Numerical root-finding tracking critical set det(Df2) = 0
│   ├── verify_fold_slope.py           # Numerical verification of the Einstein limit (f0 -> 0-)
│   └── f2_fold_analysis.png           # Plot showing the fold line and sqrt(3) asymptote overlay
│
└── relative_entropy/                  # Goal 4: Deruelle–Schulze Relative Entropy & Trace at Infinity
    ├── compute_finite_trace.py        # Python trace engine
    ├── find_matching_solitons.py      # 2D nonlinear root-finding matching identical asymptotic cones on D2 x S2
    ├── triple_solitons_d3s1.py        # Multi-topology solver isolating geometric triples on D2 x S2 and D3 x S1
    ├── plot_second_derivative.py      # Second derivative d^2Y/dX^2 concavity & positive initial slope analysis
    ├── plot_trace_surface_3d.py       # 3D manifold of finite trace C_AB(a'_infty, b'_infty) on D2 x S2
    ├── plot_trace_surface_d3s1_3d.py  # 3D cross-topological trace manifold C_AC comparing D2 x S2 vs D3 x S1
    ├── plot_triple_trace_surfaces_3d.py # Multi-surface 3D comparison of all 3 traces (C_AB, C_AC, C_BC)
    ├── trace_data_s36.json            # High-precision trajectory checkpoints and finite trace dataset (s <= 12)
    ├── trace_surface_cache.npz        # Precomputed simulation grid cache for D2 x S2 trace surface C_AB
    ├── trace_surface_d3s1_cache.npz   # Precomputed simulation grid cache for D3 x S1 trace surface C_AC
    ├── triple_trace_surfaces_cache.npz# Precomputed simulation grid cache for all 3 trace surfaces
    ├── julia_soliton_matcher.jl       # Julia solver for high-precision 2D root finding
    └── julia_trace_engine.jl          # Julia solver for variational trace cross-verification
```

## Goal 4: Relative Entropy & Trace at Infinity Workflow

Under Deruelle & Schulze (2023, CPAM, Theorem 1.1), two expanding gradient Ricci solitons sharing the same asymptotic cone satisfy:
$$\lim_{s \to \infty} s^4 e^{s^2/4} \|g_1 - g_2\|_\gamma = C_{\text{trace}} < \infty$$

### Key Modules:
- **High-Precision Engine (`compute_finite_trace.py`)**: 100% pure Python implementation using `mpmath` (45 decimal places) and Gauss-Legendre Order 8 collocation to evaluate trajectories out past $s = 10.0 \sim 12.0$.
- **Reference Dataset (`trace_data_s36.json`)**: Precomputed, certified high-precision $(s, X, Y)$ trajectory checkpoints providing instant calibration without re-integrating ODEs.
- **Concavity Verification (`plot_second_derivative.py`)**: Evaluates the second derivative $\frac{d^2Y}{dX^2}$ versus theoretical shape factor $X = s^{-4} e^{-s^2/4}$, showing that $\left.\frac{dY}{dX}\right|_{X=0} = C_{\text{trace}} > 0$ and $\frac{d^2Y}{dX^2} < 0$ everywhere, geometrically guaranteeing that $0 < Y(X) < C_{\text{trace}} X$.
- **Geometric Triples (`triple_solitons_d3s1.py`)**: Identifies expanding solitons sharing the exact same cone across distinct complete 4-manifold topologies: $D^2 \times S^2$ (lower branch $S_A$ & upper branch $S_B$) and $D^3 \times S^1$ (circle core $S_C$).
- **3D Manifold Visualizations**:
  - `plot_trace_surface_3d.py`: 3D surface of $C_{AB}(a'_\infty, b'_\infty)$ between pairs on $D^2 \times S^2$.
  - `plot_trace_surface_d3s1_3d.py`: 3D surface of cross-topological trace $C_{AC}(a'_\infty, b'_\infty)$ comparing $D^2 \times S^2$ and $D^3 \times S^1$.
  - `plot_triple_trace_surfaces_3d.py`: Simultaneous 3D comparison of all 3 pairwise traces ($C_{AB}, C_{AC}, C_{BC}$) over the asymptotic cone plane.
