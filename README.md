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
    └── (In progress)                  # Coordinate reparametrization & trace at infinity calculation
```
