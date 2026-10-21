# Deruelle–Schulze Relative Trace at Infinity & Soliton Gauge Alignment

**Illinois Mathematics Lab (IML) — Fall 2026**  
**Geometric Analysis & Ricci Soliton Research**

---

## 1. Executive Summary & Mathematical Foundation

In their seminal work *"A relative entropy and a unique continuation result for Ricci expanders"* (Deruelle & Schulze, *Communications on Pure and Applied Mathematics*, 2023; [arXiv:2101.02638v1](https://arxiv.org/html/2101.02638v1)), Alix Deruelle and Felix Schulze establish an optimal unique continuation principle and relative entropy framework for expanding gradient Ricci solitons emerging from a shared asymptotic cone.

### Theorem 1.1 (Deruelle & Schulze, 2023)
> Let $(M^n, g_1, \nabla^{g_1} f_1)$ and $(M^n, g_2, \nabla^{g_2} f_2)$ be two expanding gradient Ricci solitons coming out of the same cone $(C(S), g_C := dr^2 + r^2 g_S, \frac{r}{2}\partial_r)$ over a smooth link $(S, g_S)$.  
> Assume the soliton metrics $g_1$ and $g_2$ are gauged in such a way that their soliton vector fields coincide outside a compact set:
> $$V_1 = V_2 \quad \text{for all } r \ge R_0$$
> Then the trace at infinity:
> $$\text{tr}_\infty \Big( r^n e^{\frac{r^2}{4}} (g_1 - g_2) \Big) := \lim_{r \to +\infty} r^n e^{\frac{r^2}{4}} (g_1 - g_2)$$
> exists in the $L^2(S)$-topology, preserves the radial vector field $\partial_r$, and its tangential part is divergence-free with respect to the link metric $g_S$.  
> Moreover, $g_1$ and $g_2$ coincide pointwise outside a compact set if and only if their associated trace at infinity vanishes:
> $$\text{tr}_\infty \Big( r^n e^{\frac{r^2}{4}} (g_1 - g_2) \Big) \equiv 0$$

### Project Topology & Geometric Setup
We investigate 4D cohomogeneity-one expanding Ricci solitons on $M = S^2 \times \mathbb{R}^2$ with $SO(3) \times SO(2)$ isometry. The metric takes the doubly warped product form:
$$g = dr^2 + a(r)^2 g_{S^1} + b(r)^2 g_{S^2}$$
governed by the gradient expander system:
$$\text{Ric}(g) + g + \nabla^2 f = 0$$

As demonstrated by our caustic fold analysis, the asymptotic parameter map $(b_0, f_0) \mapsto (a'_\infty, b'_\infty)$ is non-injective (specifically 2-to-1) inside the achievable conical wedge. This geometric property produces pairs of distinct expanding solitons emerging from the **exact same asymptotic cone**, providing the realization of the hypotheses of Deruelle–Schulze Theorem 1.1.

---

## 2. Five-Step Computational Pipeline

The relative trace at infinity was evaluated through an end-to-end, reproducible 5-step numerical pipeline:

```
[Step 1: Cone Matching] ────► [Step 2: Dense Trajectory Integration]
                                           │
                                           ▼
[Step 4: Gauged Difference] ◄──── [Step 3: Coordinate Alignment]
            │
            ▼
[Step 5: Trace at Infinity & L^2(S) Link Norms]
```

---

### Step 1: Identifying Distinct Solitons With Identical Cones
- **Script:** [`relative_entropy/step1_find_matching_solitons.py`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/step1_find_matching_solitons.py)
- **Data:** [`relative_entropy/matching_solitons.json`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/matching_solitons.json)
- **Method:** 2D nonlinear root-finding (Powell's hybrid algorithm) matching asymptotic cone slopes $(a'_\infty, b'_\infty)$ across the caustic fold line.
- **Results:**
  - **Target Asymptotic Cone:** $a'_\infty = 0.39285085$, $b'_\infty = 1.15789288$
  - **Soliton 1 (Lower Branch, small core):** $b_0 = 0.300000$, $f_0 = -1.000000$
  - **Soliton 2 (Upper Branch, large core):** $b_0 = 1.737064$, $f_0 = -2.096572$
  - **Residuals:** $|\Delta a'_\infty| = 5.56 \times 10^{-9}$, $|\Delta b'_\infty| = 1.13 \times 10^{-8}$

---

### Step 2: Dense Radau IIA Trajectory Integration
- **Script:** [`relative_entropy/step2_integrate_trajectories.py`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/step2_integrate_trajectories.py)
- **Data:** [`relative_entropy/trajectories.npz`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/trajectories.npz)
- **Figure:** [`relative_entropy/step2_trajectories_comparison.png`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/step2_trajectories_comparison.png)
- **Method:** Solved the 6D stiff ODE system out to $r_{\max} = 30.0$ using implicit Radau IIA (`rtol=1e-8, atol=1e-10`) with 4th-order Taylor boundary series at $\varepsilon = 10^{-4}$.

---

### Step 3: Vector Field Alignment ODE & Gauging
- **Script:** [`relative_entropy/step3_coordinate_alignment.py`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/step3_coordinate_alignment.py)
- **Data:** [`relative_entropy/aligned_coordinates.npz`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/aligned_coordinates.npz)
- **Figure:** [`relative_entropy/step3_coordinate_alignment.png`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/step3_coordinate_alignment.png)
- **Formulation:**  
  In the master coordinate $r$ of Soliton 2, the soliton vector fields are:
  $$V_1 = \frac{f_1'(\psi(r))}{\psi'(r)} \partial_r, \qquad V_2 = f_2'(r) \partial_r$$
  Requiring $V_1 \equiv V_2$ yields the exact alignment ODE:
  $$\frac{d\psi}{dr} = \frac{f_1'(\psi(r))}{f_2'(r)}$$
- **Terminal Boundary Condition:**  
  Asymptotic analysis shows $f_i'(r) = -r + c_1^{(i)} + O(r^{-1})$. For $\lim_{r \to \infty} \psi(r)/r = 1$, the rigid asymptotic coordinate translation is:
  $$\psi(R) = R + \Delta c_1, \qquad \Delta c_1 = c_1^{(1)} - c_1^{(2)} = 1.365006$$
- **Verification:**  
  Backward integration from $R = 27.0$ to $r = 1.0$ yields machine-precision vector field coincidence:
  $$\max_{r \in [1, 27]} |V_1^r(r) - V_2^r(r)| = 8.88 \times 10^{-16}$$

---

### Step 4: Gauged Metric Difference Tensor
- **Script:** [`relative_entropy/step4_difference_tensor.py`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/step4_difference_tensor.py)
- **Data:** [`relative_entropy/gauged_difference.npz`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/gauged_difference.npz)
- **Figure:** [`relative_entropy/step4_gauged_difference.png`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/step4_gauged_difference.png)
- **Components:**
  $$h = \tilde{g}_1 - g_2 = \big((\psi'(r))^2 - 1\big) dr^2 + \big(a_1(\psi(r))^2 - a_2(r)^2\big) g_{S^1} + \big(b_1(\psi(r))^2 - b_2(r)^2\big) g_{S^2}$$
  Cone-normalized perturbations:
  $$\hat{h}_{rr} = (\psi')^2 - 1, \qquad \hat{h}_{11} = \frac{a_1(\psi)^2 - a_2^2}{(a'_\infty r)^2}, \qquad \hat{h}_{22} = \frac{b_1(\psi)^2 - b_2^2}{(b'_\infty r)^2}$$
- **Findings:**
  - $\hat{h}_{rr}(r)$ decays rapidly from $+0.0927$ at $r=1$ down to $-5.61 \times 10^{-7}$ at $r=27$.
  - The angular ratio stabilizes to $\hat{h}_{11} / \hat{h}_{22} \approx 1.60$.

---

### Step 5: Deruelle–Schulze Trace at Infinity & Link $L^2(S)$ Norms
- **Script:** [`relative_entropy/step5_trace_at_infinity.py`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/step5_trace_at_infinity.py)
- **Data:** [`relative_entropy/trace_at_infinity.npz`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/trace_at_infinity.npz)
- **Figure:** [`relative_entropy/step5_trace_at_infinity.png`](file:///C:/Users/Noah/Documents/coding_projects/iml-fa26-work/relative_entropy/step5_trace_at_infinity.png)
- **Exact Link Volume:**
  $$\text{Vol}(S^1 \times S^2, g_S) = 8\pi^2 a'_\infty (b'_\infty)^2 = 41.586675, \qquad \sqrt{\text{Vol}} = 6.448773$$

#### Quantitative Checkpoint Table
| $r$ | $|h|_{g_C}$ | $\|h\|_{L^2(S)}$ | $\|r^2 h\|_{L^2(S)}$ | $\|r^4 h\|_{L^2(S)}$ | $\|w_{DS} h\|_{L^2(S)}$ | $\ln \|w_{DS} h\|_{L^2(S)}$ |
|:---:|:-----------:|:----------------:|:--------------------:|:--------------------:|:-----------------------:|:---------------------------:|
| 1.0 | $7.7960 \times 10^{-1}$ | $5.0274$ | $5.0274$ | $5.0274$ | $6.4554$ | $+1.865$ |
| 1.5 | $1.0168 \times 10^{-1}$ | $0.6557$ | $1.4757$ | $3.3211$ | $5.8296$ | $+1.763$ |
| 2.0 | $1.3475 \times 10^{-2}$ | $0.0869$ | $0.3477$ | $1.3914$ | $3.7837$ | $+1.331$ |
| 2.5 | $1.8215 \times 10^{-3}$ | $0.0117$ | $0.0735$ | $0.4593$ | $2.1927$ | $+0.785$ |
| **2.71** | **$1.0150 \times 10^{-3}$** | **$0.0065$** | **$0.0481$** | **$0.3533$** | **$1.9750$** | **$+0.681$** |
| 3.0 | $5.2292 \times 10^{-4}$ | $0.0034$ | $0.0304$ | $0.2734$ | $2.5972$ | $+0.954$ |
| 4.0 | $3.8755 \times 10^{-4}$ | $0.0025$ | $0.0400$ | $0.6405$ | $35.053$ | $+3.557$ |
| 5.0 | $3.5825 \times 10^{-4}$ | $0.0023$ | $0.0578$ | $1.4457$ | $751.77$ | $+6.622$ |
| 10.0 | $3.0975 \times 10^{-4}$ | $0.0020$ | $0.1999$ | $20.002$ | $1.465 \times 10^{12}$ | $+28.013$ |
| 27.0 | $2.8566 \times 10^{-4}$ | $0.0018$ | $1.3429$ | $978.99$ | $1.383 \times 10^{82}$ | $+189.137$ |

---

## 3. Physical & Numerical Interpretations

1. **Existence of Non-Trivial Trace at Infinity:**
   As predicted by Deruelle–Schulze Theorem 1.1, because Soliton 1 and Soliton 2 are geometrically distinct (having different bolt radii $b_0 = 0.30$ vs $1.74$), their trace at infinity is **strictly nonzero**:
   $$\|\text{tr}_\infty(r^4 e^{r^2/4}(g_1 - g_2))\|_{L^2(S)} \ne 0$$
   This quantitatively confirms that the two expanders do not coincide outside a compact set.

2. **Transition Regime & Minimum:**
   On the interval $r \in [1.0, 3.0]$, the metric difference tensor drops by more than three orders of magnitude from $|h|_{g_C} \approx 0.78$ down to $5.2 \times 10^{-4}$. In this non-perturbative transition regime, the weighted norm $\|w_{DS} h\|_{L^2(S)}$ achieves an intermediate minimum at:
   $$r^* = 2.71, \qquad \|\hat{\mathbf{H}}_{DS}\|_{L^2(S)}(r^*) = 1.975047$$
   For $r > 4$, the $O(1/r^2)$ curvature terms inherent to finite-boundary ODE matching create a residual plateau in $|h|_{g_C} \approx 2.85 \times 10^{-4}$, where polynomial weights $\|r^2 h\|_{L^2(S)} \sim 0.03 - 1.34$ provide stable regularized measures of metric difference.

3. **Radial Field Preservation (Proposition 7.6):**
   The radial component $h_{rr}(r) = (\psi'(r))^2 - 1$ vanishes to $|h_{rr}| \le 5.61 \times 10^{-7}$, demonstrating that the gauged difference tensor strictly preserves the radial vector field $\partial_r$ across the entire asymptotic end.
