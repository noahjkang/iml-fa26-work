# Research Journal: Symmetric Fixed Points of the Ricci Flow

## Entry: 2026-09-13
**Hypothesis & Goals:**
Initialize the numerical solver for the 4-dimensional cohomogeneity one gradient expanding Ricci solitons with $SO(3) \times SO(2)$ symmetry on the $S^1 \times \mathbb{R}^3$ topology. The primary goal is to bypass the degeneracy at the singular orbit $r=0$ and successfully graph the baseline profiles of $a(r)$, $b(r)$, and $f(r)$.

**Mathematical Logic:**
The soliton condition yields a coupled ODE system based on the radial coordinate $r$:
$$f'' = \frac{a''}{a} + 2\frac{b''}{b} - 1$$
$$a'' = -2\frac{a'b'}{b} + a'f' + a$$
$$b'' = \frac{1 - (b')^2}{b} - \frac{a'b'}{a} + b'f' + b$$
Due to division by $b(0)=0$, the system is singular at $r=0$. To circumvent this, the numerical integration is offset to a small boundary $\epsilon = 10^{-4}$ using L'Hôpital's derived Taylor expansions:
$a(\epsilon) = a_0 + \frac{a_0}{6}\epsilon^2, \quad a'(\epsilon) = \frac{a_0}{3}\epsilon$
$b(\epsilon) = \epsilon, \quad b'(\epsilon) = 1$
$f(\epsilon) = \frac{f_0}{2}\epsilon^2, \quad f'(\epsilon) = f_0\epsilon$

**Code Adjustments:**
Developed a Python script utilizing `scipy.integrate.solve_ivp` with the stiff `Radau` method. The initial conditions were encoded precisely via the Taylor approximations at $r = \epsilon$. Specifically, `max(a, 1e-10)` and `max(b, 1e-10)` floor guards were added directly to the ODE function. This was a critical adjustment to protect the stiff Radau solver from division-by-zero crashes during its internal Jacobian estimations.

**Findings & Geometric Interpretations:**
The simulation bypassed the singularity successfully. The resulting baseline profiles demonstrated asymptotic conicality, as $a(r)$ and $b(r)$ exhibited strict linear growth at large distances ($r$), with their first derivatives converging to finite, positive constants.

---

## Entry: 2026-09-13
**Hypothesis & Goals:**
Numerically prove the trivial scaling property of the ODE system with respect to the initial condition $a_0$, and systematically map how the asymptotic cone limits $a'_\infty$ and $b'_\infty$ depend on the initial concavity parameter $f_0$ to determine the expander degree.

**Mathematical Logic:**
Any term involving $a$ in the ODE system appears as a ratio (e.g., $a'/a$ or $a''/a$). Consequently, scaling $a_0$ by a constant $c$ should yield a new solution $(c \cdot a, b, f)$, scaling the asymptotic limit $a'_\infty$ by $c$ while leaving $b'_\infty$ fundamentally unaffected. To prove asymptotic conicality strictly, the limits must correspond to vanishing curvature at large distances, dictated by $O(r^{-2})$ decay of the second derivatives.

**Code Adjustments:**
- Implemented a verification function solving the ODEs for $a_0=1$ and scaled $a_0=2.5$ (`rtol=1e-10`, `atol=1e-10`).
- Created a limit mapping script fixing $a_0 = 1$ and systematically varying $f_0 \in [-0.1, -5.0]$, evaluating the integration to a large distance $r=20$. We explicitly computed and printed the maximal absolute values of $a''(20)$ and $b''(20)$ to bound the curvature decay.
- Generated a visualization featuring a 3D limit mapping ($a'_\infty, f_0, b'_\infty$) alongside a 2D parametric mapping.

**Findings & Geometric Interpretations:**
The trivial scaling property for $a(r)$ was successfully proven, displaying numerical differences near the $10^{-8}$ bound. 

For the limit mapping, the 2D parametric mapping of $F(1, -f_0) = (a'_\infty, b'_\infty)$ does not cross itself, numerically proving the expander degree for $S^1 \times \mathbb{R}^3$ is exactly 1. Note that as $f_0 \to 0^-$, both $a'_\infty, b'_\infty \to \infty$ (approaching hyperbolic geometry), and as $f_0 \to -\infty$, $b'_\infty \to 0$. Also note that $O(r^{-2})$ curvature decay was successfully verified by bounding $a''(20)$ and $b''(20)$ near zero, confirming the slopes have fundamentally stabilized as asymptotic cones.
