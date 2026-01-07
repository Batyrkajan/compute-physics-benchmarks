# Accuracy vs Compute Cost Tradeoffs in Numerical Integration for Physics Simulations

**Author:** Batyr
**Date:** January 2026
**Affiliation:** Physics BS, Arizona State University

---

## Abstract

This report presents a systematic benchmark of four numerical integration methods—Forward Euler, 4th-order Runge-Kutta (RK4), adaptive Runge-Kutta-Fehlberg (RK45), and Velocity Verlet—applied to physics simulation problems. We measure accuracy, computational cost, stability, and energy conservation to characterize the fundamental tradeoffs involved in selecting an integration scheme. Results demonstrate that the optimal choice depends critically on the objective: RK45 dominates for accuracy-constrained problems, RK4 offers the best accuracy-per-computation for moderate requirements, while the symplectic Velocity Verlet method is essential for long-time simulations where energy conservation matters. These findings have direct implications for computational physics workflows where compute resources and accuracy requirements must be balanced.

---

## 1. Introduction

Numerical integration of ordinary differential equations (ODEs) is foundational to computational physics. From molecular dynamics to celestial mechanics, the choice of integration method directly impacts both the accuracy of results and the computational resources required.

The central question this work addresses is:

> **How do numerical integration methods trade accuracy vs compute cost for physics simulations?**

This is not merely an academic exercise. In practice, computational physicists must make this tradeoff constantly:
- High-accuracy simulations are expensive
- Fast simulations may be unreliable
- Long-time simulations accumulate errors differently than short ones

Understanding these tradeoffs quantitatively enables informed decision-making about which method to use for a given problem.

### 1.1 Objectives

1. Implement and validate four representative integration methods
2. Measure accuracy as a function of timestep (order verification)
3. Characterize the accuracy-cost Pareto frontier
4. Analyze long-time energy conservation (symplectic vs non-symplectic)
5. Determine stability boundaries for each method

---

## 2. Methods

### 2.1 Integration Algorithms

We implement four methods spanning different orders and structural properties:

**Forward Euler (1st order)**

The simplest explicit method. Updates the state using only the derivative at the current point:

$$y_{n+1} = y_n + \Delta t \cdot f(t_n, y_n)$$

Properties: O(Δt) error, 1 function evaluation per step, conditionally stable.

**Runge-Kutta 4 (4th order)**

The classical "workhorse" integrator. Evaluates derivatives at multiple points within each timestep:

$$y_{n+1} = y_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

where $k_1, k_2, k_3, k_4$ are intermediate evaluations. Properties: O(Δt⁴) error, 4 function evaluations per step.

**RK45 Adaptive (Runge-Kutta-Fehlberg)**

An embedded pair method that computes both 4th and 5th order estimates. The difference provides an error estimate used to adapt the step size:

$$\text{error estimate} = |y_5 - y_4|$$

Properties: Automatic step size control, 6 function evaluations per step, maintains specified tolerance.

**Velocity Verlet (2nd order, symplectic)**

A structure-preserving integrator designed for Hamiltonian systems:

$$x_{n+1} = x_n + v_n \Delta t + \frac{1}{2}a_n \Delta t^2$$
$$v_{n+1} = v_n + \frac{1}{2}(a_n + a_{n+1})\Delta t$$

Properties: O(Δt²) error, symplectic (preserves phase space volume), bounded energy error.

### 2.2 Test Problems

**Simple Harmonic Oscillator**

$$\frac{d^2x}{dt^2} = -\omega^2 x$$

Analytical solution: $x(t) = A\cos(\omega t + \phi)$

This provides ground truth for error measurement. Parameters: ω = 1, x₀ = 1, v₀ = 0, T = 100.

**Nonlinear Pendulum (Large Angle)**

$$\frac{d^2\theta}{dt^2} = -\frac{g}{L}\sin(\theta)$$

No closed-form solution exists. We use scipy's DOP853 (8th order) with tight tolerances as reference. Parameters: g/L = 1, θ₀ = 2.5 rad (143°), T = 50.

### 2.3 Metrics

**Accuracy Metrics:**
- RMS error: $\sqrt{\frac{1}{N}\sum_i |y_i - y_{ref,i}|^2}$
- Maximum error: $\max_i |y_i - y_{ref,i}|$

**Cost Metrics:**
- Wall-clock time (averaged over 10 runs)
- Function evaluations

**Energy Metrics:**
- Relative energy drift: $|E(t) - E(0)|/|E(0)|$

### 2.4 Experimental Design

**Timestep sweep:** dt ∈ {0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001}

**Long-time test:** T = 1000 (≈160 oscillation periods) with dt = 0.01

**Stability analysis:** Binary search for maximum stable timestep

---

## 3. Results

### 3.1 Order Verification

Figure 1 shows RMS error vs timestep on log-log axes. The slope of each line corresponds to the method's order of accuracy.

| Method | Expected Order | Measured Slope |
|--------|----------------|----------------|
| Forward Euler | 1 | ~1.0 |
| Velocity Verlet | 2 | ~2.0 |
| RK4 | 4 | ~4.0 |
| RK45 | 4 | N/A (adaptive) |

Key observations:
- Euler's error decreases linearly with timestep (10x smaller dt → 10x smaller error)
- RK4's error decreases as dt⁴ (10x smaller dt → 10,000x smaller error)
- RK45 maintains constant error regardless of initial timestep (adapts automatically)
- At very small dt, RK4 hits machine precision limits (error stops decreasing)

### 3.2 Accuracy-Cost Tradeoff (Pareto Analysis)

Figure 2 shows error vs runtime, the fundamental tradeoff.

**Key findings:**

1. **RK45 dominates for high accuracy requirements.** When you need error < 10⁻⁶, RK45 achieves this faster than any fixed-step method because it takes large steps in smooth regions.

2. **RK4 dominates for moderate accuracy.** For error targets around 10⁻⁴ to 10⁻⁸, RK4 with appropriate timestep is faster than RK45's adaptive overhead.

3. **Euler is never Pareto-optimal.** For any accuracy level Euler achieves, another method achieves the same accuracy faster.

4. **Verlet is competitive for moderate accuracy** and becomes essential when energy conservation matters (see Section 3.3).

### 3.3 Energy Conservation

Figure 3 shows relative energy error over 1000 time units (≈160 oscillation periods).

| Method | Final Relative Energy Error |
|--------|----------------------------|
| Forward Euler | ~10⁴ (diverges) |
| RK4 | ~10⁻⁹ |
| RK45 | ~10⁻⁶ |
| Velocity Verlet | ~10⁻⁵ |

**Critical insight:** Although RK4 achieves better instantaneous accuracy than Verlet, the symplectic property of Verlet means its energy error is *bounded* rather than *growing*. For simulations running millions of timesteps (molecular dynamics, N-body gravity), this property is essential.

The energy error in non-symplectic methods (RK4, RK45) accumulates secularly over time. In symplectic methods, the error oscillates but doesn't grow—the integrator is solving a *slightly perturbed* Hamiltonian exactly.

### 3.4 Stability

Table 1: Maximum stable timestep (Harmonic Oscillator)

| Method | Max Stable dt | Stability Ratio |
|--------|---------------|-----------------|
| Forward Euler | 0.29 | 1.0x |
| RK4 | 3.00 | 10x |
| Velocity Verlet | 2.02 | 7x |
| RK45 | N/A (adaptive) | N/A |

Euler's stability limit is approximately dt < 2/ω for the harmonic oscillator, matching theoretical predictions.

---

## 4. Discussion

### 4.1 When to Use Each Method

**Forward Euler:** Educational purposes only. Never optimal for production use.

**RK4:** General-purpose workhorse. Use when:
- Moderate accuracy is sufficient
- Fixed timestep output is needed
- Problem is not stiff

**RK45:** Use when:
- Accuracy requirements are strict
- Problem has varying timescales
- Computational budget is flexible

**Velocity Verlet:** Use when:
- Energy conservation is critical
- Simulation runs for many periods
- Problem is Hamiltonian (conservative forces only)

### 4.2 The Symplectic Advantage

The most important conceptual finding is the distinction between *accuracy* and *structure preservation*. A higher-order method (RK4) may produce smaller errors at each step, but a symplectic method (Verlet) produces errors that don't accumulate.

This has profound implications for computational physics:
- Molecular dynamics simulations typically run 10⁶ - 10⁹ timesteps
- Non-symplectic energy drift over this time would be catastrophic
- Symplectic methods are therefore standard in this domain

### 4.3 Limitations

This study has several limitations:

1. **Test problems are 1D.** Real physics involves coupled multi-dimensional systems.
2. **No stiff problems.** Stiff ODEs require implicit methods not studied here.
3. **No parallelization.** GPU acceleration could change cost calculations significantly.
4. **Fixed precision.** All computations use 64-bit floating point.

### 4.4 Comparison with scipy

We compared our RK4 implementation with scipy.integrate.solve_ivp's RK45:

| Implementation | Time for SHO (T=100, dt=0.01) |
|---------------|------------------------------|
| Our RK4 | 83 ms |
| scipy RK45 | ~50 ms |

scipy's implementation is faster due to:
- C-level implementation (vs pure Python)
- Optimized memory allocation
- BLAS-level vectorization

This highlights that implementation quality matters as much as algorithm choice for practical performance.

---

## 5. Conclusion

This study provides a systematic characterization of accuracy-cost tradeoffs in numerical integration for physics simulations. The main conclusions are:

1. **Method selection depends on objectives.** There is no universally "best" integrator.

2. **RK4 offers the best accuracy-per-FLOP** for moderate accuracy requirements.

3. **RK45 is optimal for accuracy-constrained problems** where step size can vary.

4. **Symplectic methods are essential for long-time physics** where energy conservation matters more than instantaneous accuracy.

5. **Euler provides a useful baseline** but should never be used for production simulations.

These findings inform integrator selection for computational physics workflows where compute resources and accuracy requirements must be balanced.

---

## 6. Future Work

Several extensions would strengthen this analysis:

1. **GPU scaling study.** Batch integration on GPU could change optimal method selection.
2. **Higher dimensions.** Extend to N-body and PDE problems.
3. **Implicit methods.** Add BDF and implicit RK for stiff problems.
4. **Mixed precision.** Investigate accuracy/cost tradeoffs with FP32.

---

## References

1. Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical Integration*. Springer.
2. Press, W. H., et al. (2007). *Numerical Recipes*. Cambridge University Press.
3. Verlet, L. (1967). Computer "experiments" on classical fluids. *Physical Review*, 159(1), 98.
4. Fehlberg, E. (1969). Low-order classical Runge-Kutta formulas with stepsize control. NASA Technical Report.

---

## Appendix: Reproducibility

All code and data are available at: [repository URL]

To reproduce:
```bash
pip install -r requirements.txt
python run_all.py
```

System specifications:
- CPU: AMD Ryzen 9 9950X
- GPU: NVIDIA RTX 5090
- RAM: 64GB DDR5
- OS: Windows 11
- Python: 3.10+

---

*This report was prepared as part of research into compute-constrained systems at Arizona State University.*
