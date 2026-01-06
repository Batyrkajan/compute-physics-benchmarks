# Experiment Design: Numerical Integration Methods for Physics Simulations
## Accuracy vs Compute Cost Tradeoff Analysis

---

## 1. RESEARCH QUESTION

**Primary:** How do numerical integration methods trade accuracy vs compute cost for physics simulations?

**Sub-questions:**
1. At what timestep does each method become unstable?
2. What is the accuracy-per-FLOP efficiency of each method?
3. How does GPU parallelization change the cost-accuracy tradeoff?
4. Which method is optimal for different accuracy requirements?

---

## 2. INTEGRATION METHODS (4 methods)

### 2.1 Forward Euler (1st order)
- **Why:** Baseline. Simplest possible integrator.
- **Expected:** Fast but inaccurate, unstable for stiff problems
- **Complexity:** O(n) per step, 1 function evaluation

### 2.2 Runge-Kutta 4 (RK4) (4th order)
- **Why:** Industry workhorse. Gold standard for non-adaptive methods.
- **Expected:** Good accuracy, 4x cost of Euler per step
- **Complexity:** O(n) per step, 4 function evaluations

### 2.3 Runge-Kutta-Fehlberg (RK45) - Adaptive
- **Why:** Demonstrates adaptive stepping concept. Used in scipy.integrate.
- **Expected:** Best accuracy-per-compute for smooth problems
- **Complexity:** 6 function evaluations, but fewer total steps

### 2.4 Velocity Verlet (2nd order, symplectic)
- **Why:** Physics-specific. Conserves energy in Hamiltonian systems.
- **Expected:** Moderate accuracy but excellent long-term stability
- **Complexity:** O(n) per step, 2 function evaluations
- **Key insight:** Sometimes "less accurate" methods are better for physics

---

## 3. TEST PROBLEMS (2 problems)

### 3.1 Simple Harmonic Oscillator
```
d²x/dt² = -ω²x
```
- **Why:** Analytical solution exists (x = A·cos(ωt + φ))
- **Tests:** Basic accuracy, energy conservation, phase drift
- **Parameters:** ω = 1.0, x₀ = 1.0, v₀ = 0.0
- **Integration time:** T = 100 (≈16 periods)

### 3.2 Nonlinear Pendulum (Large Angle)
```
d²θ/dt² = -(g/L)·sin(θ)
```
- **Why:** No closed-form solution, tests nonlinear handling
- **Tests:** Accuracy in nonlinear regime, stability under large oscillations
- **Parameters:** g/L = 1.0, θ₀ = 2.5 rad (≈143°), ω₀ = 0.0
- **Integration time:** T = 50

**Why these two?**
- SHO: Ground truth comparison (we KNOW the answer)
- Pendulum: Real-world physics where analytical solutions fail

---

## 4. METRICS & MEASUREMENTS

### 4.1 Accuracy Metrics

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **Position Error** | \|x_numerical - x_analytical\| | Direct accuracy measure |
| **Energy Drift** | \|E(t) - E(0)\| / E(0) | Physics conservation |
| **Phase Error** | Accumulated timing drift | Long-term reliability |
| **Max Error** | max(\|error\|) over trajectory | Worst-case behavior |
| **RMS Error** | sqrt(mean(error²)) | Average behavior |

### 4.2 Compute Cost Metrics

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **Wall Time** | Actual runtime (ms) | Real-world cost |
| **Function Evaluations** | # of f(x,t) calls | Hardware-independent cost |
| **Steps Taken** | # of integration steps | For adaptive methods |
| **Time per Step** | Wall time / steps | Per-operation efficiency |

### 4.3 Efficiency Metrics (Derived)

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **Accuracy per Second** | 1 / (error × time) | Bang for buck |
| **Pareto Frontier** | Non-dominated (error, time) pairs | Optimal tradeoff curve |

---

## 5. EXPERIMENTAL DESIGN

### 5.1 Timestep Sweep (Main Experiment)

```
Timesteps: dt = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]
```

For each (method, problem, dt):
1. Run integration
2. Measure wall time (average of 10 runs)
3. Compute error vs analytical/reference solution
4. Record energy drift
5. Log function evaluations

### 5.2 Stability Analysis

For SHO:
- Run with increasing dt until method diverges
- Record critical dt for each method
- Plot stability boundary

### 5.3 Long-Time Behavior

For SHO with dt = 0.01:
- Run for T = 1000 (≈160 periods)
- Track energy drift over time
- Compare symplectic (Verlet) vs non-symplectic (RK4)

### 5.4 GPU Scaling Experiment

**Batch Integration:** Integrate N independent oscillators simultaneously

```
N = [1, 10, 100, 1000, 10000, 100000, 1000000]
```

Compare:
- CPU (NumPy): Sequential
- CPU (NumPy): Vectorized
- GPU (CuPy): Vectorized

**Key question:** At what N does GPU become faster?

---

## 6. PLOTS (Publication Quality)

### Plot 1: Error vs Timestep (Log-Log)
- X: timestep (log scale)
- Y: RMS error (log scale)
- Lines: One per method
- **Shows:** Order of accuracy (slope = order)

### Plot 2: Runtime vs Timestep (Log-Log)
- X: timestep (log scale)
- Y: wall time (log scale)
- Lines: One per method
- **Shows:** Computational cost scaling

### Plot 3: Error vs Runtime (Pareto Plot)
- X: wall time (log scale)
- Y: RMS error (log scale)
- Points: Each (method, dt) combination
- **Shows:** Accuracy-cost tradeoff, Pareto frontier

### Plot 4: Energy Drift Over Time
- X: time
- Y: relative energy error
- Lines: One per method (fixed dt = 0.01)
- **Shows:** Symplectic advantage

### Plot 5: Stability Regions
- Heatmap or boundary plot
- X: dt
- Y: method
- Color: stable/unstable
- **Shows:** Practical stability limits

### Plot 6: GPU Scaling
- X: batch size N (log scale)
- Y: wall time (log scale)
- Lines: CPU-seq, CPU-vec, GPU
- **Shows:** GPU crossover point

### Plot 7: Phase Portrait (Bonus)
- X: position
- Y: velocity
- Trajectories for each method
- **Shows:** Qualitative behavior, energy conservation visually

---

## 7. CODE ARCHITECTURE

```
compute-physics-benchmarks/
├── src/
│   ├── __init__.py
│   ├── integrators/
│   │   ├── __init__.py
│   │   ├── euler.py          # Forward Euler
│   │   ├── rk4.py            # Runge-Kutta 4
│   │   ├── rk45.py           # Adaptive RK45
│   │   ├── verlet.py         # Velocity Verlet
│   │   └── base.py           # Abstract base class
│   ├── problems/
│   │   ├── __init__.py
│   │   ├── harmonic.py       # Simple harmonic oscillator
│   │   ├── pendulum.py       # Nonlinear pendulum
│   │   └── base.py           # Abstract base class
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── accuracy.py       # Error computation
│   │   ├── timing.py         # Runtime measurement
│   │   └── runner.py         # Main benchmark orchestrator
│   └── gpu/
│       ├── __init__.py
│       ├── integrators_gpu.py  # CuPy implementations
│       └── scaling.py          # Batch scaling experiments
├── experiments/
│   ├── run_accuracy_sweep.py
│   ├── run_stability_analysis.py
│   ├── run_longterm.py
│   └── run_gpu_scaling.py
├── results/
│   ├── data/                 # Raw CSV/JSON results
│   └── figures/              # Generated plots
├── writeup/
│   └── technical_report.md   # The document
├── requirements.txt
├── README.md
└── run_all.py                # Master script
```

---

## 8. TECHNICAL SPECIFICATIONS

### Dependencies
```
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0        # For reference solutions
cupy>=12.0.0         # GPU compute (optional)
numba>=0.57.0        # JIT compilation (optional)
pandas>=2.0.0        # Data handling
```

### Design Principles
1. **Reproducibility:** Fixed random seeds, version pinning
2. **Modularity:** Each integrator is independent, testable
3. **Documentation:** Every function has docstring with math
4. **Type hints:** Full typing for clarity
5. **Benchmarking rigor:** Multiple runs, proper timing isolation

---

## 9. EXPECTED RESULTS (Hypotheses)

### Accuracy vs Timestep
- Euler: slope ≈ 1 (1st order)
- RK4: slope ≈ 4 (4th order)
- Verlet: slope ≈ 2 (2nd order)
- RK45: adaptive, will cluster at target tolerance

### Pareto Frontier
- RK45 dominates for high accuracy requirements
- RK4 dominates for moderate accuracy
- Verlet wins for long-time physics (energy conservation)
- Euler never optimal (but useful as baseline)

### GPU Scaling
- Crossover at N ≈ 1000-10000 particles
- GPU speedup ≈ 10-100x for large N on RTX 5090

### Key Insight to Highlight
**"The best integrator depends on what you're optimizing for."**
- Accuracy? → RK45
- Speed at fixed accuracy? → RK4
- Long-term physics fidelity? → Verlet
- This is the systems thinking that PhD programs want to see.

---

## 10. WRITEUP STRUCTURE (4-6 pages)

1. **Abstract** (100 words)
2. **Introduction** (0.5 page)
   - Why numerical integration matters
   - The accuracy-cost tradeoff
3. **Methods** (1 page)
   - Integration algorithms (math)
   - Test problems
   - Experimental setup
4. **Results** (2 pages)
   - Plots with interpretation
   - Key findings
5. **Discussion** (1 page)
   - When to use which method
   - GPU scaling implications
   - Limitations
6. **Conclusion** (0.25 page)
7. **References**

---

## 11. TIMELINE

| Days | Task |
|------|------|
| 1-2 | Project setup, implement integrators |
| 3-4 | Implement problems, basic benchmarking |
| 5-6 | Run experiments, collect data |
| 7-8 | GPU implementation and scaling |
| 9-10 | Generate plots, analyze results |
| 11-12 | Write technical report |
| 13-14 | Polish, package for GitHub |

---

## 12. SUCCESS CRITERIA

The artifact is "done" when:

- [ ] All 4 integrators implemented and tested
- [ ] Both test problems working
- [ ] All 7 plots generated
- [ ] GPU comparison completed
- [ ] Technical writeup finished
- [ ] README explains how to reproduce
- [ ] Can explain any plot in 60 seconds
- [ ] Ready to attach to professor email

---

## APPROVAL CHECKLIST

Before building, confirm:

1. **Scope:** Is this the right scope? (Not too big, not too small)
2. **Methods:** Are these 4 integrators the right choices?
3. **Problems:** SHO + Pendulum sufficient?
4. **Plots:** Are these 7 plots what you want?
5. **GPU:** Include GPU comparison or defer?
6. **Timeline:** 12-14 days realistic?

**Your call. Approve and I start building.**
