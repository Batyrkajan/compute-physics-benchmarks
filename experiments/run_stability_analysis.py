"""
Stability Analysis Experiment
=============================

Determines the stability boundaries for each integrator.

For each method, finds the maximum timestep that produces
stable (non-diverging) results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
from src.integrators import ForwardEuler, RK4, RK45, VelocityVerlet
from src.problems import HarmonicOscillator, NonlinearPendulum


def check_stability(integrator, problem, dt, threshold=1e6):
    """
    Check if integration is stable at given timestep.

    Returns True if stable, False if divergent.
    """
    try:
        result = integrator.integrate(
            problem.f,
            problem.y0,
            problem.t_span,
            dt
        )

        # Check for NaN/inf
        if not np.all(np.isfinite(result.y)):
            return False

        # Check for explosion
        max_val = np.max(np.abs(result.y))
        if max_val > threshold:
            return False

        return True

    except Exception:
        return False


def find_stability_boundary(integrator, problem, dt_range, n_points=50):
    """
    Binary search for stability boundary.

    Returns the maximum stable timestep.
    """
    dt_min, dt_max = dt_range

    # First check if any dt is stable
    if not check_stability(integrator, problem, dt_min):
        return 0.0

    # Binary search
    for _ in range(n_points):
        dt_mid = (dt_min + dt_max) / 2

        if check_stability(integrator, problem, dt_mid):
            dt_min = dt_mid
        else:
            dt_max = dt_mid

    return dt_min


def run_stability_analysis():
    """Run stability analysis for all integrators."""

    print("="*70)
    print("STABILITY ANALYSIS EXPERIMENT")
    print("="*70)

    integrators = [
        ForwardEuler(),
        RK4(),
        RK45(rtol=1e-6, atol=1e-9),
        VelocityVerlet()
    ]

    problems = [
        ('Harmonic Oscillator', HarmonicOscillator(HarmonicOscillator.default_config())),
        ('Nonlinear Pendulum', NonlinearPendulum(NonlinearPendulum.default_config())),
    ]

    results = []

    for problem_name, problem in problems:
        print(f"\n{problem_name}:")
        print("-" * 40)

        for integrator in integrators:
            print(f"  {integrator.name}...", end=" ")

            # Different search ranges based on expected stability
            if integrator.name == "Forward Euler":
                dt_range = (0.001, 3.0)
            else:
                dt_range = (0.001, 5.0)

            max_dt = find_stability_boundary(integrator, problem, dt_range)

            print(f"max stable dt = {max_dt:.4f}")

            results.append({
                'problem': problem_name,
                'integrator': integrator.name,
                'max_stable_dt': max_dt,
                'order': integrator.order,
                'symplectic': integrator.is_symplectic
            })

    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'stability_analysis.csv', index=False)

    with open(output_dir / 'stability_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("STABILITY SUMMARY")
    print("="*70)
    print(df.to_string(index=False))

    return results


if __name__ == '__main__':
    run_stability_analysis()
