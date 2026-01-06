"""
Quick Test Script
=================

Verifies all components work correctly before running full experiments.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def test_integrators():
    """Test all integrators on simple problem."""
    print("Testing integrators...")

    from src.integrators import ForwardEuler, RK4, RK45, VelocityVerlet
    from src.problems import HarmonicOscillator

    # Simple config
    config = HarmonicOscillator.default_config()
    config.t_span = (0, 10)  # Short run
    problem = HarmonicOscillator(config)

    dt = 0.01
    integrators = [ForwardEuler(), RK4(), RK45(), VelocityVerlet()]

    for integrator in integrators:
        result = integrator.integrate(problem.f, problem.y0, problem.t_span, dt)

        # Check result is valid
        assert np.all(np.isfinite(result.y)), f"{integrator.name} produced NaN/inf"

        # Get analytical solution
        y_exact = problem.analytical_solution(result.t)

        # Compute error
        error = np.max(np.abs(result.y - y_exact))

        print(f"  {integrator.name:20s}: max_error={error:.2e}, "
              f"steps={result.n_steps}, evals={result.n_evaluations}")

    print("  All integrators OK!")


def test_problems():
    """Test all physics problems."""
    print("\nTesting problems...")

    from src.problems import HarmonicOscillator, NonlinearPendulum

    # Harmonic oscillator
    sho = HarmonicOscillator(HarmonicOscillator.default_config())
    E0 = sho.initial_energy()
    y_test = np.array([0.5, 0.5])
    E1 = sho.energy(y_test)
    print(f"  HarmonicOscillator: E0={E0:.4f}, E(test)={E1:.4f}")

    # Test analytical solution
    t_test = np.array([0, 1, 2])
    y_anal = sho.analytical_solution(t_test)
    assert y_anal.shape == (3, 2), "Wrong shape for analytical solution"
    print(f"  HarmonicOscillator analytical solution OK")

    # Nonlinear pendulum
    pend = NonlinearPendulum(NonlinearPendulum.default_config())
    E0 = pend.initial_energy()
    print(f"  NonlinearPendulum: E0={E0:.4f}, max_angle={pend.max_angle:.4f} rad")
    print("  All problems OK!")


def test_benchmarks():
    """Test benchmark infrastructure."""
    print("\nTesting benchmarks...")

    from src.integrators import RK4
    from src.problems import HarmonicOscillator
    from src.benchmarks import compute_errors, compute_energy_drift, time_integration

    # Setup
    problem = HarmonicOscillator(HarmonicOscillator.default_config())
    problem.config.t_span = (0, 10)
    integrator = RK4()
    dt = 0.01

    # Run integration
    result = integrator.integrate(problem.f, problem.y0, (0, 10), dt)
    y_exact = problem.analytical_solution(result.t)

    # Test accuracy metrics
    accuracy = compute_errors(result.y, y_exact)
    print(f"  Accuracy: rms={accuracy.rms_error:.2e}, max={accuracy.max_error:.2e}")

    # Test energy metrics
    energy = compute_energy_drift(result.y, problem.energy)
    print(f"  Energy: drift={energy.relative_drift:.2e}")

    # Test timing
    timing = time_integration(integrator, problem.f, problem.y0, (0, 10), dt, n_runs=3)
    print(f"  Timing: {timing.mean_time*1000:.2f} +/- {timing.std_time*1000:.2f} ms")

    print("  All benchmarks OK!")


def main():
    print("="*60)
    print("RUNNING BASIC TESTS")
    print("="*60)

    try:
        test_integrators()
        test_problems()
        test_benchmarks()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        return True

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
