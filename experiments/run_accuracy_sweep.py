"""
Accuracy Sweep Experiment
=========================

Sweeps through timesteps to measure accuracy vs compute cost.

This is the main experiment that produces:
- Error vs timestep plots (showing order of accuracy)
- Runtime vs timestep plots
- Pareto frontier (accuracy vs cost tradeoff)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.integrators import ForwardEuler, RK4, RK45, VelocityVerlet
from src.problems import HarmonicOscillator, NonlinearPendulum
from src.benchmarks import BenchmarkRunner


def run_accuracy_sweep():
    """Run the main accuracy vs timestep sweep."""

    print("="*70)
    print("ACCURACY SWEEP EXPERIMENT")
    print("="*70)

    # Configure integrators
    integrators = [
        ForwardEuler(),
        RK4(),
        RK45(rtol=1e-8, atol=1e-10),
        VelocityVerlet()
    ]

    # Configure problems
    problems = [
        HarmonicOscillator(HarmonicOscillator.default_config()),
        NonlinearPendulum(NonlinearPendulum.default_config())
    ]

    # Timestep range
    timesteps = [
        0.5, 0.2, 0.1, 0.05, 0.02, 0.01,
        0.005, 0.002, 0.001, 0.0005, 0.0001
    ]

    # Create runner
    output_dir = Path(__file__).parent.parent / 'results' / 'data'
    runner = BenchmarkRunner(
        integrators=integrators,
        problems=problems,
        timesteps=timesteps,
        n_timing_runs=10,
        output_dir=output_dir
    )

    # Run benchmarks
    results = runner.run_all(verbose=True)

    # Save results
    runner.save_results(results, filename='accuracy_sweep')

    # Print summary
    df = runner.results_to_dataframe(results)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for problem_name in df['problem'].unique():
        print(f"\n{problem_name}:")
        problem_df = df[df['problem'] == problem_name]

        for integrator in df['integrator'].unique():
            int_df = problem_df[problem_df['integrator'] == integrator]
            if len(int_df) > 0:
                best = int_df.loc[int_df['rms_error'].idxmin()]
                print(f"  {integrator}:")
                print(f"    Best RMS error: {best['rms_error']:.2e} at dt={best['dt']:.4f}")
                print(f"    Time: {best['mean_time']*1000:.2f} ms")

    return results


if __name__ == '__main__':
    run_accuracy_sweep()
