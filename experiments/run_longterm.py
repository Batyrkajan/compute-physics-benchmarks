"""
Long-Term Integration Experiment
================================

Studies energy conservation over extended integration times.

This experiment demonstrates why symplectic integrators (Verlet)
are preferred for long-time physics simulations despite having
lower order accuracy than RK4.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
from src.integrators import ForwardEuler, RK4, RK45, VelocityVerlet
from src.problems import HarmonicOscillator
from src.benchmarks.accuracy import compute_energy_drift


def run_longterm_experiment():
    """Run long-term energy drift experiment."""

    print("="*70)
    print("LONG-TERM INTEGRATION EXPERIMENT")
    print("="*70)

    # Use harmonic oscillator with long integration time
    problem = HarmonicOscillator(HarmonicOscillator.long_time_config())

    # Fixed timestep for fair comparison
    dt = 0.01

    integrators = [
        ForwardEuler(),
        RK4(),
        RK45(rtol=1e-8, atol=1e-10),
        VelocityVerlet()
    ]

    results = []
    energy_histories = {}

    print(f"\nIntegrating for T={problem.t_span[1]} with dt={dt}")
    print("-" * 50)

    for integrator in integrators:
        print(f"  {integrator.name}...", end=" ")

        try:
            result = integrator.integrate(
                problem.f,
                problem.y0,
                problem.t_span,
                dt
            )

            if not np.all(np.isfinite(result.y)):
                print("UNSTABLE")
                continue

            # Compute energy metrics
            energy_metrics = compute_energy_drift(result.y, problem.energy)

            # Store for plotting
            energy_histories[integrator.name] = {
                't': result.t.tolist(),
                'energy': energy_metrics.energy_history.tolist(),
                'initial': energy_metrics.initial_energy
            }

            results.append({
                'integrator': integrator.name,
                'order': integrator.order,
                'symplectic': integrator.is_symplectic,
                'initial_energy': energy_metrics.initial_energy,
                'final_energy': energy_metrics.final_energy,
                'max_drift': energy_metrics.max_drift,
                'relative_drift': energy_metrics.relative_drift,
                'n_steps': result.n_steps,
                'n_evaluations': result.n_evaluations
            })

            print(f"relative drift = {energy_metrics.relative_drift:.2e}")

        except Exception as e:
            print(f"ERROR: {e}")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'longterm_energy.csv', index=False)

    with open(output_dir / 'longterm_energy.json', 'w') as f:
        json.dump({
            'summary': results,
            'histories': energy_histories,
            'config': {
                'problem': 'Harmonic Oscillator',
                'T': problem.t_span[1],
                'dt': dt,
                'omega': problem.omega
            }
        }, f, indent=2)

    print("\n" + "="*70)
    print("LONG-TERM ENERGY CONSERVATION SUMMARY")
    print("="*70)
    print(df.to_string(index=False))

    print("\n KEY INSIGHT:")
    print("-" * 50)
    print("Notice how Velocity Verlet maintains bounded energy drift")
    print("while RK4's energy error grows over time. This is the")
    print("symplectic advantage: geometric structure preservation.")

    return results, energy_histories


if __name__ == '__main__':
    run_longterm_experiment()
