"""
Benchmark Runner
================

Orchestrates benchmark experiments across integrators, problems, and timesteps.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pathlib import Path

from ..integrators.base import Integrator, IntegrationResult
from ..problems.base import PhysicsProblem
from .accuracy import compute_errors, compute_energy_drift, AccuracyMetrics, EnergyMetrics
from .timing import time_integration, TimingResult


@dataclass
class BenchmarkResult:
    """
    Complete result of a single benchmark run.

    Attributes
    ----------
    integrator_name : str
        Name of the integrator
    problem_name : str
        Name of the test problem
    dt : float
        Timestep used
    timing : TimingResult
        Timing measurements
    accuracy : AccuracyMetrics
        Accuracy metrics
    energy : EnergyMetrics
        Energy conservation metrics
    n_steps : int
        Number of integration steps
    n_evaluations : int
        Number of function evaluations
    """
    integrator_name: str
    problem_name: str
    dt: float
    timing: TimingResult
    accuracy: AccuracyMetrics
    energy: EnergyMetrics
    n_steps: int
    n_evaluations: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'integrator': self.integrator_name,
            'problem': self.problem_name,
            'dt': self.dt,
            'mean_time': self.timing.mean_time,
            'std_time': self.timing.std_time,
            'rms_error': self.accuracy.rms_error,
            'max_error': self.accuracy.max_error,
            'position_rms': self.accuracy.position_rms,
            'velocity_rms': self.accuracy.velocity_rms,
            'energy_drift': self.energy.relative_drift,
            'max_energy_drift': self.energy.max_drift,
            'n_steps': self.n_steps,
            'n_evaluations': self.n_evaluations
        }


class BenchmarkRunner:
    """
    Orchestrates benchmark experiments.

    Parameters
    ----------
    integrators : List[Integrator]
        Integrators to benchmark
    problems : List[PhysicsProblem]
        Test problems
    timesteps : List[float]
        Timesteps to test
    n_timing_runs : int
        Number of timing runs per configuration
    output_dir : Path
        Directory for saving results

    Usage
    -----
    runner = BenchmarkRunner(
        integrators=[ForwardEuler(), RK4(), RK45(), VelocityVerlet()],
        problems=[HarmonicOscillator.default_config(), NonlinearPendulum.default_config()],
        timesteps=[0.1, 0.05, 0.01, 0.005, 0.001]
    )
    results = runner.run_all()
    runner.save_results(results)
    """

    def __init__(
        self,
        integrators: List[Integrator],
        problems: List[PhysicsProblem],
        timesteps: List[float],
        n_timing_runs: int = 10,
        output_dir: Optional[Path] = None
    ):
        self.integrators = integrators
        self.problems = problems
        self.timesteps = sorted(timesteps, reverse=True)  # Largest first
        self.n_timing_runs = n_timing_runs
        self.output_dir = output_dir or Path('results/data')

    def run_single(
        self,
        integrator: Integrator,
        problem: PhysicsProblem,
        dt: float
    ) -> Optional[BenchmarkResult]:
        """
        Run a single benchmark configuration.

        Returns None if the integration fails (e.g., instability).
        """
        try:
            # Timing runs
            timing = time_integration(
                integrator,
                problem.f,
                problem.y0,
                problem.t_span,
                dt,
                n_runs=self.n_timing_runs
            )

            # One more run to get the solution for accuracy
            result = integrator.integrate(
                problem.f,
                problem.y0,
                problem.t_span,
                dt
            )

            # Check for instability (NaN or inf)
            if not np.all(np.isfinite(result.y)):
                print(f"  UNSTABLE: {integrator.name} dt={dt}")
                return None

            # Reference solution
            if problem.has_analytical_solution:
                y_ref = problem.analytical_solution(result.t)
            else:
                y_ref = problem.compute_reference_solution(result.t)

            # Compute metrics
            accuracy = compute_errors(result.y, y_ref)
            energy = compute_energy_drift(result.y, problem.energy)

            return BenchmarkResult(
                integrator_name=integrator.name,
                problem_name=problem.name,
                dt=dt,
                timing=timing,
                accuracy=accuracy,
                energy=energy,
                n_steps=result.n_steps,
                n_evaluations=result.n_evaluations
            )

        except Exception as e:
            print(f"  ERROR: {integrator.name} dt={dt}: {e}")
            return None

    def run_all(self, verbose: bool = True) -> List[BenchmarkResult]:
        """
        Run all benchmark configurations.

        Parameters
        ----------
        verbose : bool
            Print progress (default: True)

        Returns
        -------
        results : List[BenchmarkResult]
            All successful benchmark results
        """
        results = []
        total = len(self.integrators) * len(self.problems) * len(self.timesteps)
        current = 0

        for problem in self.problems:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Problem: {problem.name}")
                print(f"{'='*60}")

            for integrator in self.integrators:
                if verbose:
                    print(f"\n  Integrator: {integrator.name}")

                for dt in self.timesteps:
                    current += 1
                    if verbose:
                        print(f"    dt={dt:.6f} ({current}/{total})...", end=" ")

                    result = self.run_single(integrator, problem, dt)

                    if result is not None:
                        results.append(result)
                        if verbose:
                            print(f"RMS={result.accuracy.rms_error:.2e}, "
                                  f"time={result.timing.mean_time*1000:.2f}ms")
                    else:
                        if verbose:
                            print("FAILED")

        return results

    def results_to_dataframe(self, results: List[BenchmarkResult]) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        records = [r.to_dict() for r in results]
        return pd.DataFrame(records)

    def save_results(
        self,
        results: List[BenchmarkResult],
        filename: str = "benchmark_results"
    ):
        """
        Save results to CSV and JSON.

        Parameters
        ----------
        results : List[BenchmarkResult]
            Benchmark results
        filename : str
            Base filename (without extension)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        df = self.results_to_dataframe(results)
        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # Save as JSON (more complete)
        json_data = [r.to_dict() for r in results]
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved: {json_path}")


def create_standard_benchmark() -> BenchmarkRunner:
    """
    Create benchmark runner with standard configuration.

    Returns a fully configured runner ready to execute.
    """
    from ..integrators import ForwardEuler, RK4, RK45, VelocityVerlet
    from ..problems import HarmonicOscillator, NonlinearPendulum

    integrators = [
        ForwardEuler(),
        RK4(),
        RK45(rtol=1e-6, atol=1e-9),
        VelocityVerlet()
    ]

    problems = [
        HarmonicOscillator(HarmonicOscillator.default_config()),
        NonlinearPendulum(NonlinearPendulum.default_config())
    ]

    timesteps = [
        0.5, 0.2, 0.1, 0.05, 0.02, 0.01,
        0.005, 0.002, 0.001, 0.0005, 0.0001
    ]

    return BenchmarkRunner(
        integrators=integrators,
        problems=problems,
        timesteps=timesteps
    )
