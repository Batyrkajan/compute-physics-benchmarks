"""
Timing Infrastructure
=====================

Precise timing measurements for benchmarking integrator performance.

Uses multiple runs and statistical analysis to get reliable timings.
"""

import time
from dataclasses import dataclass
from typing import Callable, List
import numpy as np
from numpy.typing import NDArray

from ..integrators.base import Integrator, IntegrationResult


@dataclass
class TimingResult:
    """
    Result of timing measurement.

    Attributes
    ----------
    mean_time : float
        Mean execution time (seconds)
    std_time : float
        Standard deviation of execution time
    min_time : float
        Minimum execution time
    max_time : float
        Maximum execution time
    n_runs : int
        Number of timing runs
    times : List[float]
        Individual run times
    """
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    n_runs: int
    times: List[float]


def time_integration(
    integrator: Integrator,
    f: Callable,
    y0: NDArray,
    t_span: tuple,
    dt: float,
    n_runs: int = 10,
    warmup: int = 2
) -> TimingResult:
    """
    Time an integration with multiple runs.

    Parameters
    ----------
    integrator : Integrator
        The integrator to benchmark
    f : callable
        ODE right-hand side
    y0 : NDArray
        Initial conditions
    t_span : tuple
        (t_start, t_end)
    dt : float
        Timestep
    n_runs : int
        Number of timing runs (default: 10)
    warmup : int
        Number of warmup runs before timing (default: 2)

    Returns
    -------
    result : TimingResult
        Timing statistics
    """
    # Warmup runs (not timed)
    for _ in range(warmup):
        integrator.integrate(f, y0.copy(), t_span, dt)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        integrator.integrate(f, y0.copy(), t_span, dt)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    return TimingResult(
        mean_time=np.mean(times),
        std_time=np.std(times),
        min_time=np.min(times),
        max_time=np.max(times),
        n_runs=n_runs,
        times=list(times)
    )


def time_single_run(
    integrator: Integrator,
    f: Callable,
    y0: NDArray,
    t_span: tuple,
    dt: float
) -> tuple:
    """
    Single timed integration run.

    Parameters
    ----------
    integrator : Integrator
        The integrator to use
    f : callable
        ODE right-hand side
    y0 : NDArray
        Initial conditions
    t_span : tuple
        (t_start, t_end)
    dt : float
        Timestep

    Returns
    -------
    result : IntegrationResult
        Integration result
    elapsed : float
        Elapsed time in seconds
    """
    start = time.perf_counter()
    result = integrator.integrate(f, y0.copy(), t_span, dt)
    end = time.perf_counter()

    return result, end - start


class Timer:
    """
    Context manager for timing code blocks.

    Usage
    -----
    with Timer() as t:
        # code to time
    print(f"Elapsed: {t.elapsed:.3f}s")
    """

    def __init__(self):
        self.elapsed = 0.0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def estimate_optimal_runs(
    integrator: Integrator,
    f: Callable,
    y0: NDArray,
    t_span: tuple,
    dt: float,
    target_time: float = 1.0,
    min_runs: int = 3,
    max_runs: int = 100
) -> int:
    """
    Estimate optimal number of runs for reliable timing.

    Aims for total timing time around target_time seconds.

    Parameters
    ----------
    integrator : Integrator
        The integrator to benchmark
    f : callable
        ODE right-hand side
    y0 : NDArray
        Initial conditions
    t_span : tuple
        (t_start, t_end)
    dt : float
        Timestep
    target_time : float
        Target total timing time (default: 1.0)
    min_runs : int
        Minimum runs (default: 3)
    max_runs : int
        Maximum runs (default: 100)

    Returns
    -------
    n_runs : int
        Recommended number of runs
    """
    # Single test run
    _, single_time = time_single_run(integrator, f, y0, t_span, dt)

    # Estimate runs needed
    estimated_runs = int(target_time / max(single_time, 1e-6))

    return max(min_runs, min(estimated_runs, max_runs))
