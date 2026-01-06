"""
Benchmarking Infrastructure
===========================

Tools for measuring integrator performance:
- Accuracy metrics (error vs analytical solution)
- Timing measurements
- Energy conservation tracking
- Benchmark runner orchestration
"""

from .accuracy import compute_errors, compute_energy_drift
from .timing import time_integration, TimingResult
from .runner import BenchmarkRunner, BenchmarkResult

__all__ = [
    'compute_errors',
    'compute_energy_drift',
    'time_integration',
    'TimingResult',
    'BenchmarkRunner',
    'BenchmarkResult'
]
