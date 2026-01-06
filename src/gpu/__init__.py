"""
GPU Acceleration Module
=======================

CuPy/Numba implementations for GPU-accelerated integration.
Used for batch scaling experiments.
"""

from .integrators_gpu import EulerGPU, RK4GPU
from .scaling import run_scaling_experiment

__all__ = ['EulerGPU', 'RK4GPU', 'run_scaling_experiment']
