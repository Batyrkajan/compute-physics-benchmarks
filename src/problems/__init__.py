"""
Physics Test Problems
=====================

Test problems with known analytical solutions (where available)
for benchmarking numerical integrators.

Available problems:
- HarmonicOscillator: Simple harmonic motion with exact solution
- NonlinearPendulum: Large-angle pendulum (no closed-form solution)
"""

from .base import PhysicsProblem, ProblemConfig
from .harmonic import HarmonicOscillator
from .pendulum import NonlinearPendulum

__all__ = ['PhysicsProblem', 'ProblemConfig', 'HarmonicOscillator', 'NonlinearPendulum']
