"""
Simple Harmonic Oscillator
==========================

The simplest oscillating system with an exact analytical solution.

Equation of motion:
    d²x/dt² = -ω²x

As a first-order system (y = [x, v]):
    dx/dt = v
    dv/dt = -ω²x

Analytical solution:
    x(t) = A·cos(ωt + φ)
    v(t) = -Aω·sin(ωt + φ)

where A and φ are determined by initial conditions.

Energy:
    E = (1/2)mv² + (1/2)kx² = (1/2)v² + (1/2)ω²x²  (with m=1)

This problem is ideal for testing integrators because:
1. We know the exact answer
2. Energy should be conserved
3. Periodic motion reveals phase drift
4. Stability issues are easy to detect
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from .base import PhysicsProblem, ProblemConfig


class HarmonicOscillator(PhysicsProblem):
    """
    Simple harmonic oscillator: d²x/dt² = -ω²x

    A mass on a spring (or equivalently, a small-angle pendulum).
    The fundamental test problem for any numerical integrator.

    Parameters (in config.params)
    ----------
    omega : float
        Angular frequency (default: 1.0)
        Period T = 2π/ω

    Initial conditions (in config.y0)
    ----------
    y0 = [x0, v0]
        x0: initial position
        v0: initial velocity
    """

    name = "Simple Harmonic Oscillator"
    has_analytical_solution = True

    def __init__(self, config: ProblemConfig):
        super().__init__(config)
        self.omega = self.params.get('omega', 1.0)

        # Compute amplitude and phase from initial conditions
        x0 = self.y0[0]
        v0 = self.y0[1]
        self.amplitude = np.sqrt(x0**2 + (v0 / self.omega)**2)
        self.phase = np.arctan2(-v0 / self.omega, x0)

    def f(self, t: float, y: NDArray) -> NDArray:
        """
        ODE right-hand side: dy/dt = [v, -ω²x]

        Parameters
        ----------
        t : float
            Current time (unused, system is autonomous)
        y : NDArray
            State [x, v]

        Returns
        -------
        dydt : NDArray
            [dx/dt, dv/dt] = [v, -ω²x]
        """
        x, v = y[0], y[1]
        return np.array([v, -self.omega**2 * x])

    def energy(self, y: NDArray) -> float:
        """
        Total energy: E = (1/2)v² + (1/2)ω²x²

        For a unit mass system, this is:
        - Kinetic energy: (1/2)v²
        - Potential energy: (1/2)ω²x² = (1/2)kx² with k=ω²

        Parameters
        ----------
        y : NDArray
            State [x, v]

        Returns
        -------
        E : float
            Total mechanical energy
        """
        x, v = y[0], y[1]
        kinetic = 0.5 * v**2
        potential = 0.5 * self.omega**2 * x**2
        return kinetic + potential

    def analytical_solution(self, t: NDArray) -> NDArray:
        """
        Exact solution: x(t) = A·cos(ωt + φ), v(t) = -Aω·sin(ωt + φ)

        Parameters
        ----------
        t : NDArray
            Time points

        Returns
        -------
        y : NDArray
            Shape (len(t), 2) with [x, v] at each time
        """
        x = self.amplitude * np.cos(self.omega * t + self.phase)
        v = -self.amplitude * self.omega * np.sin(self.omega * t + self.phase)
        return np.column_stack([x, v])

    @property
    def period(self) -> float:
        """Oscillation period T = 2π/ω."""
        return 2 * np.pi / self.omega

    @classmethod
    def default_config(cls) -> ProblemConfig:
        """
        Standard test configuration.

        Initial conditions: x0=1, v0=0 (released from rest)
        Angular frequency: ω=1 (period = 2π ≈ 6.28)
        Integration time: 100 (about 16 periods)
        """
        return ProblemConfig(
            y0=np.array([1.0, 0.0]),
            t_span=(0.0, 100.0),
            params={'omega': 1.0}
        )

    @classmethod
    def long_time_config(cls) -> ProblemConfig:
        """
        Long-time configuration for energy drift studies.

        Integration time: 1000 (about 160 periods)
        """
        return ProblemConfig(
            y0=np.array([1.0, 0.0]),
            t_span=(0.0, 1000.0),
            params={'omega': 1.0}
        )
