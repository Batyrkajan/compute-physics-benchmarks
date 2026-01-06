"""
Nonlinear Pendulum (Large Angle)
================================

A pendulum without the small-angle approximation.

Equation of motion:
    d²θ/dt² = -(g/L)·sin(θ)

As a first-order system (y = [θ, ω]):
    dθ/dt = ω
    dω/dt = -(g/L)·sin(θ)

Analytical solution:
    No closed-form solution exists for arbitrary initial angles.
    The period depends on amplitude and involves elliptic integrals.

Energy:
    E = (1/2)L²ω² + gL(1 - cos(θ))  (with m=1)
    Or normalized: E = (1/2)ω² + (g/L)(1 - cos(θ))

This problem is important because:
1. No closed-form solution → tests integrator on "real" problems
2. Nonlinear → small errors can grow
3. Still has energy conservation → tests symplectic properties
4. Large angles reveal differences between methods
"""

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .base import PhysicsProblem, ProblemConfig


class NonlinearPendulum(PhysicsProblem):
    """
    Nonlinear pendulum: d²θ/dt² = -(g/L)·sin(θ)

    A simple pendulum with arbitrary amplitude oscillations.
    Unlike the harmonic oscillator, there's no analytical solution.

    Parameters (in config.params)
    ----------
    g_over_L : float
        Ratio g/L where g is gravity, L is pendulum length
        Default: 1.0 (natural frequency = 1 rad/s for small angles)

    Initial conditions (in config.y0)
    ----------
    y0 = [θ0, ω0]
        θ0: initial angle (radians)
        ω0: initial angular velocity

    Notes
    -----
    For θ0 < 0.1 rad, the pendulum is approximately harmonic.
    For θ0 > 1 rad, nonlinear effects are significant.
    For θ0 → π, the pendulum approaches the separatrix.
    """

    name = "Nonlinear Pendulum"
    has_analytical_solution = False

    def __init__(self, config: ProblemConfig):
        super().__init__(config)
        self.g_over_L = self.params.get('g_over_L', 1.0)

        # Cache reference solution (computed lazily)
        self._reference_solution = None
        self._reference_t = None

    def f(self, t: float, y: NDArray) -> NDArray:
        """
        ODE right-hand side: dy/dt = [ω, -(g/L)·sin(θ)]

        Parameters
        ----------
        t : float
            Current time (unused, system is autonomous)
        y : NDArray
            State [θ, ω]

        Returns
        -------
        dydt : NDArray
            [dθ/dt, dω/dt] = [ω, -(g/L)·sin(θ)]
        """
        theta, omega = y[0], y[1]
        return np.array([omega, -self.g_over_L * np.sin(theta)])

    def energy(self, y: NDArray) -> float:
        """
        Total energy: E = (1/2)ω² + (g/L)(1 - cos(θ))

        Normalized form with m=1, L=1:
        - Kinetic energy: (1/2)ω²
        - Potential energy: (g/L)(1 - cos(θ))
          (zero at θ=0, max at θ=π)

        Parameters
        ----------
        y : NDArray
            State [θ, ω]

        Returns
        -------
        E : float
            Total mechanical energy
        """
        theta, omega = y[0], y[1]
        kinetic = 0.5 * omega**2
        potential = self.g_over_L * (1 - np.cos(theta))
        return kinetic + potential

    def compute_reference_solution(
        self,
        t: NDArray,
        rtol: float = 1e-12,
        atol: float = 1e-14
    ) -> NDArray:
        """
        Compute high-accuracy reference solution using scipy.

        Since no analytical solution exists, we use scipy's
        DOP853 (8th order) with very tight tolerances.

        Parameters
        ----------
        t : NDArray
            Time points
        rtol : float
            Relative tolerance for reference solver
        atol : float
            Absolute tolerance for reference solver

        Returns
        -------
        y : NDArray
            Reference solution at time points
        """
        sol = solve_ivp(
            self.f,
            self.t_span,
            self.y0,
            method='DOP853',  # 8th order, very accurate
            t_eval=t,
            rtol=rtol,
            atol=atol
        )
        return sol.y.T

    def analytical_solution(self, t: NDArray) -> NDArray:
        """
        Returns reference solution (not truly analytical).

        Since no closed-form exists, we compute a high-accuracy
        numerical solution and treat it as "ground truth".
        """
        return self.compute_reference_solution(t)

    @property
    def max_angle(self) -> float:
        """Maximum angle during oscillation (estimated from energy)."""
        E = self.initial_energy()
        # E = (g/L)(1 - cos(θ_max)) when ω = 0
        # cos(θ_max) = 1 - E·L/g
        cos_max = 1 - E / self.g_over_L
        if cos_max < -1:
            return np.pi  # Full rotation possible
        return np.arccos(max(-1, cos_max))

    @property
    def is_oscillating(self) -> bool:
        """Check if pendulum oscillates or rotates."""
        return self.initial_energy() < 2 * self.g_over_L

    @classmethod
    def default_config(cls) -> ProblemConfig:
        """
        Standard test configuration with large initial angle.

        Initial angle: 2.5 rad ≈ 143° (very nonlinear)
        Integration time: 50 (several periods)
        """
        return ProblemConfig(
            y0=np.array([2.5, 0.0]),  # Large angle, released from rest
            t_span=(0.0, 50.0),
            params={'g_over_L': 1.0}
        )

    @classmethod
    def small_angle_config(cls) -> ProblemConfig:
        """
        Small angle configuration (approximately harmonic).

        Initial angle: 0.1 rad ≈ 5.7°
        Should behave nearly like SHO for comparison.
        """
        return ProblemConfig(
            y0=np.array([0.1, 0.0]),
            t_span=(0.0, 50.0),
            params={'g_over_L': 1.0}
        )

    @classmethod
    def near_separatrix_config(cls) -> ProblemConfig:
        """
        Near-separatrix configuration (very challenging).

        Initial angle: 3.1 rad ≈ 178°
        Almost at the separatrix - tests integrator limits.
        """
        return ProblemConfig(
            y0=np.array([3.1, 0.0]),
            t_span=(0.0, 100.0),
            params={'g_over_L': 1.0}
        )
