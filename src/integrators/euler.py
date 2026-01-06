"""
Forward Euler Integrator
========================

The simplest explicit integrator. First-order accurate.

Mathematical formulation:
    y_{n+1} = y_n + dt * f(t_n, y_n)

Properties:
    - Order: 1 (error ~ O(dt))
    - Function evaluations per step: 1
    - Stability: Conditionally stable (dt must be small)
    - Not symplectic

This is the baseline against which other methods are compared.
It's fast but inaccurate, and becomes unstable for stiff problems
or large timesteps.
"""

from typing import Callable
import numpy as np
from numpy.typing import NDArray

from .base import Integrator


class ForwardEuler(Integrator):
    """
    Forward Euler (explicit Euler) integrator.

    The most basic numerical integration method. Uses the derivative
    at the current point to extrapolate to the next point.

    Good for: Understanding, baseline comparisons
    Bad for: Accuracy, stability, energy conservation
    """

    name = "Forward Euler"
    order = 1
    is_symplectic = False

    def step(
        self,
        f: Callable[[float, NDArray], NDArray],
        t: float,
        y: NDArray,
        dt: float
    ) -> NDArray:
        """
        Advance solution by one Euler step.

        y_{n+1} = y_n + dt * f(t_n, y_n)

        Parameters
        ----------
        f : callable
            Right-hand side function f(t, y)
        t : float
            Current time
        y : NDArray
            Current state vector
        dt : float
            Timestep size

        Returns
        -------
        y_new : NDArray
            State at t + dt
        """
        k1 = f(t, y)
        self._n_evaluations += 1

        return y + dt * k1
