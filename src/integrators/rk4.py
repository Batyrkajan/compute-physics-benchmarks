"""
Classical Runge-Kutta 4th Order Integrator (RK4)
================================================

The "gold standard" fixed-step integrator. Fourth-order accurate.

Mathematical formulation:
    k1 = f(t_n, y_n)
    k2 = f(t_n + dt/2, y_n + dt*k1/2)
    k3 = f(t_n + dt/2, y_n + dt*k2/2)
    k4 = f(t_n + dt, y_n + dt*k3)

    y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

Properties:
    - Order: 4 (error ~ O(dt^4))
    - Function evaluations per step: 4
    - Stability: Better than Euler, but still conditionally stable
    - Not symplectic

This is the workhorse of numerical integration. Good balance of
accuracy and simplicity. Used when you need reliable results
without adaptive stepping complexity.
"""

from typing import Callable
import numpy as np
from numpy.typing import NDArray

from .base import Integrator


class RK4(Integrator):
    """
    Classical 4th-order Runge-Kutta integrator.

    Evaluates the derivative at multiple points within each timestep
    and combines them for 4th-order accuracy. The most commonly used
    integrator in physics and engineering.

    Good for: Most problems, good accuracy/cost balance
    Bad for: Stiff problems, long-time energy conservation
    """

    name = "RK4"
    order = 4
    is_symplectic = False

    def step(
        self,
        f: Callable[[float, NDArray], NDArray],
        t: float,
        y: NDArray,
        dt: float
    ) -> NDArray:
        """
        Advance solution by one RK4 step.

        Uses 4 function evaluations to achieve 4th-order accuracy.
        The weights (1/6, 2/6, 2/6, 1/6) are derived from Taylor
        series matching conditions.

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
        # Stage 1: Evaluate at current point
        k1 = f(t, y)

        # Stage 2: Evaluate at midpoint using k1
        k2 = f(t + dt / 2, y + dt * k1 / 2)

        # Stage 3: Evaluate at midpoint using k2
        k3 = f(t + dt / 2, y + dt * k2 / 2)

        # Stage 4: Evaluate at endpoint using k3
        k4 = f(t + dt, y + dt * k3)

        self._n_evaluations += 4

        # Weighted combination
        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
