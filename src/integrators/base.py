"""
Base Integrator Class
=====================

Abstract base class defining the interface for all numerical integrators.

All integrators solve first-order ODE systems of the form:
    dy/dt = f(t, y)

where y can be a vector (e.g., [position, velocity] for mechanics problems).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class IntegrationResult:
    """
    Result of numerical integration.

    Attributes
    ----------
    t : NDArray
        Time points at which solution is computed
    y : NDArray
        Solution values, shape (n_steps, n_vars)
    n_steps : int
        Number of integration steps taken
    n_evaluations : int
        Number of function evaluations (f calls)
    """
    t: NDArray[np.float64]
    y: NDArray[np.float64]
    n_steps: int
    n_evaluations: int

    @property
    def positions(self) -> NDArray[np.float64]:
        """Extract position (first half of state vector)."""
        n_vars = self.y.shape[1]
        return self.y[:, :n_vars // 2]

    @property
    def velocities(self) -> NDArray[np.float64]:
        """Extract velocity (second half of state vector)."""
        n_vars = self.y.shape[1]
        return self.y[:, n_vars // 2:]


class Integrator(ABC):
    """
    Abstract base class for numerical integrators.

    Subclasses must implement the `step` method which advances
    the solution by one timestep.

    Attributes
    ----------
    name : str
        Human-readable name of the integrator
    order : int
        Order of accuracy (error ~ O(dt^order))
    is_symplectic : bool
        Whether the integrator preserves phase space volume
    """

    name: str = "BaseIntegrator"
    order: int = 0
    is_symplectic: bool = False

    def __init__(self):
        self._n_evaluations = 0

    @abstractmethod
    def step(
        self,
        f: Callable[[float, NDArray], NDArray],
        t: float,
        y: NDArray,
        dt: float
    ) -> NDArray:
        """
        Advance solution by one timestep.

        Parameters
        ----------
        f : callable
            Right-hand side function f(t, y) returning dy/dt
        t : float
            Current time
        y : NDArray
            Current state vector
        dt : float
            Timestep size

        Returns
        -------
        y_new : NDArray
            State vector at time t + dt
        """
        pass

    def integrate(
        self,
        f: Callable[[float, NDArray], NDArray],
        y0: NDArray,
        t_span: Tuple[float, float],
        dt: float
    ) -> IntegrationResult:
        """
        Integrate ODE system over a time interval.

        Parameters
        ----------
        f : callable
            Right-hand side function f(t, y)
        y0 : NDArray
            Initial state vector
        t_span : tuple
            (t_start, t_end) time interval
        dt : float
            Timestep size

        Returns
        -------
        result : IntegrationResult
            Contains time points, solution, and diagnostics
        """
        t_start, t_end = t_span

        # Compute number of steps
        n_steps = int(np.ceil((t_end - t_start) / dt))

        # Allocate arrays
        t = np.zeros(n_steps + 1)
        y = np.zeros((n_steps + 1, len(y0)))

        # Initial conditions
        t[0] = t_start
        y[0] = y0.copy()

        # Reset evaluation counter
        self._n_evaluations = 0

        # Integration loop
        current_t = t_start
        for i in range(n_steps):
            # Adjust final step if needed
            current_dt = min(dt, t_end - current_t)

            # Take one step
            y[i + 1] = self.step(f, current_t, y[i], current_dt)
            current_t += current_dt
            t[i + 1] = current_t

        return IntegrationResult(
            t=t,
            y=y,
            n_steps=n_steps,
            n_evaluations=self._n_evaluations
        )

    def __repr__(self) -> str:
        return f"{self.name}(order={self.order}, symplectic={self.is_symplectic})"
