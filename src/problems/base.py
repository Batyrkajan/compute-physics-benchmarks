"""
Base Physics Problem Class
==========================

Abstract base class defining the interface for physics test problems.

All problems are formulated as first-order ODE systems:
    dy/dt = f(t, y)

For mechanical systems, we use y = [x, v] (position, velocity).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class ProblemConfig:
    """
    Configuration for a physics problem.

    Attributes
    ----------
    y0 : NDArray
        Initial state vector [position, velocity]
    t_span : tuple
        Time interval (t_start, t_end)
    params : dict
        Problem-specific parameters (e.g., omega, g/L)
    """
    y0: NDArray[np.float64]
    t_span: Tuple[float, float]
    params: dict


class PhysicsProblem(ABC):
    """
    Abstract base class for physics test problems.

    Subclasses must implement:
    - `f`: The ODE right-hand side function
    - `energy`: Total energy of the system

    And optionally:
    - `analytical_solution`: If an exact solution exists
    """

    name: str = "BaseProblem"
    has_analytical_solution: bool = False

    def __init__(self, config: ProblemConfig):
        """
        Initialize problem with configuration.

        Parameters
        ----------
        config : ProblemConfig
            Initial conditions, time span, and parameters
        """
        self.config = config
        self.y0 = config.y0
        self.t_span = config.t_span
        self.params = config.params

    @abstractmethod
    def f(self, t: float, y: NDArray) -> NDArray:
        """
        Right-hand side of the ODE: dy/dt = f(t, y)

        Parameters
        ----------
        t : float
            Current time
        y : NDArray
            Current state vector

        Returns
        -------
        dydt : NDArray
            Time derivative of state vector
        """
        pass

    @abstractmethod
    def energy(self, y: NDArray) -> float:
        """
        Compute total energy of the system.

        Parameters
        ----------
        y : NDArray
            State vector [position, velocity]

        Returns
        -------
        E : float
            Total energy (kinetic + potential)
        """
        pass

    def analytical_solution(self, t: NDArray) -> Optional[NDArray]:
        """
        Analytical solution if available.

        Parameters
        ----------
        t : NDArray
            Time points

        Returns
        -------
        y : NDArray or None
            Exact solution at time points, or None if unavailable
        """
        return None

    def initial_energy(self) -> float:
        """Compute energy at initial conditions."""
        return self.energy(self.y0)

    def energy_error(self, y: NDArray) -> float:
        """
        Compute relative energy error.

        Parameters
        ----------
        y : NDArray
            State vector at some time

        Returns
        -------
        error : float
            Relative energy error |E - E0| / |E0|
        """
        E0 = self.initial_energy()
        E = self.energy(y)
        if E0 == 0:
            return abs(E)
        return abs(E - E0) / abs(E0)

    def __repr__(self) -> str:
        return f"{self.name}(params={self.params})"
