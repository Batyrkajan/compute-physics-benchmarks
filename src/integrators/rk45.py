"""
Runge-Kutta-Fehlberg Adaptive Integrator (RK45)
===============================================

Adaptive step-size integrator using embedded RK4/RK5 pair.

Mathematical formulation:
    Uses 6 function evaluations to compute both a 4th-order and
    5th-order estimate. The difference provides an error estimate
    used to adapt the step size.

    Butcher tableau (Fehlberg):
    0    |
    1/4  | 1/4
    3/8  | 3/32       9/32
    12/13| 1932/2197  -7200/2197  7296/2197
    1    | 439/216    -8          3680/513   -845/4104
    1/2  | -8/27      2          -3544/2565   1859/4104  -11/40
    -----|--------------------------------------------------
    y5   | 16/135     0           6656/12825  28561/56430 -9/50   2/55
    y4   | 25/216     0           1408/2565   2197/4104   -1/5    0

Properties:
    - Order: 4 (with 5th order error estimate)
    - Function evaluations per step: 6
    - Adaptive: Automatically adjusts dt for desired accuracy
    - Not symplectic

This is the foundation of scipy.integrate.solve_ivp's 'RK45' method.
Best choice when you want accuracy without manually tuning dt.
"""

from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray

from .base import Integrator, IntegrationResult


class RK45(Integrator):
    """
    Adaptive Runge-Kutta-Fehlberg 4(5) integrator.

    Automatically adjusts step size to maintain a specified error
    tolerance. Uses an embedded pair of 4th and 5th order methods
    to estimate local truncation error.

    Parameters
    ----------
    rtol : float
        Relative error tolerance (default: 1e-6)
    atol : float
        Absolute error tolerance (default: 1e-9)
    max_step : float
        Maximum allowed step size (default: inf)
    min_step : float
        Minimum allowed step size (default: 1e-12)

    Good for: Achieving target accuracy efficiently
    Bad for: Long-time energy conservation, very stiff problems
    """

    name = "RK45 (Adaptive)"
    order = 4
    is_symplectic = False

    # Fehlberg coefficients
    # Time coefficients
    C = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])

    # Stage coefficients (A matrix in Butcher tableau)
    A = [
        [],
        [1/4],
        [3/32, 9/32],
        [1932/2197, -7200/2197, 7296/2197],
        [439/216, -8, 3680/513, -845/4104],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ]

    # 5th order weights
    B5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])

    # 4th order weights
    B4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_step: float = np.inf,
        min_step: float = 1e-12
    ):
        super().__init__()
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.min_step = min_step

    def step(
        self,
        f: Callable[[float, NDArray], NDArray],
        t: float,
        y: NDArray,
        dt: float
    ) -> NDArray:
        """
        Single RK45 step (non-adaptive, for interface compatibility).

        For full adaptive behavior, use `integrate` method instead.
        """
        k = self._compute_stages(f, t, y, dt)
        y_new = y + dt * np.dot(self.B5, k)
        return y_new

    def _compute_stages(
        self,
        f: Callable[[float, NDArray], NDArray],
        t: float,
        y: NDArray,
        dt: float
    ) -> NDArray:
        """Compute all 6 RK stages."""
        n = len(y)
        k = np.zeros((6, n))

        k[0] = f(t, y)
        for i in range(1, 6):
            t_i = t + self.C[i] * dt
            y_i = y + dt * sum(self.A[i][j] * k[j] for j in range(i))
            k[i] = f(t_i, y_i)

        self._n_evaluations += 6
        return k

    def _estimate_error(
        self,
        k: NDArray,
        dt: float,
        y: NDArray
    ) -> Tuple[float, NDArray]:
        """
        Estimate local truncation error.

        Returns the error norm and the error vector.
        """
        # Difference between 5th and 4th order solutions
        error = dt * np.dot(self.B5 - self.B4, k)

        # Scale by tolerance
        scale = self.atol + self.rtol * np.abs(y)
        error_norm = np.sqrt(np.mean((error / scale) ** 2))

        return error_norm, error

    def _compute_new_step(self, dt: float, error_norm: float) -> float:
        """
        Compute new step size based on error estimate.

        Uses standard PI controller formula with safety factor.
        """
        if error_norm == 0:
            return min(dt * 2, self.max_step)

        # Safety factor and order for step adjustment
        safety = 0.9
        order = 5  # Using 5th order error estimate

        # New step size
        factor = safety * (1 / error_norm) ** (1 / order)

        # Limit change rate
        factor = max(0.1, min(factor, 5.0))

        new_dt = dt * factor
        return max(self.min_step, min(new_dt, self.max_step))

    def integrate(
        self,
        f: Callable[[float, NDArray], NDArray],
        y0: NDArray,
        t_span: Tuple[float, float],
        dt: float
    ) -> IntegrationResult:
        """
        Integrate with adaptive step size control.

        Parameters
        ----------
        f : callable
            Right-hand side function f(t, y)
        y0 : NDArray
            Initial state vector
        t_span : tuple
            (t_start, t_end) time interval
        dt : float
            Initial timestep (will be adapted)

        Returns
        -------
        result : IntegrationResult
            Contains time points, solution, and diagnostics
        """
        t_start, t_end = t_span
        current_t = t_start
        current_y = y0.copy()
        current_dt = min(dt, self.max_step)

        # Storage (will grow dynamically)
        t_list = [current_t]
        y_list = [current_y.copy()]

        self._n_evaluations = 0
        n_steps = 0

        while current_t < t_end:
            # Don't overshoot
            if current_t + current_dt > t_end:
                current_dt = t_end - current_t

            # Compute stages
            k = self._compute_stages(f, current_t, current_y, current_dt)

            # Estimate error
            error_norm, _ = self._estimate_error(k, current_dt, current_y)

            if error_norm <= 1.0:
                # Accept step
                current_y = current_y + current_dt * np.dot(self.B5, k)
                current_t += current_dt
                n_steps += 1

                t_list.append(current_t)
                y_list.append(current_y.copy())

            # Compute new step size (whether accepted or not)
            current_dt = self._compute_new_step(current_dt, error_norm)

        return IntegrationResult(
            t=np.array(t_list),
            y=np.array(y_list),
            n_steps=n_steps,
            n_evaluations=self._n_evaluations
        )

    def __repr__(self) -> str:
        return f"RK45(rtol={self.rtol}, atol={self.atol})"
