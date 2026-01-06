"""
Velocity Verlet Integrator
==========================

A symplectic integrator specifically designed for Hamiltonian systems.
Second-order accurate but with excellent energy conservation.

Mathematical formulation:
    For systems of the form: d²x/dt² = a(x)  (acceleration depends only on position)

    x_{n+1} = x_n + v_n * dt + 0.5 * a(x_n) * dt²
    v_{n+1} = v_n + 0.5 * (a(x_n) + a(x_{n+1})) * dt

Properties:
    - Order: 2 (error ~ O(dt²))
    - Function evaluations per step: 2 (or 1 with FSAL trick)
    - Symplectic: Preserves phase space volume
    - Time-reversible

This is THE method for molecular dynamics, celestial mechanics, and
any problem where energy conservation over long times matters more
than instantaneous accuracy.

Key insight: Sometimes a "less accurate" method is better for physics
because it respects the underlying structure of the equations.
"""

from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray

from .base import Integrator, IntegrationResult


class VelocityVerlet(Integrator):
    """
    Velocity Verlet (leapfrog) symplectic integrator.

    Designed for second-order ODEs of the form:
        d²x/dt² = a(x)

    where acceleration depends only on position, not velocity.
    This covers most mechanical systems (harmonic oscillators,
    pendulums, N-body gravity, molecular dynamics).

    The symplectic property means:
    - Energy errors are bounded, not growing
    - Phase space volume is conserved
    - Long-term behavior is qualitatively correct

    Good for: Long simulations, energy conservation, Hamiltonian systems
    Bad for: Dissipative systems, velocity-dependent forces
    """

    name = "Velocity Verlet"
    order = 2
    is_symplectic = True

    def __init__(self):
        super().__init__()
        self._last_acceleration = None

    def step(
        self,
        f: Callable[[float, NDArray], NDArray],
        t: float,
        y: NDArray,
        dt: float
    ) -> NDArray:
        """
        Advance solution by one Verlet step.

        Assumes y = [x, v] where x is position and v is velocity.
        The function f should return [v, a] where a is acceleration.

        Parameters
        ----------
        f : callable
            Right-hand side function f(t, y) returning [dx/dt, dv/dt]
        t : float
            Current time
        y : NDArray
            Current state [position, velocity]
        dt : float
            Timestep size

        Returns
        -------
        y_new : NDArray
            State at t + dt
        """
        n = len(y) // 2
        x = y[:n]
        v = y[n:]

        # Get current acceleration
        dydt = f(t, y)
        a = dydt[n:]  # acceleration is second half of derivative
        self._n_evaluations += 1

        # Update position (full step)
        x_new = x + v * dt + 0.5 * a * dt**2

        # Get new acceleration at new position
        y_temp = np.concatenate([x_new, v])  # temporary state
        dydt_new = f(t + dt, y_temp)
        a_new = dydt_new[n:]
        self._n_evaluations += 1

        # Update velocity (using average acceleration)
        v_new = v + 0.5 * (a + a_new) * dt

        return np.concatenate([x_new, v_new])

    def integrate(
        self,
        f: Callable[[float, NDArray], NDArray],
        y0: NDArray,
        t_span: Tuple[float, float],
        dt: float
    ) -> IntegrationResult:
        """
        Integrate with optimized Verlet (reusing accelerations).

        Uses FSAL (First Same As Last) optimization to reuse
        the acceleration computed at the end of each step.
        """
        t_start, t_end = t_span
        n_steps = int(np.ceil((t_end - t_start) / dt))

        n = len(y0) // 2  # number of position variables
        t = np.zeros(n_steps + 1)
        y = np.zeros((n_steps + 1, len(y0)))

        t[0] = t_start
        y[0] = y0.copy()

        self._n_evaluations = 0

        # Get initial acceleration
        dydt = f(t_start, y0)
        a = dydt[n:].copy()
        self._n_evaluations += 1

        current_t = t_start
        for i in range(n_steps):
            current_dt = min(dt, t_end - current_t)

            x = y[i, :n]
            v = y[i, n:]

            # Position update
            x_new = x + v * current_dt + 0.5 * a * current_dt**2

            # Get new acceleration
            y_temp = np.concatenate([x_new, v])
            dydt_new = f(current_t + current_dt, y_temp)
            a_new = dydt_new[n:]
            self._n_evaluations += 1

            # Velocity update
            v_new = v + 0.5 * (a + a_new) * current_dt

            # Store
            y[i + 1, :n] = x_new
            y[i + 1, n:] = v_new
            current_t += current_dt
            t[i + 1] = current_t

            # FSAL: reuse acceleration for next step
            a = a_new.copy()

        return IntegrationResult(
            t=t,
            y=y,
            n_steps=n_steps,
            n_evaluations=self._n_evaluations
        )
