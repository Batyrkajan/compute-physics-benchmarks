"""
Numerical Integration Methods
=============================

Implementations of various ODE integrators for physics simulations.

Available integrators:
- ForwardEuler: 1st order, simplest baseline
- RK4: 4th order Runge-Kutta, industry standard
- RK45: Adaptive Runge-Kutta-Fehlberg
- VelocityVerlet: 2nd order symplectic integrator
"""

from .base import Integrator
from .euler import ForwardEuler
from .rk4 import RK4
from .rk45 import RK45
from .verlet import VelocityVerlet

__all__ = ['Integrator', 'ForwardEuler', 'RK4', 'RK45', 'VelocityVerlet']
