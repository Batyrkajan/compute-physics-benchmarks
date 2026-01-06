"""
Accuracy Metrics
================

Functions for computing error metrics between numerical and reference solutions.

Metrics implemented:
- RMS error (root mean square)
- Max error (infinity norm)
- Position error
- Velocity error
- Phase error
- Energy drift
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class AccuracyMetrics:
    """
    Collection of accuracy metrics.

    Attributes
    ----------
    rms_error : float
        Root mean square error
    max_error : float
        Maximum absolute error
    position_rms : float
        RMS error in position only
    velocity_rms : float
        RMS error in velocity only
    final_error : float
        Error at final time point
    """
    rms_error: float
    max_error: float
    position_rms: float
    velocity_rms: float
    final_error: float


def compute_errors(
    y_numerical: NDArray,
    y_reference: NDArray,
    t: Optional[NDArray] = None
) -> AccuracyMetrics:
    """
    Compute error metrics between numerical and reference solutions.

    Parameters
    ----------
    y_numerical : NDArray
        Numerical solution, shape (n_steps, n_vars)
    y_reference : NDArray
        Reference/analytical solution, same shape
    t : NDArray, optional
        Time points (for weighted error computation)

    Returns
    -------
    metrics : AccuracyMetrics
        Collection of error metrics
    """
    # Handle size mismatch (adaptive methods may have different lengths)
    min_len = min(len(y_numerical), len(y_reference))
    y_num = y_numerical[:min_len]
    y_ref = y_reference[:min_len]

    # Error vector
    error = y_num - y_ref
    error_norm = np.linalg.norm(error, axis=1)

    # RMS error (normalized by number of points)
    rms_error = np.sqrt(np.mean(error_norm**2))

    # Max error
    max_error = np.max(error_norm)

    # Position and velocity errors (assuming y = [x, v])
    n_vars = y_num.shape[1]
    n_pos = n_vars // 2

    pos_error = y_num[:, :n_pos] - y_ref[:, :n_pos]
    vel_error = y_num[:, n_pos:] - y_ref[:, n_pos:]

    position_rms = np.sqrt(np.mean(np.linalg.norm(pos_error, axis=1)**2))
    velocity_rms = np.sqrt(np.mean(np.linalg.norm(vel_error, axis=1)**2))

    # Final error
    final_error = np.linalg.norm(error[-1])

    return AccuracyMetrics(
        rms_error=rms_error,
        max_error=max_error,
        position_rms=position_rms,
        velocity_rms=velocity_rms,
        final_error=final_error
    )


@dataclass
class EnergyMetrics:
    """
    Energy conservation metrics.

    Attributes
    ----------
    initial_energy : float
        Energy at t=0
    final_energy : float
        Energy at t=t_end
    max_drift : float
        Maximum absolute energy deviation
    rms_drift : float
        RMS energy deviation
    relative_drift : float
        Final relative energy error |E-E0|/|E0|
    energy_history : NDArray
        Energy at each time point
    """
    initial_energy: float
    final_energy: float
    max_drift: float
    rms_drift: float
    relative_drift: float
    energy_history: NDArray[np.float64]


def compute_energy_drift(
    y: NDArray,
    energy_func: callable
) -> EnergyMetrics:
    """
    Compute energy conservation metrics.

    Parameters
    ----------
    y : NDArray
        Solution trajectory, shape (n_steps, n_vars)
    energy_func : callable
        Function that computes energy from state vector

    Returns
    -------
    metrics : EnergyMetrics
        Collection of energy metrics
    """
    # Compute energy at each time point
    energy = np.array([energy_func(y[i]) for i in range(len(y))])

    initial_energy = energy[0]
    final_energy = energy[-1]

    # Energy deviation from initial
    drift = energy - initial_energy

    max_drift = np.max(np.abs(drift))
    rms_drift = np.sqrt(np.mean(drift**2))

    # Relative drift
    if initial_energy != 0:
        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
    else:
        relative_drift = abs(final_energy - initial_energy)

    return EnergyMetrics(
        initial_energy=initial_energy,
        final_energy=final_energy,
        max_drift=max_drift,
        rms_drift=rms_drift,
        relative_drift=relative_drift,
        energy_history=energy
    )


def compute_phase_error(
    t: NDArray,
    y_numerical: NDArray,
    y_reference: NDArray,
    omega: float
) -> float:
    """
    Estimate phase error for oscillatory solutions.

    Computes the accumulated timing error by comparing zero-crossings
    or peak positions.

    Parameters
    ----------
    t : NDArray
        Time points
    y_numerical : NDArray
        Numerical solution (position in first column)
    y_reference : NDArray
        Reference solution
    omega : float
        Expected angular frequency

    Returns
    -------
    phase_error : float
        Estimated phase error in radians
    """
    # Extract positions
    x_num = y_numerical[:, 0]
    x_ref = y_reference[:, 0]

    # Find zero crossings (from negative to positive)
    def find_zero_crossings(x):
        crossings = []
        for i in range(len(x) - 1):
            if x[i] <= 0 < x[i + 1]:
                # Linear interpolation
                frac = -x[i] / (x[i + 1] - x[i])
                crossings.append(i + frac)
        return np.array(crossings)

    cross_num = find_zero_crossings(x_num)
    cross_ref = find_zero_crossings(x_ref)

    if len(cross_num) < 2 or len(cross_ref) < 2:
        return 0.0

    # Compare periods
    n_compare = min(len(cross_num), len(cross_ref)) - 1
    if n_compare < 1:
        return 0.0

    # Accumulated phase difference
    dt_num = np.diff(cross_num[:n_compare + 1]) * (t[1] - t[0])
    dt_ref = np.diff(cross_ref[:n_compare + 1]) * (t[1] - t[0])

    period_error = np.mean(dt_num - dt_ref)
    phase_error = period_error * omega * n_compare

    return phase_error
