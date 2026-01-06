"""
GPU-Accelerated Integrators
===========================

CuPy implementations of integrators for batch processing.

These implementations integrate N independent oscillators simultaneously,
demonstrating GPU parallelism advantages for large batch sizes.

Requires: cupy (pip install cupy-cuda12x for CUDA 12.x)
"""

from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray

# Try to import CuPy, fall back to NumPy
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np  # Fallback to NumPy
    HAS_CUPY = False


class EulerGPU:
    """
    GPU-accelerated Forward Euler for batch integration.

    Integrates N independent systems simultaneously.
    State shape: (N, n_vars) where N is batch size.
    """

    name = "Forward Euler (GPU)"

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np

    def integrate_batch(
        self,
        f_batch: Callable,
        y0: NDArray,
        t_span: Tuple[float, float],
        dt: float
    ) -> Tuple[NDArray, NDArray]:
        """
        Integrate batch of systems.

        Parameters
        ----------
        f_batch : callable
            Vectorized RHS: f(t, y) where y has shape (N, n_vars)
        y0 : NDArray
            Initial conditions, shape (N, n_vars)
        t_span : tuple
            (t_start, t_end)
        dt : float
            Timestep

        Returns
        -------
        t : NDArray
            Time points
        y : NDArray
            Solution, shape (n_steps, N, n_vars)
        """
        xp = self.xp
        t_start, t_end = t_span

        # Move to GPU if available
        y0_device = xp.asarray(y0)

        n_steps = int(np.ceil((t_end - t_start) / dt))
        N, n_vars = y0_device.shape

        # Allocate output on device
        t = np.linspace(t_start, t_end, n_steps + 1)
        y = xp.zeros((n_steps + 1, N, n_vars), dtype=y0_device.dtype)
        y[0] = y0_device

        # Integration loop
        current_t = t_start
        current_y = y0_device.copy()

        for i in range(n_steps):
            current_dt = min(dt, t_end - current_t)
            k1 = f_batch(current_t, current_y)
            current_y = current_y + current_dt * k1
            current_t += current_dt
            y[i + 1] = current_y

        # Move back to CPU
        if self.use_gpu:
            y = cp.asnumpy(y)

        return t, y


class RK4GPU:
    """
    GPU-accelerated RK4 for batch integration.

    Integrates N independent systems simultaneously.
    """

    name = "RK4 (GPU)"

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np

    def integrate_batch(
        self,
        f_batch: Callable,
        y0: NDArray,
        t_span: Tuple[float, float],
        dt: float
    ) -> Tuple[NDArray, NDArray]:
        """
        Integrate batch of systems with RK4.

        Parameters
        ----------
        f_batch : callable
            Vectorized RHS: f(t, y) where y has shape (N, n_vars)
        y0 : NDArray
            Initial conditions, shape (N, n_vars)
        t_span : tuple
            (t_start, t_end)
        dt : float
            Timestep

        Returns
        -------
        t : NDArray
            Time points
        y : NDArray
            Solution, shape (n_steps, N, n_vars)
        """
        xp = self.xp
        t_start, t_end = t_span

        # Move to GPU if available
        y0_device = xp.asarray(y0)

        n_steps = int(np.ceil((t_end - t_start) / dt))
        N, n_vars = y0_device.shape

        # Allocate output on device
        t = np.linspace(t_start, t_end, n_steps + 1)
        y = xp.zeros((n_steps + 1, N, n_vars), dtype=y0_device.dtype)
        y[0] = y0_device

        # Integration loop
        current_t = t_start
        current_y = y0_device.copy()

        for i in range(n_steps):
            current_dt = min(dt, t_end - current_t)

            k1 = f_batch(current_t, current_y)
            k2 = f_batch(current_t + current_dt / 2, current_y + current_dt * k1 / 2)
            k3 = f_batch(current_t + current_dt / 2, current_y + current_dt * k2 / 2)
            k4 = f_batch(current_t + current_dt, current_y + current_dt * k3)

            current_y = current_y + (current_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            current_t += current_dt
            y[i + 1] = current_y

        # Move back to CPU
        if self.use_gpu:
            y = cp.asnumpy(y)

        return t, y


def harmonic_batch_cpu(t: float, y: NDArray, omega: float = 1.0) -> NDArray:
    """
    Batch harmonic oscillator RHS (CPU/NumPy).

    y has shape (N, 2) where y[:, 0] is position, y[:, 1] is velocity.
    """
    dydt = np.zeros_like(y)
    dydt[:, 0] = y[:, 1]          # dx/dt = v
    dydt[:, 1] = -omega**2 * y[:, 0]  # dv/dt = -omega^2 * x
    return dydt


def harmonic_batch_gpu(t: float, y, omega: float = 1.0):
    """
    Batch harmonic oscillator RHS (GPU/CuPy).

    y has shape (N, 2) where y[:, 0] is position, y[:, 1] is velocity.
    """
    xp = cp if HAS_CUPY else np
    dydt = xp.zeros_like(y)
    dydt[:, 0] = y[:, 1]
    dydt[:, 1] = -omega**2 * y[:, 0]
    return dydt
