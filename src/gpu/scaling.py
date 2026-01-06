"""
GPU Scaling Experiment
======================

Measures GPU vs CPU performance as a function of batch size.

Key question: At what batch size does GPU become faster?
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from .integrators_gpu import EulerGPU, RK4GPU, harmonic_batch_cpu, harmonic_batch_gpu, HAS_CUPY

# Try to import CuPy
try:
    import cupy as cp
except ImportError:
    cp = None


@dataclass
class ScalingResult:
    """Result of scaling experiment."""
    batch_size: int
    cpu_time: float
    gpu_time: Optional[float]
    speedup: Optional[float]
    method: str


def run_scaling_experiment(
    batch_sizes: List[int] = None,
    t_span: tuple = (0, 10),
    dt: float = 0.01,
    n_runs: int = 5,
    warmup: int = 2
) -> List[ScalingResult]:
    """
    Run GPU scaling experiment.

    Parameters
    ----------
    batch_sizes : List[int]
        Batch sizes to test (default: powers of 10 from 1 to 1M)
    t_span : tuple
        Integration time span
    dt : float
        Timestep
    n_runs : int
        Number of timing runs
    warmup : int
        Number of warmup runs

    Returns
    -------
    results : List[ScalingResult]
        Scaling results for each configuration
    """
    if batch_sizes is None:
        batch_sizes = [1, 10, 100, 1000, 10000, 100000]
        if HAS_CUPY:
            batch_sizes.append(1000000)

    print("="*60)
    print("GPU SCALING EXPERIMENT")
    print("="*60)
    print(f"CuPy available: {HAS_CUPY}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Integration: t=[{t_span[0]}, {t_span[1]}], dt={dt}")
    print("-"*60)

    results = []

    # Test both Euler and RK4
    methods = [
        ("Euler", EulerGPU(use_gpu=False), EulerGPU(use_gpu=True)),
        ("RK4", RK4GPU(use_gpu=False), RK4GPU(use_gpu=True))
    ]

    for method_name, cpu_integrator, gpu_integrator in methods:
        print(f"\n{method_name}:")

        for N in batch_sizes:
            print(f"  N={N:>8d}...", end=" ")

            # Random initial conditions
            np.random.seed(42)
            y0 = np.random.randn(N, 2).astype(np.float64)

            # CPU timing
            cpu_times = []
            for run in range(warmup + n_runs):
                start = time.perf_counter()
                cpu_integrator.integrate_batch(
                    lambda t, y: harmonic_batch_cpu(t, y),
                    y0, t_span, dt
                )
                elapsed = time.perf_counter() - start
                if run >= warmup:
                    cpu_times.append(elapsed)

            cpu_time = np.mean(cpu_times)

            # GPU timing (if available)
            gpu_time = None
            speedup = None

            if HAS_CUPY:
                # Convert to GPU
                y0_gpu = cp.asarray(y0)

                # Warmup
                for _ in range(warmup):
                    gpu_integrator.integrate_batch(
                        lambda t, y: harmonic_batch_gpu(t, y),
                        y0, t_span, dt
                    )
                    cp.cuda.Stream.null.synchronize()

                # Timing
                gpu_times = []
                for _ in range(n_runs):
                    start = time.perf_counter()
                    gpu_integrator.integrate_batch(
                        lambda t, y: harmonic_batch_gpu(t, y),
                        y0, t_span, dt
                    )
                    cp.cuda.Stream.null.synchronize()
                    elapsed = time.perf_counter() - start
                    gpu_times.append(elapsed)

                gpu_time = np.mean(gpu_times)
                speedup = cpu_time / gpu_time

            # Report
            if gpu_time is not None:
                print(f"CPU={cpu_time*1000:.2f}ms, GPU={gpu_time*1000:.2f}ms, "
                      f"Speedup={speedup:.2f}x")
            else:
                print(f"CPU={cpu_time*1000:.2f}ms, GPU=N/A")

            results.append(ScalingResult(
                batch_size=N,
                cpu_time=cpu_time,
                gpu_time=gpu_time,
                speedup=speedup,
                method=method_name
            ))

    return results


def save_scaling_results(
    results: List[ScalingResult],
    output_dir: Path = None
):
    """Save scaling results to CSV."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'results' / 'data'

    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for r in results:
        records.append({
            'batch_size': r.batch_size,
            'cpu_time': r.cpu_time,
            'gpu_time': r.gpu_time,
            'speedup': r.speedup,
            'method': r.method
        })

    df = pd.DataFrame(records)
    df.to_csv(output_dir / 'gpu_scaling.csv', index=False)
    print(f"\nSaved: {output_dir / 'gpu_scaling.csv'}")

    return df


def plot_scaling_results(
    results: List[ScalingResult] = None,
    data_path: Path = None,
    output_path: Path = None
):
    """Plot GPU scaling results."""
    import matplotlib.pyplot as plt

    if results is None:
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / 'results' / 'data'
        df = pd.read_csv(data_path / 'gpu_scaling.csv')
    else:
        df = pd.DataFrame([{
            'batch_size': r.batch_size,
            'cpu_time': r.cpu_time,
            'gpu_time': r.gpu_time,
            'speedup': r.speedup,
            'method': r.method
        } for r in results])

    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / 'results' / 'figures'

    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: Runtime vs Batch Size
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for method in df['method'].unique():
        method_df = df[df['method'] == method]

        # Runtime plot
        axes[0].loglog(
            method_df['batch_size'],
            method_df['cpu_time'] * 1000,
            'o-',
            label=f'{method} (CPU)'
        )
        if method_df['gpu_time'].notna().any():
            axes[0].loglog(
                method_df['batch_size'],
                method_df['gpu_time'] * 1000,
                's--',
                label=f'{method} (GPU)'
            )

    axes[0].set_xlabel('Batch Size (N)')
    axes[0].set_ylabel('Runtime (ms)')
    axes[0].set_title('Runtime vs Batch Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Speedup plot
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        if method_df['speedup'].notna().any():
            axes[1].semilogx(
                method_df['batch_size'],
                method_df['speedup'],
                'o-',
                label=method
            )

    axes[1].axhline(y=1, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Batch Size (N)')
    axes[1].set_ylabel('GPU Speedup (CPU time / GPU time)')
    axes[1].set_title('GPU Speedup vs Batch Size')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'gpu_scaling.png', dpi=150)
    plt.close()

    print(f"Saved: {output_path / 'gpu_scaling.png'}")


if __name__ == '__main__':
    results = run_scaling_experiment()
    df = save_scaling_results(results)
    plot_scaling_results(results)
