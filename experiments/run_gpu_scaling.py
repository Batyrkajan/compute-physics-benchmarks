"""
GPU Scaling Experiment Runner
=============================

Measures performance as a function of batch size.
Compares CPU vs GPU for batch integration.

Note: Requires CuPy for GPU acceleration.
      pip install cupy-cuda12x  (for CUDA 12.x)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gpu.scaling import run_scaling_experiment, save_scaling_results, plot_scaling_results


def main():
    """Run GPU scaling experiment."""

    # Smaller batch sizes for CPU-only testing
    batch_sizes = [1, 10, 100, 1000, 10000, 100000]

    results = run_scaling_experiment(
        batch_sizes=batch_sizes,
        t_span=(0, 10),
        dt=0.01,
        n_runs=3,
        warmup=1
    )

    df = save_scaling_results(results)
    plot_scaling_results(results)

    print("\n" + "="*60)
    print("GPU SCALING SUMMARY")
    print("="*60)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
