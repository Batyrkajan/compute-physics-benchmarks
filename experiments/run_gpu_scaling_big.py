"""
GPU Scaling - Large Batch Sizes
================================

Tests GPU performance with up to 1 million particles.
Shows the true power of RTX 5090.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gpu.scaling import run_scaling_experiment, save_scaling_results, plot_scaling_results


def main():
    # Big batch sizes to show GPU advantage
    batch_sizes = [1000, 10000, 100000, 500000, 1000000]

    print("Testing with large batch sizes...")
    print("This will show the true power of your RTX 5090!")
    print()

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
    print("GPU SCALING SUMMARY (Large Batches)")
    print("="*60)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
