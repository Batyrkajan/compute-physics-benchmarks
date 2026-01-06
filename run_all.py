"""
Master Benchmark Script
=======================

Runs all experiments and generates all plots.

Usage:
    python run_all.py           # Run all experiments and plots
    python run_all.py --quick   # Quick run with fewer timesteps
    python run_all.py --plots   # Only generate plots (assumes data exists)
"""

import sys
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def run_all_experiments(quick: bool = False):
    """Run all experiments."""

    from experiments.run_accuracy_sweep import run_accuracy_sweep
    from experiments.run_stability_analysis import run_stability_analysis
    from experiments.run_longterm import run_longterm_experiment

    start = time.time()

    print("\n" + "="*70)
    print("RUNNING ALL EXPERIMENTS")
    print("="*70)

    # 1. Accuracy sweep
    print("\n[1/3] Accuracy Sweep...")
    run_accuracy_sweep()

    # 2. Stability analysis
    print("\n[2/3] Stability Analysis...")
    run_stability_analysis()

    # 3. Long-term energy
    print("\n[3/3] Long-term Energy...")
    run_longterm_experiment()

    elapsed = time.time() - start
    print(f"\nAll experiments completed in {elapsed:.1f} seconds")


def generate_all_plots():
    """Generate all publication-quality plots."""

    # Will implement in next step
    print("\nPlot generation will be implemented next...")


def main():
    parser = argparse.ArgumentParser(
        description='Run physics integration benchmarks'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick run with fewer configurations'
    )
    parser.add_argument(
        '--plots', action='store_true',
        help='Only generate plots (skip experiments)'
    )
    parser.add_argument(
        '--experiments', action='store_true',
        help='Only run experiments (skip plots)'
    )

    args = parser.parse_args()

    if args.plots:
        generate_all_plots()
    elif args.experiments:
        run_all_experiments(quick=args.quick)
    else:
        run_all_experiments(quick=args.quick)
        generate_all_plots()


if __name__ == '__main__':
    main()
