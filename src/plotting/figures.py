"""
Publication-Quality Figures
===========================

Generates all figures for the technical report.

Style guide:
- Clean, minimal design
- Consistent color scheme
- Proper axis labels with units
- Legends that don't obscure data
- High DPI for publication
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 6

# Color scheme
COLORS = {
    'Forward Euler': '#e41a1c',      # Red
    'RK4': '#377eb8',                 # Blue
    'RK45 (Adaptive)': '#4daf4a',     # Green
    'Velocity Verlet': '#984ea3',     # Purple
}

MARKERS = {
    'Forward Euler': 'o',
    'RK4': 's',
    'RK45 (Adaptive)': '^',
    'Velocity Verlet': 'D',
}


def plot_error_vs_timestep(
    df: pd.DataFrame,
    problem_name: str,
    output_path: Path,
    show: bool = False
):
    """
    Plot RMS error vs timestep (log-log).

    This plot demonstrates the order of accuracy for each method.
    The slope of the line equals the order of the method.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    problem_df = df[df['problem'] == problem_name]

    for integrator in problem_df['integrator'].unique():
        data = problem_df[problem_df['integrator'] == integrator]
        data = data.sort_values('dt')

        ax.loglog(
            data['dt'],
            data['rms_error'],
            marker=MARKERS.get(integrator, 'o'),
            color=COLORS.get(integrator, 'gray'),
            label=integrator,
            linewidth=2,
            markersize=7
        )

    # Add reference slopes
    dt_ref = np.array([1e-3, 1e-1])

    # O(dt) reference
    ax.loglog(dt_ref, 0.01 * dt_ref, 'k--', alpha=0.3, linewidth=1)
    ax.text(0.03, 0.0006, 'O(dt)', fontsize=8, alpha=0.5)

    # O(dt^2) reference
    ax.loglog(dt_ref, 0.1 * dt_ref**2, 'k--', alpha=0.3, linewidth=1)
    ax.text(0.03, 0.00015, 'O(dt²)', fontsize=8, alpha=0.5)

    # O(dt^4) reference
    ax.loglog(dt_ref, 10 * dt_ref**4, 'k--', alpha=0.3, linewidth=1)
    ax.text(0.03, 1e-7, 'O(dt⁴)', fontsize=8, alpha=0.5)

    ax.set_xlabel('Timestep dt')
    ax.set_ylabel('RMS Error')
    ax.set_title(f'Accuracy vs Timestep: {problem_name}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f'error_vs_timestep_{problem_name.lower().replace(" ", "_")}.png')
    if show:
        plt.show()
    plt.close()


def plot_runtime_vs_timestep(
    df: pd.DataFrame,
    problem_name: str,
    output_path: Path,
    show: bool = False
):
    """
    Plot runtime vs timestep (log-log).

    Shows computational cost as a function of timestep.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    problem_df = df[df['problem'] == problem_name]

    for integrator in problem_df['integrator'].unique():
        data = problem_df[problem_df['integrator'] == integrator]
        data = data.sort_values('dt')

        ax.loglog(
            data['dt'],
            data['mean_time'] * 1000,  # Convert to ms
            marker=MARKERS.get(integrator, 'o'),
            color=COLORS.get(integrator, 'gray'),
            label=integrator,
            linewidth=2,
            markersize=7
        )

    ax.set_xlabel('Timestep dt')
    ax.set_ylabel('Runtime (ms)')
    ax.set_title(f'Runtime vs Timestep: {problem_name}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f'runtime_vs_timestep_{problem_name.lower().replace(" ", "_")}.png')
    if show:
        plt.show()
    plt.close()


def plot_pareto_frontier(
    df: pd.DataFrame,
    problem_name: str,
    output_path: Path,
    show: bool = False
):
    """
    Plot error vs runtime (Pareto plot).

    This is the "money plot" showing the accuracy-cost tradeoff.
    Points on the lower-left Pareto frontier are optimal.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    problem_df = df[df['problem'] == problem_name]

    for integrator in problem_df['integrator'].unique():
        data = problem_df[problem_df['integrator'] == integrator]

        ax.loglog(
            data['mean_time'] * 1000,
            data['rms_error'],
            marker=MARKERS.get(integrator, 'o'),
            color=COLORS.get(integrator, 'gray'),
            label=integrator,
            linewidth=1,
            linestyle='-',
            markersize=8,
            alpha=0.8
        )

    # Find and highlight Pareto frontier
    all_points = problem_df[['mean_time', 'rms_error']].values
    pareto_mask = _compute_pareto_frontier(all_points)
    pareto_points = all_points[pareto_mask]
    pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

    ax.plot(
        pareto_points[:, 0] * 1000,
        pareto_points[:, 1],
        'k--',
        linewidth=2,
        alpha=0.5,
        label='Pareto Frontier'
    )

    ax.set_xlabel('Runtime (ms)')
    ax.set_ylabel('RMS Error')
    ax.set_title(f'Accuracy vs Cost: {problem_name}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate(
        'Better\n(faster, more accurate)',
        xy=(0.15, 0.15),
        xycoords='axes fraction',
        fontsize=9,
        alpha=0.6,
        ha='center'
    )
    ax.annotate(
        '',
        xy=(0.05, 0.05),
        xytext=(0.25, 0.25),
        xycoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_path / f'pareto_{problem_name.lower().replace(" ", "_")}.png')
    if show:
        plt.show()
    plt.close()


def _compute_pareto_frontier(points: np.ndarray) -> np.ndarray:
    """Compute Pareto frontier mask for 2D points (minimize both axes)."""
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # Point j dominates point i if j is better in both dimensions
                if (points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1] and
                    (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1])):
                    is_pareto[i] = False
                    break

    return is_pareto


def plot_energy_drift(
    data_path: Path,
    output_path: Path,
    show: bool = False
):
    """
    Plot energy drift over time for long-term integration.

    Demonstrates the symplectic advantage of Verlet.
    """
    with open(data_path / 'longterm_energy.json', 'r') as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    histories = data['histories']
    config = data['config']

    for integrator_name, hist in histories.items():
        t = np.array(hist['t'])
        energy = np.array(hist['energy'])
        E0 = hist['initial']

        # Relative energy error
        relative_error = (energy - E0) / E0

        ax.plot(
            t,
            relative_error,
            color=COLORS.get(integrator_name, 'gray'),
            label=integrator_name,
            linewidth=1.5
        )

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Energy Error (E - E0) / E0')
    ax.set_title(f'Energy Conservation: T={config["T"]}, dt={config["dt"]}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Scientific notation for y-axis if needed
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    plt.tight_layout()
    plt.savefig(output_path / 'energy_drift.png')
    if show:
        plt.show()
    plt.close()


def plot_stability_regions(
    data_path: Path,
    output_path: Path,
    show: bool = False
):
    """
    Plot stability boundaries as bar chart.
    """
    df = pd.read_csv(data_path / 'stability_analysis.csv')

    fig, ax = plt.subplots(figsize=(10, 6))

    problems = df['problem'].unique()
    integrators = df['integrator'].unique()

    x = np.arange(len(integrators))
    width = 0.35

    for i, problem in enumerate(problems):
        problem_df = df[df['problem'] == problem]
        max_dts = [problem_df[problem_df['integrator'] == intg]['max_stable_dt'].values[0]
                   for intg in integrators]

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, max_dts, width, label=problem, alpha=0.8)

    ax.set_xlabel('Integrator')
    ax.set_ylabel('Maximum Stable Timestep')
    ax.set_title('Stability Boundaries by Integrator')
    ax.set_xticks(x)
    ax.set_xticklabels(integrators, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'stability_regions.png')
    if show:
        plt.show()
    plt.close()


def plot_phase_portrait(
    output_path: Path,
    show: bool = False
):
    """
    Plot phase portrait comparing integrators.

    Shows qualitative behavior in phase space.
    """
    import sys
    sys.path.insert(0, str(output_path.parent.parent))

    from src.integrators import ForwardEuler, RK4, VelocityVerlet
    from src.problems import HarmonicOscillator, ProblemConfig

    # Long integration with moderate timestep to show differences
    config = ProblemConfig(
        y0=np.array([1.0, 0.0]),
        t_span=(0.0, 100.0),
        params={'omega': 1.0}
    )
    problem = HarmonicOscillator(config)
    dt = 0.1  # Large enough to show method differences

    integrators = [
        ('Forward Euler', ForwardEuler()),
        ('RK4', RK4()),
        ('Velocity Verlet', VelocityVerlet())
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Exact circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    x_exact = np.cos(theta)
    v_exact = -np.sin(theta)

    for ax, (name, integrator) in zip(axes, integrators):
        result = integrator.integrate(problem.f, problem.y0, problem.t_span, dt)

        # Plot numerical trajectory
        ax.plot(
            result.y[:, 0],
            result.y[:, 1],
            color=COLORS.get(name, 'blue'),
            linewidth=0.5,
            alpha=0.8,
            label=f'{name}'
        )

        # Plot exact solution
        ax.plot(x_exact, v_exact, 'k--', linewidth=1.5, alpha=0.3, label='Exact')

        # Mark start point
        ax.plot(result.y[0, 0], result.y[0, 1], 'go', markersize=8, label='Start')

        # Mark end point
        ax.plot(result.y[-1, 0], result.y[-1, 1], 'rx', markersize=8, label='End')

        ax.set_xlabel('Position x')
        ax.set_ylabel('Velocity v')
        ax.set_title(name)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Phase Portraits: dt={dt}, T={problem.t_span[1]}', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'phase_portrait.png', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def generate_all_figures(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    show: bool = False
):
    """
    Generate all publication-quality figures.

    Parameters
    ----------
    data_dir : Path
        Directory containing benchmark data
    output_dir : Path
        Directory for saving figures
    show : bool
        Whether to display figures interactively
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / 'results' / 'data'
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'results' / 'figures'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating publication-quality figures...")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")

    # Load accuracy sweep data
    accuracy_csv = data_dir / 'accuracy_sweep.csv'
    if accuracy_csv.exists():
        df = pd.read_csv(accuracy_csv)

        for problem in df['problem'].unique():
            print(f"  Plotting {problem}...")

            plot_error_vs_timestep(df, problem, output_dir, show)
            plot_runtime_vs_timestep(df, problem, output_dir, show)
            plot_pareto_frontier(df, problem, output_dir, show)

    # Energy drift plot
    longterm_json = data_dir / 'longterm_energy.json'
    if longterm_json.exists():
        print("  Plotting energy drift...")
        plot_energy_drift(data_dir, output_dir, show)

    # Stability regions
    stability_csv = data_dir / 'stability_analysis.csv'
    if stability_csv.exists():
        print("  Plotting stability regions...")
        plot_stability_regions(data_dir, output_dir, show)

    # Phase portrait (generated from scratch)
    print("  Plotting phase portrait...")
    plot_phase_portrait(output_dir, show)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    generate_all_figures(show=False)
