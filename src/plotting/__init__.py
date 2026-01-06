"""
Plotting Module
===============

Publication-quality figure generation for benchmark results.
"""

from .figures import (
    plot_error_vs_timestep,
    plot_runtime_vs_timestep,
    plot_pareto_frontier,
    plot_energy_drift,
    plot_stability_regions,
    plot_phase_portrait,
    generate_all_figures
)

__all__ = [
    'plot_error_vs_timestep',
    'plot_runtime_vs_timestep',
    'plot_pareto_frontier',
    'plot_energy_drift',
    'plot_stability_regions',
    'plot_phase_portrait',
    'generate_all_figures'
]
