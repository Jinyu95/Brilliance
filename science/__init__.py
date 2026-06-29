"""BRILLIANCE discovery harness — parameter-space exploration and analysis.

This module supports the PRL-targeted experiments:
  Candidate 1 — dipole grading pattern discovery
  Candidate 2 — new lattice family discovery (N-bend sweeps)
  Candidate 3 — empirical scaling-law corrections
  Candidate 4 — human-vs-AI comparison protocol
  Candidate 5 — asymmetry benefit quantification

Core components:
  explore.py  — parameter grid generation + batch agent runner
  analyze.py  — Pareto frontier, anomaly detection, scaling analysis
  plots.py    — publication-quality figures (plotly, no BLAS dependency)
"""

from science.explore import DesignSpaceExplorer
from science.analyze import (
    ResultDatabase,
    detect_anomalies,
    compare_to_reference,
    summarize_by_family,
)
from science.plots import (
    plot_pareto_frontier,
    plot_scaling_law,
    plot_emittance_vs_energy,
    plot_dipole_grading_pattern,
)

__all__ = [
    "DesignSpaceExplorer",
    "ResultDatabase",
    "detect_anomalies",
    "compare_to_reference",
    "summarize_by_family",
    "plot_pareto_frontier",
    "plot_scaling_law",
    "plot_emittance_vs_energy",
    "plot_dipole_grading_pattern",
]
