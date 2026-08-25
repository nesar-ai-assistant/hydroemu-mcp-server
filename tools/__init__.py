"""Science tool package. Only names in __all__ become MCP tools.

The hydro-emulator tools live in hydro_tools.py; add your own modules
here and extend __all__ to expose new tools — nothing else changes.
"""

from .hydro_tools import (
    ArtifactResult,
    list_observables,
    describe_parameters,
    predict_observable,
    predict_observable_redshift,
    plot_observable_comparison,
)

__all__ = [
    "list_observables",
    "describe_parameters",
    "predict_observable",
    "predict_observable_redshift",
    "plot_observable_comparison",
]
