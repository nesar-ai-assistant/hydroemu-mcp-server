"""Science tool package. Only names in __all__ become MCP tools.

The hydro-emulator tools live in hydro_tools.py. The emulator backend is
provided by the cosmohydro_emu package (lazily imported inside each tool
function).
"""

from .hydro_tools import (
    ArtifactResult,
    list_observables,
    describe_parameters,
    predict_observable,
    plot_prediction,
    plot_observable_comparison,
)

__all__ = [
    "list_observables",
    "describe_parameters",
    "predict_observable",
    "plot_prediction",
    "plot_observable_comparison",
]
