"""DhanRakshak — core AML mule-account detection library."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dhanrakshak")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "data_loader",
    "feature_engineering",
    "ensemble_models",
    "graph_analysis",
    "temporal_window_generator",
    "evaluation",
    "pipeline",
]
