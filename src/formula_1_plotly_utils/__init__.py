from importlib.metadata import PackageNotFoundError, version

from . import plot
from . import values
from ._core.annotations import ChartMarker
from .values.colors import apply_driver_colors

try:
    __version__ = version("formula-1-plotly-utils")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "plot",
    "values",
    "ChartMarker",
    "apply_driver_colors",
]
