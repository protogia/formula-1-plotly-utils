try:
    from fastf1.plotting import setup_mpl
    setup_mpl(misc_mpl_mods=False)
except Exception as exc:               # pragma: no cover
    import warnings
    warnings.warn(
        "fastf1 is not installed – colour palette will fall back to Plotly defaults. "
        f"Original error: {exc}",
        RuntimeWarning,
    )

from . import plot
from . import values

__all__ = [
    "plot",
    "values"
]