"""
Public API of the *formula_1_plotly_utils* package.
"""
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


from .plot.environment import plot_weather_data, plot_track, plot_track_elevation
from .plot.strategy import plot_tyre_strategies, plot_total_pitstop_time
from .plot.lap_times import (
    plot_laptime_distribution_weatherdependent,
    plot_laptime_distribution_per_compound,
    plot_best_laptime,
)
from .plot.qualifying import (
    plot_qualifying_results,
    plot_driver_position_per_lap,
)
from .plot.gap import (
    plot_gap_between_d1_d2,
    plot_leading_laptime_evolution,
    plot_leading_laptimes,
)

__all__ = [
    "plot_track",
    "plot_track_elevation",
    "plot_weather_data",
    "plot_tyre_strategies",
    "plot_total_pitstop_time",
    "plot_laptime_distribution_weatherdependent",
    "plot_laptime_distribution_per_compound",
    "plot_best_laptime",
    "plot_qualifying_results",
    "plot_driver_position_per_lap",
    "plot_gap_between_d1_d2",
    "plot_leading_laptime_evolution",
    "plot_leading_laptimes",
]