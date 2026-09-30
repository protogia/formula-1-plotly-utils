from __future__ import annotations

from ..values import constants
from ..values.colors import _colors_similar, _driver_styles, category_palette
from ..values.informations import _LapTimeline

from .._core import geometry
from .._core import layout
from .._core import telemetry
from .._core.annotations import (
    MarkerInput, _add_markers, _add_track_status_on_laps, _lap_x, _normalize_markers,
)
from . import overview

import pandas as pd
import numpy as np

from typing import Optional, List, Union
import warnings

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d


def plot_gap_between_d1_d2(
        laps: pd.DataFrame,
        driver1_code: str,
        driver2_code: str,
        track_status: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        markers: MarkerInput = None,
    ) -> go.Figure:
    """
    Plots the gap between two drivers at the end of each lap.

    A positive gap means driver2 is behind driver1. The area is filled in the
    color of the driver who is ahead.

    Parameters:
    -----------
    laps : pd.DataFrame
        The FastF1 laps DataFrame (session.laps).
    driver1_code, driver2_code : str
        Driver abbreviations, e.g. 'VER'.
    track_status : pd.DataFrame, optional
        fastf1 ``session.track_status`` to highlight SC/VSC/red flag phases.
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``lap`` or ``time``).
    """
    d1_laps = laps[laps['Driver'] == driver1_code][['LapNumber', 'Time']].dropna()
    d2_laps = laps[laps['Driver'] == driver2_code][['LapNumber', 'Time']].dropna()

    # merge to align lap numbers
    gap_df = pd.merge(d1_laps, d2_laps, on='LapNumber', suffixes=('_d1', '_d2')).sort_values('LapNumber')

    # driver2_code is chasing driver1_code, a positive value indicates d2 is behind d1
    gap_df['Gap'] = (gap_df['Time_d2'] - gap_df['Time_d1']).dt.total_seconds()

    styles = _driver_styles(laps, [driver1_code, driver2_code])
    fig = go.Figure()

    for driver, values in (
        (driver1_code, gap_df['Gap'].clip(lower=0)),
        (driver2_code, gap_df['Gap'].clip(upper=0)),
    ):
        fig.add_trace(go.Scatter(
            x=gap_df['LapNumber'], y=values, mode='lines', line=dict(width=0),
            fill='tozeroy', fillcolor=layout._rgba(styles[driver].color, 0.35),
            name=f'{driver} ahead', hoverinfo='skip',
        ))

    fig.add_trace(go.Scatter(
        x=gap_df['LapNumber'],
        y=gap_df['Gap'],
        mode='lines+markers',
        name=f'Gap {driver2_code} to {driver1_code}',
        line=dict(color=layout.TEXT, width=2),
        marker=dict(size=5),
        hovertemplate='Lap %{x}<br>Gap: %{y:+.3f} s<extra></extra>',
    ))

    timeline = _LapTimeline(laps)
    _add_track_status_on_laps(fig, track_status, timeline)

    def _resolve(m):
        x = _lap_x(m, timeline)
        if m.y is not None or x is None:
            return x, m.y
        return x, float(np.interp(x, gap_df['LapNumber'], gap_df['Gap']))
    _add_markers(fig, markers, _resolve)

    layout._apply_layout(
        fig,
        title=title or f"Gap between {driver1_code} and {driver2_code}",
        subtitle=layout._session_subtitle(laps),
        x_title=layout._axis_title('Lap'),
        y_title=layout._axis_title(f'Gap {driver2_code} to {driver1_code}', 's'),
    )
    fig.update_layout(hovermode='x unified')
    return fig


def _pick_lap(laps: pd.DataFrame, driver: str, lap: Union[str, int]):
    driver_laps = laps.pick_drivers(driver)
    if driver_laps.empty:
        return None
    if str(lap) == 'fastest':
        return driver_laps.pick_fastest()
    selected = driver_laps[driver_laps['LapNumber'] == int(lap)]
    return selected.iloc[0] if not selected.empty else None


def plot_lap_telemetry_comparison(
    laps: pd.DataFrame,
    circuit_info: Optional['fastf1.mvapi.CircuitInfo'],
    driver1_code: str,
    driver2_code: str,
    driver1_lap: Union[str, int] = 'fastest',
    driver2_lap: Union[str, int] = 'fastest',
    metrics_to_plot: Optional[List[str]] = None,
    title: Optional[str] = None,
    markers: MarkerInput = None,
) -> Union[go.Figure, List[go.Figure], None]:
    """
    Compares the telemetry of two laps (one figure per metric): track map colored
    by the difference (left) and the metric over distance (right).

    Parameters:
    -----------
    laps : fastf1.core.Laps
        The FastF1 laps (session.laps), telemetry must be loaded.
    circuit_info : fastf1 CircuitInfo, optional
        Used for track rotation and corner annotations.
    driver1_code, driver2_code : str
        Driver abbreviations (can be the same driver to compare two laps).
    driver1_lap, driver2_lap : 'fastest' or lap number
    metrics_to_plot : list, optional
        Telemetry channels, default: Speed, Throttle, Brake, RPM, nGear.
    title : str, optional
        Custom chart title (the metric is appended if several metrics are plotted).
    markers : ChartMarker | dict | list, optional
        Events to highlight (``distance`` in m, optionally ``driver``).
    """
    lap1 = _pick_lap(laps, driver1_code, driver1_lap)
    lap2 = _pick_lap(laps, driver2_code, driver2_lap)

    if lap1 is None or lap2 is None or lap1.empty or lap2.empty:
        print(f"One or both laps are missing or empty for Driver 1 ({driver1_code}, Lap {driver1_lap}) or Driver 2 ({driver2_code}, Lap {driver2_lap}).")
        return None

    # telemetry (car + position data incl. distance)
    try:
        tel1 = lap1.get_telemetry()
        tel2 = lap2.get_telemetry()
        if len(tel1) == 0 or len(tel2) == 0:
            raise ValueError("Telemetry is empty for one of the laps.")
    except Exception as e:
        print(f"Could not retrieve telemetry for {driver1_code} or {driver2_code}: {e}")
        return None

    lap1_num = int(lap1['LapNumber'])
    lap2_num = int(lap2['LapNumber'])
    lap1_label = f"{driver1_code} L{lap1_num}"
    lap2_label = f"{driver2_code} L{lap2_num}"

    # driver colors; similar colors (teammates, same driver, ...) are hard to compare,
    # so the lines get two distinct standard colors and the legend shows the team color.
    # Stricter threshold than elsewhere: both lines overlap on most of the lap.
    styles = _driver_styles(laps, [driver1_code, driver2_code])
    team_color1, team_color2 = styles[driver1_code].color, styles[driver2_code].color
    if _colors_similar(team_color1, team_color2, threshold=300.0):
        color1, color2 = category_palette[0], category_palette[1]
        legend1 = f"{lap1_label} <span style='color:{team_color1}'>■</span>"
        legend2 = f"{lap2_label} <span style='color:{team_color2}'>■</span>"
    else:
        color1, color2 = team_color1, team_color2
        legend1, legend2 = lap1_label, lap2_label
    diff_colorscale = [[0, color2], [0.5, '#F2F2F2'], [1, color1]]

    track_angle = circuit_info.rotation / 180 * np.pi if circuit_info is not None else 0
    rot_coords = geometry._rotate(tel1[['X', 'Y']].to_numpy(), angle=track_angle)

    corner_distances = []
    if circuit_info is not None:
        for _, corner in circuit_info.corners.iterrows():
            closest = ((tel1['X'] - corner['X'])**2 + (tel1['Y'] - corner['Y'])**2).idxmin()
            corner_distances.append((tel1.loc[closest, 'Distance'], f"{corner['Number']}{corner['Letter']}", corner))

    available_columns = set(tel1.columns).intersection(set(tel2.columns))
    candidates = metrics_to_plot or ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear']
    metrics = [m for m in candidates if m in available_columns]

    distance_markers = [m for m in _normalize_markers(markers) if m.distance is not None]
    if len(distance_markers) < len(_normalize_markers(markers)):
        warnings.warn("Markers without 'distance' cannot be placed on telemetry charts and are skipped.", stacklevel=2)

    max_dist = max(tel1['Distance'].max(), tel2['Distance'].max())
    figures = []

    for metric in metrics:
        unit = constants.telemetry_units.get(metric, '')
        values1 = tel1[metric].astype(float)
        values2 = tel2[metric].astype(float)

        fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], horizontal_spacing=0.06)

        for label, legend_name, tel, values, color in (
            (lap1_label, legend1, tel1, values1, color1),
            (lap2_label, legend2, tel2, values2, color2),
        ):
            fig.add_trace(
                go.Scatter(
                    x=tel['Distance'], y=values, mode='lines', name=legend_name,
                    line=dict(color=color, width=2),
                    hovertemplate=f'{label}<br>Distance: %{{x:.0f}} m<br>{metric}: %{{y:.1f}} {unit}<extra></extra>',
                ),
                row=1, col=2,
            )

        # corner lines on telemetry plot
        for corner_dist, corner_label, _ in corner_distances:
            fig.add_vline(
                x=corner_dist, line=dict(width=1, dash='dash', color=layout.MUTED),
                annotation_text=corner_label, annotation_position='bottom right',
                annotation_font=dict(color=layout.MUTED, size=10),
                row=1, col=2,
            )

        # difference on track map
        tel2_interp = interp1d(tel2['Distance'], values2, kind='linear', fill_value="extrapolate")(tel1['Distance'])
        diff = values1 - tel2_interp
        max_diff = float(np.nanmax(np.abs(diff))) or 1.0

        fig.add_trace(layout._track_outline(rot_coords[:, 0], rot_coords[:, 1]), row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=rot_coords[:, 0], y=rot_coords[:, 1],
                mode='markers', showlegend=False,
                marker=dict(
                    size=5, color=diff, colorscale=diff_colorscale,
                    cmin=-max_diff, cmax=max_diff, cmid=0, showscale=True,
                    colorbar=dict(
                        orientation='h', x=0.2, xanchor='center', y=-0.12, len=0.36, thickness=12,
                        title=dict(text=f"Higher {metric}", side='top'),
                        tickvals=[-max_diff, 0, max_diff],
                        ticktext=[lap2_label, "Equal", lap1_label],
                    ),
                ),
                customdata=tel1['Distance'],
                hovertemplate=f'Distance: %{{customdata:.0f}} m<br>{metric} difference: %{{marker.color:+.1f}} {unit}<extra></extra>',
            ),
            row=1, col=1,
        )

        # corner annotations on map
        for _, corner_label, corner in corner_distances:
            track_x, track_y = geometry._rotate([corner['X'], corner['Y']], angle=track_angle)
            fig.add_annotation(
                x=track_x, y=track_y, text=corner_label, showarrow=False,
                bgcolor=layout.GRID, font=dict(color=layout.TEXT, size=10),
                row=1, col=1,
            )

        # markers: telemetry (right) and track map (left)
        def _resolve_tel(m, _v1=values1, _v2=values2):
            if m.y is not None:
                return m.distance, m.y
            if m.driver == driver1_code:
                return m.distance, float(np.interp(m.distance, tel1['Distance'], _v1))
            if m.driver == driver2_code:
                return m.distance, float(np.interp(m.distance, tel2['Distance'], _v2))
            return m.distance, None

        def _resolve_map(m):
            idx = int(np.abs(tel1['Distance'].to_numpy() - m.distance).argmin())
            return rot_coords[idx, 0], rot_coords[idx, 1]

        _add_markers(fig, distance_markers, _resolve_tel, row=1, col=2)
        _add_markers(fig, distance_markers, _resolve_map, row=1, col=1)

        default_title = f"{metric} Comparison: {driver1_code} (Lap {lap1_num}) vs {driver2_code} (Lap {lap2_num})"
        chart_title = (f"{title} – {metric}" if len(metrics) > 1 else title) if title else default_title

        layout._apply_layout(fig, title=chart_title, subtitle=layout._session_subtitle(laps), height=550)
        fig.update_xaxes(range=[0, max_dist], title_text=layout._axis_title('Distance', 'm'), row=1, col=2)
        fig.update_yaxes(title_text=layout._axis_title(metric, unit), row=1, col=2)
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_layout(margin=dict(b=110))

        figures.append(fig)

    return figures[0] if len(figures) == 1 else figures


def plot_laptime_evolution(
    laps: pd.DataFrame,
    drivers: List[str],
    track_status: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    markers: MarkerInput = None,
) -> go.Figure:
    """
    Plots the lap time per lap for the given drivers.

    The y-axis is zoomed to the racing pace, slow laps (pit stops, SC) stay
    available via zoom/hover.

    Parameters:
    -----------
    laps : pd.DataFrame
        The FastF1 laps DataFrame (session.laps).
    drivers : list
        Driver abbreviations.
    track_status : pd.DataFrame, optional
        fastf1 ``session.track_status`` to highlight SC/VSC/red flag phases.
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``lap`` or ``time``, optionally ``driver``).
    """
    laps_pace = laps[laps['Driver'].isin(drivers)].copy()
    laps_pace['LapTimeSeconds'] = laps_pace['LapTime'].dt.total_seconds()

    styles = _driver_styles(laps, drivers)

    fig = go.Figure()
    for driver in drivers:
        driver_laps = laps_pace[laps_pace['Driver'] == driver].sort_values('LapNumber')
        fig.add_trace(go.Scatter(
            x=driver_laps['LapNumber'],
            y=driver_laps['LapTimeSeconds'],
            mode='lines+markers',
            name=driver,
            line=dict(color=styles[driver].color, dash=styles[driver].dash, width=2),
            marker=dict(size=5),
            hovertemplate=f"{driver}: %{{y:.3f}} s<extra></extra>",
        ))

    timeline = _LapTimeline(laps)
    _add_track_status_on_laps(fig, track_status, timeline)

    def _resolve(m):
        x = _lap_x(m, timeline)
        if m.y is not None or x is None or m.driver not in drivers:
            return x, m.y
        driver_laps = laps_pace[laps_pace['Driver'] == m.driver].dropna(subset=['LapTimeSeconds']).sort_values('LapNumber')
        if driver_laps.empty:
            return x, None
        return x, float(np.interp(x, driver_laps['LapNumber'], driver_laps['LapTimeSeconds']))
    _add_markers(fig, markers, _resolve)

    # zoom to racing pace
    pace = telemetry._filter_slow_laps(laps_pace, group_columns=['Driver'])['LapTimeSeconds']
    y_range = None
    if not pace.empty:
        y_range = [laps_pace['LapTimeSeconds'].min() - 0.5, pace.max() + 0.5]

    layout._apply_layout(
        fig,
        title=title or 'Lap Time Evolution',
        subtitle=layout._session_subtitle(laps),
        x_title=layout._axis_title('Lap'),
        y_title=layout._axis_title('Lap Time', 's'),
        legend_title='Driver',
    )
    fig.update_layout(hovermode='x unified', yaxis_range=y_range)
    return fig


def plot_standings_evolution_of_lap(
    laps: pd.DataFrame,
    results: pd.DataFrame,
    title: Optional[str] = None,
    markers: MarkerInput = None,
) -> go.Figure:
    """Positions gained/lost on lap 1 (grid vs. end of lap 1)."""
    return overview.plot_position_evolution_for_lap_x(
        laps, results, 1,
        title=title or 'Positions Gained/Lost on Lap 1 (Grid vs Lap 1 Finish)',
        markers=markers,
    )
