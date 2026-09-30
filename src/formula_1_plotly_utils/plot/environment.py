from __future__ import annotations
from typing import Sequence, Literal, Optional
import warnings

import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .._core import geometry
from .._core import layout
from .._core import telemetry
from .._core.annotations import MarkerInput, _add_markers, _add_track_status, _normalize_markers, _time_x
from ..values import constants
from ..values.colors import category_palette, condition_colors
from ..values.informations import _to_minutes


_METRIC_STYLE = {
    'elevation': dict(name='Elevation Gradient', unit='%'),
    'speed': dict(name='Speed', unit='km/h'),
    'lat_g': dict(name='Lateral G', unit='g'),
    'lon_g': dict(name='Longitudinal G', unit='g'),
}


def _cumulative_distance(position: pd.DataFrame) -> np.ndarray:
    """Distance along the track in meters (fastf1 X/Y are in 1/10 m)."""
    if 'Distance' in position.columns:
        return position['Distance'].to_numpy(dtype=float)
    steps = np.sqrt(position['X'].diff().fillna(0) ** 2 + position['Y'].diff().fillna(0) ** 2)
    return (steps.cumsum() / 10).to_numpy()


def _distance_markers(markers: MarkerInput):
    normalized = _normalize_markers(markers)
    valid = [m for m in normalized if m.distance is not None]
    if len(valid) < len(normalized):
        warnings.warn("Markers without 'distance' cannot be placed on track charts and are skipped.", stacklevel=3)
    return valid


def plot_track(
    position: pd.DataFrame,
    circuit_info: Optional['fastf1.mvapi.CircuitInfo'] = None,
    reference_altitude: int = 0,
    metrics: Sequence[Literal['elevation', 'speed', 'lat_g', 'lon_g']] = ('elevation',),
    all_telemetry: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    markers: MarkerInput = None,
) -> go.Figure:
    """
    Plot the track layout colored by metrics using subplots (max 2 columns).

    Parameters:
    -----------
    position : pd.DataFrame
        Position data with 'X', 'Y', 'Z' (e.g. ``lap.get_telemetry()``).
    circuit_info : fastf1 CircuitInfo, optional
        Used for track rotation and corner annotations.
    metrics : sequence of 'elevation', 'speed', 'lat_g', 'lon_g'
    all_telemetry : pd.DataFrame, optional
        Telemetry of several laps/drivers to average the metrics over.
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``distance`` in m along the lap).
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    num_metrics = len(metrics)
    cols = min(2, num_metrics)
    rows = int(np.ceil(num_metrics / cols))

    # Pre-process telemetry data once
    if all_telemetry is not None:
        group_cols = [c for c in ['Driver', 'LapNumber'] if c in all_telemetry.columns]
        if group_cols:
            processed_chunks = [telemetry._compute_telemetry_metrics(group) for _, group in all_telemetry.groupby(group_cols)]
            tel_df = pd.concat(processed_chunks)
        else:
            tel_df = telemetry._compute_telemetry_metrics(all_telemetry)
    else:
        tel_df = telemetry._compute_telemetry_metrics(position)

    # Rotate track map once
    track = position[['X', 'Y']].to_numpy()
    if circuit_info and hasattr(circuit_info, 'rotation'):
        track_angle = circuit_info.rotation / 180 * np.pi
        rotated_track = geometry._rotate(track, angle=track_angle)
    else:
        track_angle = 0
        rotated_track = track
    track_distance = _cumulative_distance(position)
    distance_markers = _distance_markers(markers)

    titles = [_METRIC_STYLE.get(m.lower(), dict(name=m))['name'] for m in metrics]

    # Calculate spacing offsets so colorbars don't overlap in subplots
    horizontal_spacing = 0.15 if cols > 1 else 0.1
    vertical_spacing = 0.12 if rows > 1 else 0.1

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing
    )

    for idx, metric in enumerate(metrics):
        r = idx // cols + 1
        c = idx % cols + 1
        m_key = metric.lower()
        style = _METRIC_STYLE.get(m_key, dict(name=metric, unit=''))

        # Retrieve metric values
        if all_telemetry is not None:
            if 'Distance' in position.columns and 'Distance' in tel_df.columns:
                ref_dist = position['Distance'].values
                bins = np.concatenate([[-np.inf], (ref_dist[:-1] + ref_dist[1:]) / 2, [np.inf]])
                tel_df['dist_bin'] = pd.cut(tel_df['Distance'], bins=bins, labels=False)

                avg_series = tel_df.groupby('dist_bin')[m_key].mean()
                metric_values = avg_series.reindex(range(len(ref_dist))).bfill().ffill().values
            else:
                metric_values = tel_df[m_key].values[:len(position)]
        else:
            metric_values = tel_df[m_key].values

        max_abs_val = float(np.nanmax(np.abs(metric_values))) if len(metric_values) > 0 else 1.0
        min_val = float(np.nanmin(metric_values)) if len(metric_values) > 0 else 0.0
        max_val = float(np.nanmax(metric_values)) if len(metric_values) > 0 else 1.0

        # Calculate exact colorbar coordinates per subplot cell
        col_width = (1 - (cols - 1) * horizontal_spacing) / cols
        row_height = (1 - (rows - 1) * vertical_spacing) / rows
        x_pos = (c - 1) * (col_width + horizontal_spacing) + col_width
        y_pos = 1 - (r - 1) * (row_height + vertical_spacing) - (row_height / 2)

        marker_opts = {
            'size': 5,
            'color': metric_values,
            'opacity': 0.9,
            'colorbar': dict(
                len=row_height * 0.85,
                x=x_pos + 0.01,
                y=y_pos,
                thickness=12,
                title=layout._axis_title(style['name'], style.get('unit')),
            )
        }

        if m_key == 'lon_g':
            bound = max(max_abs_val, 1.0)
            marker_opts.update({'colorscale': 'RdBu_r', 'cmid': 0.0, 'cmin': -bound, 'cmax': bound})
        elif m_key == 'elevation':
            bound = max(max_abs_val, 0.5)
            marker_opts.update({'colorscale': 'Spectral_r', 'cmid': 0.0, 'cmin': -bound, 'cmax': bound})
        elif m_key == 'lat_g':
            marker_opts.update({'colorscale': 'Magma', 'cmin': 0.0, 'cmax': max(max_val, 1.0)})
        else:  # 'speed'
            marker_opts.update({'colorscale': 'Turbo', 'cmin': min_val, 'cmax': max_val})

        fig.add_trace(layout._track_outline(rotated_track[:, 0], rotated_track[:, 1]), row=r, col=c)
        fig.add_trace(
            go.Scatter(
                x=rotated_track[:, 0],
                y=rotated_track[:, 1],
                mode='markers',
                marker=marker_opts,
                customdata=track_distance,
                hovertemplate=(
                    f"{style['name']}: %{{marker.color:.2f}} {style.get('unit', '')}"
                    "<br>Distance: %{customdata:.0f} m<extra></extra>"
                ),
                showlegend=False
            ),
            row=r, col=c
        )

        # Add corner annotations
        if circuit_info and hasattr(circuit_info, 'corners'):
            for _, corner in circuit_info.corners.iterrows():
                txt = f"{corner['Number']}{corner['Letter']}"
                track_x, track_y = geometry._rotate([corner['X'], corner['Y']], angle=track_angle)
                fig.add_annotation(
                    x=track_x,
                    y=track_y,
                    text=txt,
                    showarrow=False,
                    bgcolor=layout.GRID,
                    font=dict(color=layout.TEXT, size=10),
                    row=r, col=c
                )

        def _resolve(m):
            idx_nearest = int(np.abs(track_distance - m.distance).argmin())
            return rotated_track[idx_nearest, 0], rotated_track[idx_nearest, 1]
        _add_markers(fig, distance_markers, _resolve, row=r, col=c)

        # 1:1 aspect ratio, no axes on a track map
        axis_num = (r - 1) * cols + c
        anchor_target = f"x{axis_num}" if axis_num > 1 else "x"
        fig.update_yaxes(scaleanchor=anchor_target, scaleratio=1, visible=False, row=r, col=c)
        fig.update_xaxes(visible=False, row=r, col=c)

    layout._apply_layout(
        fig,
        title=title or ('Track Map: ' + ', '.join(titles)),
        subtitle=layout._session_subtitle(position),
        height=max(600, 500 * rows),
    )
    return fig


def plot_track_elevation(
        position: pd.DataFrame,
        circuit_info: Optional['fastf1.mvapi.CircuitInfo'] = None,
        reference_altitude: int = 0,
        title: Optional[str] = None,
        markers: MarkerInput = None,
    ) -> go.Figure:
    """Plot the altitude gradient along the track with corner annotations.

    Parameters:
        position: Dataframe containing 'X', 'Y', and 'Z' coordinates.
            Usually obtained from :func:`fastf1.core.Telemetry.get_pos_data`.
        circuit_info (Optional): Circuit information containing corner
            locations.
        reference_altitude (Optional): An offset value added to the altitude
            shown in the hover text (e.g. to normalize to sea level).
        title (Optional): Custom chart title.
        markers (Optional): Events to highlight (``distance`` in m).

    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    delta_x = position['X'].diff().fillna(0)
    delta_y = position['Y'].diff().fillna(0)
    distances = np.sqrt(delta_x**2 + delta_y**2)
    cumulative_distance = (distances.cumsum() / 10).to_numpy()  # 1/10 m -> m

    altitude_meters = position['Z'].to_numpy() / 10 + reference_altitude
    altitude_diff = position['Z'].diff().fillna(0)
    altitude_gradient = np.where(distances > 0, (altitude_diff / distances.replace(0, np.nan)) * 100, 0)
    altitude_gradient = np.nan_to_num(altitude_gradient)

    bound = float(np.max(np.abs(altitude_gradient))) or 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_distance,
        y=altitude_gradient,
        mode='lines',
        line=dict(color=layout.MUTED, width=1),
        hoverinfo='skip',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=cumulative_distance,
        y=altitude_gradient,
        mode='markers',
        marker=dict(
            size=4, color=altitude_gradient, colorscale='Spectral_r', cmin=-bound, cmax=bound, cmid=0,
            colorbar=dict(title=layout._axis_title('Gradient', '%'), thickness=12),
        ),
        customdata=altitude_meters,
        hovertemplate='Distance: %{x:.0f} m<br>Gradient: %{y:+.2f} %<br>Altitude: %{customdata:.1f} m<extra></extra>',
        showlegend=False,
    ))

    # vertical lines for corner information
    if circuit_info is not None:
        for _, corner in circuit_info.corners.iterrows():
            distances_to_corner = np.sqrt((position['X'] - corner['X'])**2 + (position['Y'] - corner['Y'])**2)
            corner_distance = cumulative_distance[int(np.argmin(distances_to_corner.to_numpy()))]
            fig.add_vline(
                x=corner_distance,
                line=dict(width=1, dash='dash', color=layout.MUTED),
                annotation_text=f"C{corner['Number']}{corner['Letter']}",
                annotation_position="bottom right",
                annotation_font=dict(color=layout.MUTED, size=10),
            )

    _add_markers(fig, _distance_markers(markers), lambda m: (
        m.distance,
        m.y if m.y is not None else float(np.interp(m.distance, cumulative_distance, altitude_gradient)),
    ))

    layout._apply_layout(
        fig,
        title=title or 'Altitude Gradient along the Track',
        subtitle=layout._session_subtitle(position),
        x_title=layout._axis_title('Distance', 'm'),
        y_title=layout._axis_title('Altitude Gradient', '%'),
    )
    return fig


def plot_weather_data(
        weather_data: pd.DataFrame,
        airTemp: bool = True,
        trackTemp: bool = True,
        humidity: bool = True,
        pressure: bool = True,
        windSpeed: bool = True,
        track_status: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        markers: MarkerInput = None,
    ) -> go.Figure:
    """Plot weather metrics over the session time.

    One row per metric group (temperatures share a row) with a common time axis.
    Rain is highlighted by blue bands.

    Parameters
    ----------
    weather_data : pd.DataFrame
        fastf1 ``session.weather_data``.
    airTemp, trackTemp, humidity, pressure, windSpeed : bool
        Select the metrics to plot.
    track_status : pd.DataFrame, optional
        fastf1 ``session.track_status`` to highlight SC/VSC/red flag phases.
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``time`` as session time).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    data = weather_data.copy()
    data['Minutes'] = data['Time'].dt.total_seconds() / 60

    panels = []
    temps = [(col, name) for col, name, enabled in (
        ('AirTemp', 'Air', airTemp), ('TrackTemp', 'Track', trackTemp)) if enabled]
    if temps:
        panels.append(('Temperature', '°C', temps))
    for col, name, enabled in (
        ('Humidity', 'Humidity', humidity), ('Pressure', 'Pressure', pressure), ('WindSpeed', 'Wind Speed', windSpeed)
    ):
        if enabled:
            panels.append((name, constants.weather_units[col], [(col, name)]))

    rows = max(1, len(panels))
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04)

    color_idx = 0
    for r, (panel_name, unit, series) in enumerate(panels, start=1):
        for col, name in series:
            fig.add_trace(
                go.Scatter(
                    x=data['Minutes'], y=data[col], mode='lines', name=name,
                    line=dict(color=category_palette[color_idx % len(category_palette)], width=2),
                    hovertemplate=f'{name}: %{{y:.1f}} {unit}<extra></extra>',
                ),
                row=r, col=1,
            )
            color_idx += 1
        fig.update_yaxes(title_text=layout._axis_title(panel_name, unit), row=r, col=1)

    # rain bands
    rain = data[data['Rainfall'].astype(bool)].copy()
    if not rain.empty:
        rain['rain_group'] = (rain['Time'].diff() > pd.to_timedelta(65, unit='s')).cumsum()
        for _, group_df in rain.groupby('rain_group'):
            fig.add_vrect(
                x0=group_df['Minutes'].min(), x1=group_df['Minutes'].max(),
                fillcolor=condition_colors['Rain'], opacity=0.2, layer='below', line_width=0,
                row='all', col=1,
            )
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol='square', size=12, color=condition_colors['Rain'], opacity=0.5),
            name='Rain', hoverinfo='skip',
        ), row=1, col=1)

    _add_track_status(
        fig, track_status, _to_minutes, data['Time'].max(),
        row='all', col=1, label_row=1, label_col=1,
    )
    _add_markers(fig, markers, lambda m: (_time_x(m), None), row='all', col=1, label_row=1, label_col=1)

    layout._apply_layout(
        fig,
        title=title or 'Weather Data During the Session',
        subtitle=layout._session_subtitle(weather_data),
        legend_title='Metric',
        height=max(450, 200 * rows + 150),
    )
    fig.update_xaxes(title_text=layout._axis_title('Session Time', 'min'), row=rows, col=1)
    fig.update_layout(hovermode='x unified')
    return fig
