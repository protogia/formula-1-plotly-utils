from __future__ import annotations
from typing import Sequence, Literal, Optional

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots

from _core import geometry
from _core import telemetry

from fastf1.plotting._plotting import _COLOR_PALETTE
from fastf1.mvapi import CircuitInfo



def plot_track(
    position: pd.DataFrame,
    circuit_info: Optional['fastf1.mvapi.CircuitInfo'] = None,
    reference_altitude: int = 0,
    metrics: Sequence[Literal['elevation', 'speed', 'lat_g', 'lon_g']] = ('elevation',),
    all_telemetry: Optional[pd.DataFrame] = None
) -> go.Figure:
    """Plot the track layout with customizable metrics using subplots (max 2 columns)."""
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
        rotated_track = geometry.rotate(track, angle=track_angle)
    else:
        track_angle = 0
        rotated_track = track

    titles = [m.replace('_', ' ').title() for m in metrics]
    
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
            'opacity': 0.85,
            'colorbar': dict(
                len=row_height * 0.85,
                x=x_pos + 0.01,
                y=y_pos,
                thickness=12
            )
        }

        if m_key == 'lon_g':
            bound = max(max_abs_val, 1.0)
            marker_opts.update({
                'colorscale': 'RdBu_r',
                'cmid': 0.0,
                'cmin': -bound,
                'cmax': bound,
            })
            marker_opts['colorbar']['title'] = 'Longitudinal G (g)'
            hover_text = [f"Lon G: {v:+.2f}g" for v in metric_values]

        elif m_key == 'elevation':
            bound = max(max_abs_val, 0.5)
            marker_opts.update({
                'colorscale': 'Spectral_r',
                'cmid': 0.0,
                'cmin': -bound,
                'cmax': bound,
            })
            marker_opts['colorbar']['title'] = 'Elevation Gradient (%)'
            hover_text = [f"Gradient: {v:+.2f}%" for v in metric_values]

        elif m_key == 'lat_g':
            marker_opts.update({
                'colorscale': 'Magma',
                'cmin': 0.0,
                'cmax': max(max_val, 1.0),
            })
            marker_opts['colorbar']['title'] = 'Lateral G (g)'
            hover_text = [f"Lat G: {v:.2f}g" for v in metric_values]

        else:  # 'speed'
            marker_opts.update({
                'colorscale': 'Turbo',
                'cmin': min_val,
                'cmax': max_val,
            })
            marker_opts['colorbar']['title'] = 'Speed (km/h)'
            hover_text = [f"Speed: {v:.1f} km/h" for v in metric_values]

        fig.add_trace(
            go.Scatter(
                x=rotated_track[:, 0],
                y=rotated_track[:, 1],
                mode='lines+markers',
                marker=marker_opts,
                line=dict(color=_COLOR_PALETTE[0], width=4),
                hoverinfo='text',
                text=hover_text,
                showlegend=False
            ),
            row=r, col=c
        )

        # Add corner annotations
        if circuit_info and hasattr(circuit_info, 'corners'):
            for _, corner in circuit_info.corners.iterrows():
                txt = f"{corner['Number']}{corner['Letter']}"
                track_x, track_y = geometry.rotate([corner['X'], corner['Y']], angle=track_angle)
                fig.add_annotation(
                    x=track_x,
                    y=track_y,
                    text=txt,
                    showarrow=False,
                    bgcolor="grey",
                    font=dict(color="white", size=10),
                    row=r, col=c
                )

        # Configure aspect ratio for 1:1 mapping scale
        axis_num = (r - 1) * cols + c
        
        # Plotly layout keys: xaxis, yaxis, xaxis2, yaxis2, etc.
        x_axis_key = f"xaxis{axis_num}" if axis_num > 1 else "xaxis"
        y_axis_key = f"yaxis{axis_num}" if axis_num > 1 else "yaxis"
        
        # Valid scaleanchor target values: x, x2, x3, etc.
        anchor_target = f"x{axis_num}" if axis_num > 1 else "x"

        fig.layout[y_axis_key].update(
            scaleanchor=anchor_target, 
            scaleratio=1,
            showgrid=False, 
            zeroline=False, 
            showticklabels=False
        )
        fig.layout[x_axis_key].update(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False
        )

    return fig


def plot_track_elevation(
        position: pd.DataFrame,
        circuit_info: Optional['fastf1.mvapi.CircuitInfo'] = None,
        reference_altitude: int = 0
    ) -> 'plotly.graph_objects.Figure': 
    """Plot the track elevation with corner annotations 
    using Plotly.

    The plot is interactive, allowing for zooming and hovering to see 
    specific altitude gradients and corner details.

    Parameters:
        position: Dataframe containing 'X', 'Y', and 'Z' coordinates. 
            Usually obtained from :func:`fastf1.core.Telemetry.get_pos_data`.
        circuit_info (Optional): Circuit information containing corner 
            locations and track rotation.
        reference_altitude (Optional): An offset value added to the 'Z' coordinate 
            (useful for normalizing altitude to sea level or track minimum).

    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """

    # calculate the distance along the track
    # difference in x and y between consecutive points
    delta_x = position['X'].diff().fillna(0)
    delta_y = position['Y'].diff().fillna(0)

    # distance between consecutive points
    distances = np.sqrt(delta_x**2 + delta_y**2)

    # cumulative distance along track
    cumulative_distance = distances.cumsum()/10

    # aclc gradient
    altitude_meters = position['Z'].values + reference_altitude
    altitude_diff = position['Z'].diff().fillna(0)

    altitude_gradient = np.where(distances > 0, (altitude_diff / distances) * 100, 0)

    # color scale based on the altitude gradient values
    colorscale = 'Plasma'
    min_gradient, max_gradient = np.min(altitude_gradient), np.max(altitude_gradient)

    plasma_colors = pcolors.get_colorscale(colorscale)

    # list of segments with start and end points and corresponding gradient and color
    segments = []
    for i in range(len(altitude_gradient) - 1):
        segment_gradient = (altitude_gradient[i] + altitude_gradient[i+1]) / 2 # Average gradient for the segment
        normalized_segment_gradient = (segment_gradient - min_gradient) / (max_gradient - min_gradient) if (max_gradient - min_gradient) != 0 else 0

        # interpolate color from colorscale
        segment_color = pcolors.sample_colorscale(plasma_colors, normalized_segment_gradient)[0]
        segment = {
            'x': [cumulative_distance.iloc[i], cumulative_distance.iloc[i+1]], 
            'y': [altitude_gradient[i], altitude_gradient[i+1]],
            'gradient': segment_gradient,
            'color': segment_color 
        }
        segments.append(segment)

    fig = go.Figure()

    for segment in segments:
        fig.add_trace(go.Scatter(
            x=segment['x'],
            y=segment['y'],
            mode='lines',
            line=dict(color=segment['color'], width=2), # color the line by segment gradient
            hoverinfo='text',
            text=f'Altitude Gradient: {segment["gradient"]:.2f}',
            showlegend=False 
        ))

    fig.add_trace(go.Scatter(
        x=[None], 
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title='Altitude Gradient'),
            cmin=min_gradient,
            cmax=max_gradient,
            color=altitude_gradient 
        ),
        hoverinfo='none',
        showlegend=False
    ))

    # vertical lines for corner information
    for _, corner in circuit_info.corners.iterrows():
        # match X, Y and cumulatative distance via index
        distances_to_corner = np.sqrt((position['X'] - corner['X'])**2 + (position['Y'] - corner['Y'])**2)
        closest_pos_index = distances_to_corner.idxmin()
        corner_cumulative_distance = cumulative_distance.iloc[closest_pos_index]

        fig.add_vline(
            x=corner_cumulative_distance,
            line_width=1,
            line_dash="dash",
            line_color="red",
            annotation_text=f"C-{corner['Number']}{corner['Letter']}",
            annotation_position="top right"
        )

    fig.update_layout(
        title='Altitude Gradient Along the Track with Corners',
        xaxis_title='Distance along Track [m]', # Update x-axis title
        yaxis_title='Altitude Gradient [%]',
    )
    return fig


def plot_weather_data(
        weather_data: pd.DataFrame,
        airTemp: bool = True,
        trackTemp: bool = True,
        humidity: bool = True,
        pressure: bool = True,
        windSpeed: bool = True,
    ) -> 'plotly.graph_objects.Figure':
    """Plot multiple weather metrics over time.

    Creates an interactive Plotly figure containing optional sub‑plots for
    air temperature, track temperature, humidity, pressure and wind speed.
    Rain events are highlighted by shading the corresponding time intervals.

    Parameters
    ----------
    weather_data : pd.DataFrame
        DataFrame containing at least the columns ``Time``, ``AirTemp``,
        ``TrackTemp``, ``Humidity``, ``Pressure``, ``WindSpeed`` and
        ``Rainfall`` (boolean).  The ``Time`` column should be a datetime
        type.
    airTemp : bool, default=True
        If ``True`` plot the air temperature trace.
    trackTemp : bool, default=True
        If ``True`` plot the track temperature trace.
    humidity : bool, default=True
        If ``True`` plot the humidity trace.
    pressure : bool, default=True
        If ``True`` plot the atmospheric pressure trace.
    windSpeed : bool, default=True
        If ``True`` plot the wind‑speed trace.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure with the selected weather traces and a
        shaded region for rain periods.

    """
    # time column to string for plotting
    weather_data_str_time = weather_data.copy()
    weather_data_str_time['Time_str'] = weather_data_str_time['Time'].apply(lambda x: str(x).split(' ')[-1]) # Extract HH:MM:SS

    # Create subplots with multiple y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if airTemp:
        fig.add_trace(
            go.Scatter(x=weather_data_str_time['Time_str'], y=weather_data_str_time['AirTemp'], name='Air Temp'),
            secondary_y=False,
        ) 

    if trackTemp:
        fig.add_trace(
            go.Scatter(x=weather_data_str_time['Time_str'], y=weather_data_str_time['TrackTemp'], name='Track Temp'),
            secondary_y=False,
        )

    if humidity:
        fig.add_trace(
            go.Scatter(x=weather_data_str_time['Time_str'], y=weather_data_str_time['Humidity'], name='Humidity'),
            secondary_y=True,
        )

    if pressure:
        fig.add_trace(
            go.Scatter(x=weather_data_str_time['Time_str'], y=weather_data_str_time['Pressure'], name='Pressure'),
            secondary_y=True,
        )

    if windSpeed:
        fig.add_trace(
            go.Scatter(x=weather_data_str_time['Time_str'], y=weather_data_str_time['WindSpeed'], name='Wind Speed'),
            secondary_y=True,
        )

    # ensure y-axis range is set
    fig.update_layout(
        title='Weather Data During the Race',
        xaxis_title='Time', # Keep Time as x-axis title
        legend_title='Metric'
    )

    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Value", secondary_y=True)

    # get the y-axis range after adding traces and updating layout
    y_range_primary = fig.layout.yaxis.range


    # shading to indicate rain
    rain_periods_str_time = weather_data_str_time[weather_data_str_time['Rainfall'] == True].copy()
    if not rain_periods_str_time.empty:
        rain_periods_str_time['rain_group'] = (rain_periods_str_time['Time'].diff() > pd.Timedelta(seconds=65)).cumsum()
        for group_id, group_df in rain_periods_str_time.groupby('rain_group'):
            start_time_str = group_df['Time_str'].min()
            end_time_str = group_df['Time_str'].max()

            y0_val = y_range_primary[0] if y_range_primary is not None else 0
            y1_val = y_range_primary[1] if y_range_primary is not None else 100 
            

            fig.add_shape(
                type="rect",
                x0=start_time_str,
                y0=y0_val,  # start at the bottom of the primary y-axis
                x1=end_time_str,
                y1=y1_val,  # end at the top of the primary y-axis
                fillcolor="blue",
                opacity=0.2,
                layer="below",
                line_width=0,
            )

        # single legend entry for rain
        fig.add_trace(go.Scatter(
            x=[None], y=[None], # invisible trace
            mode='markers',
            marker=dict(size=10, color="blue", opacity=0.5),
            legendgroup='Rain',
            showlegend=True,
            name='Rain'
        ))
    return fig
