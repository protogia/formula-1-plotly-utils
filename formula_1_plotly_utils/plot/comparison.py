from __future__ import annotations

from values import constants
from _core import geometry

import pandas as pd
import numpy as np

import fastf1.plotting

from typing import Optional, List, Dict

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d


def plot_gap_between_d1_d2(
        laps: pd.DataFrame,
        driver1_code: str,
        driver2_code: str,
        event_lap: int,
        event_label: str
    ):
    # Filter laps for both drivers
    d1_laps = laps.pick_driver(driver1_code)[['LapNumber', 'Time']]
    d2_laps = laps.pick_driver(driver2_code)[['LapNumber', 'Time']]

    d1_laps['TotalTime'] = d1_laps['Time'].dt.total_seconds()
    d2_laps['TotalTime'] = d2_laps['Time'].dt.total_seconds()

    # merge to align lap numbers
    gap_df = pd.merge(d1_laps, d2_laps, on='LapNumber', suffixes=(f'_{driver1_code}', f'_{driver2_code}'))

    # driver2_code is chasing driver1_code, a positive value indicates d2 is behind d1
    gap_df['Gap'] = gap_df[f'TotalTime_{driver2_code}'] - gap_df[f'TotalTime_{driver1_code}']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gap_df['LapNumber'],
        y=gap_df['Gap'],
        mode='lines+markers',
        name=f'Gap: {driver2_code} to {driver1_code}',
        fill='tozeroy'
    ))

    fig.update_layout(
        title=f"The Hunt: Gap between {driver1_code} and {driver2_code}",
        xaxis_title="Lap Number",
        yaxis_title="Gap (Seconds)",
    )

    if event_lap:
        fig.add_shape(dict(
            type="line", x0=event_lap, x1=event_lap, y0=gap_df['Gap'].min(), y1=gap_df['Gap'].max(),
            line=dict(color="red", dash="dash")
        ))
        if event_label:
            fig.add_annotation(x=event_lap, y=gap_df['Gap'].max(), text=event_label, showarrow=True)

    return fig



def plot_lap_telemetry_comparison(
    laps: pd.DataFrame,
    circuit_info: Optional['fastf1.mvapi.CircuitInfo'],
    driver1_code: str,
    driver2_code: str,
    driver1_lap: str,
    driver2_lap: str,
    metrics_to_plot: List = None,
    highlight_distance = None,
    highlight_label: str = None,
    custom_title: str = None
):
    # filter chosen laps
    if driver1_lap == 'fastest':
        driver1_laps_filtered = laps.pick_drivers(driver1_code)
        lap1 = driver1_laps_filtered.loc[driver1_laps_filtered['LapTime'].idxmin()] if not driver1_laps_filtered.empty else pd.Series()
    else:
        lap1 = laps.pick_drivers(driver1_code).pick_lap(int(driver1_lap) if str(driver1_lap).isdigit() else driver1_lap)

    if driver2_lap == 'fastest':
        driver2_laps_filtered = laps.pick_drivers(driver2_code)
        lap2 = driver2_laps_filtered.loc[driver2_laps_filtered['LapTime'].idxmin()] if not driver2_laps_filtered.empty else pd.Series()
    else:
        lap2 = laps.pick_drivers(driver2_code).pick_lap(int(driver2_lap) if str(driver2_lap).isdigit() else driver2_lap)

    if lap1.empty or lap2.empty:
        print(f"One or both laps are missing or empty for Driver 1 ({driver1_code}, Lap {driver1_lap}) or Driver 2 ({driver2_code}, Lap {driver2_lap}).")
        return None

    lap1_label = driver1_code
    lap2_label = driver2_code

    # telemetry & position data
    try:
        tel1 = lap1.get_telemetry()
        tel2 = lap2.get_telemetry()
        pos1 = lap1.get_pos_data()
        
        if len(tel1) == 0 or len(tel2) == 0:
            raise ValueError("Telemetry is empty for one of the laps.")
    except Exception as e:
        print(f"Could not retrieve telemetry for {lap1_label} or {lap2_label}: {e}")
        return None

    # Track rotation angle
    track_angle = circuit_info.rotation / 180 * np.pi if circuit_info is not None else 0

    lap1_num = int(lap1['LapNumber']) if 'LapNumber' in lap1 else "N/A"
    lap2_num = int(lap2['LapNumber']) if 'LapNumber' in lap2 else "N/A"

    comparison_title_suffix = custom_title if custom_title else f"{lap1_label} (Lap {lap1_num}) vs {lap2_label} (Lap {lap2_num})"

    available_columns = set(tel1.columns).intersection(set(tel2.columns))
    
    # Ensure fallback metrics if definition module isn't loaded
    if metrics_to_plot is None:
        metrics = [m for m in ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear'] if m in available_columns]
    else:
        metrics = [m for m in metrics_to_plot if m in available_columns]

    color1, color2 = 'red', 'lightblue'
    max_dist = max(tel1['Distance'].max(), tel2['Distance'].max())
    
    figures = []

    for metric in metrics:
        unit = constants.telemetry_units.get(metric, '') if 'definitions' in globals() else ''
        fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], horizontal_spacing=0.05)

        # FIXED: hovertemplate is now properly INSIDE go.Scatter()
        fig.add_trace(
            go.Scatter(
                x=tel1['Distance'], 
                y=tel1[metric], 
                mode='lines', 
                name=f"{lap1_label} L{lap1_num}", 
                line=dict(color=color1), 
                legendgroup="l1",
                hovertemplate=f'Distance: %{{x:.1f}} m<br>{metric}: %{{y:.2f}} {unit}<extra></extra>'
            ), 
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=tel2['Distance'], 
                y=tel2[metric], 
                mode='lines', 
                name=f"{lap2_label} L{lap2_num}", 
                line=dict(color=color2), 
                legendgroup="l2",
                hovertemplate=f'Distance: %{{x:.1f}} m<br>{metric}: %{{y:.2f}} {unit}<extra></extra>'
            ), 
            row=1, col=2
        )

        # Highlight distance line
        if highlight_distance is not None:
            fig.add_trace(
                go.Scatter(
                    x=[highlight_distance, highlight_distance],
                    y=[tel1[metric].min(), tel1[metric].max()],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dot'),
                    name=highlight_label if highlight_label else 'Highlighted Distance',
                    showlegend=True
                ), 
                row=1, col=2
            )

        # Track corner annotations on telemetry plot
        if circuit_info is not None and 'Date' in pos1.columns and not pos1.empty:
            for _, corner in circuit_info.corners.iterrows():
                dist_sq = (pos1['X'] - corner['X'])**2 + (pos1['Y'] - corner['Y'])**2
                if not dist_sq.empty:
                    closest_idx = dist_sq.idxmin()
                    closest_time = pos1.loc[closest_idx, 'Date']
                    tel_idx = (tel1['Date'] - closest_time).abs().idxmin()
                    corner_dist = tel1.loc[tel_idx, 'Distance']
                    fig.add_vline(
                        x=corner_dist, 
                        line_width=1, 
                        line_dash="dash", 
                        line_color="grey", 
                        annotation_text=f"C-{corner['Number']}{corner['Letter']}", 
                        annotation_position="top right", 
                        row=1, col=2
                    )

        # Interpolation & Difference calculation
        interp_func = interp1d(tel2['Distance'], tel2[metric], kind='linear', fill_value="extrapolate")
        tel2_interp = interp_func(tel1['Distance'])
        diff = tel1[metric] - tel2_interp
        max_diff = np.max(np.abs(diff)) if np.max(np.abs(diff)) > 0 else 1

        # Use pos1 or tel1 coordinates for spatial track map
        map_coords = pos1[['X', 'Y']].to_numpy() if {'X', 'Y'}.issubset(pos1.columns) else tel1[['X', 'Y']].to_numpy()
        rot_coords = geometry._rotate(map_coords, angle=track_angle)

        # Add Track Map Trace
        fig.add_trace(
            go.Scatter(
                x=rot_coords[:, 0],
                y=rot_coords[:, 1],
                mode='lines+markers',
                name='Track Map',
                showlegend=True,
                marker=dict(
                    size=4, 
                    color=diff, 
                    colorscale='RdBu', 
                    reversescale=True, 
                    cmin=-max_diff, 
                    cmax=max_diff, 
                    cmid=0, 
                    showscale=True, 
                    colorbar=dict(
                        thickness=15, 
                        x=-0.15, 
                        title=dict(text=f"Higher {metric}", side='top'), 
                        tickvals=[-max_diff, 0, max_diff], 
                        ticktext=[f"{lap2_label} L{lap2_num}", "Equal", f"{lap1_label} L{lap1_num}"]
                    )
                ), 
                hovertemplate=f'{metric} Difference: %{{marker.color:.2f}} {unit}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add Corner Annotations to Map
        if circuit_info is not None:
            for _, corner in circuit_info.corners.iterrows():
                track_x, track_y = geometry._rotate([corner['X'], corner['Y']], angle=track_angle)
                fig.add_annotation(
                    x=track_x, 
                    y=track_y, 
                    text=f"{corner['Number']}{corner['Letter']}", 
                    showarrow=False, 
                    bgcolor="grey", 
                    font=dict(color="white", size=10), 
                    row=1, col=1
                )

        fig.update_xaxes(range=[0, max_dist], title_text="Distance (m)", row=1, col=2)
        fig.update_yaxes(title_text=f"{metric} [{unit}]", row=1, col=2)
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_layout(
            title=dict(text=f"{metric} Analysis: {comparison_title_suffix}", x=0.5, xanchor='center'), 
            height=500, 
            margin=dict(l=100, r=50, t=80, b=50), 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        figures.append(fig)

    return figures[0] if len(figures) == 1 else figures



from values.colors import get_driver_colors

def plot_laptime_evolution(
    laps: pd.DataFrame,
    drivers: List[str],
    event_lap: int,
    event_label: str,
) -> go.Figure:
    laps_pace = laps.pick_drivers(drivers).copy()

    laps_pace['LapTimeSeconds'] = laps_pace['LapTime'].dt.total_seconds()

    driver_colors = get_driver_colors(laps_pace, drivers)

    fig = go.Figure()
    for driver in drivers:
        driver_laps = laps_pace[laps_pace['Driver'] == driver]
        
        color = driver_colors.get(driver, '#808080')

        fig.add_trace(go.Scatter(
            x=driver_laps['LapNumber'],
            y=driver_laps['LapTimeSeconds'],
            mode='lines+markers',
            name=driver,
            line=dict(color=color),
            marker=dict(size=6),
            hovertemplate=f"Driver: {driver}<br>Lap: %{{x}}<br>Time: %{{y:.3f}}s<extra></extra>"
        ))

    # Annotation for chosen lap
    fig.add_annotation(
        x=34,
        y=laps_pace[(laps_pace['LapNumber'] == event_lap) & (laps_pace['Driver'] == 'ANT')]['LapTimeSeconds'].iloc[0],
        text=event_label,
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        bgcolor="rgba(0,0,0,0.5)",
        font=dict(color="white")
    )

    fig.update_layout(
        title='Pace Comparison',
        xaxis_title='Lap Number',
        yaxis_title='Lap Time (Seconds)',
        hovermode='x unified',
        yaxis=dict(range=[
            min(laps_pace['LapTimeSeconds'].dropna()) - 0.5,
            max(laps_pace[laps_pace['LapTimeSeconds'] < 120]['LapTimeSeconds'].dropna()) + 0.5
        ])
    )
    return fig



def plot_standings_evolution_of_lap(
    laps: pd.DataFrame,
    results: List,
    color_map: Dict
):
    import fastf1.plotting

    # Get Grid positions from results
    grid_data = results[['Abbreviation', 'GridPosition']].copy()
    grid_data.columns = ['Driver', 'GridPosition']

    # Get position at the end of Lap 1
    lap1_pos = laps.pick_lap(1)[['Driver', 'Position']].copy()
    lap1_pos.columns = ['Driver', 'Lap1Position']

    # Merge and calculate gain
    gain_df = pd.merge(grid_data, lap1_pos, on='Driver')
    gain_df['Lap1Gain'] = gain_df['GridPosition'] - gain_df['Lap1Position']
    gain_df = gain_df.sort_values('Lap1Gain', ascending=False)

    fig_gain = px.bar(
        gain_df,
        x='Driver',
        y='Lap1Gain',
        color='Driver',
        color_discrete_map=color_map,
        title='Positions Gained/Lost on Lap 1 (Grid vs Lap 1 Finish)',
        labels={'Lap1Gain': 'Positions Gained'}
    )
    fig_gain.update_layout(showlegend=False)
    fig_gain.show()