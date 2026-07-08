import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as pcolors
import pandas as pd
import numpy as np
import pandas as pd
from datetime import  datetime
from typing import Optional
from fastf1.plotting._plotting import _COLOR_PALETTE
from fastf1.logger import get_logger
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from formula_1_plotly_utils import definitions


_logger = get_logger(__package__)


# Helper function to rotate track coordinates
def _rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)


def setup_plotly(color_scheme: str = None):
    """
    Configures Plotly with the FastF1 dark theme.
    """
    if color_scheme == "FastF1":
        _enable_fastf1_color_scheme()


def _enable_fastf1_color_scheme():
    # Defining the colors to match exactly
    bg_color = '#292625'      # figure.facecolor
    plot_bg_color = '#1e1c1b' # axes.facecolor
    grid_color = '#2d2928'    # axes.edgecolor
    text_color = '#F1F1F3'    # text.color / axes.labelcolor

    fastf1_template = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=plot_bg_color,
            font=dict(
                family="sans-serif", # Gravity is usually not standard on all systems
                color=text_color,
                size=14
            ),
            # Title styling (matches axes.titlesize 19 and titlepad 12)
            title=dict(
                font=dict(size=19, color=text_color),
                pad=dict(t=12),
                x=0.5, # Centered title is often cleaner in Plotly
                xanchor='center'
            ),
            xaxis=dict(
                gridcolor=grid_color,
                linecolor=grid_color,
                zerolinecolor=grid_color,
                showline=True,
                tickfont=dict(color='#f1f2f3'), # Matches xtick.color
                titlefont=dict(color='#f1f2f3'),
                mirror=True # Matches the "box" look of MPL
            ),
            yaxis=dict(
                gridcolor=grid_color,
                linecolor=grid_color,
                zerolinecolor=grid_color,
                showline=True,
                tickfont=dict(color='#f1f2f3'), # Matches ytick.color
                titlefont=dict(color='#f1f2f3'),
                mirror=True
            ),
            # Legend styling (matches (0.1, 0.1, 0.1, 0.7))
            legend=dict(
                bgcolor='rgba(25, 25, 25, 0.7)',
                bordercolor='rgba(25, 25, 25, 0.9)',
                borderwidth=1,
                font=dict(color=text_color)
            ),
            # The line colors
            colorway=_COLOR_PALETTE,
            # Plotly specific: Ensure the hover label matches the theme
            hoverlabel=dict(
                bgcolor=plot_bg_color,
                font=dict(color=text_color, size=13)
            )
        )
    )
    
    pio.templates["fastf1"] = fastf1_template
    pio.templates.default = "fastf1"


################################################
# utils
################################################

def plot_track(
        position: pd.DataFrame,
        circuit_info: Optional['fastf1.mvapi.CircuitInfo'] = None,
        reference_altitude: int = 0
    ) -> 'plotly.graph_objects.Figure': 
    """Plot the track layout with elevation markers and corner annotations 
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

    # rotate track
    track = position.loc[:, ('X', 'Y')].to_numpy()
    track_angle = circuit_info.rotation / 180 * np.pi
    rotated_track = _rotate(track, angle=track_angle)

    # calc gradient
    altitude_meters = position['Z'].values + reference_altitude
    altitude_diff = position['Z'].diff().fillna(0)

    delta_x = position['X'].diff().fillna(0)
    delta_y = position['Y'].diff().fillna(0)
    distances = np.sqrt(delta_x**2 + delta_y**2)
    altitude_gradient = np.where(distances > 0, (altitude_diff / distances) * 100, 0)

    # scatter plot with color scale based on the altitude gradient
    fig = go.Figure(data=go.Scatter(
        x=rotated_track[:, 0],
        y=rotated_track[:, 1],
        mode='lines+markers',
        marker=dict(
            size=5,
            color=altitude_gradient,
            colorscale='Plasma',
            colorbar=dict(title='Altitude Gradient'),
            opacity=0.8
        ),
        line=dict(
                color=_COLOR_PALETTE[0],
                width=4
            ),
        hoverinfo='text',
        text=[f'Altitude Gradient: {grad:.2f}%' for grad in altitude_gradient]
    ))

    if circuit_info:
        # add corner information as annotations
        for _, corner in circuit_info.corners.iterrows():
            # Rotate the center of the corner equivalently to the rest of the track map
            txt = f"{corner['Number']}{corner['Letter']}"
            track_x, track_y = _rotate([corner['X'], corner['Y']], angle=track_angle)
            fig.add_annotation(
                x=track_x,
                y=track_y,
                text=txt,
                showarrow=False, # Do not show arrow
                bgcolor="grey",
                font=dict(
                    color="white",
                    size=10
                )
            )

    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1), 
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis_showgrid=False, yaxis_zeroline=False, yaxis_showticklabels=False,
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


def plot_tyre_strategies(
        drivers: list,
        laps: pd.DataFrame,
        track_status: pd.DataFrame,
    ) -> 'plotly.graph_objects.Figure':
    """Visualise tyre strategy and track status for multiple drivers.

    Generates a stacked horizontal bar chart that shows the number of laps
    each driver spent on each tyre compound.  Vertical dashed lines
    indicate track‑status changes (e.g. safety car, yellow flag).  For
    each status change a coloured marker is plotted on the y‑axis next
    to the driver bar.

    Parameters
    ----------
    drivers : list
        List of driver names to include in the plot.  The order determines
        the order on the y‑axis.
    laps : pd.DataFrame
        DataFrame containing at least ``Driver``, ``Stint``, ``Compound`` and
        ``LapNumber`` columns.
    track_status : pd.DataFrame
        DataFrame containing at least ``Message`` and ``Time`` columns.
        ``Message`` should be one of the keys in the ``track_status_colors``
        mapping.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure showing stacked tyre‑compound bars
        for each driver and vertical markers for track‑status events.
    """
        
    stints = laps[['Driver', 'Stint', 'Compound', 'LapNumber']]
    stints = stints.groupby(['Driver', 'Stint', 'Compound']).count().reset_index()
    stints = stints.rename(columns={'LapNumber': 'LapCount'})

    track_status_changes = track_status.copy()

    fig = go.Figure()

    added_compounds = set()

    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver].sort_values(by='Stint') # sort by stint to ensure correct stacking

        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            compound = row["Compound"]
            color = definitions.compound_colors.get(compound.upper(), 'gray') 
            
            # determine whether to show the legend entry for this compound
            show_legend_entry = False
            if compound not in added_compounds:
                added_compounds.add(compound)
                show_legend_entry = True

            fig.add_trace(go.Bar(
                y=[driver],
                x=[row["LapCount"]],
                name=compound,
                orientation='h',
                marker=dict(
                    color=color,
                    line=dict(color='white', width=2)
                ),
                base=previous_stint_end,
                customdata=[compound], # compound for hovertext in next line
                hovertemplate='Driver: %{y}<br>Compound: %{customdata}<br>Laps: %{x}<extra></extra>',
                showlegend=show_legend_entry
            ))

            previous_stint_end += row["LapCount"]

    fig.update_layout(
        title='Tyre Strategy per Driver',
        xaxis_title='Lap Number',
        yaxis_title='Driver',
        barmode='stack',
        legend_title='Compound',
        yaxis=dict(autorange="reversed"), # invert y-axis
        height=800 
    )

    grouped_track_status = _get_track_status_changes(laps, track_status)

    # vertical lines for track status changes
    for lap, lap_events in grouped_track_status:
        line_color = definitions.track_status_colors.get(lap_events.iloc[0]['Message'], 'gray')

        fig.add_vline(
            x=lap,
            line_width=2,
            line_dash="dash",
            line_color=line_color, 
            layer="above",
        )

        # scatter markers for each event
        num_events = len(lap_events)
        # vertical offset for each marker in the same lap
        vertical_offsets = np.linspace(-0.2, 0.2, num_events) 
        
        # index of the first driver as a reference point for the vertical position of markers
        if drivers.size > 0:
            driver_y_index = fig.layout.yaxis.categoryarray.index(drivers[0]) if fig.layout.yaxis.categoryarray is not None else 0
        else:
            driver_y_index = 0 # Default to 0 if no drivers are found

        for i, (index, row) in enumerate(lap_events.iterrows()):
            event_color = definitions.track_status_colors.get(row['Message'], 'gray')

            fig.add_trace(go.Scatter(
                x=[row['Lap']],
                y=[driver_y_index + vertical_offsets[i]], 
                mode='markers',
                marker=dict(
                    size=10,
                    color=event_color,
                    symbol='circle', 
                    line=dict(color='black', width=1)
                ),
                hoverinfo='text',
                text=f"Track Status: {row['Message']}, Lap {row['Lap']}",
                showlegend=False,
            ))

    for status, color in definitions.track_status_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], 
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color, symbol='circle'),
            legendgroup='Track Status',
            showlegend=True,
            name=status
        ))
    return fig


def plot_pitstop_durations(
        choosen_drivers: list, 
        laps: pd.DataFrame, 
        track_status: pd.DataFrame
    ):

    pitstop_times = {}
    individual_pitstop_durations = {}

    for driver in choosen_drivers:
        driver_laps = laps.pick_driver(driver).reset_index(drop=True)

        # laps where the driver entered pits
        pit_in_laps = driver_laps.loc[driver_laps['PitInTime'].notnull()]

        total_pitstop_duration = pd.Timedelta(seconds=0)
        driver_pitstop_list = []

        for index, pit_in_lap in pit_in_laps.iterrows():
            # lap after the pit-in lap where PitOutTime not null
            next_lap_index = pit_in_lap.name + 1
            if next_lap_index < len(driver_laps):
                pit_out_lap = driver_laps.loc[next_lap_index]
                if pd.notnull(pit_out_lap['PitOutTime']):
                    # calc timediff
                    if isinstance(pit_in_lap['PitInTime'], pd.Timedelta) and isinstance(pit_out_lap['PitOutTime'], pd.Timedelta):
                        pitstop_duration = pit_out_lap['PitOutTime'] - pit_in_lap['PitInTime']
                    else:
                        try:
                            pitstop_duration = pd.to_timedelta(pit_out_lap['PitOutTime']) - pd.to_timedelta(pit_in_lap['PitInTime'])
                        except ValueError:
                            pitstop_duration = pd.Timedelta(seconds=0) # Handle cases where conversion fails


                    total_pitstop_duration += pitstop_duration
                    driver_pitstop_list.append({'LapNumber': pit_in_lap['LapNumber'], 'Duration': pitstop_duration})
                else:
                    print(f"Warning: Could not find PitOutTime for pit stop starting on Lap {pit_in_lap['LapNumber']} for driver {driver}")


        pitstop_times[driver] = total_pitstop_duration
        individual_pitstop_durations[driver] = driver_pitstop_list
    
    # conv pitstop durations dict to df
    individual_pitstops_list = []
    for driver, stops in individual_pitstop_durations.items():
        for stop in stops:
            individual_pitstops_list.append({'Driver': driver, 'LapNumber': stop['LapNumber'], 'PitStopDurationSeconds': stop['Duration'].total_seconds()})

    individual_pitstops_df = pd.DataFrame(individual_pitstops_list)

    # plot
    fig = px.bar(individual_pitstops_df,
                x='LapNumber',
                y='PitStopDurationSeconds',
                color='Driver',
                title='Individual Pit Stop Durations per Driver',
                labels={'LapNumber': 'Lap Number', 'PitStopDurationSeconds': 'Pit Stop Duration (seconds)'},
                barmode='group' # group shows bars side by side for each lap
                )

    fig.update_layout(xaxis_title='Lap Number', yaxis_title='Pit Stop Duration (seconds)')

    grouped_track_status = _get_track_status_changes(laps, track_status)

    # vertical lines for track status changes
    for lap, lap_events in grouped_track_status:
        if lap > 23 and lap < 35:
            line_color = definitions.track_status_colors.get(lap_events.iloc[0]['Message'], 'gray')

            fig.add_vline(
                x=lap,
                line_width=2,
                line_dash="dash",
                line_color=line_color,
                layer="above", 
            )

            num_events = len(lap_events)
            vertical_offsets = np.linspace(0, fig.layout.yaxis.range[1] if fig.layout.yaxis.range else 50, num_events) # Adjust the range and number of points as needed

            for i, (index, row) in enumerate(lap_events.iterrows()):
                event_color = definitions.track_status_colors.get(row['Message'], 'gray')

                fig.add_trace(go.Scatter(
                    x=[row['Lap']],
                    y=[vertical_offsets[i]], 
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=event_color,
                        symbol='circle', 
                        line=dict(color='black', width=1)
                    ),
                    hoverinfo='text',
                    text=f"Track Status: {row['Message']}, Lap {row['Lap']}",
                    showlegend=False 
                ))

    # legend for the track status colors by adding invisible traces
    for status, color in definitions.track_status_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], # No data
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color, symbol='circle'),
            legendgroup='Track Status',
            showlegend=True,
            name=status
        ))
    return fig


def plot_laptime_distribution_weatherdependent(
        laps: pd.DataFrame,
        session_start_time: datetime,
        drivers: list,
        weather_data: pd.DataFrame = None,
    ) -> pd.DataFrame:         

    drivers_laps = laps[laps['Driver'].isin(drivers)].copy()
    drivers_laps['DateTime'] = session_start_time + drivers_laps['Time']
    drivers_laps['LapTimeSeconds'] = drivers_laps['LapTime'].dt.total_seconds()
    drivers_laps['LapTimeZScore'] = (drivers_laps['LapTimeSeconds'] - drivers_laps['LapTimeSeconds'].mean()) / drivers_laps['LapTimeSeconds'].std()
    drivers_laps_filtered = drivers_laps[abs(drivers_laps['LapTimeZScore']) <= 3].copy()
    
    if weather_data is not None:
        weather_data_datetime = weather_data['Time']

        # copy of weather_data with datetime index for merging
        weather_data_for_merge = weather_data.copy()
        weather_data_for_merge['DateTime'] = weather_data_datetime

        drivers_laps_filtered = drivers_laps.sort_values(by='DateTime')
        weather_data_for_merge_sorted = weather_data_for_merge.sort_values(by='DateTime')

        merged_laps_weather = pd.merge_asof(
            drivers_laps_filtered,
            weather_data_for_merge_sorted[['DateTime', 'Rainfall', 'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed']],
            on='DateTime',
            direction='backward' # find the closest timestamp before or at the lap time
        )

        merged_laps_weather.dropna(subset=['Rainfall'], inplace=True)

        rainy_laps_df = merged_laps_weather[merged_laps_weather['Rainfall'] == True].copy()
        dry_laps_df = merged_laps_weather[merged_laps_weather['Rainfall'] == False].copy()

        average_rainy_lap_times = rainy_laps_df.groupby('Driver')['LapTime'].mean().dt.total_seconds()
        average_dry_lap_times = dry_laps_df.groupby('Driver')['LapTime'].mean().dt.total_seconds()

        # combine for plot
        combined_laps_df = pd.concat([rainy_laps_df.assign(Condition='Raining'),
                                    dry_laps_df.assign(Condition='Not Raining')])

        fig = px.violin(combined_laps_df,
                            y='LapTimeSeconds',
                            x='Driver',
                            color='Condition',
                            box=True, #  box plot inside violin
                            points='all', 
                            title='Lap Time Distribution by Driver and Condition',
                            labels={'Driver': 'Driver', 'LapTimeSeconds': 'Lap Time (seconds)', 'Condition': 'Condition'},
                            color_discrete_map={'Raining': 'blue', 'Not Raining': 'orange'}
                            )

        fig.update_layout(xaxis_title='Driver', yaxis_title='Lap Time (seconds)')
    else:
        raise Exception
    return fig


def plot_laptime_distribution_per_compound(laps: pd.DataFrame, drivers: list, results: pd.DataFrame):
    filtered_laps = laps[laps['Driver'].isin(drivers)].copy()
    filtered_laps['LapTimeSeconds'] = filtered_laps['LapTime'].dt.total_seconds()

    # box plot compounds
    fig = px.box(filtered_laps,
        x='Driver',
        y='LapTimeSeconds',
        color='Compound',
        points='all',
        hover_data=['LapNumber'],
        title='Lap Time Performance per Driver and Tyre Compound')

    driver_positions = results.sort_values(by='Position')['Abbreviation'].tolist()
    fig.update_layout(
        xaxis_title='Driver',
        yaxis_title='Lap Time (seconds)',
        legend_title='Tyre Compound',
        xaxis=dict(categoryorder='array', categoryarray=driver_positions)
    )
    return fig


def plot_laptime_distribution_per_qualifyinground(laps: pd.DataFrame, drivers: list, results: pd.DataFrame):
    filtered_laps = laps[laps['Driver'].isin(drivers)].copy()
    
    # identify the border of qualifying rounds
    if 'q2_end_lap' not in locals() or 'q3_end_lap' not in locals():
        if not results.empty:
            q2_end_lap = laps[laps['DriverNumber'].isin(results[results['Position'] == 16]['DriverNumber'].values)]['LapNumber'].max()
            q3_end_lap = laps[laps['DriverNumber'].isin(results[results['Position'] == 11]['DriverNumber'].values)]['LapNumber'].max()
        else:
            q2_end_lap = None
            q3_end_lap = None

    filtered_laps['QualifyingRound'] = 'SQ1'
    if q2_end_lap is not None:
        filtered_laps.loc[filtered_laps['LapNumber'] > q2_end_lap, 'QualifyingRound'] = 'SQ2'
    if q3_end_lap is not None:
        filtered_laps.loc[filtered_laps['LapNumber'] > q3_end_lap, 'QualifyingRound'] = 'SQ3'

    # laptime to seconds for plotting
    filtered_laps['LapTimeSeconds'] = filtered_laps['LapTime'].dt.total_seconds()

    # final driver positions sorted
    if not results.empty:
        driver_positions = results.sort_values(by='Position')['Abbreviation'].tolist()
        filtered_laps['Driver_Category'] = pd.Categorical(filtered_laps['Driver'], categories=driver_positions, ordered=True)
        filtered_laps.sort_values(by='Driver_Category', inplace=True)

    fig = px.box(filtered_laps,
                    x='Driver',
                    y='LapTimeSeconds',
                    color='QualifyingRound',
                    points='all',
                    hover_data=['LapNumber', 'Compound'],
                    title='Lap Time Performance per Driver and Qualifying Round')

    fig.update_layout(
        xaxis_title='Driver',
        yaxis_title='Lap Time (seconds)',
        legend_title='Qualifying Round',
        xaxis=dict(categoryorder='array', categoryarray=driver_positions)
    )
    return fig


def plot_best_laptime(results: pd.DataFrame, drivers: list, criteria: str=None):
    filtered_results = results[results['Abbreviation'].isin(drivers)].copy()

    if criteria == "qualifying":
        # Q1, Q2, Q3 columns to seconds
        best_lap_times_official = filtered_results[['Abbreviation', 'Q1', 'Q2', 'Q3']].copy()
        for col in ['Q1', 'Q2', 'Q3']:
            best_lap_times_official[col] = best_lap_times_official[col].apply(lambda x: x.total_seconds() if pd.notna(x) else np.nan)

        value_vars=['Q1', 'Q2', 'Q3']
        var_name='QualifyingRound'

    elif criteria == "compound":
        #!todo
        value_vars=None
        var_name='Compound'
        
    elif criteria == "weather":
        #!todo rain/dry
        value_vars=None
        var_name='Rainy/Dry'

    best_lap_times_official = best_lap_times_official.melt(
        id_vars='Abbreviation',
        value_vars=value_vars,
        var_name=var_name,
        value_name='BestLapTime'
    ).dropna(subset=['BestLapTime'])

    # best overall lap time per driver from the official results for sorting
    best_overall_lap_time_driver_official = best_lap_times_official.groupby('Abbreviation')['BestLapTime'].min().reset_index()
    best_overall_lap_time_driver_official = best_overall_lap_time_driver_official.rename(columns={'BestLapTime': 'BestOverallLapTime'})

    # merge best lap times with overall best lap time for sorting
    best_lap_times_official = pd.merge(best_lap_times_official, best_overall_lap_time_driver_official, on='Abbreviation', how='left')

    if not best_overall_lap_time_driver_official.empty:
        driver_order_official = best_overall_lap_time_driver_official.sort_values(by='BestOverallLapTime')['Abbreviation'].tolist()
        best_lap_times_official['Driver_Category'] = pd.Categorical(best_lap_times_official['Abbreviation'], categories=driver_order_official, ordered=True)
        best_lap_times_official.sort_values(by='Driver_Category', inplace=True)

    # plot
    fig = px.scatter(best_lap_times_official,
                            x='Abbreviation',
                            y='BestLapTime',
                            color=var_name, 
                            symbol=var_name,
                            hover_data=[var_name, 'BestLapTime'],
                            title=f'Best Lap Time per Driver by {var_name}')

    fig.update_layout(
        xaxis_title='Driver',
        yaxis_title='Best Lap Time (seconds)',
        legend_title=var_name,
        xaxis=dict(categoryorder='array', categoryarray=driver_order_official) # order of drivers on x-axis
    )
    return fig


def plot_driver_position_per_lap():
    pass

def _get_track_status_changes(
        laps: pd.DataFrame,
        track_status: pd.DataFrame
    ) -> pd.DataFrame:

    filtered_track_status_changes = track_status[
        track_status['Message'].isin(definitions.track_status_colors.keys())
    ].copy()

    # add lap-column by finding the lap number closest to event time
    filtered_track_status_changes['Lap'] = filtered_track_status_changes['Time'].apply(
        lambda event_time: laps.loc[laps['Time'] <= event_time, 'LapNumber'].max() if not laps.loc[laps['Time'] <= event_time].empty else None
    )
    filtered_track_status_changes.dropna(subset=['Lap'], inplace=True)
    filtered_track_status_changes['Lap'] = filtered_track_status_changes['Lap'].astype(int)

    # group to handle multiple events per lap
    return filtered_track_status_changes.groupby('Lap')
    


def plot_leading_laptimes(drivers: list, laps: pd.DataFrame, track_status: pd.DataFrame):
    # Ensure 'LapTime' is in timedelta format
    if 'LapTime' not in laps.columns or not pd.api.types.is_timedelta64_dtype(laps['LapTime']):
        laps['LapTime'] = pd.to_timedelta(laps['LapTime'])
    cleaned_laps = laps.dropna(subset=['LapNumber', 'LapTime']).copy()

    # Convert LapTime to total seconds for numerical comparison
    cleaned_laps['LapTimeSeconds'] = cleaned_laps['LapTime'].dt.total_seconds()

    plot_data = []
    current_fastest_overall_time = float('inf')
    current_fastest_overall_driver = None

    # Filter laps up to and including the current lap number
    unique_lap_numbers = sorted(cleaned_laps['LapNumber'].unique())
    for lap_num in unique_lap_numbers:
        laps_up_to_current = cleaned_laps[cleaned_laps['LapNumber'] <= lap_num]

        if not laps_up_to_current.empty:
            # Find the index of the absolute fastest lap among all laps recorded so far
            fastest_idx_so_far = laps_up_to_current['LapTimeSeconds'].idxmin()
            fastest_row_so_far = laps_up_to_current.loc[fastest_idx_so_far]

            # Check if this new fastest lap is an improvement over the current overall fastest
            if fastest_row_so_far['LapTimeSeconds'] < current_fastest_overall_time:
                current_fastest_overall_time = fastest_row_so_far['LapTimeSeconds']
                current_fastest_overall_driver = fastest_row_so_far['Driver']

        # Append the current overall fastest lap time and its associated driver for this lap_num
        # Only add if current_fastest_overall_driver is not None (i.e., we found at least one lap)
        if current_fastest_overall_driver is not None:
            plot_data.append({
                'LapNumber': lap_num,
                'LapTimeSeconds': current_fastest_overall_time,
                'Driver': current_fastest_overall_driver
            })

    # Create a DataFrame from the collected plot data
    leading_laps_plot_df = pd.DataFrame(plot_data)

    if leading_laps_plot_df.empty:
        print("No valid lap data to plot after processing.")
    else:
        fig = go.Figure()

        # Get unique drivers who held the leading lap time
        unique_leading_drivers = leading_laps_plot_df['Driver'].unique()

        for driver in unique_leading_drivers:
            driver_laps_leading = leading_laps_plot_df[leading_laps_plot_df['Driver'] == driver]

            # Add a trace for each driver, allowing for discontinuous segments
            # by using connectgaps=False. This ensures one legend entry per driver
            # and consistent coloring for all their leading segments.
            fig.add_trace(go.Scatter(
                x=driver_laps_leading['LapNumber'],
                y=driver_laps_leading['LapTimeSeconds'],
                mode='lines+markers',
                name=driver, # This ensures a single legend entry per driver
                hoverinfo='text',
                text=[
                    f"Lap: {int(row['LapNumber'])}<br>Driver: {row['Driver']}<br>Lap Time: {row['LapTimeSeconds']:.3f}s"
                    for idx, row in driver_laps_leading.iterrows()
                ],
                connectgaps=False, # Crucial for showing discontinuous leading periods
                showlegend=True
            ))

    grouped_track_status = _get_track_status_changes(laps, track_status)

    # vertical lines for track status changes
    for lap, lap_events in grouped_track_status:
        line_color = definitions.track_status_colors.get(lap_events.iloc[0]['Message'], 'gray')

        fig.add_vline(
            x=lap,
            line_width=2,
            line_dash="dash",
            line_color=line_color, 
            layer="above",
        )

        # scatter markers for each event
        num_events = len(lap_events)
        # vertical offset for each marker in the same lap
        vertical_offsets = np.linspace(-0.2, 0.2, num_events) 
        
        # index of the first driver as a reference point for the vertical position of markers
        if drivers.size > 0:
            driver_y_index = fig.layout.yaxis.categoryarray.index(drivers[0]) if fig.layout.yaxis.categoryarray is not None else 0
        else:
            driver_y_index = 0 # Default to 0 if no drivers are found

        for i, (index, row) in enumerate(lap_events.iterrows()):
            event_color = definitions.track_status_colors.get(row['Message'], 'gray')

            fig.add_trace(go.Scatter(
                x=[row['Lap']],
                y=[driver_y_index + vertical_offsets[i]], 
                mode='markers',
                marker=dict(
                    size=10,
                    color=event_color,
                    symbol='circle', 
                    line=dict(color='black', width=1)
                ),
                hoverinfo='text',
                text=f"Track Status: {row['Message']}, Lap {row['Lap']}",
                showlegend=False,
            ))
        


def plot_lap_telemetry_comparison(
    laps: pd.DataFrame,
    circuit_info: Optional['fastf1.mvapi.CircuitInfo'],
    driver1_code: str,
    driver2_code: str,
    driver1_lap: str,
    driver2_lap: str,
    metrics_to_plot: List = None,
    highlight_distance=None,
    highlight_label: str = None,
    custom_title: str = None):

    # filter choosen lap
    if driver1_lap == 'fastest':
        driver1_laps_filtered = laps.pick_driver(driver1_code)
        lap1 = driver1_laps_filtered.loc[driver1_laps_filtered['LapTime'].idxmin()] if not driver1_laps_filtered.empty else pd.Series()
    else:
        lap1 = laps.pick_driver(driver1_code).pick_lap(driver1_lap)

    if driver2_lap == 'fastest':
        driver2_laps_filtered = laps.pick_driver(driver2_code)
        lap2 = driver2_laps_filtered.loc[driver2_laps_filtered['LapTime'].idxmin()] if not driver2_laps_filtered.empty else pd.Series()
    else:
        lap2 = laps.pick_driver(driver2_code).pick_lap(driver2_lap)

    if lap1.empty or lap2.empty:
        print(f"One or both laps are missing or empty for Driver 1 ({driver1_code}, Lap {driver1_lap}) or Driver 2 ({driver2_code}, Lap {driver2_lap}).")
        return

    lap1_label = driver1_code
    lap2_label = driver2_code

    # telemetry data
    try:
        tel1 = lap1.get_telemetry()
        tel2 = lap2.get_telemetry()
        if len(tel1) == 0 or len(tel2) == 0:
            raise ValueError("Telemetry is empty for one of the laps.")
    except Exception as e:
        print(f"Could not retrieve telemetry for {lap1_label} L{int(lap1['LapNumber'])} or {lap2_label} L{int(lap2['LapNumber'])}: {e}")
        return

    # rotate annoations to match track-map
    track_angle = circuit_info.rotation / 180 * np.pi if circuit_info is not None else 0

    lap1_num = int(lap1['LapNumber']) if 'LapNumber' in lap1 else "N/A"
    lap2_num = int(lap2['LapNumber']) if 'LapNumber' in lap2 else "N/A"

    if custom_title:
        comparison_title_suffix = custom_title
    else:
        comparison_title_suffix = f"{lap1_label} (Lap {lap1_num}) vs {lap2_label} (Lap {lap2_num})"

    pos1 = lap1.get_pos_data()

    available_columns = set(tel1.columns).intersection(set(tel2.columns))
    if metrics_to_plot is None:
        metrics = [m for m in definitions.telemetry_metrics if m in available_columns]
    else:
        metrics = [m for m in metrics_to_plot if m in available_columns]

    color1, color2 = 'red', 'lightblue'
    max_dist = max(tel1['Distance'].max(), tel2['Distance'].max())

    for metric in metrics:
        unit = definitions.telemetry_metrics.get(metric, '')
        fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], horizontal_spacing=0.05)

        fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1[metric], mode='lines', name=f"{lap1_label} L{lap1_num}", line=dict(color=color1), legendgroup="l1"), row=1, col=2)
        fig.add_trace(go.Scatter(x=tel2['Distance'], y=tel2[metric], mode='lines', name=f"{lap2_label} L{lap2_num}", line=dict(color=color2), legendgroup="l2"), row=1, col=2)

        # highlight
        if highlight_distance is not None:
            fig.add_trace(go.Scatter(
                x=[highlight_distance, highlight_distance],
                y=[tel1[metric].min(), tel1[metric].max()],
                mode='lines',
                line=dict(color='green', width=2, dash='dot'),
                name=highlight_label if highlight_label else 'Highlighted Distance',
                showlegend=True
            ), row=1, col=2)

        # match track annotations to telemetry-data based on time an position
        if circuit_info is not None:
            for _, corner in circuit_info.corners.iterrows():
                # ensure Datecolumn exists in pos1, or use Time for telemetry alignment
                if 'Date' in pos1.columns and not pos1.empty: # Check if pos1 is not empty
                    dist_sq = (pos1['X'] - corner['X'])**2 + (pos1['Y'] - corner['Y'])**2
                    if not dist_sq.empty:
                        closest_idx = dist_sq.idxmin()
                        closest_time = pos1.loc[closest_idx, 'Date']
                        # find the closest telemetry point by Date
                        tel_idx = (tel1['Date'] - closest_time).abs().idxmin()
                        corner_dist = tel1.loc[tel_idx, 'Distance']
                        fig.add_vline(x=corner_dist, line_width=1, line_dash="dash", line_color="grey", annotation_text=f"C-{corner['Number']}{corner['Letter']}", annotation_position="top right", row=1, col=2)
                else:
                    print("Position data 'Date' column not available or position data is empty for corner annotation.")

        interp_func = interp1d(tel2['Distance'], tel2[metric], kind='linear', fill_value="extrapolate")
        tel2_interp = interp_func(tel1['Distance'])
        diff = tel1[metric] - tel2_interp
        max_diff = np.max(np.abs(diff)) if np.max(np.abs(diff)) > 0 else 1
        rot_coords = _rotate(tel1[['X', 'Y']].to_numpy(), angle=track_angle)

        # map
        fig.add_trace(go.Scatter(
            x=rot_coords[:, 0],
            y=rot_coords[:, 1],
            mode='lines+markers',
            name='Track Map',
            showlegend=True,
            marker=dict(size=4, color=diff, colorscale='RdBu', reversescale=True, cmin=-max_diff, cmax=max_diff, cmid=0, showscale=True, colorbar=dict(thickness=15, x=-0.15, title=dict(text=f"Higher {metric}", side='top'), tickvals=[-max_diff, 0, max_diff], ticktext=[f"{lap2_label} L{lap2_num}", "Equal", f"{lap1_label} L{lap1_num}"]))
        ), row=1, col=1)

        if circuit_info is not None:
            for _, corner in circuit_info.corners.iterrows():
                track_x, track_y = _rotate([corner['X'], corner['Y']], angle=track_angle)
                fig.add_annotation(x=track_x, y=track_y, text=f"{corner['Number']}{corner['Letter']}", showarrow=False, bgcolor="grey", font=dict(color="white", size=10), row=1, col=1)

        fig.update_xaxes(range=[0, max_dist], title_text="Distance (m)", row=1, col=2)
        fig.update_yaxes(title_text=f"{metric} [{unit}]", row=1, col=2)
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_layout(title=dict(text=f"{metric} Analysis: {comparison_title_suffix}", x=0.5, xanchor='center'), height=500, template="plotly_white", margin=dict(l=100, r=50, t=80, b=50), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
        
        return fig


