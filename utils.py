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
from definitions import track_status_colors, compound_colors

_logger = get_logger(__package__)



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

    def _rotate(xy, *, angle):
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                            [-np.sin(angle), np.cos(angle)]])
        return np.matmul(xy, rot_mat)

    # rotate track
    track = position.loc[:, ('X', 'Y')].to_numpy()
    track_angle = circuit_info.rotation / 180 * np.pi
    rotated_track = _rotate(track, angle=track_angle)

    # calc gradient
    altitude_meters = position['Z'].values + reference_altitude
    altitude_gradient = np.gradient(altitude_meters)

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

    # gradient
    altitude_meters = position['Z'].values + reference_altitude
    altitude_gradient = np.gradient(altitude_meters)

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

    fig.show()


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
            color = compound_colors.get(compound.upper(), 'gray') 
            
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
        line_color = track_status_colors.get(lap_events.iloc[0]['Message'], 'gray')

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
            event_color = track_status_colors.get(row['Message'], 'gray')

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

    for status, color in track_status_colors.items():
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
            line_color = track_status_colors.get(lap_events.iloc[0]['Message'], 'gray')

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
                event_color = track_status_colors.get(row['Message'], 'gray')

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
    for status, color in track_status_colors.items():
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


def plot_laptime_distribution(
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


def _get_track_status_changes(
        laps: pd.DataFrame,
        track_status: pd.DataFrame
    ) -> pd.DataFrame:

    filtered_track_status_changes = track_status[
        track_status['Message'].isin(track_status_colors.keys())
    ].copy()

    # add lap-column by finding the lap number closest to event time
    filtered_track_status_changes['Lap'] = filtered_track_status_changes['Time'].apply(
        lambda event_time: laps.loc[laps['Time'] <= event_time, 'LapNumber'].max() if not laps.loc[laps['Time'] <= event_time].empty else None
    )
    filtered_track_status_changes.dropna(subset=['Lap'], inplace=True)
    filtered_track_status_changes['Lap'] = filtered_track_status_changes['Lap'].astype(int)

    # group to handle multiple events per lap
    return filtered_track_status_changes.groupby('Lap')
    