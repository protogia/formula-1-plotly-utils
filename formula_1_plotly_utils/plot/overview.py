from __future__ import annotations
from typing import List, Dict, Optional

from datetime import datetime

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

import fastf1.plotting

def plot_laptime_distribution_weatherdependent(
        laps: pd.DataFrame,
        session_start_time: datetime,
        drivers: List,
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


def plot_laptime_distribution_per_compound(laps: pd.DataFrame, drivers: List, results: pd.DataFrame):
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


def plot_laptime_distribution_per_qualifyinground(laps: pd.DataFrame, drivers: List, results: pd.DataFrame):
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


def plot_driver_position_per_lap(
        laps: pd.DataFrame,
        drivers: pd.DataFrame,
        track_status
    ):

    fig = go.Figure()

    for driver in drivers:
        drv_laps = laps.pick_drivers(driver)

        if not drv_laps.empty:
            abb = drv_laps['Driver'].iloc[0]
            fig.add_trace(go.Scatter(
                x=drv_laps['LapNumber'],
                y=drv_laps['Position'],
                mode='lines+markers',
                name=abb,
                hoverinfo='text',
                text=[f'Driver: {abb}<br>Lap: {lap}<br>Position: {pos}' for lap, pos in zip(drv_laps['LapNumber'], drv_laps['Position'])]
            ))

    fig.update_layout(
        title='Driver Positions Per Lap (Grand Prix Race)',
        xaxis_title='Lap Number',
        yaxis_title='Position',
        yaxis=dict(
            autorange='reversed', # P1 at the top
        ),
        legend_title='Driver'
    )
    return fig


def plot_gap_to_leader_evolution(
    laps: pd.DataFrame,
    track_status: pd.DataFrame,
):
    # gap to leader
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

    # cumulative time per driver
    laps['TotalTime'] = laps.groupby('Driver')['LapTimeSeconds'].cumsum()

    # leader's cumulative time for each lap
    leader_times = laps.groupby('LapNumber')['TotalTime'].min().reset_index()
    leader_times.columns = ['LapNumber', 'LeaderTotalTime']

    # merge and calculate gap
    trace_df = pd.merge(laps, leader_times, on='LapNumber')
    trace_df['GapToLeader'] = trace_df['TotalTime'] - trace_df['LeaderTotalTime']

    # plot
    fig = go.Figure()

    # lines for each driver
    for driver in trace_df['Driver'].unique():
        driver_data = trace_df[trace_df['Driver'] == driver]
        try:
            color = fastf1.plotting.get_driver_color(driver, session=R)
        except:
            color = 'gray'

        fig.add_trace(go.Scatter(
            x=driver_data['LapNumber'],
            y=driver_data['GapToLeader'],
            mode='lines',
            name=driver,
            line=dict(color=color, width=2),
            hovertemplate=f"Driver: {driver}<br>Lap: %{{x}}<br>Gap: %{{y:.3f}}s<extra></extra>"
        ))

    # 3. Highlight Track Status (SC/VSC)
    # Status codes: '3' is Safety Car, '6' is VSC
    status_mapping = {'3': ('Safety Car', 'rgba(0, 0, 255, 0.1)'), '6': ('VSC', 'rgba(255, 165, 0, 0.15)')}

    # Identify periods
    current_status = '1'
    start_lap = None
    for idx, row in track_status.iterrows():
        status = row['Status']
        # Simplified: map time to lap number
        event_lap = laps[laps['LapStartTime'] <= row['Time']]['LapNumber'].max()

        if status != current_status:
            if status in status_mapping:
                start_lap = event_lap
            elif current_status in status_mapping and start_lap is not None:
                # Close the region
                name, color = status_mapping[current_status]
                fig.add_vrect(
                    x0=start_lap, x1=event_lap,
                    fillcolor=color, opacity=0.5, layer="below", line_width=0,
                    annotation_text=name, annotation_position="top left"
                )
            current_status = status

    fig.update_layout(
        title='Race Trace: Gap to Leader (Highlighting VSC Overcut)',
        xaxis_title='Lap Number',
        yaxis_title='Gap to Leader (Seconds)',
        yaxis=dict(autorange='reversed'), # Leader is at 0, others are positive gaps
        hovermode='x unified',
    )
    return fig


def plot_qualifying_results(
    results: pd.DataFrame,
    qualifying_session: str = 'Q3',
    show_gaps: bool = True,
    custom_title: str = None,
    highlight_driver: str = None,
    sort_by: str = 'time') -> go.Figure:
    """
    Plot qualifying results from a FastF1 session.
    """
    
    # resolve column naming (fastf1 uses Q1,Q2,Q3. fallback to Q3Time
    time_col = qualifying_session if qualifying_session in results.columns else f'{qualifying_session}Time'
    
    if time_col not in results.columns:
        print(f"Qualifying session '{qualifying_session}' (searched column '{time_col}') not found in results.")
        return None
    
    # filter drivers with a time in this session
    results = results[results[time_col].notna()].copy()
    
    if results.empty:
        print(f"No qualifying times available for {qualifying_session}.")
        return None
    
    # timing & gaps
    results['TimeSeconds'] = results[time_col].dt.total_seconds()
    results['TimeMs'] = results['TimeSeconds'] * 1000
    
    pole_time_s = results['TimeSeconds'].min()
    results['GapToPole'] = results['TimeSeconds'] - pole_time_s
    
    # sort
    pos_col = 'Position' if 'Position' in results.columns else 'GridPosition'
    if sort_by == 'time':
        results = results.sort_values('TimeSeconds', ascending=False)
    else:
        results = results.sort_values(pos_col, ascending=False)
    
    # driver colors
    if 'TeamColor' in results.columns:
        results['DriverColor'] = results['TeamColor'].apply(
            lambda c: f"#{c}" if isinstance(c, str) and not c.startswith('#') else ('#cccccc' if pd.isna(c) else c)
        )
    else:
        results['DriverColor'] = '#1f77b4'
    
    # driver labels
    driver_code_col = 'Abbreviation' if 'Abbreviation' in results.columns else 'DriverCode'
    results['DriverLabel'] = results[driver_code_col].astype(str)
    if 'TeamName' in results.columns:
        results['DriverLabel'] += ' (' + results['TeamName'] + ')'
    
    # highlight
    border_widths = [3 if str(code) == str(highlight_driver) else 0 for code in results[driver_code_col]]
    border_colors = ['#FFD700' if str(code) == str(highlight_driver) else 'rgba(0,0,0,0)' for code in results[driver_code_col]]
    
    hover_texts = []
    for _, row in results.iterrows():
        td = row[time_col]
        minutes, seconds = divmod(td.seconds, 60)
        milliseconds = td.microseconds // 1000
        time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        
        gap_str = f"<br>Gap to pole: +{row['GapToPole']:.3f}s" if show_gaps else ""
        pos_val = int(row[pos_col]) if pos_col in row and pd.notna(row[pos_col]) else "N/A"
        
        driver_code = row.get(driver_code_col, '')
        team_name = row.get('TeamName', '')
        
        hover_text = (
            f"<b>{driver_code}</b> ({team_name})<br>"
            f"Qualifying Time: {time_str}<br>"
            f"Position: {pos_val}"
            f"{gap_str}"
        )
        hover_texts.append(hover_text)
    
    results['HoverText'] = hover_texts
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=results['DriverLabel'],
        x=results['TimeSeconds'],
        orientation='h',
        marker=dict(
            color=results['DriverColor'],
            line=dict(color=border_colors, width=border_widths)
        ),
        customdata=results['HoverText'],
        hovertemplate='%{customdata}<extra></extra>',
        showlegend=False,
        name='Qualifying Time'
    ))
    
    title = custom_title if custom_title else f"Qualifying {qualifying_session} Results"
    
    min_x = results['TimeSeconds'].min() - 0.5
    max_x = results['TimeSeconds'].max() + 0.5
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Lap Time (seconds)",
        yaxis_title="Driver",
        height=max(500, len(results) * 30),
        margin=dict(l=150, r=50, t=80, b=50),
        showlegend=False
    )
    
    fig.update_xaxes(range=[min_x, max_x], showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=False)
    
    return fig



def plot_leading_laptime_evolution(drivers: List, laps: pd.DataFrame, track_status: pd.DataFrame):

    # Ensure 'LapTime' is in timedelta format
    if 'LapTime' not in laps.columns or not pd.api.types.is_timedelta64_dtype(laps['LapTime']):
        laps['LapTime'] = pd.to_timedelta(laps['LapTime'])

    # Drop rows where LapNumber or LapTime is NaN
    cleaned_laps = laps.dropna(subset=['LapNumber', 'LapTime']).copy()
    cleaned_laps['LapTimeSeconds'] = cleaned_laps['LapTime'].dt.total_seconds()

    plot_data = []
    current_fastest_overall_time = float('inf')
    current_fastest_overall_driver = None
    unique_lap_numbers = sorted(cleaned_laps['LapNumber'].unique())

    for lap_num in unique_lap_numbers:
        laps_up_to_current = cleaned_laps[cleaned_laps['LapNumber'] <= lap_num]
        if not laps_up_to_current.empty:
            fastest_idx_so_far = laps_up_to_current['LapTimeSeconds'].idxmin()
            fastest_row_so_far = laps_up_to_current.loc[fastest_idx_so_far]
            if fastest_row_so_far['LapTimeSeconds'] < current_fastest_overall_time:
                current_fastest_overall_time = fastest_row_so_far['LapTimeSeconds']
                current_fastest_overall_driver = fastest_row_so_far['Driver']
        if current_fastest_overall_driver is not None:
            plot_data.append({
                'LapNumber': lap_num,
                'LapTimeSeconds': current_fastest_overall_time,
                'Driver': current_fastest_overall_driver
            })

    leading_laps_plot_df = pd.DataFrame(plot_data)

    if leading_laps_plot_df.empty:
        print("No valid lap data to plot.")
    else:
        fig = go.Figure()
        unique_leading_drivers = leading_laps_plot_df['Driver'].unique()

        for driver in unique_leading_drivers:
            driver_laps_leading = leading_laps_plot_df[leading_laps_plot_df['Driver'] == driver]
            fig.add_trace(go.Scatter(
                x=driver_laps_leading['LapNumber'],
                y=driver_laps_leading['LapTimeSeconds'],
                mode='lines+markers',
                name=driver,
                hoverinfo='text',
                text=[
                    f"Lap: {int(row['LapNumber'])}<br>Driver: {row['Driver']}<br>Lap Time: {row['LapTimeSeconds']:.3f}s"
                    for idx, row in driver_laps_leading.iterrows()
                ],
                connectgaps=False,
                showlegend=True
            ))

        # Track Status Annotations
        track_status_changes = []
        previous_status = None
        for index, row in track_status.iterrows():
            if row['Status'] != previous_status:
                track_status_changes.append({'Time': row['Time'], 'Status': row['Status']})
            previous_status = row['Status']

        STATUS_MAPPING = {
            '1': {'text': 'Green Flag', 'color': 'green'},
            '2': {'text': 'Yellow Flag', 'color': 'goldenrod'},
            '3': {'text': 'Safety Car', 'color': 'blue'},
            '4': {'text': 'Red Flag', 'color': 'red'},
            '6': {'text': 'VSC', 'color': 'orange'},
            '13': {'text': 'Chequered Flag', 'color': 'grey'}
        }

        min_plot_lap = leading_laps_plot_df['LapNumber'].min()
        max_plot_lap = leading_laps_plot_df['LapNumber'].max()
        
        # Dictionary to track how many status labels are placed per lap
        lap_annotation_counts = {}

        for event in track_status_changes:
            candidate_laps = laps[laps['LapStartTime'] <= event['Time']]
            if not candidate_laps.empty:
                lap_num_for_event = candidate_laps['LapNumber'].max()

                if min_plot_lap <= lap_num_for_event <= max_plot_lap:
                    mapped_status = STATUS_MAPPING.get(event['Status'])
                    if mapped_status:
                        # Increment count for this lap to offset vertically
                        count = lap_annotation_counts.get(lap_num_for_event, 0)
                        y_offset = count * 20 # 20px shift per label
                        
                        fig.add_vline(
                            x=lap_num_for_event,
                            line_width=1,
                            line_dash="dot",
                            line_color=mapped_status['color']
                        )
                        
                        fig.add_annotation(
                            x=lap_num_for_event,
                            y=1, # Anchor to top of plot area
                            yref="paper",
                            yshift=-y_offset, # Move down based on count
                            text=mapped_status['text'],
                            showarrow=False,
                            font=dict(color=mapped_status['color'], size=10),
                            bgcolor="rgba(255,255,255,0.8)",
                            xanchor="left"
                        )
                        
                        lap_annotation_counts[lap_num_for_event] = count + 1

        fig.update_layout(
            title='Evolution of Leading Qualifying Lap Times with Stacked Status Flags',
            xaxis_title='Lap Number',
            yaxis_title='Lap Time (seconds)',
            hovermode='x unified',
        )
    return fig


def plot_leading_laptimes(drivers: List, laps: pd.DataFrame, track_status: pd.DataFrame):
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

    return fig
        


def plot_position_evolution_for_lap_x(
    laps: pd.DataFrame,
    results: pd.DataFrame,
    lap_num: int,
    color_map: Dict
):
    grid_data = results[['Abbreviation', 'GridPosition']].copy()
    grid_data.columns = ['Driver', 'GridPosition']

    # Get position at the end of lap_num
    lapx_pos = laps.pick_lap(lap_num)[['Driver', 'Position']].copy()
    lapx_pos.columns = ['Driver', 'Lap_x_Position']

    # Merge and calculate gain
    gain_df = pd.merge(grid_data, lapx_pos, on='Driver')
    gain_df['Lap_x_Gain'] = gain_df['GridPosition'] - gain_df['Lap_x_Position']
    gain_df = gain_df.sort_values('Lap_x_Gain', ascending=False)


    fig = px.bar(
        gain_df,
        x='Driver',
        y='Lap_x_Gain',
        color='Driver',
        color_discrete_map=color_map,
        title=f'Positions Gained/Lost on Lap {lap_num} (Grid vs Lap {lap_num} Finish)',
        labels={'Lap_x_Gain': 'Positions Gained'}
    )
    fig.update_layout(showlegend=False)
    return fig