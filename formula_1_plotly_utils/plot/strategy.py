from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import fastf1
import fastf1.plotting

import pandas as pd
from typing import List, Optional
from values.colors import track_status_colors, compound_colors, get_driver_colors
from values.informations import _get_track_status_changes

def plot_tyre_strategies(
        laps: pd.DataFrame,
        track_status: pd.DataFrame,
        drivers: Optional[List[str]],

    ) -> go.Figure:
    """Visualise tyre strategy and track status for multiple drivers.

    Generates a stacked horizontal bar chart that shows the number of laps
    each driver spent on each tyre compound.  Vertical dashed lines
    indicate track‑status changes (e.g. safety car, yellow flag).  For
    each status change a coloured marker is plotted on the y‑axis next
    to the driver bar.

    Parameters
    ----------
    laps : pd.DataFrame
        DataFrame containing at least ``Driver``, ``Stint``, ``Compound`` and
        ``LapNumber`` columns.
    track_status : pd.DataFrame
        DataFrame containing at least ``Message`` and ``Time`` columns.
        ``Message`` should be one of the keys in the ``track_status_colors``
        mapping.
    drivers : list
        List of driver names to include in the plot.  The order determines
        the order on the y‑axis.
    
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

    if drivers is None:
        drivers = laps['Driver'].unique().tolist()

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


def plot_total_pitstop_time(
    laps: pd.DataFrame,
):
    """
    Plots the total time spent in the pits for each driver.
    
    Parameters:
    -----------
    laps : pd.DataFrame
        The FastF1 laps DataFrame (session.laps).
    """
    individual_pitstops_list = []
    choosen_drivers = laps['Driver'].unique().tolist()

    # Manual extraction using the laps DataFrame
    for driver in choosen_drivers:
        # Filter for the driver and reset index to match your original logic
        driver_laps = laps[laps['Driver'] == driver].reset_index(drop=True)
        stops = driver_laps[driver_laps['PitOutTime'].notnull()]

        for i, stop in stops.iterrows():
            duration = 0
            if i > 0:
                p_in = driver_laps.loc[i-1, 'PitInTime']
                p_out = stop['PitOutTime']
                if pd.notnull(p_in) and pd.notnull(p_out):
                    duration = (pd.to_timedelta(p_out) - pd.to_timedelta(p_in)).total_seconds()

            if duration > 0:
                individual_pitstops_list.append({
                    'Driver': driver,
                    'Lap': f"Lap {int(stop['LapNumber'])}",
                    'Seconds': duration
                })
                
    df_stops = pd.DataFrame(individual_pitstops_list)

    if df_stops.empty:
        print("No pit stop duration data is available for this session.")
        return

    # sort drivers by total time spent
    totals = df_stops.groupby('Driver')['Seconds'].sum().sort_values().index.tolist()

    # driver colors extraction without session object
    unique_drivers = df_stops['Driver'].unique()
    color_map = {}
    for d in unique_drivers:
        try:
            # Look up the team name from the laps DataFrame for this driver
            driver_laps = laps[laps['Driver'] == d]
            if not driver_laps.empty:
                team_name = driver_laps['Team'].iloc[0]
                color_map[d] = fastf1.plotting.get_team_color(team_name)
            else:
                color_map[d] = '#808080'
        except:
            color_map[d] = '#808080' # Fallback

    # Plotly visualization
    fig = px.bar(
        df_stops,
        x='Driver', 
        y='Seconds', 
        color='Driver', 
        text='Lap', 
        title='Total Time Spent in Pit Box',
        labels={'Seconds': 'Seconds', 'Driver': 'Driver'},
        category_orders={'Driver': totals},
        color_discrete_map=color_map,
    )

    fig.update_traces(textposition='inside')
    fig.update_layout(
        barmode='stack',
        showlegend=False,
        yaxis_title="Total Seconds (Pit In to Pit Out)",
        xaxis_title="Drivers"
    )

    fig.show()
