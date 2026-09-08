import plotly.express as px
from ..constants.colors import compound_colors, track_status_colors



def plot_tyre_strategies(
        drivers: List,
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




def plot_total_pitstop_time(session):
    # extract data from pit_stops table
    if hasattr(session, 'pit_stops') and not session.pit_stops.empty:
        df_stops = session.pit_stops.copy()
        df_stops = df_stops.rename(columns={'Duration': 'Seconds', 'StopNumber': 'Number'})
        if pd.api.types.is_timedelta64_dtype(df_stops['Seconds']):
            df_stops['Seconds'] = df_stops['Seconds'].dt.total_seconds()
        # Ensure Lap column exists for the text display
        if 'LapNumber' in df_stops.columns:
            df_stops['Lap'] = df_stops['LapNumber'].apply(lambda x: f"Lap {int(x)}")
        else:
            df_stops['Lap'] = "Stop"
    else:
        # Fallback manual extraction
        individual_pitstops_list = []
        choosen_drivers = session.laps['Driver'].unique().tolist()

        for driver in choosen_drivers:
            driver_laps = session.laps.pick_driver(driver).reset_index(drop=True)
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

    # driver colors
    unique_drivers = df_stops['Driver'].unique()
    color_map = {}
    for d in unique_drivers:
        try:
            color_map[d] = fastf1.plotting.get_driver_color(d, session=session)
        except:
            color_map[d] = '#808080' # Fallback

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

