from __future__ import annotations
from typing import List, Optional
import warnings

import plotly.graph_objects as go
import pandas as pd

from .._core import layout
from .._core.annotations import MarkerInput, _add_markers, _add_track_status_on_laps, _lap_x
from ..values.colors import _compound_color_map, _driver_styles
from ..values.informations import _LapTimeline


def plot_tyre_strategies(
        laps: pd.DataFrame,
        track_status: Optional[pd.DataFrame] = None,
        drivers: Optional[List[str]] = None,
        title: Optional[str] = None,
        markers: MarkerInput = None,
    ) -> go.Figure:
    """Visualise tyre strategy and track status for multiple drivers.

    Generates a horizontal bar chart that shows the stints of each driver,
    colored by tyre compound. Neutralisations (SC, VSC, red flag) are
    highlighted as bands.

    Parameters
    ----------
    laps : pd.DataFrame
        DataFrame containing at least ``Driver``, ``Stint``, ``Compound`` and
        ``LapNumber`` columns.
    track_status : pd.DataFrame, optional
        fastf1 ``session.track_status``.
    drivers : list, optional
        Drivers to include. The order determines the order on the y‑axis.
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``lap`` or ``time``, optionally ``driver``).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if drivers is None:
        drivers = laps['Driver'].dropna().unique().tolist()
    drivers = [str(d) for d in drivers]

    driver_laps = laps[laps['Driver'].isin(drivers)].dropna(subset=['LapNumber']).copy()
    driver_laps['Compound'] = driver_laps['Compound'].fillna('UNKNOWN').str.upper()
    stints = (
        driver_laps.groupby(['Driver', 'Stint'])
        .agg(
            Compound=('Compound', 'first'),
            FirstLap=('LapNumber', 'min'),
            LastLap=('LapNumber', 'max'),
            TyreLife=('TyreLife', 'min'),
        )
        .reset_index()
    )
    stints['Length'] = stints['LastLap'] - stints['FirstLap'] + 1

    compound_colors = _compound_color_map(laps)
    fig = go.Figure()

    for compound, compound_stints in stints.groupby('Compound', sort=False):
        color = compound_colors.get(compound, compound_colors.get('UNKNOWN', '#808080'))
        fig.add_trace(go.Bar(
            y=compound_stints['Driver'],
            x=compound_stints['Length'],
            base=compound_stints['FirstLap'] - 1,
            orientation='h',
            name=compound.title(),
            marker=dict(
                color=color,
                # white separators between stints, light compounds (hard) get a visible outline
                line=dict(color=layout.MUTED if layout._outline(color) == layout.MUTED else layout.BACKGROUND, width=1.5),
            ),
            customdata=compound_stints[['Stint', 'FirstLap', 'LastLap', 'Length', 'TyreLife']].to_numpy(),
            hovertemplate=(
                '<b>%{y}</b><br>Compound: ' + compound.title() +
                '<br>Stint %{customdata[0]:.0f}: Lap %{customdata[1]:.0f}–%{customdata[2]:.0f}'
                ' (%{customdata[3]:.0f} laps)<br>Tyre age at start: %{customdata[4]:.0f} laps<extra></extra>'
            ),
        ))

    timeline = _LapTimeline(laps)
    _add_track_status_on_laps(fig, track_status, timeline, layer='above')
    _add_markers(fig, markers, lambda m: (_lap_x(m, timeline), m.driver if m.driver in drivers else None))

    layout._apply_layout(
        fig,
        title=title or 'Tyre Strategy per Driver',
        subtitle=layout._session_subtitle(laps),
        x_title=layout._axis_title('Lap'),
        y_title='Driver',
        legend_title='Compound',
        height=max(450, 28 * len(drivers) + 180),
    )
    styles = _driver_styles(laps, drivers)
    fig.update_layout(barmode='overlay', bargap=0.25)
    fig.update_yaxes(autorange='reversed')
    layout._style_driver_axis(fig, drivers, styles, axis='y')
    return fig


def plot_total_pitstop_time(
    laps: pd.DataFrame,
    drivers: Optional[List[str]] = None,
    title: Optional[str] = None,
    markers: MarkerInput = None,
) -> Optional[go.Figure]:
    """
    Plots the total time spent in the pit lane (pit in to pit out) for each driver,
    stacked per pit stop.

    Parameters:
    -----------
    laps : pd.DataFrame
        The FastF1 laps DataFrame (session.laps).
    drivers : list, optional
        Drivers to include (default: all).
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``driver``, optionally ``y`` in seconds).
    """
    if drivers is None:
        drivers = laps['Driver'].dropna().unique().tolist()
    drivers = [str(d) for d in drivers]

    stops = []
    for driver in drivers:
        driver_laps = laps[laps['Driver'] == driver].sort_values('LapNumber')
        pit_in = driver_laps['PitInTime'].shift(1)
        durations = (driver_laps['PitOutTime'] - pit_in).dt.total_seconds()
        valid = driver_laps['PitOutTime'].notna() & pit_in.notna() & (durations > 0)
        for lap, seconds in zip(driver_laps.loc[valid, 'LapNumber'], durations[valid]):
            stops.append({'Driver': driver, 'Lap': int(lap), 'Seconds': seconds})

    df_stops = pd.DataFrame(stops)
    if df_stops.empty:
        warnings.warn("No pit stop duration data is available for this session.", stacklevel=2)
        return None

    totals = df_stops.groupby('Driver')['Seconds'].sum().sort_values()
    order = totals.index.tolist()
    styles = _driver_styles(laps, order)

    fig = go.Figure()
    for driver in order:
        driver_stops = df_stops[df_stops['Driver'] == driver]
        fig.add_trace(go.Bar(
            x=driver_stops['Driver'],
            y=driver_stops['Seconds'],
            name=driver,
            text=[f"L{lap}" for lap in driver_stops['Lap']],
            textposition='inside',
            marker=dict(color=styles[driver].color, line=dict(color=layout.BACKGROUND, width=1.5)),
            customdata=driver_stops['Lap'],
            hovertemplate='<b>%{x}</b><br>Stop on lap %{customdata}<br>Pit lane time: %{y:.1f} s<extra></extra>',
            showlegend=False,
        ))

    _add_markers(fig, markers, lambda m: (
        m.driver if m.driver in order else None,
        m.y if m.y is not None else (totals.get(m.driver) if m.driver in order else None),
    ))

    layout._apply_layout(
        fig,
        title=title or 'Total Pit Lane Time per Driver',
        subtitle=layout._session_subtitle(laps),
        x_title='Driver',
        y_title=layout._axis_title('Pit Lane Time', 's'),
    )
    fig.update_layout(barmode='stack')
    layout._style_driver_axis(fig, order, styles)
    return fig
