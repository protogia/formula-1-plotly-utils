from __future__ import annotations
from typing import List, Optional
import warnings

import pandas as pd
import numpy as np

import plotly.graph_objects as go

from .._core import layout
from .._core import telemetry
from .._core.annotations import MarkerInput, _add_markers, _add_track_status_on_laps, _lap_x
from ..values.colors import (
    _compound_color_map, _driver_styles, category_palette, condition_colors,
)
from ..values.informations import _LapTimeline


def _category_resolver(drivers: List[str]):
    """Markers on a driver category axis: ``driver`` -> x, ``y`` -> value."""
    return lambda m: (m.driver if m.driver in drivers else None, m.y)


def _merge_weather(laps: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    """Adds the weather at the end of each lap (both use fastf1 session time)."""
    merged = pd.merge_asof(
        laps.dropna(subset=['Time']).sort_values('Time'),
        weather_data.sort_values('Time')[['Time', 'Rainfall', 'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed']],
        on='Time',
        direction='backward',  # closest weather sample before or at the lap end
    )
    merged = merged.dropna(subset=['Rainfall'])
    merged['Condition'] = np.where(merged['Rainfall'].astype(bool), 'Rain', 'Dry')
    return merged


def _add_distribution(fig: go.Figure, data: pd.DataFrame, group_col: str, color_map: dict, kind: str = 'box'):
    """One box/violin trace per group with all laps as points."""
    for group in [g for g in color_map if g in set(data[group_col])]:
        group_data = data[data[group_col] == group]
        common = dict(
            x=group_data['Driver'], y=group_data['LapTimeSeconds'], name=str(group).title() if group_col == 'Compound' else str(group),
            legendgroup=str(group), offsetgroup=str(group),
            marker=dict(color=layout._outline(color_map[group]), size=4), line=dict(color=layout._outline(color_map[group])),
            customdata=group_data[['LapNumber']].to_numpy(),
            hovertemplate='<b>%{x}</b> · Lap %{customdata[0]:.0f}<br>Lap time: %{y:.3f} s<extra>' + str(group) + '</extra>',
        )
        if kind == 'violin':
            fig.add_trace(go.Violin(**common, points='all', pointpos=0, jitter=0.5, box_visible=True, meanline_visible=False, fillcolor=layout._rgba(color_map[group], 0.25)))
        else:
            fig.add_trace(go.Box(**common, boxpoints='all', pointpos=0, jitter=0.5, fillcolor=layout._rgba(color_map[group], 0.25)))
    fig.update_layout(boxmode='group', violinmode='group')


def _driver_order(drivers: List[str], results: Optional[pd.DataFrame], laps: pd.DataFrame) -> List[str]:
    """Order by classification if results are given, otherwise by median lap time."""
    if results is not None and not results.empty:
        ordered = results.sort_values(by='Position')['Abbreviation'].tolist()
        return [d for d in ordered if d in drivers] + [d for d in drivers if d not in ordered]
    medians = laps.groupby('Driver')['LapTimeSeconds'].median().sort_values()
    return [d for d in medians.index if d in drivers] + [d for d in drivers if d not in medians.index]


def plot_laptime_distribution_weatherdependent(
        laps: pd.DataFrame,
        drivers: List,
        weather_data: pd.DataFrame,
        title: Optional[str] = None,
        markers: MarkerInput = None,
    ) -> go.Figure:
    """
    Lap time distribution per driver, split into dry and rain laps.

    Parameters:
    -----------
    laps : pd.DataFrame
        The FastF1 laps DataFrame (session.laps).
    drivers : list
        Driver abbreviations.
    weather_data : pd.DataFrame
        fastf1 ``session.weather_data``.
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``driver``, optionally ``y`` in seconds).
    """
    if weather_data is None or weather_data.empty:
        raise ValueError("weather_data is required to split laps into dry and rain conditions.")

    drivers_laps = laps[laps['Driver'].isin(drivers)].copy()
    drivers_laps['LapTimeSeconds'] = drivers_laps['LapTime'].dt.total_seconds()
    merged = _merge_weather(drivers_laps, weather_data)
    merged = telemetry._filter_slow_laps(merged, group_columns=['Driver', 'Condition'])

    order = _driver_order(list(drivers), None, merged)
    fig = go.Figure()
    _add_distribution(fig, merged, 'Condition', condition_colors, kind='violin')
    _add_markers(fig, markers, _category_resolver(order))

    layout._apply_layout(
        fig,
        title=title or 'Lap Time Distribution by Driver and Condition',
        subtitle=layout._session_subtitle(laps),
        x_title='Driver',
        y_title=layout._axis_title('Lap Time', 's'),
        legend_title='Condition',
    )
    layout._style_driver_axis(fig, order, _driver_styles(laps, order))
    return fig


def plot_laptime_distribution_per_compound(
        laps: pd.DataFrame,
        drivers: List,
        results: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        markers: MarkerInput = None,
    ) -> go.Figure:
    """
    Lap time distribution per driver and tyre compound (slow laps filtered).

    Parameters:
    -----------
    laps : pd.DataFrame
        The FastF1 laps DataFrame (session.laps).
    drivers : list
        Driver abbreviations.
    results : pd.DataFrame, optional
        fastf1 ``session.results`` to order drivers by classification.
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``driver``, optionally ``y`` in seconds).
    """
    filtered_laps = laps[laps['Driver'].isin(drivers)].copy()
    filtered_laps['LapTimeSeconds'] = filtered_laps['LapTime'].dt.total_seconds()
    filtered_laps = telemetry._filter_slow_laps(laps=filtered_laps, group_columns=["Driver", "Compound"])
    filtered_laps['Compound'] = filtered_laps['Compound'].fillna('UNKNOWN').str.upper()

    order = _driver_order(list(drivers), results, filtered_laps)
    fig = go.Figure()
    _add_distribution(fig, filtered_laps, 'Compound', _compound_color_map(laps))
    _add_markers(fig, markers, _category_resolver(order))

    layout._apply_layout(
        fig,
        title=title or 'Lap Time Performance per Driver and Tyre Compound',
        subtitle=layout._session_subtitle(laps),
        x_title='Driver',
        y_title=layout._axis_title('Lap Time', 's'),
        legend_title='Tyre Compound',
    )
    layout._style_driver_axis(fig, order, _driver_styles(laps, order))
    return fig


def _split_qualifying_rounds(laps: pd.DataFrame) -> pd.DataFrame:
    """Assigns Q1/Q2/Q3 (or SQ1..SQ3 for sprint qualifying) based on fastf1's session split."""
    session = getattr(laps, 'session', None)
    prefix = 'SQ' if session is not None and 'Sprint' in str(getattr(session, 'name', '')) else 'Q'
    try:
        parts = laps.split_qualifying_sessions()
    except Exception as exc:
        warnings.warn(f"Could not split qualifying sessions ({exc}); all laps are shown as one round.", stacklevel=3)
        return laps.assign(QualifyingRound='Qualifying')
    return pd.concat([
        pd.DataFrame(part).assign(QualifyingRound=f'{prefix}{i + 1}')
        for i, part in enumerate(parts) if part is not None and not part.empty
    ])


def plot_laptime_distribution_per_qualifyinground(
        laps: pd.DataFrame,
        drivers: List,
        results: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        markers: MarkerInput = None,
        quicklaps_threshold: Optional[float] = 1.07,
    ) -> go.Figure:
    """
    Lap time distribution per driver and qualifying round.

    Parameters:
    -----------
    laps : fastf1.core.Laps
        The FastF1 laps (session.laps of a (sprint) qualifying).
    drivers : list
        Driver abbreviations.
    results : pd.DataFrame, optional
        fastf1 ``session.results`` to order drivers by classification.
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``driver``, optionally ``y`` in seconds).
    quicklaps_threshold : float, optional
        Only laps within this factor of the session best are shown (removes out/in laps).
        ``None`` shows all laps.
    """
    rounds = _split_qualifying_rounds(laps)
    filtered_laps = rounds[rounds['Driver'].isin(drivers)].dropna(subset=['LapTime']).copy()
    filtered_laps['LapTimeSeconds'] = filtered_laps['LapTime'].dt.total_seconds()
    if quicklaps_threshold is not None and not filtered_laps.empty:
        best = rounds['LapTime'].dt.total_seconds().min()
        filtered_laps = filtered_laps[filtered_laps['LapTimeSeconds'] <= best * quicklaps_threshold]

    round_names = list(dict.fromkeys(rounds['QualifyingRound']))
    round_colors = {r: category_palette[i % len(category_palette)] for i, r in enumerate(round_names)}

    order = _driver_order(list(drivers), results, filtered_laps)
    fig = go.Figure()
    _add_distribution(fig, filtered_laps, 'QualifyingRound', round_colors)
    _add_markers(fig, markers, _category_resolver(order))

    layout._apply_layout(
        fig,
        title=title or 'Lap Time Performance per Driver and Qualifying Round',
        subtitle=layout._session_subtitle(laps),
        x_title='Driver',
        y_title=layout._axis_title('Lap Time', 's'),
        legend_title='Qualifying Round',
    )
    layout._style_driver_axis(fig, order, _driver_styles(laps, order))
    return fig


def plot_best_laptime(
        results: pd.DataFrame,
        drivers: list,
        criteria: str = "qualifying",
        laps: Optional[pd.DataFrame] = None,
        weather_data: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        markers: MarkerInput = None,
    ) -> go.Figure:
    """
    Best lap time per driver, split by a criteria.

    Parameters:
    -----------
    results : pd.DataFrame
        fastf1 ``session.results``.
    drivers : list
        Driver abbreviations.
    criteria : 'qualifying' | 'compound' | 'weather'
        'qualifying' uses Q1/Q2/Q3 from results, 'compound' needs ``laps``,
        'weather' needs ``laps`` and ``weather_data``.
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``driver``, optionally ``y`` in seconds).
    """
    if criteria == "qualifying":
        best = results[results['Abbreviation'].isin(drivers)][['Abbreviation', 'Q1', 'Q2', 'Q3']].melt(
            id_vars='Abbreviation', var_name='Category', value_name='BestLapTime',
        ).rename(columns={'Abbreviation': 'Driver'})
        best['BestLapTime'] = best['BestLapTime'].dt.total_seconds()
        color_map = {q: category_palette[i] for i, q in enumerate(['Q1', 'Q2', 'Q3'])}
        legend_title = 'Qualifying Round'

    elif criteria in ("compound", "weather"):
        if laps is None:
            raise ValueError(f"criteria '{criteria}' requires laps.")
        driver_laps = laps[laps['Driver'].isin(drivers)].dropna(subset=['LapTime']).copy()
        if criteria == "compound":
            driver_laps['Category'] = driver_laps['Compound'].fillna('UNKNOWN').str.upper()
            color_map = _compound_color_map(laps)
            legend_title = 'Tyre Compound'
        else:
            if weather_data is None:
                raise ValueError("criteria 'weather' requires weather_data.")
            driver_laps = _merge_weather(driver_laps, weather_data).rename(columns={'Condition': 'Category'})
            color_map = condition_colors
            legend_title = 'Condition'
        best = driver_laps.groupby(['Driver', 'Category'])['LapTime'].min().dt.total_seconds().reset_index(name='BestLapTime')

    else:
        raise ValueError("criteria must be 'qualifying', 'compound' or 'weather'.")

    best = best.dropna(subset=['BestLapTime'])
    order = best.groupby('Driver')['BestLapTime'].min().sort_values().index.tolist()
    styles = _driver_styles(laps if laps is not None else results, order)

    fig = go.Figure()
    symbols = ['circle', 'diamond', 'square', 'triangle-up', 'x', 'star']
    for i, category in enumerate([c for c in color_map if c in set(best['Category'])]):
        data = best[best['Category'] == category]
        fig.add_trace(go.Scatter(
            x=data['Driver'], y=data['BestLapTime'], mode='markers',
            name=category.title() if criteria == "compound" else category,
            marker=dict(color=color_map[category], symbol=symbols[i % len(symbols)], size=11,
                        line=dict(color=layout.BACKGROUND, width=1)),
            hovertemplate='<b>%{x}</b><br>' + category + ': %{y:.3f} s<extra></extra>',
        ))
    _add_markers(fig, markers, _category_resolver(order))

    layout._apply_layout(
        fig,
        title=title or f'Best Lap Time per Driver by {legend_title}',
        subtitle=layout._session_subtitle(laps),
        x_title='Driver',
        y_title=layout._axis_title('Best Lap Time', 's'),
        legend_title=legend_title,
    )
    layout._style_driver_axis(fig, order, styles)
    return fig


def plot_driver_position_per_lap(
        laps: pd.DataFrame,
        drivers: Optional[List[str]] = None,
        track_status: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        markers: MarkerInput = None,
    ) -> go.Figure:
    """
    Position of each driver at the end of every lap.

    Parameters:
    -----------
    laps : pd.DataFrame
        The FastF1 laps DataFrame (session.laps).
    drivers : list, optional
        Driver abbreviations (default: all).
    track_status : pd.DataFrame, optional
        fastf1 ``session.track_status`` to highlight SC/VSC/red flag phases.
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``lap`` or ``time``, optionally ``driver``).
    """
    if drivers is None:
        drivers = laps['Driver'].dropna().unique().tolist()
    drivers = [str(d) for d in drivers]
    styles = _driver_styles(laps, drivers)

    fig = go.Figure()
    for driver in drivers:
        drv_laps = laps[laps['Driver'] == driver].sort_values('LapNumber')
        if drv_laps.empty:
            continue
        fig.add_trace(go.Scatter(
            x=drv_laps['LapNumber'],
            y=drv_laps['Position'],
            mode='lines+markers',
            name=driver,
            line=dict(color=styles[driver].color, dash=styles[driver].dash, width=2),
            marker=dict(size=4),
            hovertemplate=f'<b>{driver}</b><br>Lap %{{x}}<br>Position: P%{{y:.0f}}<extra></extra>',
        ))

    timeline = _LapTimeline(laps)
    _add_track_status_on_laps(fig, track_status, timeline)

    def _resolve(m):
        x = _lap_x(m, timeline)
        if m.y is not None or x is None or m.driver not in drivers:
            return x, m.y
        drv_laps = laps[laps['Driver'] == m.driver].dropna(subset=['Position']).sort_values('LapNumber')
        return x, float(np.interp(x, drv_laps['LapNumber'], drv_laps['Position'])) if not drv_laps.empty else None
    _add_markers(fig, markers, _resolve)

    layout._apply_layout(
        fig,
        title=title or 'Driver Positions per Lap',
        subtitle=layout._session_subtitle(laps),
        x_title=layout._axis_title('Lap'),
        y_title='Position',
        legend_title='Driver',
    )
    max_pos = laps['Position'].max()
    fig.update_yaxes(range=[(max_pos if pd.notna(max_pos) else 20) + 0.5, 0.5], dtick=1)  # P1 at the top
    return fig


def plot_gap_to_leader_evolution(
    laps: pd.DataFrame,
    track_status: Optional[pd.DataFrame] = None,
    drivers: Optional[List[str]] = None,
    title: Optional[str] = None,
    markers: MarkerInput = None,
) -> go.Figure:
    """
    Race trace: gap of each driver to the leader at the end of every lap.

    Parameters:
    -----------
    laps : pd.DataFrame
        The FastF1 laps DataFrame (session.laps).
    track_status : pd.DataFrame, optional
        fastf1 ``session.track_status`` to highlight SC/VSC/red flag phases.
    drivers : list, optional
        Driver abbreviations (default: all).
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``lap`` or ``time``, optionally ``driver``).
    """
    trace_df = laps.dropna(subset=['LapNumber', 'Time'])[['Driver', 'LapNumber', 'Time']].copy()
    # gap = session time at the end of the lap compared to the first car completing that lap
    trace_df['GapToLeader'] = (
        trace_df['Time'] - trace_df.groupby('LapNumber')['Time'].transform('min')
    ).dt.total_seconds()

    if drivers is None:
        drivers = trace_df['Driver'].unique().tolist()
    drivers = [str(d) for d in drivers]
    styles = _driver_styles(laps, drivers)

    fig = go.Figure()
    for driver in drivers:
        driver_data = trace_df[trace_df['Driver'] == driver].sort_values('LapNumber')
        fig.add_trace(go.Scatter(
            x=driver_data['LapNumber'],
            y=driver_data['GapToLeader'],
            mode='lines',
            name=driver,
            line=dict(color=styles[driver].color, dash=styles[driver].dash, width=2),
            hovertemplate=f"{driver}: +%{{y:.3f}} s<extra></extra>",
        ))

    timeline = _LapTimeline(laps)
    _add_track_status_on_laps(fig, track_status, timeline)

    def _resolve(m):
        x = _lap_x(m, timeline)
        if m.y is not None or x is None or m.driver not in drivers:
            return x, m.y
        driver_data = trace_df[trace_df['Driver'] == m.driver].sort_values('LapNumber')
        return x, float(np.interp(x, driver_data['LapNumber'], driver_data['GapToLeader'])) if not driver_data.empty else None
    _add_markers(fig, markers, _resolve)

    layout._apply_layout(
        fig,
        title=title or 'Race Trace: Gap to Leader',
        subtitle=layout._session_subtitle(laps),
        x_title=layout._axis_title('Lap'),
        y_title=layout._axis_title('Gap to Leader', 's'),
        legend_title='Driver',
    )
    fig.update_layout(hovermode='x unified')
    fig.update_yaxes(autorange='reversed')  # leader at the top
    return fig


def plot_qualifying_results(
    results: pd.DataFrame,
    qualifying_session: str = 'Q3',
    show_gaps: bool = True,
    title: Optional[str] = None,
    highlight_driver: Optional[str] = None,
    sort_by: str = 'time',
    markers: MarkerInput = None,
) -> Optional[go.Figure]:
    """
    Plot qualifying results from a FastF1 session.

    Parameters:
    -----------
    results : pd.DataFrame
        fastf1 ``session.results`` (ideally passed through ``apply_driver_colors(results, session)``).
    qualifying_session : 'Q1' | 'Q2' | 'Q3'
    show_gaps : bool
        Show the gap to pole in the hover text.
    title : str, optional
        Custom chart title.
    highlight_driver : str, optional
        Driver abbreviation to outline.
    sort_by : 'time' | 'position'
    markers : ChartMarker | dict | list, optional
        Events to highlight (``driver``, optionally ``y`` as lap time in seconds).
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
    pole_time_s = results['TimeSeconds'].min()
    results['GapToPole'] = results['TimeSeconds'] - pole_time_s

    # sort
    pos_col = 'Position' if 'Position' in results.columns else 'GridPosition'
    if sort_by == 'time':
        results = results.sort_values('TimeSeconds', ascending=False)
    else:
        results = results.sort_values(pos_col, ascending=False)

    driver_code_col = 'Abbreviation' if 'Abbreviation' in results.columns else 'DriverCode'
    codes = results[driver_code_col].astype(str).tolist()
    styles = _driver_styles(results, codes)

    # driver labels
    if 'DriverLabel' not in results.columns:
        results['DriverLabel'] = results[driver_code_col].astype(str)
        if 'TeamName' in results.columns:
            results['DriverLabel'] += ' (' + results['TeamName'] + ')'
    labels = dict(zip(codes, results['DriverLabel']))

    # highlight
    border_widths = [3 if code == str(highlight_driver) else 0 for code in codes]
    border_colors = [layout.TEXT if code == str(highlight_driver) else 'rgba(0,0,0,0)' for code in codes]

    hover_texts = []
    for _, row in results.iterrows():
        td = row[time_col]
        minutes, seconds = divmod(td.seconds, 60)
        time_str = f"{minutes:d}:{seconds:02d}.{td.microseconds // 1000:03d}"
        gap_str = f"<br>Gap to pole: +{row['GapToPole']:.3f} s" if show_gaps else ""
        pos_val = int(row[pos_col]) if pd.notna(row.get(pos_col)) else "N/A"
        hover_texts.append(
            f"<b>{row.get(driver_code_col, '')}</b> ({row.get('TeamName', '')})<br>"
            f"{qualifying_session} time: {time_str}<br>"
            f"Position: P{pos_val}{gap_str}"
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=results['DriverLabel'],
        x=results['TimeSeconds'],
        orientation='h',
        marker=dict(
            color=[styles[c].color for c in codes],
            line=dict(color=border_colors, width=border_widths),
        ),
        customdata=hover_texts,
        hovertemplate='%{customdata}<extra></extra>',
        showlegend=False,
        name='Qualifying Time',
    ))

    times = dict(zip(codes, results['TimeSeconds']))
    _add_markers(fig, markers, lambda m: (
        m.y if m.y is not None else times.get(m.driver),
        labels.get(m.driver),
    ))

    layout._apply_layout(
        fig,
        title=title or f"Qualifying {qualifying_session} Results",
        x_title=layout._axis_title('Lap Time', 's'),
        y_title='Driver',
        height=max(500, len(results) * 30 + 150),
        showlegend=False,
    )
    fig.update_xaxes(range=[results['TimeSeconds'].min() - 0.5, results['TimeSeconds'].max() + 0.5])
    fig.update_yaxes(showgrid=False)
    return fig


def _leading_laptimes(laps: pd.DataFrame, drivers: Optional[List]) -> pd.DataFrame:
    """Fastest lap time so far and its driver, per lap number."""
    if drivers is not None:
        laps = laps[laps['Driver'].isin(drivers)]
    cleaned = laps.dropna(subset=['LapNumber', 'LapTime']).copy()
    cleaned['LapTimeSeconds'] = pd.to_timedelta(cleaned['LapTime']).dt.total_seconds()

    plot_data = []
    best_time, best_driver = float('inf'), None
    for lap_num in sorted(cleaned['LapNumber'].unique()):
        lap_best = cleaned[cleaned['LapNumber'] == lap_num].nsmallest(1, 'LapTimeSeconds')
        if not lap_best.empty and lap_best['LapTimeSeconds'].iloc[0] < best_time:
            best_time = lap_best['LapTimeSeconds'].iloc[0]
            best_driver = lap_best['Driver'].iloc[0]
        if best_driver is not None:
            plot_data.append({'LapNumber': lap_num, 'LapTimeSeconds': best_time, 'Driver': best_driver})
    return pd.DataFrame(plot_data)


def plot_leading_laptime_evolution(
    drivers: Optional[List],
    laps: pd.DataFrame,
    track_status: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    markers: MarkerInput = None,
) -> Optional[go.Figure]:
    """
    Evolution of the fastest lap time so far, colored by the driver holding it.

    Parameters:
    -----------
    drivers : list, optional
        Driver abbreviations to consider (None: all).
    laps : pd.DataFrame
        The FastF1 laps DataFrame (session.laps).
    track_status : pd.DataFrame, optional
        fastf1 ``session.track_status`` to highlight SC/VSC/red flag phases.
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``lap`` or ``time``).
    """
    leading = _leading_laptimes(laps, drivers)
    if leading.empty:
        print("No valid lap data to plot.")
        return None

    all_laps = leading['LapNumber']
    styles = _driver_styles(laps, leading['Driver'].unique().tolist())

    fig = go.Figure()
    for driver in leading['Driver'].unique():
        # NaN where another driver holds the record -> separate segments per driver
        held = leading['LapTimeSeconds'].where(leading['Driver'] == driver)
        # connect a segment to the start of the next one so the step is visible
        held = held.fillna(leading['LapTimeSeconds'].where(leading['Driver'].shift(1) == driver))
        fig.add_trace(go.Scatter(
            x=all_laps, y=held, mode='lines+markers', name=driver, line_shape='hv',
            line=dict(color=styles[driver].color, width=2.5), marker=dict(size=5),
            connectgaps=False,
            hovertemplate=f"{driver}: %{{y:.3f}} s<extra></extra>",
        ))

    timeline = _LapTimeline(laps)
    _add_track_status_on_laps(fig, track_status, timeline)

    def _resolve(m):
        x = _lap_x(m, timeline)
        if m.y is not None or x is None:
            return x, m.y
        return x, float(np.interp(x, leading['LapNumber'], leading['LapTimeSeconds']))
    _add_markers(fig, markers, _resolve)

    layout._apply_layout(
        fig,
        title=title or 'Evolution of the Leading Lap Time',
        subtitle=layout._session_subtitle(laps),
        x_title=layout._axis_title('Lap'),
        y_title=layout._axis_title('Lap Time', 's'),
        legend_title='Record Holder',
    )
    fig.update_layout(hovermode='x unified')
    return fig


def plot_leading_laptimes(
    drivers: Optional[List],
    laps: pd.DataFrame,
    track_status: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    markers: MarkerInput = None,
) -> Optional[go.Figure]:
    """Alias of :func:`plot_leading_laptime_evolution` (kept for backwards compatibility)."""
    return plot_leading_laptime_evolution(drivers, laps, track_status, title=title, markers=markers)


def plot_position_evolution_for_lap_x(
    laps: pd.DataFrame,
    results: pd.DataFrame,
    lap_num: int,
    title: Optional[str] = None,
    markers: MarkerInput = None,
) -> go.Figure:
    """
    Positions gained/lost between the grid and the end of ``lap_num``.

    Parameters:
    -----------
    laps : pd.DataFrame
        The FastF1 laps DataFrame (session.laps).
    results : pd.DataFrame
        fastf1 ``session.results`` (grid positions).
    lap_num : int
    title : str, optional
        Custom chart title.
    markers : ChartMarker | dict | list, optional
        Events to highlight (``driver``, optionally ``y``).
    """
    grid_data = results[['Abbreviation', 'GridPosition']].copy()
    grid_data.columns = ['Driver', 'GridPosition']
    # pit lane starters have grid position 0
    grid_data.loc[grid_data['GridPosition'] == 0, 'GridPosition'] = len(grid_data)

    # Get position at the end of lap_num
    lapx_pos = laps[laps['LapNumber'] == lap_num][['Driver', 'Position']].copy()
    lapx_pos.columns = ['Driver', 'LapPosition']

    # Merge and calculate gain
    gain_df = pd.merge(grid_data, lapx_pos, on='Driver').dropna()
    gain_df['Gain'] = gain_df['GridPosition'] - gain_df['LapPosition']
    gain_df = gain_df.sort_values(['Gain', 'LapPosition'], ascending=[False, True])

    order = gain_df['Driver'].tolist()
    styles = _driver_styles(laps, order)

    fig = go.Figure(go.Bar(
        x=gain_df['Driver'],
        y=gain_df['Gain'],
        marker=dict(color=[styles[d].color for d in order]),
        customdata=gain_df[['GridPosition', 'LapPosition']].to_numpy(),
        hovertemplate=(
            '<b>%{x}</b><br>Grid: P%{customdata[0]:.0f}<br>'
            f'Lap {lap_num}: P%{{customdata[1]:.0f}}<br>Gain: %{{y:+.0f}}<extra></extra>'
        ),
        showlegend=False,
    ))

    gains = dict(zip(gain_df['Driver'], gain_df['Gain']))
    _add_markers(fig, markers, lambda m: (
        m.driver if m.driver in gains else None,
        m.y if m.y is not None else gains.get(m.driver),
    ))

    layout._apply_layout(
        fig,
        title=title or f'Positions Gained/Lost on Lap {lap_num} (Grid vs Lap {lap_num} Finish)',
        subtitle=layout._session_subtitle(laps),
        x_title='Driver',
        y_title=layout._axis_title('Positions Gained'),
        showlegend=False,
    )
    layout._style_driver_axis(fig, order, styles)
    fig.update_yaxes(dtick=1)
    return fig
