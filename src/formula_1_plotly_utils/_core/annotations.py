from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Union
import warnings

import pandas as pd
import plotly.graph_objects as go

from . import layout
from ..values import colors
from ..values.informations import _LapTimeline, _get_track_status_periods, _to_minutes


@dataclass
class ChartMarker:
    """
    Event to highlight in a chart, e.g. ``ChartMarker("VER crash", lap=1, driver="VER")``.

    Which position is used depends on the x-axis of the chart:

    - lap axis: ``lap`` (or ``time``, converted via the leader's laps)
    - time axis: ``time`` as session time (timedelta, 'HH:MM:SS' or minutes as number)
    - distance axis: ``distance`` in meters
    - driver axis (bar/box charts): ``driver``

    ``driver`` additionally pins the marker to that driver's value in line charts,
    ``y`` sets the vertical position explicitly. Without a vertical position the
    marker is drawn as a vertical line with a label.
    """
    label: str
    lap: Optional[float] = None
    time: Optional[Union[pd.Timedelta, str, float]] = None
    distance: Optional[float] = None
    driver: Optional[str] = None
    y: Optional[float] = None
    color: Optional[str] = None


MarkerInput = Union[ChartMarker, Mapping[str, Any], Iterable[Union[ChartMarker, Mapping[str, Any]]], None]
Resolver = Callable[[ChartMarker], Tuple[Any, Any]]


def _normalize_markers(markers: MarkerInput) -> List[ChartMarker]:
    """Accepts a ChartMarker, a dict with the same keys, or a list of both."""
    if markers is None:
        return []
    if isinstance(markers, (ChartMarker, Mapping)):
        markers = [markers]
    normalized = []
    for m in markers:
        if isinstance(m, ChartMarker):
            normalized.append(m)
        elif isinstance(m, Mapping):
            normalized.append(ChartMarker(**m))
        else:
            raise TypeError(f"Marker must be a ChartMarker or dict, got {type(m).__name__}.")
    return normalized


def _lap_x(marker: ChartMarker, timeline: Optional[_LapTimeline] = None) -> Optional[float]:
    """Marker position on a lap axis."""
    if marker.lap is not None:
        return float(marker.lap)
    if marker.time is not None and timeline is not None and not timeline.is_empty:
        return timeline.to_lap(pd.to_timedelta(_to_minutes(marker.time), unit="m"))
    return None


def _time_x(marker: ChartMarker, timeline: Optional[_LapTimeline] = None) -> Optional[float]:
    """Marker position on a session time axis (minutes)."""
    if marker.time is not None:
        return _to_minutes(marker.time)
    if marker.lap is not None and timeline is not None and not timeline.is_empty:
        return timeline.to_time(marker.lap).total_seconds() / 60.0
    return None


def _add_markers(
    fig: go.Figure,
    markers: MarkerInput,
    resolve: Resolver,
    *,
    row: Any = None,
    col: Any = None,
    label_row: Optional[int] = None,
    label_col: Optional[int] = None,
) -> go.Figure:
    """
    Draws markers in one consistent style.

    ``resolve`` maps a marker to ``(x, y)`` of the chart; a missing ``y`` draws a
    vertical line, a missing ``x`` a horizontal line, both a highlighted point.
    """
    label_row = label_row if label_row is not None else (row if row != "all" else 1)
    label_col = label_col if label_col is not None else (col if col != "all" else 1)
    label_box = dict(
        font=dict(color=layout.TEXT, size=12),
        bgcolor="rgba(255,255,255,0.9)",
        borderpad=4,
        borderwidth=1,
    )

    line_count = 0
    for marker in _normalize_markers(markers):
        x, y = resolve(marker)
        color = marker.color or colors.marker_color
        if x is None and y is None:
            warnings.warn(
                f"Marker '{marker.label}' has no position usable for this chart "
                "(check lap/time/distance/driver) and is skipped.",
                stacklevel=3,
            )
            continue

        if y is None:
            fig.add_vline(x=x, line=dict(color=color, width=1.5, dash="dot"), row=row, col=col)
            fig.add_annotation(
                # below the (up to 3) staggered track status labels
                x=x, y=1, yref="y domain", yanchor="top", yshift=-50 - 26 * (line_count % 4),
                text=marker.label, showarrow=False, xanchor="left", xshift=4,
                bordercolor=color, row=label_row, col=label_col, **label_box,
            )
            line_count += 1
        elif x is None:
            fig.add_hline(y=y, line=dict(color=color, width=1.5, dash="dot"), row=row, col=col)
            fig.add_annotation(
                x=1, xref="x domain", y=y, text=marker.label, showarrow=False,
                xanchor="right", yanchor="bottom", bordercolor=color,
                row=label_row, col=label_col, **label_box,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[x], y=[y], mode="markers",
                    marker=dict(symbol="circle-open", size=16, color=color, line=dict(width=2.5)),
                    hovertemplate=f"{marker.label}<extra></extra>",
                    showlegend=False,
                ),
                row=label_row, col=label_col,
            )
            fig.add_annotation(
                x=x, y=y, text=marker.label, showarrow=True, arrowhead=2, arrowcolor=color,
                ax=0, ay=-45, bordercolor=color, row=label_row, col=label_col, **label_box,
            )
    return fig


def _add_track_status(
    fig: go.Figure,
    track_status: Optional[pd.DataFrame],
    to_x: Callable[[pd.Timedelta], float],
    end_time: pd.Timedelta,
    *,
    include_yellow: bool = False,
    layer: str = "below",
    row: Any = None,
    col: Any = None,
    label_row: Optional[int] = None,
    label_col: Optional[int] = None,
) -> go.Figure:
    """
    Highlights neutralisations (SC, VSC, red flag) in one consistent style:
    a translucent band per period with its label at the top, and a dotted line
    at the moment the VSC ending was announced.
    """
    if track_status is None or track_status.empty:
        return fig

    label_row = label_row if label_row is not None else (row if row != "all" else 1)
    label_col = label_col if label_col is not None else (col if col != "all" else 1)

    span = max(to_x(end_time) - to_x(pd.to_timedelta(0, unit="s")), 1e-9)
    shown_kinds, has_ending = [], False
    prev_label_x, level = None, 0
    for period in _get_track_status_periods(track_status, include_yellow=include_yellow):
        style = colors.track_status_styles[period.kind]
        x0 = to_x(period.start)
        x1 = to_x(period.end if period.end is not None else end_time)
        if x1 <= x0:
            continue

        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=style["color"], opacity=0.18,
            line_width=0, layer=layer, row=row, col=col,
        )
        if style["label"]:
            # stagger labels of periods close to each other
            level = level + 1 if prev_label_x is not None and x0 - prev_label_x < 0.06 * span and level < 2 else 0
            prev_label_x = x0
            fig.add_annotation(
                x=x0, y=1, yref="y domain", text=f"<b>{style['label']}</b>", showarrow=False,
                xanchor="left", yanchor="top", xshift=2, yshift=-15 * level,
                font=dict(color=style["text"], size=11),
                row=label_row, col=label_col,
            )
        if period.ending is not None:
            fig.add_vline(
                x=to_x(period.ending), line=dict(color=style["color"], width=1.5, dash="dot"),
                row=row, col=col,
            )
            has_ending = True
        if period.kind not in shown_kinds:
            shown_kinds.append(period.kind)

    # legend entries
    for i, kind in enumerate(shown_kinds):
        style = colors.track_status_styles[kind]
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(symbol="square", size=12, color=style["color"], opacity=0.5),
                name=style["name"], legendgroup="track_status",
                legendgrouptitle_text="Track Status" if i == 0 else None,
                hoverinfo="skip",
            ),
            row=label_row, col=label_col,
        )
    if has_ending:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=colors.track_status_styles["VSC"]["color"], dash="dot", width=1.5),
                name="VSC ending", legendgroup="track_status", hoverinfo="skip",
            ),
            row=label_row, col=label_col,
        )
    return fig


def _add_track_status_on_laps(
    fig: go.Figure,
    track_status: Optional[pd.DataFrame],
    timeline: _LapTimeline,
    **kwargs,
) -> go.Figure:
    """Track status on a lap axis."""
    if timeline.is_empty:
        return fig
    return _add_track_status(fig, track_status, timeline.to_lap, timeline.end_time, **kwargs)
