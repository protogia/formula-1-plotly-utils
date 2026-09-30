from __future__ import annotations
from typing import Iterable, Mapping, Optional

import plotly.graph_objects as go
import plotly.io as pio


# Light theme based on plotly's default look.
BACKGROUND = "#FFFFFF"
PLOT_BACKGROUND = "#E5ECF6"
GRID = "#FFFFFF"
TEXT = "#2A3F5F"
MUTED = "#6B7A90"

TEMPLATE_NAME = "formula_1_plotly_utils"

_template = go.layout.Template(pio.templates["plotly"])
_template.layout.update(
    paper_bgcolor=BACKGROUND,
    plot_bgcolor=PLOT_BACKGROUND,
    font=dict(color=TEXT, size=13),
    title=dict(x=0.02, xanchor="left", font=dict(size=20)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    margin=dict(l=70, r=30, t=90, b=60),
    height=600,
)
_axis_style = dict(gridcolor=GRID, zerolinecolor=GRID, title=dict(font=dict(color=MUTED)))
_template.layout.xaxis.update(_axis_style)
_template.layout.yaxis.update(_axis_style)
pio.templates[TEMPLATE_NAME] = _template


def _axis_title(label: str, unit: Optional[str] = None) -> str:
    """Formats an axis title consistently as ``Label [unit]``."""
    return f"{label} [{unit}]" if unit else label


def _rgba(hex_color: str, alpha: float) -> str:
    """'#RRGGBB' -> 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _outline(hex_color: str) -> str:
    """Outline color for marks: very light colors (e.g. hard tyre) get a muted outline."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return MUTED if luminance > 0.85 else hex_color


def _track_outline(x, y) -> go.Scatter:
    """Grey track line drawn below colored track maps, so light values stay visible."""
    return go.Scatter(
        x=x, y=y, mode="lines", line=dict(color="#B8C2D0", width=9),
        hoverinfo="skip", showlegend=False,
    )


def _session_subtitle(*data) -> Optional[str]:
    """Builds ``<Event> <Year> · <Session>`` from the first object carrying a fastf1 session."""
    for d in data:
        session = getattr(d, "session", None)
        if session is None:
            continue
        try:
            return f"{session.event['EventName']} {session.event.year} · {session.name}"
        except Exception:
            continue
    return None


def _apply_layout(
    fig: go.Figure,
    *,
    title: str,
    subtitle: Optional[str] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    legend_title: Optional[str] = None,
    height: Optional[int] = None,
    showlegend: Optional[bool] = None,
) -> go.Figure:
    """Applies the shared theme, title and axis titles to a figure."""
    title_opts = dict(text=title)
    if subtitle:
        title_opts["subtitle"] = dict(text=subtitle, font=dict(color=MUTED, size=13))

    fig.update_layout(template=TEMPLATE_NAME, title=title_opts)
    if x_title is not None:
        fig.update_layout(xaxis_title=x_title)
    if y_title is not None:
        fig.update_layout(yaxis_title=y_title)
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    if height is not None:
        fig.update_layout(height=height)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


def _style_driver_axis(fig: go.Figure, drivers: Iterable[str], styles: Mapping, axis: str = "x") -> go.Figure:
    """
    Orders a categorical driver axis and labels each driver in black with a
    border in the driver color (colored text alone is hard to read for light colors).

    The regular tick labels stay as invisible placeholders so plotly reserves their space.
    """
    drivers = list(drivers)
    axis_opts = dict(
        categoryorder="array", categoryarray=drivers,
        tickmode="array", tickvals=drivers, ticktext=[f"<b>{d}</b>" for d in drivers],
        tickfont=dict(color="rgba(0,0,0,0)", size=13), ticks="",
    )
    if axis == "x":
        fig.update_xaxes(**axis_opts)
    else:
        fig.update_yaxes(**axis_opts)

    for d in drivers:
        if d not in styles:
            continue
        position = (
            dict(x=d, xref="x", y=0, yref="y domain", yanchor="top", yshift=-5)
            if axis == "x" else
            dict(y=d, yref="y", x=0, xref="x domain", xanchor="right", xshift=-5)
        )
        fig.add_annotation(
            **position, text=f"<b>{d}</b>", showarrow=False,
            font=dict(color="#000000", size=12), bgcolor=BACKGROUND,
            bordercolor=styles[d].color, borderwidth=2, borderpad=2,
        )
    return fig
