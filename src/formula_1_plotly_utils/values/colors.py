from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd
import fastf1
import fastf1.plotting


_FALLBACK_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#8cd17d",
]

# matplotlib linestyles (as returned by fastf1) -> plotly dash styles
_LINESTYLE_TO_DASH = {
    "solid": "solid", "-": "solid",
    "dashed": "dash", "--": "dash",
    "dotted": "dot", ":": "dot",
    "dashdot": "dashdot", "-.": "dashdot",
}


@dataclass(frozen=True)
class DriverStyle:
    color: str
    dash: str = "solid"


def _driver_column(data: pd.DataFrame) -> str:
    """Returns the column holding the driver abbreviation (laps: 'Driver', results: 'Abbreviation')."""
    for col in ("Driver", "Abbreviation"):
        if col in data.columns:
            return col
    raise KeyError("Data needs a 'Driver' or 'Abbreviation' column to assign driver colors.")


def _normalize_hex(color) -> Optional[str]:
    if not isinstance(color, str) or not color or color.lower() == "nan":
        return None
    return color if color.startswith("#") else f"#{color}"


def _driver_styles(
    data: pd.DataFrame,
    drivers: Optional[Iterable[str]] = None,
    session: Optional["fastf1.core.Session"] = None,
    colormap: str = "default",
) -> Dict[str, DriverStyle]:
    """
    Resolves color and line style per driver.

    Priority: 'DriverColor' column (see apply_driver_colors) > fastf1 driver style
    (requires a session, taken from ``data.session`` if not given) > 'TeamColor' column
    > fallback palette. Teammates sharing a color get a dashed line.
    """
    col = _driver_column(data)
    if drivers is None:
        drivers = data[col].dropna().unique().tolist()
    drivers = [str(d) for d in drivers]
    session = session if session is not None else getattr(data, "session", None)

    first_rows = data.dropna(subset=[col]).drop_duplicates(col).set_index(col)
    team_col = next((c for c in ("Team", "TeamName") if c in first_rows.columns), None)

    colors: Dict[str, str] = {}
    dashes: Dict[str, Optional[str]] = {}
    for i, drv in enumerate(drivers):
        row = first_rows.loc[drv] if drv in first_rows.index else None
        color, dash = None, None

        if row is not None and "DriverColor" in first_rows.columns:
            color = _normalize_hex(row["DriverColor"])
            if "DriverLineStyle" in first_rows.columns and isinstance(row["DriverLineStyle"], str):
                dash = row["DriverLineStyle"]

        if color is None and session is not None:
            try:
                style = fastf1.plotting.get_driver_style(drv, ["color", "linestyle"], session, colormap=colormap)
                color = style["color"]
                dash = _LINESTYLE_TO_DASH.get(style["linestyle"], "solid")
            except Exception:
                pass

        if color is None and row is not None and "TeamColor" in first_rows.columns:
            color = _normalize_hex(row["TeamColor"])

        if color is None:
            team = row[team_col] if (row is not None and team_col) else drv
            known = list(dict.fromkeys(first_rows[team_col])) if team_col else drivers
            idx = known.index(team) if team in known else i
            color = _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)]

        colors[drv], dashes[drv] = color, dash

    # teammates without explicit line style: first solid, second dashed
    seen_colors = set()
    styles = {}
    for drv in drivers:
        dash = dashes[drv]
        if dash is None:
            dash = "dash" if colors[drv].lower() in seen_colors else "solid"
        seen_colors.add(colors[drv].lower())
        styles[drv] = DriverStyle(color=colors[drv], dash=dash)
    return styles


def _colors_similar(color_a: str, color_b: str, threshold: float = 200.0) -> bool:
    """Whether two hex colors are hard to tell apart (weighted RGB 'redmean' distance)."""
    (r1, g1, b1), (r2, g2, b2) = (
        tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for c in (color_a, color_b)
    )
    rmean = (r1 + r2) / 2
    distance = (
        (2 + rmean / 256) * (r1 - r2) ** 2 + 4 * (g1 - g2) ** 2 + (2 + (255 - rmean) / 256) * (b1 - b2) ** 2
    ) ** 0.5
    return distance < threshold


def apply_driver_colors(
    data: pd.DataFrame,
    session: Optional["fastf1.core.Session"] = None,
    colormap: str = "default",
) -> pd.DataFrame:
    """
    Applies driver colors based on fastf1-driver-colors.

    Should be used directly after loading a session. All plot functions prefer the
    added columns, so colors stay consistent across charts.

    Parameters:
    -----------
    data : pd.DataFrame (either session.laps or session.results)
    session : fastf1 session (optional, taken from ``data.session`` for laps). Needed
        for session.results to use fastf1's color map instead of the raw 'TeamColor'.
    colormap : 'default' | 'official'
        fastf1 color map; 'official' uses the (brighter) colors of the F1 timing app.

    Returns:
    --------
    data : pd.DataFrame (copy with 'DriverColor', 'DriverLineStyle' and 'DriverLabel')
    """
    out = data.copy()
    col = _driver_column(out)
    styles = _driver_styles(
        out.drop(columns=["DriverColor", "DriverLineStyle"], errors="ignore"),
        session=session if session is not None else getattr(data, "session", None),
        colormap=colormap,
    )

    out["DriverColor"] = out[col].map(lambda d: styles[d].color if d in styles else None)
    out["DriverLineStyle"] = out[col].map(lambda d: styles[d].dash if d in styles else None)
    out["DriverLabel"] = out[col].astype(str)
    if "TeamName" in out.columns:
        out["DriverLabel"] += " (" + out["TeamName"].astype(str) + ")"
    return out


# fastf1 compound colors (fastf1.plotting.get_compound_mapping)
compound_colors = {
    "SOFT": "#da291c",
    "MEDIUM": "#ffd12e",
    "HARD": "#f0f0ec",
    "INTERMEDIATE": "#43b02a",
    "WET": "#0067ad",
    "UNKNOWN": "#00ffff",
    "TEST-UNKNOWN": "#434649",
}


def _compound_color_map(data: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """Compound colors from fastf1 (session-specific if available)."""
    session = getattr(data, "session", None)
    if session is not None:
        try:
            return {k.upper(): v for k, v in fastf1.plotting.get_compound_mapping(session).items()}
        except Exception:
            pass
    return dict(compound_colors)


# Track status: one style used by every chart (see _core.annotations._add_track_status)
track_status_styles = {
    # color: band / lines, text: label (darker for readability on the light background)
    "SC": {"label": "SC", "name": "Safety Car", "color": "#F5C400", "text": "#9A7B00"},
    "VSC": {"label": "VSC", "name": "Virtual Safety Car", "color": "#FF8700", "text": "#C25E00"},
    "RED": {"label": "RED FLAG", "name": "Red Flag", "color": "#E10600", "text": "#B00500"},
    "YELLOW": {"label": "", "name": "Yellow Flag", "color": "#FFF200", "text": "#9A9200"},
}

# Kept for backwards compatibility: fastf1 track status message -> color
track_status_colors = {
    "AllClear": "#43B02A",
    "Yellow": track_status_styles["YELLOW"]["color"],
    "Red": track_status_styles["RED"]["color"],
    "SCDeployed": track_status_styles["SC"]["color"],
    "VSCDeployed": track_status_styles["VSC"]["color"],
    "VSCEnding": track_status_styles["VSC"]["color"],
}

# Color for chart markers (events passed via the ``markers`` parameter)
marker_color = "#2A3F5F"

# Categories which are not drivers or compounds (qualifying rounds, weather metrics, ...)
category_palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]

condition_colors = {"Dry": "#FFA15A", "Rain": "#0067AD"}
