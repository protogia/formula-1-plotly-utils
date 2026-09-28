from __future__ import annotations
from typing import List, Optional, Dict

import pandas as pd
import fastf1
import fastf1.plotting


import fastf1
import pandas as pd

_FALLBACK_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#8cd17d",
]


def apply_driver_colors(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies driver colors based on fastf1-team-colors.

    Parameters:
    -----------
    data : pd.DataFrame (either session.laps or session.results)

    Returns:
    --------
    data : pd.DataFrame (enhanced with driver color data)
    """
    is_laps = 'Team' in data.columns and 'DriverCode' not in data.columns

    # per-driver color map (shared logic)
    def _map_from_teams(key_col, value_col, palette_index):
        out = {}
        for i, v in enumerate(pd.Series(value_col).unique()):
            if pd.isna(v):
                out[v] = _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)]
                continue
            try:
                c = fastf1.plotting.get_team_color(str(v))
            except Exception:
                c = None
            out[v] = c if (c and c != '#808080') else _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)]
        return out

    if is_laps:
        teams = data.groupby('Driver')['Team'].first().dropna()
        team_color_map = _map_from_teams('Team', teams, None)
        results = pd.DataFrame({
            'Driver': list(teams.index),
            'TeamColor': teams.values,
            'Abbreviation': teams.index,
        })
    else:
        results = data.copy()

    # colors
    if 'TeamColor' in results.columns:
        results['DriverColor'] = results['TeamColor'].apply(
            lambda c: f"#{c}" if isinstance(c, str) and not c.startswith('#')
                      else ('#cccccc' if pd.isna(c) else c)
        )
    elif 'Team' in results.columns:
        team_color_map = _map_from_teams('Team', results['Team'], None)
        results['DriverColor'] = results['Team'].map(team_color_map)
    elif 'Abbreviation' in results.columns:
        color_map = {}
        for i, d in enumerate(results['Abbreviation'].unique()):
            try:
                c = fastf1.plotting.get_team_color(
                    results.loc[results['Abbreviation'] == d, 'Team'].dropna().iloc[0]
                    if 'Team' in results.columns else None
                )
            except (ValueError, IndexError):
                c = None
            color_map[d] = c if (c and c != '#808080') else _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)]
        results['DriverColor'] = results['Abbreviation'].map(color_map).fillna('#cccccc')
    else:
        results['DriverColor'] = '#1f77b4'

    # labels
    code_col = 'Abbreviation' if 'Abbreviation' in results.columns else 'DriverCode'
    results['DriverLabel'] = results[code_col].astype(str)
    if 'TeamName' in results.columns:
        results['DriverLabel'] += ' (' + results['TeamName'].astype(str) + ')'

    return results



track_status_colors = {
    "AllClear": "green",
    "Yellow": "yellow",
    "Red": "red",
    "SCDeployed": "purple",
    "VSCDeployed": "violet",
    "VSCEnding": "orange",
}


compound_colors = {
    'SOFT': 'red',
    'MEDIUM': 'yellow',
    'HARD': 'white',
    'INTERMEDIATE': 'green',
    'WET': 'blue'
}
