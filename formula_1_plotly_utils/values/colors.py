from __future__ import annotations
from typing import List, Optional, Dict
import fastf1
import fastf1.plotting


from __future__ import annotations
from typing import List, Optional, Dict
import pandas as pd
import fastf1.plotting

def get_driver_colors(
    laps: pd.DataFrame, 
    drivers: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Returns driver colors based on fastf1-team-colors.
    
    Parameters:
    -----------
    laps : pd.DataFrame
        FastF1 laps DataFrame (session.laps).
    drivers : List[str], optional
        Optional driver abbreviations.
        
    Returns:
    --------
    Dict[str, str]
        dict of drivers and HEX-colors.
    """
    if drivers is None:
        drivers = laps['Driver'].unique().tolist() # all drivers

    color_map: Dict[str, str] = {}
    for d in drivers:
        try:
            driver_laps = laps[laps['Driver'] == d]
            if not driver_laps.empty:
                team_name = driver_laps['Team'].iloc
                color_map[d] = fastf1.plotting.get_team_color(team_name)
            else:
                color_map[d] = '#808080'  # fallback
        except:
            color_map[d] = '#808080'  
    return color_map



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
