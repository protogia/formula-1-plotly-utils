from __future__ import annotations

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from values.colors import track_status_colors
def _get_track_status_changes(
    laps: pd.DataFrame,
    track_status: pd.DataFrame
) -> DataFrameGroupBy:
    """
    Filters state-changes of the track and assigns them to laps.
    
    Params:
    -----------
    laps : pd.DataFrame
        DataFrame from FastF1 (session.laps).
    track_status : pd.DataFrame
        Track-Status-DataFrame from FastF1 (session.track_status).
        
    Returns:
    --------
    DataFrameGroupBy
        pd.DataFrame sorted by lap.
    """
    filtered_track_status_changes = track_status[
        track_status['Message'].isin(track_status_colors.keys())
    ].copy()

    # add lap-column by finding the lap number closest to event time
    filtered_track_status_changes['Lap'] = filtered_track_status_changes['Time'].apply(
        lambda event_time: laps.loc[laps['Time'] <= event_time, 'LapNumber'].max() 
        if not laps.loc[laps['Time'] <= event_time].empty else None
    )
    filtered_track_status_changes.dropna(subset=['Lap'], inplace=True)
    filtered_track_status_changes['Lap'] = filtered_track_status_changes['Lap'].astype(int)

    # group to handle multiple events per lap
    return filtered_track_status_changes.groupby('Lap')
