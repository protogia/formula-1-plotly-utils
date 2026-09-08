import fastf1.plotting
from typing import Optional

def get_driver_colors(
        session: Optional[''], 
        drivers: List = None
):
    if drivers is None:
        drivers = session.laps['Driver'].unique() # all drivers

    return {d: fastf1.plotting.get_driver_color(d, session=session) for d in drivers}