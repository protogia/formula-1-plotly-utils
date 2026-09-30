from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# fastf1 track status codes -> period kind ('1' = all clear ends a period)
_STATUS_KIND = {
    "2": "YELLOW",
    "4": "SC",
    "5": "RED",
    "6": "VSC",
}
_VSC_ENDING = "7"


@dataclass
class _TrackStatusPeriod:
    kind: str
    start: pd.Timedelta
    end: Optional[pd.Timedelta] = None      # None: lasts until the end of the data
    ending: Optional[pd.Timedelta] = None   # e.g. 'VSC ending' message


def _get_track_status_periods(
    track_status: pd.DataFrame,
    include_yellow: bool = False,
) -> List[_TrackStatusPeriod]:
    """
    Converts fastf1 track status changes into neutralisation periods.

    Params:
    -----------
    track_status : pd.DataFrame
        Track-Status-DataFrame from FastF1 (session.track_status).
    include_yellow : bool
        Yellow flags are frequent and short, so they are skipped by default.

    Returns:
    --------
    List of periods (SC, VSC, RED, YELLOW) in session time.
    """
    periods: List[_TrackStatusPeriod] = []
    current: Optional[_TrackStatusPeriod] = None

    for _, row in track_status.sort_values("Time").iterrows():
        status, time = str(row["Status"]), row["Time"]

        if status == _VSC_ENDING:
            if current is None or current.kind != "VSC":
                if current is not None:
                    current.end = time
                    periods.append(current)
                current = _TrackStatusPeriod("VSC", start=time)
            current.ending = time
            continue

        kind = _STATUS_KIND.get(status)
        if current is not None and kind != current.kind:
            current.end = time
            periods.append(current)
            current = None
        if kind is not None and current is None:
            current = _TrackStatusPeriod(kind, start=time)

    if current is not None:
        periods.append(current)

    if not include_yellow:
        periods = [p for p in periods if p.kind != "YELLOW"]
    return periods


def _to_seconds(value) -> float:
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, str):
        return pd.to_timedelta(value).total_seconds()
    if hasattr(value, "total_seconds"):
        return value.total_seconds()
    return float(value)


def _to_minutes(value) -> float:
    """Session time (timedelta / 'HH:MM:SS' string) -> minutes. Plain numbers are taken as minutes."""
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return _to_seconds(value) / 60.0


class _LapTimeline:
    """
    Maps session time <-> (fractional) race lap, based on the leader.

    A lap axis shows lap ``n`` at the end of lap n, so an event during lap n is
    placed between ``n - 1`` and ``n``.
    """

    def __init__(self, laps: pd.DataFrame):
        valid = laps.dropna(subset=["LapNumber"])
        starts = valid.dropna(subset=["LapStartTime"]).groupby("LapNumber")["LapStartTime"].min()
        ends = valid.dropna(subset=["Time"]).groupby("LapNumber")["Time"].min()

        points = pd.concat([
            pd.DataFrame({"t": starts.dt.total_seconds().values, "lap": starts.index.values - 1.0}),
            pd.DataFrame({"t": ends.dt.total_seconds().values, "lap": ends.index.values.astype(float)}),
        ]).sort_values(["t", "lap"])

        self._t = points["t"].to_numpy()
        self._lap = np.maximum.accumulate(points["lap"].to_numpy())

    @property
    def is_empty(self) -> bool:
        return len(self._t) == 0

    @property
    def end_time(self) -> pd.Timedelta:
        return pd.to_timedelta(float(self._t[-1]), unit="s")

    def to_lap(self, time) -> float:
        return float(np.interp(_to_seconds(time), self._t, self._lap))

    def to_time(self, lap: float) -> pd.Timedelta:
        return pd.to_timedelta(float(np.interp(lap, self._lap, self._t)), unit="s")
