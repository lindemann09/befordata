import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._data import BeForData


def detect_sessions(data: BeForData, time_column: str, time_gap: float) -> BeForData:
    """detects sessions based on time gaps in the time column"""
    sessions = [0]
    breaks = np.flatnonzero(np.diff(data.dat[time_column]) >= time_gap) + 1
    sessions.extend(breaks.tolist())
    return BeForData(data.dat, sampling_rate=data.sampling_rate,
                     columns=data.columns, sessions=sessions,
                     meta=data.meta)

def find_times(timeline: ArrayLike, needles: ArrayLike) -> NDArray[np.int_]:
    """returns index (i) of the closes time. If not found, it return next larger
    element.

    ``time_stamps[i-1] <= t < time_stamps[i]``

    Parameter
    ---------
    timeline : ArrayLike
        the sorted array of time stamps

    needles : number or ArrayLike
        the time(s) to be found in the timeline

    """

    return np.searchsorted(timeline, np.atleast_1d(needles), 'right')
