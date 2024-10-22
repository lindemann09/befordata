import numpy as np

from ._data import BeForData


def detect_sessions(data: BeForData, time_column: str, time_gap: float) -> BeForData:
    """detects sessions based on time gaps in the time column"""
    sessions = [0]
    breaks = np.flatnonzero(np.diff(data.dat[time_column]) >= time_gap) + 1
    sessions.extend(breaks.tolist())
    return BeForData(data.dat, sampling_rate=data.sampling_rate,
                     columns=data.columns, sessions=sessions,
                     meta=data.meta)
