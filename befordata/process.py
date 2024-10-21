import numpy as np

from .force_data import BeForData

def detect_sessions(data:BeForData, time_column:str, time_gap:float) -> BeForData:
    """detects sessions based on time gaps in the time column"""
    new_sessions = [0]
    breaks = np.flatnonzero(np.diff(data.dat[time_column]) >= time_gap) + 1
    new_sessions.extend(breaks.tolist())
    return BeForData(data.dat, sampling_rate=data.sampling_rate,
              columns=data.columns, new_sessions=new_sessions,
              meta=data.meta)

def lowpass_filter():
    raise NotImplementedError() ## TODO