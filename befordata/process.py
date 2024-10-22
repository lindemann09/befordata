from copy import copy
from typing import Union, List
import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd
from .force_data import BeForData


def detect_sessions(data: BeForData, time_column: str, time_gap: float) -> BeForData:
    """detects sessions based on time gaps in the time column"""
    sessions = [0]
    breaks = np.flatnonzero(np.diff(data.dat[time_column]) >= time_gap) + 1
    sessions.extend(breaks.tolist())
    return BeForData(data.dat, sampling_rate=data.sampling_rate,
                     columns=data.columns, sessions=sessions,
                     meta=data.meta)

def _butter_lowpass_filter(data:pd.Series, cutoff: float, sampling_rate: float, order: int):
    b, a = butter(order, cutoff, fs=sampling_rate,
                  btype='lowpass', analog=False)
    y = filtfilt(b, a, data - data.iat[0]) + data.iat[0] # filter centered data
    return y


def lowpass_filter(d: BeForData,
                   cutoff_freq: float,
                   butterworth_order: int,
                   columns: Union[None, str, List[str]] = None):
    """Lowpass Butterworth filter of BeforData"""

    if columns is None:
        columns = d.columns
    elif not isinstance(columns, List):
        columns = [columns]

    df = d.dat.copy()
    for s in range(d.n_sessions):
        f, t = d.session_rows(s)
        for c in columns: # type: ignore
            df.loc[f:t, c] = _butter_lowpass_filter(
                                    data=df.loc[f:t, c],
                                    cutoff=cutoff_freq,
                                    sampling_rate=d.sampling_rate,
                                    order=butterworth_order)
    meta = copy(d.meta)
    meta["cutoff_freq"]  = cutoff_freq
    meta["butterworth_order"]  = butterworth_order
    return BeForData(df, d.sampling_rate, d.columns, d.sessions, meta)
