import typing as _tp
from copy import copy as _copy

import numpy as _np
import pandas as _pd
from numpy.typing import ArrayLike as _ArrayLike
from numpy.typing import NDArray as _NDArray
from scipy import signal as _signal

from ._data import BeForData


def detect_sessions(data: BeForData, time_column: str, time_gap: float) -> BeForData:
    """detects sessions based on time gaps in the time column"""
    sessions = [0]
    breaks = _np.flatnonzero(_np.diff(data.dat[time_column]) >= time_gap) + 1
    sessions.extend(breaks.tolist())
    return BeForData(data.dat, sampling_rate=data.sampling_rate,
                     columns=data.columns, sessions=sessions,
                     meta=data.meta)


def find_times(timeline: _ArrayLike, needles: _ArrayLike) -> _NDArray[_np.int_]:
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

    return _np.searchsorted(timeline, _np.atleast_1d(needles), 'right')

# filtering


def _butter_lowpass_filter(data: _pd.Series, cutoff: float, sampling_rate: float, order: int):
    b, a = _signal.butter(order, cutoff, fs=sampling_rate,
                          btype='lowpass', analog=False)
    # filter centered data
    y = _signal.filtfilt(b, a, data - data.iat[0]) + data.iat[0]
    return y


def lowpass_filter(d: BeForData,
                   cutoff_freq: float,
                   butterworth_order: int,
                   columns: _tp.Union[None, str, _tp.List[str]] = None):
    """Lowpass Butterworth filter of BeforData"""

    if columns is None:
        columns = d.columns
    elif not isinstance(columns, _tp.List):
        columns = [columns]

    df = d.dat.copy()
    for s in range(d.n_sessions):
        f, t = d.session_rows(s)
        for c in columns:  # type: ignore
            df.loc[f:t, c] = _butter_lowpass_filter(
                data=df.loc[f:t, c],
                cutoff=cutoff_freq,
                sampling_rate=d.sampling_rate,
                order=butterworth_order)
    meta = _copy(d.meta)
    meta["cutoff_freq"] = cutoff_freq
    meta["butterworth_order"] = butterworth_order
    return BeForData(df, d.sampling_rate, d.columns, d.sessions, meta)
