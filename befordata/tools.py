"""
Collection of useful function to deal with BeForData structures

(c) O. Lindemann
"""

import warnings as _warnings
from copy import deepcopy as _deepcopy

import numpy as _np
import pandas as _pd
from scipy import signal as _signal

from ._record import BeForRecord


def detect_sessions(
    rec: BeForRecord, time_gap: float) -> BeForRecord:
    """Detect recording sessions in the BeForRecord based on time gaps

    Parameters
    ----------
    rec : BeForRecord
        the data
    time_gap : float
        smallest time gap that should be considered as pause of the recording
        and the start of a new session

    Returns
    -------
    BeForRecord
    """

    if len(rec.time_column) == 0:
        _warnings.warn("No time column defined!", RuntimeWarning)
        return rec
    else:
        time_column = rec.time_column
    sessions = [0]
    breaks = _np.flatnonzero(_np.diff(rec.dat[time_column]) >= time_gap) + 1
    sessions.extend(breaks.tolist())
    return BeForRecord(
        rec.dat,
        sampling_rate=rec.sampling_rate,
        sessions=sessions,
        time_column=time_column,
        meta=rec.meta,
    )

def __butter_lowpass_filter(
    rec: _pd.Series, order: int, cutoff: float, sampling_rate: float, center_data: bool
):
    b, a = _signal.butter(order, cutoff, fs=sampling_rate, btype='lowpass', analog=False)  # type: ignore
    if center_data:
        # filter centred data (first sample = 0)
        return _signal.filtfilt(b, a, rec - rec.iat[0]) + rec.iat[0]
    else:
        return _signal.filtfilt(b, a, rec)


def lowpass_filter(
    rec: BeForRecord,
    cutoff: float,
    order: int,
    center_data: bool = True
) -> BeForRecord:
    """Lowpass Butterworth filter of BeforRecord

    Parameters
    ----------
    rec : BeForRecord
        the data
    cutoff : float
        cutoff frequency
    order : int
        order of the filter.
    center_data : bool (default: True)
        temporarily centred data (first sample = 0) will be used for the filtering

    Returns
    -------
    BeForRecord with filtered data

    Notes
    -----
    see documentation of `scipy.signal.butter` for information about the filtering

    """

    df = rec.dat.copy()
    for idx in rec.session_ranges():
        for c in rec.force_cols:
            df.iloc[idx, c] = __butter_lowpass_filter(rec=df.iloc[idx, c], # type: ignore
                cutoff=cutoff,
                sampling_rate=rec.sampling_rate,
                order=order,
                center_data=center_data,
            )
    meta = _deepcopy(rec.meta)
    meta["filter"] = f"butterworth: cutoff={cutoff}, order={order}"
    return BeForRecord(
        df,
        sampling_rate=rec.sampling_rate,
        sessions=rec.sessions,
        time_column=rec.time_column,
        meta=meta,
    )

