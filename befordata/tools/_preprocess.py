import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import signal

from .._record import BeForRecord

pd.set_option("mode.copy_on_write", True)


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
        warnings.warn("No time column defined!", RuntimeWarning)
        return rec
    else:
        time_column = rec.time_column
    sessions = [0]
    breaks = np.flatnonzero(np.diff(rec.dat[time_column]) >= time_gap) + 1
    sessions.extend(breaks.tolist())
    return BeForRecord(
        rec.dat,
        sampling_rate=rec.sampling_rate,
        sessions=sessions,
        time_column=time_column,
        meta=rec.meta,
    )

def __butter_lowpass_filter(
    rec: pd.Series, order: int, cutoff: float, sampling_rate: float, center_data: bool
):
    b, a = signal.butter(order, cutoff, fs=sampling_rate, btype='lowpass', analog=False)  # type: ignore
    if center_data:
        # filter centred data (first sample = 0)
        return signal.filtfilt(b, a, rec - rec.iat[0]) + rec.iat[0]
    else:
        return signal.filtfilt(b, a, rec)


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
        for c in rec.force_cols():
            df.iloc[idx, c] = __butter_lowpass_filter(rec=df.iloc[idx, c], # type: ignore
                cutoff=cutoff,
                sampling_rate=rec.sampling_rate,
                order=order,
                center_data=center_data,
            )
    meta = deepcopy(rec.meta)
    meta["filter"] = f"butterworth: cutoff={cutoff}, order={order}"
    return BeForRecord(
        df,
        sampling_rate=rec.sampling_rate,
        sessions=rec.sessions,
        time_column=rec.time_column,
        meta=meta,
    )
