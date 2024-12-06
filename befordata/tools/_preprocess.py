from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from scipy import signal

from .._record import BeForRecord


def detect_sessions(
    rec: BeForRecord, time_gap: float, time_column: str | None = None
) -> BeForRecord:
    """Detect recording sessions in the BeForRecord based on time gaps

    Parameters
    ----------
    rec : BeForRecord
        the data
    time_gap : float
        smallest time gap that should be considered as pause of the recording
        and the start of a new session
    time_column : str | None, optional
        name of column that represents the time

    Returns
    -------
    BeForRecord
    """

    if time_column is None:
        time_column = rec.time_column
    sessions = [0]
    breaks = np.flatnonzero(np.diff(rec.dat[time_column]) >= time_gap) + 1
    sessions.extend(breaks.tolist())
    return BeForRecord(
        rec.dat,
        sampling_rate=rec.sampling_rate,
        columns=rec.columns,
        sessions=sessions,
        time_column=time_column,
        meta=rec.meta,
    )


def _butter_lowpass_filter(
    rec: pd.Series, order: int, cutoff: float, sampling_rate: float, btype: str
):
    b, a = signal.butter(order, cutoff, fs=sampling_rate, btype=btype, analog=False)
    # filter shifted data (first sample = 0)
    y = signal.filtfilt(b, a, rec - rec.iat[0]) + rec.iat[0]
    return y


def butter_filter(
    rec: BeForRecord,
    order: int,
    cutoff: float,
    btype: str = "lowpass",
    columns: str | List[str] | None = None,
) -> BeForRecord:
    """Lowpass Butterworth filter of BeforRecord

    temporarily shifted data (first sample = 0) will be used for the filtering

    Parameters
    ----------
    rec : BeForRecord
        the data
    order : int
        order of the filter.
    cutoff : float
        cutoff frequency
    btype : {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
        type of filter, by default "lowpass"
    columns : str | List[str] | None, optional
        column(s) of data the should be filtered. By default None; in this case,
        all data columns will be filtered

    Returns
    -------
    BeForRecord with filtered data

    Notes
    -----
    see documentation of `scipy.signal.butter` for information about the filtering

    """

    if columns is None:
        columns = rec.columns
    elif not isinstance(columns, List):
        columns = [columns]

    df = rec.dat.copy()
    for s in range(rec.n_sessions()):
        f, t = rec.session_samples(s)
        for c in columns:  # type: ignore
            df.loc[f:t, c] = _butter_lowpass_filter(
                rec=df.loc[f:t, c],
                cutoff=cutoff,
                sampling_rate=rec.sampling_rate,
                order=order,
                btype=btype,
            )
    meta = deepcopy(rec.meta)
    meta["cutoff_freq"] = cutoff
    meta["butterworth_order"] = order
    return BeForRecord(
        df,
        sampling_rate=rec.sampling_rate,
        columns=rec.columns,
        sessions=rec.sessions,
        time_column=rec.time_column,
        meta=meta,
    )
