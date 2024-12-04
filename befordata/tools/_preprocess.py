from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from scipy import signal

from .._record import BeForRecord


def detect_sessions(
    data: BeForRecord, time_gap: float, time_column: str | None = None
) -> BeForRecord:
    """Detect recording sessions in the BeForRecord based on time gaps

    Parameters
    ----------
    data : BeForRecord
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
        time_column = data.time_column
    sessions = [0]
    breaks = np.flatnonzero(np.diff(data.dat[time_column]) >= time_gap) + 1
    sessions.extend(breaks.tolist())
    return BeForRecord(
        data.dat,
        sampling_rate=data.sampling_rate,
        columns=data.columns,
        sessions=sessions,
        time_column=time_column,
        meta=data.meta,
    )


def _butter_lowpass_filter(
    data: pd.Series, order: int, cutoff: float, sampling_rate: float, btype: str
):
    b, a = signal.butter(order, cutoff, fs=sampling_rate, btype=btype, analog=False)
    # filter shifted data (first sample = 0)
    y = signal.filtfilt(b, a, data - data.iat[0]) + data.iat[0]
    return y


def butter_filter(
    d: BeForRecord,
    order: int,
    cutoff: float,
    btype: str = "lowpass",
    columns: str | List[str] | None = None,
) -> BeForRecord:
    """Lowpass Butterworth filter of BeforRecord

    temporarily shifted data (first sample = 0) will be used for the filtering

    Parameters
    ----------
    d : BeForRecord
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
        columns = d.columns
    elif not isinstance(columns, List):
        columns = [columns]

    df = d.dat.copy()
    for s in range(d.n_sessions()):
        f, t = d.session_samples(s)
        for c in columns:  # type: ignore
            df.loc[f:t, c] = _butter_lowpass_filter(
                data=df.loc[f:t, c],
                cutoff=cutoff,
                sampling_rate=d.sampling_rate,
                order=order,
                btype=btype,
            )
    meta = deepcopy(d.meta)
    meta["cutoff_freq"] = cutoff
    meta["butterworth_order"] = order
    return BeForRecord(
        df,
        sampling_rate=d.sampling_rate,
        columns=d.columns,
        sessions=d.sessions,
        time_column=d.time_column,
        meta=meta,
    )
