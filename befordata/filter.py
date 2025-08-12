"""
Collection of functions for filtering BeForeRecord data.
"""


from copy import deepcopy as _deepcopy

import pandas as _pd
from scipy import signal as _signal

from ._record import BeForRecord


def __butter_lowpass_filter(
    rec: _pd.Series, order: int, cutoff: float, sampling_rate: float, center_data: bool
):
    b, a = _signal.butter(  # type: ignore
        order, cutoff, fs=sampling_rate, btype="lowpass", analog=False
    )
    if center_data:
        # filter centred data (first sample = 0)
        return _signal.filtfilt(b, a, rec - rec.iat[0]) + rec.iat[0]
    else:
        return _signal.filtfilt(b, a, rec)


def lowpass_filter(
    rec: BeForRecord, cutoff: float, order: int, center_data: bool = True
) -> BeForRecord:
    """
    Applies a lowpass Butterworth filter to the force data in a `BeForRecord`.

    This function filters each force data column in every session of the provided
    `BeForRecord` using a zero-phase Butterworth lowpass filter. Optionally, the
    data can be centred (subtracting the first sample) before filtering to reduce
    edge artifacts.

    Returns a `BeForRecord` instance with the filtered force data. No inplace
    modification is performed to preserve the original data.

    Parameters
    ----------
    rec : BeForRecord
    cutoff : float
        The cutoff frequency of the lowpass filter (in Hz).
    order : int
        The order of the Butterworth filter.
    center_data : bool, optional (default: True)
        If True, center the data by subtracting the first sample before filtering.


    Notes
    -----
    Filtering is performed using `scipy.signal.butter` and `scipy.signal.filtfilt`
    for zero-phase filtering. See the SciPy documentation for more details:
    https://docs.scipy.org/doc/scipy/reference/signal.html
    """

    df = rec.dat.copy()
    for idx in rec.session_ranges():
        for c in rec.force_cols:
            df.iloc[idx, c] = __butter_lowpass_filter(  # type: ignore
                rec=df.iloc[idx, c],  # type: ignore
                cutoff=cutoff,
                sampling_rate=rec.sampling_rate,
                order=order,
                center_data=center_data,
            )
    meta = _deepcopy(rec.meta)
    meta["filter"] = f"butterworth: cutoff={cutoff}, order={order}"
    return BeForRecord(
        dat=df,
        sampling_rate=rec.sampling_rate,
        sessions=rec.sessions,
        time_column=rec.time_column,
        meta=meta,
    )