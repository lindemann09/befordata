"""
Collection of useful functions to work with BeForData structures.

This module provides utilities for manipulating and analysing BeForData objects,
including scaling force data, detecting recording sessions based on time gaps,
and applying lowpass Butterworth filters to force signals.
"""

import warnings as _warnings
from copy import deepcopy as _deepcopy

import numpy as _np
import pandas as _pd
from scipy import signal as _signal

from ._record import BeForEpochs, BeForRecord


def scale_record(data: BeForRecord, factor: float) -> BeForRecord:
    """
    Scales the force data in a BeForRecord by a specified factor.

    This function creates a deep copy of the input BeForRecord, multiplies the columns
    corresponding to force measurements by the given scaling factor, and returns a new
    BeForRecord with the scaled data. All other attributes (sampling rate, sessions,
    time column, and metadata) are preserved via deep copy to avoid side effects.

    Parameters
    ----------
    data : BeForRecord
    factor : float
        The scaling factor to apply to the force data columns.

    Returns
    -------
    BeForRecord
        A new BeForRecord instance with the force data columns scaled by the given factor.
        All other attributes are copied from the input record.
    """

    df = data.dat.copy()
    df.iloc[:, data.force_cols] *= factor
    return BeForRecord(
        dat=df,
        sampling_rate=data.sampling_rate,
        sessions=_deepcopy(data.sessions),
        time_column=data.time_column,
        meta=_deepcopy(data.meta),
    )


def scale_epochs(data: BeForEpochs, factor: float) -> BeForEpochs:
    """
    Scales the force data in a BeForEpochs object by a specified factor.

    This function multiplies all force data in the BeForEpochs instance by the
    given scaling factor. The baseline is also scaled accordingly. All other
    attributes (sampling rate, design, zero_sample) are preserved.

    Parameters
    ----------
    data : BeForEpochs
        The BeForEpochs instance containing the force data to be scaled.
    factor : float
        The scaling factor to apply to the force data and baseline.

    Returns
    -------
    BeForEpochs
        A new BeForEpochs instance with the force data and baseline scaled by
        the given factor.
    """

    return BeForEpochs(
        dat=data.dat * factor,
        sampling_rate=data.sampling_rate,
        design=data.design.copy(),
        baseline=data.baseline * factor,
        zero_sample=data.zero_sample,
    )


def detect_sessions(rec: BeForRecord, time_gap: float) -> BeForRecord:
    """
    Detects recording sessions in a BeForRecord based on time gaps.

    This function analyses the time column of the provided BeForRecord and identifies
    breaks in the recording where the time difference between consecutive samples
    exceeds the specified time_gap. Each detected break marks the start of a new session.

    Parameters
    ----------
    rec : BeForRecord
        The BeForRecord instance containing the data to analyze.
    time_gap : float
        The minimum time difference (in the same units as the time column) that is
        considered a pause in the recording and thus the start of a new session.

    Returns
    -------
    BeForRecord
        A new BeForRecord instance with updated session indices reflecting detected sessions.

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
        dat=rec.dat,
        sampling_rate=rec.sampling_rate,
        sessions=sessions,
        time_column=time_column,
        meta=rec.meta,
    )


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
    Applies a lowpass Butterworth filter to the force data in a BeForRecord.

    This function filters each force data column in every session of the provided BeForRecord
    using a zero-phase Butterworth lowpass filter. Optionally, the data can be centred
    (subtracting the first sample) before filtering to reduce edge artifacts.

    Parameters
    ----------
    rec : BeForRecord
        The BeForRecord instance containing the data to be filtered.
    cutoff : float
        The cutoff frequency of the lowpass filter (in Hz).
    order : int
        The order of the Butterworth filter.
    center_data : bool, optional (default: True)
        If True, center the data by subtracting the first sample before filtering.

    Returns
    -------
    BeForRecord
        A new BeForRecord instance with the force data columns filtered.

    Notes
    -----
    Filtering is performed using `scipy.signal.butter` and `scipy.signal.filtfilt`
    for zero-phase filtering. See the SciPy documentation for more details.
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
