"""
Collection of useful functions to work with BeForData structures.
"""

import typing as _tp
import warnings as _warnings

import numpy as _np
import pandas as _pd
from numpy.typing import NDArray as _NDArray

from ._epochs import BeForEpochs
from ._record import BeForRecord


### BeForRecord ###
def scale_record(rec: BeForRecord, factor: float) -> None:
    """
    Scales the force data (inplace) of the BeForRecord by a specified factor.

    This function creates a deep copy of the input BeForRecord, multiplies the columns
    corresponding to force measurements by the given scaling factor, and returns a new
    BeForRecord with the scaled data. All other attributes (sampling rate, sessions,
    time column, and metadata) are preserved via deep copy to avoid side effects.

    Parameters
    ----------
    rec : BeForRecord
    factor : float
        The scaling factor to apply to the force data columns.

    """

    rec.dat.iloc[:, rec.force_cols] *= factor


def concat_records(
    record_list: _tp.List[BeForRecord] | _tp.Tuple[BeForRecord, ...],
    no_sessions: bool = False,
) -> BeForRecord:
    """
    Concatenate a list or tuple of `BeForRecord` instances into a single
    `BeForRecord`.

    Returns a `BeForRecord` instance containing the concatenated records from
        all input objects.

    Parameters
    ----------
    record_list : list or tuple of BeForRecord
        A list or tuple containing BeForRecord objects to concatenate. All
        objects must have matching data columns, time_column, sampling rate.
    no_sessions : bool, optional
        If True, session information will be ignored during concatenation.
        Default is False.


    Notes
    -----
    Metadata will not be copied and only the metadata of the first
    BeForRecord will be used.

    Raises
    ------
    ValueError
        If the input list or tuple is empty.
    TypeError
        If any element in the input is not a BeForRecord instance.

    """
    if len(record_list) == 0:
        raise ValueError("No records to concatenate.")

    rtn = record_list[0]
    for ep in record_list[1:]:
        if not isinstance(ep, BeForRecord):
            raise TypeError("All items in the list must be BeForRecord instances.")
        rtn = _concat_records(rtn, ep, no_sessions)

    return rtn


def _concat_records(
    epochs_a: BeForRecord, epochs_b: BeForRecord, no_sessions: bool
) -> BeForRecord:
    """
    Helper function to concatenate two BeForRecord instances.

    Ensures compatibility of the two objects before concatenation.

    Notes
    -----
    Meta data will not be copied and only the meta data of the first
    BeForRecord will be used in the resulting BeForRecord.

    """

    if epochs_b.sampling_rate != epochs_a.sampling_rate:
        raise ValueError("Sampling rates are not the same.")
    if epochs_b.time_column != epochs_a.time_column:
        raise ValueError("Time columns are not the same.")
    if not _np.array_equal(  # Column names are not the same?
        _np.sort(epochs_b.dat.columns), _np.sort(epochs_a.dat.columns)
    ):
        raise ValueError("Data column names are not the same.")

    if no_sessions:
        sessions = []
    else:
        new_session_ids = [x + epochs_a.n_samples() for x in epochs_b.sessions]
        sessions = epochs_a.sessions + new_session_ids

    return BeForRecord(
        dat=_pd.concat((epochs_a.dat, epochs_b.dat), ignore_index=True),
        sampling_rate=epochs_a.sampling_rate,
        time_column=epochs_a.time_column,
        sessions=sessions,
        meta=epochs_a.meta,
    )


def detect_sessions(rec: BeForRecord, time_gap: float) -> None:
    """
    Detects recording sessions in a BeForRecord based on time gaps. The function
    modifies the `sessions` attribute of the BeForRecord in place.

    This function analyses the time column of the provided BeForRecord and identifies
    breaks in the recording where the time difference between consecutive samples
    exceeds the specified time_gap. Each detected break marks the start of a new
    session.

    Parameters
    ----------
    rec : BeForRecord
    time_gap : float
        The minimum time difference (in the same units as the time column) that is
        considered a pause in the recording and thus the start of a new session.


    """

    if len(rec.time_column) == 0:
        _warnings.warn("No time column defined!", RuntimeWarning)
        return

    sessions = [0]
    breaks = _np.flatnonzero(_np.diff(rec.dat[rec.time_column]) >= time_gap) + 1
    sessions.extend(breaks.tolist())
    rec.sessions = sessions


def split_sessions(rec: BeForRecord) -> _tp.List[BeForRecord]:
    """Split the record into a list of `BeForRecord` objects, one per session.

    Returns a list of `BeForRecord` objects, each containing the data for one session.

    Parameters
    ----------
    rec : BeForRecord

    """
    rtn = []
    for idx in rec.session_ranges():
        dat = BeForRecord(
            dat=rec.dat.iloc[idx.start : idx.stop, :],
            sampling_rate=rec.sampling_rate,
            time_column=rec.time_column,
            meta=rec.meta,
        )
        rtn.append(dat)
    return rtn


def extract_epochs(
    rec: BeForRecord,
    column: str | int,
    n_samples: int,
    n_samples_before: int,
    zero_samples: _tp.List[int] | _NDArray[_np.int_] | None = None,
    zero_times: _tp.List[float] | _NDArray[_np.floating] | None = None,
    design: _pd.DataFrame = _pd.DataFrame(),
    suppress_warnings: bool = False,
) -> BeForEpochs:
    """
    Extracts epochs centred around specified zero samples or zero times, with a given
    number of samples before and after each zero point.

    Returns an `BeForEpochs` object containing the extracted epochs.

    Parameters
    ----------
    rec : BeForRecord
    column : str
        Name of the column containing the force data to extract.
    n_samples : int
        Number of samples to extract after the zero sample.
    n_samples_before : int
        Number of samples to extract before the zero sample.
    zero_samples : list of int or np.ndarray, optional
        List of sample indices to center epochs on.
    zero_times : list of float or np.ndarray, optional
        List of time stamps to center epochs on.
    design : pd.DataFrame, optional
        Optional design matrix or metadata for the epochs.
    suppress_warnings : bool, optional (default: False)
        If True, suppress incomplete epoch warnings during epoch extraction.

    Raises
    ------
    ValueError
        If neither or both of `zero_samples` and `zero_times` are provided.

    Notes
    -----
    Provide either `zero_samples` or `zero_times`, not both.
    Use `find_samples_by_time` to convert times to sample indices if needed.

    Examples
    --------
    >>> ep = extract_epochs(my_record, "Fx",
    ...          n_samples=5000,
    ...          n_samples_before=100,
    ...          design=my_design,
    ...          zero_times=my_design.trial_time)

    """

    if zero_samples is None and zero_times is None:
        raise ValueError(
            "Define either the samples or times where to extract the epochs "
            "(i.e. parameter zero_samples or zero_time)"
        )

    elif zero_samples is not None and zero_times is not None:
        raise ValueError(
            "Define only one the samples or times where to extract the epochs, "
            "not both."
        )

    elif zero_times is not None:
        return extract_epochs(
            rec,
            column=column,
            n_samples=n_samples,
            n_samples_before=n_samples_before,
            zero_samples=rec.find_samples_by_time(zero_times),
            design=design,
            suppress_warnings=suppress_warnings,
        )

    assert zero_samples is not None  # always!

    if isinstance(column, int):
        fd = rec.dat.iloc[:, column]
    else:
        fd = rec.dat.loc[:, column]

    n = len(fd)  # samples for data
    n_epochs = len(zero_samples)
    n_col = n_samples_before + n_samples
    force_mtx = _np.empty((n_epochs, n_col), dtype=_np.float64)
    for r, zs in enumerate(zero_samples):
        f = zs - n_samples_before
        if f > 0 and f < n:
            t = zs + n_samples
            if t > n:
                if not suppress_warnings:
                    _warnings.warn(
                        f"extract_force_epochs: force epoch {r} is incomplete, "
                        f"{t - n} samples missing.",
                        RuntimeWarning,
                    )
                tmp = fd[f:].to_numpy(copy=True)  # make copy
                force_mtx[r, : len(tmp)] = tmp
                force_mtx[r, len(tmp) :] = 0
            else:
                force_mtx[r, :] = fd[f:t]

    return BeForEpochs(
        force_mtx,
        sampling_rate=rec.sampling_rate,
        design=design,
        zero_sample=n_samples_before,
    )


### BeForEpochs ###


def scale_epochs(epochs: BeForEpochs, factor: float) -> None:
    """
    Scales the force data (inplace) of the `BeForEpochs` structure by a specified
    factor.

    This function multiplies all force data in the `BeForEpochs` instance by the
    given scaling factor. The baseline is also scaled accordingly. All other
    attributes (sampling rate, design, zero_sample) are preserved.

    Parameters
    ----------
    epochs : BeForEpochs
    factor : float
        The scaling factor to apply to the force data and baseline.

    """
    epochs.dat *= factor
    epochs.baseline *= factor


def concat_epochs(
    epochs_list: _tp.List[BeForEpochs] | _tp.Tuple[BeForEpochs, ...],
) -> BeForEpochs:
    """
    Concatenate a list or tuple of BeForEpochs instances into a single
    `BeForEpochs` object.

    Parameters
    ----------
    epochs_list : list or tuple of BeForEpochs
        A list or tuple containing BeForEpochs objects to concatenate. All
        objects must have matching sample count, sampling rate, zero sample,
        baseline adjustment status, and design columns.

    Raises
    ------
    ValueError
        If the input list or tuple is empty.
    TypeError
        If any element in the input is not a BeForEpochs instance.

    """
    if len(epochs_list) == 0:
        raise ValueError("No epochs to concatenate.")

    rtn = epochs_list[0]
    for ep in epochs_list[1:]:
        if not isinstance(ep, BeForEpochs):
            raise TypeError("All items in the list must be BeForEpochs instances.")
        rtn = _concat_epochs(rtn, ep)

    return rtn


def _concat_epochs(epochs_a: BeForEpochs, epochs_b: BeForEpochs) -> BeForEpochs:
    """
    Helper function to concatenate two BeForEpochs instances.

    Ensures compatibility of the two objects before concatenation.
    """

    if epochs_b.n_samples() != epochs_a.n_samples():
        raise ValueError("Number of samples per epoch are not the same")
    if epochs_b.sampling_rate != epochs_a.sampling_rate:
        raise ValueError("Sampling rates are not the same.")
    if epochs_b.zero_sample != epochs_a.zero_sample:
        raise ValueError("Zero samples are not the same.")
    if epochs_b.is_baseline_adjusted() != epochs_a.is_baseline_adjusted():
        raise ValueError("One data structure is baseline adjusted, the other not.")

    if not _np.array_equal(  # Column names are not the same?
        _np.sort(epochs_b.design.columns), _np.sort(epochs_a.design.columns)
    ):
        raise ValueError("Design column names are not the same.")

    return BeForEpochs(
        dat=_np.concatenate((epochs_a.dat, epochs_b.dat), axis=0),
        sampling_rate=epochs_a.sampling_rate,
        design=_pd.concat([epochs_a.design, epochs_b.design], ignore_index=True),
        baseline=_np.append(epochs_a.baseline, epochs_b.baseline),
        zero_sample=epochs_a.zero_sample,
    )


def adjust_baseline(epochs: BeForEpochs, reference_window: _tp.Tuple[int, int]) -> None:
    """
    Adjust the baseline of each epoch (inplace) of the BeForEpochs data using the
    mean value of a defined sample window.

    Parameters
    ----------
    reference_window : Tuple[int, int]
        Tuple specifying the sample range (start, stop) used for baseline adjustment.
        Samples from "start" to "stop-1" will be included in the adjustment.
    """

    if epochs.is_baseline_adjusted():
        dat = epochs.dat + _np.atleast_2d(epochs.baseline).T  # restore baseline
    else:
        dat = epochs.dat
    i = range(reference_window[0], reference_window[1])
    epochs.baseline = _np.mean(dat[:, i], axis=1)
    epochs.dat = dat - _np.atleast_2d(epochs.baseline).T
    epochs.dat = dat - _np.atleast_2d(epochs.baseline).T
    epochs.dat = dat - _np.atleast_2d(epochs.baseline).T


def subset_epochs(
    epochs: BeForEpochs,
    idx: _tp.List[int]
    | _tp.List[bool]
    | _NDArray[_np.bool]
    | _NDArray[_np.int_]
    | _pd.Series,
) -> BeForEpochs:
    """
    Extract a subset of epochs from a BeForEpochs instance.

    Parameters
    ----------
    epochs : BeForEpochs
        The original BeForEpochs instance to extract the subset from.
    idx : list of int or list of bool or np.ndarray or pd.Series
        The indices of the epochs to extract.

    Returns
    -------
    BeForEpochs
        A new BeForEpochs instance containing the extracted subset.

    """
    if len(epochs.baseline) > 0:
        bsl = epochs.baseline[idx]
    else:
        bsl = _np.array([])
    if isinstance(idx, _pd.Series):
        idx = idx.to_numpy()

    return BeForEpochs(
        dat=epochs.dat[idx, :],
        sampling_rate=epochs.sampling_rate,
        design=epochs.design.iloc[idx, :],
        baseline=bsl,
        zero_sample=epochs.zero_sample,
    )
