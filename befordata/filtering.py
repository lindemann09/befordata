"""
Collection of functions for filtering BeForeRecord data.
"""

from copy import deepcopy as _deepcopy

from scipy import signal as _signal

from ._record import BeForRecord


def filtfilt(b, a, rec: BeForRecord, inplace: bool = False, **kwargs) -> BeForRecord:
    """
    Applies a digital filter forward and backward to each force data column in
    a BeForRecord.

    This function basically wraps `scipy.signal.filtfilt` to provide a zero-phase
    filtering for all force columns in every session of the provided BeForRecord.

    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    a : (N,) array_like
        The denominator coefficient vector of the filter.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    rec : BeForRecord
        The BeForRecord instance containing the data to filter.
    inplace : bool, optional (default: False)
        If True, perform filtering in place and modify the original data.
    **kwargs
        Additional keyword arguments passed to `scipy.signal.filtfilt`.


    Notes
    -----
    Documentation of `scipy.signal.filtfilt`:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html


    Examples
    --------
    >>> from scipy import signal
    >>> from befordata import BeForRecord, filtering
    >>> rec = BeForRecord(...)  # Create a BeForRecord instance
    >>> b, a = signal.butter(N=4, Wn=30, fs=1000, btype='low')
    >>> filtered = filtering.filtfilt(b, a, rec)

    """

    if not inplace:
        rec = _deepcopy(rec)

    rec.meta["filter"] = "filtered data"
    for idx in rec.session_ranges():
        for c in rec.force_cols:
            # idx is expected to be a range or slice for rows; c is the column index
            idx = range(idx.start, idx.stop)
            rec.dat.iloc[idx.start : idx.stop, c] = _signal.filtfilt(
                b, a, rec.dat.iloc[idx.start : idx.stop, c], **kwargs
            )
    return rec


def lowpass_filter(rec: BeForRecord, cutoff: float, order: int) -> BeForRecord:
    """
    Convenience function to apply a lowpass Butterworth filter to the force data
    in a `BeForRecord`.

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

    Examples
    --------
    >>> from befordata import BeForRecord, filtering
    >>> rec = BeForRecord(...)  # Create a BeForRecord instance
    >>> filtered = filtering.lowpass_filter(rec, cutoff=30, order=4)

    """

    rtn = _deepcopy(rec)

    b, a = _signal.butter(  # type: ignore
        order, cutoff, fs=rec.sampling_rate, btype="lowpass", analog=False
    )
    filtfilt(b, a, rtn, inplace=True)
    rtn.meta["filter"] = f"butterworth: cutoff={cutoff}, order={order}"

    return rtn
