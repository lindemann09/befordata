from copy import copy as _copy
import typing as _tp

import pandas as _pd
from scipy import signal as _signal

from ._data import BeForData


def _butter_lowpass_filter(data:_pd.Series, cutoff: float, sampling_rate: float, order: int):
    b, a = _signal.butter(order, cutoff, fs=sampling_rate,
                  btype='lowpass', analog=False)
    y = _signal.filtfilt(b, a, data - data.iat[0]) + data.iat[0] # filter centered data
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
        for c in columns: # type: ignore
            df.loc[f:t, c] = _butter_lowpass_filter(
                                    data=df.loc[f:t, c],
                                    cutoff=cutoff_freq,
                                    sampling_rate=d.sampling_rate,
                                    order=butterworth_order)
    meta = _copy(d.meta)
    meta["cutoff_freq"]  = cutoff_freq
    meta["butterworth_order"]  = butterworth_order
    return BeForData(df, d.sampling_rate, d.columns, d.sessions, meta)
