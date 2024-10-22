"""Epochs Data"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from pyarrow import Table, feather
from ._data import BeForData

# NPEpochs = NDArray[np.float_]


@dataclass
class BeForEpochs:
    """Behavioural force data organized epoch-wis

    Attributes
    ----------
    dat:

    sample_rate: float
        the sampling rate of the force measurements
    XXX
    """

    dat: NDArray[np.floating]
    sampling_rate: float
    design: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    baseline: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64))
    zero_sample: int = 0

    def __post_init__(self):
        self.dat = np.atleast_2d(self.dat)
        if self.dat.ndim != 2:
            raise ValueError("Epoch data but be a 2D numpy array")

        ne = self.n_epochs
        if self.design.shape[0] > 0 and self.design.shape[0] != ne:
            raise ValueError(
                "Epoch data and design must have the same number of rows")

        self.baseline = np.atleast_1d(self.baseline)
        if self.baseline.ndim != 1 and len(self.baseline) != ne:
            raise ValueError(
                "baseline must be a 1D array. The number of elements must match the of epochs")

    @property
    def n_epochs(self):
        return self.dat.shape[0]

    @property
    def n_samples(self):
        return self.dat.shape[1]


def epochs(d: BeForData,
           column: str,
           zero_sample: List[int],
           n_samples: int,
           n_samples_before: int=0,
           design: pd.DataFrame = pd.DataFrame()) -> BeForEpochs:  # FIXME How to handle sessions?

    fd = d.dat.loc[:, column]
    n = len(fd)  # samples for data
    n_epochs = len(zero_sample)
    n_col = n_samples_before + n_samples
    force_mtx = np.empty((n_epochs, n_col), dtype=np.float64)
    for (r, zs) in enumerate(zero_sample):
        f = zs - n_samples_before
        if f > 0 and f < n:
            t = zs + n_samples
            if t > n:
                warnings.warn(
                    f"extract_force_epochs: last force epoch is incomplete, {t-n} samples missing.",
                    RuntimeWarning)
                tmp = fd[f:]
                force_mtx[r, :len(tmp)]  = tmp
                force_mtx[r, len(tmp):]  = 0
            else:
                force_mtx[r, :] = fd[f:t]

    return BeForEpochs(force_mtx,
                       sampling_rate=d.sampling_rate,
                       design=design,
                       zero_sample=n_samples_before)

