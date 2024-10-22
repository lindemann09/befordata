"""Epochs Data"""

import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

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
    design: pd.DataFrame = field(
        default_factory=pd.DataFrame())  # type: ignore
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
        """number of epochs"""
        return self.dat.shape[0]

    @property
    def n_samples(self):
        """number of sample of one epoch"""
        return self.dat.shape[1]

    def adjust_baseline(self, reference_window: Tuple[int, int]):
        """Adjust the baseline of each epoch using the mean value of
        a defined range of sample (reference window)

        Parameter
        ---------
        reference_window : Tuple[int, int]
            sample range that is used for the baseline adjustment

        """

        if len(self.baseline) > 0:
            dat = self.dat + np.atleast_2d(self.baseline).T  # rest baseline
        else:
            dat = self.dat
        i = range(reference_window[0], reference_window[1])
        self.baseline = np.mean(dat[:, i], axis=1)
        self.dat = dat - np.atleast_2d(self.baseline).T


def epochs(d: BeForData,
           column: str,
           zero_samples: Union[List[int], NDArray[np.int_]],
           n_samples: int,
           n_samples_before: int = 0,
           design: pd.DataFrame = pd.DataFrame()) -> BeForEpochs:
    """extracts epochs from BeForData

    Parameter
    ---------
    d: BeForData
        data
    column: str
        name of column containing the force data to be used
    zero_samples: List[int]
        zero sample that define the epochs
    n_samples: int
        number of samples to be extract (from zero sample on)
    n_samples_before: int, optional
        number of samples to be extracted before the zero sample (default=0)

    design: pd.DataFrame, optional
        design information

    Note
    ----
    use `find_times` to detect zero samples with time-based

    """

    fd = d.dat.loc[:, column]
    n = len(fd)  # samples for data
    n_epochs = len(zero_samples)
    n_col = n_samples_before + n_samples
    force_mtx = np.empty((n_epochs, n_col), dtype=np.float64)
    for (r, zs) in enumerate(zero_samples):
        f = zs - n_samples_before
        if f > 0 and f < n:
            t = zs + n_samples
            if t > n:
                warnings.warn(
                    f"extract_force_epochs: last force epoch is incomplete, {t-n} samples missing.",
                    RuntimeWarning)
                tmp = fd[f:]
                force_mtx[r, :len(tmp)] = tmp
                force_mtx[r, len(tmp):] = 0
            else:
                force_mtx[r, :] = fd[f:t]

    return BeForEpochs(force_mtx,
                       sampling_rate=d.sampling_rate,
                       design=design,
                       zero_sample=n_samples_before)
