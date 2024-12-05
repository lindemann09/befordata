"""Epochs Data Structure"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class BeForEpochs:
    """Behavioural force data organized epoch-wis

    Args
    ----
    dat: : 2d numpy array
        data. Each row of the 2D numpy array represents one epoch. Thus, the number
        of rows equals the number of epochs and number of columns equals the number
        of samples in each epoch.

    sample_rate: float
        sampling rate of the force measurements

    design : pd.DataFrame
        design data frame

    baseline : numpy array
        baseline for each epoch at `zero_sample`

    zero_sample : int, optional
        sample index that represents the time 0

    """

    dat: NDArray[np.floating]
    sampling_rate: float
    design: pd.DataFrame = field(default_factory=pd.DataFrame())  # type: ignore
    baseline: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    zero_sample: int = 0

    def __post_init__(self):
        self.dat = np.atleast_2d(self.dat)
        if self.dat.ndim != 2:
            raise ValueError("Epoch data but be a 2D numpy array")

        ne = self.n_epochs()
        if self.design.shape[0] > 0 and self.design.shape[0] != ne:
            raise ValueError("Epoch data and design must have the same number of rows")

        self.baseline = np.atleast_1d(self.baseline)
        if self.baseline.ndim != 1 and len(self.baseline) != ne:
            raise ValueError(
                "baseline must be a 1D array. The number of elements must match the of epochs"
            )

    def __repr__(self):
        rtn = "BeForEpochs"
        rtn += f"\n  n epochs: {self.n_epochs()}"
        rtn += f", n_samples: {self.n_samples()}"
        rtn += f"\n  sampling_rate: {self.sampling_rate}"
        rtn += f", zero_sample: {self.zero_sample}"
        if len(self.design) == 0:
            rtn += "\n  design: None"
        else:
            rtn += f"\n  design: {list(self.design.columns)}".replace("[", "").replace(
                "]", ""
            )
        # rtn += "\n" + str(self.dat)
        return rtn

    def n_epochs(self) -> int:
        """number of epochs"""
        return self.dat.shape[0]

    def n_samples(self) -> int:
        """number of sample of one epoch"""
        return self.dat.shape[1]

    def adjust_baseline(self, reference_window: Tuple[int, int]):
        """Adjust the baseline of each epoch using the mean value of
        a defined range of sample (reference window)

        Parameters
        ----------
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

    def to_arrow(self):
        # FIXME
        pass
        pass
