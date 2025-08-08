"""
Epochs Data Structure

This module defines the BeForEpochs data class for organizing and managing
behavioural force data segmented into epochs. Each epoch is represented as a
row in a 2D numpy array, with columns corresponding to samples within that
epoch. The class also maintains metadata such as sampling rate, experimental
design, baseline values, and the zero sample index.

"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class BeForEpochs:
    """
    Behavioural force data organized epoch-wise.

    This data structure stores and manages behavioural force data segmented
    into epochs. Each epoch is represented as a row in a 2D numpy array,
    with columns corresponding to samples within that epoch. Additional
    metadata such as sampling rate, experimental design, baseline values,
    and the zero sample index are also maintained.

    Parameters
    ----------
    dat : NDArray[np.floating]
        2D numpy array containing the force data. Each row is an epoch,
        each column a sample.
    sampling_rate : float
        Sampling rate of the force measurements (Hz).
    design : pd.DataFrame
        DataFrame containing design/metadata for each epoch (one row per
        epoch).
    baseline : NDArray[np.float64]
        1D numpy array containing baseline values for each epoch at
        `zero_sample`.
    zero_sample : int, optional
        Sample index representing the sample of the time zero within
        each epoch (default: 0).

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
        if self.baseline.ndim != 1:
            raise ValueError("Baseline must be a 1D array.")
        if len(self.baseline) > 0 and len(self.baseline) != ne:
            raise ValueError(
                "If baseline is not empty, the number of elements must match number of epochs."
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
        """Return the number of epochs."""
        return self.dat.shape[0]

    def n_samples(self) -> int:
        """Return the number of samples per epoch."""
        return self.dat.shape[1]


    def is_baseline_adjusted(self):
        """
        Check if baseline adjustment has been applied.

        Returns `True` if baseline adjustment has been applied, `False` otherwise.
        """
        return len(self.baseline) > 0
