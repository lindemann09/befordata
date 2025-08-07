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
from typing import Tuple

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

    Attributes
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
        Sample index representing time zero within each epoch (default: 0).

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

    def append(self, other: BeForEpochs):
        """
        Append epochs from another BeForEpochs instance.

        Parameters
        ----------
        other : BeForEpochs
            Another BeForEpochs object to append. Must have matching sample count,
            sampling rate, zero sample, baseline adjustment status, and design columns.
        """
        if other.n_samples() != self.n_samples():
            raise ValueError("Number of samples per epoch are not the same")
        if other.sampling_rate != self.sampling_rate:
            raise ValueError("Sampling rates are not the same.")
        if other.zero_sample != self.zero_sample:
            raise ValueError("Zero samples are not the same.")
        if other.is_baseline_adjusted() != self.is_baseline_adjusted():
            raise ValueError("One data structure is baseline adjusted, the other not.")
        if np.any(other.design.columns != self.design.columns):
            raise ValueError("Design column names are not the same.")

        self.dat = np.concat([self.dat, other.dat], axis=0)
        self.design = pd.concat([self.design, other.design], ignore_index=True)
        self.baseline = np.append(self.baseline, other.baseline)

    def is_baseline_adjusted(self):
        """
        Check if baseline adjustment has been applied.

        Returns
        -------
        bool
            True if baseline adjustment has been applied, False otherwise.
        """
        return len(self.baseline) > 0

    def adjust_baseline(self, reference_window: Tuple[int, int]):
        """
        Adjust the baseline of each epoch using the mean value of a defined sample window.

        Parameters
        ----------
        reference_window : Tuple[int, int]
            Tuple specifying the sample range (start, end) used for baseline adjustment.
        """
        if self.is_baseline_adjusted():
            dat = self.dat + np.atleast_2d(self.baseline).T  # restore baseline
        else:
            dat = self.dat
        i = range(reference_window[0], reference_window[1])
        self.baseline = np.mean(dat[:, i], axis=1)
        self.dat = dat - np.atleast_2d(self.baseline).T
