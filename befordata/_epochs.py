"""Epochs Data Structure"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray

from .misc import ENC, try_num, values_as_string

BSL_COL_NAME = '__befor_baseline__'

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
        """converts BeForEpochs to ``pyarrow.Table``

        Samples and design will be concatenated to one arrow table. If baseline
        is adjusted, additionally the baseline value will be added a column.

        Zero sample and sampling_rate will be included to schema meta data.
        of schema will be defined.

        Arrow tables can converted back to BeForRecord struct using
        ``BeForEpochs.from_arrow()``

        Returns
        -------
        pyarrow.Table

        """

        dat = pd.concat([pd.DataFrame(self.dat), self.design], axis=1)
        if len(self.baseline) >0:
            dat[BSL_COL_NAME] = self.baseline
        tbl = pa.Table.from_pandas(dat, preserve_index=False)

        schema_metadata = {
            "sampling_rate": str(self.sampling_rate),
            "zero_sample": str(self.zero_sample)
        }
        return tbl.replace_schema_metadata(schema_metadata)


    @staticmethod
    def from_arrow(
        tbl: pa.Table,
        sampling_rate: float | None = None,
        zero_sample: int | None = None,
    ) -> BeForEpochs:
        """Creates BeForEpoch struct from `pyarrow.Table`

        Parameters
        ----------
        tbl : pyarrow.Table

        """

        if not isinstance(tbl, pa.Table):
            raise TypeError(f"must be pyarrow.Table, not {type(tbl)}")

        # search arrow meta data for befor parameter
        if tbl.schema.metadata is not None:
            for k, v in tbl.schema.metadata.items():
                if k == b"sampling_rate":
                    if sampling_rate is None:
                        sampling_rate = try_num(v)
                elif k == b"zero_sample":
                    if zero_sample is None:
                        try:
                            zero_sample = int(try_num(v))
                        except ValueError:
                            zero_sample = 0

        if sampling_rate is None:
            raise RuntimeError("No sampling rate defined!")
        if zero_sample is None:
            zero_sample = 0

        dat = tbl.to_pandas()

        try:
            baseline = np.array(dat.pop(BSL_COL_NAME))
        except KeyError:
            baseline = np.array([])

        # count columns_name that have not int as name
        n_epoch_samples = dat.shape[1]
        for cn in reversed(dat.columns):
            try:
                int(cn)
                break
            except ValueError:
                n_epoch_samples -= 1

        return BeForEpochs(
            dat= dat.iloc[:, :n_epoch_samples],
            sampling_rate=sampling_rate,
            design=dat.iloc[:, n_epoch_samples:],
            baseline=baseline,
            zero_sample=zero_sample
        )