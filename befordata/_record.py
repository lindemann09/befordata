"""
BeForData: Behavioural Force Data Management

This module defines the `BeForRecord` class for handling behavioural force
measurement datasets. It provides tools for session management, force data
manipulation, and conversion to an epoch-based representation. The structure
encapsulates measurement data, session boundaries, and metadata, offering a
unified and extensible interface.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ._epochs import BeForEpochs

pd.set_option("mode.copy_on_write", True)


@dataclass
class BeForRecord:
    """Data Structure for handling behavioural force measurements.

    This data structure encapsulates force measurement data, session information,
    and metadata, providing methods for session management, force extraction,
    and epoch extraction.

    Parameters
    ----------
    dat : pd.DataFrame
        The main data table containing force measurements and optionally a time column.
    sampling_rate : float
        The sampling rate (Hz) of the force measurements.
    sessions : list of int, optional
        List of sample indices where new recording sessions start. Defaults to [0].
    time_column : str, optional
        Name of the column containing time stamps. If empty, time stamps are generated.
    meta : dict, optional
        Arbitrary metadata associated with the record.
    """

    dat: pd.DataFrame
    sampling_rate: float
    sessions: List[int] = field(default_factory=list[int])
    time_column: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate and initialize the BeForRecord instance."""
        if not isinstance(self.dat, pd.DataFrame):
            raise TypeError(f"must be pandas.DataFrame, not {type(self.dat)}")

        if len(self.sessions) == 0:
            self.sessions.append(0)
        else:
            if isinstance(self.sessions, int):
                self.sessions = [self.sessions]
            if self.sessions[0] < 0:
                self.sessions[0] = 0
            elif self.sessions[0] > 0:
                self.sessions.insert(0, 0)

        if len(self.time_column) > 0 and self.time_column not in self.dat:
            raise ValueError(f"Time column {self.time_column} not found in DataFrame")

        self.force_cols = np.flatnonzero(self.dat.columns != self.time_column)

    def __repr__(self):
        """Return a string representation of the BeForRecord instance."""
        rtn = "BeForRecord"
        rtn += f"\n  sampling_rate: {self.sampling_rate}"
        rtn += f", n sessions: {self.n_sessions()}"
        if len(self.time_column) >= 0:
            rtn += f"\n  time_column: {self.time_column}"
        rtn += "\n  metadata"
        for k, v in self.meta.items():
            rtn += f"\n  - {k}: {v}".rstrip()
        rtn += "\n" + str(self.dat)
        return rtn

    def n_samples(self) -> int:
        """Return the total number of samples across all sessions.

        Returns
        -------
        int
            Number of samples (rows) in the data.
        """
        return self.dat.shape[0]

    def n_forces(self) -> int:
        """Return the number of force columns.

        Returns
        -------
        int
            Number of force measurement columns (excluding the time column).
        """
        return len(self.force_cols)

    def n_sessions(self) -> int:
        """Return the number of recording sessions.

        Returns
        -------
        int
            Number of sessions.
        """
        return len(self.sessions)

    def time_stamps(self) -> NDArray[np.floating]:
        """Return the time stamps as a numpy array.

        If a time column is specified, its values are returned.
        Otherwise, time stamps are generated based on the sampling rate.

        Returns
        -------
        np.ndarray
            Array of time stamps (float).
        """
        if len(self.time_column) > 0:
            return self.dat.loc[:, self.time_column].to_numpy()
        else:
            step = 1000.0 / self.sampling_rate
            final_time = self.dat.shape[0] * step
            return np.arange(0, final_time, step)

    def forces(self, session: int | None = None) -> pd.DataFrame | pd.Series:
        """Return force data for all samples or a specific session.

        Parameters
        ----------
        session : int or None, optional
            If specified, returns force data for the given session index.
            If None, returns force data for all samples.

        Returns
        -------
        pd.DataFrame or pd.Series
            Force data for the specified session or all data.
        """
        if session is None:
            return self.dat.loc[:, self.force_cols]  # type: ignore
        else:
            r = self.session_range(session)
            return self.dat.loc[r.start : r.stop, self.force_cols]  # type: ignore

    def add_session(self, dat: pd.DataFrame):
        """Append a new recording session to the data.

        The new DataFrame must have the same columns as the existing data.

        Parameters
        ----------
        dat : pd.DataFrame
            DataFrame containing new session data to append.
        """
        nbefore = self.dat.shape[0]
        self.dat = pd.concat([self.dat, dat], ignore_index=True)
        self.sessions.append(nbefore)

    def split_sessions(self) -> List[BeForRecord]:
        """Split the record into a list of BeForRecord objects, one per session.

        Returns
        -------
        list of BeForRecord
            Each element contains data for a single session.
        """
        rtn = []
        for idx in self.session_ranges():
            dat = BeForRecord(
                dat=self.dat.iloc[idx, :],  # type: ignore
                sampling_rate=self.sampling_rate,
                time_column=self.time_column,
                meta=self.meta,
            )
            rtn.append(dat)
        return rtn

    def session_ranges(self) -> List[range]:
        """Return a list of sample index ranges for all sessions.

        Returns
        -------
        list of range
            Each range corresponds to the sample indices of a session.
        """
        return [self.session_range(s) for s in range(len(self.sessions))]

    def session_range(self, session: int) -> range:
        """Return the sample index range for a specific session.

        Parameters
        ----------
        session : int
            Session index.

        Returns
        -------
        range
            Range of sample indices for the session.
        """
        f = self.sessions[session]
        try:
            t = self.sessions[session + 1]
        except IndexError:
            t = self.dat.shape[0]
        return range(f, t - 1)

    def find_samples_by_time(self, times: ArrayLike) -> NDArray:
        """Find the sample indices closest to the given time stamps.

        For each time in `times`, returns the index of the next larger or equal time stamp.

        Parameters
        ----------
        times : ArrayLike
            Array of time stamps to search for.

        Returns
        -------
        np.ndarray
            Array of sample indices corresponding to the input times.

        Notes
        -----
        Uses numpy.searchsorted with 'right' side.
        """
        return np.searchsorted(self.time_stamps(), np.atleast_1d(times), "right")

    def extract_epochs(
        self,
        column: str,
        n_samples: int,
        n_samples_before: int,
        zero_samples: List[int] | NDArray[np.int_] | None = None,
        zero_times: List[float] | NDArray[np.floating] | None = None,
        design: pd.DataFrame = pd.DataFrame(),
    ) -> BeForEpochs:
        """Extract epochs from the force data.

        Extracts epochs centered around specified zero samples or zero times, with a given
        number of samples before and after each zero point.

        Parameters
        ----------
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

        Returns
        -------
        BeForEpochs
            Object containing the extracted epochs.

        Raises
        ------
        ValueError
            If neither or both of `zero_samples` and `zero_times` are provided.

        Notes
        -----
        Provide either `zero_samples` or `zero_times`, not both.
        Use `find_samples_by_time` to convert times to sample indices if needed.
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
            return self.extract_epochs(
                column=column,
                n_samples=n_samples,
                n_samples_before=n_samples_before,
                zero_samples=self.find_samples_by_time(zero_times),
                design=design,
            )

        assert zero_samples is not None  # always!
        fd = self.dat.loc[:, column]
        n = len(fd)  # samples for data
        n_epochs = len(zero_samples)
        n_col = n_samples_before + n_samples
        force_mtx = np.empty((n_epochs, n_col), dtype=np.float64)
        for r, zs in enumerate(zero_samples):
            f = zs - n_samples_before
            if f > 0 and f < n:
                t = zs + n_samples
                if t > n:
                    warnings.warn(
                        "extract_force_epochs: last force epoch is incomplete, "
                        f"{t - n} samples missing.",
                        RuntimeWarning,
                    )
                    tmp = fd[f:]
                    force_mtx[r, : len(tmp)] = tmp
                    force_mtx[r, len(tmp) :] = 0
                else:
                    force_mtx[r, :] = fd[f:t]

        return BeForEpochs(
            force_mtx,
            sampling_rate=self.sampling_rate,
            design=design,
            zero_sample=n_samples_before,
        )
