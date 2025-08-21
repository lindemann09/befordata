"""
BeForData: Behavioural Force Data Management

This module defines the `BeForRecord` class for handling behavioural force
measurement datasets. It provides tools for session management, force data
manipulation, and conversion to an epoch-based representation. The structure
encapsulates measurement data, session boundaries, and metadata, offering a
unified and extensible interface.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

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
        """Returns the total number of samples across all sessions."""
        return self.dat.shape[0]

    def n_forces(self) -> int:
        """Returns the number of force columns."""
        return len(self.force_cols)

    def n_sessions(self) -> int:
        """Returns the number of recording sessions."""
        return len(self.sessions)

    def time_stamps(self) -> NDArray:
        """Returns the time stamps as a numpy array.

        If a time column is specified, its values are returned.
        Otherwise, time stamps are generated based on the sampling rate.

        """
        if len(self.time_column) > 0:
            return self.dat.loc[:, self.time_column].to_numpy()
        else:
            step = 1000.0 / self.sampling_rate
            final_time = self.dat.shape[0] * step
            return np.arange(0, final_time, step)

    def forces(self, session: int | None = None) -> pd.DataFrame | pd.Series:
        """Returns force data for all samples or a specific session.

        Parameters
        ----------
        session : int or None, optional
            If specified, returns force data for the given session index.
            If None, returns force data for all samples.

        """
        if session is None:
            return self.dat.iloc[:, self.force_cols]
        else:
            r = self.session_range(session)
            return self.dat.iloc[r.start : r.stop, self.force_cols]

    def append(self, dat: pd.DataFrame, new_session: bool = False) -> None:
        """
        Appends new recordings to the existing data.

        Parameters
        ----------
        dat : pd.DataFrame
            DataFrame containing the new data to append. Must have the same
            columns as the current data.
        new_session : bool, optional
            If True, records will be marked as new session in the `sessions`
            list. Default is False.

        """

        nbefore = self.dat.shape[0]
        self.dat = pd.concat([self.dat, dat], ignore_index=True)
        if new_session:
            self.sessions.append(nbefore)

    def session_ranges(self) -> List[range]:
        """Returns a list of sample index ranges for all sessions.  Each range
        corresponds to the sample indices of a session.

        """
        return [self.session_range(s) for s in range(len(self.sessions))]

    def session_range(self, session: int) -> range:
        """Returns the sample index range for a specific session.

        Parameters
        ----------
        session : int
            Session index.

        """
        f = self.sessions[session]
        try:
            t = self.sessions[session + 1]
        except IndexError:
            t = self.dat.shape[0]
        return range(f, t - 1)

    def find_samples_by_time(self, times: ArrayLike) -> NDArray:
        """Find the indices of samples corresponding to or immediately following
        the given time stamps.

        This method searches for the indices of time stamps in the dataset that
        are equal to or the next larger value for each time in the input array.
        If an exact match is not found,the index of the next larger time stamp
        is returned.

        Parameters
        ----------
        times : ArrayLike
            Array of time stamps to search for.

        Notes
        -----
        Uses numpy.searchsorted with 'left' side.
        """
        return np.searchsorted(self.time_stamps(), np.atleast_1d(times), "left")
