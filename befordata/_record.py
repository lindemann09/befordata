"""Before Data"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pyarrow import Table

from ._epochs import BeForEpochs

ENC = "utf-8"


@dataclass
class BeForRecord:
    """Data Structure for handling behavioural force measurements

    Args
    ----
    dat: Pandas Dataframe
        data
    sample_rate: float
        the sampling rate of the force measurements
    columns: list of strings
        the columns in 'dat' that comprise the force measurements
    sessions: list of integer
        sample numbers at which a new recording session starts, if the exists
    time_column :
            str = ""
    meta: dictionary
        any kind of meta data
    """

    dat: pd.DataFrame
    sampling_rate: float
    columns: List[str] = field(default_factory=list[str])
    sessions: List[int] = field(default_factory=list[int])
    time_column: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.dat, pd.DataFrame):
            raise TypeError(f"must be pandas.DataFrame, not {type(self.dat)}")

        if isinstance(self.columns, str):
            self.columns = [self.columns]
        elif len(self.columns) == 0:
            # make col forces
            self.columns = self.dat.columns.values.tolist()

        if isinstance(self.sessions, int):
            self.sessions = [self.sessions]
        elif len(self.sessions) == 0:
            self.sessions.append(0)
        if len(self.time_column) > 0 and self.time_column not in self.dat:
            raise ValueError(f"Time column {self.time_column} not found in DataFrame")

    def __repr__(self):
        rtn = "BeForRecord"
        rtn += f"\n  sampling_rate: {self.sampling_rate}"
        rtn += f", n sessions: {self.n_sessions()}"
        rtn += f"\n  columns: {self.columns}".replace("[", "").replace("]", "")
        if len(self.time_column) >= 0:
            rtn += f"\n  time_column: {self.time_column}"
        rtn += "\n  metadata"
        for k, v in self.meta.items():
            rtn += f"\n  - {k}: {v}".rstrip()
        rtn += "\n" + str(self.dat)
        return rtn

    def add_session(self, dat: pd.DataFrame):
        """Adds data (dataframe) as a new recording

        Dataframe has to have the same columns as the already existing data
        """
        nbefore = self.dat.shape[0]
        self.dat = pd.concat([self.dat, dat], ignore_index=True)
        self.sessions.append(nbefore)

    def n_samples(self) -> int:
        """Number of sample in all sessions"""

        return self.dat.shape[0]

    def n_forces(self) -> int:
        """Number of force columns"""
        return len(self.columns)

    def n_sessions(self) -> int:
        """Number of recoding sessions"""
        return len(self.sessions)

    def time_stamps(self) -> NDArray:
        """The time stamps (numpy array)

        Creates time stamps, of they are not define in the data
        """
        if len(self.time_column) > 0:
            return self.dat.loc[:, self.time_column].to_numpy()
        else:
            step = 1000.0 / self.sampling_rate
            final_time = self.dat.shape[0] * step
            return np.arange(0, final_time, step)

    def session_samples(self, session: int) -> Tuple[int, int]:
        """tuple represents the range of the samples (from, to) of this sessions"""
        f = self.sessions[session]
        try:
            t = self.sessions[session + 1]
        except IndexError:
            t = self.dat.shape[0]
        return f, t - 1

    def get_data(
        self, columns: str | List[str] | None = None, session: int | None = None
    ) -> pd.DataFrame | pd.Series:
        """Returns data of a particular column and/or a particular session"""
        if columns is None:
            columns = self.dat.columns.values.tolist()

        if session is None:
            return self.dat.loc[:, columns]  # type: ignore
        else:
            f, t = self.session_samples(session)
            return self.dat.loc[f:t, columns]  # type: ignore

    def get_forces(self, session: int | None = None) -> pd.DataFrame | pd.Series:
        """Returns force data of a particular session"""
        return self.get_data(self.columns, session)

    def add_column(
        self, name: str, data: List | pd.Series, is_force_column: bool = True
    ):
        """Add data column (in place).

        Parameters
        ----------
        name : str
            columns name
        data : List or Pandas Series
            column data, that has to have the correct length
        is_force_column : bool, optional
            set this to False, if the added data do not comprise force
            measurements (default=true)
        """

        self.dat[name] = data
        if is_force_column:
            self.columns.append(name)

    def drop_column(self, name: str):
        """Drop a column for the data (in place)"""
        self.dat = self.dat.drop(name, axis=1)
        try:
            self.columns.remove(name)
        except ValueError:
            pass
        if self.time_column == name:
            self.time_column = ""

    def find_samples_by_time(self, times: ArrayLike) -> NDArray:
        """returns sample index (i) of the closes time in the BeForRecord.
        Takes the next larger element, if the exact time could not be found.

        .. math:: \\text{time_stamps}[i-1] <= t < \\text{time_stamps}[i]

        Parameters
        ----------
        timeline : ArrayLike
            the sorted array of time stamps

        """
        return np.searchsorted(self.time_stamps(), np.atleast_1d(times), "right")

    def extract_epochs(
        self,
        column: str,
        zero_samples: List[int] | NDArray[np.int_],
        n_samples: int,
        n_samples_before: int = 0,
        design: pd.DataFrame = pd.DataFrame(),
    ) -> BeForEpochs:
        """extracts epochs from BeForRecord

        Parameters
        ----------
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

        Notes
        -----
        use `find_times` to detect zero samples with time-based

        """

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
                        f"extract_force_epochs: last force epoch is incomplete, {t-n} samples missing.",
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

    def to_arrow(self) -> Table:
        """converts BeForRecord to ``pyarrow.Table``

        metadata of schema will be defined. Files can converted back to
        BeForRecord struct using ``BeForRecord.from_arrow()``

        Returns
        -------
        pyarrow.Table

        Examples
        --------
        >>> from pyarrow import feather
        >>> feather.write_feather(data.to_arrow(), "filename.feather",
                          compression="lz4", compression_level=6)
        """

        # Convert the DataFrame to a PyArrow table
        table = Table.from_pandas(self.dat, preserve_index=False)

        # Add metadata to the schema (serialize sampling_rate, timestamp, trigger, and meta)
        schema_metadata = {
            "sampling_rate": str(self.sampling_rate),
            "columns": ",".join(self.columns),
            "time_column": self.time_column,
            "sessions": ",".join([str(x) for x in self.sessions]),
        }
        schema_metadata.update(self.meta)
        return table.replace_schema_metadata(schema_metadata)

    @staticmethod
    def from_arrow(
        tbl: Table,
        sampling_rate: float | None = None,
        columns: str | List[str] | None = None,
        sessions: List[int] | None = None,
        time_column: str | None = None,
        meta: dict | None = None,
    ) -> BeForRecord:
        """Creates BeForRecord struct from `pyarrow.Table`

        Parameters
        ----------
        tbl : pyarrow.Table

        Examples
        --------
        >>> from pyarrow import feather
        >>> dat = feather.read_table("my_force_data.feather")
        >>> dat = BeforeData.from_arrow(dat)

        """

        if not isinstance(tbl, Table):
            raise TypeError(f"must be pyarrow.Table, not {type(tbl)}")

        if isinstance(columns, str):
            columns = [columns]

        # search info in meta data
        file_meta = {}
        if tbl.schema.metadata is not None:
            for k, v in tbl.schema.metadata.items():
                if k == b"sampling_rate":
                    if sampling_rate is None:
                        sampling_rate = float(v)
                elif k == b"columns":
                    if columns is None:
                        columns = v.decode(ENC).split(",")
                elif k == b"time_column":
                    if time_column is None:
                        time_column = v.decode(ENC)
                elif k == b"sessions":
                    if sessions is None:
                        sessions = [int(x) for x in v.decode(ENC).split(",")]
                else:
                    file_meta[k.decode(ENC)] = v.decode(ENC).strip()

        if sampling_rate is None:
            raise RuntimeError("No sampling rate defined!")
        if columns is None:
            columns = []
        if time_column is None:
            time_column = ""
        if sessions is None:
            sessions = []
        if isinstance(meta, dict):
            meta.update(file_meta)
        else:
            meta = file_meta

        return BeForRecord(
            dat=tbl.to_pandas(),
            sampling_rate=sampling_rate,
            columns=columns,  # type: ignore
            sessions=sessions,
            time_column=time_column,
            meta=meta,
        )
