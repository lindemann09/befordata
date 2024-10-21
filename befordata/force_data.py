"""Before Data"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import pandas as pd
from pyarrow import Table, feather


@dataclass
class BeForData:
    """Data Structure for handling behavioural force Data

    Attributes
    ----------
    dat:  Pandas Dataframe
        data
    sample_rate: float
        the sampling rate of the force measurements
    columns: list of strings
        the columns in 'dat' that comprise the force measurements
    new_session: list of integer
        sample numbers at which a new recording session starts, if the exists
    meta: dictionary
        any kind of meta data
    """

    dat: pd.DataFrame
    sampling_rate: float
    columns: List[str] = field(default_factory=list[str])
    new_sessions: List[int] = field(default_factory=list[int])
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.columns) == 0:
            # make col forces
            self.columns = self.dat.columns.values.tolist()

        if len(self.new_sessions) == 0:
            self.new_sessions.append(0)

    def add_session(self, dat: pd.DataFrame):
        """Adds data (dataframe) as a new recording

        Dataframe has to have the same columns as the already existing data
        """
        nbefore = self.dat.shape[0]
        self.dat = pd.concat([self.dat, dat], ignore_index=True)
        self.new_sessions.append(nbefore)

    @property
    def n_total_samples(self) -> int:
        """Number of sample in all sessions"""
        return self.dat.shape[0]

    @property
    def n_forces(self) -> int:
        """Number of force columns"""
        return len(self.columns)

    @property
    def n_sessions(self) -> int:
        """Number of recoding sessions"""
        return len(self.new_sessions)

    def get_data(self,
               columns: Union[None,  str, List[str]] = None,
               session: Optional[int] = None) -> Union[pd.DataFrame, pd.Series]:
        """Returns data of a particular column and/or a particular session"""
        if columns is None:
            columns = self.dat.columns.values.tolist()

        if session is None:
            return self.dat.loc[:, columns] # type: ignore
        else:
            f = self.new_sessions[session]
            try:
                t = self.new_sessions[session+1]
            except IndexError:
                t = self.dat.shape[0]

            return self.dat.loc[f:(t-1), columns] # type: ignore

    def forces(self, session: Optional[int] = None) -> Union[pd.DataFrame, pd.Series]:
        """Returns force data of a particular session"""
        return self.get_data(self.columns, session)

    def add_column(self, name:str, data:Union[List, pd.Series],
                        is_force_column:bool=True):
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

    def drop_column(self, name:str):
        """Drop a column for the data (in place)"""
        self.dat = self.dat.drop(name, axis=1)
        try:
            self.columns.remove(name)
        except ValueError:
            pass

    def write_arrow(self, filepath: str) -> None:
        """Write the data a Arrow data file"""

        # Convert the DataFrame to a PyArrow table
        table = Table.from_pandas(self.dat)

        # Add metadata to the schema (serialize sampling_rate, timestamp, trigger, and meta)
        schema_metadata = {
            'sampling_rate': str(self.sampling_rate),
            'columns': ",".join(self.columns),
            'new_sessions': ",".join([str(x) for x in self.new_sessions])
        }
        schema_metadata.update(self.meta)
        table = table.replace_schema_metadata(schema_metadata)

        feather.write_feather(table, filepath, compression="lz4",
                            compression_level=6)

def to_befordata(pyarrow_table:Table) -> BeForData:
    """Converts a PyArrow table to BeforData object"""

    sr = 0
    columns = []
    new_sessions = []
    meta = {}
    for k, v in pyarrow_table.schema.metadata.items():
        if k == b"sampling_rate":
            sr = float(v)
        elif k == b"columns":
            columns = v.decode("utf-8").split(",")
        elif k == b"new_sessions":
            new_sessions = [int(x) for x in v.decode("utf-8").split(",")]
        else:
            meta[k.decode("utf-8")] = v.decode("utf-8")

    return BeForData(dat=pyarrow_table.to_pandas(),
                     sampling_rate=sr,
                     columns=columns,
                     new_sessions=new_sessions,
                     meta=meta)

def read_force_data(filepath: str) -> BeForData:
    """Read BeForData file in Arrow format"""

    return to_befordata(feather.read_table(filepath))


