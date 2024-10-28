import gzip as _gzip
import lzma as _lzma
from io import StringIO as _StringIO
from pathlib import Path as _Path
import typing as _tp

import pandas as _pd

from ._data import BeForData

def read_csv(file_path: _tp.Union[str, _Path],
            columns: _tp.Union[None, str, _tp.List[str]] = None,
             encoding: str = "utf-8",
             comment_char: str = '#'):
    """Reads CSV file

    The function can handle comments as well as compressed CSVs, if the end
    with `.csv.xz` or `.csv.gz`

    Parameter
    ---------
    file_path : str | Path
        the path to the CSV file. If file end with `.csv.xz` or `.csv.gz`,
        decompression will be used
    columns : str | list[str], Optional
        the columns that should be read. If no columns are specified all columns
        are read
    encoding : str, optional
        file encoding, default="utf-8",

    comment_char: str, Optional
        line starting with character or string will be treated as comments and
        returned as a list of strings.

    Returns
    -------
    pandas.DataFrame and list[str] with all comments
    """

    p = _Path(file_path)
    if p.suffix.endswith("xz"):
        fl = _lzma.open(p, 'rt', encoding=encoding)
    elif p.suffix.endswith("gz"):
        fl = _gzip.open(file_path, 'rt', encoding=encoding)
    else:
        fl = open(file_path, 'r', encoding=encoding)

    csv_str = ""
    comments = []
    for l in fl.readlines():
        if l.startswith(comment_char):
            comments.append(l)
        else:
            csv_str += l
    fl.close()

    df = _pd.read_csv(_StringIO(csv_str))
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(columns, list):
        df = df.loc[:, columns]

    return df, comments


def read_csv_as_befordata(file_path: _tp.Union[str, _Path],
                          sampling_rate: float,
                          columns: _tp.Union[None, str, _tp.List[str]] = None,
                          sessions: _tp.Optional[_tp.List[int]] = None,
                          time_column: _tp.Optional[str] = None,
                          meta: _tp.Optional[dict] = None,
                          encoding: str = "utf-8",
                          comment_char: str = '#'):
    """Read CSV file as befordata
    """

    df, _ = read_csv(file_path=file_path,
                     encoding=encoding,
                     comment_char=comment_char)
    if columns is None:
        columns = []
    elif isinstance(columns, str):
        columns = [columns]

    if time_column is None:
        time_column = ""
    if sessions is None:
        sessions = []
    if not isinstance(meta, dict):
        meta = {}

    return BeForData(dat=df,
                     sampling_rate=sampling_rate,
                     columns=columns,
                     sessions=sessions,
                     time_column=time_column,
                     meta=meta)
