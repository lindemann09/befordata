"""Reading compressed csv files with comments"""

import gzip as _gzip
import lzma as _lzma
from io import StringIO as _StringIO
from pathlib import Path as _Path
from typing import List as _List
from typing import Tuple as _Tuple

import pandas as _pd


def read_csv(
    file_path: str | _Path,
    columns: str | _List[str] | None = None,
    encoding: str = "utf-8",
    comment_char: str = "#",
) -> _Tuple[_pd.DataFrame, _List[str]]:
    """
    Reads a CSV file, supporting comments and compression.

    This function reads a CSV file and returns a Pandas DataFrame,
    optionally compressed with `.xz` or `.gz`, and extracts any comment
    lines (lines starting with `comment_char`). The comments are returned
    as a list, and the CSV data is loaded into a pandas DataFrame.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the CSV file. Supports uncompressed, `.csv.xz`, or
        `.csv.gz` files.
    columns : str or list of str, optional
        Column name or list of column names to select from the CSV.
        If None, all columns are read.
    encoding : str, default "utf-8"
        File encoding to use when reading the file.
    comment_char : str, default "#"
        Lines starting with this character are treated as comments and
        returned separately.


    """

    p = _Path(file_path)
    if p.suffix.endswith("xz"):
        fl = _lzma.open(p, "rt", encoding=encoding)
    elif p.suffix.endswith("gz"):
        fl = _gzip.open(file_path, "rt", encoding=encoding)
    else:
        fl = open(file_path, "r", encoding=encoding)

    csv_str = ""
    comments = []
    for line in fl.readlines():
        if line.startswith(comment_char):
            comments.append(line)
        else:
            csv_str += line
    fl.close()

    df = _pd.read_csv(_StringIO(csv_str))
    df = df.copy()  # copy: to solve potential fragmented dataframe problem
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(columns, list):
        df = df.loc[:, columns]

    return df, comments
