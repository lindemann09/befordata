"""
converting BeForData struct from and to arrow tables. Using for saving data.

(c) O. Lindemann
"""

import typing as _tp

import numpy as _np
import pandas as _pd
import pyarrow as _pa

from . import BeForEpochs, BeForRecord, _misc
from ._misc import ENC

BSL_COL_NAME = '__befor_baseline__'

def record_to_arrow(rec:BeForRecord) -> _pa.Table:
    """converts BeForRecord to ``pyarrow.Table``

    metadata of schema will be defined.

    Returns
    -------
    pyarrow.Table

    Examples
    --------
    >>> from pyarrow import feather
    >>> tbl = record_to_arrow(my_record)
    >>> feather.write_feather(tbl, "filename.feather",
                        compression="lz4", compression_level=6)
    """

    # Convert the DataFrame to a PyArrow table
    table = _pa.Table.from_pandas(rec.dat, preserve_index=False)

    # Add metadata to the schema (serialize sampling_rate, timestamp, trigger, and meta)
    schema_metadata = {
        "sampling_rate": str(rec.sampling_rate),
        "time_column": rec.time_column,
        "sessions": ",".join([str(x) for x in rec.sessions]),
    }
    schema_metadata.update(_misc.values_as_string(rec.meta))
    return table.replace_schema_metadata(schema_metadata)

def arrow_to_record(
    tbl: _pa.Table,
    sampling_rate: float | None = None,
    sessions: _tp.List[int] | None = None,
    time_column: str | None = None,
    meta: dict | None = None,
) -> BeForRecord:
    """Creates BeForRecord struct from `pyarrow.Table`

    Parameters
    ----------
    tbl : pyarrow.Table

    Examples
    --------
    >>> from pyarrow.feather import read_table
    >>> dat = arrow_to_record(read_table("my_force_data.feather"))

    """

    if not isinstance(tbl, _pa.Table):
        raise TypeError(f"must be pyarrow.Table, not {type(tbl)}")

    # search arrow meta data for befor record parameter
    arrow_meta = {}
    if tbl.schema.metadata is not None:
        for k, v in tbl.schema.metadata.items():
            if k == b"sampling_rate":
                if sampling_rate is None:
                    sampling_rate = _misc.try_num(v)
            elif k == b"time_column":
                if time_column is None:
                    time_column = v.decode(ENC)
            elif k == b"sessions":
                if sessions is None:
                    sessions = [int(x) for x in v.decode(ENC).split(",")]
            else:
                arrow_meta[k.decode(ENC)] = _misc.try_num(v.decode(ENC).strip())

    # make befor meta data
    if isinstance(meta, dict):
        meta.update(arrow_meta)
    else:
        meta = arrow_meta

    if sampling_rate is None:
        raise RuntimeError("No sampling rate defined!")
    if time_column is None:
        time_column = ""
    if sessions is None:
        sessions = []

    return BeForRecord(
        dat=tbl.to_pandas(),
        sampling_rate=sampling_rate,
        sessions=sessions,
        time_column=time_column,
        meta=meta
    )


def epochs_to_arrow(rec:BeForEpochs) -> _pa.Table:
    """converts BeForEpochs to ``pyarrow.Table``

    Samples and design will be concatenated to one arrow table. If baseline
    is adjusted, additionally the baseline value will be added a column.

    Zero sample and sampling_rate will be included to schema meta data.
    of schema will be defined.

    Returns
    -------
    pyarrow.Table

    Examples
    --------
    >>> from pyarrow import feather
    >>> tbl = record_to_arrow(my_epochs)
    >>> feather.write_feather(tbl, "my_epochs.feather",
                        compression="lz4", compression_level=6)
    """

    dat = _pd.concat([_pd.DataFrame(rec.dat), rec.design], axis=1)
    if rec.is_baseline_adjusted():
        dat[BSL_COL_NAME] = rec.baseline
    tbl = _pa.Table.from_pandas(dat, preserve_index=False)

    schema_metadata = {
        "sampling_rate": str(rec.sampling_rate),
        "zero_sample": str(rec.zero_sample)
    }
    return tbl.replace_schema_metadata(schema_metadata)


def arrow_to_epochs(
    tbl: _pa.Table,
    sampling_rate: float | None = None,
    zero_sample: int | None = None,
) -> BeForEpochs:
    """Creates BeForEpoch struct from `pyarrow.Table`

    Parameters
    ----------
    tbl : pyarrow.Table

    Examples
    --------
    >>> from pyarrow.feather import read_table
    >>> dat = arrow_to_epochs(read_table("my_epochs.feather"))

    """

    if not isinstance(tbl, _pa.Table):
        raise TypeError(f"must be pyarrow.Table, not {type(tbl)}")

    # search arrow meta data for befor parameter
    if tbl.schema.metadata is not None:
        for k, v in tbl.schema.metadata.items():
            if k == b"sampling_rate":
                if sampling_rate is None:
                    sampling_rate = _misc.try_num(v)
            elif k == b"zero_sample":
                if zero_sample is None:
                    try:
                        zero_sample = int(_misc.try_num(v))
                    except ValueError:
                        zero_sample = 0

    if sampling_rate is None:
        raise RuntimeError("No sampling rate defined!")
    if zero_sample is None:
        zero_sample = 0

    dat = tbl.to_pandas()

    try:
        baseline = _np.array(dat.pop(BSL_COL_NAME))
    except KeyError:
        baseline = _np.array([])

    # n_epoch_samples: count columns_name that have not int as name
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