"""
This module provides utility functions to extract and convert XDF stream data
(as returned by `pyxdf.load_xdf`) into pandas DataFrames and BeForRecord objects
for further analysis in the BeForData framework.

Key Features:
- Retrieve channel information and labels from XDF streams.
- Convert XDF channel data to pandas DataFrames with customizable time stamp column names.
- Create BeForRecord objects from XDF streams for use in BeForData workflows.

Global Variables:
- TIME_STAMPS: str (default = "time")
    Column name for time stamps in the resulting DataFrame. Can be changed
    by setting `befordata.xdf.before.TIME_STAMPS`.

(c) O. Lindemann
"""

import typing as _tp

import numpy as _np
import pandas as _pd

from ._record import BeForRecord

TIME_STAMPS = "time"


def _get_channel_id(xdf_streams: _tp.List[dict], name_or_id: int | str) -> int:
    if isinstance(name_or_id, int):
        return name_or_id

    for id_, stream in enumerate(xdf_streams):
        if stream["info"]["name"][0] == name_or_id:
            return id_
    raise ValueError(f"Can't find channel {name_or_id}")


def channel_info(xdf_streams: _tp.List[dict], channel: int | str) -> _tp.Dict:
    """
    Retrieve metadata information for a specific channel from XDF streaming data.

    Returns a dictionary containing channel metadata such as name, type, channel
    count, channel format, clock times, and clock values.

    Parameters
    ----------
    xdf_streams : list of dict
        List of XDF streams as returned by `pyxdf.load_xdf`.
    channel : int or str
        Channel index (int) or channel name (str).

    Raises
    ------
    ValueError
        If the specified channel cannot be found.
    """
    channel_id = _get_channel_id(xdf_streams, channel)
    info = xdf_streams[channel_id]["info"]
    fields = ("name", "type", "channel_count", "channel_format")
    rtn = {k: info[k][0] for k in fields}

    rtn["clock_times"] = xdf_streams[channel_id]["clock_times"]
    rtn["clock_values"] = xdf_streams[channel_id]["clock_values"]
    return rtn


def _channel_labels(xdf_streams: _tp.List[dict], channel: int | str) -> _tp.List[str]:
    """
    Extract channel labels from XDF streaming data.

    Given a list of XDF stream dictionaries (as returned by `pyxdf.load_xdf`),
    this function retrieves the labels for the specified channel, which can be
    identified either by its integer index or by its name.

    xdf_streams : List[dict]
        List of XDF stream dictionaries, typically as returned by
        `pyxdf.load_xdf`.
    channel : int or str
        Channel identifier. Can be an integer index or a string representing
        the channel name.


    Raises
    ------
    KeyError
        If the specified channel cannot be found in the provided streams.

    """

    channel_id = _get_channel_id(xdf_streams, channel)
    info = xdf_streams[channel_id]["info"]
    try:
        ch_info = info["desc"][0]["channels"][0]["channel"]
    except TypeError:
        ch_info = []

    if len(ch_info) == 0:
        return [info["name"][0]]
    else:
        return [x["label"][0] for x in ch_info]


def data(xdf_streams: _tp.List[dict], channel: int | str) -> _pd.DataFrame:
    """
    Convert XDF channel data to a pandas DataFrame.

    Returns a Pandas DataFrame containing time stamps and channel data, with
    columns named according to the global `TIME_STAMPS` variable and channel labels.

    Parameters
    ----------
    xdf_streams : list of dict
        List of XDF streams as returned by `pyxdf.load_xdf`.
    channel : int or str
        Channel index (int) or channel name (str).


    Raises
    ------
    ValueError
        If the specified channel cannot be found.
    """

    channel_id = _get_channel_id(xdf_streams, channel)
    lbs = [TIME_STAMPS] + _channel_labels(xdf_streams, channel_id)
    dat = _np.atleast_2d(xdf_streams[channel_id]["time_series"])
    t = _np.atleast_2d(xdf_streams[channel_id]["time_stamps"]).T
    return _pd.DataFrame(_np.hstack((t, dat)), columns=lbs)


def before_record(
    xdf_streams: _tp.List[dict], channel: int | str, sampling_rate: float
) -> BeForRecord:
    """
    Create a BeForRecord object from XDF stream data.

    Returns a `BeForRecord` object containing the channel data, sampling rate,
    time column name, and channel metadata.

    Parameters
    ----------
    xdf_streams : list of dict
        List of XDF streams as returned by `pyxdf.load_xdf`.
    channel : int or str
        Channel index (int) or channel name (str) to extract data from.
    sampling_rate : float
        Sampling rate of the channel data.

    Raises
    ------
    ValueError
        If the specified channel cannot be found.

    Examples
    --------

    >>> from pyxdf import load_xdf
    >>> streams, header = load_xdf("my_lsl_recording.xdf")
    >>> rec = xdf.before_record(streams, "Force", 1000)

    """
    channel_id = _get_channel_id(xdf_streams, channel)
    return BeForRecord(
        dat=data(xdf_streams, channel_id),
        sampling_rate=sampling_rate,
        time_column=TIME_STAMPS,
        meta=channel_info(xdf_streams, channel_id),
    )
