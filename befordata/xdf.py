"""
Converting XDF streaming data from `pyxdf` to BeForData

(c) O. Lindemann
"""

import typing as _tp

import numpy as _np
import pandas as _pd

from ._record import BeForRecord

TIME_STAMPS = "time"

def _get_channel_id(xdf_streams:_tp.List[dict], name_or_id: int | str) -> int:
    if isinstance(name_or_id, int):
        return name_or_id

    for id_, stream in enumerate(xdf_streams):
        if stream["info"]["name"][0] == name_or_id:
            return id_
    raise ValueError(f"Can't find channel {name_or_id}")

def channel_info(xdf_streams: _tp.List[dict], channel : int | str) -> _tp.Dict:
    """channel info from xdf stream data

    Parameters
    ----------
    xdf: list of dicts
        xdf streams (as returned by `pyxdf.load_xdf`)
    channel: int | str
        channel id (int) or channel name (str)

    Returns
    -------
    dict

    """
    channel_id = _get_channel_id(xdf_streams, channel)
    info = xdf_streams[channel_id]["info"]
    fields = ("name", "type", "channel_count", "channel_format")
    rtn = {k: info[k][0] for k in fields}

    rtn["clock_times"] = xdf_streams[channel_id]["clock_times"]
    rtn["clock_values"] = xdf_streams[channel_id]["clock_values"]
    return rtn

def _channel_labels(xdf_streams: _tp.List[dict], channel : int | str) -> _tp.List[str]:
    """channel labels from xdf stream data

    Parameters
    ----------
    xdf: list of dicts
        xdf streams (as returned by `pyxdf.load_xdf`)
    channel: int | str
        channel id (int) or channel name (str)

    Returns
    -------
    List[str]

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

def data(xdf_streams: _tp.List[dict], channel : int | str) -> _pd.DataFrame :
    """Channel Pandas dataframe from xdf stream data

    Parameters
    ----------
    xdf: list of dicts
        xdf streams (as returned by `pyxdf.load_xdf`)
    channel: int | str
        channel id (int) or channel name (str)

    Returns
    -------
    pandas.DataFrame

    """
    channel_id = _get_channel_id(xdf_streams, channel)
    lbs = [TIME_STAMPS] + _channel_labels(xdf_streams, channel_id)
    dat = _np.atleast_2d(xdf_streams[channel_id]["time_series"])
    t = _np.atleast_2d(xdf_streams[channel_id]["time_stamps"]).T
    return _pd.DataFrame(_np.hstack((t, dat)), columns=lbs)

def before_record(xdf_streams: _tp.List[dict],
                  channel : int | str,
                  sampling_rate:float) -> BeForRecord:
    """BeforeData from xdf stream data

    Parameters
    ----------
    xdf: list of dicts
        xdf streams (as returned by `pyxdf.load_xdf`)
    channel: int | str
        channel id (int) or channel name (str)
    sampling_rate: float
        the sampling rate of the force measurements

    Returns
    -------
    BeForRecord

    """
    channel_id = _get_channel_id(xdf_streams, channel)
    return BeForRecord(dat = data(xdf_streams, channel_id),
                       sampling_rate=sampling_rate,
                       time_column=TIME_STAMPS,
                       meta=channel_info(xdf_streams, channel_id))


