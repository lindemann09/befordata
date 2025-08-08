"""Miscellaneous helper functions

"""

ENC = "utf-8"

def values_as_string(d: dict) -> dict:
    """Helper function returns all keys as strings"""
    rtn = {}
    for v, k in d.items():
        if isinstance(k, (list, tuple)):
            rtn[v] = ",".join([str(x) for x in k])
        else:
            rtn[v] = str(k)
    return rtn

def try_num(val):
    """
    Attempts to convert the input value to an integer or float.

    Args:
        val: The value to attempt to convert.

    Returns an int, float, or original value: The converted value if possible,
        otherwise the original input.
    """
    if isinstance(val, (int, float)):
        return val
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val
