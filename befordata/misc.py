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
    if isinstance(val, (int, float)):
        return val
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val
