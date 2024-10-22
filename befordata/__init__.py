"""Data structure for handling behavioural force data"""

__author__ = "Oliver Lindemann"
__version__ = "0.1.8"

from ._data import BeForData, arrow2befor, read_befor_feather
from ._epochs import BeForEpochs
from ._process import detect_sessions, find_times
