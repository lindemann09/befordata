"""Data structure for handling behavioural force data"""

__author__ = "Oliver Lindemann"
__version__ = "0.1.7"

from ._data import BeForData, arrow2befor, read_befor_feather
from ._epochs import BeForEpochs, epochs
from ._process import detect_sessions