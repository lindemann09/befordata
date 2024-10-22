"""Data structure for handling behavioural force data"""

__author__ = "Oliver Lindemann"
__version__ = "0.1.7"

from ._force_data import BeForData, arrow2befor, read_befor_feather
from ._process import detect_sessions, lowpass_filter