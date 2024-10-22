"""Data structure for handling behavioural force data"""

__author__ = "Oliver Lindemann"
__version__ = "0.1.6"

from .force_data import BeForData, arrow2befor, read_befor_feather
from .process import detect_sessions, lowpass_filter