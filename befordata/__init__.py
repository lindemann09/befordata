"""
BeForData: Python data structures for handling behavioural force data.

This package provides core classes and utilities for loading, processing, and analysing
behavioural force data, such as those collected in experimental psychology or neuroscience.
It offers a structured approach to manage epochs and records of force measurements,
enabling efficient data manipulation and analysis.

MIT License

Author: Oliver Lindemann

"""

__author__ = "Oliver Lindemann"
__version__ = "0.4.3"


from ._epochs import BeForEpochs
from ._record import BeForRecord
from ._tools import (
                     adjust_baseline,
                     concat_epochs,
                     concat_records,
                     detect_sessions,
                     extract_epochs,
                     scale_epochs,
                     scale_record,
                     split_sessions,
                     subset_epochs,
)
