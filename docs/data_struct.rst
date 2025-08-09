
Data Structures
================

.. autosummary::

   befordata.BeForRecord
   befordata.BeForEpochs


- **BeForRecord**
   Represents a single continuous recording of force data, including metadata such as
   sampling rate, channel information, and experimental annotations. BeForRecord
   supports data cleaning, resampling, and extraction of epochs, and provides
   convenient access to raw and processed force signals.

   The data structure has the following attributes:

   - **`dat`**: DataFrame containing force measurements and optionally a time column.
   - **`sampling_rate`**: Sampling rate of the force measurements (Hz).
   - **`sessions`**: List of sample indices where new recording sessions start.
   - **`time_column`** (optional): Name of the column containing time stamps (if any).
   - **`meta`** (optional): Arbitrary metadata associated with the record.

- **BeForEpochs**
   A container class for managing multiple epochs of force data. Each epoch
   represents a segment of continuous force measurements, typically corresponding
   to a trial or experimental condition. BeForEpochs provides methods for slicing,
   indexing, and batch-processing epochs, as well as for loading and saving epoch
   data from various formats.

   The data structure has the following attributes:

   - **`dat`**: 2D numpy array containing the force data (epochs x samples).
   - **`sampling_rate`**: Sampling rate of the force measurements (Hz).
   - **`design`**: DataFrame containing design/metadata for each epoch.
   - **`baseline`** (optional): 1D numpy array containing baseline values for each epoch at `zero_sample`.
   - **`zero_sample`**: Sample index representing the sample of the time zero within each epoch (default: 0).


BeForRecord
----------------------


.. autoclass:: befordata.BeForRecord
   :members:
   :member-order: bysource

BeForEpochs
----------------------

.. autoclass:: befordata.BeForEpochs
   :members:
   :member-order: bysource
