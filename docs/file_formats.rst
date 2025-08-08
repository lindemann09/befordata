File Formats
============

Arrow & Feather
---------------

**Loading and saving BeForRecord data in Apache Arrow format**

To use this module, please install the python library *pyarrow*.


.. autosummary::

    befordata.arrow.record_to_arrow
    befordata.arrow.arrow_to_record
    befordata.arrow.epochs_to_arrow
    befordata.arrow.arrow_to_epochs


XDF
------------

**Converting XDF streaming data to BeForData**

Use the library *pyxdf* to read XDF files.

.. autosummary::

   befordata.xdf.before_record
   befordata.xdf.data
   befordata.xdf.channel_info


CSV
------------

**Support for reading compressed CSV files with embedded comments**

.. autosummary::

    befordata.csv.read_csv



befordata.arrow
=======================

.. autofunction:: befordata.arrow.record_to_arrow

.. autofunction:: befordata.arrow.arrow_to_record

.. autofunction:: befordata.arrow.epochs_to_arrow

.. autofunction:: befordata.arrow.arrow_to_epochs


befordata.xdf
=======================

.. autofunction:: befordata.xdf.before_record

.. autofunction:: befordata.xdf.data

.. autofunction:: befordata.xdf.channel_info


Globals
--------

To change the column name for time stamps in the dataframe, modify the global string
variable ``befordata.xdf.before.TIME_STAMPS`` (default: ``"time"``). Set this variable
to your preferred column name before loading data.


befordata.csv
=======================

.. autofunction:: befordata.csv.read_csv
