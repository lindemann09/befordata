Module: xdf
============

**Converting XDF streaming data to BeForData**

Use the library *pyxdf* to read XDF files.

.. autofunction:: befordata.xdf.before_record

.. autofunction:: befordata.xdf.data

.. autofunction:: befordata.xdf.channel_info


Globals
--------

To change the column name for time stamps in the dataframe, modify the global string
variable ``befordata.xdf.before.TIME_STAMPS`` (default: ``"time"``). Set this variable
to your preferred column name before loading data.

