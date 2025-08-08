Behavioural Force Data (BeForData)
==================================

**Data structures for handling behavioural force data**

This package provides core classes and utilities for loading, processing,
and analysing behavioural force data, such as those collected in experimental
psychology or neuroscience. It offers a structured approach to manage epochs and
records of force measurements, enabling efficient data manipulation and analysis.

BeForData is based on two :doc:`data_struct` of force data: one for the representation of
the raw time-based force measurements in the shape of a dataframe (**BeForRecord**) and
one for epoch-based representations as matrices (**BeForEpochs**).


**Features**

- Flexible loading and saving of force data in common formats (e.g., CSV, XDF).
- Efficient slicing and indexing of epochs and records for batch analysis.
- Metadata management for experimental context, including event markers and annotations.
- Utilities for preprocessing, such as filtering and baseline correction.
- Integration with scientific Python libraries (NumPy, pandas) for advanced analysis.

Source code: https://github.com/lindemann09/befordata

\(c\) Oliver Lindemann

|GitHub license| |PyPI|


Typical Workflow
----------------

1. Load raw force data into a BeForRecord object.
2. Preprocess and annotate the data as needed.
3. Segment the data into epochs using event markers, creating a BeForEpochs object.

See :doc:`usage` for a detailed usage guide.

Install via pip
----------------

::

   pip install befordata


Julia
-----

A `Julia implementation of BeForData <https://github.com/lindemann09/BeForData.jl>`_
is available as a beta release.

.. |GitHub license| image:: https://img.shields.io/github/license/lindemann09/befordata
   :target: https://github.com/lindemann09/befordata/blob/master/LICENSE
.. |PyPI| image:: https://img.shields.io/pypi/v/befordata?style=flat
   :target: https://pypi.org/project/befordata/


Contents
========

.. toctree::
   :maxdepth: 2

   usage
   api


