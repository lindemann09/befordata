# Behavioural Force Data (BeForData)

**Data structures for handling behavioural force data**


This package provides core classes and utilities for loading, processing, and analysing behavioural force data, such as those collected in experimental psychology or neuroscience. It offers a structured approach to manage epochs and records of force measurements, enabling efficient data manipulation and analysis.

**Features**

- Flexible loading and saving of force data in common formats (e.g., CSV, XDF).
- Efficient slicing and indexing of epochs and records for batch analysis.
- Metadata management for experimental context, including event markers and annotations.
- Utilities for preprocessing, such as filtering and baseline correction.
- Integration with scientific Python libraries (NumPy, pandas) for advanced analysis.

Source code: https://github.com/lindemann09/befordata

Documentation: https://lindemann09.github.io/befordata/

(c) Oliver Lindemann

[![GitHub license](https://img.shields.io/github/license/lindemann09/befordata)](https://github.com/lindemann09/befordata/blob/master/LICENSE) [![PyPI](https://img.shields.io/pypi/v/befordata?style=flat)](https://pypi.org/project/befordata/)


## Main Components

- **BeForRecord**
    Represents a single continuous recording of force data, including metadata such as sampling rate, channel information, and experimental annotations. BeForRecord supports data cleaning, resampling, and extraction of epochs, and provides convenient access to raw and processed force signals.

- **BeForEpochs**
    A container class for managing multiple epochs of force data. Each epoch represents a segment of continuous force measurements, typically corresponding to a trial or experimental condition. BeForEpochs provides methods for slicing, indexing, and batch-processing epochs, as well as for loading and saving epoch data from various formats.


## Typical Workflow

1. Load raw force data into a BeForRecord object.
2. Preprocess and annotate the data as needed.
3. Segment the data into epochs using event markers, creating a BeForEpochs object.


### Install via pip

```
pip install befordata
```

### Julia

A [Julia implementation of BeForData](https://github.com/lindemann09/BeForData.jl) is available as a beta release.

