# Usage

## Create BeForRecord from csv-file


```python
import pandas as pd
from befordata import BeForRecord

# 1. read csv with Pandas
df = pd.read_csv("demo_force_data.csv")

# 2. converting to before record
mydata = BeForRecord(df, sampling_rate = 1000)
print(mydata)
```




    BeForRecord
      sampling_rate: 1000, n sessions: 1
      columns: 'Fx', 'Fy', 'time', 'trigger'
      time_column:
      metadata
                 Fx      Fy     time  trigger
    0       -0.1717 -0.1143   601676   0.0000
    1       -0.1719 -0.1136   601678   0.0000
    2       -0.1719 -0.1133   601679   0.0000
    3       -0.1718 -0.1209   601680   0.0000
    4       -0.1697 -0.1020   601681   0.0000
    ...         ...     ...      ...      ...
    2334873  0.0991 -0.3851  3120147   0.9656
    2334874  0.1034 -0.3789  3120147   0.9650
    2334875  0.1013 -0.3704  3120149   0.9653
    2334876  0.1013 -0.3875  3120149   0.9653
    2334877  0.0992 -0.3883  3120151   0.9660

    [2334878 rows x 4 columns]



Adding some additional information


```python
mydata = BeForRecord(df, sampling_rate = 1000,
                         columns=["Fx"],
                         time_column = "time",
                         meta = {"Exp": "my experiment"})

print(mydata)
```

    BeForRecord
      sampling_rate: 1000, n sessions: 1
      columns: 'Fx'
      time_column: time
      metadata
      - Exp: my experiment
                 Fx      Fy     time  trigger
    0       -0.1717 -0.1143   601676   0.0000
    1       -0.1719 -0.1136   601678   0.0000
    2       -0.1719 -0.1133   601679   0.0000
    3       -0.1718 -0.1209   601680   0.0000
    4       -0.1697 -0.1020   601681   0.0000
    ...         ...     ...      ...      ...
    2334873  0.0991 -0.3851  3120147   0.9656
    2334874  0.1034 -0.3789  3120147   0.9650
    2334875  0.1013 -0.3704  3120149   0.9653
    2334876  0.1013 -0.3875  3120149   0.9653
    2334877  0.0992 -0.3883  3120151   0.9660

    [2334878 rows x 4 columns]


## Epochs-based representation

Epochs are represented as matrix. Each row is one trial

Example

* Extracting epochs of the length 2000 from `Fx` (plus 100 samples before)
* the 6 epochs  start at the 6 "zero samples"


```python
epochs = mydata.extract_epochs("Fx",
            zero_samples = [1530, 6021, 16983, 28952, 67987],
            n_samples=2000,
            n_samples_before=10)
print(epochs)
```

    BeForEpochs
      n epochs: 5, n_samples: 2010
      sampling_rate: 1000, zero_sample: 10
      design: None


## Pyarrow format

Arrow and feather file format is very fast and platform & language independent


### Saving to feather file


```python
from pyarrow.feather import write_feather

# 1. convert to pyarrow table
tbl = mydata.to_arrow()


# 2. e.g. save pyarrow table to feather file
write_feather(tbl, "demo.feather", compression="lz4",
               compression_level=6)
```

### Loading BeforRecord from feather file


```python

from pyarrow.feather import read_table

# 1. load files as arrow table
tbl = read_table("demo.feather")

# 2. Convert to BeForRecord
mydata = BeForRecord.from_arrow(tbl)
```

## Example of the data preprocessing of experimental data with design


```python
import pandas as pd
from befordata import BeForRecord, tools

# 1. read csv with Pandas
df = pd.read_csv("demo_force_data.csv")


# 2. converting pandas data to before record
mydata = BeForRecord(df,
                   sampling_rate=1000,
                   columns=["Fx", "Fy"],
                   time_column = "time",
                   meta = {"Exp": "my experiment"})

# 3. detect pauses and treat data as recording with different sessions
mydata = tools.detect_sessions(mydata, time_gap=2000)

# 4. filter data (takes into account the different sessions)
mydata = tools.butter_filter(mydata, cutoff=30, order=4, btype="lowpass")

# 5. read design data (csv)
design = pd.read_csv("demo_design_data.csv")

# 6. get samples from times of the trial onset in the design (`trial_time`)
samples = mydata.find_samples_by_time(design.trial_time)

# 7. extract epochs
ep = mydata.extract_epochs("Fx", samples,
                    n_samples = 5000, n_samples_before=100, design=design)

print(ep)
```

    BeForEpochs
      n epochs: 391, n_samples: 5100
      sampling_rate: 1000, zero_sample: 100
      design: 'operand_1', 'operand_2', 'operator', 'correct_response', 'response', 'resp_number_digits', 'resp_num_category', 'subject_id', 'trial', 'trial_time'



