# Usage


BeForData Usage Examples

## Create BeForRecord from csv-file

If your data includes time stamps, you must specify the time column when
creating the record. The example below also demonstrates how to add a
metadata dictionary.

``` python
import pandas as pd
from befordata import BeForRecord

# 1. read csv with Pandas
df = pd.read_csv("samples/demo_force_data.csv")

# 2. converting to before record
mydata = BeForRecord(df, sampling_rate=1000)
print(mydata)
```

    BeForRecord
      sampling_rate: 1000, n sessions: 1
      time_column:
      metadata
                 Fx      Fy     time
    0       -0.1717 -0.1143   601676
    1       -0.1719 -0.1136   601678
    2       -0.1719 -0.1133   601679
    3       -0.1718 -0.1209   601680
    4       -0.1697 -0.1020   601681
    ...         ...     ...      ...
    2334873  0.0991 -0.3851  3120147
    2334874  0.1034 -0.3789  3120147
    2334875  0.1013 -0.3704  3120149
    2334876  0.1013 -0.3875  3120149
    2334877  0.0992 -0.3883  3120151

    [2334878 rows x 3 columns]

``` python
mydata = BeForRecord(
    df, sampling_rate=1000, time_column="time", meta={"Exp": "my experiment"}
)
```

## Epochs-based representation

Epochs are represented as matrix. Each row is one trial

Example

-   Extracting epochs of the length 2000 from `Fx` (plus 100 samples
    before)
-   the 6 epochs start at the 6 “zero samples”

``` python
from befordata import tools

epochs = tools.extract_epochs(mydata, "Fx",
    zero_samples=[1530, 6021, 16983, 28952, 67987],
    n_samples=2000,
    n_samples_before=10,
)
print(epochs)
```

    BeForEpochs
      n epochs: 5, n_samples: 2010
      sampling_rate: 1000, zero_sample: 10
      design: None

**Note**: BeForEpochs should usually contain information about the
experimental design. See the example of data preprocessing below.

## Pyarrow & Feather Format

Arrow and feather file format is fast and platform & language
independent

### Writing and reading BeforRecord using the feather format

``` python
from pyarrow.feather import write_feather, read_table
from befordata import arrow

# writing
tbl = arrow.record_to_arrow(mydata)
write_feather(tbl, "demo.feather")

# reading
mydata2 = arrow.arrow_to_record(read_table("demo.feather"))
```

### Writing and reading epochs

``` python
# writing
tbl = arrow.epochs_to_arrow(epochs)
write_feather(tbl, "epochs.feather",
    compression="lz4", compression_level=8) # optional compression parameters

# reading
epochs2 = arrow.arrow_to_epochs(read_table("epochs.feather"))
```

## Extracting data for XDF files (as used by LSL)

``` python
from pyxdf import load_xdf
from befordata import xdf

# read xdf
streams, header = load_xdf("samples/xdf_sample.xdf")

# extract data
rec = xdf.before_record(streams, "MousePosition", 1000)
rec
```

    BeForRecord
      sampling_rate: 1000, n sessions: 1
      time_column: time
      metadata
      - name: MousePosition
      - type: Position
      - channel_count: 2
      - channel_format: float32
      - clock_times: [29570.8205096245, 29575.8203263595, 29580.8201211015]
      - clock_values: [-7.819500751793385e-06, -7.819498932803981e-06, -5.864501872565597e-06]
                 time  MouseX  MouseY
    0    29570.255833   592.0   373.0
    1    29570.268411   575.0   373.0
    2    29570.280998   553.0   373.0
    3    29570.293601   523.0   376.0
    4    29570.305144   493.0   380.0
    ..            ...     ...     ...
    204  29581.561514   192.0   146.0
    205  29581.574091   194.0   146.0
    206  29581.599295   195.0   146.0
    207  29581.637042   197.0   146.0
    208  29581.662244   198.0   146.0

    [209 rows x 3 columns]

## Example of data preprocessing with experimental design

``` python
import pandas as pd
from befordata import BeForRecord, BeForEpochs, tools

# 1. read csv with Pandas
df = pd.read_csv("samples/demo_force_data.csv")


# 2. converting pandas data to before record
mydata = BeForRecord(
    df, sampling_rate=1000, time_column="time", meta={"Exp": "my experiment"}
)

# 3. detect pauses and treat data as recording with different sessions
tools.detect_sessions(mydata, time_gap=2000)

# 4. filter data (takes into account the different sessions)
flt_data = tools.lowpass_filter(mydata, cutoff=30, order=4)

# 5. read design data (csv)
design = pd.read_csv("samples/demo_design_data.csv")

# 6. extract epochs
ep = tools.extract_epochs(flt_data, "Fx",
    n_samples=5000, n_samples_before=100, design=design,
    zero_times = design.trial_time
)

# 7. adjust baseline
tools.adjust_baseline(ep, (80, 100))

print(ep)
```

    BeForEpochs
      n epochs: 391, n_samples: 5100
      sampling_rate: 1000, zero_sample: 100
      design: 'operand_1', 'operand_2', 'operator', 'correct_response', 'response', 'resp_number_digits', 'resp_num_category', 'subject_id', 'trial', 'trial_time'
