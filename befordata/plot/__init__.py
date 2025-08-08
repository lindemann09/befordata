from befordata import BeForRecord

from .interactive_plot import InteractiveDataFramePlot, plt


def plot_record(
    rec: BeForRecord,
    win_width=100,
    y_factor: float = 1.1,
    time_scaling_factor: float | None = None,
    figsize=(14, 4),
):

    """
    This function visualizes the data contained in a BeForRecord object using an interactive plot.
    It supports optional time scaling and windowing for large datasets.

    Parameters
    ----------
    rec : BeForRecord
        The BeForRecord instance containing the data to plot.
    win_width : int, optional
        The width of the interactive window (number of samples), by default 100.
    y_factor : float, optional
        Factor to scale the y-axis for better visualization, by default 1.1.
    time_scaling_factor : float or None, optional
        If provided, scales the time axis by this factor. Must be a positive number or None.
        If None, no scaling is applied unless a time column is missing, in which case it defaults to 1.
    figsize : tuple, optional
        Size of the figure in inches (width, height), by default (14, 4).

    Raises
    ------
    AssertionError
        If `time_scaling_factor` is not None and is not a positive number.

    """

    assert time_scaling_factor is None or time_scaling_factor > 0, (
        "time_scaling_factor must be a positive number or None"
    )

    # data frame with time column
    if len(rec.time_column) > 0:  # has time column
        time_col = rec.time_column
    else:
        # no time column, enforce add time column
        time_col = "time"
        if time_scaling_factor is None:
            time_scaling_factor = 1

    if time_scaling_factor is not None and time_scaling_factor > 0:
        data = rec.dat.copy()
        data[time_col] = rec.time_stamps() * time_scaling_factor
    else:
        data = rec.dat

    ip = InteractiveDataFramePlot(
        dat=data,
        x_column=time_col,
        win_width=win_width,
        y_factor=y_factor,
        figsize=figsize,
    )
    plt.show()

    return ip
