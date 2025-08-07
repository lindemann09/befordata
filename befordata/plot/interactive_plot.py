import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
except ImportError as exc:
    raise ImportError(
        "Please install matplotlib to use interactive plotting features."
    ) from exc


class InteractiveXYPlot(object):
    """Interactive plot with sliders."""

    def __init__(
        self,
        x_vals: NDArray,
        y_vals: NDArray,
        win_width:float,
        y_factor: float = 1.1,
        figsize=(14, 4),
    ):
        self._create_plot(figsize)
        self.lines.append(plt.plot(x_vals, y_vals))
        self._init_slider(
            min_x=np.min(x_vals),
            max_x=np.max(x_vals),
            win_width=win_width,
            y_factor=y_factor,
            y_range=(np.max(y_vals) - np.min(y_vals)),
            center_y=np.mean(y_vals[0:win_width])
        )

    def _create_plot(self, figsize):
        # Plot the initial data
        self.fig, self.ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(bottom=0.25, top=0.95, left=0.05, right=0.95)
        self.lines = []

    def _init_slider(
        self,
        min_x: float,
        max_x: float,
        win_width: float,
        y_factor: float,
        y_range: float,
        center_y: float | np.floating,
    ) -> None:
        """Initialize sliders for the interactive plot."""
        max_window_width = max_x - min_x
        if win_width <= 0:
            win_width = max_window_width * 0.10
        elif win_width > max_window_width:
            win_width = max_window_width

        if y_factor < 0.1:
            y_factor = 0.1
        elif y_factor > 1.5:
            y_factor = 1.5

        self._y_range2 = y_range / 2  # needed for update

        # make slider
        x_slider_axis = plt.axes((0.2, 0.15, 0.65, 0.03), facecolor="White")
        self.slider_x = Slider(
            x_slider_axis, "x", valmin=min_x, valmax=max_x, valinit=min_x
        )
        self.slider_x.on_changed(self._update_window)

        winwidth_slider_axis = plt.axes((0.2, 0.1, 0.65, 0.03), facecolor="White")
        self.slider_winwidth = Slider(
            winwidth_slider_axis,
            "range",
            valmin=0,
            valmax=max_window_width,
            valinit=win_width,
        )
        self.slider_winwidth.on_changed(self._update_window)

        y_slider_axis = plt.axes((0.2, 0.05, 0.65, 0.03), facecolor="White")
        self.slider_y = Slider(y_slider_axis, "y scale", 0.1, 1.5, valinit=y_factor)
        self.slider_y.on_changed(self._update_y_lim)

        ## init plot
        self._set_ylim(factor=y_factor, center=center_y)
        self._update_window(None)

    def _set_ylim(
        self, factor: float, center: float | np.floating | None = None
    ) -> tuple:
        """Set the y-axis limits with a factor of extra space.
        factor of maximum y range.
        """
        if center is None:
            ymin, ymax = self.ax.get_ylim()
            center = (ymin + ymax) / 2

        range_size = factor * self._y_range2
        y_range = (float(center - range_size), float(center + range_size))
        self.ax.set_ylim(y_range[0], y_range[1])
        return y_range

    def _update_window(self, _):
        x_pos = self.slider_x.val
        ww = self.slider_winwidth.val
        ymin, ymax = self.ax.get_ylim()
        self.ax.axis((x_pos, x_pos + ww, ymin, ymax))
        self.fig.canvas.draw_idle()

    def _update_y_lim(self, _):
        self._set_ylim(factor=self.slider_y.val)
        self.fig.canvas.draw_idle()


class InteractiveDataFramePlot(InteractiveXYPlot):
    """Interactive plot with y and x swapped."""

    def __init__(
        self,
        dat: pd.DataFrame,
        x_column: str,
        win_width:float,
        y_factor: float = 1.1,
        figsize=(14, 4),
    ):

        self._create_plot(figsize)

        x_vals = dat[x_column]
        # Loop through all columns except the x_column, plot each, and compute overall min, max, and center for y
        min_y = np.inf
        max_y = -np.inf
        y_center_lst = []
        for col in dat.columns:
            if col == x_column:
                continue
            # plot each column
            y_vals = dat[col].to_numpy()
            line_plt = plt.plot(x_vals, y_vals, label=col)
            self.lines.append(line_plt)

            min_y = min(y_vals.min(), min_y)
            max_y = max(y_vals.max(), max_y)
            y_center_lst.append(np.mean(y_vals[0:win_width]))
        center_y = np.mean(y_center_lst)

        plt.legend()  # Show legend

        self._init_slider(
            min_x=x_vals.min(),
            max_x=x_vals.max(),
            win_width=win_width,
            y_factor=y_factor,
            y_range=(max_y - min_y),
            center_y=center_y
        )
