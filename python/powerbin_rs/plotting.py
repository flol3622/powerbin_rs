"""
Plotting utilities for PowerBin diagrams.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections, ticker

try:
    from plotbin.display_pixels import display_pixels

    HAS_PLOTBIN = True
except ImportError:
    HAS_PLOTBIN = False


def plot_power_diagram(xy, dens, bin_num, xybin, rbin, npix, magrange=20):
    """
    Plots a 2D Power Diagram tessellation colored by bin assignment.
    """
    single = npix == 1
    rng = np.random.default_rng(826)
    rnd = rng.permutation(rbin.size)

    if HAS_PLOTBIN:
        display_pixels(xy[:, 0], xy[:, 1], rnd[bin_num], pixelsize=1, cmap="Set3")
    else:
        # Fallback scatter if plotbin is not installed
        plt.scatter(
            xy[:, 0], xy[:, 1], c=rnd[bin_num], s=12, cmap="Set3", edgecolors="none"
        )

    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")

    ax = plt.gca()
    diam = 2 * rbin[~single]
    max_diam = np.max(diam) if diam.size > 0 else 1.0
    linewidth = 0.5 * (diam / max_diam) ** 0.3 if max_diam > 0 else 0.5

    if diam.size > 0:
        circles = collections.EllipseCollection(
            diam,
            diam,
            0,
            offsets=xybin[~single],
            units="xy",
            facecolor="none",
            edgecolors="k",
            lw=linewidth,
            transOffset=ax.transData,
        )
        ax.add_collection(circles)

    diam_dots = np.clip(rbin / 4.0, 0.3, None)
    circles_dots = collections.EllipseCollection(
        diam_dots,
        diam_dots,
        0,
        offsets=xybin,
        units="xy",
        facecolor="k",
        edgecolors="none",
        transOffset=ax.transData,
    )
    ax.add_collection(circles_dots)

    if dens is not None:
        max_dens = np.max(dens)
        if max_dens > 0:
            levels = max_dens * 10 ** (-0.4 * np.arange(magrange + 1)[::-1])
            plt.tricontour(*xy.T, dens, levels=levels, colors="indigo", linewidths=1)


class CustomAsinhLocator(ticker.AutoLocator):
    """
    A custom locator that combines AsinhLocator for large values
    and MaxNLocator for values near zero.
    """

    def __init__(self, linear_width=1.0):
        super().__init__()
        self._asinh_locator = ticker.AsinhLocator(linear_width, subs=None)
        self._linear_locator = ticker.MaxNLocator(steps=[1, 2, 5])

    def tick_values(self, vmin, vmax):
        asinh_ticks = np.asarray(self._asinh_locator.tick_values(vmin, vmax))
        linear_ticks = np.asarray(self._linear_locator.tick_values(vmin, vmax))
        ticks = np.union1d(
            asinh_ticks[np.abs(asinh_ticks) >= 1],
            linear_ticks[np.abs(linear_ticks) < 1],
        )
        return ticks


def format_asinh_axis(ax, axis="y", linear_width=1.0, max_labels=9):
    """
    Install major and minor formatters/locators for an 'asinh' axis.
    """

    def major_formatter(x, pos):
        if abs(x) < 1000:
            fmt = ".2g" if abs(x) < 1 else ".0f"
            return rf"${x:{fmt}}$"
        ex = int(np.floor(np.log10(abs(x))))
        ma = x / 10**ex
        if np.isclose(abs(ma), 1):
            return rf"${np.sign(ma) * 10:.0f}^{ex}$"
        return rf"${ma:.1f}\times10^{ex}$"

    def make_minor_formatter(subs):
        def minor_formatter(x, pos):
            if abs(x) < 1:
                return ""
            ex = int(np.floor(np.log10(abs(x))))
            ma = x / 10**ex
            if abs(ma) not in subs:
                return ""
            if abs(x) < 1000:
                return rf"${x:.0f}$"
            return rf"${ma:.1f}\times10^{ex}$"

        return minor_formatter

    ax_obj = ax.xaxis if axis == "x" else ax.yaxis
    ax_obj.set_major_locator(CustomAsinhLocator(linear_width))
    ax_obj.set_major_formatter(major_formatter)

    vmin, vmax = ax_obj.get_view_interval()
    major_ticks = ax_obj.get_major_locator().tick_values(vmin, vmax)
    major_ticks_in_view = major_ticks[(major_ticks >= vmin) & (major_ticks <= vmax)]
    n_major_labels = len(major_ticks_in_view)

    dense_minor_locator = ticker.AsinhLocator(linear_width, subs=range(1, 10))
    minor_ticks = dense_minor_locator.tick_values(vmin, vmax)
    minor_ticks_in_view = minor_ticks[(minor_ticks >= vmin) & (minor_ticks <= vmax)]
    all_potential_ticks = np.union1d(minor_ticks_in_view, major_ticks_in_view)

    subs_candidates = [[2, 3, 4, 6], [2, 5], []]
    chosen_subs = []

    for subs in subs_candidates:
        minor_fmt = make_minor_formatter(subs)
        n_minor_labels = sum(bool(minor_fmt(x, None)) for x in all_potential_ticks)
        if n_major_labels + n_minor_labels <= max_labels:
            chosen_subs = subs
            break

    ax_obj.set_minor_locator(dense_minor_locator)
    if chosen_subs:
        ax_obj.set_minor_formatter(make_minor_formatter(chosen_subs))
