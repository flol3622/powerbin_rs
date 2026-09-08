"""
PowerBin: Fast Adaptive 2D Data Binning with Centroidal Power Diagrams
Optimized Rust Core Implementation
"""
from time import perf_counter
from typing import Callable, Optional, Union
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy import sparse

from ._core import PowerBinCore, power_diagram as _core_power_diagram
from .plotting import format_asinh_axis, plot_power_diagram


def power_diagram(xy: ArrayLike, xybin: ArrayLike, rbin: ArrayLike) -> np.ndarray:
    """
    Computes a Power Diagram tessellation for a set of 2D points.

    Parameters
    ----------
    xy : array_like of shape (npix, 2)
        Input pixel coordinates.
    xybin : array_like of shape (nbin, 2)
        Generator coordinates.
    rbin : array_like of shape (nbin,)
        Generator radii.

    Returns
    -------
    bin_num : ndarray of int of shape (npix,)
        Index of the closest generator under power distance for each point.
    """
    xy = np.asarray(xy, dtype=np.float64)
    xybin = np.asarray(xybin, dtype=np.float64)
    rbin = np.asarray(rbin, dtype=np.float64)
    return np.asarray(_core_power_diagram(xy, xybin, rbin), dtype=int)


def update_bins(capacity_spec, xy, xybin, rbin, args=()):
    """
    Updates bin properties based on a Power Diagram tessellation.
    """
    bin_num = power_diagram(xy, xybin, rbin)
    N = len(xy)
    S = sparse.csr_array((np.ones(N), (bin_num, np.arange(N))), shape=(len(rbin), N))
    npix = S.count_nonzero(1)
    xybin = (S @ xy) / npix[:, None]

    if callable(capacity_spec):
        groups = np.split(S.indices, S.indptr[1:-1])
        capacity = np.fromiter((capacity_spec(idx, *args) for idx in groups), float)
    else:
        capacity = S @ capacity_spec

    return xybin, npix, capacity, bin_num


class PowerBin:
    """
    PowerBin Class
    ==============

    Performs 2D adaptive spatial binning using Centroidal Power Diagrams.
    This implementation uses a high-performance multithreaded Rust engine.

    Parameters
    ----------
    xy: array_like of shape (npix, 2)
        Coordinates of the pixels to be binned.
    capacity_spec: callable or array_like of shape (npix,)
        Additive array or callable returning total capacity of indices.
    target_capacity: float
        The target capacity value for each bin.
    pixelsize: float, optional
        Pixel size in coordinate units. If None, estimated automatically.
    verbose: int, optional
        Verbosity level: 0 = silent, 1 = summary, 2 = detailed.
    regul: bool, optional
        Whether to perform iterative regularization (default: True).
    args: tuple, optional
        Additional positional arguments passed to capacity_spec if callable.
    maxiter: int, optional
        Maximum number of regularization iterations (default: 50).

    Attributes
    ----------
    xy: ndarray of shape (npix, 2)
    bin_num: ndarray of int of shape (npix,)
    pixel_capacity: ndarray of shape (npix,)
    bin_capacity: ndarray of shape (nbin,)
    xybin: ndarray of shape (nbin, 2)
    rbin: ndarray of shape (nbin,)
    npix: ndarray of int of shape (nbin,)
    single: ndarray of bool of shape (nbin,)
    rms_frac: float
    target_capacity: float
    pixelsize: float
    verbose: int
    args: tuple
    time_accretion: float
    time_regularization: float
    it: int
    """

    def __init__(
        self,
        xy: ArrayLike,
        capacity_spec: Union[Callable, ArrayLike],
        target_capacity: float,
        pixelsize: Optional[float] = None,
        verbose: int = 1,
        regul: bool = True,
        args: tuple = (),
        maxiter: int = 50,
    ) -> None:
        # --- Input Validation (100% parity with reference) ---
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(
                f"xy must be a 2D array-like with shape (npix, 2), but got shape {xy.shape}"
            )
        if xy.shape[0] == 0:
            raise ValueError("Input 'xy' cannot be empty.")
        if not np.all(np.isfinite(xy)):
            raise ValueError("xy must contain only finite values.")

        npix_in = xy.shape[0]
        if callable(capacity_spec):
            if np.ndim(capacity_spec([0, 1], *args)) != 0:
                raise ValueError("If 'capacity_spec' is a callable, it must return a single scalar number.")
        else:
            capacity_spec = np.asarray(capacity_spec, dtype=float)
            if capacity_spec.ndim != 1 or capacity_spec.shape[0] != npix_in:
                raise ValueError(f"'capacity_spec' must have shape ({npix_in},), but got {capacity_spec.shape}")
            if not np.all(np.isfinite(capacity_spec)):
                raise ValueError("If 'capacity_spec' is an array, it must contain only finite values.")

        if not isinstance(target_capacity, (int, float)) or target_capacity <= 0:
            raise ValueError("target_capacity must be a positive number.")

        if pixelsize is not None and (not isinstance(pixelsize, (int, float)) or pixelsize <= 0):
            raise ValueError("pixelsize, if provided, must be a positive number.")

        if not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer.")

        if not isinstance(args, tuple):
            raise TypeError("args must be a tuple.")

        if not isinstance(maxiter, int) or maxiter <= 0:
            raise ValueError("maxiter must be a positive integer.")
        # --- End Validation ---

        # Execute high-performance Rust core
        core = PowerBinCore(
            xy,
            capacity_spec,
            float(target_capacity),
            pixelsize=float(pixelsize) if pixelsize is not None else None,
            verbose=verbose,
            regul=bool(regul),
            args=args,
            maxiter=int(maxiter),
        )

        # Populate Python attributes as standard numpy arrays
        self.xy = np.asarray(core.xy, dtype=np.float64)
        self.capacity = capacity_spec
        self.target_capacity = float(target_capacity)
        self.pixelsize = float(core.pixelsize)
        self.verbose = int(verbose)
        self.args = tuple(args)

        self.bin_num = np.asarray(core.bin_num, dtype=int)
        self.xybin = np.asarray(core.xybin, dtype=np.float64)
        self.rbin = np.asarray(core.rbin, dtype=np.float64)
        self.bin_capacity = np.asarray(core.bin_capacity, dtype=np.float64)
        self.pixel_capacity = np.asarray(core.pixel_capacity, dtype=np.float64)
        self.npix = np.asarray(core.npix, dtype=int)
        self.single = np.asarray(core.single, dtype=bool)
        self.rms_frac = float(core.rms_frac)

        self.time_accretion = float(core.time_accretion)
        self.time_regularization = float(core.time_regularization)
        self.time_total = float(core.time_total)
        self.it = int(core.it)

        # Match exact console summary output of reference PowerBin
        if verbose >= 1:
            print(f"Bins: {self.rbin.size}; Single Pixels: {np.sum(self.single)}/{len(xy)}")
            print(f"Capacity Fractional RMS Scatter (%): {self.rms_frac:.2f}")
            print(f"Time Accretion: {self.time_accretion:.2f} s")
            if regul:
                print(f"Time Regularization (it={self.it}): {self.time_regularization:.2f} s")

    def plot(
        self,
        capacity_scale: str = "raw",
        ylabel: Optional[str] = None,
        ylim: Optional[tuple[float, float]] = None,
        magrange: float = 10.0,
        left_title: Optional[str] = None,
        abscissa: str = "radius",
        points_alpha: Optional[float] = None,
        rasterize_points: bool = True,
        legend_loc: str = "best",
    ) -> None:
        """
        Generates a two-panel diagnostic plot summarizing the binning results.
        """
        if capacity_scale not in ("raw", "sqrt"):
            raise ValueError("capacity_scale must be either 'raw' or 'sqrt'.")

        if ylabel is not None and not isinstance(ylabel, str):
            raise TypeError("ylabel, if provided, must be a string.")

        if ylim is not None:
            if not isinstance(ylim, (list, tuple)) or len(ylim) != 2 or \
               not all(isinstance(v, (int, float)) for v in ylim):
                raise ValueError("ylim must be a tuple or list of two numbers, e.g., (bottom, top).")

        if not isinstance(magrange, (int, float)) or magrange <= 0:
            raise ValueError("magrange must be a positive number.")

        if points_alpha is not None and (not isinstance(points_alpha, (int, float)) or not (0 <= points_alpha <= 1)):
            raise ValueError("points_alpha must be a float between 0 and 1.")

        if not isinstance(rasterize_points, bool):
            raise TypeError("rasterize_points must be a boolean.")

        if not isinstance(legend_loc, str):
            raise TypeError("legend_loc must be a string.")

        # Scaled to pixel units
        xy = self.xy / self.pixelsize
        xybin = self.xybin / self.pixelsize
        rbin = self.rbin / self.pixelsize
        pixel_capacity = self.pixel_capacity.copy()
        bin_capacity = self.bin_capacity.copy()
        target_capacity = self.target_capacity
        rms_frac = self.rms_frac
        single = self.single

        if capacity_scale == "sqrt":
            pixel_capacity = np.sqrt(pixel_capacity)
            bin_capacity = np.sqrt(bin_capacity)
            target_capacity = np.sqrt(target_capacity)
            non_single = bin_capacity[~single]
            rms_frac = np.std(non_single, ddof=1) / np.mean(non_single) * 100 if len(non_single) > 1 else 0.0

        if ylabel is None:
            ylabel = "Capacity" if capacity_scale == "raw" else r"$\sqrt{\mathrm{Capacity}}$"

        rx, ry = np.ptp(xy, axis=0)
        rx = max(rx, 1e-4)
        ry = max(ry, 1e-4)
        _, (ax0, ax1) = plt.subplots(1, 2, width_ratios=[3 / 4, ry / rx], layout="constrained")
        ax1.set_box_aspect(3 / 4)

        # Left panel: Power Diagram
        plt.sca(ax0)
        plot_power_diagram(xy, pixel_capacity, self.bin_num, xybin, rbin, self.npix, magrange)
        ax0.set_title(left_title if left_title is not None else "Centroidal Power Diagram")
        ax0.set_xlabel('X (pixels)')
        ax0.set_ylabel('Y (pixels)')

        # Right panel: Capacity vs. Radius
        plt.sca(ax1)
        if abscissa == "radius":
            x_pix = np.hypot(*xy.T)
            x_bin = np.hypot(*xybin.T)
            xlabel = 'R (pixels)'
            x_left = -0.5
            x_right = np.max(x_pix)
        elif abscissa == "x":
            x_pix = xy[:, 0]
            x_bin = xybin[:, 0]
            xlabel = 'X (pixels)'
            x_left, x_right = np.min(x_pix), np.max(x_pix)
        else:  # "y"
            x_pix = xy[:, 1]
            x_bin = xybin[:, 1]
            xlabel = 'Y (pixels)'
            x_left, x_right = np.min(x_pix), np.max(x_pix)

        if points_alpha is None:
            points_alpha = min(46.0 / (len(x_pix) ** 0.67), 1.0)

        ax1.plot(x_pix, pixel_capacity, '.k', alpha=points_alpha, markeredgewidth=0,
                 label='Input', rasterized=rasterize_points)
        if np.sum(single) > 0:
            ax1.plot(x_bin[single], bin_capacity[single], 'xb', markersize=3, label='Single')
        ax1.plot(x_bin[~single], bin_capacity[~single], 'or', markersize=4.2, markeredgewidth=0, label='Bins')
        ax1.plot(x_bin[~single], bin_capacity[~single], 'ok', markersize=1, markeredgewidth=0)
        ax1.axhline(target_capacity, linestyle='--', linewidth=1, color='gray')
        ax1.axis([x_left, x_right, np.min(pixel_capacity), np.max(bin_capacity) * 1.5])

        if ylim is not None:
            ax1.set_ylim(ylim)

        ax1.set_title(rf'Fractional RMS Scatter $\sigma={rms_frac:.1f}$ %')
        ax1.set_yscale('asinh')
        format_asinh_axis(ax1)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)

        leg = ax1.legend(loc=legend_loc, handletextpad=0, labelspacing=0)
        if leg.legend_handles:
            leg.legend_handles[0].set_alpha(0.5)
