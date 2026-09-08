from typing import Any

import numpy as np
from numpy.typing import NDArray

class PowerBinCore:
    xy: NDArray[np.float64]
    bin_num: NDArray[np.int_]
    xybin: NDArray[np.float64]
    rbin: NDArray[np.float64]
    bin_capacity: NDArray[np.float64]
    pixel_capacity: NDArray[np.float64]
    npix: NDArray[np.int_]
    single: NDArray[np.bool_]
    rms_frac: float
    target_capacity: float
    pixelsize: float
    verbose: int
    args: tuple[Any, ...]
    time_accretion: float
    time_regularization: float
    time_total: float
    it: int

    def __init__(
        self,
        xy: Any,
        capacity_spec: Any,
        target_capacity: float,
        pixelsize: float | None = None,
        verbose: int = 1,
        regul: bool = True,
        args: tuple[Any, ...] = (),
        maxiter: int = 50,
    ) -> None: ...

def power_diagram(
    xy: Any,
    xybin: Any,
    rbin: Any,
) -> list[int]: ...
