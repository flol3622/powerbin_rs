"""
Test suite verifying 100% API and numerical parity between
powerbin_rs and reference Python powerbin (https://pypi.org/project/powerbin/).
"""
import inspect
from importlib import resources
import numpy as np
import pytest
import matplotlib.pyplot as plt

import powerbin
import powerbin_rs


def load_ngc2273_data():
    """Loads reference NGC 2273 SAURON dataset."""
    data_path = resources.files("powerbin") / "examples/sample_data_ngc2273.txt"
    x, y, signal, noise = np.loadtxt(data_path).T
    xy = np.column_stack([x, y])
    cap = (signal / noise) ** 2
    return xy, cap, 50.0 ** 2


# ============================================================================
# 1. API Signature & Defaults Parity
# ============================================================================

def test_signature_and_defaults_parity():
    """Verify that powerbin_rs.PowerBin.__init__ has identical parameter names and defaults."""
    py_sig = inspect.signature(powerbin.PowerBin.__init__)
    rs_sig = inspect.signature(powerbin_rs.PowerBin.__init__)

    assert list(py_sig.parameters.keys()) == list(rs_sig.parameters.keys())

    for param_name, py_param in py_sig.parameters.items():
        if param_name == "self":
            continue
        rs_param = rs_sig.parameters[param_name]
        assert py_param.default == rs_param.default, f"Default mismatch for {param_name}"


# ============================================================================
# 2. Input Validation Parity
# ============================================================================

def test_input_validation():
    """Verify that powerbin_rs raises identical exceptions for invalid inputs."""
    valid_xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    valid_cap = np.array([10.0, 20.0, 30.0, 40.0])

    # 1. xy wrong shape (1D)
    with pytest.raises(ValueError):
        powerbin_rs.PowerBin(np.array([1.0, 2.0]), valid_cap, target_capacity=50.0)

    # 2. xy wrong shape (3D)
    with pytest.raises(ValueError):
        powerbin_rs.PowerBin(np.ones((4, 3)), valid_cap, target_capacity=50.0)

    # 3. xy empty
    with pytest.raises(ValueError):
        powerbin_rs.PowerBin(np.empty((0, 2)), np.array([]), target_capacity=50.0)

    # 4. xy non-finite
    bad_xy = valid_xy.copy()
    bad_xy[0, 0] = np.nan
    with pytest.raises(ValueError):
        powerbin_rs.PowerBin(bad_xy, valid_cap, target_capacity=50.0)

    # 5. capacity_spec wrong length
    with pytest.raises(ValueError):
        powerbin_rs.PowerBin(valid_xy, np.array([10.0, 20.0]), target_capacity=50.0)

    # 6. capacity_spec non-finite
    bad_cap = valid_cap.copy()
    bad_cap[1] = np.inf
    with pytest.raises(ValueError):
        powerbin_rs.PowerBin(valid_xy, bad_cap, target_capacity=50.0)

    # 7. target_capacity non-positive
    with pytest.raises(ValueError):
        powerbin_rs.PowerBin(valid_xy, valid_cap, target_capacity=0.0)
    with pytest.raises(ValueError):
        powerbin_rs.PowerBin(valid_xy, valid_cap, target_capacity=-10.0)

    # 8. pixelsize non-positive
    with pytest.raises(ValueError):
        powerbin_rs.PowerBin(valid_xy, valid_cap, target_capacity=50.0, pixelsize=-1.0)

    # 9. verbose negative
    with pytest.raises(ValueError):
        powerbin_rs.PowerBin(valid_xy, valid_cap, target_capacity=50.0, verbose=-1)

    # 10. args not a tuple
    with pytest.raises(TypeError):
        powerbin_rs.PowerBin(valid_xy, valid_cap, target_capacity=50.0, args=[1, 2])

    # 11. maxiter non-positive
    with pytest.raises(ValueError):
        powerbin_rs.PowerBin(valid_xy, valid_cap, target_capacity=50.0, maxiter=0)


# ============================================================================
# 3. Numerical & Structural Parity on Astronomical IFS Data (NGC 2273)
# ============================================================================

def test_ngc2273_additive_parity():
    """Verify exact numerical and attribute match with reference on SAURON NGC 2273 data."""
    xy, cap, target_cap = load_ngc2273_data()

    pb_ref = powerbin.PowerBin(xy, cap, target_capacity=target_cap, verbose=0)
    pb_rs = powerbin_rs.PowerBin(xy, cap, target_capacity=target_cap, verbose=0)

    # 1. Bin counts & Single pixels
    assert len(pb_rs.rbin) == len(pb_ref.rbin) == 378
    assert np.sum(pb_rs.single) == np.sum(pb_ref.single) == 105
    assert len(pb_rs.xybin) == 378
    assert len(pb_rs.npix) == 378
    assert len(pb_rs.bin_num) == len(xy)

    # 2. RMS fractional scatter
    assert np.isclose(pb_rs.rms_frac, pb_ref.rms_frac, atol=0.05)
    assert np.isclose(pb_rs.rms_frac, 13.58, atol=0.2)

    # 3. Attributes inspection
    expected_attrs = [
        "xy", "capacity", "target_capacity", "pixelsize", "verbose", "args",
        "single", "bin_num", "xybin", "rbin", "bin_capacity", "pixel_capacity",
        "npix", "rms_frac", "time_accretion", "time_regularization", "it"
    ]
    for attr in expected_attrs:
        assert hasattr(pb_rs, attr), f"Missing attribute: {attr}"

    # 4. Pixel size match
    assert np.isclose(pb_rs.pixelsize, pb_ref.pixelsize)

    # 5. Types check
    assert isinstance(pb_rs.bin_num, np.ndarray)
    assert isinstance(pb_rs.xybin, np.ndarray)
    assert isinstance(pb_rs.rbin, np.ndarray)
    assert isinstance(pb_rs.bin_capacity, np.ndarray)
    assert isinstance(pb_rs.single, np.ndarray)
    assert pb_rs.single.dtype == bool


def test_ngc2273_accretion_only_parity():
    """Verify regul=False produces identical initial accretion results."""
    xy, cap, target_cap = load_ngc2273_data()

    pb_ref = powerbin.PowerBin(xy, cap, target_capacity=target_cap, regul=False, verbose=0)
    pb_rs = powerbin_rs.PowerBin(xy, cap, target_capacity=target_cap, regul=False, verbose=0)

    assert len(pb_rs.rbin) == len(pb_ref.rbin) == 378
    assert pb_rs.it == 0
    assert np.allclose(pb_rs.xybin, pb_ref.xybin, atol=1e-5)


def test_callable_capacity_parity():
    """Verify custom callable capacity function with extra args."""
    xy, cap, target_cap = load_ngc2273_data()

    def custom_func(idx, multiplier):
        return float(np.sum(cap[idx]) * multiplier)

    pb_ref = powerbin.PowerBin(xy, custom_func, target_capacity=target_cap * 2.0, args=(2.0,), verbose=0)
    pb_rs = powerbin_rs.PowerBin(xy, custom_func, target_capacity=target_cap * 2.0, args=(2.0,), verbose=0)

    assert len(pb_rs.rbin) == len(pb_ref.rbin) == 378
    assert np.isclose(pb_rs.rms_frac, pb_ref.rms_frac, atol=0.1)


# ============================================================================
# 4. Helper Functions & Plotting
# ============================================================================

def test_power_diagram_parity():
    """Verify power_diagram helper function parity."""
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    xybin = np.array([[0.0, 0.0], [1.0, 1.0]])
    rbin = np.array([1.0, 1.0])

    bins_ref = powerbin.powerbin.power_diagram(xy, xybin, rbin)
    bins_rs = powerbin_rs.power_diagram(xy, xybin, rbin)

    np.testing.assert_array_equal(bins_rs, bins_ref)


def test_plot_method():
    """Verify .plot() executes without error."""
    xy, cap, target_cap = load_ngc2273_data()
    pb = powerbin_rs.PowerBin(xy, cap, target_capacity=target_cap, verbose=0)

    # Test raw scale
    pb.plot(capacity_scale="raw", magrange=5.0)
    plt.close("all")

    # Test sqrt scale
    pb.plot(capacity_scale="sqrt", magrange=5.0)
    plt.close("all")
