use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use crate::{
    powerbin, power_diagram as rs_power_diagram,
    CapacitySpec, PowerBinConfig,
};

/// Core Rust computational engine for PowerBin.
#[pyclass(name = "PowerBinCore", subclass)]
pub struct PowerBinCore {
    #[pyo3(get)]
    pub xy: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub bin_num: Py<PyArray1<usize>>,
    #[pyo3(get)]
    pub xybin: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub rbin: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub bin_capacity: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub pixel_capacity: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub npix: Py<PyArray1<usize>>,
    #[pyo3(get)]
    pub single: Py<PyArray1<bool>>,
    #[pyo3(get)]
    pub rms_frac: f64,
    #[pyo3(get)]
    pub pixelsize: f64,
    #[pyo3(get)]
    pub target_capacity: f64,
    #[pyo3(get)]
    pub verbose: usize,
    #[pyo3(get)]
    pub it: usize,
    #[pyo3(get)]
    pub time_accretion: f64,
    #[pyo3(get)]
    pub time_regularization: f64,
    #[pyo3(get)]
    pub time_total: f64,
}

#[pymethods]
impl PowerBinCore {
    #[new]
    #[pyo3(signature = (xy, capacity_spec, target_capacity, pixelsize=None, verbose=1, regul=true, args=None, maxiter=50))]
    pub fn new(
        py: Python<'_>,
        xy: PyReadonlyArray2<f64>,
        capacity_spec: &Bound<'_, PyAny>,
        target_capacity: f64,
        pixelsize: Option<f64>,
        verbose: usize,
        regul: bool,
        args: Option<&Bound<'_, PyTuple>>,
        maxiter: usize,
    ) -> PyResult<Self> {
        let xy_view = xy.as_array();
        let npix_total = xy_view.shape()[0];
        let mut xy_vec = Vec::with_capacity(npix_total);
        for row in xy_view.rows() {
            xy_vec.push([row[0], row[1]]);
        }

        let config = PowerBinConfig {
            target_capacity,
            pixelsize,
            verbose,
            regul,
            maxiter,
        };

        let is_callable = capacity_spec.is_callable();
        let result = if is_callable {
            let empty_tuple = PyTuple::empty(py);
            let extra_args = args.unwrap_or(&empty_tuple);

            let cap_func = |indices: &[usize]| -> f64 {
                Python::attach(|py_inner| {
                    let py_indices = indices.to_vec();
                    let mut call_args = vec![py_indices.into_pyobject(py_inner).unwrap().into_any()];
                    for item in extra_args.iter() {
                        call_args.push(item);
                    }
                    let tuple_args = PyTuple::new(py_inner, call_args).unwrap();
                    let res = capacity_spec.call1(tuple_args).expect("capacity callable failed");
                    res.extract::<f64>().expect("capacity callable must return float")
                })
            };

            powerbin(&xy_vec, CapacitySpec::Custom(&cap_func), &config)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?
        } else {
            let array_cap: PyReadonlyArray1<f64> = capacity_spec.extract()?;
            let cap_slice = array_cap.as_slice()?;
            powerbin(&xy_vec, CapacitySpec::Additive(cap_slice), &config)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?
        };

        let py_xy = xy.to_owned_array().into_pyarray(py).unbind();
        let py_bin_num = PyArray1::from_vec(py, result.bin_num).unbind();

        let py_xybin = PyArray2::from_vec2(py, &result.xybin.iter().map(|p| vec![p[0], p[1]]).collect::<Vec<_>>())
            .unwrap()
            .unbind();

        let py_rbin = PyArray1::from_vec(py, result.rbin).unbind();
        let py_bin_cap = PyArray1::from_vec(py, result.bin_capacity).unbind();
        let py_pixel_cap = PyArray1::from_vec(py, result.pixel_capacity).unbind();
        let py_npix = PyArray1::from_vec(py, result.npix).unbind();
        let py_single = PyArray1::from_vec(py, result.single).unbind();

        Ok(PowerBinCore {
            xy: py_xy,
            bin_num: py_bin_num,
            xybin: py_xybin,
            rbin: py_rbin,
            bin_capacity: py_bin_cap,
            pixel_capacity: py_pixel_cap,
            npix: py_npix,
            single: py_single,
            rms_frac: result.rms_frac,
            pixelsize: result.pixelsize,
            target_capacity,
            verbose,
            it: result.iterations,
            time_accretion: result.time_accretion_sec,
            time_regularization: result.time_regularization_sec,
            time_total: result.time_total_sec,
        })
    }
}

#[pyfunction]
#[pyo3(signature = (xy, xybin, rbin))]
pub fn power_diagram<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<f64>,
    xybin: PyReadonlyArray2<f64>,
    rbin: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<usize>>> {
    let xy_view = xy.as_array();
    let xybin_view = xybin.as_array();
    let rbin_slice = rbin.as_slice()?;

    let mut xy_vec = Vec::with_capacity(xy_view.shape()[0]);
    for row in xy_view.rows() {
        xy_vec.push([row[0], row[1]]);
    }

    let mut xybin_vec = Vec::with_capacity(xybin_view.shape()[0]);
    for row in xybin_view.rows() {
        xybin_vec.push([row[0], row[1]]);
    }

    let bin_num = rs_power_diagram(&xy_vec, &xybin_vec, rbin_slice);
    Ok(PyArray1::from_vec(py, bin_num))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PowerBinCore>()?;
    m.add_function(wrap_pyfunction!(power_diagram, m)?)?;
    Ok(())
}
