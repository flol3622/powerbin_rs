pub mod accretion;
pub mod geometry;
pub mod regularization;
pub mod stopper;

#[cfg(feature = "python")]
pub mod python;

use std::time::Instant;

pub use accretion::{bin_accretion_impl, reassign_bad_bins};
pub use geometry::{build_delaunay_adjacency, estimate_pixelsize};
pub use regularization::{power_diagram, regularization_impl, update_bins};
pub use stopper::EarlyStopper;

/// Function signature for evaluating custom, non-additive capacities of a bin.
pub type CapacityFn<'a> = &'a (dyn Fn(&[usize]) -> f64 + 'a);

/// Capacity specification: either additive per-pixel density or custom callback.
pub enum CapacitySpec<'a> {
    /// Additive pixel capacities (e.g. S/N squared for Poisson noise).
    Additive(&'a [f64]),
    /// General non-additive callable function returning capacity of a subset of pixels.
    Custom(CapacityFn<'a>),
}

/// Configuration options for the PowerBin solver.
#[derive(Debug, Clone)]
pub struct PowerBinConfig {
    /// Target capacity per bin (e.g. target_sn^2).
    pub target_capacity: f64,
    /// Pixel size in coordinate units. If None, estimated automatically.
    pub pixelsize: Option<f64>,
    /// Verbosity level: 0 = silent, 1 = summary, 2 = iter-by-iter.
    pub verbose: usize,
    /// Whether to perform Centroidal Power Diagram regularization (default: true).
    pub regul: bool,
    /// Maximum number of regularization iterations (default: 50).
    pub maxiter: usize,
}

impl Default for PowerBinConfig {
    fn default() -> Self {
        Self {
            target_capacity: 1.0,
            pixelsize: None,
            verbose: 1,
            regul: true,
            maxiter: 50,
        }
    }
}

/// Result of the PowerBin spatial binning algorithm.
#[derive(Debug, Clone)]
pub struct PowerBinResult {
    /// Bin index for each input pixel: shape (N,).
    pub bin_num: Vec<usize>,
    /// Coordinates of bin generators in original coordinate units: shape (M, 2).
    pub xybin: Vec<[f64; 2]>,
    /// Power radii of bin generators in original coordinate units: shape (M,).
    pub rbin: Vec<f64>,
    /// Measured capacity of each bin: shape (M,).
    pub bin_capacity: Vec<f64>,
    /// Pixel capacity of each input pixel: shape (N,).
    pub pixel_capacity: Vec<f64>,
    /// Number of pixels in each bin: shape (M,).
    pub npix: Vec<usize>,
    /// Boolean flag indicating if bin consists of a single pixel: shape (M,).
    pub single: Vec<bool>,
    /// Fractional root-mean-square scatter of bin capacities (%) for multi-pixel bins.
    pub rms_frac: f64,
    /// Estimated or provided pixel size.
    pub pixelsize: f64,
    /// Number of regularization iterations performed.
    pub iterations: usize,
    /// Time spent in bin accretion stage (seconds).
    pub time_accretion_sec: f64,
    /// Time spent in regularization stage (seconds).
    pub time_regularization_sec: f64,
    /// Total execution time (seconds).
    pub time_total_sec: f64,
}

/// Runs the PowerBin adaptive binning algorithm.
pub fn powerbin(
    xy: &[[f64; 2]],
    capacity_spec: CapacitySpec,
    config: &PowerBinConfig,
) -> Result<PowerBinResult, String> {
    let npix_total = xy.len();
    if npix_total == 0 {
        return Err("Input 'xy' cannot be empty.".to_string());
    }
    if config.target_capacity <= 0.0 || !config.target_capacity.is_finite() {
        return Err("target_capacity must be a positive finite number.".to_string());
    }

    let t_start = Instant::now();

    // Determine pixel capacity
    let (dens, custom_fn): (Vec<f64>, Option<CapacityFn>) = match capacity_spec {
        CapacitySpec::Additive(slice) => {
            if slice.len() != npix_total {
                return Err(format!(
                    "Capacity array length {} must match xy length {}",
                    slice.len(),
                    npix_total
                ));
            }
            (slice.to_vec(), None)
        }
        CapacitySpec::Custom(func) => {
            let mut d = Vec::with_capacity(npix_total);
            for i in 0..npix_total {
                d.push(func(&[i]));
            }
            (d, Some(func))
        }
    };

    for &val in &dens {
        if !val.is_finite() {
            return Err("Capacity values must be finite.".to_string());
        }
    }
    if dens.iter().all(|&d| d > config.target_capacity) {
        return Err("All pixels have capacity > target and binning is not needed.".to_string());
    }

    // Determine pixelsize
    let pixelsize = match config.pixelsize {
        Some(ps) => {
            if ps <= 0.0 || !ps.is_finite() {
                return Err("pixelsize must be a positive finite number.".to_string());
            }
            ps
        }
        None => estimate_pixelsize(xy),
    };

    // Normalize coordinates to pixel units
    let mut norm_xy = Vec::with_capacity(npix_total);
    for p in xy {
        norm_xy.push([p[0] / pixelsize, p[1] / pixelsize]);
    }

    // Stage 1: Bin Accretion
    let t1 = Instant::now();
    let (xybin_norm, pixel_capacity, _bin_num_init) = bin_accretion_impl(
        &norm_xy,
        &dens,
        config.target_capacity,
        custom_fn,
        config.verbose,
    );
    let t2 = Instant::now();
    let time_accretion = (t2 - t1).as_secs_f64();

    // Stage 2: Regularization
    let (final_bin_num, final_xybin_norm, final_rbin_norm, bin_capacity, npix, it) = if config.regul
    {
        if config.verbose >= 1 {
            println!("Regularization...");
        }
        regularization_impl(
            &norm_xy,
            xybin_norm,
            &dens,
            config.target_capacity,
            custom_fn,
            config.verbose,
            config.maxiter,
        )
    } else {
        let m = xybin_norm.len();
        let rbin_zero = vec![0.0; m];
        let (updated_xybin, updated_npix, updated_cap, updated_bin_num) =
            update_bins(&norm_xy, &xybin_norm, &rbin_zero, &dens, custom_fn);
        let mut rbin = Vec::with_capacity(m);
        for &cnt in &updated_npix {
            rbin.push((cnt as f64 / std::f64::consts::PI).sqrt());
        }
        (
            updated_bin_num,
            updated_xybin,
            rbin,
            updated_cap,
            updated_npix,
            0,
        )
    };
    let t3 = Instant::now();
    let time_regularization = (t3 - t2).as_secs_f64();
    let time_total = (t3 - t_start).as_secs_f64();

    // Convert coordinates back to world units
    let mut final_xybin = Vec::with_capacity(final_xybin_norm.len());
    for p in &final_xybin_norm {
        final_xybin.push([p[0] * pixelsize, p[1] * pixelsize]);
    }
    let mut final_rbin = Vec::with_capacity(final_rbin_norm.len());
    for &r in &final_rbin_norm {
        final_rbin.push(r * pixelsize);
    }

    let m = final_rbin.len();
    let single: Vec<bool> = npix.iter().map(|&cnt| cnt == 1).collect();

    // RMS scatter of non-single bins
    let non_single_caps: Vec<f64> = (0..m)
        .filter(|&j| !single[j])
        .map(|j| bin_capacity[j])
        .collect();

    let rms_frac = if non_single_caps.len() > 1 {
        let count = non_single_caps.len() as f64;
        let mean = non_single_caps.iter().sum::<f64>() / count;
        let var = non_single_caps
            .iter()
            .map(|&c| (c - mean) * (c - mean))
            .sum::<f64>()
            / (count - 1.0);
        var.sqrt() / mean * 100.0
    } else {
        0.0
    };

    let single_sum: usize = single.iter().map(|&s| if s { 1 } else { 0 }).sum();

    if config.verbose >= 1 {
        println!("Bins: {}; Single Pixels: {}/{}", m, single_sum, npix_total);
        println!("Capacity Fractional RMS Scatter (%): {:.2}", rms_frac);
        println!("Time Accretion: {:.2} s", time_accretion);
        if config.regul {
            println!(
                "Time Regularization (it={}): {:.2} s",
                it, time_regularization
            );
        }
    }

    Ok(PowerBinResult {
        bin_num: final_bin_num,
        xybin: final_xybin,
        rbin: final_rbin,
        bin_capacity,
        pixel_capacity,
        npix,
        single,
        rms_frac,
        pixelsize,
        iterations: it,
        time_accretion_sec: time_accretion,
        time_regularization_sec: time_regularization,
        time_total_sec: time_total,
    })
}
