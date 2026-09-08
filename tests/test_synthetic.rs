use powerbin_rs::{powerbin, CapacitySpec, PowerBinConfig};

#[test]
fn test_error_empty_input() {
    let xy: Vec<[f64; 2]> = Vec::new();
    let dens: Vec<f64> = Vec::new();
    let config = PowerBinConfig::default();

    let res = powerbin(&xy, CapacitySpec::Additive(&dens), &config);
    assert!(res.is_err(), "Empty input should return error");
}

#[test]
fn test_error_invalid_target_capacity() {
    let xy = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let dens = vec![1.0, 1.0, 1.0, 1.0];
    let mut config = PowerBinConfig::default();
    config.target_capacity = -5.0;

    let res = powerbin(&xy, CapacitySpec::Additive(&dens), &config);
    assert!(res.is_err(), "Negative target capacity should return error");

    config.target_capacity = 0.0;
    let res = powerbin(&xy, CapacitySpec::Additive(&dens), &config);
    assert!(res.is_err(), "Zero target capacity should return error");
}

#[test]
fn test_error_all_pixels_exceed_target() {
    let xy = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let dens = vec![100.0, 100.0, 100.0, 100.0];
    let mut config = PowerBinConfig::default();
    config.target_capacity = 10.0; // All pixels have capacity > target

    let res = powerbin(&xy, CapacitySpec::Additive(&dens), &config);
    assert!(res.is_err(), "All pixels > target should return error indicating binning not needed");
}

#[test]
fn test_uniform_grid_binning() {
    let mut xy = Vec::new();
    let mut dens = Vec::new();
    let w = 20;
    let h = 20;
    for y in 0..h {
        for x in 0..w {
            xy.push([x as f64, y as f64]);
            dens.push(1.0); // uniform capacity 1.0 per pixel
        }
    }

    let config = PowerBinConfig {
        target_capacity: 16.0, // Should partition 400 pixels into ~25 bins of 16 pixels each
        pixelsize: Some(1.0),
        verbose: 0,
        regul: true,
        maxiter: 30,
    };

    let res = powerbin(&xy, CapacitySpec::Additive(&dens), &config).expect("uniform grid run");
    assert!((res.xybin.len() as f64 - 25.0).abs() <= 3.0, "Should produce ~25 bins, got {}", res.xybin.len());
    assert!(res.rms_frac < 15.0, "RMS fraction should be low for uniform grid");
}
