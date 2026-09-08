use powerbin_rs::{CapacitySpec, PowerBinConfig, powerbin};

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
    let config_neg = PowerBinConfig {
        target_capacity: -5.0,
        ..Default::default()
    };

    let res = powerbin(&xy, CapacitySpec::Additive(&dens), &config_neg);
    assert!(res.is_err(), "Negative target capacity should return error");

    let config_zero = PowerBinConfig {
        target_capacity: 0.0,
        ..Default::default()
    };
    let res = powerbin(&xy, CapacitySpec::Additive(&dens), &config_zero);
    assert!(res.is_err(), "Zero target capacity should return error");
}

#[test]
fn test_error_all_pixels_exceed_target() {
    let xy = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let dens = vec![100.0, 100.0, 100.0, 100.0];
    let config = PowerBinConfig {
        target_capacity: 10.0,
        ..Default::default()
    };

    let res = powerbin(&xy, CapacitySpec::Additive(&dens), &config);
    assert!(
        res.is_err(),
        "All pixels > target should return error indicating binning not needed"
    );
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
    assert!(
        (res.xybin.len() as f64 - 25.0).abs() <= 3.0,
        "Should produce ~25 bins, got {}",
        res.xybin.len()
    );
    assert!(
        res.rms_frac < 15.0,
        "RMS fraction should be low for uniform grid"
    );
}

#[test]
fn test_power_diagram_direct() {
    let xy = vec![[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]];
    let xybin = vec![[0.5, 0.0], [10.5, 10.0]];
    let rbin = vec![1.0, 1.0];

    let bins = powerbin_rs::power_diagram(&xy, &xybin, &rbin);
    assert_eq!(bins, vec![0, 0, 1, 1]);
}

#[test]
fn test_update_bins_direct() {
    let xy = vec![[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 10.0]];
    let xybin = vec![[0.0, 0.0], [10.0, 10.0]];
    let rbin = vec![1.0, 1.0];
    let dens = vec![5.0, 15.0, 20.0, 30.0];

    let (new_xybin, npix, capacity, bin_num) =
        powerbin_rs::update_bins(&xy, &xybin, &rbin, &dens, None);

    assert_eq!(npix, vec![2, 2]);
    assert_eq!(bin_num, vec![0, 0, 1, 1]);
    assert_eq!(capacity, vec![20.0, 50.0]);
    assert!((new_xybin[0][0] - 1.0).abs() < 1e-5);
    assert!((new_xybin[1][0] - 11.0).abs() < 1e-5);
}

#[test]
fn test_high_aspect_ratio_grid() {
    let mut xy = Vec::new();
    let mut dens = Vec::new();
    for y in 0..50 {
        for x in 0..10 {
            xy.push([x as f64, y as f64]);
            dens.push(1.0);
        }
    }
    let config = PowerBinConfig {
        target_capacity: 25.0, // 500 pixels / 25 = 20 bins
        pixelsize: Some(1.0),
        verbose: 0,
        regul: true,
        maxiter: 30,
    };
    let res = powerbin(&xy, CapacitySpec::Additive(&dens), &config).expect("aspect ratio run");
    assert!((res.xybin.len() as f64 - 20.0).abs() <= 3.0);
    assert_eq!(res.bin_num.len(), 500);
}

#[test]
fn test_single_bin_large_target() {
    let mut xy = Vec::new();
    let mut dens = Vec::new();
    for y in 0..5 {
        for x in 0..5 {
            xy.push([x as f64, y as f64]);
            dens.push(2.0);
        }
    }
    let config = PowerBinConfig {
        target_capacity: 45.0, // Total flux is 50.0 >= 0.8 * 45.0, so all accretes to 1 bin
        pixelsize: Some(1.0),
        verbose: 0,
        regul: true,
        maxiter: 10,
    };
    let res = powerbin(&xy, CapacitySpec::Additive(&dens), &config).expect("single bin run");
    assert_eq!(res.xybin.len(), 1);
    assert_eq!(res.bin_num.len(), 25);
    assert_eq!(res.npix[0], 25);
    assert_eq!(res.bin_capacity[0], 50.0);
}
