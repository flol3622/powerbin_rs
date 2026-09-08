use powerbin_rs::{estimate_pixelsize, powerbin, CapacitySpec, PowerBinConfig};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn load_ngc2273() -> (Vec<[f64; 2]>, Vec<f64>) {
    let file = File::open(".venv/lib/python3.14/site-packages/powerbin/examples/sample_data_ngc2273.txt")
        .expect("Failed to open sample_data_ngc2273.txt");
    let reader = BufReader::new(file);
    let mut xy = Vec::new();
    let mut dens = Vec::new();

    for line in reader.lines() {
        let line = line.expect("line");
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            let x: f64 = parts[0].parse().unwrap();
            let y: f64 = parts[1].parse().unwrap();
            let sig: f64 = parts[2].parse().unwrap();
            let noise: f64 = parts[3].parse().unwrap();
            xy.push([x, y]);
            dens.push((sig / noise) * (sig / noise));
        }
    }
    (xy, dens)
}

#[test]
fn test_pixelsize_estimation() {
    let (xy, _) = load_ngc2273();
    let ps = estimate_pixelsize(&xy);
    assert!((ps - 0.799908).abs() < 1e-4, "Estimated pixelsize {} should be close to 0.799908", ps);
}

#[test]
fn test_powerbin_ngc2273_additive() {
    let (xy, dens) = load_ngc2273();
    let config = PowerBinConfig {
        target_capacity: 2500.0, // 50^2
        pixelsize: None,
        verbose: 0,
        regul: true,
        maxiter: 50,
    };

    let result = powerbin(&xy, CapacitySpec::Additive(&dens), &config).expect("powerbin run");
    assert_eq!(result.bin_num.len(), xy.len());
    assert_eq!(result.xybin.len(), 378, "Should produce 378 bins on NGC 2273");
    assert_eq!(result.rbin.len(), 378);
    assert_eq!(result.bin_capacity.len(), 378);

    let single_count = result.single.iter().filter(|&&s| s).count();
    assert_eq!(single_count, 105, "Should have 105 single-pixel bins");
    assert!((result.rms_frac - 13.58).abs() < 0.2, "RMS frac {} should match Python reference ~13.58%", result.rms_frac);
}

#[test]
fn test_powerbin_ngc2273_accretion_only() {
    let (xy, dens) = load_ngc2273();
    let config = PowerBinConfig {
        target_capacity: 2500.0,
        pixelsize: None,
        verbose: 0,
        regul: false, // No regularization
        maxiter: 50,
    };

    let result = powerbin(&xy, CapacitySpec::Additive(&dens), &config).expect("powerbin run");
    assert_eq!(result.xybin.len(), 378);
    assert_eq!(result.iterations, 0);
}

#[test]
fn test_powerbin_custom_callable() {
    let (xy, dens) = load_ngc2273();
    let custom_fn = |idx: &[usize]| -> f64 {
        idx.iter().map(|&i| dens[i]).sum()
    };

    let config = PowerBinConfig {
        target_capacity: 2500.0,
        pixelsize: None,
        verbose: 0,
        regul: true,
        maxiter: 50,
    };

    let result = powerbin(&xy, CapacitySpec::Custom(&custom_fn), &config).expect("powerbin custom callable");
    assert_eq!(result.xybin.len(), 378);
    assert!((result.rms_frac - 13.58).abs() < 0.2);
}
