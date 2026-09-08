use powerbin_rs::{estimate_pixelsize, powerbin, CapacitySpec, PowerBinConfig};
use std::time::Instant;

const SAMPLE_DATA: &str = include_str!("../tests/sample_data_ngc2273.txt");

fn main() {
    println!("PowerBin (Rust) — Example on NGC 2273 SAURON data\n");

    let mut xy = Vec::new();
    let mut dens = Vec::new();

    for line in SAMPLE_DATA.lines() {
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

    let n = xy.len();
    println!("Loaded {} spatial pixels from NGC 2273", n);

    let target_sn = 50.0;
    let target_capacity = target_sn * target_sn;

    let ps = estimate_pixelsize(&xy);
    println!("Estimated pixelsize: {:.6}\"", ps);

    let config = PowerBinConfig {
        target_capacity,
        pixelsize: Some(ps),
        verbose: 1,
        regul: true,
        maxiter: 50,
    };

    let t0 = Instant::now();
    let res = powerbin(&xy, CapacitySpec::Additive(&dens), &config).expect("PowerBin execution failed");
    let dt = t0.elapsed();

    println!("\n=== Summary ===");
    println!("Final bins:           {}", res.xybin.len());
    let single_count = res.single.iter().filter(|&&s| s).count();
    println!("Single-pixel bins:    {}/{}", single_count, n);
    println!("Fractional RMS (%):   {:.2}%", res.rms_frac);
    println!("Regularization iters: {}", res.iterations);
    println!("Total Rust Time:      {:?}", dt);
}
