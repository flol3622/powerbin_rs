import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from powerbin_rs import PowerBin

# Output directories
ARTIFACT_DIR = "/Users/psoubrie/.gemini/antigravity-cli/brain/dbd23c97-d4f1-49c6-9cf2-834d78fe8423"
PLOTS_DIR = "gradient_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def create_grid(size=120, extent=60.0):
    """Create square grid of (x, y) coordinates centered at 0."""
    coords = np.linspace(-extent, extent, size)
    xx, yy = np.meshgrid(coords, coords)
    x = xx.ravel()
    y = yy.ravel()
    return x, y, xx, yy, size

def generate_gradients():
    """Generates 6 distinct gradient datasets."""
    datasets = {}
    
    # 1. Linear Ramp Gradient
    x, y, xx, yy, size = create_grid(size=120, extent=50.0)
    # Ramp from bottom-left to top-right
    ramp = (x - x.min() + y - y.min()) / (np.ptp(x) + np.ptp(y))
    signal = 2.0 + 98.0 * (ramp ** 1.5)
    noise = np.sqrt(signal)
    datasets["linear_ramp"] = {
        "name": "Linear Ramp Gradient",
        "description": "Smooth 2D directional linear/polynomial slope across field",
        "x": x, "y": y, "signal": signal, "noise": noise,
        "target_sn": 38.0, "abscissa": "x",
        "extent": [-50, 50, -50, 50],
    }

    # 2. Radial Exponential Disk Gradient (Sersic n=1)
    x, y, xx, yy, size = create_grid(size=120, extent=50.0)
    r = np.hypot(x, y)
    r_d = 12.0
    signal = 100.0 * np.exp(-r / r_d) + 1.0
    noise = np.sqrt(signal)
    datasets["exponential_disk"] = {
        "name": "Exponential Disk (Sérsic n=1)",
        "description": "Smooth exponential radial decay typical of galactic disks",
        "x": x, "y": y, "signal": signal, "noise": noise,
        "target_sn": 30.0, "abscissa": "radius",
        "extent": [-50, 50, -50, 50],
    }

    # 3. Steep Core / Cusp Gradient (Sersic n=4 de Vaucouleurs)
    x, y, xx, yy, size = create_grid(size=120, extent=50.0)
    r = np.hypot(x, y)
    r_e = 15.0
    b4 = 7.669
    # Steep cusp
    signal = 800.0 * np.exp(-b4 * ((np.maximum(r, 0.5) / r_e)**(1.0/4.0) - 1.0)) + 0.5
    noise = np.sqrt(signal)
    datasets["steep_cusp"] = {
        "name": "Steep Core Cusp (Sérsic n=4)",
        "description": "Extreme dynamic range central cusp (elliptical galaxy profile)",
        "x": x, "y": y, "signal": signal, "noise": noise,
        "target_sn": 65.0, "abscissa": "radius",
        "extent": [-50, 50, -50, 50],
    }

    # 4. Multi-Source / Galaxy Group Gradient
    x, y, xx, yy, size = create_grid(size=130, extent=65.0)
    # Background sky + 4 separate sources
    sky_bg = 15.0
    s_tot = np.full_like(x, sky_bg)
    # Source 1 (Main galaxy)
    s_tot += 300.0 * np.exp(-np.hypot(x + 18, y + 15) / 9.0)
    # Source 2 (Secondary companion)
    s_tot += 180.0 * np.exp(-np.hypot(x - 22, y + 20) / 7.0)
    # Source 3 (Dwarf 1)
    s_tot += 100.0 * np.exp(-np.hypot(x + 20, y - 24) / 5.5)
    # Source 4 (Dwarf 2)
    s_tot += 70.0 * np.exp(-np.hypot(x - 25, y - 20) / 4.5)
    noise = np.sqrt(s_tot)
    datasets["galaxy_group"] = {
        "name": "Galaxy Group (Multi-Peak)",
        "description": "Four disparate brightness peaks on background-limited field",
        "x": x, "y": y, "signal": s_tot, "noise": noise,
        "target_sn": 30.0, "abscissa": "radius",
        "extent": [-65, 65, -65, 65],
    }

    # 5. Annular / Ring Gradient (Non-Monotonic)
    x, y, xx, yy, size = create_grid(size=120, extent=50.0)
    r = np.hypot(x, y)
    r_ring = 22.0
    sigma_ring = 5.5
    # Central hole, bright ring, faint outskirts
    signal = 80.0 * np.exp(-0.5 * ((r - r_ring) / sigma_ring)**2) + 2.0 + 5.0 * np.exp(-r / 8.0)
    noise = np.sqrt(signal)
    datasets["annular_ring"] = {
        "name": "Annular Ring (Gradient Inversion)",
        "description": "Non-monotonic gradient with central hole and bright surrounding torus",
        "x": x, "y": y, "signal": signal, "noise": noise,
        "target_sn": 28.0, "abscissa": "radius",
        "extent": [-50, 50, -50, 50],
    }

    # 6. Two-Arm Spiral Galaxy Gradient
    x, y, xx, yy, size = create_grid(size=120, extent=50.0)
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    # Exponential disk
    disk = 80.0 * np.exp(-r / 14.0)
    # Logarithmic spiral arms: theta - k*ln(r)
    k_spiral = 2.2
    arm_modulation = 0.7 * np.cos(2 * theta - k_spiral * np.log(np.maximum(r, 2.0) / 5.0))
    signal = disk * (1.0 + np.clip(arm_modulation, -0.6, 1.0)) + 1.5
    noise = np.sqrt(signal)
    datasets["spiral_arms"] = {
        "name": "Two-Arm Spiral Galaxy",
        "description": "Curvilinear anisotropic gradient tracing spiral arm features",
        "x": x, "y": y, "signal": signal, "noise": noise,
        "target_sn": 32.0, "abscissa": "radius",
        "extent": [-50, 50, -50, 50],
    }

    return datasets

def plot_single_gradient_suite(key, data, pb):
    """Produces a clean, detailed 3-panel figure for an individual gradient."""
    x = data["x"]
    y = data["y"]
    signal = data["signal"]
    target_sn = data["target_sn"]
    abscissa = data["abscissa"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    # Panel 1: Input Gradient (Flux/Signal)
    ax0 = axes[0]
    sc0 = ax0.scatter(x, y, c=np.log10(np.maximum(signal, 1e-3)), cmap='inferno', s=12, marker='s', edgecolors='none')
    cb0 = fig.colorbar(sc0, ax=ax0, shrink=0.85, pad=0.03)
    cb0.set_label(r'$\log_{10}(\mathrm{Signal})$', fontsize=11)
    ax0.set_title(f"Input Signal: {data['name']}", fontsize=12, fontweight='bold')
    ax0.set_xlabel("X (pixels)", fontsize=11)
    ax0.set_ylabel("Y (pixels)", fontsize=11)
    ax0.set_aspect('equal')

    # Panel 2: PowerBin Tessellation & Generators
    ax1 = axes[1]
    rng = np.random.default_rng(42)
    rand_colors = rng.permutation(len(pb.rbin))
    pix_colors = rand_colors[pb.bin_num]
    sc1 = ax1.scatter(x, y, c=pix_colors, cmap='Set3', s=12, marker='s', edgecolors='none')
    
    # Overlay generator soap bubbles
    xybin = pb.xybin
    rbin = pb.rbin
    single = pb.single
    diam = 2 * rbin[~single]
    lw = 0.6 * (diam / np.maximum(np.max(diam), 1e-5))**0.3
    circles = collections.EllipseCollection(
        diam, diam, 0, offsets=xybin[~single], units='xy',
        facecolor='none', edgecolors='black', lw=lw, transOffset=ax1.transData
    )
    ax1.add_collection(circles)

    # Plot generator centroid markers
    ax1.plot(xybin[:, 0], xybin[:, 1], 'k.', markersize=2.0, alpha=0.8)

    ax1.set_title(f"PowerBin Tessellation ({len(pb.rbin)} bins)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("X (pixels)", fontsize=11)
    ax1.set_ylabel("Y (pixels)", fontsize=11)
    ax1.set_aspect('equal')

    # Panel 3: Capacity / S/N Uniformity vs Coordinate
    ax2 = axes[2]
    if abscissa == "radius":
        r_pix = np.hypot(x, y)
        r_bin = np.hypot(pb.xybin[:, 0], pb.xybin[:, 1])
        coord_label = "Radius R (pixels)"
    else:
        r_pix = x
        r_bin = pb.xybin[:, 0]
        coord_label = "X coordinate (pixels)"

    pix_sn = signal / data["noise"]
    bin_sn = np.sqrt(pb.bin_capacity)

    ax2.plot(r_pix, pix_sn, '.', color='#bdc3c7', alpha=0.35, markersize=3, label='Pixel S/N', rasterized=True)
    if np.sum(pb.single) > 0:
        ax2.plot(r_bin[pb.single], bin_sn[pb.single], 'x', color='#2980b9', markersize=5, label=f'Single pixels ({np.sum(pb.single)})')
    ax2.plot(r_bin[~pb.single], bin_sn[~pb.single], 'o', color='#e74c3c', markersize=4.5, label='Binned cells')
    ax2.axhline(target_sn, color='#2c3e50', linestyle='--', lw=1.5, label=f'Target S/N = {target_sn}')

    ax2.set_xlabel(coord_label, fontsize=11)
    ax2.set_ylabel("Signal-to-Noise Ratio (S/N)", fontsize=11)
    ax2.set_title(rf"Capacity Enforcement: RMS Scatter = {pb.rms_frac:.2f}%", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='best', frameon=True, fontsize=9)

    out_file = os.path.join(PLOTS_DIR, f"gradient_{key}.png")
    fig.savefig(out_file, dpi=180)
    plt.close(fig)

    shutil.copy(out_file, os.path.join(ARTIFACT_DIR, f"gradient_{key}.png"))
    print(f"Saved {out_file} (Bins: {len(pb.rbin)}, Singles: {np.sum(pb.single)}, RMS: {pb.rms_frac:.2f}%)")
    return out_file

def plot_master_summary(datasets, pb_results):
    """Creates a comprehensive 6x3 master figure showing ALL gradient images side-by-side."""
    fig, axes = plt.subplots(6, 3, figsize=(17, 24), constrained_layout=True)

    row = 0
    for key, data in datasets.items():
        pb = pb_results[key]
        x = data["x"]
        y = data["y"]
        signal = data["signal"]
        target_sn = data["target_sn"]
        abscissa = data["abscissa"]

        # 1. Input Image
        ax0 = axes[row, 0]
        sc0 = ax0.scatter(x, y, c=np.log10(np.maximum(signal, 1e-3)), cmap='magma', s=9, marker='s', edgecolors='none')
        ax0.set_title(f"{data['name']}\n(Input Flux Gradient)", fontsize=11, fontweight='bold')
        ax0.set_aspect('equal')
        ax0.tick_params(labelsize=8)

        # 2. PowerBin Tessellation
        ax1 = axes[row, 1]
        rng = np.random.default_rng(row * 100 + 42)
        rand_colors = rng.permutation(len(pb.rbin))
        pix_colors = rand_colors[pb.bin_num]
        ax1.scatter(x, y, c=pix_colors, cmap='Set3', s=9, marker='s', edgecolors='none')

        # Circles
        xybin = pb.xybin
        rbin = pb.rbin
        single = pb.single
        diam = 2 * rbin[~single]
        lw = 0.5 * (diam / np.maximum(np.max(diam), 1e-5))**0.3
        circles = collections.EllipseCollection(
            diam, diam, 0, offsets=xybin[~single], units='xy',
            facecolor='none', edgecolors='black', lw=lw, transOffset=ax1.transData
        )
        ax1.add_collection(circles)
        ax1.plot(xybin[:, 0], xybin[:, 1], 'k.', markersize=1.5, alpha=0.8)
        ax1.set_title(f"PowerBin Regularized Tessellation\n({len(pb.rbin)} bins, {np.sum(pb.single)} singles)", fontsize=11, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.tick_params(labelsize=8)

        # 3. Capacity / SN scatter
        ax2 = axes[row, 2]
        if abscissa == "radius":
            r_pix = np.hypot(x, y)
            r_bin = np.hypot(pb.xybin[:, 0], pb.xybin[:, 1])
            coord_label = "R (pixels)"
        else:
            r_pix = x
            r_bin = pb.xybin[:, 0]
            coord_label = "X (pixels)"

        pix_sn = signal / data["noise"]
        bin_sn = np.sqrt(pb.bin_capacity)

        ax2.plot(r_pix, pix_sn, '.', color='#bdc3c7', alpha=0.25, markersize=2.5, label='Pixel S/N', rasterized=True)
        if np.sum(pb.single) > 0:
            ax2.plot(r_bin[pb.single], bin_sn[pb.single], 'x', color='#2980b9', markersize=3.5, label='Single')
        ax2.plot(r_bin[~pb.single], bin_sn[~pb.single], 'o', color='#e74c3c', markersize=3.5, label='Bins')
        ax2.axhline(target_sn, color='#2c3e50', linestyle='--', lw=1.2, label=f'Target={target_sn}')
        ax2.set_title(rf"Target Capacity Enforcement ($\sigma = {pb.rms_frac:.2f}\%$)", fontsize=11, fontweight='bold')
        ax2.set_xlabel(coord_label, fontsize=9)
        ax2.set_ylabel("S/N", fontsize=9)
        ax2.tick_params(labelsize=8)
        ax2.grid(True, linestyle=':', alpha=0.5)
        if row == 0:
            ax2.legend(loc='best', frameon=True, fontsize=7.5)

        row += 1

    out_file = "all_gradients_summary.png"
    fig.savefig(out_file, dpi=160)
    plt.close(fig)

    shutil.copy(out_file, os.path.join(ARTIFACT_DIR, out_file))
    print(f"\nMaster summary plot saved to {out_file} and copied to artifact dir.")
    return out_file

def main():
    print("Generating 6 gradient synthetic images...")
    datasets = generate_gradients()
    pb_results = {}

    for key, data in datasets.items():
        print(f"\nProcessing '{key}' ({data['name']})...")
        x = data["x"]
        y = data["y"]
        signal = data["signal"]
        noise = data["noise"]
        target_sn = data["target_sn"]

        xy = np.column_stack([x, y])
        cap = (signal / noise)**2
        target_capacity = target_sn ** 2

        # Run Rust PowerBin
        pb = PowerBin(xy, cap, target_capacity=target_capacity, regul=True, verbose=0)
        pb_results[key] = pb

        # Save single plot
        plot_single_gradient_suite(key, data, pb)

    # Master plot showing ALL of them
    print("\nCreating Master Overview Plot for ALL gradients...")
    plot_master_summary(datasets, pb_results)

if __name__ == "__main__":
    main()
