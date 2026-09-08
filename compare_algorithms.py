import os
import shutil
import time
import warnings
from importlib import resources
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from scipy.spatial import ConvexHull

from vorbin.voronoi_2d_binning import voronoi_2d_binning
from powerbin import PowerBin as PyPowerBin
from powerbin_rs import PowerBin as RsPowerBin
from generate_gradient_tests import generate_gradients

warnings.filterwarnings('ignore')

ARTIFACT_DIR = "/Users/psoubrie/.gemini/antigravity-cli/brain/dbd23c97-d4f1-49c6-9cf2-834d78fe8423"
PLOTS_DIR = "algorithm_comparison_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def check_bin_convexity(x, y, bin_ids):
    """
    Evaluates convexity of discrete bins.
    For each bin with >= 4 pixels, computes 2D ConvexHull.
    Counts how many intruder pixels from OTHER bins lie within the convex hull.
    Returns (num_non_convex_bins, total_multi_pixel_bins).
    """
    unique_bins = np.unique(bin_ids)
    non_convex_count = 0
    multi_pixel_bins = 0

    # Grid lookup for fast point-in-hull checks
    xy = np.column_stack([x, y])
    
    for b in unique_bins:
        idx = np.where(bin_ids == b)[0]
        if len(idx) < 4:
            continue
        multi_pixel_bins += 1
        pts = xy[idx]
        try:
            hull = ConvexHull(pts)
            # Check bounding box first
            min_x, min_y = pts.min(axis=0)
            max_x, max_y = pts.max(axis=0)
            candidates = np.where(
                (x >= min_x) & (x <= max_x) & 
                (y >= min_y) & (y <= max_y) & 
                (bin_ids != b)
            )[0]
            if len(candidates) == 0:
                continue
            
            # Test candidate points against hull equations: Ax + By + C <= 1e-9
            cand_pts = xy[candidates]
            # hull.equations: [A, B, C] such that Ax + By + C <= 0 inside
            inside = np.all(cand_pts @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-7, axis=1)
            if np.any(inside):
                non_convex_count += 1
        except Exception:
            pass

    return non_convex_count, multi_pixel_bins

def run_benchmarks():
    datasets = generate_gradients()
    
    # Also add NGC 2273
    data_path = resources.files('powerbin') / 'examples/sample_data_ngc2273.txt'
    x_ngc, y_ngc, sig_ngc, noi_ngc = np.loadtxt(data_path).T
    datasets["ngc2273"] = {
        "name": "NGC 2273 (SAURON IFS)",
        "description": "Real astronomical observations from SAURON",
        "x": x_ngc, "y": y_ngc, "signal": sig_ngc, "noise": noi_ngc,
        "target_sn": 50.0, "abscissa": "radius",
    }

    results = []

    print("=" * 90)
    print("ALGORITHM COMPARISON BENCHMARK: VorBin vs PowerBin (Python) vs PowerBin (Rust)")
    print("=" * 90)

    for key, data in datasets.items():
        x = data["x"]
        y = data["y"]
        signal = data["signal"]
        noise = data["noise"]
        target_sn = data["target_sn"]
        xy = np.column_stack([x, y])
        cap = (signal / noise)**2
        target_cap = target_sn**2
        dx = np.min(np.diff(np.unique(x))) if len(np.unique(x)) > 1 else 1.0

        n_pts = len(x)
        print(f"\nEvaluating: {data['name']} (N = {n_pts:,} pixels, Target S/N = {target_sn})")

        # 1. Rust PowerBin Regularized
        # Warmup
        _ = RsPowerBin(xy, cap, target_capacity=target_cap, regul=True, verbose=0)
        t_runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            pb_rs_reg = RsPowerBin(xy, cap, target_capacity=target_cap, regul=True, verbose=0)
            t_runs.append(time.perf_counter() - t0)
        t_rs_reg = np.median(t_runs)

        # 2. Rust PowerBin Accretion-Only
        t_runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            pb_rs_acc = RsPowerBin(xy, cap, target_capacity=target_cap, regul=False, verbose=0)
            t_runs.append(time.perf_counter() - t0)
        t_rs_acc = np.median(t_runs)

        # 3. Python PowerBin Regularized
        t_runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            pb_py_reg = PyPowerBin(xy, cap, target_capacity=target_cap, regul=True, verbose=0)
            t_runs.append(time.perf_counter() - t0)
        t_py_reg = np.median(t_runs)

        # 4. Python PowerBin Accretion-Only
        t_runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            pb_py_acc = PyPowerBin(xy, cap, target_capacity=target_cap, regul=False, verbose=0)
            t_runs.append(time.perf_counter() - t0)
        t_py_acc = np.median(t_runs)

        # 5. Classic VorBin
        t0 = time.perf_counter()
        vb = voronoi_2d_binning(x, y, signal, noise, target_sn, pixelsize=dx, plot=False, quiet=True)
        t_vb = time.perf_counter() - t0
        vb_bin_ids = vb[0]
        vb_bins = len(np.unique(vb_bin_ids))

        # Calculate VorBin RMS scatter
        bin_sn_list = []
        for b in np.unique(vb_bin_ids):
            idx = np.where(vb_bin_ids == b)[0]
            bin_sn_list.append(np.sum(signal[idx]) / np.sqrt(np.sum(noise[idx]**2)))
        bin_sn_arr = np.array(bin_sn_list)
        npix_bin = np.bincount(vb_bin_ids)
        not_single = npix_bin[np.unique(vb_bin_ids)] > 1
        vb_rms = np.std(bin_sn_arr[not_single], ddof=1) / np.mean(bin_sn_arr[not_single]) * 100 if np.sum(not_single) > 1 else 0.0

        # Check non-convexity
        nc_vb, tot_vb = check_bin_convexity(x, y, vb_bin_ids)
        nc_rs_acc, tot_rs_acc = check_bin_convexity(x, y, pb_rs_acc.bin_num)
        nc_rs_reg, tot_rs_reg = check_bin_convexity(x, y, pb_rs_reg.bin_num)

        speedup_py = t_py_reg / t_rs_reg
        speedup_vb = t_vb / t_rs_reg

        res = {
            "key": key,
            "name": data["name"],
            "N": n_pts,
            "target_sn": target_sn,
            "t_rs_reg": t_rs_reg,
            "t_rs_acc": t_rs_acc,
            "t_py_reg": t_py_reg,
            "t_py_acc": t_py_acc,
            "t_vb": t_vb,
            "speedup_vs_py": speedup_py,
            "speedup_vs_vb": speedup_vb,
            "bins_rs_reg": len(pb_rs_reg.rbin),
            "bins_rs_acc": len(pb_rs_acc.rbin),
            "bins_vb": vb_bins,
            "singles_rs": int(np.sum(pb_rs_reg.single)),
            "rms_rs_reg": pb_rs_reg.rms_frac,
            "rms_rs_acc": pb_rs_acc.rms_frac,
            "rms_py_reg": pb_py_reg.rms_frac,
            "rms_vb": vb_rms,
            "nc_vb": nc_vb,
            "tot_vb": tot_vb,
            "nc_rs_reg": nc_rs_reg,
            "tot_rs_reg": tot_rs_reg,
            "nc_rs_acc": nc_rs_acc,
            "pb_rs_reg": pb_rs_reg,
            "pb_rs_acc": pb_rs_acc,
            "vb_bin_ids": vb_bin_ids,
            "x": x, "y": y, "signal": signal, "noise": noise,
        }
        results.append(res)

        print(f"  Rust PowerBin:     {t_rs_reg*1000:6.2f} ms | Bins: {len(pb_rs_reg.rbin):4d} | RMS: {pb_rs_reg.rms_frac:5.2f}% | Non-convex: {nc_rs_reg}/{tot_rs_reg}")
        print(f"  Python PowerBin:   {t_py_reg*1000:6.2f} ms | Bins: {len(pb_py_reg.rbin):4d} | RMS: {pb_py_reg.rms_frac:5.2f}% | Speedup: {speedup_py:5.1f}x")
        print(f"  Classic VorBin:    {t_vb*1000:6.2f} ms | Bins: {vb_bins:4d} | RMS: {vb_rms:5.2f}% | Non-convex: {nc_vb}/{tot_vb} | Speedup: {speedup_vb:5.1f}x")

    return results

def plot_algorithm_timing_comparison(results):
    """Generates grouped bar charts comparing execution times and speedups."""
    names = [r["name"].replace(" (SAURON IFS)", "").replace(" (Sérsic n=1)", "").replace(" (Sérsic n=4)", "") for r in results]
    t_rs = [r["t_rs_reg"] * 1000 for r in results]
    t_py = [r["t_py_reg"] * 1000 for r in results]
    t_vb = [r["t_vb"] * 1000 for r in results]

    x = np.arange(len(names))
    width = 0.26

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 11), constrained_layout=True)

    # Top: Execution Time (Log scale)
    rects_vb = ax0.bar(x - width, t_vb, width, label='Classic VorBin (Python WVT)', color='#e74c3c', alpha=0.9)
    rects_py = ax0.bar(x, t_py, width, label='PowerBin (Reference Python)', color='#f39c12', alpha=0.9)
    rects_rs = ax0.bar(x + width, t_rs, width, label='PowerBin (Optimized Rust)', color='#27ae60', alpha=0.9)

    ax0.set_yscale('log')
    ax0.set_ylabel("Execution Time (milliseconds, log scale)", fontsize=12)
    ax0.set_title("Algorithm Execution Time Comparison across 7 Datasets", fontsize=14, fontweight='bold')
    ax0.set_xticks(x)
    ax0.set_xticklabels(names, fontsize=10, fontweight='bold')
    ax0.grid(True, which="both", linestyle="--", alpha=0.4)
    ax0.legend(frameon=True, fontsize=11, loc='upper left')

    # Annotate time on Rust bars
    for i, (t_r, t_p, t_v) in enumerate(zip(t_rs, t_py, t_vb)):
        ax0.annotate(f"{t_r:.1f} ms", (x[i] + width, t_r),
                     textcoords="offset points", xytext=(0, 5),
                     ha='center', fontsize=8.5, fontweight='bold', color='#1e8449')

    # Bottom: Speedup factors of Rust over Python and VorBin
    sp_py = [r["speedup_vs_py"] for r in results]
    sp_vb = [r["speedup_vs_vb"] for r in results]

    width2 = 0.35
    rects_sp_py = ax1.bar(x - width2/2, sp_py, width2, label='Rust Speedup vs Python PowerBin', color='#3498db', alpha=0.9)
    rects_sp_vb = ax1.bar(x + width2/2, sp_vb, width2, label='Rust Speedup vs Classic VorBin', color='#9b59b6', alpha=0.9)

    ax1.set_ylabel("Speedup Factor (×)", fontsize=12)
    ax1.set_title("Rust Acceleration Factor (Speedup)", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=11, loc='upper left')

    for i, (sp_p, sp_v) in enumerate(zip(sp_py, sp_vb)):
        ax1.annotate(f"{sp_p:.1f}×", (x[i] - width2/2, sp_p),
                     textcoords="offset points", xytext=(0, 4),
                     ha='center', fontsize=9, fontweight='bold')
        ax1.annotate(f"{sp_v:.1f}×", (x[i] + width2/2, sp_v),
                     textcoords="offset points", xytext=(0, 4),
                     ha='center', fontsize=9, fontweight='bold', color='#6c3483')

    out_file = "algorithm_timing_comparison.png"
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    shutil.copy(out_file, os.path.join(ARTIFACT_DIR, out_file))
    print(f"\nSaved timing comparison chart to {out_file}")

def plot_visual_tessellation_comparison(results):
    """
    Generates a visual comparison of the output tessellations for representative datasets:
    Comparing Classic VorBin vs PowerBin Accretion-Only vs PowerBin Regularized (Rust).
    Shows the geometric differences: VorBin non-convexity vs Accretion vs CPD convexity.
    """
    # Select 3 representative datasets: Exponential Disk, Annular Ring, Spiral Arms
    selected_keys = ["exponential_disk", "annular_ring", "spiral_arms", "galaxy_group"]
    selected_res = [r for r in results if r["key"] in selected_keys]

    fig, axes = plt.subplots(len(selected_res), 4, figsize=(20, 5 * len(selected_res)), constrained_layout=True)

    for row, r in enumerate(selected_res):
        x = r["x"]
        y = r["y"]
        signal = r["signal"]
        
        # 1. Input Signal Map
        ax0 = axes[row, 0]
        ax0.scatter(x, y, c=np.log10(np.maximum(signal, 1e-3)), cmap='magma', s=8, marker='s', edgecolors='none')
        ax0.set_title(f"{r['name']}\nInput Signal", fontsize=11, fontweight='bold')
        ax0.set_aspect('equal')
        ax0.tick_params(labelsize=8)

        # 2. Classic VorBin (WVT)
        ax1 = axes[row, 1]
        vb_ids = r["vb_bin_ids"]
        rng = np.random.default_rng(row * 77 + 1)
        rand_colors_vb = rng.permutation(len(np.unique(vb_ids)))
        # Map bin IDs to 0..K-1
        _, inv_vb = np.unique(vb_ids, return_inverse=True)
        ax1.scatter(x, y, c=rand_colors_vb[inv_vb], cmap='Set3', s=8, marker='s', edgecolors='none')
        ax1.set_title(f"Classic VorBin (WVT)\n{r['bins_vb']} bins | Non-convex: {r['nc_vb']}/{r['tot_vb']}\nTime: {r['t_vb']*1000:.1f} ms", fontsize=11, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.tick_params(labelsize=8)

        # 3. PowerBin Accretion-Only (Rust)
        ax2 = axes[row, 2]
        pb_acc = r["pb_rs_acc"]
        rand_colors_acc = rng.permutation(len(pb_acc.rbin))
        ax2.scatter(x, y, c=rand_colors_acc[pb_acc.bin_num], cmap='Set3', s=8, marker='s', edgecolors='none')
        ax2.set_title(f"PowerBin Accretion Only\n{r['bins_rs_acc']} bins | RMS: {r['rms_rs_acc']:.1f}%\nTime: {r['t_rs_acc']*1000:.1f} ms", fontsize=11, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.tick_params(labelsize=8)

        # 4. PowerBin Regularized (Rust CPD)
        ax3 = axes[row, 3]
        pb_reg = r["pb_rs_reg"]
        rand_colors_reg = rng.permutation(len(pb_reg.rbin))
        ax3.scatter(x, y, c=rand_colors_reg[pb_reg.bin_num], cmap='Set3', s=8, marker='s', edgecolors='none')
        
        # Overlay soap-bubble circles
        xybin = pb_reg.xybin
        rbin = pb_reg.rbin
        single = pb_reg.single
        diam = 2 * rbin[~single]
        lw = 0.5 * (diam / np.maximum(np.max(diam), 1e-5))**0.3
        circles = collections.EllipseCollection(
            diam, diam, 0, offsets=xybin[~single], units='xy',
            facecolor='none', edgecolors='black', lw=lw, transOffset=ax3.transData
        )
        ax3.add_collection(circles)
        ax3.plot(xybin[:, 0], xybin[:, 1], 'k.', markersize=1.5, alpha=0.8)

        ax3.set_title(f"PowerBin Regularized (Rust CPD)\n{r['bins_rs_reg']} bins | Non-convex: {r['nc_rs_reg']}/{r['tot_rs_reg']}\nTime: {r['t_rs_reg']*1000:.1f} ms ({r['speedup_vs_vb']:.1f}× vs VorBin)", fontsize=11, fontweight='bold')
        ax3.set_aspect('equal')
        ax3.tick_params(labelsize=8)

    out_file = "algorithm_visual_tessellation_comparison.png"
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    shutil.copy(out_file, os.path.join(ARTIFACT_DIR, out_file))
    print(f"Saved visual tessellation comparison to {out_file}")

def main():
    results = run_benchmarks()
    plot_algorithm_timing_comparison(results)
    plot_visual_tessellation_comparison(results)

if __name__ == "__main__":
    main()
