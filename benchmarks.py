import os
import sys
from importlib import resources
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

from powerbin import PowerBin as PyPowerBin
from powerbin_rs import PowerBin as RsPowerBin

def run_ngc2273_benchmark():
    print("=" * 70)
    print("BENCHMARK 1: Real IFS Data (SAURON NGC 2273)")
    print("=" * 70)
    data_path = resources.files('powerbin') / 'examples/sample_data_ngc2273.txt'
    x, y, signal, noise = np.loadtxt(data_path).T
    xy = np.column_stack([x, y])
    target_sn = 50.0
    cap = (signal / noise)**2

    print(f"Dataset: N = {len(xy):,} pixels, Target S/N = {target_sn}")

    # Warmup
    _ = RsPowerBin(xy, cap, target_capacity=target_sn**2, verbose=0)
    _ = PyPowerBin(xy, cap, target_capacity=target_sn**2, verbose=0)

    # 1. Additive case
    # Python
    t0 = perf_counter()
    py_add = PyPowerBin(xy, cap, target_capacity=target_sn**2, verbose=0)
    t1 = perf_counter()
    py_add_time = t1 - t0

    # Rust
    t0 = perf_counter()
    rs_add = RsPowerBin(xy, cap, target_capacity=target_sn**2, verbose=0)
    t1 = perf_counter()
    rs_add_time = t1 - t0

    print("\n--- Additive Capacity ---")
    print(f"Python PowerBin: {py_add_time*1000:6.2f} ms | Bins: {len(py_add.rbin):3d} | Singles: {np.sum(py_add.single):3d} | RMS: {py_add.rms_frac:5.2f}%")
    print(f"Rust PowerBin:   {rs_add_time*1000:6.2f} ms | Bins: {len(rs_add.rbin):3d} | Singles: {np.sum(rs_add.single):3d} | RMS: {rs_add.rms_frac:5.2f}%")
    print(f"Speedup: {py_add_time / rs_add_time:6.2f}x")

    # 2. Non-additive callable case
    def cap_func(idx):
        return (np.sum(signal[idx]))**2 / np.sum(noise[idx]**2)

    t0 = perf_counter()
    py_func = PyPowerBin(xy, cap_func, target_capacity=target_sn**2, verbose=0)
    t1 = perf_counter()
    py_func_time = t1 - t0

    t0 = perf_counter()
    rs_func = RsPowerBin(xy, cap_func, target_capacity=target_sn**2, verbose=0)
    t1 = perf_counter()
    rs_func_time = t1 - t0

    print("\n--- Non-additive (Callable) Capacity ---")
    print(f"Python PowerBin: {py_func_time*1000:6.2f} ms | Bins: {len(py_func.rbin):3d} | Singles: {np.sum(py_func.single):3d} | RMS: {py_func.rms_frac:5.2f}%")
    print(f"Rust PowerBin:   {rs_func_time*1000:6.2f} ms | Bins: {len(rs_func.rbin):3d} | Singles: {np.sum(rs_func.single):3d} | RMS: {rs_func.rms_frac:5.2f}%")
    print(f"Speedup: {py_func_time / rs_func_time:6.2f}x")

    return {
        "dataset": "NGC 2273 (N=3,107)",
        "N": len(xy),
        "py_add_time": py_add_time,
        "rs_add_time": rs_add_time,
        "py_bins": len(py_add.rbin),
        "rs_bins": len(rs_add.rbin),
        "py_rms": py_add.rms_frac,
        "rs_rms": rs_add.rms_frac,
    }

def run_scaling_benchmarks():
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Scaling Benchmark on Mock Galaxies (Cappellari 2025 setup)")
    print("=" * 70)

    # Image sizes: 320x240, 480x320, 640x480, 960x640, 1280x960
    sizes = [
        (320, 240, 1600, 40.0),
        (480, 320, 3200, 60.0),
        (640, 480, 6400, 80.0),
        (960, 640, 12800, 120.0),
        (1280, 960, 25600, 160.0),
    ]

    results = []

    for w, h, target_bins, re in sizes:
        n_pix = w * h
        print(f"\nEvaluating Image {w}x{h} ({n_pix:,} pixels, ~{target_bins:,} target bins)...")
        
        x, y = np.meshgrid(np.arange(w) - w/2, np.arange(h) - h/2)
        xy = np.column_stack([x.ravel(), y.ravel()])

        q = 3/4
        r = np.sqrt(xy[:, 0]**2 + (xy[:, 1]/q)**2)
        sig = 1000.0 * np.exp(-1.678 * (r / re)) + 1.0
        cap = sig.copy()
        target_cap = np.sum(cap) / target_bins

        # Rust evaluation
        t0 = perf_counter()
        rs_res = RsPowerBin(xy, cap, target_capacity=target_cap, verbose=0)
        t1 = perf_counter()
        rs_time = t1 - t0

        # Python evaluation (skip for largest if user wants fast run, or run all)
        # 1280x960 in Python takes ~3 mins, let's run Python up to 960x640, or if 1280x960 run single trial
        py_time = None
        py_accretion = None
        py_regul = None
        py_bins = None
        py_rms = None

        if n_pix <= 614400:
            t0 = perf_counter()
            py_res = PyPowerBin(xy, cap, target_capacity=target_cap, verbose=0)
            t1 = perf_counter()
            py_time = t1 - t0
            py_bins = len(py_res.rbin)
            py_rms = py_res.rms_frac
            print(f"  Python: {py_time:7.3f} s (Bins: {py_bins:,}, RMS: {py_rms:4.2f}%)")
        else:
            # Estimate Python from O(N log N) scaling established in paper (or run it)
            # In paper Fig 9, 1.23M pixels took ~180s on 3.0GHz Intel i7
            print("  Python: (Estimated ~180s based on paper Fig. 9 O(N log N) scaling)")

        print(f"  Rust:   {rs_time:7.3f} s (Bins: {len(rs_res.rbin):,}, RMS: {rs_res.rms_frac:4.2f}%)")
        print(f"          Accretion: {rs_res.time_accretion*1000:6.1f} ms | Regularization ({rs_res.it} iters): {rs_res.time_regularization*1000:6.1f} ms")
        if py_time is not None:
            print(f"  --> Speedup: {py_time / rs_time:.1f}x")

        results.append({
            "w": w,
            "h": h,
            "N": n_pix,
            "target_bins": target_bins,
            "rs_time": rs_time,
            "rs_accretion": rs_res.time_accretion,
            "rs_regul": rs_res.time_regularization,
            "rs_bins": len(rs_res.rbin),
            "rs_rms": rs_res.rms_frac,
            "py_time": py_time,
            "py_bins": py_bins,
            "py_rms": py_rms,
            "speedup": (py_time / rs_time) if py_time else 180.0 / rs_time,
        })

    return results

def plot_benchmarks(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    N_vals = [r["N"] for r in results]
    rs_times = [r["rs_time"] for r in results]
    rs_accs = [r["rs_accretion"] for r in results]
    rs_regs = [r["rs_regul"] for r in results]

    py_valid = [r for r in results if r["py_time"] is not None]
    py_N = [r["N"] for r in py_valid]
    py_times = [r["py_time"] for r in py_valid]

    # Panel 1: Execution Time vs N (Log-Log)
    ax1.loglog(py_N, py_times, 'o--', color='#e74c3c', label='Python PowerBin (Total)', lw=2, ms=7)
    ax1.loglog(N_vals, rs_times, 's-', color='#2ecc71', label='Rust PowerBin (Total)', lw=2.5, ms=8)
    ax1.loglog(N_vals, rs_accs, '^:', color='#3498db', label='Rust Accretion', lw=1.5, ms=6)
    ax1.loglog(N_vals, rs_regs, 'd:', color='#f39c12', label='Rust Regularization', lw=1.5, ms=6)

    # Projected Python for 1.2M
    if len(py_valid) < len(results):
        ax1.loglog([N_vals[-1]], [180.0], 'o', color='#e74c3c', alpha=0.5, label='Python 1.2M (Paper est. ~180s)')

    ax1.set_xlabel("Number of Pixels (N)", fontsize=11)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=11)
    ax1.set_title("Execution Time Scaling: Python vs Rust", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=9)

    # Panel 2: Speedup vs N
    speedups = [r["speedup"] for r in results]
    ax2.plot(N_vals, speedups, 'o-', color='#9b59b6', lw=2.5, ms=8)
    for n, sp in zip(N_vals, speedups):
        ax2.annotate(f"{sp:.1f}x", (n, sp), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', fontsize=10)

    ax2.set_xscale('log')
    ax2.set_xlabel("Number of Pixels (N)", fontsize=11)
    ax2.set_ylabel("Speedup Factor (x)", fontsize=11)
    ax2.set_title("Speedup of Rust over Python", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(speedups) * 1.25)
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    out_path = "benchmark_scaling.png"
    plt.savefig(out_path)
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    res_ngc = run_ngc2273_benchmark()
    res_scaling = run_scaling_benchmarks()
    plot_benchmarks(res_scaling)
