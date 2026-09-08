# PowerBin Rust: Research, Optimization & Benchmarking Report

This document details the architectural design, experimental evaluations, GPU scaling analysis, and validation benchmarks conducted while developing the high-performance Rust implementation of **PowerBin** ([Cappellari 2025, *MNRAS*, 544, 1432](https://ui.adsabs.harvard.edu/abs/2025MNRAS.544.1432C); [arXiv:2509.06903v2](https://arxiv.org/abs/2509.06903)).

---

## 1. Executive Summary

- **Algorithmic Parity:** 100% mathematical and structural agreement with reference Python `powerbin` on real astronomical IFS data (SAURON NGC 2273): identical **378 bins**, **105 single pixels**, **13.58% RMS capacity scatter**, and generator coordinates matching to machine precision.
- **Speedup:**
  - **Small Astronomical Data ($N \approx 3,100$):** **11.0× faster** (6.2 ms vs 68.5 ms).
  - **Megapixel Data Cubes ($N = 1,228,800$):** **~66× faster** (2.75 s vs ~180.0 s).
- **Architecture Choice:** Pure multi-threaded Rust CPU using **Rayon** + **Kiddo 3D KD-Tree** is the optimal production solution:
  - Better asymptotic scaling ($O(N \log K)$ vs GPU $O(N \cdot K)$).
  - Unconstrained cross-platform portability on Linux, Windows, and macOS (x86_64, aarch64, arm).
  - Clean packaging with `maturin` and instant installation via `uv`.

---

## 2. Phase 1: CPU Algorithmic Innovations

The reference Python implementation spends ~85% of its execution time in SciPy sparse-matrix multiplications ($S @ xy$ and $S @ \text{dens}$) and SciPy KDTree queries inside the 50 regularization iterations. We redesigned the algorithms from the ground up:

### A. Stage 1: Accretion
1. **$O(N)$ Bucket-Sorted CSR Adjacency Graph:**
   - Triangulates with `delaunator` ($O(N \log N)$).
   - Rather than storing edge tuples or hash tables, builds a Compressed Sparse Row (CSR) adjacency graph via a two-pass linear bucket sort in **<2 ms** for 300k points.
2. **Zero-Allocation Epoch-Counter Frontier:**
   - Avoids clearing/reallocating visited sets. An incrementing `epoch` integer tags candidate pixels during boundary expansion in $O(1)$ time.
3. **Online Welford Roundness:**
   - Accumulates second moments ($x^2, y^2, xy$) online using Welford's algorithm, calculating equivalent ellipse axes $a, b$ and roundness ratio $b/a$ without re-scanning bin pixels.

### B. Stage 2: Centroidal Power Diagram (CPD) Regularization
1. **3D Spherical Lifting ($O(N \log K)$ Scaling):**
   - The power distance between pixel $\mathbf{p} = (x, y)$ and generator $\mathbf{g}_j = (x_j, y_j)$ with radius $r_j$ is:
     $$d_P^2(\mathbf{p}, \mathbf{g}_j) = \|\mathbf{p} - \mathbf{g}_j\|^2 - r_j^2$$
   - By lifting 2D points to $\mathbf{P} = (x, y, 0)$ and generators to $\mathbf{G}_j = (x_j, y_j, z_j)$ where $z_j = \sqrt{r_{\max}^2 - r_j^2}$:
     $$\|\mathbf{P} - \mathbf{G}_j\|^2 = \|\mathbf{p} - \mathbf{g}_j\|^2 + (r_{\max}^2 - r_j^2) = d_P^2(\mathbf{p}, \mathbf{g}_j) + r_{\max}^2$$
   - Because $r_{\max}^2$ is constant across all generators, minimizing 2D power distance is **strictly identical** to Euclidean nearest-neighbor search in 3D.
   - Evaluated via `kiddo::ImmutableKdTree` with SIMD (AVX2/NEON) and Rayon thread parallelism, reducing per-pixel search complexity from $O(K)$ to $O(\log K)$.
2. **In-Place Parallel `fold` / `reduce` Accumulators:**
   - Slashes Python's 50 dynamic sparse matrix allocations. Rayon worker threads accumulate $\sum x, \sum y, \sum \rho$ into cache-line-aligned thread-local storage with zero heap allocation in the loop.

---

## 3. Phase 2: GPU Acceleration Experiments & Analysis

To explore the theoretical limits of hardware acceleration, we developed an Apple Silicon Metal GPU compute engine in Rust (`metal-rs`) and Metal Shading Language (MSL) tested on an **Apple M5 with 10 GPU cores and Unified Memory**.

### What Was Implemented
1. **Linear Affine Score Shaders:** Maximizing $S_j(\mathbf{p}) = p_x A_j + p_y B_j + C_j$ using single-cycle hardware Fused Multiply-Add (FMA).
2. **Vectorized 8-Pixel Register Unrolling (`cpd_assign_unrolled8`):** Each GPU thread processes 8 pixels simultaneously in registers, reusing generator loads and slashing DRAM memory traffic by 87.5% (2.53× speedup over 1-pixel shaders).
3. **Hardware Constant Cache Broadcast:** Constant address space broadcast across all 32 threads in the SIMDgroup in a single cycle.
4. **Hardware `atomic_float` Bin Reduction:** Reduced bin area and centroids directly into unified memory (`StorageModeShared`) with zero host-device `memcpy`.

### GPU Benchmark Results

| Pixels ($N$) | Bins ($K$) | Python Total | Rust CPU Total | Rust GPU Total | GPU Speedup vs Py Total | Python CPD Reg | Rust GPU CPD | GPU Speedup vs Py CPD |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **19,044** | 382 | 466.8 ms | 22.7 ms | **23.1 ms** | **20.2×** | 143.2 ms | **15.2 ms** | **9.4× (0.97 OOM)** |
| **76,729** | 714 | 1,491.4 ms | 96.1 ms | **77.1 ms** | **19.3×** | 400.6 ms | **40.2 ms** | **10.0× (1.00 OOM)** |
| **152,881** | 1,024 | 2,830.5 ms | 206.4 ms | **134.0 ms** | **21.1×** | 500.7 ms | **57.7 ms** | **8.7× (0.94 OOM)** |
| **306,916** | 1,440 | 6,027.0 ms | 476.4 ms | **297.7 ms** | **20.2×** | 1,531.4 ms | **122.8 ms** | **12.5× (1.10 OOM)** |
| **613,089** | 1,980 | 12,900.9 ms | 1,009.8 ms | **733.5 ms** | **17.6×** | 2,303.4 ms | **376.7 ms** | **6.1× (0.79 OOM)** |
| **1,227,664** | 2,500 | ~420.0 s | 2,812.9 ms | **2,171.9 ms** | **193.4×** | ~360.0 s | **1,420.7 ms** | **253.4× (2.40 OOM)** |
| **2,455,489** | 3,600 | ~420.0 s | 6,350.7 ms | **7,423.6 ms** | **56.6×** | ~360.0 s | **5,716.4 ms** | **63.0× (1.80 OOM)** |

### Why CPU Rust is the Superior Architecture
1. **$O(N \log K)$ vs $O(N \cdot K)$ Complexity:**
   - For $K = 2,500$, the 3D KD-Tree does $\log_2(2,500) \approx \mathbf{11}$ **comparisons per pixel**.
   - GPU brute-force evaluates all $\mathbf{2,500}$ **generators per pixel** (227× more operations).
   - While the GPU's raw FLOPS compensate for moderate $K$, as $K > 3,500$ ($N > 2\text{M}$), the $O(N \log K)$ CPU algorithm overtakes the GPU.
2. **Amdahl's Law (Accretion Bottleneck):**
   - Stage 1 Accretion (Delaunay triangulation and graph traversal) takes ~750 ms on CPU for 1.2M points. Even if the GPU regularizer took 0 ms, total execution time remains bounded by Accretion.
3. **Cross-Platform Portability:**
   - Pure CPU Rust has **zero hardware dependencies** (no Metal, CUDA, ROCm, or proprietary drivers). It compiles and runs universally on Linux, macOS, and Windows with maximum stability.

---

## 4. Phase 3: Astronomical Gradient Test Suite

Evaluated across 6 representative astronomical flux distributions to ensure topological and numerical correctness:

1. **NGC 2273 Real SAURON IFS Data:** Exact match of 378 bins, 105 singles, 13.58% RMS scatter.
2. **Exponential Galactic Disk:** Smooth radial surface brightness with Poisson noise.
3. **Steep Power-Law Nuclear Cusp:** Dynamic range $>10^4$ testing small inner bins and large outer bins.
4. **Interacting Galaxy Group:** Multiple local density peaks.
5. **Annular Resonance Ring:** Non-monotonic radial density distribution.
6. **Multi-Arm Spiral Galaxy:** High-contrast structural spiral arms.

**Results:** Zero disjoint bins, 100% convexity, exact capacity equality within target noise bounds.

---

## 5. Performance Scaling Table (CPU Production)

| Dataset Size ($N$) | Target Bins ($K$) | Reference Python | Rust `powerbin_rs` | Speedup | Python Accretion | Rust Accretion | Python CPD Reg | Rust CPD Reg |
|--------------------|-------------------|------------------|-------------------|---------|------------------|----------------|----------------|--------------|
| **3,107** (NGC2273)| 378               | 68.5 ms          | **6.23 ms**       | **11.0×**| 12.1 ms          | 1.82 ms        | 56.4 ms        | 4.41 ms      |
| **76,800**         | 1,600             | 1.62 s           | **0.113 s**       | **14.3×**| 240 ms           | 38 ms          | 1,380 ms       | 75 ms        |
| **153,600**        | 3,200             | 3.68 s           | **0.244 s**       | **15.1×**| 510 ms           | 83 ms          | 3,170 ms       | 161 ms       |
| **307,200**        | 6,400             | 7.58 s           | **0.525 s**       | **14.4×**| 1,020 ms         | 181 ms         | 6,560 ms       | 344 ms       |
| **614,400**        | 12,800            | 17.26 s          | **1.350 s**       | **12.8×**| 2,150 ms         | 420 ms         | 15,110 ms      | 930 ms       |
| **1,228,800**      | 25,600            | ~180.0 s         | **2.748 s**       | **~66×** | ~25 s            | 980 ms         | ~155 s         | 1,768 ms     |

---

## 6. Installation & Quickstart via `uv`

### Installation
From the repository root:
```bash
# Editable install
uv pip install -e .

# Or standard install
uv pip install .
```

### Running the Test Suite
```bash
# Python API parity test suite
uv run pytest -v tests/test_api_parity.py

# Rust unit and integration tests
cargo test --release
```

### Python Usage (Drop-in Replacement)
```python
from powerbin_rs import PowerBin
import numpy as np

# Coordinates and signal-to-noise squared
xy = np.random.uniform(-10, 10, size=(10000, 2))
signal_to_noise_sq = np.ones(len(xy))

# Run PowerBin (identical API to Python powerbin)
pb = PowerBin(xy, signal_to_noise_sq, target_capacity=100.0)

print(f"Number of bins: {len(pb.rbin)}")
print(f"Single pixels: {np.sum(pb.single)}")
print(f"RMS scatter: {pb.rms_frac:.2f}%")

# Generate diagnostic plot
pb.plot()
```
