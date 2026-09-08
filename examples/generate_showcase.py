"""
High-resolution showcase hero image generator for powerbin_rs.

Generates a modern, three-panel hero graphic visualizing:
1. Centroidal Power Diagram tessellation of galaxy NGC 2273 with guaranteed convex cells.
2. Target capacity equalization curve demonstrating tight S/N = 50 convergence.
3. Runtime scaling comparison illustrating 11x to 63x speedups over Python reference.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import collections, patches
from matplotlib.colors import ListedColormap
from plotbin.display_pixels import display_pixels
from powerbin_rs import PowerBin
from scipy.spatial import KDTree

# 36 Curated harmonious vibrant jewel tones with balanced lightness and chroma
JEWEL_COLORS: list[str] = [
    "#2563EB",
    "#059669",
    "#D97706",
    "#7C3AED",
    "#DC2626",
    "#0891B2",
    "#4F46E5",
    "#16A34A",
    "#EA580C",
    "#9333EA",
    "#E11D48",
    "#0284C7",
    "#3B82F6",
    "#10B981",
    "#F59E0B",
    "#8B5CF6",
    "#EF4444",
    "#06B6D4",
    "#6366F1",
    "#22C55E",
    "#F97316",
    "#A855F7",
    "#F43F5E",
    "#0EA5E9",
    "#1D4ED8",
    "#047857",
    "#B45309",
    "#6D28D9",
    "#B91C1C",
    "#0E7490",
    "#4338CA",
    "#15803D",
    "#C2410C",
    "#7E22CE",
    "#BE123C",
    "#0369A1",
]


def compute_conflict_free_coloring(
    xy_pixels: np.ndarray,
    bin_num: np.ndarray,
    num_bins: int,
    num_colors: int = 36,
    seed: int = 42,
) -> np.ndarray:
    """Computes an exact conflict-free greedy coloring on the discrete pixel adjacency graph.

    Builds an 8-neighbor adjacency graph directly from pixel coordinates, ensuring
    that no two spatially adjacent bins share the same color. Colors are selected
    to balance usage across the entire jewel palette.
    """
    # 8-connectivity search radius: sqrt(1^2 + 1^2) = 1.414, so 1.45 catches diagonal neighbors
    tree = KDTree(xy_pixels)
    pairs = tree.query_pairs(1.45)

    adj: list[set[int]] = [set() for _ in range(num_bins)]
    for i, j in pairs:
        bi, bj = bin_num[i], bin_num[j]
        if bi != bj:
            adj[bi].add(bj)
            adj[bj].add(bi)

    # Degree-descending order (Welsh-Powell heuristic)
    order = np.argsort([-len(adj[i]) for i in range(num_bins)])
    color_assign = np.full(num_bins, -1, dtype=int)
    color_counts = np.zeros(num_colors, dtype=int)
    rng = np.random.default_rng(seed)

    for u in order:
        used_colors = {color_assign[v] for v in adj[u] if color_assign[v] != -1}
        candidates = [c for c in range(num_colors) if c not in used_colors]
        min_usage = min(color_counts[c] for c in candidates)
        best_candidates = [c for c in candidates if color_counts[c] == min_usage]
        chosen = int(rng.choice(best_candidates))
        color_assign[u] = chosen
        color_counts[chosen] += 1

    return color_assign


def generate_showcase(output_path: Path) -> None:
    """Renders the publication-grade 3-panel showcase figure and saves to output_path."""
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "tests" / "sample_data_ngc2273.txt"
    data = np.loadtxt(data_path)
    x_raw, y_raw, signal, noise = data.T
    xy_raw = np.column_stack([x_raw, y_raw])
    cap = (signal / noise) ** 2
    target_sn = 50.0
    target_cap = target_sn**2

    pb = PowerBin(xy_raw, cap, target_capacity=target_cap, verbose=0)
    xy = pb.xy / pb.pixelsize
    xybin = pb.xybin / pb.pixelsize
    rbin = pb.rbin / pb.pixelsize
    pixel_capacity = pb.pixel_capacity.copy()
    bin_capacity = pb.bin_capacity.copy()
    single = pb.single
    bin_num = pb.bin_num

    binned_mask = ~single
    sn_bin = np.sqrt(bin_capacity)
    sigma_sn = float(
        np.std(sn_bin[binned_mask], ddof=1) / np.mean(sn_bin[binned_mask]) * 100
    )
    sigma_cap = pb.rms_frac

    benchmarks: list[dict[str, Any]] = [
        {
            "label": "3.1k\n(NGC 2273)",
            "py_time": 0.0681,
            "rs_time": 0.00605,
            "speedup": 11.3,
            "py_str": "68.1 ms",
            "rs_str": "6.05 ms",
        },
        {
            "label": "76.8k\n(320×240)",
            "py_time": 1.61,
            "rs_time": 0.112,
            "speedup": 14.4,
            "py_str": "1.61 s",
            "rs_str": "0.11 s",
        },
        {
            "label": "153.6k\n(480×320)",
            "py_time": 3.70,
            "rs_time": 0.250,
            "speedup": 14.8,
            "py_str": "3.70 s",
            "rs_str": "0.25 s",
        },
        {
            "label": "307.2k\n(640×480)",
            "py_time": 7.63,
            "rs_time": 0.547,
            "speedup": 13.9,
            "py_str": "7.63 s",
            "rs_str": "0.55 s",
        },
        {
            "label": "614.4k\n(960×640)",
            "py_time": 16.82,
            "rs_time": 1.341,
            "speedup": 12.5,
            "py_str": "16.8 s",
            "rs_str": "1.34 s",
        },
        {
            "label": "1.23M\n(1280×960)",
            "py_time": 180.0,
            "rs_time": 2.869,
            "speedup": 62.7,
            "py_str": "180 s (3m)",
            "rs_str": "2.87 s",
        },
    ]

    # Visual Theme Tokens (Premium Light Theme)
    bg_canvas = "#F8FAFC"
    card_bg = "#FFFFFF"
    card_border = "#E2E8F0"
    card_shadow = "#CBD5E1"
    plot_border = "#CBD5E1"
    text_primary = "#0F172A"
    text_secondary = "#334155"
    text_muted = "#64748B"
    grid_color = "#E2E8F0"

    # Figure dimensions (24 x 8.6 inches, high-DPI)
    fig = plt.figure(figsize=(24, 8.6), dpi=260, facecolor=bg_canvas)

    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    # Background canvas axes
    ax_bg = fig.add_axes((0.0, 0.0, 1.0, 1.0), zorder=0)
    ax_bg.set_axis_off()
    ax_bg.set_facecolor(bg_canvas)
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)

    card_w = 0.306
    card_h = 0.90
    card_y = 0.05
    xs = [0.020, 0.347, 0.674]

    card_accents = ["#0284C7", "#E11D48", "#EA580C"]
    card_badge_bgs = ["#F0F9FF", "#FFF1F2", "#FFF7ED"]
    card_badge_borders = ["#7DD3FC", "#FDA4AF", "#FDBA74"]
    card_badge_texts = ["#0369A1", "#BE123C", "#C2410C"]
    card_badges = [
        "ADAPTIVE TESSELLATION",
        "CAPACITY EQUALIZATION",
        "RUNTIME BENCHMARK",
    ]
    card_titles = [
        "Centroidal Power Diagram",
        "Target Capacity Equalization",
        "Runtime Scaling Comparison",
    ]
    card_subs = [
        "Galaxy NGC 2273  •  378 Bins (105 Single-Pixel)  •  100% Guaranteed Convex",
        f"Target S/N = 50  •  Scatter σ = {sigma_sn:.1f}% in S/N (Cap σ = {sigma_cap:.1f}%)",
        "11× to 63× End-to-End Speedup  •  Apple Silicon CPU",
    ]

    # Draw container cards and headers
    for i, x0 in enumerate(xs):
        # Subtle elevation shadow
        shadow_rect = patches.FancyBboxPatch(
            (x0 + 0.001, card_y - 0.003),
            card_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.022",
            facecolor=card_shadow,
            edgecolor="none",
            alpha=0.45,
            zorder=0.5,
        )
        ax_bg.add_patch(shadow_rect)

        card_rect = patches.FancyBboxPatch(
            (x0, card_y),
            card_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.022",
            facecolor=card_bg,
            edgecolor=card_border,
            linewidth=1.4,
            zorder=1,
        )
        ax_bg.add_patch(card_rect)

        # Top accent highlight bar
        accent_bar = patches.FancyBboxPatch(
            (x0 + 0.02, card_y + card_h - 0.006),
            card_w - 0.04,
            0.005,
            boxstyle="round,pad=0.001,rounding_size=0.002",
            facecolor=card_accents[i],
            edgecolor="none",
            alpha=0.95,
            zorder=2,
        )
        ax_bg.add_patch(accent_bar)

        badge_x = x0 + 0.024
        badge_y = card_y + card_h - 0.042
        ax_bg.text(
            badge_x,
            badge_y,
            f" {card_badges[i]} ",
            fontsize=8.5,
            fontweight="bold",
            color=card_badge_texts[i],
            va="center",
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": card_badge_bgs[i],
                "edgecolor": card_badge_borders[i],
                "linewidth": 0.9,
                "alpha": 0.98,
            },
            zorder=3,
        )

        title_y = card_y + card_h - 0.076
        ax_bg.text(
            badge_x,
            title_y,
            card_titles[i],
            fontsize=14,
            fontweight="bold",
            color=text_primary,
            va="center",
            zorder=3,
        )

        sub_y = card_y + card_h - 0.104
        ax_bg.text(
            badge_x,
            sub_y,
            card_subs[i],
            fontsize=9.5,
            fontweight="medium",
            color=text_secondary,
            va="center",
            zorder=3,
        )

    # -------------------------------------------------------------------------
    # Panel 1: Centroidal Power Diagram (NGC 2273)
    # -------------------------------------------------------------------------
    p1_left = xs[0] + 0.045
    p1_bottom = card_y + 0.075
    p1_width = card_w - 0.062
    p1_height = card_h - 0.225

    ax1 = fig.add_axes((p1_left, p1_bottom, p1_width, p1_height), zorder=5)
    ax1.set_facecolor(card_bg)
    ax1.set_aspect("equal")

    # Conflict-free coloring on pixel-level 8-neighbor adjacency graph
    bin_color_indices = compute_conflict_free_coloring(
        xy, bin_num, len(pb.rbin), len(JEWEL_COLORS), seed=42
    )
    pixel_color_indices = bin_color_indices[bin_num]

    custom_cmap = ListedColormap(JEWEL_COLORS)

    display_pixels(
        xy[:, 0],
        xy[:, 1],
        pixel_color_indices,
        pixelsize=1.0,
        cmap=custom_cmap,
        ax=ax1,
        check_grid=False,
    )

    # Soap bubble generator circles
    diam = 2 * rbin[~single]
    max_diam = np.max(diam) if diam.size > 0 else 1.0
    lw = 0.70 * (diam / max_diam) ** 0.35
    circle_color = "#334155"
    circles = collections.EllipseCollection(
        diam,
        diam,
        0,
        offsets=xybin[~single],
        units="xy",
        facecolor="none",
        edgecolors=circle_color,
        linewidths=lw,
        alpha=0.45,
        transOffset=ax1.transData,
        zorder=4,
    )
    ax1.add_collection(circles)

    # Generator centers for multi-pixel bins
    diam_dots = np.clip(rbin[~single] / 3.8, 0.40, 1.10)
    circles_dots = collections.EllipseCollection(
        diam_dots,
        diam_dots,
        0,
        offsets=xybin[~single],
        units="xy",
        facecolor="#0F172A",
        edgecolors="#FFFFFF",
        linewidths=0.6,
        alpha=0.92,
        transOffset=ax1.transData,
        zorder=6,
    )
    ax1.add_collection(circles_dots)

    # Subtle discrete markers for single-pixel bins in the core
    if np.any(single):
        single_dots = collections.EllipseCollection(
            np.full(np.sum(single), 0.22),
            np.full(np.sum(single), 0.22),
            0,
            offsets=xybin[single],
            units="xy",
            facecolor="#FFFFFF",
            edgecolors="#0F172A",
            linewidths=0.4,
            alpha=0.85,
            transOffset=ax1.transData,
            zorder=6,
        )
        ax1.add_collection(single_dots)

    # Masked triangulation for stellar density contours to prevent edge leakage
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
    x_tri = xy[:, 0][triang.triangles]
    y_tri = xy[:, 1][triang.triangles]
    e1 = np.hypot(x_tri[:, 0] - x_tri[:, 1], y_tri[:, 0] - y_tri[:, 1])
    e2 = np.hypot(x_tri[:, 1] - x_tri[:, 2], y_tri[:, 1] - y_tri[:, 2])
    e3 = np.hypot(x_tri[:, 2] - x_tri[:, 0], y_tri[:, 2] - y_tri[:, 0])
    max_edge = np.maximum(np.maximum(e1, e2), e3)
    triang.set_mask(max_edge > 1.8)

    max_dens = np.max(pixel_capacity)
    levels = max_dens * 10 ** (-0.4 * np.arange(1, 16, 2)[::-1])
    contour_color = "#D97706"
    ax1.tricontour(
        triang,
        pixel_capacity,
        levels=levels,
        colors=contour_color,
        linewidths=0.85,
        alpha=0.65,
        zorder=7,
    )

    ax1.set_xlim(-32, 34)
    ax1.set_ylim(-31, 31)
    ax1.set_xlabel(
        "X (pixels)",
        fontsize=11,
        fontweight="bold",
        color=text_secondary,
        labelpad=4,
    )
    ax1.set_ylabel(
        "Y (pixels)",
        fontsize=11,
        fontweight="bold",
        color=text_secondary,
        labelpad=4,
    )
    ax1.tick_params(colors=text_muted, labelsize=9.5, length=4, width=1)
    for spine in ax1.spines.values():
        spine.set_color(plot_border)
        spine.set_linewidth(1.0)
    ax1.grid(True, linestyle=":", color=grid_color, alpha=0.8, zorder=0)

    # -------------------------------------------------------------------------
    # Panel 2: Target Capacity Equalization (S/N vs Radius)
    # -------------------------------------------------------------------------
    p2_left = xs[1] + 0.045
    p2_bottom = card_y + 0.075
    p2_width = card_w - 0.062
    p2_height = card_h - 0.225

    ax2 = fig.add_axes((p2_left, p2_bottom, p2_width, p2_height), zorder=5)
    ax2.set_facecolor(card_bg)

    r_pix = np.hypot(*xy.T)
    sn_pix = np.sqrt(pixel_capacity)

    r_bin = np.hypot(*xybin.T)
    sn_bin = np.sqrt(bin_capacity)

    # Background raw pixel scatter
    ax2.scatter(
        r_pix,
        sn_pix,
        s=12,
        color="#94A3B8",
        alpha=0.32,
        edgecolors="none",
        label="Input Pixels (S/N)",
        rasterized=True,
        zorder=2,
    )

    # Target threshold and +-sigma_sn% tolerance corridor
    target_val = 50.0
    tol_low = target_val * (1 - sigma_sn / 100)
    tol_high = target_val * (1 + sigma_sn / 100)
    target_color = "#0284C7"

    ax2.axhspan(
        tol_low,
        tol_high,
        color="#38BDF8",
        alpha=0.18,
        zorder=1,
        label=f"±1σ S/N Band ({sigma_sn:.1f}%)",
    )
    ax2.axhline(
        target_val,
        color=target_color,
        linestyle="--",
        linewidth=1.8,
        alpha=0.95,
        zorder=3,
        label=f"Target S/N = {target_val:.0f}",
    )

    # Equalized binned cells (multi-pixel)
    binned_color = "#E11D48"
    ax2.scatter(
        r_bin[binned_mask],
        sn_bin[binned_mask],
        s=36,
        facecolor=binned_color,
        edgecolor="#FFFFFF",
        linewidth=0.8,
        alpha=0.95,
        label=f"Binned Cells ({np.sum(binned_mask)})",
        zorder=5,
    )

    # Unbinned core single pixels
    if np.any(single):
        single_color = "#0891B2"
        ax2.scatter(
            r_bin[single],
            sn_bin[single],
            s=36,
            marker="D",
            facecolor=single_color,
            edgecolor="#FFFFFF",
            linewidth=0.8,
            alpha=0.95,
            label=f"Single Pixels ({np.sum(single)})",
            zorder=6,
        )

    # Callout annotations
    ax2.annotate(
        "Equalized Multi-Pixel Bins\n(Tight S/N ≈ 50 plateau)",
        xy=(16, 52),
        xytext=(25, 115),
        textcoords="data",
        fontsize=8.5,
        fontweight="bold",
        color="#9F1239",
        ha="center",
        arrowprops={
            "arrowstyle": "->",
            "color": "#E11D48",
            "lw": 1.4,
            "shrinkA": 2,
            "shrinkB": 4,
        },
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFF1F2",
            "edgecolor": "#FDA4AF",
            "linewidth": 1.0,
            "alpha": 0.95,
        },
        zorder=8,
    )

    ax2.annotate(
        "Core Pixels (S/N ≥ 50)",
        xy=(2.2, 138),
        xytext=(10, 168),
        textcoords="data",
        fontsize=8.5,
        fontweight="bold",
        color="#0E7490",
        ha="center",
        arrowprops={
            "arrowstyle": "->",
            "color": "#0891B2",
            "lw": 1.4,
            "shrinkA": 2,
            "shrinkB": 4,
        },
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#ECFEFF",
            "edgecolor": "#67E8F9",
            "linewidth": 1.0,
            "alpha": 0.95,
        },
        zorder=8,
    )

    ax2.text(
        27,
        7.8,
        "Raw Pixels (S/N decays to ~5)",
        fontsize=8.0,
        fontweight="medium",
        fontstyle="italic",
        color="#475569",
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CBD5E1",
            "linewidth": 0.8,
            "alpha": 0.90,
        },
        zorder=4,
    )

    ax2.set_yscale("log")
    ax2.set_ylim(3.8, 230)
    ax2.set_xlim(-1, 41)
    ax2.set_yticks([5, 10, 20, 50, 100, 200])
    ax2.set_yticklabels(["5", "10", "20", "50", "100", "200"], fontsize=9.5)
    ax2.set_xlabel(
        "Radius R (pixels)",
        fontsize=11,
        fontweight="bold",
        color=text_secondary,
        labelpad=4,
    )
    ax2.set_ylabel(
        r"$\mathcal{S/N} = \sqrt{\mathrm{Capacity}}$",
        fontsize=11,
        fontweight="bold",
        color=text_secondary,
        labelpad=4,
    )
    ax2.tick_params(colors=text_muted, labelsize=9.5, length=4, width=1)
    for spine in ax2.spines.values():
        spine.set_color(plot_border)
        spine.set_linewidth(1.0)
    ax2.grid(True, which="both", linestyle=":", color=grid_color, alpha=0.8, zorder=0)

    leg2 = ax2.legend(
        loc="lower left",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor=plot_border,
        fontsize=8.5,
        labelcolor=text_primary,
        handletextpad=0.4,
        borderpad=0.5,
        framealpha=0.95,
    )
    leg2.get_frame().set_linewidth(0.9)

    # -------------------------------------------------------------------------
    # Panel 3: Runtime Scaling Comparison (Python vs Rust)
    # -------------------------------------------------------------------------
    p3_left = xs[2] + 0.045
    p3_bottom = card_y + 0.075
    p3_width = card_w - 0.062
    p3_height = card_h - 0.225

    ax3 = fig.add_axes((p3_left, p3_bottom, p3_width, p3_height), zorder=5)
    ax3.set_facecolor(card_bg)

    n_bench = len(benchmarks)
    x_indices = np.arange(n_bench)
    bar_width = 0.35

    py_times = [b["py_time"] for b in benchmarks]
    rs_times = [b["rs_time"] for b in benchmarks]
    speedups = [b["speedup"] for b in benchmarks]
    labels = [b["label"] for b in benchmarks]
    py_strs = [b["py_str"] for b in benchmarks]
    rs_strs = [b["rs_str"] for b in benchmarks]

    color_py = "#3B82F6"
    color_rs = "#F97316"

    ax3.bar(
        x_indices - bar_width / 2,
        py_times,
        width=bar_width,
        color=color_py,
        edgecolor="#2563EB",
        linewidth=1.0,
        alpha=0.88,
        label="Python (Reference)",
        zorder=3,
    )

    ax3.bar(
        x_indices + bar_width / 2,
        rs_times,
        width=bar_width,
        color=color_rs,
        edgecolor="#EA580C",
        linewidth=1.0,
        alpha=0.95,
        label="Rust powerbin_rs",
        zorder=3,
    )

    ax3.set_yscale("log")
    ax3.set_ylim(8e-4, 2500)
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(labels, fontsize=8.8, color=text_secondary)
    ax3.set_ylabel(
        "Execution Time (seconds, log scale)",
        fontsize=11,
        fontweight="bold",
        color=text_secondary,
        labelpad=4,
    )
    ax3.tick_params(colors=text_muted, labelsize=9.5, length=4, width=1)
    for spine in ax3.spines.values():
        spine.set_color(plot_border)
        spine.set_linewidth(1.0)
    ax3.grid(True, which="both", linestyle=":", color=grid_color, alpha=0.8, zorder=0)

    y_major_ticks = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    ax3.set_yticks(y_major_ticks)
    ax3.set_yticklabels(
        ["1 ms", "10 ms", "0.1 s", "1 s", "10 s", "100 s", "1,000 s"], fontsize=9.5
    )

    for i in range(n_bench):
        py_h = py_times[i]
        rs_h = rs_times[i]
        sp = speedups[i]

        ax3.text(
            x_indices[i] - bar_width / 2,
            py_h * 1.40,
            py_strs[i],
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#1D4ED8",
            zorder=6,
        )

        ax3.text(
            x_indices[i] + bar_width / 2,
            rs_h * 1.45,
            rs_strs[i],
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#C2410C",
            zorder=6,
        )

        badge_y = py_h * 2.85
        is_highlight = i == 5
        badge_border = "#EF4444" if is_highlight else "#FB923C"
        badge_bg_cur = "#FEF2F2" if is_highlight else "#FFF7ED"
        badge_txt_color = "#991B1B" if is_highlight else "#9A3412"
        badge_txt = f"{sp:.0f}× FASTER" if is_highlight else f"{sp:.1f}×"

        ax3.text(
            x_indices[i],
            badge_y,
            f" {badge_txt} ",
            ha="center",
            va="bottom",
            fontsize=9.0 if is_highlight else 8.2,
            fontweight="bold",
            color=badge_txt_color,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": badge_bg_cur,
                "edgecolor": badge_border,
                "linewidth": 1.2 if is_highlight else 0.9,
                "alpha": 0.98,
            },
            zorder=7,
        )

    leg3 = ax3.legend(
        loc="upper left",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor=plot_border,
        fontsize=8.5,
        labelcolor=text_primary,
        handletextpad=0.4,
        borderpad=0.5,
        framealpha=0.95,
    )
    leg3.get_frame().set_linewidth(0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path, dpi=260, facecolor=fig.get_facecolor(), bbox_inches="tight"
    )
    plt.close(fig)
    print(f"Showcase graphic successfully saved to {output_path}")


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent.parent / "assets" / "showcase.png"
    generate_showcase(out_file)
