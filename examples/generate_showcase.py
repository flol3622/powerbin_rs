"""Generate the README showcase with Matplotlib and the NGC 2273 sample.

Chart contract: a spatial categorical map shows adaptive cell size; a pixel/bin
scatter compares S/N with its target; six paired runtime observations compare
implementations on an explicitly logarithmic axis. Benchmark values come from
README.md (not a fresh timing run); the largest Python timing is approximate.
The static 3600 x 1328 PNG is designed for a full-width README. Blue/orange and
circle/diamond encodings distinguish series. The map uses a separate categorical
palette because adjacent cell identity, rather than magnitude, is the subject.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from plotbin.display_pixels import display_pixels
from powerbin_rs import PowerBin
from scipy.spatial import KDTree

# Balanced categorical tones; adjacency coloring keeps neighboring bins distinct.
JEWEL_COLORS = [
    "#4477AA",
    "#66AACC",
    "#228877",
    "#88BBAA",
    "#DDAA55",
    "#CC6677",
    "#AA77AA",
    "#7777AA",
    "#BB8855",
    "#99AA55",
]
INK = "#172B3A"
MUTED = "#586B78"
BLUE = "#2878AD"
ORANGE = "#CB6534"
GRID = "#E3E9ED"


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


def style_axis(ax: plt.Axes) -> None:
    """Keep scaffolding quiet and consistent across the quantitative panels."""
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color(GRID)
    ax.tick_params(colors=MUTED, length=3, pad=6)
    ax.minorticks_off()


def format_time(seconds: float) -> str:
    return f"{seconds * 1000:.3g} ms" if seconds < 0.1 else f"{seconds:.3g} s"


def generate_showcase(output_path: Path) -> None:
    """Render data-derived bin statistics alongside the documented benchmarks."""
    repo_root = Path(__file__).resolve().parent.parent
    x, y, signal, noise = np.loadtxt(repo_root / "tests" / "sample_data_ngc2273.txt").T
    target_sn = 50.0
    pb = PowerBin(
        np.column_stack([x, y]),
        (signal / noise) ** 2,
        target_capacity=target_sn**2,
        verbose=0,
    )
    xy = pb.xy / pb.pixelsize
    centers = pb.xybin / pb.pixelsize
    single = pb.single
    sn = np.sqrt(pb.bin_capacity)
    scatter = np.std(sn[~single], ddof=1) / np.mean(sn[~single]) * 100

    # README benchmark table; last reference timing is cited there as approximate.
    sizes = ["3.1k", "76.8k", "153.6k", "307.2k", "614.4k", "1.23M"]
    py_times = np.array([0.0681, 1.61, 3.70, 7.63, 16.82, 180.0])
    rs_times = np.array([0.00605, 0.112, 0.250, 0.547, 1.341, 2.869])

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.facecolor": "#FFFFFF",
        }
    ):
        fig = plt.figure(figsize=(18, 6.64), dpi=200, facecolor="white")

        lefts = [0.04, 0.365, 0.695]
        titles = [
            "Spatial tessellation",
            "Signal-to-noise equalization",
            "Runtime comparison",
        ]
        metrics = [
            f"{len(sn):,} adaptive bins",
            f"{scatter:.1f}% S/N scatter",
            "11–63× faster",
        ]
        subtitles = [
            f"{len(x):,} pixels  ·  NGC 2273",
            f"Target S/N = {target_sn:g}  ·  multi-pixel bins",
            "Python reference vs. Rust  ·  lower is better",
        ]
        for left, title, metric, subtitle in zip(lefts, titles, metrics, subtitles):
            fig.text(left, 0.785 / 0.83, title, fontsize=12, weight="bold")
            fig.text(left, 0.724 / 0.83, metric, fontsize=23, weight="bold")
            fig.text(left, 0.684 / 0.83, subtitle, fontsize=10.5, color=MUTED)

        ax1 = fig.add_axes((0.073, 0.205 / 0.83, 0.255, 0.425 / 0.83))
        ax2 = fig.add_axes((0.399, 0.205 / 0.83, 0.255, 0.425 / 0.83))
        ax3 = fig.add_axes((0.744, 0.205 / 0.83, 0.188, 0.425 / 0.83))

        # Pixel footprint, without overlapping generator circles or contours.
        colors = compute_conflict_free_coloring(
            xy,
            pb.bin_num,
            len(sn),
            len(JEWEL_COLORS),
            seed=42,
        )
        display_pixels(
            xy[:, 0],
            xy[:, 1],
            colors[pb.bin_num],
            pixelsize=1,
            cmap=ListedColormap(JEWEL_COLORS),
            vmin=-0.5,
            vmax=len(JEWEL_COLORS) - 0.5,
            ax=ax1,
            check_grid=False,
        )
        # Draw shared pixel edges only where bin identity changes.
        # Coordinates contain small measurement jitter, so snap to the pixel grid.
        grid_xy = np.rint(xy).astype(int)
        lookup = {tuple(point): int(b) for point, b in zip(grid_xy, pb.bin_num)}
        edges = []
        for (px, py), b in lookup.items():
            if (px + 1, py) in lookup and lookup[px + 1, py] != b:
                edges.append([(px + 0.5, py - 0.5), (px + 0.5, py + 0.5)])
            if (px, py + 1) in lookup and lookup[px, py + 1] != b:
                edges.append([(px - 0.5, py + 0.5), (px + 0.5, py + 0.5)])
        ax1.add_collection(
            LineCollection(edges, colors="white", linewidths=0.35, alpha=0.65)
        )
        ax1.set(
            xlim=(-33, 34), ylim=(-31, 31), xlabel="X (pixels)", ylabel="Y (pixels)"
        )
        ax1.set_xticks([-30, -15, 0, 15, 30])
        ax1.set_yticks([-30, -15, 0, 15, 30])
        style_axis(ax1)

        radius = np.hypot(*xy.T)
        bin_radius = np.hypot(*centers.T)
        ax2.scatter(
            radius,
            np.sqrt(pb.pixel_capacity),
            s=7,
            c="#AEBBC4",
            alpha=0.35,
            edgecolors="none",
            label="Input pixels",
        )
        ax2.axhline(target_sn, color=INK, ls="--", lw=1.2)
        ax2.scatter(
            bin_radius[~single],
            sn[~single],
            s=22,
            c=ORANGE,
            edgecolors="white",
            linewidths=0.4,
            label="Multi-pixel bins",
            zorder=3,
        )
        ax2.scatter(
            bin_radius[single],
            sn[single],
            s=20,
            c=BLUE,
            marker="D",
            edgecolors="white",
            linewidths=0.4,
            label="Single pixels",
            zorder=4,
        )
        ax2.text(39, 57, "Target 50", ha="right", fontsize=10, color=INK)
        ax2.set(
            yscale="log",
            ylim=(4, 230),
            xlim=(-1, 41),
            xlabel="Radius (pixels)",
            ylabel="Signal-to-noise ratio (log scale)",
        )
        ax2.set_yticks(
            [5, 10, 20, 50, 100, 200], labels=["5", "10", "20", "50", "100", "200"]
        )
        ax2.set_xticks([0, 10, 20, 30, 40])
        style_axis(ax2)
        ax2.grid(axis="y", color=GRID, lw=0.7)
        ax2.legend(
            loc="upper right",
            frameon=False,
            fontsize=9,
            handletextpad=0.4,
            borderaxespad=0.3,
            labelspacing=0.6,
        )

        # Paired dots preserve meaningful distances on a log scale, without
        # implying that logarithmic bar lengths are proportional to runtime.
        rows = np.arange(len(sizes))
        ax3.hlines(rows, rs_times, py_times, color="#C3CDD4", lw=2, zorder=1)
        ax3.scatter(rs_times, rows, color=ORANGE, s=48, zorder=3, label="Rust")
        ax3.scatter(
            py_times,
            rows,
            facecolors="white",
            edgecolors=BLUE,
            linewidths=1.7,
            marker="D",
            s=45,
            zorder=3,
            label="Python",
        )
        for i, (ref, rust) in enumerate(zip(py_times, rs_times)):
            ax3.annotate(
                format_time(rust),
                (rust, i),
                xytext=(0, -15),
                textcoords="offset points",
                ha="center",
                color=ORANGE,
                fontsize=9,
            )
            ax3.annotate(
                ("≈" if i == 5 else "") + format_time(ref),
                (ref, i),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                color=BLUE,
                fontsize=9,
            )
            ax3.text(
                1.055,
                i,
                f"{'≈' if i == 5 else ''}{ref / rust:.1f}×",
                transform=ax3.get_yaxis_transform(),
                va="center",
                fontsize=10,
                weight="bold",
            )
        ax3.set(
            xscale="log",
            xlim=(0.003, 400),
            ylim=(5.65, -0.65),
            xlabel="Runtime (log scale)",
        )
        ax3.set_yticks(rows, labels=sizes)
        ax3.set_xticks([0.01, 1, 100], labels=["10 ms", "1 s", "100 s"])
        style_axis(ax3)
        ax3.grid(axis="x", color=GRID, lw=0.7)
        ax3.tick_params(axis="y", length=0)
        ax3.spines["left"].set_visible(False)
        ax3.text(
            -0.04,
            1.075,
            "Pixels",
            transform=ax3.transAxes,
            ha="right",
            fontsize=9,
            color=MUTED,
        )
        ax3.text(
            1.055, 1.075, "Speedup", transform=ax3.transAxes, fontsize=9, color=MUTED
        )
        ax3.legend(
            loc="lower center",
            bbox_to_anchor=(0.52, 1.045),
            ncol=2,
            frameon=False,
            fontsize=9,
            handletextpad=0.3,
            columnspacing=1,
        )

        notes = [
            f"Smaller cells in the bright core.\n{np.sum(single)} single-pixel bins retained.",
            "Scatter = std / mean across multi-pixel bins.\nCapacity = (S/N)²; single pixels excluded from scatter.",
            "Apple Silicon, 10-core CPU · README benchmarks.\n≈180 s reference: Cappellari (2025, §6).",
        ]
        for left, note in zip(lefts, notes):
            fig.text(
                left, 0.093 / 0.83, note, fontsize=9.5, color=MUTED, linespacing=1.6
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    print(f"Showcase graphic saved to {output_path}")


if __name__ == "__main__":
    generate_showcase(
        Path(__file__).resolve().parent.parent / "assets" / "showcase.png"
    )
