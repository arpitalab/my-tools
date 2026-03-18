"""
nsf_hic_map.py — Hi-C style award similarity map.

Computes pairwise cosine similarity between NSF awards sorted by
directorate → division and plots as a heatmap.  Tight red diagonal blocks
= research "TADs" (divisions whose awards are internally similar).
Off-diagonal warmth = cross-domain overlap.

Two modes
---------
centroid   One point per division (L2-normalised mean embedding).
           Produces a clean ~51×51 overview. Fast (<1 s).
sampled    Up to --n-per-div individual awards per division.
           Produces a richer NxN map. Heavier (~1500×1500 default).

Usage
-----
python nsf_hic_map.py \\
    --db          ./output/nsf_awards.db \\
    --embeddings  ./output/embeddings_specter.npy \\
    --award-ids   ./output/award_ids_specter.npy \\
    --mode        centroid \\
    --output-dir  ./output
"""
from __future__ import annotations

import os
import sqlite3
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Hi-C colormap: white → orange-red (matches genomics convention)
# ---------------------------------------------------------------------------

HIC_CMAP = LinearSegmentedColormap.from_list(
    "hic",
    ["#FFFFFF", "#FFE0D0", "#FF9966", "#CC2222", "#660000"],
    N=256,
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_division_map(db_path: str) -> dict[str, tuple[str, str, str]]:
    """award_id → (dir_abbr, div_abbr, div_long_name)  ordered by (dir, div)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        """
        SELECT a.award_id,
               dr.abbreviation AS dir,
               v.abbreviation  AS div,
               v.long_name     AS div_name
        FROM award a
        JOIN division    v  ON v.id  = a.division_id
        JOIN directorate dr ON dr.id = v.directorate_id
        ORDER BY dr.abbreviation, v.abbreviation, a.award_id
        """
    ).fetchall()
    conn.close()
    return {r[0]: (r[1], r[2], r[3] or r[2]) for r in rows}


def load_embeddings(emb_path: str, ids_path: str) -> tuple[np.ndarray, dict[str, int]]:
    emb = np.load(emb_path)
    ids = np.load(ids_path, allow_pickle=True).tolist()
    id2row = {aid: i for i, aid in enumerate(ids)}
    return emb, id2row


# ---------------------------------------------------------------------------
# Division ordering
# ---------------------------------------------------------------------------

def division_order(div_map: dict) -> list[tuple[str, str, str]]:
    """Return sorted unique (dir_abbr, div_abbr, div_name) tuples."""
    seen = {}
    for aid, (d, v, vn) in div_map.items():
        seen[(d, v)] = vn
    return [(d, v, vn) for (d, v), vn in sorted(seen.items())]


# ---------------------------------------------------------------------------
# Centroid mode
# ---------------------------------------------------------------------------

def compute_centroids(
    emb: np.ndarray,
    id2row: dict[str, int],
    div_map: dict[str, tuple],
    div_order: list[tuple],
) -> np.ndarray:
    """Return L2-normalized mean embedding per division, ordered by div_order."""
    # accumulate
    sums: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(emb.shape[1], dtype=np.float64))
    counts: dict[tuple, int] = defaultdict(int)
    for aid, (d, v, _) in div_map.items():
        row = id2row.get(aid)
        if row is None:
            continue
        sums[(d, v)] += emb[row].astype(np.float64)
        counts[(d, v)] += 1
    centroids = []
    for d, v, _ in div_order:
        key = (d, v)
        if counts[key] == 0:
            centroids.append(np.zeros(emb.shape[1], dtype=np.float32))
        else:
            c = sums[key] / counts[key]
            c /= np.linalg.norm(c) + 1e-10
            centroids.append(c.astype(np.float32))
    return np.stack(centroids)  # (n_div, D)


# ---------------------------------------------------------------------------
# Sampled mode
# ---------------------------------------------------------------------------

def sample_awards(
    emb: np.ndarray,
    id2row: dict[str, int],
    div_map: dict[str, tuple],
    div_order: list[tuple],
    n_per_div: int,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """
    Sample up to n_per_div awards per division.
    Returns:
        matrix   (N, D)
        labels   list of award_ids
        div_labels  list of div_abbr per row
        boundaries  cumulative row counts for div boundaries
    """
    div_awards: dict[tuple, list[str]] = defaultdict(list)
    for aid, (d, v, _) in div_map.items():
        div_awards[(d, v)].append(aid)

    rows_list, labels, div_labels, boundaries = [], [], [], []
    for d, v, _ in div_order:
        bucket = div_awards.get((d, v), [])
        rng = np.random.default_rng(seed=42)
        sample = rng.choice(bucket, size=min(n_per_div, len(bucket)), replace=False).tolist()
        for aid in sorted(sample):
            r = id2row.get(aid)
            if r is not None:
                rows_list.append(emb[r])
                labels.append(aid)
                div_labels.append(v)
        boundaries.append(len(rows_list))

    matrix = np.stack(rows_list) if rows_list else np.zeros((0, emb.shape[1]))
    return matrix, labels, div_labels, boundaries


# ---------------------------------------------------------------------------
# Directorate boundary positions
# ---------------------------------------------------------------------------

def dir_boundaries(div_order: list[tuple], div_sizes: list[int] | None = None) -> list[tuple]:
    """
    Returns list of (dir_abbr, start_idx, end_idx) where indices are into the div_order list
    (centroid mode) or into the flattened award array (sampled mode, using div_sizes).
    """
    result = []
    cur_dir = None
    start = 0
    cumulative = 0

    if div_sizes is None:
        # centroid mode: one row per division
        for i, (d, v, _) in enumerate(div_order):
            if d != cur_dir:
                if cur_dir is not None:
                    result.append((cur_dir, start, i))
                cur_dir = d
                start = i
        result.append((cur_dir, start, len(div_order)))
    else:
        # sampled mode: use cumulative sizes
        for i, (d, v, _) in enumerate(div_order):
            n = div_sizes[i]
            if d != cur_dir:
                if cur_dir is not None:
                    result.append((cur_dir, start, cumulative))
                cur_dir = d
                start = cumulative
            cumulative += n
        result.append((cur_dir, start, cumulative))

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Directorate colours (for boundary annotations)
DIR_COLORS = {
    "BIO": "#2ca02c", "CSE": "#1f77b4", "ENG": "#ff7f0e",
    "MPS": "#9467bd", "GEO": "#8c564b", "EDU": "#e377c2",
    "SBE": "#17becf", "TIP": "#bcbd22", "O/D": "#7f7f7f",
    "IRM": "#aec7e8", "BFA": "#c5b0d5", "OCIO": "#ffbb78",
}


def plot_hic(
    sim_matrix: np.ndarray,
    tick_labels: list[str],
    dir_blocks: list[tuple],
    title: str,
    output_path: str,
    figsize: float = 14.0,
    show_div_lines: bool = True,
    div_boundaries: list[int] | None = None,
) -> None:
    N = sim_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.88))

    # Clip lower tail to enhance contrast (like Hi-C display normalization)
    vmin = float(np.percentile(sim_matrix, 5))
    vmax = 1.0
    im = ax.imshow(sim_matrix, cmap=HIC_CMAP, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    # Division boundary lines (thin, grey)
    if show_div_lines and div_boundaries:
        for b in div_boundaries[:-1]:
            ax.axhline(b - 0.5, color="#AAAAAA", lw=0.5, alpha=0.7)
            ax.axvline(b - 0.5, color="#AAAAAA", lw=0.5, alpha=0.7)

    # Directorate boundary lines (thick, dark)
    for dir_abbr, start, end in dir_blocks:
        for pos in (start, end):
            ax.axhline(pos - 0.5, color="#222222", lw=1.8)
            ax.axvline(pos - 0.5, color="#222222", lw=1.8)

    # Directorate labels centred on their blocks (data coordinates)
    for dir_abbr, start, end in dir_blocks:
        mid = (start + end - 1) / 2
        color = DIR_COLORS.get(dir_abbr, "#333333")
        ax.text(mid, -1.8, dir_abbr, ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=color, clip_on=False)
        ax.text(-1.8, mid, dir_abbr, ha="right", va="center",
                fontsize=8, fontweight="bold", color=color, clip_on=False)

    # Tick labels
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlim(-2, N - 0.5)
    ax.set_ylim(N - 0.5, -2)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Cosine similarity", fontsize=10)

    ax.set_title(title, fontsize=13, pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}  ({os.path.getsize(output_path)//1024} KB)")


# ---------------------------------------------------------------------------
# Public API (used by streamlit app)
# ---------------------------------------------------------------------------

def build_centroid_matrix(
    db_path: str,
    emb_path: str,
    ids_path: str,
) -> tuple[np.ndarray, list[str], list[str], list[tuple]]:
    """
    Returns (sim_matrix, tick_labels, dir_abbrs_per_row, dir_blocks).
    Suitable for plotly heatmap in the Streamlit app.
    """
    div_map  = load_division_map(db_path)
    emb, id2row = load_embeddings(emb_path, ids_path)
    div_ord  = division_order(div_map)
    centroids = compute_centroids(emb, id2row, div_map, div_ord)
    sim      = centroids @ centroids.T
    np.fill_diagonal(sim, 1.0)

    tick_labels = [f"{d}/{v}" for d, v, _ in div_ord]
    dir_abbrs   = [d for d, v, _ in div_ord]
    dir_blocks  = dir_boundaries(div_ord)
    return sim, tick_labels, dir_abbrs, dir_blocks, div_ord


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Hi-C style NSF award similarity map")
    p.add_argument("--db",         default="./output/nsf_awards.db")
    p.add_argument("--embeddings", default="./output/embeddings_specter.npy")
    p.add_argument("--award-ids",  default="./output/award_ids_specter.npy")
    p.add_argument("--mode",       choices=["centroid", "sampled"], default="centroid")
    p.add_argument("--n-per-div",  type=int, default=30,
                   help="Awards per division in sampled mode")
    p.add_argument("--output-dir", default="./output")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading data …")
    div_map     = load_division_map(args.db)
    emb, id2row = load_embeddings(args.embeddings, args.award_ids)
    div_ord     = division_order(div_map)
    print(f"  {len(emb):,} awards · {len(div_ord)} divisions")

    if args.mode == "centroid":
        print("Computing division centroids …")
        centroids  = compute_centroids(emb, id2row, div_map, div_ord)
        sim        = centroids @ centroids.T
        np.fill_diagonal(sim, 1.0)
        tick_labels = [f"{d}/{v}" for d, v, _ in div_ord]
        dir_blocks  = dir_boundaries(div_ord)
        title = "NSF Award Space — Division Centroids (SPECTER)\nDiagonal blocks = cohesive research domains"
        out   = os.path.join(args.output_dir, "hic_centroid.png")
        plot_hic(sim, tick_labels, dir_blocks, title, out,
                 figsize=14, show_div_lines=False)

    else:  # sampled
        print(f"Sampling up to {args.n_per_div} awards per division …")
        matrix, labels, div_labels, cum_bounds = sample_awards(
            emb, id2row, div_map, div_ord, args.n_per_div
        )
        print(f"  {len(labels)} total awards sampled")
        print("Computing similarity matrix …")
        sim = matrix @ matrix.T
        np.fill_diagonal(sim, 1.0)

        # Div boundaries from cumulative counts
        div_sizes  = [cum_bounds[0]] + [cum_bounds[i] - cum_bounds[i-1]
                                         for i in range(1, len(cum_bounds))]
        dir_blocks = dir_boundaries(div_ord, div_sizes)
        tick_labels = div_labels
        title = (f"NSF Award Space — {len(labels)} sampled awards "
                 f"(≤{args.n_per_div}/division, SPECTER)\n"
                 "Diagonal blocks = cohesive research domains")
        out = os.path.join(args.output_dir, f"hic_sampled_{args.n_per_div}.png")
        plot_hic(sim, tick_labels, dir_blocks, title, out,
                 figsize=18, show_div_lines=True, div_boundaries=cum_bounds)

    print("Done.")


if __name__ == "__main__":
    main()
