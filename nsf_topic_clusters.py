"""
nsf_topic_clusters.py — Phase 1: Year-by-year topic clustering of NSF awards.

Strategy: fit ONE global BERTopic model over all awards using the existing
SPECTER2 embeddings (no re-encoding), then compute per-year cluster statistics.
This gives stable cluster IDs that are consistent across years — essential for
Phase 2 trajectory tracking.

Outputs (written to output_dir):
  topic_model/                — saved BERTopic model (reload without re-fitting)
  cluster_assignments.parquet — award_id, cluster_id, cluster_label, year,
                                directorate, division, award_amount
  cluster_year_stats.parquet  — per (cluster_id, year): size, growth_rate,
                                pct_new_pis, cross_dir_entropy, top_terms

These two parquets are the inputs for Phase 2 (trajectory analysis) and
Phase 4 (feature computation for breakthrough prediction).

Usage:
  python nsf_topic_clusters.py
  python nsf_topic_clusters.py --db ./output/nsf_awards.db --output-dir ./output
  python nsf_topic_clusters.py --refit   # force refit even if model exists
"""
from __future__ import annotations

import argparse
import math
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_HERE       = Path(__file__).parent
_OUTPUT     = _HERE / "output"
_DB         = _OUTPUT / "nsf_awards.db"
_EMB        = _OUTPUT / "embeddings_specter2.npy"
_IDS        = _OUTPUT / "award_ids_specter2.npy"
_MODEL_DIR  = _OUTPUT / "topic_model"
_ASSIGN_OUT = _OUTPUT / "cluster_assignments.parquet"
_STATS_OUT  = _OUTPUT / "cluster_year_stats.parquet"

MIN_TOPIC_SIZE = 40   # min awards per cluster globally
YEAR_START     = 2010
YEAR_END       = 2024


# ---------------------------------------------------------------------------
# 1. Load embeddings + award metadata
# ---------------------------------------------------------------------------

def load_embeddings(emb_path: Path, ids_path: Path) -> tuple[np.ndarray, list[str]]:
    print(f"Loading embeddings from {emb_path.name} …")
    emb = np.load(emb_path).astype(np.float32)
    ids = np.load(ids_path, allow_pickle=True).tolist()
    print(f"  {emb.shape[0]:,} awards  dim={emb.shape[1]}")
    return emb, ids


def load_metadata(db_path: Path, award_ids: list[str]) -> pd.DataFrame:
    """Load source_year, directorate, division, award_amount for each award_id."""
    print(f"Loading metadata from {db_path.name} …")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # Pull everything we need in one query
    rows = conn.execute(
        """
        SELECT  a.award_id,
                a.source_year                   AS year,
                a.award_amount,
                d.abbreviation                  AS directorate,
                v.abbreviation                  AS division,
                COALESCE(a.abstract_narration,
                         a.por_text, a.title)   AS text
        FROM    award a
        LEFT JOIN directorate d ON d.id = a.directorate_id
        LEFT JOIN division    v ON v.id = a.division_id
        ORDER BY a.id
        """
    ).fetchall()
    conn.close()

    meta = pd.DataFrame(rows, columns=["award_id","year","award_amount",
                                        "directorate","division","text"])
    meta["award_id"] = meta["award_id"].astype(str)

    # Keep only rows that have an embedding
    id_set = set(str(i) for i in award_ids)
    meta   = meta[meta["award_id"].isin(id_set)].copy()
    print(f"  {len(meta):,} awards with metadata  "
          f"years {int(meta.year.min())}–{int(meta.year.max())}")
    return meta


# ---------------------------------------------------------------------------
# 2. Fit (or load) BERTopic model
# ---------------------------------------------------------------------------

def fit_or_load_model(
    embeddings: np.ndarray,
    texts: list[str],
    model_dir: Path,
    refit: bool = False,
):
    """Return (topic_model, topics, probs)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    if model_dir.exists() and not refit:
        print(f"Loading saved BERTopic model from {model_dir} …")
        model = BERTopic.load(str(model_dir))
        print("  Transforming embeddings with loaded model …")
        topics, probs = model.transform(texts, embeddings)
        return model, topics, probs

    print("Fitting BERTopic on all awards …")
    print(f"  {len(texts):,} abstracts  embedding dim={embeddings.shape[1]}")

    umap_model = UMAP(
        n_components=15,        # higher than default for better cluster separation
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,   # needed for soft-clustering / transform
    )
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 3),
        stop_words="english",
        min_df=5,
        max_df=0.4,
    )

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=MIN_TOPIC_SIZE,
        calculate_probabilities=False,   # faster; we only need hard assignments
        verbose=True,
    )

    topics, probs = model.fit_transform(texts, embeddings)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir), serialization="safetensors",
               save_ctfidf=True, save_embedding_model=False)
    print(f"  Model saved to {model_dir}")

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_noise  = sum(1 for t in topics if t == -1)
    print(f"  {n_topics} topics found  |  {n_noise:,} noise awards (cluster=-1)")

    return model, topics, probs


# ---------------------------------------------------------------------------
# 3. Build cluster assignments dataframe
# ---------------------------------------------------------------------------

def build_assignments(
    model,
    topics: list[int],
    award_ids: list[str],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Join cluster assignments back to metadata."""
    print("Building cluster assignment table …")

    topic_info = model.get_topic_info()
    # topic_info has columns: Topic, Count, Name, Representation, ...
    label_map  = dict(zip(topic_info["Topic"], topic_info["Name"]))

    assign = pd.DataFrame({
        "award_id":   [str(a) for a in award_ids],
        "cluster_id": topics,
    })
    assign["cluster_label"] = assign["cluster_id"].map(label_map)

    # Merge with metadata (left join — keeps all embeddings rows)
    assign = assign.merge(
        meta[["award_id","year","directorate","division","award_amount"]],
        on="award_id", how="left",
    )

    # Filter to year range and drop noise if desired
    assign = assign[
        assign["year"].between(YEAR_START, YEAR_END)
    ].copy()

    print(f"  {len(assign):,} awards in {YEAR_START}–{YEAR_END}  "
          f"({(assign.cluster_id == -1).sum():,} noise)")
    return assign


# ---------------------------------------------------------------------------
# 4. Load PI names for pct_new_pis calculation
# ---------------------------------------------------------------------------

def load_pi_cluster_history(db_path: Path, assign: pd.DataFrame) -> pd.DataFrame:
    """
    For each award, get the PI name.  Returns dataframe:
      award_id, pi_name
    """
    print("Loading PI assignments …")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        """
        SELECT a.award_id, ai.full_name AS pi_name
        FROM   award_investigator ai
        JOIN   award a ON a.id = ai.award_id
        WHERE  ai.role_code = 'Principal Investigator'
        """
    ).fetchall()
    conn.close()

    pi_df = pd.DataFrame(rows, columns=["award_id","pi_name"])
    pi_df["award_id"] = pi_df["award_id"].astype(str)
    return pi_df


# ---------------------------------------------------------------------------
# 5. Compute per-year cluster statistics
# ---------------------------------------------------------------------------

def _entropy(counts: dict) -> float:
    """Shannon entropy of a value distribution (counts dict)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values() if c > 0
    )


def compute_cluster_year_stats(
    assign: pd.DataFrame,
    pi_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (cluster_id, year) compute:
      size              — number of awards
      total_funding     — sum of award_amount
      growth_rate       — (size_t - size_{t-1}) / size_{t-1}
      pct_new_pis       — fraction of PIs not seen in this cluster in prior years
      cross_dir_entropy — Shannon entropy of directorate distribution
    """
    print("Computing per-year cluster statistics …")

    # Join PI names
    data = assign.merge(pi_df, on="award_id", how="left")
    data = data[data["cluster_id"] != -1].copy()   # exclude noise

    years    = sorted(data["year"].dropna().unique().astype(int))
    clusters = sorted(data["cluster_id"].unique())

    # Track which PIs have been seen in each cluster (cumulative)
    pi_seen: dict[int, set] = defaultdict(set)   # cluster_id → set of pi_names

    records = []
    for yr in tqdm(years, desc="Year stats"):
        yr_data = data[data["year"] == yr]

        for cid in clusters:
            c_data = yr_data[yr_data["cluster_id"] == cid]
            if len(c_data) == 0:
                continue

            size     = len(c_data)
            funding  = c_data["award_amount"].sum()

            # Cross-directorate entropy
            dir_counts = c_data["directorate"].value_counts().to_dict()
            cross_ent  = _entropy(dir_counts)

            # pct_new_pis: PIs appearing in this cluster for the first time
            pis_this_year = set(c_data["pi_name"].dropna().tolist())
            new_pis = pis_this_year - pi_seen[cid]
            pct_new = len(new_pis) / len(pis_this_year) if pis_this_year else 0.0

            records.append({
                "cluster_id":        cid,
                "year":              yr,
                "size":              size,
                "total_funding":     funding,
                "cross_dir_entropy": round(cross_ent, 4),
                "pct_new_pis":       round(pct_new, 4),
            })

            # Update cumulative PI set
            pi_seen[cid].update(pis_this_year)

    stats = pd.DataFrame(records)

    # Growth rate: (size_t / size_{t-1}) - 1
    stats = stats.sort_values(["cluster_id","year"])
    stats["size_lag"]    = stats.groupby("cluster_id")["size"].shift(1)
    stats["growth_rate"] = ((stats["size"] - stats["size_lag"])
                            / stats["size_lag"].replace(0, np.nan))
    stats = stats.drop(columns=["size_lag"])

    print(f"  {len(stats):,} (cluster, year) rows across {len(clusters)} clusters")
    return stats


# ---------------------------------------------------------------------------
# 6. Attach top topic terms to stats
# ---------------------------------------------------------------------------

def attach_top_terms(stats: pd.DataFrame, model) -> pd.DataFrame:
    """Add a 'top_terms' column (pipe-separated keywords) from BERTopic."""
    topic_info = model.get_topic_info().set_index("Topic")
    def _terms(cid):
        if cid not in topic_info.index:
            return ""
        rep = topic_info.loc[cid, "Representation"]
        if isinstance(rep, list):
            return " | ".join(rep[:8])
        return str(rep)

    stats = stats.copy()
    stats["top_terms"] = stats["cluster_id"].map(_terms)
    return stats


# ---------------------------------------------------------------------------
# 7. Summary printout
# ---------------------------------------------------------------------------

def print_summary(assign: pd.DataFrame, stats: pd.DataFrame, model) -> None:
    topic_info = model.get_topic_info()
    top10      = topic_info[topic_info["Topic"] != -1].nlargest(10, "Count")

    print("\n─── Top 10 clusters by total award count ───")
    for _, row in top10.iterrows():
        print(f"  [{row['Topic']:>4}]  {row['Count']:>6,} awards  {row['Name']}")

    print("\n─── Awards per year ───")
    by_year = (assign[assign["cluster_id"] != -1]
               .groupby("year")["award_id"]
               .count()
               .sort_index())
    for yr, n in by_year.items():
        print(f"  {int(yr)}  {n:>6,}")

    print("\n─── Highest cross-directorate entropy clusters (any year) ───")
    top_ent = (stats.groupby("cluster_id")["cross_dir_entropy"]
               .mean()
               .sort_values(ascending=False)
               .head(10))
    ti_map  = dict(zip(topic_info["Topic"], topic_info["Name"]))
    for cid, ent in top_ent.items():
        print(f"  [{cid:>4}]  entropy={ent:.3f}  {ti_map.get(cid,'?')}")

    print("\n─── Fastest growing clusters (mean growth rate) ───")
    top_grow = (stats[stats["growth_rate"].notna()]
                .groupby("cluster_id")["growth_rate"]
                .mean()
                .sort_values(ascending=False)
                .head(10))
    for cid, gr in top_grow.items():
        print(f"  [{cid:>4}]  growth={gr:+.2f}  {ti_map.get(cid,'?')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: NSF topic clustering")
    p.add_argument("--db",         default=str(_DB),
                   help="Path to nsf_awards.db")
    p.add_argument("--output-dir", default=str(_OUTPUT),
                   help="Output directory (must contain embeddings_specter2.npy)")
    p.add_argument("--refit",      action="store_true",
                   help="Force refit even if saved model exists")
    p.add_argument("--year-start", type=int, default=YEAR_START)
    p.add_argument("--year-end",   type=int, default=YEAR_END)
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    db_path    = Path(args.db)
    emb_path   = output_dir / "embeddings_specter2.npy"
    ids_path   = output_dir / "award_ids_specter2.npy"
    model_dir  = output_dir / "topic_model"
    assign_out = output_dir / "cluster_assignments.parquet"
    stats_out  = output_dir / "cluster_year_stats.parquet"

    global YEAR_START, YEAR_END
    YEAR_START = args.year_start
    YEAR_END   = args.year_end

    # ── Step 1: embeddings + metadata ──────────────────────────────────────
    embeddings, award_ids = load_embeddings(emb_path, ids_path)
    meta                  = load_metadata(db_path, award_ids)

    # Align: keep only award_ids that have metadata, in embedding order
    meta_index = meta.set_index("award_id")
    valid_mask = [str(aid) in meta_index.index for aid in award_ids]
    embeddings = embeddings[valid_mask]
    award_ids  = [aid for aid, ok in zip(award_ids, valid_mask) if ok]
    texts      = [meta_index.loc[str(aid), "text"] or "" for aid in award_ids]

    print(f"\nAligned: {len(award_ids):,} awards with both embeddings and metadata\n")

    # ── Step 2: fit or load BERTopic ───────────────────────────────────────
    model, topics, _ = fit_or_load_model(
        embeddings, texts, model_dir, refit=args.refit
    )

    # ── Step 3: cluster assignments ────────────────────────────────────────
    assign = build_assignments(model, topics, award_ids, meta)
    assign.to_parquet(assign_out, index=False)
    print(f"Saved {assign_out.name}  ({len(assign):,} rows)")

    # ── Step 4: PI data ────────────────────────────────────────────────────
    pi_df = load_pi_cluster_history(db_path, assign)

    # ── Step 5: per-year cluster statistics ────────────────────────────────
    stats = compute_cluster_year_stats(assign, pi_df)
    stats = attach_top_terms(stats, model)
    stats.to_parquet(stats_out, index=False)
    print(f"Saved {stats_out.name}  ({len(stats):,} rows)")

    # ── Step 6: summary ────────────────────────────────────────────────────
    print_summary(assign, stats, model)

    print("\nPhase 1 complete.")
    print(f"  {assign_out}")
    print(f"  {stats_out}")
    print(f"  {model_dir}/")
    print("\nNext: run nsf_topic_trajectories.py for Phase 2 (trajectory tracking)")


if __name__ == "__main__":
    main()
