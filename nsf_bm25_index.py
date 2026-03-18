"""
nsf_bm25_index.py — BM25 index over NSF award abstracts.

BM25 (Okapi BM25) is a bag-of-words retrieval function that improves on TF-IDF
by saturating term frequency (rare terms matter but diminishing returns) and
normalizing by document length. Complements TF-IDF cosine in the hybrid scorer.

Usage:
    python nsf_bm25_index.py --db ./output/nsf_awards.db --output-dir ./output

Outputs:
    bm25_index.pkl     — BM25Okapi object (picklable)
    bm25_award_ids.npy — award_ids in same order as BM25 corpus
"""
from __future__ import annotations

import os
import re
import pickle
import sqlite3
import argparse

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_STOP = frozenset(
    "a an the and or but in on at to for of with by from is are was were be been "
    "being have has had do does did will would could should may might shall can "
    "this that these those it its we our they their he she his her i my you your "
    "us them him her which who what when where how not no nor so yet both either "
    "each few more most other some such than then there thus".split()
)

def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords and short tokens."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{1,}", text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 2]


# ---------------------------------------------------------------------------
# Load texts
# ---------------------------------------------------------------------------

def load_texts(db_path: str) -> tuple[list[str], list[str]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        """
        SELECT award_id,
               COALESCE(abstract_narration, por_text, title) AS text
        FROM award
        WHERE COALESCE(abstract_narration, por_text, title) IS NOT NULL
        ORDER BY id
        """
    ).fetchall()
    conn.close()
    return [r[0] for r in rows], [r[1] for r in rows]


# ---------------------------------------------------------------------------
# Build and save BM25
# ---------------------------------------------------------------------------

def build_bm25(texts: list[str]) -> tuple[BM25Okapi, list[list[str]]]:
    print(f"Tokenizing {len(texts):,} abstracts …")
    corpus = [tokenize(t) for t in tqdm(texts, unit="doc")]
    print("Building BM25 index …")
    index = BM25Okapi(corpus)
    return index, corpus


def save_outputs(
    output_dir: str,
    index: BM25Okapi,
    award_ids: list[str],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    idx_path = os.path.join(output_dir, "bm25_index.pkl")
    ids_path = os.path.join(output_dir, "bm25_award_ids.npy")

    tmp = idx_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, idx_path)

    np.save(ids_path, np.array(award_ids, dtype=object))

    size_mb = os.path.getsize(idx_path) / 1024 / 1024
    print(f"  {idx_path}  ({size_mb:.1f} MB)")
    print(f"  {ids_path}")


# ---------------------------------------------------------------------------
# Query helper (used by app at runtime)
# ---------------------------------------------------------------------------

def bm25_scores(query_text: str, index: BM25Okapi) -> np.ndarray:
    """
    Returns BM25 scores for all documents (same order as build corpus).
    Scores are raw BM25 values — normalize before combining with other signals.
    """
    tokens = tokenize(query_text)
    if not tokens:
        return np.zeros(len(index.idf), dtype=np.float32)
    return index.get_scores(tokens).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build BM25 index for NSF awards")
    p.add_argument("--db", default="./output/nsf_awards.db")
    p.add_argument("--output-dir", default="./output")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"DB: {args.db}")

    award_ids, texts = load_texts(args.db)
    print(f"Loaded {len(award_ids):,} award texts.")

    index, _ = build_bm25(texts)
    print("Saving outputs …")
    save_outputs(args.output_dir, index, award_ids)
    print("Done.")


if __name__ == "__main__":
    main()
