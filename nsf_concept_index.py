"""
nsf_concept_index.py — TF-IDF concept index over NSF award abstracts.

Builds a domain-aware concept layer on top of the embedding index.
TF-IDF with 1-3 grams separates "topologically associating domains" (genomics)
from "topological spaces" (math) because the n-grams are domain-specific.

Usage:
    python nsf_concept_index.py --db ./output/nsf_awards.db --output-dir ./output

Outputs:
    concept_vectorizer.pkl    — fitted TfidfVectorizer
    concept_matrix.npz        — sparse float32 matrix (N × vocab)
    concept_award_ids.npy     — award_ids in same row order as matrix
    concept_top_terms.pkl     — dict: award_id → [(phrase, weight), ...]  (top-20 per award)
"""
from __future__ import annotations

import os
import pickle
import sqlite3
import argparse

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Load texts (same ORDER as embeddings.npy for consistent indexing)
# ---------------------------------------------------------------------------

def load_texts(db_path: str) -> tuple[list[str], list[str]]:
    """Returns (award_ids, texts) ordered by award.id — matches embeddings.npy row order."""
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
# Build TF-IDF index
# ---------------------------------------------------------------------------

def build_tfidf(
    texts: list[str],
    ngram_range: tuple = (1, 3),
    min_df: int = 5,
    max_df: float = 0.4,
    max_features: int = 80_000,
) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """
    Fit TF-IDF over all abstracts.
    ngram_range=(1,3) captures phrases like "topologically associating domains"
    min_df=5: concept must appear in ≥5 abstracts to be a node (filters noise)
    max_df=0.4: removes near-universal terms (e.g. "research", "study")
    sublinear_tf: log(tf+1) — reduces dominance of very frequent terms
    """
    print(f"Fitting TF-IDF on {len(texts):,} abstracts  "
          f"(ngram={ngram_range}, min_df={min_df}, max_df={max_df}, max_feat={max_features:,}) …")
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",  # min 2-char tokens
        dtype=np.float32,
    )
    matrix = vec.fit_transform(texts)
    print(f"Vocabulary size: {len(vec.vocabulary_):,}  Matrix: {matrix.shape}  "
          f"nnz={matrix.nnz:,}")
    return vec, matrix.astype(np.float32)


# ---------------------------------------------------------------------------
# Build per-award top-term lookup (for display)
# ---------------------------------------------------------------------------

def build_top_terms(
    award_ids: list[str],
    matrix: sp.csr_matrix,
    feature_names: list[str],
    top_k: int = 20,
) -> dict[str, list[tuple[str, float]]]:
    """For each award, store its top-K TF-IDF concept phrases for UI display."""
    print(f"Extracting top-{top_k} concepts per award …")
    result: dict[str, list[tuple[str, float]]] = {}
    for i, aid in enumerate(tqdm(award_ids, unit="award")):
        row = matrix[i]
        if row.nnz == 0:
            result[aid] = []
            continue
        indices = row.indices
        data = row.data
        top_idx = np.argsort(data)[::-1][:top_k]
        result[aid] = [(feature_names[indices[j]], float(data[j])) for j in top_idx]
    return result


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: str,
    vectorizer: TfidfVectorizer,
    matrix: sp.csr_matrix,
    award_ids: list[str],
    top_terms: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    vec_path = os.path.join(output_dir, "concept_vectorizer.pkl")
    mat_path = os.path.join(output_dir, "concept_matrix.npz")
    ids_path = os.path.join(output_dir, "concept_award_ids.npy")
    top_path = os.path.join(output_dir, "concept_top_terms.pkl")

    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    sp.save_npz(mat_path, matrix)
    np.save(ids_path, np.array(award_ids, dtype=object))
    with open(top_path, "wb") as f:
        pickle.dump(top_terms, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  {vec_path}  (vocabulary: {len(vectorizer.vocabulary_):,} terms)")
    print(f"  {mat_path}  ({matrix.shape[0]:,} × {matrix.shape[1]:,}  "
          f"{os.path.getsize(mat_path)/1024/1024:.1f} MB)")
    print(f"  {ids_path}")
    print(f"  {top_path}")


def load_concept_index(output_dir: str):
    """Load all concept index artifacts. Returns (vectorizer, matrix, award_ids, top_terms)."""
    with open(os.path.join(output_dir, "concept_vectorizer.pkl"), "rb") as f:
        vec = pickle.load(f)
    mat = sp.load_npz(os.path.join(output_dir, "concept_matrix.npz"))
    ids = np.load(os.path.join(output_dir, "concept_award_ids.npy"), allow_pickle=True).tolist()
    with open(os.path.join(output_dir, "concept_top_terms.pkl"), "rb") as f:
        top_terms = pickle.load(f)
    return vec, mat, ids, top_terms


# ---------------------------------------------------------------------------
# Query helpers (used by app at runtime)
# ---------------------------------------------------------------------------

def concept_scores(
    query_text: str,
    vectorizer: TfidfVectorizer,
    matrix: sp.csr_matrix,
) -> np.ndarray:
    """
    Compute TF-IDF cosine similarity between query and all awards.
    Matrix rows are already L2-normalized (TF-IDF uses norm='l2' by default).
    Returns dense float32 array of shape (N,).
    """
    q_vec = vectorizer.transform([query_text])           # (1 × vocab) sparse
    scores = (matrix @ q_vec.T).toarray().ravel()        # (N,) via sparse dot
    return scores.astype(np.float32)


def matched_concepts(
    query_text: str,
    award_id: str,
    vectorizer: TfidfVectorizer,
    top_terms: dict[str, list[tuple[str, float]]],
    top_k: int = 5,
) -> list[str]:
    """Return concept phrases that appear in both the query and the award abstract."""
    q_vec = vectorizer.transform([query_text])
    q_nonzero = set(vectorizer.get_feature_names_out()[q_vec.indices])
    award_concepts = [phrase for phrase, _ in top_terms.get(award_id, [])]
    matched = [c for c in award_concepts if c in q_nonzero]
    return matched[:top_k]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build TF-IDF concept index for NSF awards")
    p.add_argument("--db", default="./output/nsf_awards.db")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--ngram-max", type=int, default=3,
                   help="Max n-gram size (default 3: unigram+bigram+trigram)")
    p.add_argument("--min-df", type=int, default=5,
                   help="Min document frequency for a term to be a concept")
    p.add_argument("--max-features", type=int, default=80_000)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"DB: {args.db}")

    award_ids, texts = load_texts(args.db)
    print(f"Loaded {len(award_ids):,} award texts.")

    vectorizer, matrix = build_tfidf(
        texts,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_features=args.max_features,
    )

    feature_names = vectorizer.get_feature_names_out().tolist()
    top_terms = build_top_terms(award_ids, matrix, feature_names, top_k=20)

    print("\nSaving outputs …")
    save_outputs(args.output_dir, vectorizer, matrix, award_ids, top_terms)
    print("\nDone.")


if __name__ == "__main__":
    main()
