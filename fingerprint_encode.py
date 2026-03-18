"""
fingerprint_encode.py — Phase 2: encode crawled papers and build fingerprints.

Loads all paper texts from researcher_papers (written by fingerprint_crawl.py),
deduplicates identical texts (papers co-authored by multiple PIs are encoded
once and reused), batch-encodes with SPECTER2, then assembles per-researcher
weighted fingerprints:

    fingerprint = 0.30 × mean(NSF award embeddings)
                + 0.70 × mean(paper embeddings)

Fallback: if no papers were crawled, fingerprint = NSF-only profile.
Stores results in researcher_fingerprint table.

Usage:
    python fingerprint_encode.py
    python fingerprint_encode.py --model allenai/specter2 --batch-size 128
    python fingerprint_encode.py --overwrite   # re-encode already-done researchers
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH  = "./output/nsf_awards.db"
EMB_PATH = "./output/embeddings_specter2.npy"
IDS_PATH = "./output/award_ids_specter2.npy"

W_NSF    = 0.30   # weight for NSF award embeddings
W_PAPERS = 0.70   # weight for paper embeddings


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS researcher_fingerprint (
    investigator_key  TEXT PRIMARY KEY,
    full_name         TEXT,
    inst_name         TEXT,
    openalex_id       TEXT,
    orcid             TEXT,
    n_nsf_awards      INTEGER,
    n_papers          INTEGER,
    n_with_abstract   INTEGER,
    top_topics        TEXT,      -- JSON array, top-6 by frequency
    embedding         BLOB,      -- float32 numpy array, serialized
    model_name        TEXT,
    built_at          TEXT
);
"""


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


# ---------------------------------------------------------------------------
# Load NSF award embeddings
# ---------------------------------------------------------------------------

def load_award_embeddings(emb_path: str, ids_path: str) -> tuple[np.ndarray, dict[str, int]]:
    """Returns (embedding matrix, award_id → row_index map)."""
    emb  = np.load(emb_path)                                  # (N, D) float32
    ids  = np.load(ids_path, allow_pickle=True).tolist()
    return emb, {aid: i for i, aid in enumerate(ids)}


# ---------------------------------------------------------------------------
# Load crawled papers from DB
# ---------------------------------------------------------------------------

def load_researcher_papers(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """
    Returns {investigator_key: [paper_dict, ...]} for all crawled papers.
    """
    rows = conn.execute(
        """SELECT investigator_key, paper_text, has_abstract, topics
           FROM researcher_papers
           WHERE paper_text IS NOT NULL AND LENGTH(paper_text) > 30"""
    ).fetchall()

    by_key: dict[str, list[dict]] = {}
    for r in rows:
        key = r["investigator_key"]
        if key not in by_key:
            by_key[key] = []
        by_key[key].append({
            "text":         r["paper_text"],
            "has_abstract": r["has_abstract"],
            "topics":       json.loads(r["topics"] or "[]"),
        })
    return by_key


def load_crawl_log(conn: sqlite3.Connection) -> dict[str, dict]:
    """Returns {investigator_key: crawl_log_row}."""
    rows = conn.execute(
        "SELECT investigator_key, full_name, inst_name, openalex_id, orcid "
        "FROM researcher_crawl_log"
    ).fetchall()
    return {r["investigator_key"]: dict(r) for r in rows}


def load_researcher_award_ids(conn: sqlite3.Connection,
                               keys: list[str]) -> dict[str, list[str]]:
    """
    Returns {investigator_key: [award_id, ...]} for the given keys.
    Uses the same key function as fingerprint_crawl.py.
    """
    # Build a temp mapping full_name → key
    rows = conn.execute(
        """SELECT ai.full_name, ai.email, a.award_id
           FROM award_investigator ai
           JOIN award a ON a.id = ai.award_id
           WHERE ai.role_code = 'Principal Investigator'"""
    ).fetchall()

    def _key(name: str, email: str) -> str | None:
        em = (email or "").strip().lower()
        if em:
            return f"e:{em}"
        n = (name or "").strip().lower()
        parts = [p.strip(".,") for p in n.split() if len(p.strip(".,")) > 1]
        return f"n:{' '.join(parts)}" if parts else None

    key_set = set(keys)
    result: dict[str, list[str]] = {}
    for r in rows:
        k = _key(r["full_name"], r["email"] or "")
        if k and k in key_set:
            result.setdefault(k, []).append(r["award_id"])
    return result


# ---------------------------------------------------------------------------
# Text deduplication helper
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# Fingerprint assembly
# ---------------------------------------------------------------------------

def assemble_fingerprint(
    award_ids:   list[str],
    paper_texts: list[str],
    award_emb:   np.ndarray,
    id2row:      dict[str, int],
    text2vec:    dict[str, np.ndarray],
) -> np.ndarray | None:
    """
    Build weighted fingerprint from NSF award embeddings + paper embeddings.
    Returns L2-normalized float32 vector, or None if no data available.
    """
    vecs:    list[np.ndarray] = []
    weights: list[float]      = []

    # NSF component
    nsf_rows = [award_emb[id2row[a]] for a in award_ids if a in id2row]
    if nsf_rows:
        nsf_mean = np.stack(nsf_rows).mean(axis=0).astype(np.float32)
        vecs.append(nsf_mean);  weights.append(W_NSF)

    # Papers component
    paper_vecs = [text2vec[_text_hash(t)] for t in paper_texts
                  if _text_hash(t) in text2vec]
    if paper_vecs:
        paper_mean = np.stack(paper_vecs).mean(axis=0).astype(np.float32)
        vecs.append(paper_mean);  weights.append(W_PAPERS)

    if not vecs:
        return None

    total_w  = sum(weights)
    combined = sum(w * v for w, v in zip(weights, vecs)) / total_w
    combined = combined.astype(np.float32)
    norm     = np.linalg.norm(combined)
    return combined / norm if norm > 1e-9 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode researcher fingerprints")
    p.add_argument("--db",         default=DB_PATH)
    p.add_argument("--emb-path",   default=EMB_PATH)
    p.add_argument("--ids-path",   default=IDS_PATH)
    p.add_argument("--model",      default="allenai/specter2",
                   help="Embedding model (must match emb-path)")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Encoding batch size (default 128; lower if OOM)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Re-encode researchers already in researcher_fingerprint")
    p.add_argument("--db-batch",   type=int, default=500,
                   help="Write fingerprints to DB every N researchers")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _apply_schema(conn)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading NSF award embeddings …", end=" ", flush=True)
    award_emb, id2row = load_award_embeddings(args.emb_path, args.ids_path)
    print(f"done — {award_emb.shape[0]:,} awards, dim={award_emb.shape[1]}")

    print("Loading crawled papers from DB …", end=" ", flush=True)
    papers_by_key = load_researcher_papers(conn)
    crawl_log     = load_crawl_log(conn)
    print(f"done — {len(crawl_log):,} crawled researchers, "
          f"{sum(len(v) for v in papers_by_key.values()):,} paper rows")

    all_keys = list(crawl_log.keys())

    # Skip already-encoded unless --overwrite
    if not args.overwrite:
        done = {r["investigator_key"] for r in
                conn.execute(
                    "SELECT investigator_key FROM researcher_fingerprint"
                ).fetchall()}
        all_keys = [k for k in all_keys if k not in done]
        print(f"Already encoded: {len(crawl_log)-len(all_keys):,}  |  "
              f"Remaining: {len(all_keys):,}")
    if not all_keys:
        print("Nothing to encode.")
        conn.close()
        return

    # ── Load award IDs per researcher ─────────────────────────────────────────
    print("Mapping researchers to NSF award IDs …", end=" ", flush=True)
    award_ids_by_key = load_researcher_award_ids(conn, all_keys)
    print("done.")

    # ── Collect all unique paper texts for batch encoding ─────────────────────
    print("Deduplicating paper texts …", end=" ", flush=True)
    hash2text: dict[str, str] = {}
    for key in all_keys:
        for paper in papers_by_key.get(key, []):
            h = _text_hash(paper["text"])
            if h not in hash2text:
                hash2text[h] = paper["text"]
    unique_texts = list(hash2text.values())
    unique_hashes = list(hash2text.keys())
    print(f"{len(unique_texts):,} unique texts "
          f"(from {sum(len(papers_by_key.get(k,[])) for k in all_keys):,} total)")

    # ── Load model and batch-encode all unique texts ──────────────────────────
    print(f"Loading model '{args.model}' …", end=" ", flush=True)
    from nsf_embeddings import SPECTER2Encoder
    from sentence_transformers import SentenceTransformer

    if args.model in {"allenai/specter2", "allenai/specter2_aug2023refresh"}:
        model = SPECTER2Encoder(args.model)
    else:
        model = SentenceTransformer(args.model)
    print("done.")

    print(f"Encoding {len(unique_texts):,} unique texts "
          f"(batch_size={args.batch_size}) …")
    all_vecs = model.encode(
        unique_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    text2vec: dict[str, np.ndarray] = {
        h: all_vecs[i] for i, h in enumerate(unique_hashes)
    }
    print(f"Encoding done — shape={all_vecs.shape}")

    # ── Assemble fingerprints and write to DB ─────────────────────────────────
    print(f"\nAssembling fingerprints for {len(all_keys):,} researchers …")
    now   = datetime.now(timezone.utc).isoformat()
    batch: list[tuple] = []
    built = skipped = 0

    def _flush(batch: list[tuple]) -> None:
        with conn:
            conn.executemany(
                """INSERT OR REPLACE INTO researcher_fingerprint
                   (investigator_key, full_name, inst_name, openalex_id, orcid,
                    n_nsf_awards, n_papers, n_with_abstract, top_topics,
                    embedding, model_name, built_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                batch,
            )

    for key in tqdm(all_keys, unit="researcher"):
        log     = crawl_log.get(key, {})
        papers  = papers_by_key.get(key, [])
        a_ids   = award_ids_by_key.get(key, [])

        fp = assemble_fingerprint(
            a_ids,
            [p["text"] for p in papers],
            award_emb, id2row, text2vec,
        )
        if fp is None:
            skipped += 1
            continue

        # Topic frequency across papers
        from collections import Counter
        topic_cnt: Counter = Counter()
        for p in papers:
            for t in p["topics"]:
                topic_cnt[t] += 1
        top_topics = [t for t, _ in topic_cnt.most_common(6)]

        batch.append((
            key,
            log.get("full_name", ""),
            log.get("inst_name", ""),
            log.get("openalex_id"),
            log.get("orcid"),
            len(a_ids),
            len(papers),
            sum(p["has_abstract"] for p in papers),
            json.dumps(top_topics),
            fp.tobytes(),
            args.model,
            now,
        ))
        built += 1

        if len(batch) >= args.db_batch:
            _flush(batch)
            batch.clear()

    if batch:
        _flush(batch)

    total_fp = conn.execute(
        "SELECT COUNT(*) FROM researcher_fingerprint"
    ).fetchone()[0]
    conn.close()

    print(f"\nDone.  Built={built:,}  Skipped (no data)={skipped:,}")
    print(f"Total fingerprints in DB: {total_fp:,}")


if __name__ == "__main__":
    main()
