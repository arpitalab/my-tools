"""
retrieve_similar.py — Parse one NSF XML and retrieve closest matches from the DB.

Usage (fast path with precomputed embeddings):
    python retrieve_similar.py \
        --xml /path/to/award.xml \
        --db ./output/nsf_awards.db \
        --embeddings ./output/embeddings.npy \
        --award-ids  ./output/award_ids.npy \
        --kg         ./output/nsf_kg.pkl \
        --top-n 5

Usage (slow path, original):
    python retrieve_similar.py \
        --xml /path/to/award.xml \
        --db ./output/nsf_awards.db \
        --year-pool 2021 \
        --top-n 5
"""
from __future__ import annotations

import sys
import os
import pickle
import sqlite3
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from nsf_xml_parser import NSFXMLParser


# ---------------------------------------------------------------------------
# Parse target XML
# ---------------------------------------------------------------------------

def parse_target(xml_path: str) -> dict:
    # Use a minimal parser instance — xml_dir is only needed for relpath in source_file
    parser = NSFXMLParser(xml_dir=os.path.dirname(xml_path), db_path=":memory:")
    tree = ET.parse(xml_path)
    award_el = tree.getroot().find("Award")
    if award_el is None:
        raise ValueError(f"No <Award> element in {xml_path}")
    fname = os.path.basename(xml_path)
    try:
        year = int(fname[:4])
    except ValueError:
        year = None
    return parser._extract_award(award_el, year, xml_path)


# ---------------------------------------------------------------------------
# Load pool from DB
# ---------------------------------------------------------------------------

def load_pool(db_path: str, year: int | None, limit: int | None) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    year_filter = "AND a.source_year = ?" if year else ""
    limit_clause = f"LIMIT {limit}" if limit else ""
    params = [year] if year else []
    rows = conn.execute(
        f"""
        SELECT
            a.award_id,
            a.title,
            COALESCE(a.abstract_narration, a.por_text, a.title) AS text,
            d.abbreviation  AS directorate,
            v.abbreviation  AS division
        FROM award a
        LEFT JOIN directorate d ON d.id = a.directorate_id
        LEFT JOIN division    v ON v.id = a.division_id
        WHERE COALESCE(a.abstract_narration, a.por_text, a.title) IS NOT NULL
        {year_filter}
        {limit_clause}
        """,
        params,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Precomputed embedding helpers (Change 1)
# ---------------------------------------------------------------------------

def load_precomputed(embeddings_path: str, award_ids_path: str) -> tuple[np.ndarray, list[str]]:
    """Load pre-normalized float32 embeddings + matching award_ids list."""
    embeddings = np.load(embeddings_path)           # already L2-normalized float32
    award_ids = np.load(award_ids_path, allow_pickle=True).tolist()
    return embeddings, award_ids


def fetch_metadata(db_path: str, award_ids: list[str]) -> dict[str, dict]:
    """Fetch title/directorate/division/text for a set of award_ids."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(award_ids))
    rows = conn.execute(
        f"""
        SELECT
            a.award_id,
            a.title,
            COALESCE(a.abstract_narration, a.por_text, a.title) AS text,
            d.abbreviation AS directorate,
            v.abbreviation AS division
        FROM award a
        LEFT JOIN directorate d ON d.id = a.directorate_id
        LEFT JOIN division    v ON v.id = a.division_id
        WHERE a.award_id IN ({placeholders})
        """,
        award_ids,
    ).fetchall()
    conn.close()
    return {r["award_id"]: dict(r) for r in rows}


# ---------------------------------------------------------------------------
# KG reranking (Change 3)
# ---------------------------------------------------------------------------

def kg_rerank(results: list[dict], query_pe_codes: set[str], kg, boost: float = 0.05) -> list[dict]:
    """
    Boost scores for results whose PE codes overlap with the query award's PE codes.
    query_pe_codes: set of bare PE code strings (e.g. {'139Y00', '806000'})
    kg: NetworkX graph with PE: prefix on program-element nodes.
    """
    for r in results:
        awd_node = f"AWD:{r['award_id']}"
        if not kg.has_node(awd_node):
            continue
        awd_pes = {
            nbr.replace("PE:", "")
            for nbr in kg.successors(awd_node)
            if nbr.startswith("PE:")
        }
        if awd_pes & query_pe_codes:
            r["score"] += boost
            r["kg_boosted"] = True
    return sorted(results, key=lambda x: -x["score"])


# ---------------------------------------------------------------------------
# Similarity retrieval (Change 2 fast-path + original slow-path)
# ---------------------------------------------------------------------------

def retrieve(
    query_text: str,
    pool: list[dict],
    model,
    top_n: int,
    precomputed_embs: np.ndarray | None = None,
    precomputed_ids: list[str] | None = None,
    db_path: str | None = None,
    kg=None,
    query_pe_codes: set[str] | None = None,
) -> list[dict]:
    # --- Fast path: precomputed embeddings ---
    if precomputed_embs is not None and precomputed_ids is not None:
        print("Encoding query …")
        query_emb = model.encode([query_text], normalize_embeddings=True)[0].astype(np.float32)
        # Full dot product (embeddings are already L2-normalized)
        scores = precomputed_embs @ query_emb
        top50_idx = np.argsort(scores)[::-1][:50]
        candidate_ids = [precomputed_ids[int(i)] for i in top50_idx]
        candidate_scores = {precomputed_ids[int(i)]: float(scores[i]) for i in top50_idx}

        meta = fetch_metadata(db_path, candidate_ids) if db_path else {}
        results = []
        for aid in candidate_ids:
            m = meta.get(aid, {"award_id": aid, "title": "", "text": "", "directorate": "", "division": ""})
            m["score"] = candidate_scores[aid]
            m.setdefault("kg_boosted", False)
            results.append(m)

        if kg is not None and query_pe_codes:
            results = kg_rerank(results, query_pe_codes, kg)

        return results[:top_n]

    # --- Slow path: encode full pool ---
    texts = [r["text"] for r in pool]
    print(f"Encoding {len(texts)} pool documents …")
    pool_embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)

    print("Encoding query …")
    query_emb = model.encode([query_text])[0]

    # Cosine similarity
    pool_norm = pool_embeddings / (np.linalg.norm(pool_embeddings, axis=1, keepdims=True) + 1e-10)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    scores = pool_norm @ query_norm

    top_idx = np.argsort(scores)[::-1][:top_n]
    results = []
    for i in top_idx:
        r = pool[int(i)].copy()
        r["score"] = float(scores[i])
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml",  required=True, help="Path to a single NSF XML award file")
    p.add_argument("--db",   default="./output/nsf_awards.db")
    p.add_argument("--year-pool", type=int, default=None,
                   help="Restrict pool to this source year (slow path only)")
    p.add_argument("--pool-limit", type=int, default=None,
                   help="Cap pool size for speed (slow path only)")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    # Fast-path args (Change 2)
    p.add_argument("--embeddings", default=None,
                   help="Path to precomputed embeddings.npy; enables fast path")
    p.add_argument("--award-ids", default=None,
                   help="Path to award_ids.npy; required when --embeddings is given")
    # KG reranking (Change 3)
    p.add_argument("--kg", default=None,
                   help="Path to nsf_kg.pkl; enables structural KG reranking")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Parse target
    print(f"\n=== TARGET ===")
    record = parse_target(args.xml)
    query_text = (
        record.get("abstract_narration")
        or record.get("por_text")
        or record.get("title")
        or ""
    )
    if not query_text:
        print("ERROR: no usable text in target XML")
        sys.exit(1)

    print(f"  Award ID   : {record['award_id']}")
    print(f"  Title      : {record['title']}")
    print(f"  Directorate: {record['dir_abbr']}  Division: {record['div_abbr']}")
    print(f"  Abstract   : {query_text[:200]} …\n")

    # 2. Load model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    # 3. Load optional precomputed embeddings + KG
    precomputed_embs = precomputed_ids = kg = None
    query_pe_codes: set[str] = set(record.get("program_elements") and
                                    [pe["code"] for pe in record["program_elements"]] or [])

    if args.embeddings:
        if not args.award_ids:
            print("ERROR: --award-ids is required when --embeddings is given")
            sys.exit(1)
        print("Loading precomputed embeddings …")
        precomputed_embs, precomputed_ids = load_precomputed(args.embeddings, args.award_ids)
        print(f"  {len(precomputed_ids):,} awards, shape={precomputed_embs.shape}")

    if args.kg:
        print("Loading knowledge graph …")
        with open(args.kg, "rb") as f:
            kg = pickle.load(f)
        print(f"  {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")

    # 4. Retrieve
    if precomputed_embs is not None:
        matches = retrieve(
            query_text, pool=[], model=model, top_n=args.top_n,
            precomputed_embs=precomputed_embs, precomputed_ids=precomputed_ids,
            db_path=args.db, kg=kg, query_pe_codes=query_pe_codes,
        )
    else:
        pool = load_pool(args.db, args.year_pool, args.pool_limit)
        print(f"Pool size: {len(pool)} awards")
        matches = retrieve(query_text, pool, model, args.top_n)

    # 5. Print results
    print(f"\n=== TOP {args.top_n} MATCHES ===")
    for rank, m in enumerate(matches, 1):
        boosted = "  [KG boosted]" if m.get("kg_boosted") else ""
        print(f"\n[{rank}] score={m['score']:.4f}  award={m['award_id']}"
              f"  dir={m.get('directorate','')}  div={m.get('division','')}{boosted}")
        print(f"     Title   : {m.get('title','')}")
        print(f"     Abstract: {(m.get('text') or '')[:250]} …")


if __name__ == "__main__":
    main()
