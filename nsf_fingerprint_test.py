"""
nsf_fingerprint_test.py — Digital fingerprinting via OpenAlex (field-agnostic).

Fetches abstracts from OpenAlex abstract_inverted_index for all fields
(geosciences, CS, physics, biology, engineering, etc.).
PubMed is NOT used — OpenAlex covers all NSF directorates.

Usage:
    python nsf_fingerprint_test.py          # 10 random 2024 awardees
    python nsf_fingerprint_test.py --named  # use NAMED_REVIEWERS list
"""
from __future__ import annotations

import argparse
import os
import random
import sqlite3
import time
from collections import defaultdict

import numpy as np
import pickle
import scipy.sparse as sp
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH   = "./output/nsf_awards.db"
EMB_PATH  = "./output/embeddings_specter2.npy"
IDS_PATH  = "./output/award_ids_specter2.npy"
CONCEPT_VEC   = "./output/concept_vectorizer.pkl"
CONCEPT_MAT   = "./output/concept_matrix.npz"
BM25_IDX      = "./output/bm25_index.pkl"
BM25_IDS      = "./output/bm25_award_ids.npy"

PAPER_YEARS = 5
MAX_PAPERS  = 30

OA_BASE  = "https://api.openalex.org"
OA_EMAIL = "spine.calcium@gmail.com"
OA_DELAY = 0.2   # seconds between OpenAlex requests

# Hardcoded list for --named mode
NAMED_REVIEWERS = [
    "Kevin Collins",
    "Mala Murthy",
    "Yehuda Ben-Shahar",
    "Michael Perry",
    "Amanda R Chappell",
    "Paul S Katz",
    "Mark A Bee",
]

# ---------------------------------------------------------------------------
# OpenAlex helpers
# ---------------------------------------------------------------------------

def _oa_get(path: str, params: dict) -> dict:
    """GET from OpenAlex with polite-pool header and rate limiting."""
    time.sleep(OA_DELAY)
    resp = requests.get(
        f"{OA_BASE}/{path}",
        params=params,
        headers={"User-Agent": f"mailto:{OA_EMAIL}"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """
    OpenAlex stores abstracts as {word: [pos1, pos2, ...]} — reconstruct
    into a plain string. Returns empty string if not available.
    """
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, locs in inverted_index.items():
        for pos in locs:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))


def openalex_resolve_author(name: str, institution: str) -> dict | None:
    """
    Search OpenAlex for the best-matching author.
    Validates top candidates against institution; falls back to top result.
    Rejects candidates with implausibly high work counts (wrong-person signal).
    """
    try:
        data = _oa_get("authors", {"search": name, "per_page": 8})
        results = data.get("results", [])
    except Exception as e:
        print(f"  [OpenAlex] search failed for '{name}': {e}")
        return None

    if not results:
        return None

    # Filter out mega-authors (wrong disambiguation signal)
    candidates = [r for r in results if r.get("works_count", 0) < 600]
    if not candidates:
        candidates = results   # if all are large, don't filter

    if institution:
        inst_words = {w.lower() for w in institution.split() if len(w) > 3}
        for r in candidates:
            for inst in (r.get("last_known_institutions") or []):
                oa_words = set((inst.get("display_name") or "").lower().split())
                if inst_words & oa_words:
                    return r

    return candidates[0]


def openalex_get_papers(oa_id: str, years: int = PAPER_YEARS) -> list[dict]:
    """
    Fetch recent works for an author via OpenAlex.
    Reconstructs abstracts from abstract_inverted_index.
    Falls back to title if no abstract available.
    Works for ALL fields — geosciences, CS, physics, biology, etc.
    """
    cutoff = 2026 - years
    try:
        data = _oa_get("works", {
            "filter":  f"authorships.author.id:{oa_id},"
                       f"publication_year:>{cutoff}",
            "per_page": MAX_PAPERS,
            "select":  "ids,title,publication_year,"
                       "abstract_inverted_index,topics,primary_location",
        })
    except Exception as e:
        print(f"  [OpenAlex] works fetch failed: {e}")
        return []

    papers = []
    for w in data.get("results", []):
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        text     = abstract if len(abstract) > 80 else (w.get("title") or "")
        topics   = [t["display_name"] for t in (w.get("topics") or [])[:5]]
        venue    = ((w.get("primary_location") or {})
                    .get("source") or {}).get("display_name", "")
        if text:
            papers.append({
                "title":    w.get("title") or "",
                "abstract": abstract,
                "text":     text,
                "year":     w.get("publication_year"),
                "topics":   topics,
                "venue":    venue,
                "has_abstract": len(abstract) > 80,
            })
    return papers


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_random_2024_awardees(n: int = 10, seed: int = 42) -> list[dict]:
    """
    Pick n PI-role investigators from 2024 awards, spread across different
    directorates for field diversity.
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT ai.full_name, ai.email,
               i.name  AS inst_name, i.state_code,
               a.award_id, a.title AS award_title,
               d.abbreviation AS dir_abbr, d.long_name AS dir_name
        FROM award_investigator ai
        JOIN award a      ON a.id = ai.award_id
        LEFT JOIN institution  i ON i.id = a.institution_id
        LEFT JOIN directorate  d ON d.id = a.directorate_id
        WHERE a.source_year = 2024
          AND ai.role_code  = 'Principal Investigator'
          AND ai.full_name IS NOT NULL
          AND LENGTH(ai.full_name) > 4
        ORDER BY a.award_id
        """
    ).fetchall()
    conn.close()

    # Group by directorate for diverse sampling
    by_dir: dict[str, list] = defaultdict(list)
    for r in rows:
        by_dir[r["dir_abbr"] or "UNKNOWN"].append(dict(r))

    random.seed(seed)
    selected = []
    dirs = list(by_dir.keys())
    random.shuffle(dirs)
    # Round-robin across directorates until we have n
    i = 0
    while len(selected) < n and i < n * 10:
        d = dirs[i % len(dirs)]
        if by_dir[d]:
            selected.append(random.choice(by_dir[d]))
            by_dir[d] = []   # don't reuse same directorate
        i += 1
    # Fill remainder if not enough directorates
    remaining = [r for rlist in by_dir.values() for r in rlist]
    random.shuffle(remaining)
    while len(selected) < n and remaining:
        selected.append(remaining.pop())

    # Aggregate award_ids per person from DB
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    result = []
    for r in selected:
        aids = [row["award_id"] for row in conn.execute(
            """SELECT a.award_id FROM award a
               JOIN award_investigator ai ON ai.award_id = a.id
               WHERE ai.full_name = ? AND ai.role_code = 'Principal Investigator'""",
            (r["full_name"],)
        ).fetchall()]
        result.append({**r, "award_ids": aids or [r["award_id"]]})
    conn.close()
    return result


def get_named_reviewers(names: list[str]) -> list[dict]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    result = []
    for name in names:
        parts = name.strip().split()
        rows = conn.execute(
            """SELECT ai.full_name, ai.email, i.name AS inst_name,
                      i.state_code, a.award_id,
                      d.abbreviation AS dir_abbr, d.long_name AS dir_name
               FROM award_investigator ai
               JOIN award a ON a.id = ai.award_id
               LEFT JOIN institution i ON i.id = a.institution_id
               LEFT JOIN directorate d ON d.id = a.directorate_id
               WHERE ai.full_name LIKE ? AND ai.full_name LIKE ?
               LIMIT 50""",
            (f"%{parts[0]}%", f"%{parts[-1]}%")
        ).fetchall()
        if rows:
            award_ids = list({r["award_id"] for r in rows})
            result.append({
                "full_name": rows[0]["full_name"],
                "email":     next((r["email"] for r in rows if r["email"]), ""),
                "inst_name": rows[0]["inst_name"] or "",
                "state_code":rows[0]["state_code"] or "",
                "award_ids": award_ids,
                "dir_abbr":  rows[0]["dir_abbr"] or "",
                "dir_name":  rows[0]["dir_name"] or "",
                "award_title": "",
            })
        else:
            result.append({"full_name": name, "email": "", "inst_name": "",
                           "state_code": "", "award_ids": [], "dir_abbr": "",
                           "dir_name": "", "award_title": ""})
    conn.close()
    return result


# ---------------------------------------------------------------------------
# Fingerprint builder
# ---------------------------------------------------------------------------

def build_fingerprint(award_ids: list[str], paper_texts: list[str],
                      model, emb: np.ndarray,
                      emb_ids: list[str]) -> np.ndarray | None:
    """
    Weighted mean: NSF award embeddings (30%) + paper text embeddings (70%).
    Falls back gracefully if one source is unavailable.
    """
    id2row = {aid: i for i, aid in enumerate(emb_ids)}
    nsf_rows = [emb[id2row[a]] for a in award_ids if a in id2row]

    vecs, weights = [], []
    if nsf_rows:
        vecs.append(np.stack(nsf_rows).mean(axis=0));  weights.append(0.3)

    good_texts = [t for t in paper_texts if len(t.strip()) > 60]
    if good_texts:
        paper_embs = model.encode(
            good_texts, normalize_embeddings=True,
            convert_to_numpy=True, batch_size=32,
        )
        vecs.append(paper_embs.mean(axis=0));  weights.append(0.7)

    if not vecs:
        return None
    combined = sum(w * v for w, v in zip(weights, vecs)) / sum(weights)
    norm = np.linalg.norm(combined)
    return combined / norm if norm > 1e-9 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--named", action="store_true",
                        help="Use NAMED_REVIEWERS list instead of random 2024 awardees")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 72)
    print("NSF Digital Fingerprint Test — OpenAlex (field-agnostic)")
    print("=" * 72)

    # ── Load indices ──────────────────────────────────────────────────────────
    print("\nLoading indices …", end=" ", flush=True)
    emb     = np.load(EMB_PATH)
    emb_ids = np.load(IDS_PATH, allow_pickle=True).tolist()
    print("done.")

    print("Loading SPECTER2 model …", end=" ", flush=True)
    from nsf_embeddings import SPECTER2Encoder
    model = SPECTER2Encoder("allenai/specter2")
    print("done.\n")

    # ── Select awardees ───────────────────────────────────────────────────────
    if args.named:
        awardees = get_named_reviewers(NAMED_REVIEWERS)
        print(f"Using {len(awardees)} named reviewers.\n")
    else:
        awardees = get_random_2024_awardees(n=10, seed=args.seed)
        print(f"Selected 10 random 2024 awardees (seed={args.seed}):\n")
        for a in awardees:
            print(f"  {a['full_name']:<32} {a['dir_abbr']:<6} "
                  f"{a['inst_name'][:40]}")
            print(f"    Award: {a['award_title'][:70]}")
        print()

    # ── Fingerprint each person ───────────────────────────────────────────────
    results = []
    for i, person in enumerate(awardees, 1):
        name = person["full_name"]
        inst = person["inst_name"]
        dirn = person.get("dir_abbr", "")
        print(f"── {i:2d}. {name}  [{dirn}]  ({inst})")

        # Resolve on OpenAlex
        oa      = openalex_resolve_author(name, inst)
        oa_id   = oa["id"].split("/")[-1] if oa else None
        oa_cnt  = oa.get("works_count", 0)  if oa else 0
        oa_cite = oa.get("cited_by_count", 0) if oa else 0
        orcid   = (oa or {}).get("orcid") or ""

        # Fetch papers via OpenAlex (field-agnostic)
        papers = openalex_get_papers(oa_id, years=PAPER_YEARS) if oa_id else []
        n_with_abs = sum(1 for p in papers if p["has_abstract"])

        # Aggregate topics across papers
        topic_freq: dict[str, int] = defaultdict(int)
        for p in papers:
            for t in p["topics"]:
                topic_freq[t] += 1
        top_topics = sorted(topic_freq, key=lambda x: -topic_freq[x])[:6]

        # Build fingerprint
        paper_texts = [p["text"] for p in papers]
        fp = build_fingerprint(person["award_ids"], paper_texts, model, emb, emb_ids)

        # NSF-only profile for comparison
        id2row = {a: i for i, a in enumerate(emb_ids)}
        nsf_rows = [emb[id2row[a]] for a in person["award_ids"] if a in id2row]
        if nsf_rows:
            nsf_profile = np.stack(nsf_rows).mean(axis=0)
            nsf_profile /= np.linalg.norm(nsf_profile)
        else:
            nsf_profile = None

        # Similarity of enriched profile vs NSF-only to the query
        # (Use first NSF award abstract as "self-similarity" anchor)
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        award_abstract = ""
        if person["award_ids"]:
            row = conn.execute(
                "SELECT COALESCE(abstract_narration, title) AS txt "
                "FROM award WHERE award_id = ?",
                (person["award_ids"][0],)
            ).fetchone()
            award_abstract = (row["txt"] or "") if row else ""
        conn.close()

        if award_abstract:
            q_emb = model.encode([award_abstract], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)
            sim_nsf = float(nsf_profile @ q_emb) if nsf_profile is not None else None
            sim_fp  = float(fp @ q_emb)          if fp is not None else None
        else:
            sim_nsf = sim_fp = None

        print(f"     OpenAlex : {oa_id or 'NOT FOUND'}  "
              f"({oa_cnt} works, {oa_cite:,} citations)"
              + (f"  ORCID ✓" if orcid else ""))
        print(f"     Papers   : {len(papers)} fetched, "
              f"{n_with_abs} with full abstract, "
              f"{len(papers)-n_with_abs} title-only")
        if top_topics:
            print(f"     Topics   : {', '.join(top_topics[:5])}")
        if papers:
            print(f"     Recent   :")
            for p in papers[:3]:
                marker = "●" if p["has_abstract"] else "○"
                print(f"       {marker} [{p['year']}] {p['title'][:80]}")
        s_nsf = f"{sim_nsf:.3f}" if sim_nsf is not None else "n/a"
        s_fp  = f"{sim_fp:.3f}"  if sim_fp  is not None else "n/a"
        print(f"     Self-sim : NSF-only={s_nsf}  Enriched={s_fp}  "
              f"{'▲ improved' if sim_fp and sim_nsf and sim_fp > sim_nsf else ''}")
        print()

        results.append({
            **person,
            "oa_id": oa_id, "oa_works": oa_cnt, "oa_citations": oa_cite,
            "orcid": orcid, "n_papers": len(papers),
            "n_with_abstract": n_with_abs, "top_topics": top_topics,
            "sim_nsf": sim_nsf, "sim_fp": sim_fp,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"{'#':<3} {'Name':<28} {'Dir':<5} {'OA?':<4} "
          f"{'Papers':>6} {'Abs':>4} {'NSF':>6} {'FP':>6}")
    print("-" * 72)
    for i, r in enumerate(results, 1):
        oa_ok   = "✓" if r["oa_id"] else "✗"
        s_nsf   = f"{r['sim_nsf']:.3f}" if r["sim_nsf"] is not None else "  n/a"
        s_fp    = f"{r['sim_fp']:.3f}"  if r["sim_fp"]  is not None else "  n/a"
        arrow   = "▲" if r["sim_fp"] and r["sim_nsf"] and r["sim_fp"] > r["sim_nsf"] else " "
        print(f"{i:<3} {r['full_name']:<28} {r.get('dir_abbr',''):<5} {oa_ok:<4}"
              f"{r['n_papers']:>6} {r['n_with_abstract']:>4} "
              f"{s_nsf:>6} {s_fp:>6} {arrow}")

    resolved   = sum(1 for r in results if r["oa_id"])
    has_papers = sum(1 for r in results if r["n_papers"] > 0)
    improved   = sum(1 for r in results
                     if r["sim_fp"] and r["sim_nsf"] and r["sim_fp"] > r["sim_nsf"])
    print("=" * 72)
    print(f"OpenAlex resolved: {resolved}/{len(results)}  |  "
          f"Got papers: {has_papers}/{len(results)}  |  "
          f"FP improved over NSF-only: {improved}/{len(results)}")


if __name__ == "__main__":
    main()
