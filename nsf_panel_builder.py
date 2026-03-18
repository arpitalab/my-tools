"""
nsf_panel_builder.py — Reviewer panel construction for NSF proposals.

Given N proposal abstracts, identifies a pool of potential reviewers from the
NSF awards database (PIs/Co-PIs of similar funded awards) and produces a
constrained assignment:
  - Each proposal receives `n_per_proposal` reviewers (default 3)
  - No reviewer is assigned more than `max_load` proposals (default 8)
  - The returned pool is `coi_buffer` × the minimum needed (default 2×),
    giving program officers room to drop reviewers with conflicts of interest.

Standalone module — no Streamlit dependency.
"""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Proposal parsing
# ---------------------------------------------------------------------------

# Separators: lines of 3+ dashes/equals, or "Proposal N:" / "Abstract N:" headers
_SEP_RE = re.compile(
    r"^\s*(?:-{3,}|={3,}|#{3,})\s*$"
    r"|^\s*(?:proposal|abstract|submission)\s*#?\d+\s*[:\-–]\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_proposals(text: str) -> list[dict]:
    """
    Split pasted text into individual proposals.
    Separators: lines of 3+ dashes/equals, or 'Proposal N:' headers.
    Returns list of {label, abstract} dicts (empty abstracts are dropped).
    """
    parts = _SEP_RE.split(text)
    proposals = []
    for i, part in enumerate(parts):
        body = part.strip()
        if not body:
            continue
        # Try to extract a label from first non-empty line if it looks like a title
        lines = body.splitlines()
        first = lines[0].strip()
        if len(first) < 120 and len(lines) > 1 and not first.endswith("."):
            label = first
            abstract = "\n".join(lines[1:]).strip()
        else:
            label = f"Proposal {len(proposals) + 1}"
            abstract = body
        if abstract:
            proposals.append({"label": label, "abstract": abstract})
    return proposals


# ---------------------------------------------------------------------------
# Reviewer identity & deduplication
# ---------------------------------------------------------------------------

def _reviewer_key(row: dict) -> str:
    """
    Canonical identity key: email (lowercased) if non-empty,
    else 'last_name__first3' as a fuzzy fallback.
    """
    email = (row.get("email") or "").strip().lower()
    if email and "@" in email:
        return email
    last  = (row.get("last_name")  or row.get("full_name") or "").strip().lower()
    first = (row.get("first_name") or "").strip().lower()[:3]
    return f"__name__{last}__{first}" if last else f"__name__{row.get('full_name','?').lower()}"


def _merge_reviewer(existing: dict, new_row: dict) -> None:
    """Update existing reviewer record with any richer fields from new_row."""
    for field in ("full_name", "email", "inst_name", "state_code", "nsf_id"):
        if not existing.get(field) and new_row.get(field):
            existing[field] = new_row[field]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def fetch_investigators_bulk(conn: sqlite3.Connection, award_ids: list[str]) -> dict[str, list[dict]]:
    """
    Returns {award_id: [investigator_dict, ...]} for PIs and Co-PIs.
    Joins through institution for inst_name / state_code.
    """
    if not award_ids:
        return {}
    ph = ",".join("?" * len(award_ids))
    rows = conn.execute(
        f"""
        SELECT
            a.award_id,
            ai.nsf_id,
            ai.full_name,
            ai.first_name,
            ai.last_name,
            ai.email,
            ai.role_code,
            i.name        AS inst_name,
            i.state_code
        FROM award_investigator ai
        JOIN award a ON a.id = ai.award_id
        LEFT JOIN institution i ON i.id = a.institution_id
        WHERE a.award_id IN ({ph})
          AND ai.role_code IN (
              'Principal Investigator',
              'Co-Principal Investigator'
          )
        """,
        award_ids,
    ).fetchall()

    result: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        result[r[0]].append({
            "nsf_id":     r[1] or "",
            "full_name":  r[2] or "",
            "first_name": r[3] or "",
            "last_name":  r[4] or "",
            "email":      r[5] or "",
            "role_code":  r[6] or "",
            "inst_name":  r[7] or "",
            "state_code": r[8] or "",
        })
    return dict(result)


def fetch_reviewer_pe_codes(conn: sqlite3.Connection, award_ids: list[str]) -> dict[str, list[str]]:
    """Returns {award_id: [pe_code, ...]} for the given award_ids."""
    if not award_ids:
        return {}
    ph = ",".join("?" * len(award_ids))
    rows = conn.execute(
        f"""
        SELECT a.award_id, pe.code
        FROM award_program_element ape
        JOIN award a ON a.id = ape.award_id
        JOIN program_element pe ON pe.id = ape.program_element_id
        WHERE a.award_id IN ({ph})
        """,
        award_ids,
    ).fetchall()
    result: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        result[r[0]].append(r[1])
    return dict(result)


# ---------------------------------------------------------------------------
# Score matrix construction
# ---------------------------------------------------------------------------

def build_score_matrix(
    proposals: list[dict],
    retrieve_fn,          # callable(abstract_text) -> list[dict] with 'award_id', 'score_combined'
    conn: sqlite3.Connection,
    progress_cb=None,     # optional callable(i, n) for progress reporting
) -> tuple[np.ndarray, list[str], list[dict]]:
    """
    For each proposal, retrieve similar awards and extract PI/Co-PI candidates.

    Returns:
      score_matrix  — float32 (n_reviewers × n_proposals); entry [r,p] =
                       max retrieval score of reviewer r's awards for proposal p
      reviewer_keys — list[str], one per row of score_matrix
      reviewer_info — list[dict] with name, email, inst, pe_codes, n_awards, roles
    """
    n_proposals = len(proposals)

    # Per-proposal: {reviewer_key: max_score}
    proposal_scores: list[dict[str, float]] = [{} for _ in range(n_proposals)]
    # Global reviewer registry
    reviewer_registry: dict[str, dict] = {}
    # Reviewer → set of award_ids they appeared in (for PE code aggregation)
    reviewer_awards: dict[str, set[str]] = defaultdict(set)

    for p_idx, prop in enumerate(proposals):
        if progress_cb:
            progress_cb(p_idx, n_proposals)

        matches = retrieve_fn(prop["abstract"])

        # Bulk-fetch investigators for this proposal's matched awards
        award_ids = [m["award_id"] for m in matches]
        inv_map   = fetch_investigators_bulk(conn, award_ids)

        for match in matches:
            aid   = match["award_id"]
            score = float(match.get("score_combined", 0.0))
            for inv in inv_map.get(aid, []):
                key = _reviewer_key(inv)
                # Update registry
                if key not in reviewer_registry:
                    reviewer_registry[key] = {
                        "key":        key,
                        "full_name":  inv["full_name"],
                        "first_name": inv["first_name"],
                        "last_name":  inv["last_name"],
                        "email":      inv["email"],
                        "nsf_id":     inv["nsf_id"],
                        "inst_name":  inv["inst_name"],
                        "state_code": inv["state_code"],
                        "roles":      set(),
                        "n_awards":   0,
                    }
                else:
                    _merge_reviewer(reviewer_registry[key], inv)
                reviewer_registry[key]["roles"].add(inv["role_code"])
                reviewer_registry[key]["n_awards"] += 1
                reviewer_awards[key].add(aid)

                # Max score across this reviewer's awards for this proposal
                prev = proposal_scores[p_idx].get(key, 0.0)
                proposal_scores[p_idx][key] = max(prev, score)

    if progress_cb:
        progress_cb(n_proposals, n_proposals)

    # Build ordered list of all reviewer keys
    all_keys = list(reviewer_registry.keys())
    n_reviewers = len(all_keys)
    key_to_idx = {k: i for i, k in enumerate(all_keys)}

    # Score matrix
    score_matrix = np.zeros((n_reviewers, n_proposals), dtype=np.float32)
    for p_idx, p_scores in enumerate(proposal_scores):
        for key, score in p_scores.items():
            r_idx = key_to_idx[key]
            score_matrix[r_idx, p_idx] = score

    # Enrich reviewer_info with top PE codes
    all_award_ids_flat = [aid for awards in reviewer_awards.values() for aid in awards]
    pe_map = fetch_reviewer_pe_codes(conn, list(set(all_award_ids_flat)))

    reviewer_info = []
    for key in all_keys:
        info = dict(reviewer_registry[key])
        info["roles"] = list(info["roles"])
        # Aggregate PE codes across their matched awards
        pe_counter: dict[str, int] = defaultdict(int)
        for aid in reviewer_awards[key]:
            for code in pe_map.get(aid, []):
                pe_counter[code] += 1
        top_pe = sorted(pe_counter, key=lambda c: -pe_counter[c])[:3]
        info["top_pe_codes"] = top_pe
        reviewer_info.append(info)

    return score_matrix, all_keys, reviewer_info, dict(reviewer_awards)


# ---------------------------------------------------------------------------
# Profile-based score enrichment
# ---------------------------------------------------------------------------

def enrich_with_profiles(
    score_matrix: np.ndarray,
    reviewer_keys: list[str],
    reviewer_awards: dict[str, set[str]],
    proposals: list[dict],
    emb: np.ndarray,
    emb_ids: list[str],
    encode_fn,                      # callable(list[str]) -> np.ndarray  (L2-normalized)
    profile_threshold: float = 0.25,
) -> np.ndarray:
    """
    Enriches the score matrix with reviewer profile similarity.

    For each reviewer, computes a profile embedding as the mean of all their
    award embeddings, then scores that profile against every proposal embedding.
    Where the profile similarity exceeds `profile_threshold`, it replaces a
    zero entry (or boosts a low entry) in the score matrix.

    This ensures reviewers with broad, relevant expertise are considered even
    if none of their specific awards appeared in the top-N for a given proposal.

    Returns an enriched score_matrix (same shape, float32).
    """
    emb_id_to_row = {aid: i for i, aid in enumerate(emb_ids)}
    enriched = score_matrix.copy()

    # Encode all proposal abstracts once
    prop_texts = [p["abstract"] for p in proposals]
    prop_embs  = encode_fn(prop_texts)                  # (n_proposals, D)

    n_proposals = len(proposals)

    for r_idx, key in enumerate(reviewer_keys):
        award_ids = list(reviewer_awards.get(key, []))
        row_indices = [emb_id_to_row[aid] for aid in award_ids if aid in emb_id_to_row]
        if not row_indices:
            continue

        # Reviewer profile = mean of their award embeddings, re-normalized
        profile = emb[row_indices].mean(axis=0)
        norm = np.linalg.norm(profile)
        if norm < 1e-8:
            continue
        profile = (profile / norm).astype(np.float32)

        # Cosine similarity against all proposals (already L2-normalized)
        sims = prop_embs @ profile                      # (n_proposals,)

        for p_idx in range(n_proposals):
            sim = float(sims[p_idx])
            if sim >= profile_threshold:
                # Use max of retrieval-based score and profile similarity
                enriched[r_idx, p_idx] = max(enriched[r_idx, p_idx], sim)

    return enriched


# ---------------------------------------------------------------------------
# Constrained greedy assignment
# ---------------------------------------------------------------------------

def assign_panel(
    score_matrix: np.ndarray,
    reviewer_keys: list[str],
    reviewer_info: list[dict],
    n_per_proposal: int = 3,
    max_load: int = 8,
    coi_buffer: float = 2.0,
    max_panel_size: int | None = None,
) -> dict:
    """
    Greedy constrained assignment of reviewers to proposals.

    Panel size is capped at max_panel_size (default: ceil(3 × n_proposals / 6)).
    Only the top max_panel_size reviewers (by total score) are eligible for
    assignment; the rest become reserves automatically.

    Algorithm:
      1. Pre-select top max_panel_size reviewers by total relevance score.
      2. Among those, greedily assign: reviewer r → proposal p if both caps allow.
      3. Reserves = all other candidates ranked by total score, up to coi_buffer × panel.

    Returns dict with keys:
      assignments    — list[list[str]]: assignments[p] = [reviewer_key, ...]
      primary_keys   — set[str]: keys of assigned reviewers
      n_primary      — int
      pool_keys      — list[str]: primary (ordered) + reserves
      pool_info      — list[dict]
      score_matrix   — the input matrix (for display)
      reviewer_keys  — passed through
      coverage       — list[int]: how many reviewers each proposal got
      reviewer_load  — list[int]: how many proposals each reviewer was assigned
      max_panel_size — the cap that was applied
    """
    n_reviewers, n_proposals = score_matrix.shape

    # Auto-compute panel size cap: ceil(n_reviews_per_proposal × n_proposals / avg_load)
    if max_panel_size is None:
        import math
        max_panel_size = math.ceil(n_per_proposal * n_proposals / 6)
    max_panel_size = max(max_panel_size, n_per_proposal)  # always at least n_per_proposal

    # Pre-select the top-N reviewers by total score across all proposals
    total_scores = score_matrix.sum(axis=1)
    ranked_all = np.argsort(total_scores)[::-1]
    panel_idxs  = set(int(i) for i in ranked_all[:max_panel_size])

    # Flat list of (score, r, p) sorted descending — panel members only
    triples = [
        (float(score_matrix[r, p]), r, p)
        for r in range(n_reviewers) if r in panel_idxs
        for p in range(n_proposals)
        if score_matrix[r, p] > 0
    ]
    triples.sort(reverse=True)

    assignments: list[list[int]] = [[] for _ in range(n_proposals)]
    reviewer_load = np.zeros(n_reviewers, dtype=int)
    proposal_count = np.zeros(n_proposals, dtype=int)

    for score, r, p in triples:
        if proposal_count[p] >= n_per_proposal:
            continue
        if reviewer_load[r] >= max_load:
            continue
        assignments[p].append(r)
        reviewer_load[r] += 1
        proposal_count[p] += 1

    # Convert indices to keys
    assignments_keys = [[reviewer_keys[r] for r in asn] for asn in assignments]

    # Primary reviewer set (assigned to ≥1 proposal)
    primary_idxs = {r for asn in assignments for r in asn}
    n_primary = len(primary_idxs)

    # Reserves: unassigned candidates, ranked by total score, until pool = buffer × primary
    n_reserve_target = max(int(n_primary * (coi_buffer - 1.0)), 1)
    reserve_ranked = [
        int(i) for i in np.argsort(total_scores)[::-1]
        if i not in primary_idxs
    ][:n_reserve_target]

    # Pool ordering: primary first (by load desc, then score desc), then reserves (by score)
    primary_ordered = sorted(
        primary_idxs,
        key=lambda i: (-int(reviewer_load[i]), -float(total_scores[i])),
    )
    pool_idxs = primary_ordered + reserve_ranked

    pool_keys = [reviewer_keys[i] for i in pool_idxs]
    pool_info = []
    for i in pool_idxs:
        info = dict(reviewer_info[i])
        info["is_primary"]          = (i in primary_idxs)
        info["assigned_proposals"]  = int(reviewer_load[i])
        info["total_score"]         = float(total_scores[i])
        pool_info.append(info)

    return {
        "assignments":      assignments_keys,
        "primary_keys":     {reviewer_keys[i] for i in primary_idxs},
        "n_primary":        n_primary,
        "pool_keys":        pool_keys,
        "pool_info":        pool_info,
        "score_matrix":     score_matrix,
        "reviewer_keys":    reviewer_keys,
        "coverage":         proposal_count.tolist(),
        "reviewer_load":    reviewer_load.tolist(),
        "max_panel_size":   max_panel_size,
    }


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def assignments_to_csv(proposals: list[dict], result: dict) -> str:
    """Returns CSV string of proposal → reviewer assignments."""
    lines = ["Proposal Label,Reviewer 1,Email 1,Reviewer 2,Email 2,Reviewer 3,Email 3,Coverage"]
    key_to_info = {info["key"]: info for info in result["pool_info"]}
    for p_idx, prop in enumerate(proposals):
        asn_keys = result["assignments"][p_idx]
        coverage = result["coverage"][p_idx]
        cols = [prop["label"]]
        for k in asn_keys:
            info = key_to_info.get(k, {})
            cols.append(info.get("full_name", k))
            cols.append(info.get("email", ""))
        # Pad if fewer than 3
        while len(cols) < 7:
            cols.append("")
        cols.append(str(coverage))
        lines.append(",".join(f'"{c}"' for c in cols))
    return "\n".join(lines)


def pool_to_csv(result: dict) -> str:
    """Returns CSV string of the reviewer pool."""
    lines = ["Rank,Name,Email,Institution,State,NSF ID,Status,Assigned Proposals,Top PE Codes,Total Score"]
    for rank, info in enumerate(result["pool_info"], 1):
        status = "Primary" if info.get("is_primary") else "Reserve"
        pe = "|".join(info.get("top_pe_codes", []))
        row = [
            str(rank),
            info.get("full_name", ""),
            info.get("email", ""),
            info.get("inst_name", ""),
            info.get("state_code", ""),
            info.get("nsf_id", ""),
            status,
            str(info.get("assigned_proposals", 0)),
            pe,
            f"{info.get('total_score', 0):.4f}",
        ]
        lines.append(",".join(f'"{c}"' for c in row))
    return "\n".join(lines)
