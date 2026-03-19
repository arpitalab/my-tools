"""
nsf_solr_agent.py — Ollama agent over NSF SOLR + local SQLite database.

Two data sources:
  SOLR   — all proposals (awarded + declined + pending), full text, panel/reviewer data
  SQLite — 162k awarded grants, SPECTER2 semantic search, PI profiles, SQL analytics

Default: ollama native tool-calling (llama3.1, llama3.2, qwen2.5).
Fallback: --react flag for models without tool support (phi4).

Usage:
    python nsf_solr_agent.py                               # llama3.1 (default)
    python nsf_solr_agent.py --model llama3.2              # smaller/faster
    python nsf_solr_agent.py --model phi4 --react          # ReAct fallback
    python nsf_solr_agent.py --out panel_report.md         # save session to file
    python nsf_solr_agent.py --output /path/to/output      # custom index directory

Requirements:
    pip install ollama pysolr numpy scipy
    (semantic/hybrid search also needs: sentence-transformers)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle
import re
import sqlite3
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np
import ollama
import pysolr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOLR_URL = "http://dis-checker-a01.ad.nsf.gov/solr/proposals/"
TIMEOUT  = 30

FIELD_GROUPS = {
    "fi_pid": "id",
    "fi_typ": "lead_id, lead_proposal",
    "fi_dat": "received, received_year, load_date, pm_rcom_date, received_to_rcom_days",
    "fi_stu": "status",
    "fi_tle": "title",
    "fi_sum": "summary",
    "fi_des": "description, description_length",
    "fi_bio": "bio",
    "fi_dmp": "data_management",
    "fi_fac": "facilities",
    "fi_mem": "mentoring",
    "fi_ref": "references",
    "fi_sup": "supplementary",
    "fi_add": "additional_documents",
    "fi_cap": "support",
    "fi_bju": "budget",
    "fi_uni": "directorate, division, managing_program, managing_program_code",
    "fi_req": "natr_rqst_code, rqst_mnth_cnt, requested_amount",
    "fi_sub": "program_announcement, program_director, prop_po_name",
    "fi_coi": "proposal_coi_flag",
    "fi_awd": "award_amount, award_date, awd_exp_date, awd_istr_code, awd_po_name, dd_rcom_date, funding_program, funding_program_code, funding_program_count",
    "fi_pia": "pi_id, inst_attr, pi_degree, pi_degree_year, pi_race, pi_disability, pi_ethnicity, pi_gender, pi_project_role",
    "fi_pin": "pi_all, pi_email, pi_name, inst, inst_state, pi_inst, pi_city, pi_state",
    "fi_snr": "senior_name, senior_inst, senior_title",
    "fi_sgr": "suggested_reviewers",
    "fi_pan": "panel_id",
    "fi_pnl": "panel_name, panel_org_code, panel_count, panel_end_date, panel_reviewers, panel_start_date",
    "fi_rwa": "reviewer_id, reviewer_count, reviewer_disability, reviewer_gender, reviewer_ethnicity, reviewer_race, reviewer_status",
    "fi_rwn": "reviewer_all, reviewer_name, reviewer_department, reviewer_email, reviewer_inst",
    "fi_cor": "collaborators",
    "fi_int": "foreign_colb_country, foreign_colb_country_code, foreign_country, foreign_country_code, intl_actv_flag, nsf_fund_trav_intl_flag",
    "fi_oth": "obj_clas_code, pdf_location, prc_code, summary_length, deviation_authorization, corpus",
}


# SQLite / index paths — override with --output flag at CLI
_OUTPUT = Path(__file__).parent / "output"

DB_PATH     = _OUTPUT / "nsf_awards.db"
EMB_PATH    = _OUTPUT / "embeddings_specter2.npy"
IDS_PATH    = _OUTPUT / "award_ids_specter2.npy"
CONCEPT_VEC = _OUTPUT / "concept_vectorizer.pkl"
CONCEPT_MAT = _OUTPUT / "concept_matrix.npz"
CONCEPT_IDS = _OUTPUT / "concept_award_ids.npy"
BM25_IDX    = _OUTPUT / "bm25_index.pkl"
BM25_IDS    = _OUTPUT / "bm25_award_ids.npy"


def _solr() -> pysolr.Solr:
    return pysolr.Solr(SOLR_URL, timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# SQLite lazy loaders
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH} (use --output to set path)")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@lru_cache(maxsize=1)
def _embeddings():
    emb = np.load(EMB_PATH)
    ids = np.load(IDS_PATH, allow_pickle=True).tolist()
    return emb, ids


@lru_cache(maxsize=1)
def _concept_index():
    import scipy.sparse as sp
    with open(CONCEPT_VEC, "rb") as f:
        vec = pickle.load(f)
    mat = sp.load_npz(CONCEPT_MAT)
    ids = np.load(CONCEPT_IDS, allow_pickle=True).tolist()
    return vec, mat, ids


@lru_cache(maxsize=1)
def _bm25_index():
    with open(BM25_IDX, "rb") as f:
        idx = pickle.load(f)
    ids = np.load(BM25_IDS, allow_pickle=True).tolist()
    return idx, ids


@lru_cache(maxsize=1)
def _encoder():
    from nsf_embeddings import SPECTER2Encoder
    return SPECTER2Encoder("allenai/specter2")


# SQLite helpers

_STOP = frozenset(
    "a an the and or but in on at to for of with by from is are was were "
    "be been being have has had do does did will would could should may "
    "might shall can this that these those it its we our they their he she "
    "his her i my you your us them him her which who what when where how "
    "not no nor so yet both either each few more most other some such "
    "than then there thus".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{1,}", text.lower())
            if t not in _STOP and len(t) > 2]


def _normalize_arr(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _fmt_award(row: dict, include_abstract: bool = False) -> str:
    lines = [
        f"  award_id    : {row.get('award_id', '')}",
        f"  title       : {row.get('title', '')}",
        f"  year        : {row.get('source_year', '')}",
        f"  directorate : {row.get('dir', '')} — {row.get('dir_name', '')}",
        f"  division    : {row.get('div', '')} — {row.get('div_name', '')}",
        f"  institution : {row.get('inst_name', '')}  ({row.get('state_code', '')})",
        f"  amount      : ${row.get('award_amount') or 0:,.0f}",
    ]
    if include_abstract and row.get("abstract"):
        wrapped = textwrap.fill(row["abstract"][:600], width=72,
                                subsequent_indent="               ")
        lines.append(f"  abstract    : {wrapped}")
    return "\n".join(lines)


def _fetch_award_rows(award_ids: list[str]) -> dict[str, dict]:
    if not award_ids:
        return {}
    ph   = ",".join("?" * len(award_ids))
    rows = _db().execute(
        f"""SELECT a.award_id, a.title, a.award_amount, a.source_year,
                   COALESCE(a.abstract_narration, a.por_text, a.title) AS abstract,
                   d.abbreviation AS dir,   d.long_name  AS dir_name,
                   v.abbreviation AS div,   v.long_name  AS div_name,
                   i.name         AS inst_name, i.state_code
            FROM award a
            LEFT JOIN directorate d ON d.id = a.directorate_id
            LEFT JOIN division    v ON v.id = a.division_id
            LEFT JOIN institution i ON i.id = a.institution_id
            WHERE a.award_id IN ({ph})""",
        award_ids,
    ).fetchall()
    return {r["award_id"]: dict(r) for r in rows}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_proposals(query: str, fields: str = "id,title,summary", rows: int = 5) -> str:
    try:
        results = _solr().search(query, **{"fl": fields, "rows": min(int(rows), 50)})
        if not results.hits:
            return f"No results for: {query}"
        lines = [f"Found {results.hits:,} proposals (showing {len(list(results))}):\n"]
        for doc in results:
            lines.append(f"ID: {doc.get('id','?')}  |  {doc.get('title','(no title)')}")
            for f in fields.split(","):
                f = f.strip()
                if f not in ("id", "title") and f in doc:
                    val = str(doc[f])
                    lines.append(f"  {f}: {val[:400] + '…' if len(val) > 400 else val}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


def get_proposal(proposal_id: str,
                 fields: str = "id,title,summary,description,pi_name,inst,award_amount,directorate,received_year,status") -> str:
    try:
        results = _solr().search(f'id:{proposal_id}', **{"fl": fields, "rows": 1})
        docs = list(results)
        if not docs:
            return f"No proposal found: {proposal_id}"
        doc   = docs[0]
        lines = [f"Proposal {proposal_id}\n"]
        for f in fields.split(","):
            f = f.strip()
            if f in doc:
                lines.append(f"{'─'*40}\n{f.upper()}:\n{doc[f]}")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


def fetch_proposals_by_ids(id_list: str,
                           fields: str = "id,title,summary,pi_name,award_amount") -> str:
    ids = [i.strip() for i in id_list.split(",") if i.strip()]
    if not ids:
        return "No IDs provided."
    try:
        query   = "id:(" + " OR ".join(ids) + ")"
        results = _solr().search(query, **{"fl": fields, "rows": len(ids)})
        docs    = list(results)
        if not docs:
            return "None of the IDs were found."
        found   = {str(d.get("id")): d for d in docs}
        missing = [i for i in ids if i not in found]
        lines   = [f"Retrieved {len(docs)} of {len(ids)} proposals.\n"]
        for doc in docs:
            lines.append(f"ID {doc.get('id')}  |  {doc.get('title','?')}")
            for f in fields.split(","):
                f = f.strip()
                if f not in ("id", "title") and f in doc:
                    val = str(doc[f])
                    lines.append(f"  {f}: {val[:300] + '…' if len(val) > 300 else val}")
            lines.append("")
        if missing:
            lines.append(f"Not found: {', '.join(missing)}")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


def facet_proposals(query: str, facet_field: str, limit: int = 15) -> str:
    try:
        results = _solr().search(query, **{
            "rows": 0,
            "facet": "true",
            "facet.field": facet_field,
            "facet.limit": int(limit),
            "facet.mincount": 1,
        })
        raw   = results.facets.get("facet_fields", {}).get(facet_field, [])
        pairs = list(zip(raw[::2], raw[1::2]))
        if not pairs:
            return f"No facet data for '{facet_field}' (matched {results.hits:,} proposals)"
        max_c = pairs[0][1]
        lines = [f"Top {len(pairs)} values for '{facet_field}' across {results.hits:,} proposals:\n"]
        for val, count in pairs:
            bar = "█" * int(count / max(max_c, 1) * 25)
            lines.append(f"  {str(val):<40} {count:>7,}  {bar}")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


def proposal_fields(group: str = "") -> str:
    if group:
        if group not in FIELD_GROUPS:
            return f"Unknown group '{group}'. Valid: {', '.join(FIELD_GROUPS)}"
        return f"{group}: {FIELD_GROUPS[group]}"
    lines = ["Field groups (use field names in search/get_proposal):\n"]
    for code, flds in FIELD_GROUPS.items():
        lines.append(f"  {code:<8}  {flds}")
    return "\n".join(lines)


def csv_to_panel_input(csv_path: str, id_column: str = "",
                       text_field: str = "summary", out_path: str = "") -> str:
    """Read proposal IDs from a CSV file, fetch summaries from SOLR, and write
    a formatted text file ready for the nsf_panel_builder app.

    csv_path:   path to the CSV file containing proposal IDs
    id_column:  column name containing NSF proposal IDs (auto-detected if blank)
    text_field: SOLR field to use as the abstract — 'summary' (default) or 'description'
    out_path:   where to write the panel input file (default: <csv_path>.panel_input.txt)

    Output format (one block per proposal, separated by ---):
        Proposal Title
        <summary text>
        ---
    """
    # --- Read CSV ---
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return f"File not found: {csv_path}"
    try:
        with open(csv_file, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
            cols   = reader.fieldnames or []
    except Exception as e:
        return f"CSV read error: {e}"

    if not rows:
        return "CSV file is empty."

    # --- Auto-detect ID column ---
    if not id_column:
        id_candidates = [c for c in cols if re.search(r"id|prop|award|number|num", c, re.I)]
        if not id_candidates:
            return (f"Cannot auto-detect ID column. Columns found: {', '.join(cols)}\n"
                    f"Re-run with id_column='<column name>'")
        id_column = id_candidates[0]

    if id_column not in cols:
        return f"Column '{id_column}' not found. Available: {', '.join(cols)}"

    ids = [str(r[id_column]).strip() for r in rows if r.get(id_column, "").strip()]
    if not ids:
        return f"No IDs found in column '{id_column}'."

    # --- Fetch from SOLR in batches of 50 ---
    fetched: dict[str, dict] = {}
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        query = "id:(" + " OR ".join(batch) + ")"
        try:
            results = _solr().search(query, **{
                "fl": f"id,title,{text_field}",
                "rows": len(batch),
            })
            for doc in results:
                fetched[str(doc.get("id", ""))] = doc
        except Exception as e:
            return f"SOLR error (check VPN): {e}"

    # --- Build panel input text ---
    blocks  = []
    missing = []
    for pid in ids:
        doc = fetched.get(pid)
        if not doc:
            missing.append(pid)
            continue
        title   = doc.get("title", f"Proposal {pid}").strip()
        abstract = str(doc.get(text_field, "")).strip()
        if not abstract:
            missing.append(pid)
            continue
        blocks.append(f"{title}\n{abstract}")

    if not blocks:
        return f"No text retrieved for any of {len(ids)} IDs (check VPN / field name)."

    output_text = "\n---\n".join(blocks) + "\n---\n"

    # --- Write output file ---
    if not out_path:
        out_path = str(csv_file.with_suffix("")) + ".panel_input.txt"
    try:
        Path(out_path).write_text(output_text, encoding="utf-8")
    except Exception as e:
        return f"Could not write output file: {e}"

    summary = (f"Wrote {len(blocks)} proposals to: {out_path}\n"
               f"  ID column : '{id_column}'\n"
               f"  Text field: '{text_field}'\n"
               f"  Missing   : {len(missing)}"
               + (f" ({', '.join(missing[:10])}{'…' if len(missing)>10 else ''})" if missing else ""))
    return summary


# ---------------------------------------------------------------------------
# SQLite tools (mirror of nsf_local_mcp.py)
# ---------------------------------------------------------------------------

def db_schema() -> str:
    """Show all SQLite tables with columns and row counts."""
    try:
        conn   = _db()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        lines  = ["NSF local database schema (awarded proposals, 2010–2024)\n"]
        for (tname,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM [{tname}]").fetchone()[0]
            lines.append(f"── {tname}  ({count:,} rows)")
            for col in conn.execute(f"PRAGMA table_info([{tname}])").fetchall():
                pk = " PK" if col["pk"] else ""
                lines.append(f"     {col['name']:<35} {col['type']:<12}{pk}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"DB error: {e}"


def sql_query(query: str, limit: int = 100) -> str:
    """Run a read-only SQL SELECT against the local NSF awards database.
    Tables: award, directorate, division, institution, award_investigator,
            program_element, award_program_element, researcher_fingerprint, researcher_papers.
    Key columns: award.award_id, award.title, award.abstract_narration,
                 award.award_amount, award.source_year, award.directorate_id.
    award_investigator.role_code is 'Principal Investigator' or 'Co-Principal Investigator'.
    Only SELECT is allowed.
    """
    query = query.strip()
    if not re.match(r"^\s*SELECT\b", query, re.IGNORECASE):
        return "Only SELECT queries are allowed."
    if not re.search(r"\bLIMIT\b", query, re.IGNORECASE):
        query = f"{query} LIMIT {limit}"
    try:
        rows = _db().execute(query).fetchall()
    except Exception as e:
        return f"SQL error: {e}"
    if not rows:
        return "Query returned no rows."
    keys   = rows[0].keys()
    header = "  ".join(f"{k:<20}" for k in keys)
    sep    = "  ".join("-" * 20 for _ in keys)
    lines  = [f"{len(rows)} row(s):\n", header, sep]
    for row in rows:
        lines.append("  ".join(f"{str(v or ''):<20}" for v in row))
    return "\n".join(lines)


def semantic_search(query_text: str, top_n: int = 10,
                    directorate: str = "", year_start: int = 0,
                    year_end: int = 0) -> str:
    """Find NSF awarded grants semantically similar to a natural-language query.
    Uses SPECTER2 embeddings — best for concept/topic discovery.
    directorate: optional filter e.g. 'BIO', 'CSE', 'ENG', 'MPS'
    year_start / year_end: optional year range filter
    """
    try:
        top_n    = min(int(top_n), 50)
        emb, ids = _embeddings()
        enc      = _encoder()
        q_vec    = enc.encode([query_text], normalize_embeddings=True,
                              convert_to_numpy=True)[0].astype(np.float32)
        sims     = emb @ q_vec

        if directorate or year_start or year_end:
            clauses, params = [], []
            if directorate:
                clauses.append("d.abbreviation = ?"); params.append(directorate)
            if year_start:
                clauses.append("a.source_year >= ?"); params.append(int(year_start))
            if year_end:
                clauses.append("a.source_year <= ?"); params.append(int(year_end))
            where   = " AND ".join(clauses)
            allowed = {r[0] for r in _db().execute(
                f"SELECT a.award_id FROM award a "
                f"LEFT JOIN directorate d ON d.id=a.directorate_id WHERE {where}", params
            ).fetchall()}
            for i, aid in enumerate(ids):
                if aid not in allowed:
                    sims[i] = -2.0

        top_idx = np.argsort(sims)[::-1][:top_n]
        top_ids = [ids[i] for i in top_idx]
        meta    = _fetch_award_rows(top_ids)
        lines   = [f"Top {len(top_idx)} semantically similar awards:\n"]
        for rank, i in enumerate(top_idx, 1):
            aid = ids[i]
            lines.append(f"[{rank}]  score={float(sims[i]):.3f}")
            lines.append(_fmt_award(meta.get(aid, {"award_id": aid})))
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Semantic search error: {e}"


def hybrid_search(query_text: str, top_n: int = 10,
                  directorate: str = "", year_start: int = 0,
                  year_end: int = 0) -> str:
    """Find NSF awarded grants via hybrid retrieval: SPECTER2 + TF-IDF concepts + BM25.
    More precise than semantic_search alone for specific research area queries.
    directorate: optional filter e.g. 'BIO', 'CSE'
    year_start / year_end: optional year range
    """
    try:
        import scipy.sparse as sp
        top_n = min(int(top_n), 50)

        emb, emb_ids    = _embeddings()
        vec, mat, c_ids = _concept_index()
        bm25_idx, b_ids = _bm25_index()
        enc             = _encoder()

        q_vec       = enc.encode([query_text], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)
        raw_emb     = emb @ q_vec
        q_cvec      = vec.transform([query_text])
        raw_concept = (mat @ q_cvec.T).toarray().ravel().astype(np.float32)
        tokens      = _tokenize(query_text)
        raw_bm25    = (bm25_idx.get_scores(tokens).astype(np.float32)
                       if tokens else np.zeros(len(b_ids), dtype=np.float32))

        combined = (0.45 * _normalize_arr(raw_emb) +
                    0.35 * _normalize_arr(raw_concept) +
                    0.20 * _normalize_arr(raw_bm25))

        if directorate or year_start or year_end:
            clauses, params = [], []
            if directorate:
                clauses.append("d.abbreviation = ?"); params.append(directorate)
            if year_start:
                clauses.append("a.source_year >= ?"); params.append(int(year_start))
            if year_end:
                clauses.append("a.source_year <= ?"); params.append(int(year_end))
            where   = " AND ".join(clauses)
            allowed = {r[0] for r in _db().execute(
                f"SELECT a.award_id FROM award a "
                f"LEFT JOIN directorate d ON d.id=a.directorate_id WHERE {where}", params
            ).fetchall()}
            for i, aid in enumerate(emb_ids):
                if aid not in allowed:
                    combined[i] = -1.0

        top_idx = np.argsort(combined)[::-1][:top_n]
        top_ids = [emb_ids[i] for i in top_idx]
        meta    = _fetch_award_rows(top_ids)
        lines   = [f"Top {len(top_idx)} awards (hybrid semantic+concept+BM25):\n"]
        for rank, i in enumerate(top_idx, 1):
            aid = emb_ids[i]
            lines.append(f"[{rank}]  score={combined[i]:.3f}")
            lines.append(_fmt_award(meta.get(aid, {"award_id": aid})))
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Hybrid search error: {e}"


def get_award(award_id: str) -> str:
    """Get full details for a specific NSF award: abstract, investigators, program elements.
    award_id: NSF award number e.g. '2535312'
    """
    try:
        conn = _db()
        meta = _fetch_award_rows([award_id])
        if not meta:
            return f"No award found: {award_id}"
        lines = [f"Award {award_id}\n", _fmt_award(meta[award_id], include_abstract=True), ""]

        invs = conn.execute(
            """SELECT ai.full_name, ai.role_code, ai.email, i.name AS inst
               FROM award_investigator ai
               JOIN award a ON a.id = ai.award_id
               LEFT JOIN institution i ON i.id = a.institution_id
               WHERE a.award_id = ?
               ORDER BY CASE ai.role_code WHEN 'Principal Investigator' THEN 0 ELSE 1 END""",
            (award_id,),
        ).fetchall()
        if invs:
            lines.append("  investigators:")
            for inv in invs:
                role = "PI" if "Principal" in (inv["role_code"] or "") and "Co" not in (inv["role_code"] or "") else "Co-PI"
                lines.append(f"    [{role}] {inv['full_name']}  {inv['email'] or ''}  {inv['inst'] or ''}")
            lines.append("")

        pes = conn.execute(
            """SELECT pe.code, pe.text FROM award_program_element ape
               JOIN award a ON a.id = ape.award_id
               JOIN program_element pe ON pe.id = ape.program_element_id
               WHERE a.award_id = ?""",
            (award_id,),
        ).fetchall()
        if pes:
            lines.append("  program elements:")
            for pe in pes:
                lines.append(f"    {pe['code']}  {pe['text'] or ''}")

        return "\n".join(lines)
    except Exception as e:
        return f"DB error: {e}"


def get_researcher(full_name: str = "", investigator_key: str = "") -> str:
    """Get a PI's profile: NSF funding history, publications, top research topics.
    full_name: partial name match e.g. 'Jane Smith'
    investigator_key: exact key e.g. 'e:jane.smith@nsf.gov'
    Falls back to award_investigator table if fingerprint not built yet.
    """
    try:
        conn = _db()
        if investigator_key:
            rows = conn.execute(
                "SELECT * FROM researcher_fingerprint WHERE investigator_key = ?",
                (investigator_key,),
            ).fetchall()
        elif full_name:
            rows = conn.execute(
                "SELECT * FROM researcher_fingerprint WHERE LOWER(full_name) LIKE ?",
                (f"%{full_name.lower()}%",),
            ).fetchall()
        else:
            return "Provide full_name or investigator_key."

        if not rows:
            # Fall back to award_investigator
            awards = conn.execute(
                """SELECT a.award_id, a.title, a.source_year, a.award_amount,
                          d.abbreviation AS dir, ai.role_code
                   FROM award_investigator ai
                   JOIN award a ON a.id = ai.award_id
                   LEFT JOIN directorate d ON d.id = a.directorate_id
                   WHERE LOWER(ai.full_name) LIKE ?
                   ORDER BY a.source_year DESC LIMIT 20""",
                (f"%{full_name.lower()}%",),
            ).fetchall()
            if not awards:
                return f"No records found for '{full_name}'. Try sql_query on award_investigator."
            lines = [f"Awards for '{full_name}' (no fingerprint built — showing raw awards):\n"]
            for aw in awards:
                amt = f"${aw['award_amount']:,.0f}" if aw["award_amount"] else "—"
                lines.append(f"  {aw['award_id']}  ({aw['source_year']})  [{aw['dir'] or '?'}]  {amt}  {aw['title'][:60]}")
            return "\n".join(lines)

        lines = []
        for fp in rows:
            key = fp["investigator_key"]
            lines += [
                f"Researcher: {fp['full_name']}",
                f"  institution : {fp['inst_name'] or '—'}",
                f"  NSF awards  : {fp['n_nsf_awards']}",
                f"  papers      : {fp['n_papers']}  ({fp['n_with_abstract']} with abstract)",
                f"  top topics  : {fp['top_topics'] or '—'}",
                "",
            ]
            papers = conn.execute(
                """SELECT paper_title, pub_year, venue FROM researcher_papers
                   WHERE investigator_key = ? ORDER BY pub_year DESC LIMIT 10""",
                (key,),
            ).fetchall()
            if papers:
                lines.append("  recent papers:")
                for p in papers:
                    lines.append(f"    ({p['pub_year']}) {p['paper_title'][:70]}")
                lines.append("")
            awards = conn.execute(
                """SELECT a.award_id, a.title, a.source_year, a.award_amount, d.abbreviation AS dir
                   FROM award_investigator ai
                   JOIN award a ON a.id = ai.award_id
                   LEFT JOIN directorate d ON d.id = a.directorate_id
                   WHERE ai.full_name = ? AND ai.role_code = 'Principal Investigator'
                   ORDER BY a.source_year DESC LIMIT 10""",
                (fp["full_name"],),
            ).fetchall()
            if awards:
                lines.append("  NSF awards (PI):")
                for aw in awards:
                    amt = f"${aw['award_amount']:,.0f}" if aw["award_amount"] else "—"
                    lines.append(f"    {aw['award_id']}  ({aw['source_year']})  [{aw['dir'] or '?'}]  {amt}  {aw['title'][:60]}")
        return "\n".join(lines)
    except Exception as e:
        return f"DB error: {e}"


TOOL_MAP = {
    # SOLR tools
    "search_proposals":       search_proposals,
    "get_proposal":           get_proposal,
    "fetch_proposals_by_ids": fetch_proposals_by_ids,
    "facet_proposals":        facet_proposals,
    "proposal_fields":        proposal_fields,
    "csv_to_panel_input":     csv_to_panel_input,
    # SQLite tools
    "db_schema":              db_schema,
    "sql_query":              sql_query,
    "semantic_search":        semantic_search,
    "hybrid_search":          hybrid_search,
    "get_award":              get_award,
    "get_researcher":         get_researcher,
}

# ---------------------------------------------------------------------------
# Ollama native tool definitions
# ---------------------------------------------------------------------------

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_proposals",
            "description": "Search NSF proposals using Lucene query syntax.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":  {"type": "string",  "description": "Lucene query, e.g. 'pi_name:\"Jane Smith\"' or 'summary:(quantum computing)'"},
                    "fields": {"type": "string",  "description": "comma-separated fields: id,title,summary,description,pi_name,inst,award_amount,directorate,received_year,status"},
                    "rows":   {"type": "integer", "description": "number of results (max 50)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_proposal",
            "description": "Retrieve a single NSF proposal by ID with full text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "string", "description": "NSF proposal ID, e.g. '2535312'"},
                    "fields":      {"type": "string", "description": "comma-separated fields to return"},
                },
                "required": ["proposal_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_proposals_by_ids",
            "description": "Fetch multiple proposals by comma-separated IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id_list": {"type": "string", "description": "comma-separated IDs e.g. '2535312,2535313'"},
                    "fields":  {"type": "string", "description": "fields to return for each proposal"},
                },
                "required": ["id_list"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "facet_proposals",
            "description": "Count top values for a field across matching proposals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "Lucene query selecting proposals to count"},
                    "facet_field": {"type": "string",  "description": "field to break down: directorate, division, inst, inst_state, status, received_year, funding_program"},
                    "limit":       {"type": "integer", "description": "number of top values to show"},
                },
                "required": ["query", "facet_field"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "proposal_fields",
            "description": "List available NSF proposal field groups.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group": {"type": "string", "description": "optional group code e.g. fi_pin, fi_awd"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "csv_to_panel_input",
            "description": (
                "Read proposal IDs from a CSV file, fetch their summaries from SOLR, "
                "and write a formatted text file ready for the nsf_panel_builder app. "
                "Auto-detects the ID column. Output is one block per proposal separated by ---."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path":   {"type": "string", "description": "path to the CSV file"},
                    "id_column":  {"type": "string", "description": "column name with proposal IDs (auto-detected if blank)"},
                    "text_field": {"type": "string", "description": "'summary' (default) or 'description' for full narrative"},
                    "out_path":   {"type": "string", "description": "output file path (default: <csv>.panel_input.txt)"},
                },
                "required": ["csv_path"],
            },
        },
    },
    # --- SQLite tools ---
    {
        "type": "function",
        "function": {
            "name": "db_schema",
            "description": "Show all SQLite tables with columns and row counts. Call first to understand the local database before writing SQL.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Run a read-only SQL SELECT on the local NSF awards database (162k awarded grants, 2010–2024). Good for aggregations, trends, PI counts, funding totals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SELECT statement"},
                    "limit": {"type": "integer", "description": "max rows returned (default 100)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Find awarded NSF grants semantically similar to a natural-language query using SPECTER2 embeddings. Best for concept/topic discovery without exact keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text":  {"type": "string",  "description": "natural language description or abstract"},
                    "top_n":       {"type": "integer", "description": "number of results (max 50)"},
                    "directorate": {"type": "string",  "description": "optional filter: BIO, CSE, ENG, MPS, GEO, SBE, EDU, TIP"},
                    "year_start":  {"type": "integer", "description": "optional start year"},
                    "year_end":    {"type": "integer", "description": "optional end year"},
                },
                "required": ["query_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "Find awarded NSF grants via hybrid retrieval: SPECTER2 embeddings + TF-IDF concepts + BM25. More precise than semantic_search for specific research areas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text":  {"type": "string",  "description": "research topic or description"},
                    "top_n":       {"type": "integer", "description": "number of results (max 50)"},
                    "directorate": {"type": "string",  "description": "optional filter: BIO, CSE, ENG etc."},
                    "year_start":  {"type": "integer", "description": "optional start year"},
                    "year_end":    {"type": "integer", "description": "optional end year"},
                },
                "required": ["query_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_award",
            "description": "Get full details for a specific NSF awarded grant: abstract, investigators, program element codes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "award_id": {"type": "string", "description": "NSF award number e.g. '2535312'"},
                },
                "required": ["award_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_researcher",
            "description": "Get a PI's profile: NSF funding history, publications, top research topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name":        {"type": "string", "description": "partial name match e.g. 'Jane Smith'"},
                    "investigator_key": {"type": "string", "description": "exact key e.g. 'e:jane.smith@nsf.gov'"},
                },
            },
        },
    },
]

SYSTEM_MSG = {
    "role": "system",
    "content": """You are an expert assistant for NSF research data with two data sources:

1. SOLR (VPN required) — all proposals including declined/pending, full text, panel/reviewer data
2. SQLite (local, no VPN) — 162k awarded grants 2010-2024, semantic search, PI profiles

Use SOLR tools for: proposal full text, declined proposals, panel analysis, reviewer info.
Use SQLite tools for: semantic/concept search, SQL aggregations, PI career profiles, funding trends.
Two-step pattern: semantic_search or hybrid_search to find award IDs → get_proposal for full SOLR text.

═══ SOLR FIELDS (exact spelling — do not alter field names) ═══
  id, title, summary, description, pi_name, pi_all, inst, inst_state
  award_amount, directorate, division, received_year
  panel_id, panel_name, panel_reviewers
  pi_gender, pi_race, pi_ethnicity, reviewer_name, reviewer_gender

FIELD NAME SPELLINGS — use exactly as written, no variations:
  directorate   (NOT directorato, NOT Directorate, NOT dir)
  received_year (NOT year, NOT receivedyear)
  panel_id      (NOT panelid, NOT panel)
  award_amount  (NOT amount, NOT awardAmount)

STATUS (exact strings, always quoted):
  status:"Proposal has been awarded"
  status:"Pending, PM recommends award"
  status:"Recommended for award, DDConcurred"
  status:"Decline, DDConcurred"
  status:"Pending, PM recommends decline"
  status:"Pending, Review Package Produced"
  NEVER use status:Awarded — returns nothing.

═══ SOLR QUERY EXAMPLES ═══
  panel_id:P260135 AND status:"Proposal has been awarded"
  directorate:BIO AND received_year:2024
  pi_name:"Jane Smith"
  summary:(quantum computing)

═══ SQLITE SQL EXAMPLES ═══
  -- funding by directorate 2023
  SELECT d.abbreviation, COUNT(*) n, SUM(a.award_amount)/1e6 total_m
  FROM award a JOIN directorate d ON d.id=a.directorate_id
  WHERE a.source_year=2023 GROUP BY d.abbreviation ORDER BY total_m DESC

  -- top PIs by award count
  SELECT ai.full_name, COUNT(*) n_awards, SUM(a.award_amount) total
  FROM award_investigator ai JOIN award a ON a.id=ai.award_id
  WHERE ai.role_code='Principal Investigator'
  GROUP BY ai.full_name ORDER BY n_awards DESC LIMIT 20
""",
}


# ---------------------------------------------------------------------------
# Native tool-calling loop
# ---------------------------------------------------------------------------

def _run_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Unknown tool '{name}'"
    try:
        return fn(**args)
    except TypeError as e:
        return f"Bad arguments for {name}: {e}"


def run_native(model: str, verbose: bool, outfile=None) -> None:
    history: list[dict] = []

    def emit(text: str) -> None:
        print(text)
        if outfile:
            outfile.write(text + "\n")
            outfile.flush()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "tools":
            for t in OLLAMA_TOOLS:
                print(f"  {t['function']['name']}: {t['function']['description']}")
            continue

        if outfile:
            outfile.write(f"\n## You: {user_input}\n\n")
            outfile.flush()

        history.append({"role": "user", "content": user_input})
        messages = [SYSTEM_MSG] + history[-30:]

        for _ in range(16):
            resp    = ollama.chat(model=model, messages=messages, tools=OLLAMA_TOOLS)
            msg     = resp["message"]
            calls   = msg.get("tool_calls") or []

            if not calls:
                answer = msg.get("content", "").strip()
                emit(f"\nAssistant: {answer}\n")
                history.append({"role": "assistant", "content": answer})
                break

            messages.append(msg)
            for tc in calls:
                fn_info = tc.get("function", tc)
                name    = fn_info["name"]
                args    = fn_info.get("arguments", {})
                if verbose:
                    print(f"  [tool] {name}({json.dumps(args)[:120]})")
                result = _run_tool(name, args)
                if verbose:
                    print(f"  [result] {result[:200]}")
                messages.append({"role": "tool", "content": result})
        else:
            print("(reached max steps)\n")

        if len(history) > 40:
            history = history[-40:]


# ---------------------------------------------------------------------------
# ReAct fallback (for models without native tool support, e.g. phi4)
# ---------------------------------------------------------------------------

REACT_SYSTEM = f"""You are an expert assistant for NSF research proposals.
You have these tools:

search_proposals(query, fields="id,title,summary", rows=5)
get_proposal(proposal_id, fields="id,title,summary,description,pi_name,inst,award_amount,directorate,received_year,status")
fetch_proposals_by_ids(id_list, fields="id,title,summary,pi_name,award_amount")
facet_proposals(query, facet_field, limit=15)
proposal_fields(group="")

STATUS FIELD uses full strings — always quote them:
  Awarded:  status:"Proposal has been awarded"
  Declined: status:"Decline, DDConcurred"
  Pending:  status:"Pending, Review Package Produced"
  NEVER use status:Awarded — it returns nothing.

Panel queries use panel_id field: panel_id:P260135

To call a tool output EXACTLY (no extra text before Action:):
Action: <tool_name>
Action Input: {{"key": "value"}}

After the Observation write either another Action or:
Final Answer: <your answer>
"""


def run_react(model: str, verbose: bool) -> None:
    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "tools":
            for name in TOOL_MAP:
                print(f"  {name}")
            continue

        history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": REACT_SYSTEM}] + history[-20:]

        for step in range(8):
            resp    = ollama.chat(model=model, messages=messages)
            content = resp["message"]["content"]
            if verbose:
                print(f"\n  [step {step+1}] {content[:400]}")

            fa = re.search(r"Final Answer:\s*(.*)", content, re.DOTALL)
            am = re.search(r"Action:\s*(\w+)", content)
            im = re.search(r"Action Input:\s*(\{.*?\})", content, re.DOTALL)

            if fa or not am:
                answer = fa.group(1).strip() if fa else content.strip()
                print(f"\nAssistant: {answer}\n")
                history.append({"role": "assistant", "content": answer})
                break

            name = am.group(1).strip()
            args = {}
            if im:
                try:
                    args = json.loads(im.group(1))
                except json.JSONDecodeError:
                    pass

            if verbose:
                print(f"  [tool] {name}({json.dumps(args)[:120]})")
            result = _run_tool(name, args)
            if verbose:
                print(f"  [obs]  {result[:200]}")

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation: {result}\n\nContinue."})
        else:
            print("(reached max steps)\n")

        if len(history) > 40:
            history = history[-40:]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_repl(model: str = "llama3.1", verbose: bool = False,
             react: bool = False, out: str | None = None) -> None:
    if not verbose:
        sys.stderr = open(os.devnull, "w")

    mode = "react" if react else "native"
    print(f"NSF SOLR Agent  (model={model}  mode={mode}  solr={SOLR_URL})")
    if out:
        print(f"Saving to: {out}")
    print("Type 'quit' to exit, 'tools' to list tools.\n")

    outfile = open(out, "w") if out else None
    if outfile:
        outfile.write(f"# NSF SOLR Session  model={model}\n\n")

    try:
        if react:
            run_react(model, verbose)
        else:
            run_native(model, verbose, outfile=outfile)
    finally:
        if outfile:
            outfile.close()
            print(f"\nSession saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NSF SOLR + SQLite agent (Ollama)")
    p.add_argument("--model",   default="llama3.1",
                   help="Ollama model (default: llama3.1).")
    p.add_argument("--verbose", action="store_true",
                   help="Show tool calls and results")
    p.add_argument("--react",   action="store_true",
                   help="Use ReAct text loop (for phi4 and models without native tool support)")
    p.add_argument("--out",     default=None, metavar="FILE",
                   help="Save session to a markdown file e.g. --out panel_report.md")
    p.add_argument("--output",  default=None, metavar="DIR",
                   help="Path to output/ directory containing nsf_awards.db and indices "
                        "(default: ./output). Set this on the NSF machine.")
    args = p.parse_args()

    # Override index paths if --output is given
    if args.output:
        import sys as _sys
        _out = Path(args.output)
        # Patch module-level globals
        _mod = _sys.modules[__name__]
        for _name, _fname in [
            ("DB_PATH",     "nsf_awards.db"),
            ("EMB_PATH",    "embeddings_specter2.npy"),
            ("IDS_PATH",    "award_ids_specter2.npy"),
            ("CONCEPT_VEC", "concept_vectorizer.pkl"),
            ("CONCEPT_MAT", "concept_matrix.npz"),
            ("CONCEPT_IDS", "concept_award_ids.npy"),
            ("BM25_IDX",    "bm25_index.pkl"),
            ("BM25_IDS",    "bm25_award_ids.npy"),
        ]:
            setattr(_mod, _name, _out / _fname)

    run_repl(model=args.model, verbose=args.verbose, react=args.react, out=args.out)
