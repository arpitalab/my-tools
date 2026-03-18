"""
nsf_local_mcp.py — MCP server over the local NSF assets.

Gives Claude Code (or any MCP client) direct access to:
  • SQLite award database  (162k+ NSF awards, 2010–2024)
  • SPECTER2 embeddings    (semantic similarity search)
  • TF-IDF concept index   (domain-discriminating n-gram search)
  • BM25 keyword index     (exact-term matching)
  • Researcher fingerprints (OpenAlex-augmented PI profiles)

Tools exposed:
  schema          — list DB tables, columns, row counts
  sql_query       — run any read-only SQL (most flexible tool)
  semantic_search — find awards similar to a text query
  hybrid_search   — semantic + TF-IDF + BM25 combined
  get_award       — full record for one award (investigators, PE codes, abstract)
  get_researcher  — PI profile: awards, papers, top topics

Usage — add to Claude Code:
  claude mcp add nsf-local \\
    conda run --no-capture-output -n thellmbook \\
    python /Users/sraghava/Desktop/my_llm_explore/nsf_local_mcp.py

Or add to ~/Library/Application Support/Claude/claude_desktop_config.json:
  {
    "mcpServers": {
      "nsf-local": {
        "command": "conda",
        "args": ["run", "--no-capture-output", "-n", "thellmbook",
                 "python", "/Users/sraghava/Desktop/my_llm_explore/nsf_local_mcp.py"]
      }
    }
  }

Then in Claude Code:
  "Generate a funding trend report for quantum computing 2010–2024"
  "Who are the top NSF-funded PIs in climate modeling?"
  "Find awards similar to this abstract: ..."
  "Compare BIO vs CSE directorate funding 2015–2024"
"""
from __future__ import annotations

import os
import pickle
import re
import sqlite3
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE = Path(__file__).parent / "output"

DB_PATH      = _BASE / "nsf_awards.db"
EMB_PATH     = _BASE / "embeddings_specter2.npy"
IDS_PATH     = _BASE / "award_ids_specter2.npy"
CONCEPT_VEC  = _BASE / "concept_vectorizer.pkl"
CONCEPT_MAT  = _BASE / "concept_matrix.npz"
CONCEPT_IDS  = _BASE / "concept_award_ids.npy"
BM25_IDX     = _BASE / "bm25_index.pkl"
BM25_IDS     = _BASE / "bm25_award_ids.npy"

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "nsf-local",
    instructions=(
        "Access the local NSF award database (162k awards, 2010–2024) plus "
        "semantic and keyword search indices. "
        "Start with schema() to understand the data, then use sql_query() for "
        "structured analysis or semantic_search()/hybrid_search() for topic-based "
        "discovery. Use get_award() for full details on specific awards and "
        "get_researcher() for PI profiles."
    ),
)

# ---------------------------------------------------------------------------
# Lazy-loaded resources (loaded once on first use, kept in memory)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@lru_cache(maxsize=1)
def _embeddings() -> tuple[np.ndarray, list[str]]:
    emb = np.load(EMB_PATH)                               # (N, D) float32
    ids = np.load(IDS_PATH, allow_pickle=True).tolist()
    return emb, ids


@lru_cache(maxsize=1)
def _concept_index():
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
    """Load SPECTER2 encoder (deferred — only when semantic search is called)."""
    from nsf_embeddings import SPECTER2Encoder
    return SPECTER2Encoder("allenai/specter2")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _normalize(arr: np.ndarray) -> np.ndarray:
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
        f"""
        SELECT a.award_id, a.title, a.award_amount, a.source_year,
               COALESCE(a.abstract_narration, a.por_text, a.title) AS abstract,
               d.abbreviation AS dir,   d.long_name  AS dir_name,
               v.abbreviation AS div,   v.long_name  AS div_name,
               i.name         AS inst_name, i.state_code
        FROM award a
        LEFT JOIN directorate d ON d.id = a.directorate_id
        LEFT JOIN division    v ON v.id = a.division_id
        LEFT JOIN institution i ON i.id = a.institution_id
        WHERE a.award_id IN ({ph})
        """,
        award_ids,
    ).fetchall()
    return {r["award_id"]: dict(r) for r in rows}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def schema() -> str:
    """Show all database tables with columns and row counts.

    Call this first to understand what data is available before writing
    SQL queries. Key tables: award, directorate, division, institution,
    award_investigator, program_element, researcher_fingerprint, researcher_papers.
    """
    conn  = _db()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()

    lines = ["NSF local database schema\n"]
    for (tname,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM [{tname}]").fetchone()[0]
        lines.append(f"── {tname}  ({count:,} rows)")
        cols = conn.execute(f"PRAGMA table_info([{tname}])").fetchall()
        for col in cols:
            pk  = " PK"  if col["pk"]        else ""
            nn  = " NOT NULL" if col["notnull"] else ""
            lines.append(f"     {col['name']:<35} {col['type']:<12}{pk}{nn}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def sql_query(query: str, limit: int = 100) -> str:
    """Run a read-only SQL query against the NSF awards database.

    Only SELECT statements are allowed. Results are capped at `limit` rows.

    Key tables and commonly useful columns:

      award            — award_id, title, abstract_narration, award_amount,
                         source_year, tran_type, directorate_id, division_id,
                         institution_id, po_name

      directorate      — id, abbreviation (e.g. 'BIO','CSE'), long_name

      division         — id, abbreviation, long_name

      institution      — id, name, state_code, country_name, org_uei_num

      award_investigator — award_id (FK→award.id), full_name, email,
                           role_code ('Principal Investigator' | 'Co-Principal Investigator'),
                           nsf_id

      program_element  — id, code (6-digit), text
      award_program_element — award_id (FK→award.id), program_element_id

      researcher_fingerprint — investigator_key, full_name, inst_name,
                               n_nsf_awards, n_papers, n_with_abstract,
                               top_topics (JSON), model_name, built_at

      researcher_papers — investigator_key, full_name, paper_title, paper_text,
                          has_abstract, pub_year, topics (JSON), venue

    Example queries:
      SELECT d.abbreviation, COUNT(*) AS n, SUM(a.award_amount)/1e6 AS total_m
      FROM award a JOIN directorate d ON d.id=a.directorate_id
      WHERE a.source_year=2023 GROUP BY d.abbreviation ORDER BY total_m DESC

      SELECT ai.full_name, COUNT(*) AS n_awards
      FROM award_investigator ai WHERE ai.role_code='Principal Investigator'
      GROUP BY ai.full_name ORDER BY n_awards DESC LIMIT 20
    """
    query = query.strip()
    if not re.match(r"^\s*SELECT\b", query, re.IGNORECASE):
        return "Only SELECT queries are allowed."

    # Inject LIMIT if missing
    if not re.search(r"\bLIMIT\b", query, re.IGNORECASE):
        query = f"{query} LIMIT {limit}"

    try:
        rows = _db().execute(query).fetchall()
    except sqlite3.Error as e:
        return f"SQL error: {e}"

    if not rows:
        return "Query returned no rows."

    # Header + rows
    keys   = rows[0].keys()
    header = "  ".join(f"{k:<20}" for k in keys)
    sep    = "  ".join("-" * 20 for _ in keys)
    lines  = [header, sep]
    for row in rows:
        lines.append("  ".join(f"{str(v or ''):<20}" for v in row))

    return f"{len(rows)} row(s):\n\n" + "\n".join(lines)


@mcp.tool()
def semantic_search(
    query_text: str,
    top_n: int = 10,
    directorate: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> str:
    """Find NSF awards semantically similar to a query using SPECTER2 embeddings.

    Best for: open-ended topic discovery, finding awards in a research area
    described in natural language, finding similar awards to a known abstract.

    query_text:  natural language description or a full abstract
    top_n:       number of results (max 50)
    directorate: filter to one directorate abbreviation, e.g. 'BIO', 'CSE', 'ENG'
    year_start / year_end: filter to a year range
    """
    top_n = min(top_n, 50)
    emb, ids = _embeddings()
    enc      = _encoder()

    q_vec = enc.encode([query_text], normalize_embeddings=True,
                       convert_to_numpy=True)[0].astype(np.float32)
    sims  = emb @ q_vec                                    # (N,)

    # Optional directorate / year filter
    if directorate or year_start or year_end:
        clauses, params = [], []
        if directorate:
            clauses.append("d.abbreviation = ?"); params.append(directorate)
        if year_start:
            clauses.append("a.source_year >= ?"); params.append(year_start)
        if year_end:
            clauses.append("a.source_year <= ?"); params.append(year_end)
        where = " AND ".join(clauses)
        allowed = {r[0] for r in _db().execute(
            f"SELECT a.award_id FROM award a "
            f"LEFT JOIN directorate d ON d.id=a.directorate_id "
            f"WHERE {where}", params
        ).fetchall()}
        for i, aid in enumerate(ids):
            if aid not in allowed:
                sims[i] = -2.0

    top_idx  = np.argsort(sims)[::-1][:top_n]
    top_ids  = [ids[i] for i in top_idx]
    meta     = _fetch_award_rows(top_ids)

    lines = [f"Top {len(top_idx)} semantically similar awards:\n"]
    for rank, i in enumerate(top_idx, 1):
        aid  = ids[i]
        m    = meta.get(aid, {})
        sim  = float(sims[i])
        lines.append(f"[{rank}]  score={sim:.3f}")
        lines.append(_fmt_award(m))
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def hybrid_search(
    query_text: str,
    top_n: int = 10,
    w_emb: float = 0.45,
    w_concept: float = 0.35,
    w_bm25: float = 0.20,
    directorate: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> str:
    """Find NSF awards using hybrid retrieval: semantic + TF-IDF concepts + BM25.

    Combines three signals to reduce cross-domain false matches:
      w_emb     — semantic/paraphrase similarity  (default 0.45)
      w_concept — domain-specific n-gram overlap  (default 0.35)
      w_bm25    — exact keyword matching          (default 0.20)

    Weights are renormalized to sum to 1. Increase w_bm25 for exact-term
    queries; increase w_emb for broad semantic queries.

    Best for: specific research area queries, proposal matching, finding
    reviewers whose awards overlap a topic.
    """
    top_n = min(top_n, 50)
    total_w = w_emb + w_concept + w_bm25 or 1.0
    w_emb /= total_w; w_concept /= total_w; w_bm25 /= total_w

    emb, emb_ids       = _embeddings()
    vec, mat, c_ids    = _concept_index()
    bm25_idx, b_ids    = _bm25_index()
    enc                = _encoder()

    # Signal 1: embedding
    q_vec   = enc.encode([query_text], normalize_embeddings=True,
                         convert_to_numpy=True)[0].astype(np.float32)
    raw_emb = emb @ q_vec

    # Signal 2: TF-IDF concept (same length as emb_ids — ORDER BY a.id)
    q_cvec      = vec.transform([query_text])
    raw_concept = (mat @ q_cvec.T).toarray().ravel().astype(np.float32)

    # Signal 3: BM25
    tokens   = _tokenize(query_text)
    raw_bm25 = (bm25_idx.get_scores(tokens).astype(np.float32)
                if tokens else np.zeros(len(b_ids), dtype=np.float32))

    combined = (w_emb   * _normalize(raw_emb) +
                w_concept * _normalize(raw_concept) +
                w_bm25  * _normalize(raw_bm25))

    # Optional filters
    if directorate or year_start or year_end:
        clauses, params = [], []
        if directorate:
            clauses.append("d.abbreviation = ?"); params.append(directorate)
        if year_start:
            clauses.append("a.source_year >= ?"); params.append(year_start)
        if year_end:
            clauses.append("a.source_year <= ?"); params.append(year_end)
        where   = " AND ".join(clauses)
        allowed = {r[0] for r in _db().execute(
            f"SELECT a.award_id FROM award a "
            f"LEFT JOIN directorate d ON d.id=a.directorate_id "
            f"WHERE {where}", params
        ).fetchall()}
        for i, aid in enumerate(emb_ids):
            if aid not in allowed:
                combined[i] = -1.0

    top_idx = np.argsort(combined)[::-1][:top_n]
    top_ids = [emb_ids[i] for i in top_idx]
    meta    = _fetch_award_rows(top_ids)

    lines = [
        f"Top {len(top_idx)} awards (hybrid: emb={w_emb:.2f} "
        f"concept={w_concept:.2f} bm25={w_bm25:.2f}):\n"
    ]
    for rank, i in enumerate(top_idx, 1):
        aid = emb_ids[i]
        m   = meta.get(aid, {})
        lines.append(
            f"[{rank}]  score={combined[i]:.3f}  "
            f"(emb={_normalize(raw_emb)[i]:.2f} "
            f"concept={_normalize(raw_concept)[i]:.2f} "
            f"bm25={_normalize(raw_bm25)[i]:.2f})"
        )
        lines.append(_fmt_award(m))
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def get_award(award_id: str) -> str:
    """Get full details for a specific award: abstract, investigators, PE codes.

    Use after search tools identify an interesting award and you need the
    full text, PI names, co-PIs, and program element codes.
    """
    conn = _db()
    meta = _fetch_award_rows([award_id])
    if not meta:
        return f"No award found with id: {award_id}"

    m     = meta[award_id]
    lines = [f"Award {award_id}\n", _fmt_award(m, include_abstract=True), ""]

    # Investigators
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

    # PE codes
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


@mcp.tool()
def get_researcher(
    full_name: str | None = None,
    investigator_key: str | None = None,
) -> str:
    """Get a researcher's profile: NSF funding history, papers, top topics.

    Looks up data from researcher_fingerprint and researcher_papers tables,
    built by fingerprint_crawl.py + fingerprint_encode.py.

    full_name:        name to search (partial match, case-insensitive)
    investigator_key: exact key (e.g. 'e:jane.smith@nsf.gov' or 'n:jane smith')
                      Use sql_query to find keys if needed.

    Note: researcher_fingerprint is only populated after running
    fingerprint_crawl.py + fingerprint_encode.py.
    """
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
        return (
            f"No researcher fingerprint found for '{full_name or investigator_key}'. "
            "Run fingerprint_crawl.py + fingerprint_encode.py first, or use "
            "sql_query on award_investigator to find the PI's awards directly."
        )

    lines = []
    for fp in rows:
        key = fp["investigator_key"]
        lines += [
            f"Researcher: {fp['full_name']}",
            f"  key         : {key}",
            f"  institution : {fp['inst_name'] or '—'}",
            f"  openalex_id : {fp['openalex_id'] or '—'}",
            f"  NSF awards  : {fp['n_nsf_awards']}",
            f"  papers      : {fp['n_papers']}  ({fp['n_with_abstract']} with abstract)",
            f"  top topics  : {fp['top_topics'] or '—'}",
            f"  fingerprint : built {fp['built_at'][:10]}  model={fp['model_name']}",
            "",
        ]

        # Recent papers
        papers = conn.execute(
            """SELECT paper_title, pub_year, venue, has_abstract
               FROM researcher_papers WHERE investigator_key = ?
               ORDER BY pub_year DESC LIMIT 10""",
            (key,),
        ).fetchall()
        if papers:
            lines.append("  recent papers:")
            for p in papers:
                ab = "✓" if p["has_abstract"] else "title only"
                lines.append(f"    ({p['pub_year']}) {p['paper_title'][:70]}  [{ab}]")
                if p["venue"]:
                    lines.append(f"          venue: {p['venue']}")
            lines.append("")

        # NSF awards
        awards = conn.execute(
            """SELECT a.award_id, a.title, a.source_year, a.award_amount,
                      d.abbreviation AS dir
               FROM award_investigator ai
               JOIN award a ON a.id = ai.award_id
               LEFT JOIN directorate d ON d.id = a.directorate_id
               WHERE ai.full_name = ? AND ai.role_code = 'Principal Investigator'
               ORDER BY a.source_year DESC LIMIT 10""",
            (fp["full_name"],),
        ).fetchall()
        if awards:
            lines.append("  NSF awards (PI, recent first):")
            for aw in awards:
                amt = f"${aw['award_amount']:,.0f}" if aw["award_amount"] else "—"
                lines.append(
                    f"    {aw['award_id']}  ({aw['source_year']})  "
                    f"[{aw['dir'] or '?'}]  {amt}  {aw['title'][:60]}"
                )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
