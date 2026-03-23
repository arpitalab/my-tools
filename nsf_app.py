"""
nsf_app.py — NSF Proposal Matcher (v2: hybrid retrieval)

Three-signal retrieval:
  1. Embedding similarity  — semantic/paraphrase similarity
  2. TF-IDF concept score  — domain-discriminating n-gram overlap
  3. BM25 keyword score    — exact-term matching with length normalization

All three signals are normalized to [0,1] then combined with user-tunable weights.
This separates "topologically associating domains" (BIO) from "topological spaces" (MPS)
because TF-IDF/BM25 heavily penalize math papers for genomics-specific n-grams.

Usage:
    conda run -n thellmbook streamlit run nsf_app.py
"""
from __future__ import annotations

import os
import pickle
import sqlite3
from collections import defaultdict

import re

import numpy as np
import scipy.sparse as sp
import streamlit as st
import plotly.graph_objects as go
try:
    import pysolr as _pysolr
    _PYSOLR_OK = True
except ImportError:
    _PYSOLR_OK = False

_BASE = os.path.dirname(os.path.abspath(__file__))

def _path(name: str) -> str:
    return os.path.join(_BASE, "output", name)


# ---------------------------------------------------------------------------
# Portfolio Evolution — palette & helper
# ---------------------------------------------------------------------------

_DIR_PALETTE: dict[str, str] = {
    "BIO": "#27ae60",
    "CSE": "#2980b9",
    "ENG": "#c0392b",
    "GEO": "#8e44ad",
    "MPS": "#d35400",
    "SBE": "#16a085",
    "EDU": "#f39c12",
    "TIP": "#e91e63",
    "O/D": "#7f8c8d",
    "MCA": "#795548",
}
_DIR_DEFAULT_COLOR = "#bdc3c7"


def _hex_to_rgba(hex_color: str, alpha: float = 0.6) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------

_SPECTER2_MODELS = {"allenai/specter2", "allenai/specter2_aug2023refresh"}

@st.cache_resource(show_spinner="Loading embedding model …")
def load_model(model_name: str):
    if model_name in _SPECTER2_MODELS:
        from nsf_embeddings import SPECTER2Encoder
        return SPECTER2Encoder(model_name)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


_EMB_FILES = {
    "all-MiniLM-L6-v2":                    ("embeddings.npy",             "award_ids.npy"),
    "neuml/pubmedbert-base-embeddings":     ("embeddings_pubmedbert.npy",  "award_ids_pubmedbert.npy"),
    "sentence-transformers/allenai-specter":("embeddings_specter.npy",     "award_ids_specter.npy"),
    "allenai/specter2":                     ("embeddings_specter2.npy",    "award_ids_specter2.npy"),
    "allenai/specter2_aug2023refresh":      ("embeddings_specter2_aug.npy","award_ids_specter2_aug.npy"),
}

@st.cache_resource(show_spinner="Loading embeddings …")
def load_embeddings(model_name: str):
    emb_file, ids_file = _EMB_FILES.get(model_name, ("embeddings.npy", "award_ids.npy"))
    emb = np.load(_path(emb_file))
    ids = np.load(_path(ids_file), allow_pickle=True).tolist()
    return emb, ids                                  # float32 L2-normalized, order = award.id ASC


@st.cache_resource(show_spinner="Loading concept index …")
def load_concept_index():
    with open(_path("concept_vectorizer.pkl"), "rb") as f:
        vec = pickle.load(f)
    mat = sp.load_npz(_path("concept_matrix.npz"))
    ids = np.load(_path("concept_award_ids.npy"), allow_pickle=True).tolist()
    with open(_path("concept_top_terms.pkl"), "rb") as f:
        top_terms = pickle.load(f)
    # Build award_id → row index map
    id_to_row = {aid: i for i, aid in enumerate(ids)}
    return vec, mat, ids, top_terms, id_to_row


@st.cache_resource(show_spinner="Loading BM25 index …")
def load_bm25():
    with open(_path("bm25_index.pkl"), "rb") as f:
        index = pickle.load(f)
    ids = np.load(_path("bm25_award_ids.npy"), allow_pickle=True).tolist()
    id_to_row = {aid: i for i, aid in enumerate(ids)}
    return index, ids, id_to_row


@st.cache_resource
def get_db_conn():
    conn = sqlite3.connect(
        f"file:{_path('nsf_awards.db')}?mode=ro", uri=True, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Portfolio Evolution — cached data loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def _umap_state() -> dict:
    """Mutable holder for UMAP coords shared across Streamlit reruns."""
    return {"coords": None, "ids": None}


def _load_umap_from_disk() -> bool:
    """Populate _umap_state from disk if not already loaded. Returns True if available."""
    state = _umap_state()
    if state["coords"] is not None:
        return True
    cp = _path("umap_coords.npy")
    ip = _path("umap_award_ids.npy")
    if os.path.exists(cp) and os.path.exists(ip):
        state["coords"] = np.load(cp)
        state["ids"]    = np.load(ip, allow_pickle=True).tolist()
        return True
    return False


@st.cache_data(show_spinner=False)
def _load_landscape_meta() -> dict[str, dict]:
    """Award metadata (year, dir, title) in award.id ASC order."""
    conn = get_db_conn()
    rows = conn.execute(
        """
        SELECT a.award_id,
               COALESCE(a.source_year, 0)           AS year,
               COALESCE(d.abbreviation, 'Unknown')  AS dir,
               SUBSTR(COALESCE(a.title, ''), 1, 80) AS title
        FROM award a
        LEFT JOIN directorate d ON d.id = a.directorate_id
        ORDER BY a.id
        """
    ).fetchall()
    return {r["award_id"]: dict(r) for r in rows}


@st.cache_data(show_spinner=False)
def _load_sankey_data(year_start: int, year_end: int) -> list[dict]:
    conn = get_db_conn()
    rows = conn.execute(
        """
        SELECT a.source_year                             AS year,
               COALESCE(d.abbreviation,  'Unknown')     AS dir,
               COALESCE(d.long_name,     'Unknown')     AS dir_name,
               COALESCE(v.abbreviation,  'Unknown')     AS div,
               COALESCE(v.long_name,     'Unknown')     AS div_name,
               SUM(COALESCE(a.award_amount, 0)) / 1e6  AS total_m,
               COUNT(*)                                 AS n_awards
        FROM award a
        LEFT JOIN directorate d ON d.id = a.directorate_id
        LEFT JOIN division    v ON v.id = a.division_id
        WHERE a.source_year BETWEEN ? AND ?
          AND a.source_year IS NOT NULL
        GROUP BY a.source_year, d.abbreviation, v.abbreviation
        ORDER BY a.source_year, total_m DESC
        """,
        (year_start, year_end),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_all_directorates() -> list[str]:
    conn = get_db_conn()
    rows = conn.execute(
        "SELECT abbreviation, long_name FROM directorate ORDER BY abbreviation"
    ).fetchall()
    return [f"{r[0]} — {r[1]}" for r in rows]


def fetch_award_details(conn, award_ids: list[str]) -> dict[str, dict]:
    if not award_ids:
        return {}
    ph = ",".join("?" * len(award_ids))
    rows = conn.execute(
        f"""
        SELECT
            a.award_id,  a.title,  a.award_amount,  a.source_year,
            a.award_instrument,
            COALESCE(a.abstract_narration, a.por_text, a.title) AS abstract,
            d.abbreviation  AS dir,   d.long_name AS dir_name,
            v.abbreviation  AS div,   v.long_name AS div_name,
            i.name          AS inst_name,  i.state_code,  i.country_name
        FROM award a
        LEFT JOIN directorate d ON d.id = a.directorate_id
        LEFT JOIN division    v ON v.id = a.division_id
        LEFT JOIN institution i ON i.id = a.institution_id
        WHERE a.award_id IN ({ph})
        """,
        award_ids,
    ).fetchall()
    return {r["award_id"]: dict(r) for r in rows}


def fetch_investigators(conn, award_ids: list[str]) -> dict[str, list[dict]]:
    if not award_ids:
        return {}
    ph = ",".join("?" * len(award_ids))
    rows = conn.execute(
        f"""
        SELECT a.award_id, ai.nsf_id, ai.full_name, ai.role_code, ai.email,
               i.name AS inst_name, i.state_code
        FROM award_investigator ai
        JOIN award a ON a.id = ai.award_id
        LEFT JOIN institution i ON i.id = a.institution_id
        WHERE a.award_id IN ({ph})
        ORDER BY a.award_id,
                 CASE ai.role_code WHEN 'Principal Investigator' THEN 0 ELSE 1 END
        """,
        award_ids,
    ).fetchall()
    result: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        result[r["award_id"]].append(dict(r))
    return result


def fetch_pe_codes(conn, award_ids: list[str]) -> dict[str, list[dict]]:
    if not award_ids:
        return {}
    ph = ",".join("?" * len(award_ids))
    rows = conn.execute(
        f"""
        SELECT a.award_id, pe.code, pe.text
        FROM award_program_element ape
        JOIN award a ON a.id = ape.award_id
        JOIN program_element pe ON pe.id = ape.program_element_id
        WHERE a.award_id IN ({ph})
        """,
        award_ids,
    ).fetchall()
    result: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        result[r["award_id"]].append({"code": r["code"], "text": r["text"] or ""})
    return result


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; returns zeros if constant."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def compute_concept_scores(
    query_text: str,
    vectorizer,
    matrix: sp.csr_matrix,
) -> np.ndarray:
    """TF-IDF cosine similarity (matrix rows are already L2-normalized by sklearn)."""
    q_vec = vectorizer.transform([query_text])
    return (matrix @ q_vec.T).toarray().ravel().astype(np.float32)


def compute_bm25_scores(query_text: str, bm25_index) -> np.ndarray:
    import re
    _STOP = frozenset(
        "a an the and or but in on at to for of with by from is are was were "
        "be been being have has had do does did will would could should may "
        "might shall can this that these those it its we our they their he she "
        "his her i my you your us them him her which who what when where how "
        "not no nor so yet both either each few more most other some such "
        "than then there thus".split()
    )
    tokens = [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{1,}", query_text.lower())
              if t not in _STOP and len(t) > 2]
    if not tokens:
        return np.zeros(len(bm25_index.idf), dtype=np.float32)
    return bm25_index.get_scores(tokens).astype(np.float32)


def matched_concepts_for(
    query_text: str,
    award_id: str,
    vectorizer,
    top_terms: dict,
    top_k: int = 5,
) -> list[str]:
    """Concept phrases appearing in both query and award (for display)."""
    q_vec = vectorizer.transform([query_text])
    q_terms = set(vectorizer.get_feature_names_out()[q_vec.indices])
    award_terms = [phrase for phrase, _ in top_terms.get(award_id, [])]
    return [c for c in award_terms if c in q_terms][:top_k]


# ---------------------------------------------------------------------------
# Core hybrid retrieval
# ---------------------------------------------------------------------------

def retrieve_hybrid(
    query_text: str,
    model,
    emb: np.ndarray,
    emb_ids: list[str],
    vectorizer,
    concept_matrix: sp.csr_matrix,
    concept_ids: list[str],
    concept_id_to_row: dict,
    bm25_index,
    bm25_ids: list[str],
    bm25_id_to_row: dict,
    w_emb: float,
    w_concept: float,
    w_bm25: float,
    dir_filter: str | None,
    year_filter: int | None,
    top_n: int,
) -> list[dict]:
    """
    Compute three normalized similarity scores per award, combine, return top_n.
    All three index arrays are ordered by award.id ASC so row indices align.
    """
    N = len(emb_ids)

    # --- Signal 1: embedding ---
    q_emb = model.encode(
        [query_text], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)
    raw_emb = emb @ q_emb                                         # (N,)

    # --- Signal 2: TF-IDF concept ---
    raw_concept = compute_concept_scores(query_text, vectorizer, concept_matrix)  # len = len(concept_ids)

    # --- Signal 3: BM25 ---
    raw_bm25 = compute_bm25_scores(query_text, bm25_index)        # len = len(bm25_ids)

    # All three arrays are in award.id ASC order → same row i = same award
    # (guaranteed because all builders use ORDER BY a.id)
    norm_emb     = _normalize(raw_emb)
    norm_concept = _normalize(raw_concept)
    norm_bm25    = _normalize(raw_bm25)

    combined = w_emb * norm_emb + w_concept * norm_concept + w_bm25 * norm_bm25

    # --- Optional filters: zero out non-matching rows ---
    conn = get_db_conn()
    if dir_filter or year_filter:
        dir_abbr = dir_filter.split(" — ")[0] if dir_filter else None
        year_int  = int(year_filter) if year_filter else None
        year_clause = "AND a.source_year = ?" if year_int else ""
        dir_clause  = "AND d.abbreviation = ?" if dir_abbr else ""
        params = []
        if dir_abbr:
            params.append(dir_abbr)
        if year_int:
            params.append(year_int)
        allowed = conn.execute(
            f"""
            SELECT a.award_id FROM award a
            LEFT JOIN directorate d ON d.id = a.directorate_id
            WHERE 1=1 {dir_clause} {year_clause}
            """,
            params,
        ).fetchall()
        allowed_set = {r[0] for r in allowed}
        for i, aid in enumerate(emb_ids):
            if aid not in allowed_set:
                combined[i] = -1.0

    # --- Top candidates ---
    oversample = min(max(top_n * 4, 100), N)
    top_idx = np.argsort(combined)[::-1][:oversample]
    candidate_ids = [emb_ids[int(i)] for i in top_idx]

    meta = fetch_award_details(conn, candidate_ids)

    results = []
    for i in top_idx:
        aid = emb_ids[int(i)]
        m = meta.get(aid, {})
        results.append({
            "award_id": aid,
            "score_combined": float(combined[i]),
            "score_emb":     float(norm_emb[i]),
            "score_concept": float(norm_concept[i]),
            "score_bm25":    float(norm_bm25[i]),
            **{k: m.get(k, "") for k in
               ("title", "abstract", "dir", "dir_name", "div", "div_name",
                "inst_name", "state_code", "award_amount", "source_year",
                "award_instrument")},
        })

    return results[:top_n]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_reviewers(matches, inv_map, top_n=15):
    seen: dict[str, dict] = {}
    for m in matches:
        score = m["score_combined"]
        for inv in inv_map.get(m["award_id"], []):
            key = inv["nsf_id"] if inv.get("nsf_id") else f"__name__{inv['full_name']}"
            if key not in seen:
                seen[key] = {
                    **{k: inv.get(k, "") for k in
                       ("nsf_id","full_name","role_code","email","inst_name","state_code")},
                    "relevance_score": 0.0, "n_matches": 0, "award_ids": [],
                }
            seen[key]["relevance_score"] += score
            seen[key]["n_matches"] += 1
            seen[key]["award_ids"].append(m["award_id"])
    return sorted(seen.values(), key=lambda x: -x["relevance_score"])[:top_n]


def aggregate_pe_suggestions(matches, pe_map, top_n=10):
    pe_score: dict[str, float] = defaultdict(float)
    pe_text: dict[str, str] = {}
    for m in matches:
        for pe in pe_map.get(m["award_id"], []):
            pe_score[pe["code"]] += m["score_combined"]
            pe_text[pe["code"]] = pe.get("text", "")
    ranked = sorted(pe_score.items(), key=lambda x: -x[1])[:top_n]
    return [{"code": c, "text": pe_text[c], "score": s} for c, s in ranked]


def aggregate_dir_div(matches):
    dir_scores: dict[str, float] = defaultdict(float)
    div_scores: dict[str, float] = defaultdict(float)
    for m in matches:
        if m.get("dir"):
            dir_scores[f"{m['dir']} — {m.get('dir_name','')}"] += m["score_combined"]
        if m.get("div"):
            div_scores[f"{m['div']} — {m.get('div_name','')}"] += m["score_combined"]
    return {
        "directorates": sorted(dir_scores.items(), key=lambda x: -x[1])[:3],
        "divisions":    sorted(div_scores.items(), key=lambda x: -x[1])[:3],
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def score_bar(score: float, max_score: float = 1.0) -> str:
    pct = min(int(score / max(max_score, 1e-6) * 100), 100)
    filled = pct // 5
    return f"`{'█'*filled}{'░'*(20-filled)}` {score:.3f}"


def role_badge(role: str) -> str:
    if "Principal Investigator" in role and "Co" not in role and "Former" not in role:
        return "🔵 PI"
    if "Co-Principal" in role:
        return "🟢 Co-PI"
    return "⚪ Other"


def signal_breakdown(m: dict) -> str:
    return (f"emb={m['score_emb']:.2f} · "
            f"concept={m['score_concept']:.2f} · "
            f"bm25={m['score_bm25']:.2f}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="NSF Proposal Matcher", page_icon="🔬", layout="wide")

    with st.sidebar:
        st.title("⚙️ Settings")

        st.markdown("#### Retrieval signals")
        st.caption("Weights are renormalized to sum to 1.")
        w_emb     = st.slider("Embedding (semantic)", 0.0, 1.0, 0.45, 0.05,
                              help="Captures paraphrase / semantic similarity")
        w_concept = st.slider("TF-IDF concepts (domain)", 0.0, 1.0, 0.35, 0.05,
                              help="Discriminates domain-specific n-grams (e.g. 'topologically associating domains')")
        w_bm25    = st.slider("BM25 keywords (exact)", 0.0, 1.0, 0.20, 0.05,
                              help="Exact keyword overlap with length normalization")
        total = w_emb + w_concept + w_bm25
        if total < 0.01:
            w_emb, w_concept, w_bm25 = 0.45, 0.35, 0.20
            total = 1.0
        w_emb /= total; w_concept /= total; w_bm25 /= total
        st.caption(f"Effective: emb={w_emb:.2f}  concept={w_concept:.2f}  bm25={w_bm25:.2f}")

        st.divider()
        st.markdown("#### Filters")
        dirs = ["(all)"] + get_all_directorates()
        dir_filter = st.selectbox("Directorate", dirs, index=0)
        dir_filter = None if dir_filter == "(all)" else dir_filter

        conn = get_db_conn()
        _yrs = conn.execute(
            "SELECT DISTINCT source_year FROM award WHERE source_year IS NOT NULL ORDER BY source_year"
        ).fetchall()
        years = ["(all)"] + [str(r[0]) for r in _yrs]
        year_filter = st.selectbox("Year", years, index=0)
        year_filter = None if year_filter == "(all)" else year_filter

        st.divider()
        st.markdown("#### Results")
        top_n        = st.slider("Matching awards", 5, 30, 10, 5)
        top_reviewers = st.slider("Reviewers", 5, 25, 10, 5)
        top_pe       = st.slider("PE codes", 3, 15, 7)

        st.divider()
        _available_models = {
            "all-MiniLM-L6-v2":              ("all-MiniLM-L6-v2", True),
            "PubMedBERT (bio-focused)":       ("neuml/pubmedbert-base-embeddings", True),
            "SPECTER (scientific papers)":    ("sentence-transformers/allenai-specter",
                                               os.path.exists(_path("embeddings_specter.npy"))),
            "SPECTER 2 (proximity)":          ("allenai/specter2",
                                               os.path.exists(_path("embeddings_specter2.npy"))),
            "SPECTER 2 aug2023 (proximity)":  ("allenai/specter2_aug2023refresh",
                                               os.path.exists(_path("embeddings_specter2_aug.npy"))),
        }
        model_labels = [
            label + (" ⏳ building…" if not avail else "")
            for label, (_, avail) in _available_models.items()
        ]
        model_choice = st.selectbox("Embedding model", model_labels, index=0,
                                    help="SPECTER 2 proximity: best for doc-doc similarity · PubMedBERT: best for bio")
        base_label = model_choice.replace(" ⏳ building…", "")
        model_name, model_ready = _available_models[base_label]
        if not model_ready:
            st.warning("SPECTER embeddings still computing — check back shortly.")
        st.caption("162,618 awards · NSF 2010–2024")

    # ── Page selector ────────────────────────────────────────────────────────
    page = st.sidebar.radio(
        "Page",
        ["🔬 Proposal Matcher", "🧑‍⚖️ Panel Builder", "🔍 Reviewer Lookup",
         "📈 Portfolio", "📋 SOLR Fetch"],
        label_visibility="collapsed",
    )

    if page == "🧑‍⚖️ Panel Builder":
        _render_panel_builder(
            model_name, model_ready,
            w_emb, w_concept, w_bm25,
        )
        return

    if page == "🔍 Reviewer Lookup":
        _render_reviewer_lookup(model_name, model_ready, w_emb, w_concept, w_bm25)
        return

    if page == "📈 Portfolio":
        _render_portfolio_evolution()
        return

    if page == "📋 SOLR Fetch":
        _render_solr_fetch()
        return

    st.title("🔬 NSF Proposal Matcher")
    st.markdown(
        "Paste a proposal abstract to find similar funded awards, potential reviewers, "
        "and recommended PE codes. Uses **hybrid retrieval** (semantic embeddings + "
        "domain concept n-grams + BM25 keywords) to avoid cross-domain false matches."
    )

    query_text = st.text_area("Proposal abstract", height=200,
                              placeholder="Paste your abstract here …")
    run_btn = st.button("Find matches", type="primary", disabled=not query_text.strip())

    if not run_btn or not query_text.strip():
        st.info("Enter an abstract and click **Find matches**.")
        return

    if not model_ready:
        st.error("SPECTER embeddings are still being computed. Please select a different model or try again in a few minutes.")
        return

    # Load all resources
    model                                               = load_model(model_name)
    emb, emb_ids                                        = load_embeddings(model_name)
    vec, concept_mat, concept_ids, top_terms, c_id2row  = load_concept_index()
    bm25_index, bm25_ids, b_id2row                      = load_bm25()
    conn                                                = get_db_conn()

    with st.spinner("Searching with hybrid scoring …"):
        matches = retrieve_hybrid(
            query_text.strip(), model,
            emb, emb_ids,
            vec, concept_mat, concept_ids, c_id2row,
            bm25_index, bm25_ids, b_id2row,
            w_emb, w_concept, w_bm25,
            dir_filter, year_filter,
            top_n,
        )
        award_ids  = [m["award_id"] for m in matches]
        inv_map    = fetch_investigators(conn, award_ids)
        pe_map     = fetch_pe_codes(conn, award_ids)
        reviewers  = aggregate_reviewers(matches, inv_map, top_n=top_reviewers)
        pe_suggest = aggregate_pe_suggestions(matches, pe_map, top_n=top_pe)
        dir_div    = aggregate_dir_div(matches)

    max_score = matches[0]["score_combined"] if matches else 1.0

    tab_awards, tab_reviewers, tab_pe, tab_org, tab_map = st.tabs([
        f"📄 Matching Awards ({top_n})",
        f"👥 Potential Reviewers ({top_reviewers})",
        f"🏷️ PE Code Suggestions ({top_pe})",
        "🏛️ Directorate / Division",
        "🗺️ Award Space Map",
    ])

    # ── Tab 1: Awards ──────────────────────────────────────────────────────
    with tab_awards:
        st.subheader("Most similar funded awards")

        for rank, m in enumerate(matches, 1):
            pes = pe_map.get(m["award_id"], [])
            pe_tags = "  ".join(f"`{p['code']}`" for p in pes)
            concepts = matched_concepts_for(
                query_text, m["award_id"], vec, top_terms, top_k=6
            )
            concept_tags = ("  ".join(f"`{c}`" for c in concepts)) if concepts else "_none_"

            with st.expander(
                f"**#{rank}** · {(m.get('title') or '—')[:85]}  "
                f"· {m.get('dir','?')}/{m.get('div','?')} · {m.get('source_year','?')}",
                expanded=(rank <= 3),
            ):
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    st.markdown(f"**Score** {score_bar(m['score_combined'], max_score)}  "
                                f"<span style='color:gray;font-size:0.8em'>({signal_breakdown(m)})</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"**Title:** {m.get('title','—')}")
                    abstract = m.get("abstract") or ""
                    st.markdown(f"**Abstract:** {abstract[:400]}{'…' if len(abstract)>400 else ''}")
                    st.markdown(f"**Matched concepts:** {concept_tags}")
                with c2:
                    st.markdown(f"**Directorate:** {m.get('dir_name') or m.get('dir','—')}")
                    st.markdown(f"**Division:** {m.get('div_name') or m.get('div','—')}")
                    st.markdown(f"**Year:** {m.get('source_year','—')}")
                    amt = m.get("award_amount")
                    st.markdown(f"**Amount:** ${amt/1e6:.2f}M" if amt else "**Amount:** —")
                    st.markdown(f"**Instrument:** {m.get('award_instrument') or '—'}")
                with c3:
                    st.markdown(f"**Institution:** {m.get('inst_name','—')}")
                    st.markdown(f"**State:** {m.get('state_code','—')}")
                    if pe_tags:
                        st.markdown(f"**PE codes:** {pe_tags}")
                    invs = inv_map.get(m["award_id"], [])
                    pis = [i["full_name"] for i in invs
                           if "Principal Investigator" in i.get("role_code","")
                           and "Co" not in i.get("role_code","")
                           and "Former" not in i.get("role_code","")]
                    if pis:
                        st.markdown(f"**PI:** {', '.join(pis)}")

    # ── Tab 2: Reviewers ───────────────────────────────────────────────────
    with tab_reviewers:
        st.subheader("Potential reviewers")
        st.caption("Ranked by cumulative hybrid score across matched awards.")
        if not reviewers:
            st.warning("No reviewer data found.")
        else:
            max_rev = reviewers[0]["relevance_score"]
            for rev in reviewers:
                c1, c2, c3 = st.columns([3, 3, 2])
                with c1:
                    st.markdown(f"**{rev['full_name'] or '—'}** {role_badge(rev.get('role_code',''))}")
                    st.caption(f"{rev['inst_name'] or '—'}" +
                               (f", {rev['state_code']}" if rev.get("state_code") else ""))
                with c2:
                    if rev.get("email"):
                        st.markdown(f"📧 `{rev['email']}`")
                    if rev.get("nsf_id"):
                        st.caption(f"NSF ID: {rev['nsf_id']}")
                with c3:
                    st.markdown(f"Relevance {score_bar(rev['relevance_score'], max_rev)}")
                    st.caption(f"Appears in {rev['n_matches']} matched award(s)")
                st.divider()

    # ── Tab 3: PE codes ────────────────────────────────────────────────────
    with tab_pe:
        st.subheader("Recommended Program Element codes")
        st.caption("Weighted by hybrid similarity of awards carrying each code.")
        if not pe_suggest:
            st.warning("No PE codes found for matched awards.")
        else:
            max_pe = pe_suggest[0]["score"]
            for i, pe in enumerate(pe_suggest):
                c1, c2, c3 = st.columns([1, 4, 3])
                with c1:
                    st.markdown(f"### `{pe['code']}`")
                with c2:
                    st.markdown(f"**{pe['text'] or 'N/A'}**")
                with c3:
                    st.markdown(f"{score_bar(pe['score'], max_pe)}")
                if i < len(pe_suggest) - 1:
                    st.divider()

    # ── Tab 4: Directorate / Division ──────────────────────────────────────
    with tab_org:
        st.subheader("Suggested Directorate & Division assignment")
        col_dir, col_div = st.columns(2)
        max_dir = dir_div["directorates"][0][1] if dir_div["directorates"] else 1.0
        max_div = dir_div["divisions"][0][1] if dir_div["divisions"] else 1.0

        with col_dir:
            st.markdown("#### Directorate")
            for label, score in dir_div["directorates"]:
                st.markdown(f"**{label}**")
                st.progress(int(score / max_dir * 100) / 100, text=f"{int(score/max_dir*100)}%")
                st.write("")
        with col_div:
            st.markdown("#### Division")
            for label, score in dir_div["divisions"]:
                st.markdown(f"**{label}**")
                st.progress(int(score / max_div * 100) / 100, text=f"{int(score/max_div*100)}%")
                st.write("")

    # ── Tab 5: Award Space Map ─────────────────────────────────────────────
    with tab_map:
        _render_hic_tab(model_name)


# ---------------------------------------------------------------------------
# Hi-C map tab (kept outside main() to use st.cache_data cleanly)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Computing division similarity map …")
def _build_hic_data(emb_file: str, ids_file: str) -> tuple:
    from nsf_hic_map import load_division_map, load_embeddings, division_order, \
                             compute_centroids, dir_boundaries, DIR_COLORS
    div_map      = load_division_map(_path("nsf_awards.db"))
    emb, id2row  = load_embeddings(_path(emb_file), _path(ids_file))
    div_ord      = division_order(div_map)
    centroids    = compute_centroids(emb, id2row, div_map, div_ord)
    sim          = (centroids @ centroids.T).astype(float)
    np.fill_diagonal(sim, 1.0)
    tick_labels  = [f"{d}/{v}" for d, v, _ in div_ord]
    long_labels  = [vn for _, _, vn in div_ord]
    dir_abbrs    = [d for d, v, _ in div_ord]
    dir_blocks   = dir_boundaries(div_ord)
    dir_colors   = DIR_COLORS
    return sim, tick_labels, long_labels, dir_abbrs, dir_blocks, dir_colors


def _render_hic_tab(model_name: str) -> None:
    st.subheader("NSF Award Space — Division Similarity Map")
    st.caption(
        "Each cell shows the cosine similarity between two division centroids "
        "(mean SPECTER embedding). Bright diagonal blocks = cohesive research domains "
        "(like TADs in Hi-C). Off-diagonal warmth = cross-domain overlap."
    )

    emb_file, ids_file = _EMB_FILES.get(model_name, ("embeddings.npy", "award_ids.npy"))
    if not os.path.exists(_path(emb_file)):
        st.warning(f"Embeddings for '{model_name}' not yet available.")
        return

    sim, tick_labels, long_labels, dir_abbrs, dir_blocks, dir_colors = \
        _build_hic_data(emb_file, ids_file)

    N = sim.shape[0]
    vmin = float(np.percentile(sim, 5))

    # Hover text: "BIO/MCB ↔ BIO/DEB\n0.847"
    hover = [[
        f"{tick_labels[i]} ({long_labels[i]})<br>"
        f"{tick_labels[j]} ({long_labels[j]})<br>"
        f"similarity: {sim[i,j]:.3f}"
        for j in range(N)] for i in range(N)]

    fig = go.Figure(go.Heatmap(
        z=sim,
        text=hover,
        hoverinfo="text",
        colorscale=[
            [0.0,  "#FFFFFF"],
            [0.25, "#FFE0D0"],
            [0.55, "#FF9966"],
            [0.80, "#CC2222"],
            [1.0,  "#660000"],
        ],
        zmin=vmin, zmax=1.0,
        colorbar=dict(title="Cosine<br>similarity", thickness=15),
        x=tick_labels, y=tick_labels,
        xgap=0, ygap=0,
    ))

    # Directorate boundary lines
    for dir_abbr, start, end in dir_blocks:
        color = dir_colors.get(dir_abbr, "#333333")
        for pos in (start - 0.5, end - 0.5):
            fig.add_shape(type="line", x0=pos, x1=pos, y0=-0.5, y1=N-0.5,
                          line=dict(color="#222222", width=2))
            fig.add_shape(type="line", y0=pos, y1=pos, x0=-0.5, x1=N-0.5,
                          line=dict(color="#222222", width=2))
        # Directorate label above top boundary
        mid = (start + end) / 2
        fig.add_annotation(x=mid, y=N + 0.8, text=f"<b>{dir_abbr}</b>",
                           showarrow=False, font=dict(size=10, color=color),
                           xref="x", yref="y")
        fig.add_annotation(x=-1.5, y=mid, text=f"<b>{dir_abbr}</b>",
                           showarrow=False, font=dict(size=10, color=color),
                           xref="x", yref="y", textangle=-90)

    fig.update_layout(
        width=820, height=800,
        margin=dict(l=80, r=40, t=60, b=80),
        xaxis=dict(tickfont=dict(size=8), tickangle=90, side="bottom"),
        yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
        plot_bgcolor="#FAFAFA",
        title=dict(text="NSF Division Similarity (Hi-C style)", font=dict(size=14)),
    )

    st.plotly_chart(fig, use_container_width=False)

    # Mini legend
    legend_items = [
        mpatches for _ in []  # avoid unused import warning
    ]
    cols = st.columns(len(dir_blocks))
    for col, (dir_abbr, start, end) in zip(cols, dir_blocks):
        color = dir_colors.get(dir_abbr, "#333")
        col.markdown(
            f"<span style='background:{color};color:white;"
            f"padding:2px 6px;border-radius:3px;font-size:0.75em'>"
            f"**{dir_abbr}**</span>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Panel Builder page
# ---------------------------------------------------------------------------

def _render_panel_builder(model_name, model_ready, w_emb, w_concept, w_bm25):
    from nsf_panel_builder import (
        parse_proposals, build_score_matrix, enrich_with_profiles,
        assign_panel, assignments_to_csv, pool_to_csv,
    )

    st.title("🧑‍⚖️ Panel Builder")
    st.markdown(
        "Paste **20–30 proposal abstracts** separated by `---` lines. "
        "The system finds candidate reviewers from similar funded NSF awards "
        "and builds a constrained panel assignment."
    )

    if not model_ready:
        st.error("Selected embedding model not yet available — choose another in the sidebar.")
        return

    import math as _math

    # ── Input (parsed first so settings can use proposal count) ───────────────
    raw_text = st.text_area(
        "Paste proposal abstracts (separate with ---)",
        height=300,
        placeholder="Abstract of proposal 1 …\n\n---\n\nAbstract of proposal 2 …\n\n---\n\n…",
        key="_panel_raw_text",
    )
    proposals = parse_proposals(raw_text) if raw_text.strip() else []
    if proposals:
        st.caption(f"✅ Detected **{len(proposals)} proposal(s)**")
    elif raw_text.strip():
        st.caption("⚠️ Could not detect multiple proposals — treating as one.")
        proposals = [{"label": "Proposal 1", "abstract": raw_text.strip()}]

    # ── Settings ─────────────────────────────────────────────────────────────
    _n        = len(proposals)
    _auto_cap = _math.ceil(3 * _n / 6) if _n else 1   # formula: ⌈3N/6⌉ = ⌈N/2⌉
    with st.expander("⚙️ Panel settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        n_per_proposal  = col1.number_input("Reviewers per proposal", 1, 6, 3)
        max_load        = col2.number_input("Max proposals per reviewer", 1, 20, 8)
        coi_buffer      = col3.number_input("COI buffer (×)", 1.0, 4.0, 2.0, 0.5,
                                            help="Reserve pool = (buffer-1) × primary reviewers")
        top_candidates  = col1.number_input("Top awards to scan per proposal", 10, 100, 50, 10)
        max_panel_override = col2.number_input(
            "Max panel size (primary reviewers)",
            min_value=1, max_value=200,
            value=max(_auto_cap, 1),
            help=f"Auto = ⌈3 × {_n} / 6⌉ = {_auto_cap}. "
                 "Increase if proposals can't be fully covered.",
        )
        col3.caption(f"Auto-cap: ⌈3 × N / 6⌉ = ⌈N/2⌉ = **{_auto_cap}** for {_n} proposal(s)")
        profile_threshold = col1.slider(
            "Profile similarity threshold",
            min_value=0.05, max_value=0.60, value=0.25, step=0.05,
            help=(
                "A reviewer is eligible for a proposal if the cosine similarity between "
                "their research profile (mean of all their award embeddings) and the "
                "proposal embedding exceeds this value. Lower = more lenient / broader pool."
            ),
        )

    # ── Input ─────────────────────────────────────────────────────────────────
    run_btn = st.button("Build Panel", type="primary",
                        disabled=not proposals)
    if not run_btn or not proposals:
        if not raw_text.strip():
            st.info("Paste abstracts above and click **Build Panel**.")
        return

    # ── Build ─────────────────────────────────────────────────────────────────
    model         = load_model(model_name)
    emb, emb_ids  = load_embeddings(model_name)
    vec, concept_mat, concept_ids, top_terms, c_id2row = load_concept_index()
    bm25_index, bm25_ids, b_id2row = load_bm25()
    conn          = get_db_conn()

    def _retrieve(abstract: str) -> list[dict]:
        return retrieve_hybrid(
            abstract, model,
            emb, emb_ids,
            vec, concept_mat, concept_ids, c_id2row,
            bm25_index, bm25_ids, b_id2row,
            w_emb, w_concept, w_bm25,
            None, None,          # no dir/year filter for panel building
            int(top_candidates),
        )

    progress_bar = st.progress(0, text="Retrieving candidates …")

    def _progress(i, n):
        pct = int(i / n * 100) if n else 100
        progress_bar.progress(pct / 100,
                              text=f"Proposal {i}/{n} — retrieving similar awards …")

    def _encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)

    with st.spinner("Building reviewer pool …"):
        score_matrix, reviewer_keys, reviewer_info, reviewer_awards = build_score_matrix(
            proposals, _retrieve, conn, progress_cb=_progress
        )
        progress_bar.progress(1.0, text="Enriching with reviewer expertise profiles …")
        score_matrix = enrich_with_profiles(
            score_matrix, reviewer_keys, reviewer_awards,
            proposals, emb, emb_ids,
            encode_fn=_encode,
            profile_threshold=float(profile_threshold),
        )
        result = assign_panel(
            score_matrix, reviewer_keys, reviewer_info,
            n_per_proposal=int(n_per_proposal),
            max_load=int(max_load),
            coi_buffer=float(coi_buffer),
            max_panel_size=int(max_panel_override),
        )

    progress_bar.empty()

    # ── Summary ───────────────────────────────────────────────────────────────
    n_primary   = len(result["primary_keys"])
    n_pool      = len(result["pool_keys"])
    n_proposals = len(proposals)
    coverage    = result["coverage"]
    fully_covered = sum(1 for c in coverage if c >= int(n_per_proposal))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Proposals", n_proposals)
    m2.metric("Panel cap", result["max_panel_size"])
    m3.metric("Primary reviewers", n_primary)
    m4.metric("Total pool (w/ reserves)", n_pool)
    m5.metric("Fully assigned", f"{fully_covered}/{n_proposals}")

    if fully_covered < n_proposals:
        st.warning(
            f"{n_proposals - fully_covered} proposal(s) could not be fully assigned "
            f"({n_per_proposal} reviewers each). Try increasing **Top awards to scan** or "
            f"reducing **Max proposals per reviewer**."
        )

    tab_assign, tab_pool, tab_matrix = st.tabs([
        "📋 Assignments", "👥 Reviewer Pool", "🔢 Score Matrix"
    ])

    key_to_info = {info["key"]: info for info in result["pool_info"]}

    # ── Tab: Assignments ──────────────────────────────────────────────────────
    with tab_assign:
        st.subheader("Proposal → Reviewer Assignments")
        for p_idx, prop in enumerate(proposals):
            asn_keys = result["assignments"][p_idx]
            cov = coverage[p_idx]
            label_color = "🟢" if cov >= int(n_per_proposal) else "🟡"
            with st.expander(
                f"{label_color} **{prop['label']}**  — {cov} reviewer(s) assigned",
                expanded=True,
            ):
                if not asn_keys:
                    st.warning("No reviewers assigned — insufficient candidates.")
                for rank, key in enumerate(asn_keys, 1):
                    info = key_to_info.get(key, {})
                    c1, c2, c3 = st.columns([3, 3, 2])
                    c1.markdown(
                        f"**{rank}. {info.get('full_name', key)}**  \n"
                        f"<small>{info.get('inst_name', '—')}</small>",
                        unsafe_allow_html=True,
                    )
                    c2.markdown(
                        f"`{info.get('email', '—')}`  \n"
                        f"<small>NSF ID: {info.get('nsf_id', '—') or '—'}</small>",
                        unsafe_allow_html=True,
                    )
                    score = float(score_matrix[reviewer_keys.index(key), p_idx])
                    c3.markdown(
                        f"Score: **{score:.3f}**  \n"
                        f"PE: {', '.join(info.get('top_pe_codes', [])) or '—'}"
                    )

        st.divider()
        csv_asn = assignments_to_csv(proposals, result)
        st.download_button(
            "⬇️ Download assignments CSV",
            data=csv_asn,
            file_name="panel_assignments.csv",
            mime="text/csv",
        )

    # ── Tab: Reviewer Pool ────────────────────────────────────────────────────
    with tab_pool:
        st.subheader("Reviewer Pool (Primary + Reserves)")
        st.caption(
            f"**Primary** = assigned to at least one proposal. "
            f"**Reserve** = ranked backup for COI replacements."
        )
        for rank, info in enumerate(result["pool_info"], 1):
            badge  = "🔵 Primary" if info.get("is_primary") else "⚪ Reserve"
            load   = info.get("assigned_proposals", 0)
            tscore = info.get("total_score", 0.0)
            with st.expander(
                f"**#{rank}** · {info.get('full_name','—')}  "
                f"· {badge}  · {load} proposal(s)",
                expanded=False,
            ):
                c1, c2 = st.columns(2)
                c1.markdown(
                    f"**Institution:** {info.get('inst_name', '—')}  \n"
                    f"**State:** {info.get('state_code', '—')}  \n"
                    f"**Email:** `{info.get('email', '—')}`  \n"
                    f"**NSF ID:** {info.get('nsf_id', '—') or '—'}"
                )
                c2.markdown(
                    f"**Total relevance score:** {tscore:.4f}  \n"
                    f"**Matched NSF awards:** {info.get('n_awards', 0)}  \n"
                    f"**Roles:** {', '.join(info.get('roles', []))}  \n"
                    f"**Top PE codes:** {', '.join(info.get('top_pe_codes', [])) or '—'}"
                )

        st.divider()
        csv_pool = pool_to_csv(result)
        st.download_button(
            "⬇️ Download reviewer pool CSV",
            data=csv_pool,
            file_name="reviewer_pool.csv",
            mime="text/csv",
        )

    # ── Tab: Score Matrix ─────────────────────────────────────────────────────
    with tab_matrix:
        st.subheader("Reviewer × Proposal Score Matrix")
        st.caption(
            "🔵 **Primary** rows: reviewers assigned under the 3-per-proposal / "
            "8-per-reviewer constraints.  ⚪ **Reserve** rows: top backup candidates "
            "for COI replacements.  ✓ = assigned cell."
        )
        import plotly.graph_objects as _go

        n_primary_rows = result["n_primary"]
        pool_idxs  = [reviewer_keys.index(k) for k in result["pool_keys"]]
        sub_matrix = score_matrix[pool_idxs, :]   # (pool × proposals)

        # Row labels: flag primary vs reserve and show load
        row_labels = []
        for i, info in enumerate(result["pool_info"]):
            name = (info.get("full_name") or result["pool_keys"][i])[:28]
            if info["is_primary"]:
                load = info["assigned_proposals"]
                row_labels.append(f"● {name} ({load})")
            else:
                row_labels.append(f"○ {name}")

        col_labels = [p["label"][:22] for p in proposals]

        # Build hover text and cell annotations for assigned cells
        hover      = []
        annotations = []
        for row_i, r_key in enumerate(result["pool_keys"]):
            row_hover = []
            for p_idx in range(n_proposals):
                score = float(sub_matrix[row_i, p_idx])
                asn   = r_key in result["assignments"][p_idx]
                row_hover.append(
                    f"<b>{row_labels[row_i]}</b><br>"
                    f"Proposal: {col_labels[p_idx]}<br>"
                    f"Score: {score:.3f}"
                    + ("<br><b>✓ ASSIGNED</b>" if asn else "")
                )
                if asn:
                    annotations.append(dict(
                        x=p_idx, y=row_i,
                        text="✓",
                        showarrow=False,
                        font=dict(color="white", size=11, family="Arial Black"),
                        xref="x", yref="y",
                    ))
            hover.append(row_hover)

        # Two-tone colorscale: lighter blues for reserves
        fig = _go.Figure()

        # Primary block (rows 0 .. n_primary_rows-1)
        if n_primary_rows > 0:
            fig.add_trace(_go.Heatmap(
                z=sub_matrix[:n_primary_rows].tolist(),
                x=col_labels,
                y=row_labels[:n_primary_rows],
                text=hover[:n_primary_rows],
                hoverinfo="text",
                colorscale="Blues",
                zmin=0, zmax=float(sub_matrix.max()) or 1.0,
                showscale=True,
                colorbar=dict(title="Score", thickness=12, len=0.6, y=0.7),
                name="Primary",
            ))

        # Reserve block (rows n_primary_rows .. end)
        if len(pool_idxs) > n_primary_rows:
            fig.add_trace(_go.Heatmap(
                z=sub_matrix[n_primary_rows:].tolist(),
                x=col_labels,
                y=row_labels[n_primary_rows:],
                text=hover[n_primary_rows:],
                hoverinfo="text",
                colorscale="Greys",
                zmin=0, zmax=float(sub_matrix.max()) or 1.0,
                showscale=False,
                name="Reserve",
            ))

        # Horizontal separator line between primary and reserve
        if 0 < n_primary_rows < len(pool_idxs):
            fig.add_shape(
                type="line",
                x0=-0.5, x1=n_proposals - 0.5,
                y0=n_primary_rows - 0.5, y1=n_primary_rows - 0.5,
                line=dict(color="#E63946", width=2.5, dash="dash"),
                xref="x", yref="y",
            )
            fig.add_annotation(
                x=n_proposals - 0.5, y=n_primary_rows - 0.5,
                text="  ← Reserve pool below",
                showarrow=False,
                font=dict(color="#E63946", size=10),
                xanchor="left", xref="x", yref="y",
            )

        # Assigned-cell checkmarks
        for ann in annotations:
            fig.add_annotation(**ann)

        total_rows = len(pool_idxs)
        fig.update_layout(
            height=max(350, 24 * total_rows + 100),
            margin=dict(l=200, r=20, t=50, b=130),
            xaxis=dict(tickangle=40, tickfont=dict(size=9), side="bottom"),
            yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
            plot_bgcolor="#F8F8F8",
            title=dict(
                text=f"● Primary ({n_primary_rows} reviewers)  |  ○ Reserve ({total_rows - n_primary_rows})  |  ✓ Assigned",
                font=dict(size=12),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Reviewer Lookup page
# ---------------------------------------------------------------------------

_REVIEWER_HARD_CAP = 50    # show at most this many reviewers


def _render_reviewer_lookup(model_name, model_ready, w_emb, w_concept, w_bm25):
    st.title("🔍 Reviewer Lookup")
    st.markdown(
        "Enter keywords or a short description to find NSF-funded researchers "
        "who work in that area. Results are ranked by relevance across all "
        "awards matching your query."
    )

    with st.form("reviewer_lookup_form"):
        keywords = st.text_area(
            "Keywords / description",
            height=100,
            placeholder=(
                "e.g.  chromatin organization, Hi-C, topologically associating domains\n"
                "or:   machine learning for protein structure prediction"
            ),
        )
        col1, col2 = st.columns([1, 1])
        top_awards  = col1.number_input("Awards to scan", min_value=10, max_value=500, value=100, step=10,
                                        help="Number of top-matching awards to pull reviewers from")
        top_reviewers = col2.number_input("Max reviewers to show", min_value=5, max_value=_REVIEWER_HARD_CAP,
                                          value=20, step=5,
                                          help=f"Capped at {_REVIEWER_HARD_CAP}.")
        search_btn = st.form_submit_button("Search", type="primary")

    if not search_btn or not keywords.strip():
        st.info("Enter keywords above and click **Search**.")
        return

    if not model_ready:
        st.error("Selected embedding model is not yet available. Choose a different model.")
        return

    model                                               = load_model(model_name)
    emb, emb_ids                                        = load_embeddings(model_name)
    vec, concept_mat, concept_ids, top_terms, c_id2row  = load_concept_index()
    bm25_index, bm25_ids, b_id2row                      = load_bm25()
    conn                                                = get_db_conn()

    with st.spinner("Searching …"):
        matches = retrieve_hybrid(
            keywords.strip(), model,
            emb, emb_ids,
            vec, concept_mat, concept_ids, c_id2row,
            bm25_index, bm25_ids, b_id2row,
            w_emb, w_concept, w_bm25,
            None, None,
            int(top_awards),
        )

    if not matches:
        st.warning("No awards matched your keywords. Try broader terms.")
        return

    award_ids = [m["award_id"] for m in matches]
    inv_map   = fetch_investigators(conn, award_ids)

    # Collect unique reviewers with cumulative relevance score.
    # Dedup key priority: email → normalized name (drop middle initials, lowercase).
    # This handles the same person appearing with nsf_id in XML awards but without
    # it in API-ingested awards (where nsf_id is stored as empty string).
    def _reviewer_key(inv: dict) -> str | None:
        email = (inv.get("email") or "").strip().lower()
        if email:
            return f"e:{email}"
        name = (inv.get("full_name") or "").strip().lower()
        if not name:
            return None
        # Drop single-character tokens (middle initials, punctuation fragments)
        parts = [p.strip(".,") for p in name.split() if len(p.strip(".,")) > 1]
        return f"n:{' '.join(parts)}" if parts else None

    seen: dict[str, dict] = {}
    for m in matches:
        score = m["score_combined"]
        for inv in inv_map.get(m["award_id"], []):
            key = _reviewer_key(inv)
            if key is None:
                continue
            name = (inv.get("full_name") or "").strip()
            if key not in seen:
                seen[key] = {
                    "full_name":       name,
                    "email":           (inv.get("email") or "").strip(),
                    "inst_name":       inv.get("inst_name") or "",
                    "state_code":      inv.get("state_code") or "",
                    "role_code":       inv.get("role_code") or "",
                    "relevance_score": 0.0,
                    "n_awards":        0,
                    "award_ids":       [],
                }
            else:
                # Prefer the richer record: fill in missing fields
                if not seen[key]["email"] and inv.get("email"):
                    seen[key]["email"] = inv["email"].strip()
                if not seen[key]["inst_name"] and inv.get("inst_name"):
                    seen[key]["inst_name"] = inv["inst_name"]
            seen[key]["relevance_score"] += score
            seen[key]["n_awards"]        += 1
            seen[key]["award_ids"].append(m["award_id"])

    ranked = sorted(seen.values(), key=lambda x: -x["relevance_score"])
    n_unique = len(ranked)

    # ── Display ranked table ──────────────────────────────────────────────────
    display = ranked[:min(int(top_reviewers), _REVIEWER_HARD_CAP)]
    st.info(f"Found **{n_unique} unique reviewers** across {len(matches)} matching awards — "
            f"showing top {len(display)} by relevance score.")

    max_score = display[0]["relevance_score"] if display else 1.0

    for rank, rev in enumerate(display, start=1):
        with st.expander(
            f"#{rank}  {rev['full_name']}  ·  {rev['inst_name'] or 'Unknown institution'}  "
            f"·  {rev['n_awards']} award(s)  ·  score {rev['relevance_score']:.2f}",
            expanded=(rank <= 5),
        ):
            bar = score_bar(rev["relevance_score"], max_score)
            st.markdown(f"**Relevance** {bar}")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Awards matched", rev["n_awards"])
            col_b.markdown(f"**Email**  \n{rev['email'] or '—'}")
            col_c.markdown(f"**State**  \n{rev['state_code'] or '—'}")

            # Show the matching award titles
            if rev["award_ids"]:
                ph = ",".join("?" * len(rev["award_ids"]))
                rows = conn.execute(
                    f"SELECT award_id, title, source_year FROM award WHERE award_id IN ({ph})",
                    rev["award_ids"],
                ).fetchall()
                rows_sorted = sorted(rows, key=lambda r: -(r["source_year"] or 0))
                st.markdown("**Matching awards:**")
                for r in rows_sorted[:8]:
                    yr   = r["source_year"] or "?"
                    title = r["title"] or "(no title)"
                    st.markdown(f"- `{r['award_id']}` ({yr}) {title}")
                if len(rows_sorted) > 8:
                    st.caption(f"… and {len(rows_sorted) - 8} more")


# ---------------------------------------------------------------------------
# Portfolio Evolution — figure builders
# ---------------------------------------------------------------------------

def _build_umap_figure(
    coords: np.ndarray,
    ids: list[str],
    meta: dict[str, dict],
    color_by: str,
    year_lo: int,
    year_hi: int,
) -> go.Figure:
    """Interactive UMAP scatter (WebGL) colored by directorate or year."""
    years  = np.array([meta.get(aid, {}).get("year",  0)         for aid in ids], dtype=int)
    dirs   = np.array([meta.get(aid, {}).get("dir",   "Unknown") for aid in ids])
    titles = [f"{meta.get(aid, {}).get('title', '')[:60]}" for aid in ids]

    in_range = (years >= year_lo) & (years <= year_hi)
    fig = go.Figure()

    if color_by == "Directorate":
        for dir_name, color in _DIR_PALETTE.items():
            idx_in  = np.where((dirs == dir_name) & in_range)[0]
            idx_out = np.where((dirs == dir_name) & ~in_range)[0]
            if len(idx_in):
                fig.add_trace(go.Scattergl(
                    x=coords[idx_in, 0], y=coords[idx_in, 1],
                    mode="markers", name=dir_name, legendgroup=dir_name,
                    marker=dict(size=3, color=color, opacity=0.65),
                    text=[f"<b>{titles[i]}</b><br>{dir_name} · {years[i]}" for i in idx_in],
                    hoverinfo="text",
                ))
            if len(idx_out):
                fig.add_trace(go.Scattergl(
                    x=coords[idx_out, 0], y=coords[idx_out, 1],
                    mode="markers", name=dir_name, legendgroup=dir_name,
                    showlegend=False,
                    marker=dict(size=2, color=color, opacity=0.07),
                    hoverinfo="skip",
                ))
        # Handle directorates not in palette
        idx_unk_in  = np.where((~np.isin(dirs, list(_DIR_PALETTE))) & in_range)[0]
        idx_unk_out = np.where((~np.isin(dirs, list(_DIR_PALETTE))) & ~in_range)[0]
        if len(idx_unk_in):
            fig.add_trace(go.Scattergl(
                x=coords[idx_unk_in, 0], y=coords[idx_unk_in, 1],
                mode="markers", name="Other", legendgroup="Other",
                marker=dict(size=3, color=_DIR_DEFAULT_COLOR, opacity=0.5),
                text=[f"<b>{titles[i]}</b><br>{dirs[i]} · {years[i]}" for i in idx_unk_in],
                hoverinfo="text",
            ))
        if len(idx_unk_out):
            fig.add_trace(go.Scattergl(
                x=coords[idx_unk_out, 0], y=coords[idx_unk_out, 1],
                mode="markers", name="Other", legendgroup="Other",
                showlegend=False,
                marker=dict(size=2, color=_DIR_DEFAULT_COLOR, opacity=0.05),
                hoverinfo="skip",
            ))
    else:  # color by Year
        idx_out = np.where(~in_range)[0]
        if len(idx_out):
            fig.add_trace(go.Scattergl(
                x=coords[idx_out, 0], y=coords[idx_out, 1],
                mode="markers", name="Outside range",
                marker=dict(size=2, color="#555555", opacity=0.06),
                hoverinfo="skip",
            ))
        idx_in = np.where(in_range)[0]
        if len(idx_in):
            fig.add_trace(go.Scattergl(
                x=coords[idx_in, 0], y=coords[idx_in, 1],
                mode="markers", name="Awards",
                marker=dict(
                    size=3, opacity=0.7,
                    color=years[idx_in],
                    colorscale="Plasma",
                    showscale=True,
                    colorbar=dict(title="Year", thickness=14, len=0.7),
                    cmin=year_lo, cmax=year_hi,
                ),
                text=[f"<b>{titles[i]}</b><br>{dirs[i]} · {years[i]}" for i in idx_in],
                hoverinfo="text",
            ))

    fig.update_layout(
        height=640,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        legend=dict(itemsizing="constant", font=dict(size=11)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="closest",
    )
    return fig


def _build_sankey_year_dir(rows: list[dict], dirs_filter: list[str]) -> go.Figure:
    """Year → Directorate Sankey weighted by total awarded $M."""
    data = rows if not dirs_filter else [r for r in rows if r["dir"] in dirs_filter]

    # Aggregate by (year, dir)
    agg: dict[tuple, dict] = {}
    for r in data:
        k = (r["year"], r["dir"])
        if k not in agg:
            agg[k] = {"total_m": 0.0, "n_awards": 0}
        agg[k]["total_m"]  += r["total_m"]
        agg[k]["n_awards"] += r["n_awards"]

    years = sorted({k[0] for k in agg})
    dirs  = sorted({k[1] for k in agg})
    year_idx = {y: i             for i, y in enumerate(years)}
    dir_idx  = {d: len(years)+i  for i, d in enumerate(dirs)}

    src, tgt, val, hover = [], [], [], []
    for (year, dir_), v in agg.items():
        if v["total_m"] <= 0:
            continue
        src.append(year_idx[year])
        tgt.append(dir_idx[dir_])
        val.append(v["total_m"])
        hover.append(f"{year} → {dir_}: ${v['total_m']:.1f}M · {v['n_awards']:,} awards")

    year_colors = ["rgba(80,130,210,0.85)"] * len(years)
    dir_colors  = [_hex_to_rgba(_DIR_PALETTE.get(d, _DIR_DEFAULT_COLOR), 0.85) for d in dirs]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=12, thickness=16,
            label=[str(y) for y in years] + dirs,
            color=year_colors + dir_colors,
        ),
        link=dict(
            source=src, target=tgt, value=val,
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    ))
    fig.update_layout(
        title=dict(text="Funding Flow: Year → Directorate  (width ∝ $M awarded)", font=dict(size=13)),
        height=620, font_size=11,
        margin=dict(l=15, r=15, t=45, b=10),
    )
    return fig


def _build_sankey_dir_div(rows: list[dict], dirs_filter: list[str]) -> go.Figure:
    """Directorate → Division Sankey weighted by total awarded $M."""
    data = rows if not dirs_filter else [r for r in rows if r["dir"] in dirs_filter]

    # Aggregate by (dir, div) — prefix div key to avoid cross-dir collisions
    agg: dict[tuple, dict] = {}
    for r in data:
        k = (r["dir"], r["div"])
        if k not in agg:
            agg[k] = {"total_m": 0.0, "n_awards": 0, "div_name": r["div_name"]}
        agg[k]["total_m"]  += r["total_m"]
        agg[k]["n_awards"] += r["n_awards"]

    dirs    = sorted({k[0] for k in agg})
    divkeys = sorted({f"{k[0]}:{k[1]}" for k in agg})
    dir_idx = {d: i             for i, d in enumerate(dirs)}
    div_idx = {dk: len(dirs)+i  for i, dk in enumerate(divkeys)}

    div_labels = []
    for dk in divkeys:
        d, v = dk.split(":", 1)
        label = (agg.get((d, v), {}).get("div_name") or v)[:32]
        div_labels.append(label)

    dir_colors = [_hex_to_rgba(_DIR_PALETTE.get(d, _DIR_DEFAULT_COLOR), 0.9) for d in dirs]
    div_colors = [_hex_to_rgba(_DIR_PALETTE.get(dk.split(":")[0], _DIR_DEFAULT_COLOR), 0.45)
                  for dk in divkeys]

    src, tgt, val, hover = [], [], [], []
    for (dir_, div_), v in agg.items():
        if v["total_m"] <= 0:
            continue
        dk = f"{dir_}:{div_}"
        src.append(dir_idx[dir_])
        tgt.append(div_idx[dk])
        val.append(v["total_m"])
        hover.append(f"{dir_} → {v['div_name']}: ${v['total_m']:.1f}M · {v['n_awards']:,} awards")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=8, thickness=16,
            label=dirs + div_labels,
            color=dir_colors + div_colors,
        ),
        link=dict(
            source=src, target=tgt, value=val,
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    ))
    fig.update_layout(
        title=dict(text="Funding Flow: Directorate → Division  (width ∝ $M awarded)", font=dict(size=13)),
        height=720, font_size=9,
        margin=dict(l=15, r=15, t=45, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Portfolio Evolution — render
# ---------------------------------------------------------------------------

def _render_portfolio_evolution() -> None:
    st.title("📈 Portfolio Evolution")
    st.markdown(
        "Explore how NSF's research portfolio evolved 2010–2024.  "
        "**Research Landscape** maps all 162k awards semantically; "
        "**Funding Flow** shows dollar allocations across directorates and divisions."
    )

    tab_umap, tab_sankey = st.tabs(["🗺️ Research Landscape (UMAP)", "💸 Funding Flow (Sankey)"])

    # ── Tab 1: UMAP ──────────────────────────────────────────────────────────
    with tab_umap:
        has_coords = _load_umap_from_disk()

        if not has_coords:
            st.info(
                "The 2D research landscape has not been computed yet. "
                "It projects all 162,618 SPECTER2 embeddings (768-dim) to 2D via UMAP. "
                "**First-time computation takes ~15–25 min** on CPU and is saved to disk "
                "so future loads are instant."
            )
            if st.button("🚀 Generate Research Landscape", type="primary"):
                # Check SPECTER2 embeddings exist
                emb_path = _path("embeddings_specter2.npy")
                ids_path = _path("award_ids_specter2.npy")
                if not os.path.exists(emb_path):
                    st.error(
                        f"SPECTER2 embeddings not found at {emb_path}. "
                        "Run `python nsf_embeddings.py --model allenai/specter2` first."
                    )
                else:
                    with st.spinner("Running UMAP on 162k × 768 embeddings … (~15–25 min on CPU, ~5 min on M-series)"):
                        import umap as umap_lib
                        emb  = np.load(emb_path)
                        uids = np.load(ids_path, allow_pickle=True).tolist()
                        reducer = umap_lib.UMAP(
                            n_components=2,
                            n_neighbors=15,
                            min_dist=0.1,
                            metric="cosine",
                            low_memory=True,
                            random_state=42,
                            verbose=False,
                        )
                        coords = reducer.fit_transform(emb).astype(np.float32)
                        np.save(_path("umap_coords.npy"), coords)
                        np.save(_path("umap_award_ids.npy"), np.array(uids, dtype=object))
                        state = _umap_state()
                        state["coords"] = coords
                        state["ids"]    = uids
                    st.success("UMAP computed and saved.")
                    st.rerun()
            return

        state  = _umap_state()
        coords = state["coords"]
        ids    = state["ids"]
        meta   = _load_landscape_meta()

        # Controls
        col1, col2, col3 = st.columns([2, 4, 2])
        with col1:
            color_by = st.radio("Color by", ["Directorate", "Year"], horizontal=True,
                                key="umap_color")
        with col2:
            year_lo, year_hi = st.slider(
                "Highlight year range", 2010, 2024, (2010, 2024),
                help="Points outside the range are dimmed to the background.",
                key="umap_years",
            )
        with col3:
            all_years_arr = np.array([meta.get(aid, {}).get("year", 0) for aid in ids], dtype=int)
            n_in = int(np.sum((all_years_arr >= year_lo) & (all_years_arr <= year_hi)))
            st.metric("Highlighted", f"{n_in:,}")

        with st.spinner("Rendering …"):
            fig = _build_umap_figure(coords, ids, meta, color_by, year_lo, year_hi)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Each point = one NSF award. Position reflects semantic similarity "
            "(SPECTER2 + UMAP). Awards doing similar science cluster together "
            "regardless of directorate. Hover for title. Toggle directorates in the legend."
        )

    # ── Tab 2: Sankey ─────────────────────────────────────────────────────────
    with tab_sankey:
        c1, c2, c3 = st.columns([3, 3, 2])
        with c1:
            yr_range = st.slider(
                "Year range", 2010, 2024, (2015, 2024), key="sankey_years",
            )
        with c2:
            known_dirs = sorted(set(_DIR_PALETTE.keys()))
            dirs_sel = st.multiselect(
                "Filter directorates (blank = all)", known_dirs,
                default=[], key="sankey_dirs",
            )
        with c3:
            flow_mode = st.radio(
                "View", ["Year → Directorate", "Directorate → Division"],
                key="sankey_mode",
            )

        rows = _load_sankey_data(yr_range[0], yr_range[1])
        if not rows:
            st.warning("No data for selected range.")
            return

        total_m = sum(r["total_m"] for r in rows)
        n_aw    = sum(r["n_awards"] for r in rows)
        st.caption(
            f"**{n_aw:,} awards · ${total_m:,.0f}M total** for "
            f"{yr_range[0]}–{yr_range[1]}"
            + (f" · filtered to {', '.join(dirs_sel)}" if dirs_sel else "")
        )

        if flow_mode == "Year → Directorate":
            fig = _build_sankey_year_dir(rows, dirs_sel)
        else:
            fig = _build_sankey_dir_div(rows, dirs_sel)

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Flow width ∝ total awarded dollars. "
            "Hover over any flow for exact amounts and award counts."
        )


# ---------------------------------------------------------------------------
# SOLR Fetch — constants & helpers
# ---------------------------------------------------------------------------

_SOLR_URL     = "http://dis-checker-a01.ad.nsf.gov/solr/proposals/"
_SOLR_TIMEOUT = 30

# Ordered field list shown in the extractor multiselect
_SOLR_FIELDS: list[tuple[str, str]] = [
    ("id",               "Proposal ID"),
    ("title",            "Title"),
    ("pi_name",          "PI Name"),
    ("pi_all",           "All PIs"),
    ("inst",             "Institution"),
    ("inst_state",       "State"),
    ("pi_gender",        "PI Gender"),
    ("directorate",      "Directorate"),
    ("division",         "Division"),
    ("managing_program", "Managing Program"),
    ("status",           "Status"),
    ("received_year",    "Year"),
    ("received",         "Received Date"),
    ("award_date",       "Award Date"),
    ("award_amount",     "Award Amount ($)"),
    ("requested_amount", "Requested Amount ($)"),
    ("funding_program",  "Funding Program"),
    ("panel_id",         "Panel ID"),
    ("panel_name",       "Panel Name"),
    ("summary",          "Summary"),
    ("description",      "Description (narrative)"),
    ("bio",              "PI Bio"),
    ("data_management",  "Data Management Plan"),
]

_COLLAB_RE_APP = re.compile(r'(?i)^collaborative\s+research\s*:\s*')


def _solr_client_app():
    return _pysolr.Solr(_SOLR_URL, timeout=_SOLR_TIMEOUT)


def _parse_proposal_ids(raw: str) -> list[str]:
    """Accept newline- or comma-separated IDs; filter to digit-only strings."""
    parts = re.split(r"[\n,]+", raw)
    return [p.strip() for p in parts if re.fullmatch(r"\d+", p.strip())]


def _solr_fetch_batched(ids: list[str], fields: str) -> tuple[list[dict], list[str]]:
    """Fetch SOLR docs for *ids* in batches of 50.
    Returns (docs_in_id_order, missing_ids)."""
    fetched: dict[str, dict] = {}
    client = _solr_client_app()
    for i in range(0, len(ids), 50):
        batch = ids[i : i + 50]
        results = client.search(
            "id:(" + " OR ".join(batch) + ")",
            fl=fields,
            rows=len(batch),
        )
        for doc in results:
            fetched[str(doc.get("id", ""))] = doc
    # preserve input order
    docs    = [fetched[i] for i in ids if i in fetched]
    missing = [i for i in ids if i not in fetched]
    return docs, missing


def _dedup_app(docs: list[dict], text_field: str) -> list[dict]:
    """Collapse collaborative proposals that share the same normalised title."""
    groups: dict[str, list[dict]] = {}
    for doc in docs:
        key = _COLLAB_RE_APP.sub("", str(doc.get("title", ""))).strip().lower()
        groups.setdefault(key, []).append(doc)
    result = []
    for group in groups.values():
        best = max(group, key=lambda d: len(str(d.get(text_field, ""))))
        result.append(best)
    return result


def _render_solr_fetch() -> None:
    st.title("📋 SOLR Fetch")
    st.markdown(
        "Pull proposal data directly from the NSF SOLR database by ID. "
        "**Requires NSF VPN.**"
    )

    if not _PYSOLR_OK:
        st.error("`pysolr` is not installed. Run: `pip install pysolr`")
        return

    tab_panel, tab_fields = st.tabs(
        ["📄 Panel Input Builder", "🔎 Field Extractor"]
    )

    # ------------------------------------------------------------------
    # Tab 1 — Panel Input Builder
    # ------------------------------------------------------------------
    with tab_panel:
        st.markdown(
            "Paste proposal IDs to fetch their summaries in the format "
            "expected by **nsf_panel_builder**."
        )
        ids_raw = st.text_area(
            "Proposal IDs (one per line or comma-separated)",
            height=150,
            placeholder="2412345\n2412346\n2412347",
            key="panel_ids",
        )
        text_field = st.radio(
            "Text field",
            ["summary", "description"],
            horizontal=True,
            key="panel_text_field",
        )
        fetch_btn = st.button("Fetch & Build", type="primary", key="panel_fetch")

        if fetch_btn:
            ids = _parse_proposal_ids(ids_raw)
            if not ids:
                st.warning("No valid numeric proposal IDs found.")
            else:
                with st.spinner(f"Fetching {len(ids)} proposals from SOLR …"):
                    try:
                        docs, missing = _solr_fetch_batched(
                            ids, f"id,title,{text_field}"
                        )
                    except Exception as e:
                        st.error(f"SOLR error (check VPN): {e}")
                        docs, missing = [], ids

                if docs:
                    deduped = _dedup_app(docs, text_field)
                    collapsed = len(docs) - len(deduped)
                    blocks = []
                    for doc in deduped:
                        title    = str(doc.get("title", "")).strip()
                        abstract = str(doc.get(text_field, "")).strip()
                        if abstract:
                            blocks.append(f"{title}\n{abstract}")

                    output_text = "\n---\n".join(blocks) + "\n---\n"

                    st.success(
                        f"Retrieved **{len(deduped)}** proposals"
                        + (f" ({collapsed} collaborative duplicates collapsed)" if collapsed else "")
                        + (f"  ·  **{len(missing)} not found**: {', '.join(missing)}" if missing else "")
                    )
                    st.text_area("Panel input (copy or download)", output_text, height=400)
                    st.download_button(
                        "⬇️ Download .txt",
                        data=output_text,
                        file_name="panel_input.txt",
                        mime="text/plain",
                    )
                elif missing:
                    st.error(f"None of the {len(ids)} IDs were found in SOLR (check VPN).")

    # ------------------------------------------------------------------
    # Tab 2 — Field Extractor
    # ------------------------------------------------------------------
    with tab_fields:
        st.markdown(
            "Fetch any combination of SOLR fields for one or more proposals. "
            "Results can be downloaded as CSV."
        )
        ids_raw2 = st.text_area(
            "Proposal IDs (one per line or comma-separated)",
            height=150,
            placeholder="2412345\n2412346",
            key="field_ids",
        )

        field_labels  = [label for _, label in _SOLR_FIELDS]
        field_keys    = [key   for key, _   in _SOLR_FIELDS]
        default_keys  = {"id", "title", "pi_name", "inst", "directorate",
                         "status", "received_year", "award_amount"}
        default_idx   = [i for i, k in enumerate(field_keys) if k in default_keys]

        selected_idx = st.multiselect(
            "Fields to extract",
            options=list(range(len(_SOLR_FIELDS))),
            default=default_idx,
            format_func=lambda i: f"{field_keys[i]}  —  {field_labels[i]}",
            key="field_select",
        )

        fetch_btn2 = st.button("Fetch Fields", type="primary", key="field_fetch")

        if fetch_btn2:
            ids2 = _parse_proposal_ids(ids_raw2)
            if not ids2:
                st.warning("No valid numeric proposal IDs found.")
            elif not selected_idx:
                st.warning("Select at least one field.")
            else:
                chosen_keys = [field_keys[i] for i in selected_idx]
                # always include id so we can match rows
                fl_set = list(dict.fromkeys(["id"] + chosen_keys))
                fl_str = ",".join(fl_set)

                with st.spinner(f"Fetching {len(ids2)} proposals …"):
                    try:
                        docs2, missing2 = _solr_fetch_batched(ids2, fl_str)
                    except Exception as e:
                        st.error(f"SOLR error (check VPN): {e}")
                        docs2, missing2 = [], ids2

                if docs2:
                    import pandas as pd
                    rows_out = []
                    for doc in docs2:
                        row = {k: doc.get(k, "") for k in fl_set}
                        # flatten list values (e.g. pi_all)
                        for k, v in row.items():
                            if isinstance(v, list):
                                row[k] = "; ".join(str(x) for x in v)
                        rows_out.append(row)

                    df = pd.DataFrame(rows_out, columns=fl_set)

                    if missing2:
                        st.warning(f"{len(missing2)} IDs not found: {', '.join(missing2)}")
                    st.success(f"Retrieved **{len(df)}** proposals.")
                    st.dataframe(df, use_container_width=True)

                    csv_bytes = df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download CSV",
                        data=csv_bytes,
                        file_name="proposals.csv",
                        mime="text/csv",
                    )
                elif missing2:
                    st.error(f"None of the {len(ids2)} IDs were found in SOLR (check VPN).")


if __name__ == "__main__":
    main()
