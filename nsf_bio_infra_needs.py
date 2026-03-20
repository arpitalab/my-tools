"""
nsf_bio_infra_needs.py — Infrastructure gap analysis for NSF BIO proposals.

Parses project summaries to identify where the BIO community is signalling
that it needs shared infrastructure — databases, reference resources, standards,
coordination mechanisms — that individual labs cannot provide alone.

This is NOT breakthrough prediction. It identifies coordination failures:
places where many proposals are working around a missing shared resource
(PDB, GenBank, LIGO-style investments that unlock a field).

Signals scored per proposal:
  data_resource      — explicit need for a database / repository / archive
  reference_resource — need for a reference genome / atlas / catalog / map
  standards          — need for standards / ontologies / controlled vocabularies
  scale_barrier      — science explicitly too large for a single lab
  coordination_need  — multi-site / consortium / community effort language
  computational_infra — shared software / pipeline / workflow infrastructure
  monitoring_network — long-term observational / field station needs

Output (output/bio/):
  bio_infra_scores.csv                       — one row per proposal, all signal scores
  bio_infra_division_heatmap.html            — division × signal heatmap (mean density)
  bio_infra_opportunities.html               — top proposals scored on abstract
  bio_infra_opportunities_enriched.html      — top proposals re-scored on full SOLR narrative
  bio_infra_trends.html                      — signal trends by year across BIO
  bio_infra_top_proposals.html               — ranked table top 50

Usage:
    python nsf_bio_infra_needs.py
    python nsf_bio_infra_needs.py --db output/bio/nsf_bio.db --out output/bio
    python nsf_bio_infra_needs.py --db output/nsf_awards.db  --out output/bio

    # Re-score top 300 proposals on full SOLR narrative (requires NSF VPN):
    python nsf_bio_infra_needs.py --enrich-top 300
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

SOLR_URL = "http://dis-checker-a01.ad.nsf.gov/solr/proposals/"
SOLR_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE    = Path(__file__).parent
_BIO_DB  = _HERE / "output" / "bio" / "nsf_bio.db"
_MAIN_DB = _HERE / "output" / "nsf_awards.db"
_OUT_DIR = _HERE / "output" / "bio"

# ---------------------------------------------------------------------------
# BIO-specific signal vocabulary
# Each entry: (pattern, weight)
# Patterns matched case-insensitively against the full abstract text.
# Weights within a signal allow phrase specificity to count more than
# generic terms.
# ---------------------------------------------------------------------------

SIGNALS: dict[str, list[tuple[str, float]]] = {

    "data_resource": [
        # Explicit database / repository / archive language
        (r"\bdatabase\b",               1.0),
        (r"\brepository\b",             1.0),
        (r"\bdata\s+resource\b",        1.5),
        (r"\bdata\s+portal\b",          1.5),
        (r"\bdata\s+archive\b",         1.5),
        (r"\bbiobank\b",                1.5),
        (r"\bspecimen\s+collection\b",  1.2),
        (r"\bculture\s+collection\b",   1.2),
        (r"\bdata\s+sharing\b",         1.0),
        (r"\bopen\s+data\b",            1.0),
        (r"\bpublicly\s+available\b",   0.8),
        (r"\bno\s+existing\s+(?:database|repository|resource)\b", 2.0),
        (r"\black\s+of\s+(?:a\s+)?(?:database|repository|resource)\b", 2.0),
    ],

    "reference_resource": [
        # Reference genomes, atlases, catalogs — the GenBank/PDB analogy
        (r"\breference\s+genome\b",     2.0),
        (r"\breference\s+sequence\b",   1.5),
        (r"\bpangenome\b",              2.0),
        (r"\bgenome\s+assembly\b",      1.2),
        (r"\bannotated\s+genome\b",     1.5),
        (r"\bcell\s+atlas\b",           2.0),
        (r"\batlas\s+of\b",             1.5),
        (r"\bcomprehensive\s+(?:map|atlas|catalog|survey)\b", 1.5),
        (r"\bcatalog\s+of\b",           1.2),
        (r"\bcensus\s+of\b",            1.5),
        (r"\binventory\s+of\b",         1.2),
        (r"\btaxonomic\s+(?:database|resource|backbone)\b", 1.5),
        (r"\bphylogenomic\s+(?:database|resource)\b", 1.5),
        (r"\btree\s+of\s+life\b",       1.5),
        (r"\bno\s+(?:reference|annotated)\s+genome\b", 2.5),
    ],

    "standards": [
        # Standardization / ontology / interoperability
        (r"\bstandardi[sz]",            1.0),
        (r"\bontolog",                  1.5),
        (r"\bcontrolled\s+vocabular",   1.5),
        (r"\bnomenclature\b",           1.0),
        (r"\bmetadata\s+standard",      1.5),
        (r"\bdata\s+standard",          1.5),
        (r"\binteroperab",              1.5),
        (r"\bFAIR\s+(?:data|principle)", 1.5),
        (r"\bcuration\b",               1.0),
        (r"\bannotation\s+(?:standard|framework|protocol)\b", 1.5),
        (r"\bincompatible\s+(?:data|format)\b", 2.0),
        (r"\bfragmented\s+(?:data|information)\b", 2.0),
        (r"\black\s+of\s+(?:standard|consistent)\b", 2.0),
    ],

    "scale_barrier": [
        # Science explicitly too large for a single lab
        (r"\btoo\s+large\s+for\b",      2.5),
        (r"\bbeyond\s+(?:the\s+)?(?:scope|capacity|reach)\s+of\s+(?:any\s+)?(?:single|individual)\b", 2.5),
        (r"\bcannot\s+be\s+(?:done|accomplished|achieved)\s+by\b", 2.0),
        (r"\bno\s+single\s+(?:lab|laboratory|group|investigator|institution)\b", 2.5),
        (r"\bindividual\s+(?:lab|laboratory|investigator|group)\s+(?:cannot|can.t|lacks?)\b", 2.5),
        (r"\brequires?\s+(?:a\s+)?(?:large.scale|coordinated|community.wide|concerted)\b", 2.0),
        (r"\blarge.scale\s+(?:effort|initiative|project|collaboration)\b", 1.5),
        (r"\bscale\s+(?:that|which)\s+(?:is|are)\s+(?:not|beyond|impossible)\b", 2.0),
        (r"\bprohibitively\s+(?:expensive|large|costly)\b", 2.0),
        (r"\bdata\s+(?:volumes?|scale)\s+(?:that\s+)?(?:exceed|overwhelm|surpass)\b", 1.5),
    ],

    "coordination_need": [
        # Multi-site / consortium / community coordination failures
        (r"\bconsortium\b",             1.2),
        (r"\bcoordinated\s+effort\b",   1.5),
        (r"\bcommunity\s+(?:resource|effort|initiative|standard|database)\b", 1.5),
        (r"\bmulti.site\b",             1.2),
        (r"\bmulti.institutional\b",    1.2),
        (r"\bdistributed\s+(?:data|collection|network)\b", 1.2),
        (r"\bno\s+central\s+(?:database|repository|resource)\b", 2.5),
        (r"\bfragmented\s+(?:across|among|between)\b", 2.0),
        (r"\bduplication\s+of\s+effort\b", 2.0),
        (r"\bparallel\s+(?:effort|development|work)\b", 1.5),
        (r"\bcoordination\s+(?:failure|gap|challenge|barrier)\b", 2.5),
        (r"\beach\s+(?:lab|group|investigator)\s+(?:independently|separately)\b", 2.0),
        (r"\bsiloed?\b",                2.0),
    ],

    "computational_infra": [
        # Shared computational tools / pipelines / workflows
        (r"\bbioinformatics\s+(?:tool|pipeline|workflow|resource|platform)\b", 1.5),
        (r"\bopen.source\s+(?:tool|software|pipeline|platform)\b", 1.5),
        (r"\banalysis\s+(?:pipeline|workflow|platform|framework)\b", 1.2),
        (r"\bsoftware\s+(?:tool|package|suite|platform)\b", 1.0),
        (r"\bno\s+(?:existing|available|current)\s+(?:tool|software|pipeline)\b", 2.5),
        (r"\black\s+of\s+(?:tools?|software|computational\s+methods?)\b", 2.0),
        (r"\bcomputational\s+(?:resource|infrastructure|platform|bottleneck)\b", 1.5),
        (r"\bworkflow\s+(?:management|automation|standardization)\b", 1.2),
        (r"\bscalable\s+(?:algorithm|method|tool|pipeline)\b", 1.2),
        (r"\bcloud\s+(?:computing|infrastructure|platform)\b", 1.0),
    ],

    "monitoring_network": [
        # Long-term observational / ecological / field infrastructure
        (r"\blong.term\s+(?:monitoring|dataset|data|study|observation)\b", 2.0),
        (r"\bmonitoring\s+network\b",   2.0),
        (r"\bfield\s+station\b",        1.5),
        (r"\bsensor\s+network\b",       1.5),
        (r"\blongitudinal\s+(?:data|study|dataset|monitoring)\b", 1.5),
        (r"\btime.series\s+(?:data|dataset|monitoring)\b", 1.5),
        (r"\bobservation\s+network\b",  1.5),
        (r"\bsentinel\s+(?:site|network|station)\b", 1.5),
        (r"\becological\s+(?:observatory|monitoring|network)\b", 2.0),
        (r"\bbiodiversity\s+(?:monitoring|survey|data|database)\b", 1.5),
        (r"\bgap\s+in\s+(?:long.term|temporal|monitoring)\b", 2.5),
        (r"\bno\s+(?:existing|sustained|continuous)\s+monitoring\b", 2.5),
    ],
}

# Composite weights
COMPOSITE_WEIGHTS = {
    "data_resource":      1.5,
    "reference_resource": 2.0,
    "standards":          1.2,
    "scale_barrier":      2.5,
    "coordination_need":  2.5,
    "computational_infra": 1.0,
    "monitoring_network": 1.5,
}

# Precompile all patterns
_COMPILED: dict[str, list[tuple[re.Pattern, float]]] = {
    sig: [(re.compile(pat, re.IGNORECASE), w) for pat, w in entries]
    for sig, entries in SIGNALS.items()
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_text(text: str) -> dict[str, float]:
    """Score one abstract on all signal dimensions. Returns density per 1000 words."""
    if not text or not text.strip():
        return {sig: 0.0 for sig in SIGNALS}
    words = max(len(text.split()), 1)
    scores = {}
    for sig, patterns in _COMPILED.items():
        raw = sum(w for pat, w in patterns for _ in pat.finditer(text))
        scores[sig] = round(raw / words * 1000, 4)
    return scores


def composite(row: pd.Series) -> float:
    return sum(COMPOSITE_WEIGHTS[s] * row.get(s, 0.0) for s in SIGNALS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_COLLAB_RE = re.compile(r'(?i)^collaborative\s+research\s*:\s*')


def _collab_key(title: str) -> str:
    """Normalize title for collaborative proposal deduplication."""
    return _COLLAB_RE.sub('', title or '').strip().lower()


def _dedup_collaborative(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse collaborative proposal sets (same title, different PI/institution).

    NSF collaborative proposals share a title of the form
    'Collaborative Research: <actual title>' across 2–10 partner sites.
    Keep the partner with the longest abstract text (most complete),
    record the group size in 'n_collab_parts'.
    """
    df = df.copy()
    df["_title_key"] = df["title"].apply(_collab_key)
    df["_text_len"]  = df["text"].str.len().fillna(0)

    # Within each title group, keep the row with the most text
    idx_keep = (df.groupby("_title_key")["_text_len"]
                  .idxmax()
                  .values)
    group_sizes = df.groupby("_title_key")["_title_key"].transform("count")
    df["n_collab_parts"] = group_sizes

    before = len(df)
    df = df.loc[idx_keep].drop(columns=["_title_key", "_text_len"]).reset_index(drop=True)
    collapsed = before - len(df)
    if collapsed:
        print(f"Deduplicated {collapsed:,} collaborative proposal copies "
              f"({before:,} → {len(df):,} unique projects)")
    return df


def load_proposals(db_path: Path) -> pd.DataFrame:
    """Load proposals from nsf_bio.db or nsf_awards.db (BIO only)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # nsf_bio.db schema uses abstract_narration + por_text (mapped from summary/description)
    # nsf_awards.db uses abstract_narration + por_text and has a directorate join
    try:
        # Try nsf_bio.db schema first (has status, division, pi_name directly)
        df = pd.read_sql_query(
            """SELECT award_id AS id,
                      title,
                      COALESCE(abstract_narration, por_text, title) AS text,
                      status,
                      pi_name,
                      inst,
                      received_year AS year,
                      division
               FROM award
               WHERE COALESCE(abstract_narration, por_text) IS NOT NULL""",
            conn,
        )
        df["source"] = "bio_db"
    except Exception:
        # Fall back to nsf_awards.db — join to get directorate
        df = pd.read_sql_query(
            """SELECT a.award_id AS id,
                      a.title,
                      COALESCE(a.abstract_narration, a.por_text, a.title) AS text,
                      a.source_year AS year,
                      i.name AS inst,
                      d.abbreviation AS directorate,
                      v.abbreviation AS division
               FROM award a
               LEFT JOIN directorate d ON d.id = a.directorate_id
               LEFT JOIN division    v ON v.id = a.division_id
               LEFT JOIN institution i ON i.id = a.institution_id
               WHERE d.abbreviation = 'BIO'
                 AND COALESCE(a.abstract_narration, a.por_text) IS NOT NULL""",
            conn,
        )
        df["source"] = "awards_db"

    conn.close()
    print(f"Loaded {len(df):,} proposals from {db_path.name}")
    df = _dedup_collaborative(df)
    return df


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(db_path: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_proposals(db_path)

    # Score all proposals
    print("Scoring proposals …")
    signal_cols = list(SIGNALS.keys())
    scores = [score_text(t) for t in tqdm(df["text"], unit="proposal")]
    scores_df = pd.DataFrame(scores)
    df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    df["composite"] = df.apply(composite, axis=1)

    # Save full scores
    csv_path = out_dir / "bio_infra_scores.csv"
    df.drop(columns=["text"]).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    return df


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def viz_division_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Mean signal density by BIO division."""
    signal_cols = list(SIGNALS.keys())
    div_col = "division" if "division" in df.columns else "directorate"

    agg = (df.groupby(div_col)[signal_cols]
             .mean()
             .round(4)
             .reset_index()
             .rename(columns={div_col: "division"}))
    agg = agg[agg["division"].notna() & (agg["division"] != "")]

    # Sort divisions by composite (sum of weighted means)
    agg["_composite"] = agg[signal_cols].apply(
        lambda r: sum(COMPOSITE_WEIGHTS[s] * r[s] for s in signal_cols), axis=1
    )
    agg = agg.sort_values("_composite", ascending=False).drop(columns=["_composite"])

    pretty = {
        "data_resource":      "Data / Repository",
        "reference_resource": "Reference Resource",
        "standards":          "Standards / Ontology",
        "scale_barrier":      "Scale Barrier",
        "coordination_need":  "Coordination Gap",
        "computational_infra":"Computational Tools",
        "monitoring_network": "Monitoring Network",
    }
    z      = agg[signal_cols].values
    y_labs = agg["division"].tolist()
    x_labs = [pretty[s] for s in signal_cols]

    fig = go.Figure(go.Heatmap(
        z=z, x=x_labs, y=y_labs,
        colorscale="YlOrRd",
        hoverongaps=False,
        hovertemplate="%{y}<br>%{x}: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Infrastructure Gap Signals by BIO Division (mean density per 1000 words)",
        height=max(400, 50 + 30 * len(y_labs)),
        xaxis_tickangle=-30,
        margin=dict(l=180, r=40, t=80, b=120),
    )
    path = out_dir / "bio_infra_division_heatmap.html"
    fig.write_html(str(path))
    print(f"Saved {path}")


def enrich_from_solr(df: pd.DataFrame, top_n: int, batch: int = 50) -> pd.DataFrame:
    """Re-score the top_n proposals (by composite) using full SOLR description text.

    Fetches the 'description' field (full project narrative) from SOLR in batches,
    replaces the abstract-based scores with full-text scores, and returns the
    enriched subset. Requires NSF VPN.
    """
    try:
        import pysolr
    except ImportError:
        print("pysolr not installed — skipping SOLR enrichment (pip install pysolr)")
        return df.head(0)

    top = df.nlargest(top_n, "composite").copy()
    ids = top["id"].astype(str).tolist()

    print(f"\nFetching full descriptions from SOLR for {len(ids)} proposals …")
    solr    = pysolr.Solr(SOLR_URL, timeout=SOLR_TIMEOUT)
    fetched: dict[str, str] = {}

    for i in tqdm(range(0, len(ids), batch), unit="batch"):
        chunk = ids[i:i + batch]
        query = "id:(" + " OR ".join(chunk) + ")"
        try:
            results = solr.search(query, **{"fl": "id,description", "rows": len(chunk)})
            for doc in results:
                desc = doc.get("description", "")
                if isinstance(desc, list):
                    desc = " ".join(desc)
                if desc:
                    fetched[str(doc["id"])] = desc
        except Exception as e:
            print(f"\nSOLR error (check VPN): {e}")
            break
        time.sleep(0.5)

    if not fetched:
        print("No descriptions retrieved — enriched scatter not generated.")
        return top.head(0)

    print(f"Retrieved full descriptions for {len(fetched):,} of {len(ids)} proposals.")

    signal_cols = list(SIGNALS.keys())
    enriched_rows = []
    for _, row in top.iterrows():
        pid  = str(row["id"])
        text = fetched.get(pid)
        if not text:
            continue
        new_scores = score_text(text)
        r = row.copy()
        for sig in signal_cols:
            r[sig] = new_scores[sig]
        r["composite"] = composite(r)
        r["_text_source"] = "description"
        enriched_rows.append(r)

    if not enriched_rows:
        return top.head(0)

    enriched = pd.DataFrame(enriched_rows)
    print(f"Re-scored {len(enriched):,} proposals on full narrative text.")
    return enriched


def viz_opportunities(df: pd.DataFrame, out_dir: Path, top_n: int = 200,
                      enriched: pd.DataFrame | None = None) -> None:
    """Top proposals by composite score — scatter of coordination vs reference resource."""
    top = df.nlargest(top_n, "composite").copy()
    top["year"] = top["year"].fillna(0).astype(int).astype(str)
    div_col = "division" if "division" in top.columns else "directorate"
    top[div_col] = top[div_col].fillna("Unknown")

    fig = px.scatter(
        top,
        x="coordination_need",
        y="reference_resource",
        size="composite",
        color=div_col,
        hover_name="title",
        hover_data={
            "id": True,
            "year": True,
            "inst": True,
            "composite": ":.3f",
            "data_resource": ":.3f",
            "scale_barrier": ":.3f",
            "standards": ":.3f",
        },
        size_max=30,
        title=(
            f"Top {top_n} BIO Proposals by Infrastructure Need<br>"
            "<sup>x = coordination gap signal | y = reference resource signal | "
            "size = composite score</sup>"
        ),
        labels={
            "coordination_need":  "Coordination Gap (density/1000 words)",
            "reference_resource": "Reference Resource Need (density/1000 words)",
        },
    )
    fig.update_layout(height=700)

    path = out_dir / "bio_infra_opportunities.html"
    fig.write_html(str(path))
    print(f"Saved {path}")

    # --- Enriched scatter (full SOLR description text) ---
    if enriched is not None and len(enriched) > 0:
        enr = enriched.copy()
        enr["year"] = enr["year"].fillna(0).astype(int).astype(str)
        enr[div_col] = enr[div_col].fillna("Unknown") if div_col in enr.columns else "Unknown"

        fig2 = px.scatter(
            enr,
            x="coordination_need",
            y="reference_resource",
            size="composite",
            color=div_col,
            hover_name="title",
            hover_data={
                "id": True,
                "year": True,
                "inst": True,
                "composite": ":.3f",
                "data_resource": ":.3f",
                "scale_barrier": ":.3f",
                "standards": ":.3f",
            },
            size_max=30,
            title=(
                f"Top {len(enr)} BIO Proposals — Re-scored on Full Narrative (SOLR description)<br>"
                "<sup>x = coordination gap signal | y = reference resource signal | "
                "size = composite score</sup>"
            ),
            labels={
                "coordination_need":  "Coordination Gap (density/1000 words)",
                "reference_resource": "Reference Resource Need (density/1000 words)",
            },
        )
        fig2.update_layout(height=700)
        path2 = out_dir / "bio_infra_opportunities_enriched.html"
        fig2.write_html(str(path2))
        print(f"Saved {path2}")


def viz_trends(df: pd.DataFrame, out_dir: Path) -> None:
    """Mean signal density by year — shows whether infrastructure signalling is growing."""
    signal_cols = list(SIGNALS.keys())
    df_yr = df[df["year"].notna() & (df["year"] > 2005)].copy()
    df_yr["year"] = df_yr["year"].astype(int)

    agg = df_yr.groupby("year")[signal_cols].mean().reset_index()

    pretty = {
        "data_resource":      "Data / Repository",
        "reference_resource": "Reference Resource",
        "standards":          "Standards / Ontology",
        "scale_barrier":      "Scale Barrier",
        "coordination_need":  "Coordination Gap",
        "computational_infra":"Computational Tools",
        "monitoring_network": "Monitoring Network",
    }

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, sig in enumerate(signal_cols):
        fig.add_trace(go.Scatter(
            x=agg["year"], y=agg[sig],
            name=pretty[sig],
            mode="lines+markers",
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.update_layout(
        title="Infrastructure Gap Signals Over Time — BIO Proposals",
        xaxis_title="Year",
        yaxis_title="Mean density per 1000 words",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    path = out_dir / "bio_infra_trends.html"
    fig.write_html(str(path))
    print(f"Saved {path}")


def viz_top_table(df: pd.DataFrame, out_dir: Path, top_n: int = 50) -> None:
    """Interactive ranked table of top proposals by composite score."""
    signal_cols = list(SIGNALS.keys())
    div_col = "division" if "division" in df.columns else "directorate"
    cols = ["id", "year", div_col, "title", "inst", "composite"] + signal_cols
    if "status" in df.columns:
        cols.insert(4, "status")

    top = df.nlargest(top_n, "composite")[cols].copy()
    top = top.round({s: 3 for s in signal_cols + ["composite"]})

    # Flag the dominant signal for each proposal
    top["top_signal"] = top[signal_cols].idxmax(axis=1)

    header_vals = [c.replace("_", "<br>") for c in top.columns]
    cell_vals   = [top[c].tolist() for c in top.columns]

    fig = go.Figure(go.Table(
        header=dict(
            values=header_vals,
            fill_color="#1565C0",
            font=dict(color="white", size=11),
            align="left",
        ),
        cells=dict(
            values=cell_vals,
            fill_color=[["#f9f9f9", "#ffffff"] * (len(top) // 2 + 1)]
                       * len(top.columns),
            font=dict(size=10),
            align="left",
            height=22,
        ),
    ))
    fig.update_layout(
        title=f"Top {top_n} BIO Proposals by Infrastructure Need Composite Score",
        height=900,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    path = out_dir / "bio_infra_top_proposals.html"
    fig.write_html(str(path))
    print(f"Saved {path}")


def print_summary(df: pd.DataFrame) -> None:
    signal_cols = list(SIGNALS.keys())
    div_col = "division" if "division" in df.columns else "directorate"
    print("\n" + "="*60)
    print("BIO INFRASTRUCTURE GAP SUMMARY")
    print("="*60)

    print(f"\nProposals scored: {len(df):,}")
    if "year" in df.columns:
        yr = df["year"].dropna().astype(int)
        print(f"Year range:       {yr.min()} – {yr.max()}")

    print("\nMean signal density (per 1000 words) across all proposals:")
    for sig in signal_cols:
        mean = df[sig].mean()
        top5 = df.nlargest(5, sig)[["id", "title", sig]].values
        print(f"\n  {sig:<22} mean={mean:.4f}")

    print("\nTop 10 proposals by composite infrastructure need:")
    top10 = df.nlargest(10, "composite")[["id", "title", div_col, "composite"]]
    for _, r in top10.iterrows():
        print(f"  {r['id']:<12} {r.get(div_col,'?'):<8} {r['composite']:.3f}  {str(r['title'])[:60]}")

    if div_col in df.columns:
        print(f"\nTop divisions by mean composite score:")
        div_scores = (df.groupby(div_col)["composite"]
                       .agg(["mean", "count"])
                       .sort_values("mean", ascending=False)
                       .head(10))
        for div, row in div_scores.iterrows():
            if div:
                print(f"  {str(div):<12}  mean={row['mean']:.3f}  n={int(row['count']):,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Score BIO proposals for infrastructure gap signals"
    )
    p.add_argument("--db",  default=None,
                   help=f"SQLite DB path (default: {_BIO_DB} if exists, else {_MAIN_DB})")
    p.add_argument("--out", default=str(_OUT_DIR),
                   help=f"Output directory (default: {_OUT_DIR})")
    p.add_argument("--top-n", type=int, default=200,
                   help="Number of top proposals for opportunity scatter (default 200)")
    p.add_argument("--enrich-top", type=int, default=0, metavar="N",
                   help="Re-score top N proposals on full SOLR description text "
                        "and write an enriched scatter (requires NSF VPN, default: off)")
    return p.parse_args()


def main():
    args  = parse_args()
    out   = Path(args.out)

    if args.db:
        db = Path(args.db)
    elif _BIO_DB.exists():
        db = _BIO_DB
    else:
        db = _MAIN_DB
        print(f"nsf_bio.db not found — using {_MAIN_DB} (awarded BIO only)")

    df = run(db, out)
    print_summary(df)

    enriched = None
    if args.enrich_top > 0:
        enriched = enrich_from_solr(df, top_n=args.enrich_top)

    print("\nGenerating visualizations …")
    viz_division_heatmap(df, out)
    viz_opportunities(df, out, top_n=args.top_n, enriched=enriched)
    viz_trends(df, out)
    viz_top_table(df, out)

    print(f"\nAll outputs in {out}/")


if __name__ == "__main__":
    main()
