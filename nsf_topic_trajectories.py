"""
nsf_topic_trajectories.py — Phase 2: Cluster trajectory tracking + visualization.

Reads Phase 1 outputs and computes:
  • Year-over-year cluster transition matrices (overlap tracking)
  • Transition type classification (persistent / split / merge / de_novo / disintegrate)
  • Additional features: citation_acceleration, community_invasion, award_gini,
    inst_gini, growth_sustainability, method_diversity, pe_code_age
  • Enriched cluster_year_stats with all features

Visualizations (all interactive Plotly HTML):
  1. bubble_trajectories.html  — sized circles per cluster per year, heat-mapped by
                                 pct_new_pis (inspired by NIHOPA paper)
  2. feature_dashboard.html    — time-series panel for top-N clusters
  3. breakthrough_scatter.html — pct_new_pis vs cross_dir_entropy, sized by
                                 acceleration, coloured by growth_sustainability
  4. transition_sankey.html    — Sankey flow of cluster membership across years
  5. cluster_heatmap.html      — feature heatmap across all clusters (ranked)

Usage:
    python nsf_topic_trajectories.py
    python nsf_topic_trajectories.py --top-n 30 --output-dir ./output
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE   = Path(__file__).parent
_OUTPUT = _HERE / "output"
_DB     = _OUTPUT / "nsf_awards.db"

ASSIGN_PATH = _OUTPUT / "cluster_assignments.parquet"
STATS_PATH  = _OUTPUT / "cluster_year_stats.parquet"
ENRICH_OUT  = _OUTPUT / "cluster_year_features.parquet"
TRANS_OUT   = _OUTPUT / "cluster_transitions.parquet"


# ---------------------------------------------------------------------------
# Gini helper (scipy may not be available everywhere)
# ---------------------------------------------------------------------------

def _gini(values: np.ndarray) -> float:
    """Gini coefficient — 0=equal, 1=maximally concentrated."""
    v = np.sort(np.abs(values[~np.isnan(values)]))
    if len(v) == 0 or v.sum() == 0:
        return 0.0
    n   = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


# ---------------------------------------------------------------------------
# 1. Year-over-year transition matrix
# ---------------------------------------------------------------------------

def compute_transitions(assign: pd.DataFrame, db_path: Path) -> pd.DataFrame:
    """
    Track cluster continuity via PI-level flow.

    NSF awards have a single source_year, so the same award never appears in two
    consecutive years.  Instead we track PIs: if a PI has awards in cluster A in
    year t and cluster B in year t+1, that is a PI-flow transition A→B.  The
    fraction of cluster A's PIs who move to cluster B gives the transition weight.

    Returns a DataFrame with columns:
        year_t, cluster_t, year_t1, cluster_t1,
        n_shared, frac_of_t, frac_of_t1
    """
    print("Computing PI-level cluster transition matrices …")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    pi_rows = conn.execute(
        """SELECT a.award_id, ai.full_name AS pi
           FROM award_investigator ai
           JOIN award a ON a.id = ai.award_id
           WHERE ai.role_code = 'Principal Investigator'"""
    ).fetchall()
    conn.close()

    pi_df = pd.DataFrame(pi_rows, columns=["award_id", "pi"])
    pi_df["award_id"] = pi_df["award_id"].astype(str)

    data = (assign[assign["cluster_id"] != -1]
            .merge(pi_df, on="award_id", how="inner")
            [["pi", "cluster_id", "year"]]
            .drop_duplicates())

    years   = sorted(data["year"].unique())
    records = []

    for i, yr in enumerate(tqdm(years[:-1], desc="Transitions")):
        yr1   = years[i + 1]
        t_df  = (data[data["year"] == yr]
                 [["pi", "cluster_id"]]
                 .rename(columns={"cluster_id": "cluster_t"}))
        t1_df = (data[data["year"] == yr1]
                 [["pi", "cluster_id"]]
                 .rename(columns={"cluster_id": "cluster_t1"}))

        merged = t_df.merge(t1_df, on="pi", how="inner")
        if merged.empty:
            continue

        size_t  = t_df.groupby("cluster_t").size().rename("size_t")
        size_t1 = t1_df.groupby("cluster_t1").size().rename("size_t1")

        counts = (merged.groupby(["cluster_t", "cluster_t1"])
                        .size()
                        .reset_index(name="n_shared"))
        counts["year_t"]  = yr
        counts["year_t1"] = yr1
        counts = counts.merge(size_t,  on="cluster_t")
        counts = counts.merge(size_t1, on="cluster_t1")
        counts["frac_of_t"]  = counts["n_shared"] / counts["size_t"]
        counts["frac_of_t1"] = counts["n_shared"] / counts["size_t1"]
        records.append(counts)

    if not records:
        print("  WARNING: no PI transitions found")
        return pd.DataFrame(columns=["year_t","cluster_t","year_t1","cluster_t1",
                                     "n_shared","frac_of_t","frac_of_t1",
                                     "size_t","size_t1"])

    trans = pd.concat(records, ignore_index=True)
    print(f"  {len(trans):,} PI-flow transitions across {len(years)-1} year gaps")
    return trans


# ---------------------------------------------------------------------------
# 2. Classify transition types
# ---------------------------------------------------------------------------

def classify_transitions(trans: pd.DataFrame) -> pd.DataFrame:
    """
    For each (cluster_t, year_t), classify how it evolves into year t+1:

        persistent   — one dominant successor (frac_of_t > 0.5, frac_of_t1 > 0.5)
        captured     — absorbed into a larger cluster (frac_of_t > 0.5, frac_of_t1 ≤ 0.5)
        split        — breaks into multiple clusters (max frac_of_t ≤ 0.5)
        disintegrate — < 20% of members continue into any single cluster
        de_novo      — clusters in t+1 that have no strong predecessor
    """
    print("Classifying transition types …")
    records = []

    for (cid_t, yr_t), grp in trans.groupby(["cluster_t","year_t"]):
        top = grp.sort_values("frac_of_t", ascending=False).iloc[0]
        max_frac_t  = top["frac_of_t"]
        max_frac_t1 = top["frac_of_t1"]

        if max_frac_t < 0.20:
            transition = "disintegrate"
        elif max_frac_t >= 0.5 and max_frac_t1 >= 0.5:
            transition = "persistent"
        elif max_frac_t >= 0.5 and max_frac_t1 < 0.5:
            transition = "captured"
        else:
            transition = "split"

        records.append({
            "cluster_id":       cid_t,
            "year":             yr_t,
            "transition_fwd":   transition,
            "top_successor":    top["cluster_t1"],
            "top_frac_of_t":    round(max_frac_t, 3),
            "top_frac_of_t1":   round(max_frac_t1, 3),
            "n_successors":     len(grp[grp["frac_of_t"] > 0.05]),
        })

    # De novo: clusters in t+1 with no strong predecessor
    for (cid_t1, yr_t1), grp in trans.groupby(["cluster_t1","year_t1"]):
        if grp["frac_of_t1"].max() < 0.3:
            records.append({
                "cluster_id":     cid_t1,
                "year":           yr_t1,
                "transition_fwd": "de_novo",
                "top_successor":  None,
                "top_frac_of_t":  0.0,
                "top_frac_of_t1": grp["frac_of_t1"].max(),
                "n_successors":   0,
            })

    if not records:
        print("  No transitions to classify.")
        return pd.DataFrame(columns=["cluster_id","year","transition_fwd",
                                     "top_successor","top_frac_of_t",
                                     "top_frac_of_t1","n_successors"])

    df = pd.DataFrame(records).drop_duplicates(["cluster_id","year","transition_fwd"])
    # Keep one row per (cluster, year) — prefer the non-de_novo classification
    order = {"persistent":0,"captured":1,"split":2,"disintegrate":3,"de_novo":4}
    df["_ord"] = df["transition_fwd"].map(order).fillna(9)
    df = df.sort_values("_ord").drop_duplicates(["cluster_id","year"]).drop(columns="_ord")
    print(f"  Transition type counts:\n{df['transition_fwd'].value_counts().to_string()}")
    return df


# ---------------------------------------------------------------------------
# 3. Additional features
# ---------------------------------------------------------------------------

def compute_additional_features(
    stats: pd.DataFrame,
    assign: pd.DataFrame,
    db_path: Path,
) -> pd.DataFrame:
    """
    Adds to cluster_year_stats:
        citation_acceleration   — second derivative of growth_rate
        community_invasion      — sudden increase in cross_dir_entropy
        award_gini              — funding concentration across PIs
        inst_gini               — funding concentration across institutions
        growth_sustainability   — fraction of next 3 years with positive growth
        method_diversity        — lexical diversity of abstracts in cluster-year
        pe_code_age             — avg age of program element codes in cluster
    """
    print("Computing additional features …")
    stats = stats.sort_values(["cluster_id","year"]).copy()

    # ── citation_acceleration (Δgrowth_rate) ──────────────────────────────
    stats["growth_rate_lag"] = stats.groupby("cluster_id")["growth_rate"].shift(1)
    stats["citation_acceleration"] = stats["growth_rate"] - stats["growth_rate_lag"]
    stats.drop(columns=["growth_rate_lag"], inplace=True)

    # ── community_invasion (Δcross_dir_entropy) ───────────────────────────
    stats["entropy_lag"] = stats.groupby("cluster_id")["cross_dir_entropy"].shift(1)
    stats["community_invasion"] = stats["cross_dir_entropy"] - stats["entropy_lag"]
    stats.drop(columns=["entropy_lag"], inplace=True)

    # ── growth_sustainability: fraction of next 3 years with growth > 0 ───
    def _sustainability(s):
        arr = s.values
        result = np.full(len(arr), np.nan)
        for i in range(len(arr) - 1):
            window = arr[i+1 : i+4]
            valid  = window[~np.isnan(window)]
            result[i] = (valid > 0).mean() if len(valid) > 0 else np.nan
        return pd.Series(result, index=s.index)

    stats["growth_sustainability"] = (
        stats.groupby("cluster_id")["growth_rate"]
             .transform(_sustainability)
    )

    # ── award_gini and inst_gini from assignments ─────────────────────────
    print("  Computing Gini coefficients …")
    assign_nn = assign[assign["cluster_id"] != -1].copy()

    gini_records = []
    for (cid, yr), grp in assign_nn.groupby(["cluster_id","year"]):
        amt = grp["award_amount"].dropna().values
        award_gini = _gini(amt) if len(amt) > 1 else np.nan

        inst_counts = grp["division"].value_counts().values.astype(float)
        inst_gini   = _gini(inst_counts) if len(inst_counts) > 1 else np.nan

        gini_records.append({
            "cluster_id": cid, "year": yr,
            "award_gini": round(award_gini, 4),
            "inst_gini":  round(inst_gini, 4),
        })

    gini_df = pd.DataFrame(gini_records)
    stats   = stats.merge(gini_df, on=["cluster_id","year"], how="left")

    # ── method_diversity: type-token ratio of abstracts per cluster-year ──
    print("  Computing method diversity (TTR) …")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT award_id, COALESCE(abstract_narration, por_text, title) AS text "
        "FROM award WHERE COALESCE(abstract_narration, por_text, title) IS NOT NULL"
    ).fetchall()
    conn.close()

    text_map = {str(r[0]): r[1] for r in rows}
    assign_nn["text"] = assign_nn["award_id"].map(text_map)

    def _ttr(texts):
        """Type-token ratio: unique words / total words — proxy for lexical diversity."""
        combined = " ".join(t.lower() for t in texts if isinstance(t, str))
        tokens   = combined.split()
        if not tokens:
            return np.nan
        return len(set(tokens)) / len(tokens)

    ttr_records = []
    for (cid, yr), grp in assign_nn.groupby(["cluster_id","year"]):
        ttr_records.append({
            "cluster_id":     cid,
            "year":           yr,
            "method_diversity": round(_ttr(grp["text"].tolist()), 4),
        })

    ttr_df = pd.DataFrame(ttr_records)
    stats  = stats.merge(ttr_df, on=["cluster_id","year"], how="left")

    # ── pe_code_age: how established is the program element? ──────────────
    print("  Computing PE code age …")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    pe_rows = conn.execute(
        """
        SELECT a.award_id, a.source_year, pe.code
        FROM   award a
        JOIN   award_program_element ape ON ape.award_id = a.id
        JOIN   program_element pe        ON pe.id = ape.program_element_id
        """
    ).fetchall()
    conn.close()

    pe_df = pd.DataFrame(pe_rows, columns=["award_id","award_year","pe_code"])
    pe_df["award_id"] = pe_df["award_id"].astype(str)

    # First year each PE code appears
    pe_first = pe_df.groupby("pe_code")["award_year"].min().rename("pe_first_year")
    pe_df    = pe_df.merge(pe_first, on="pe_code")
    pe_df["pe_age"] = pe_df["award_year"] - pe_df["pe_first_year"]

    # Join to assignments
    pe_assign = assign_nn[["award_id","cluster_id","year"]].merge(
        pe_df[["award_id","pe_age"]], on="award_id", how="left"
    )
    pe_age_df = (pe_assign.groupby(["cluster_id","year"])["pe_age"]
                           .mean()
                           .reset_index()
                           .rename(columns={"pe_age":"pe_code_age"}))
    pe_age_df["pe_code_age"] = pe_age_df["pe_code_age"].round(2)

    stats = stats.merge(pe_age_df, on=["cluster_id","year"], how="left")

    # ── Infrastructure-need features ──────────────────────────────────────
    print("  Computing infrastructure-need features …")

    # Vocabulary sets (compiled from PDB/GenBank/LIGO analogy)
    _INFRA_TERMS = {
        "database","repository","repositories","archive","archives",
        "shared","sharing","interoperability","interoperable","standards",
        "standard","ontology","ontologies","metadata","platform",
        "community resource","open access","open data","public data",
        "data sharing","data standard","data infrastructure","cyberinfrastructure",
        "portal","registry","catalog","catalogue","annotation","curation",
    }
    _SCALE_TERMS = {
        "limited by","lack of data","no available","insufficient data",
        "requires large","large-scale","at scale","systematic survey",
        "comprehensive survey","scarce","rarely available","not yet available",
        "undersampled","under-sampled","difficult to obtain","hard to obtain",
        "prohibitively","expensive to","cost-prohibitive",
    }
    _THEORY_TERMS = {
        "theoretical","theory","model","predict","prediction","equation",
        "derive","derived","proof","theorem","computational model",
        "simulation","simulations","framework","formalism","conjecture",
    }
    _EXPT_TERMS = {
        "measured","observed","detected","fabricated","synthesized",
        "experimental","experiment","assay","specimen","sample","trial",
        "clinical","in vivo","in vitro","cultured","sequenced","imaged",
    }

    def _vocab_score(text: str, terms: set) -> float:
        """Fraction of multi-word terms found + unigram hit rate."""
        tl = text.lower()
        hits = sum(1 for t in terms if t in tl)
        return hits / max(len(terms), 1)

    infra_records = []
    for (cid, yr), grp in assign_nn.groupby(["cluster_id","year"]):
        texts = [text_map.get(aid, "") for aid in grp["award_id"]]
        texts = [t for t in texts if t]
        if not texts:
            infra_records.append({
                "cluster_id": cid, "year": yr,
                "infra_vocab":       np.nan,
                "scale_mismatch":    np.nan,
                "theory_expt_ratio": np.nan,
                "cross_lab_dup":     np.nan,
                "mri_fraction":      np.nan,
            })
            continue

        combined = " ".join(t.lower() for t in texts)

        infra_score  = _vocab_score(combined, _INFRA_TERMS)
        scale_score  = _vocab_score(combined, _SCALE_TERMS)
        theory_score = _vocab_score(combined, _THEORY_TERMS)
        expt_score   = _vocab_score(combined, _EXPT_TERMS)
        te_ratio     = theory_score / (expt_score + 1e-6)

        # Cross-lab duplication: unique institutions / award count
        n_inst  = grp["division"].nunique()
        n_awards = len(grp)
        cross_lab = n_inst / max(n_awards, 1)

        infra_records.append({
            "cluster_id":      cid,
            "year":            yr,
            "infra_vocab":     round(infra_score,  5),
            "scale_mismatch":  round(scale_score,  5),
            "theory_expt_ratio": round(te_ratio,   4),
            "cross_lab_dup":   round(cross_lab,    4),
            "mri_fraction":    np.nan,   # filled below from PE codes
        })

    infra_df = pd.DataFrame(infra_records)

    # MRI fraction: fraction of awards with MRI-related PE codes
    mri_pe = pe_df[pe_df["pe_code"].str.startswith("118") |
                   pe_df["pe_code"].str.startswith("119") |
                   pe_df["pe_code"].str.startswith("120")]["award_id"].unique()
    assign_nn["is_mri"] = assign_nn["award_id"].isin(mri_pe)
    mri_frac = (assign_nn.groupby(["cluster_id","year"])
                          .agg(mri_fraction=("is_mri","mean"))
                          .reset_index())
    infra_df = infra_df.drop(columns=["mri_fraction"]).merge(
        mri_frac, on=["cluster_id","year"], how="left"
    )

    # Composite infrastructure opportunity score (0–1 normalised later in viz)
    infra_df["infra_opportunity"] = (
        infra_df["infra_vocab"].fillna(0)     * 3.0 +
        infra_df["scale_mismatch"].fillna(0)  * 2.0 +
        infra_df["cross_lab_dup"].fillna(0)   * 1.5 +
        infra_df["mri_fraction"].fillna(0)    * 1.0
    ).round(5)

    stats = stats.merge(infra_df, on=["cluster_id","year"], how="left")

    print(f"  Features added: citation_acceleration, community_invasion, "
          f"growth_sustainability, award_gini, inst_gini, method_diversity, "
          f"pe_code_age, infra_vocab, scale_mismatch, theory_expt_ratio, "
          f"cross_lab_dup, mri_fraction, infra_opportunity")
    return stats


# ---------------------------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------------------------

def _short_label(label: str, n: int = 35) -> str:
    return label[:n] + "…" if len(label) > n else label


def viz_bubble_trajectories(
    stats: pd.DataFrame,
    trans_types: pd.DataFrame,
    output_dir: Path,
    top_n: int = 40,
) -> None:
    """
    Circle trajectory chart: x=year, y=cluster (sorted by peak size),
    bubble area ∝ award count, colour = pct_new_pis.
    Inspired by NIHOPA paper figure style.
    """
    print("  Rendering bubble trajectory chart …")

    # Pick top_n clusters by total awards
    top_clusters = (stats.groupby("cluster_id")["size"]
                         .sum()
                         .sort_values(ascending=False)
                         .head(top_n)
                         .index.tolist())

    df = stats[stats["cluster_id"].isin(top_clusters)].copy()
    if "transition_fwd" not in df.columns:
        df = df.merge(trans_types[["cluster_id","year","transition_fwd"]],
                      on=["cluster_id","year"], how="left")

    # Sort clusters by year of peak size for a readable y-axis
    peak_year = (df.groupby("cluster_id")
                   .apply(lambda g: g.loc[g["size"].idxmax(), "year"])
                   .sort_values())
    cluster_order = peak_year.index.tolist()
    df["cluster_rank"] = df["cluster_id"].map(
        {cid: i for i, cid in enumerate(cluster_order)}
    )

    # Short readable labels
    label_map = (stats[["cluster_id","top_terms"]]
                 .drop_duplicates("cluster_id")
                 .set_index("cluster_id")["top_terms"]
                 .apply(lambda x: _short_label(str(x).split("|")[0].strip())))
    df["label"] = df["cluster_id"].map(label_map)

    symbol_map = {
        "persistent":  "circle",
        "split":       "diamond",
        "captured":    "triangle-down",
        "disintegrate":"x",
        "de_novo":     "star",
        None:          "circle",
    }
    df["symbol"] = df["transition_fwd"].map(symbol_map).fillna("circle")

    fig = go.Figure()

    for transition, grp in df.groupby("transition_fwd", dropna=False):
        fig.add_trace(go.Scatter(
            x=grp["year"],
            y=grp["cluster_rank"],
            mode="markers",
            name=str(transition),
            marker=dict(
                size=np.sqrt(grp["size"]) * 2.5,
                sizemode="diameter",
                color=grp["pct_new_pis"],
                colorscale="RdYlGn",
                cmin=0, cmax=1,
                colorbar=dict(title="pct_new_pis", x=1.02),
                symbol=symbol_map.get(transition, "circle"),
                line=dict(width=0.5, color="white"),
                opacity=0.85,
            ),
            text=grp.apply(
                lambda r: (
                    f"<b>{r['label']}</b><br>"
                    f"Year: {r['year']}<br>"
                    f"Size: {r['size']} awards<br>"
                    f"pct_new_pis: {r['pct_new_pis']:.2f}<br>"
                    f"growth_rate: {r.get('growth_rate', float('nan')):.2f}<br>"
                    f"cross_dir_entropy: {r['cross_dir_entropy']:.2f}<br>"
                    f"transition: {r.get('transition_fwd','?')}"
                ), axis=1
            ),
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title="NSF Topic Cluster Trajectories 2010–2024<br>"
              "<sup>Bubble size = award count | Color = % new PIs (green=high) | "
              "Shape = transition type</sup>",
        xaxis=dict(title="Year", dtick=1, gridcolor="#eee"),
        yaxis=dict(
            title="Cluster (sorted by peak year)",
            tickvals=list(range(len(cluster_order))),
            ticktext=[label_map.get(c, str(c)) for c in cluster_order],
            tickfont=dict(size=9),
            gridcolor="#eee",
        ),
        height=max(600, top_n * 18),
        width=1200,
        legend=dict(title="Transition type", x=1.08),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    path = output_dir / "bubble_trajectories.html"
    fig.write_html(str(path))
    print(f"    → {path.name}")


def viz_feature_dashboard(
    stats: pd.DataFrame,
    output_dir: Path,
    top_n: int = 12,
) -> None:
    """
    Multi-panel time-series dashboard for the top_n clusters.
    Shows size, growth_rate, pct_new_pis, cross_dir_entropy per cluster.
    """
    print("  Rendering feature dashboard …")

    top_clusters = (stats.groupby("cluster_id")["size"]
                         .sum()
                         .sort_values(ascending=False)
                         .head(top_n)
                         .index.tolist())

    label_map = (stats[["cluster_id","top_terms"]]
                 .drop_duplicates("cluster_id")
                 .set_index("cluster_id")["top_terms"]
                 .apply(lambda x: _short_label(str(x), 30)))

    features = ["size", "growth_rate", "pct_new_pis", "cross_dir_entropy"]
    colors   = px.colors.qualitative.Plotly

    fig = make_subplots(
        rows=len(features), cols=1,
        shared_xaxes=True,
        subplot_titles=[f.replace("_"," ").title() for f in features],
        vertical_spacing=0.06,
    )

    for i, cid in enumerate(top_clusters):
        c_df  = stats[stats["cluster_id"] == cid].sort_values("year")
        color = colors[i % len(colors)]
        name  = label_map.get(cid, str(cid))
        show_legend = True

        for row, feat in enumerate(features, 1):
            fig.add_trace(
                go.Scatter(
                    x=c_df["year"], y=c_df[feat],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=1.5),
                    marker=dict(size=5),
                    legendgroup=str(cid),
                    showlegend=(row == 1),
                    hovertemplate=f"{name}<br>Year: %{{x}}<br>{feat}: %{{y:.3f}}<extra></extra>",
                ),
                row=row, col=1,
            )

    fig.update_layout(
        title=f"Feature Dashboard — Top {top_n} Clusters by Award Count",
        height=200 * len(features),
        width=1100,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(x=1.02, font=dict(size=9)),
    )
    for row in range(1, len(features) + 1):
        fig.update_xaxes(showgrid=True, gridcolor="#eee", row=row, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#eee", row=row, col=1)

    path = output_dir / "feature_dashboard.html"
    fig.write_html(str(path))
    print(f"    → {path.name}")


def viz_breakthrough_scatter(
    stats: pd.DataFrame,
    output_dir: Path,
    min_size: int = 20,
) -> None:
    """
    The key diagnostic plot:
        x = pct_new_pis       (novelty — new entrants)
        y = cross_dir_entropy  (cross-domain breadth)
        size = abs(citation_acceleration)  (phase transition intensity)
        color = growth_sustainability      (durable vs. flash-in-pan)

    Top-right quadrant = high novelty + high cross-domain = breakthrough candidates.
    High size + low sustainability = bandwagon warning.
    """
    print("  Rendering breakthrough scatter …")

    # Aggregate to cluster level (mean across years, focus on 2015+)
    recent = stats[stats["year"] >= 2015].copy()
    agg = (recent.groupby("cluster_id").agg(
        pct_new_pis        =("pct_new_pis",         "mean"),
        cross_dir_entropy  =("cross_dir_entropy",    "mean"),
        citation_accel     =("citation_acceleration","mean"),
        growth_sustain     =("growth_sustainability","mean"),
        total_awards       =("size",                 "sum"),
        growth_rate        =("growth_rate",           "mean"),
        top_terms          =("top_terms",             "first"),
        award_gini         =("award_gini",            "mean"),
    ).reset_index())

    agg = agg[agg["total_awards"] >= min_size].copy()
    agg["label"]      = agg["top_terms"].apply(lambda x: _short_label(str(x), 40))
    agg["accel_size"] = np.abs(agg["citation_accel"].fillna(0)) * 30 + 8
    agg["award_gini"] = agg["award_gini"].fillna(0.5)

    # Quadrant annotations
    q_x = agg["pct_new_pis"].median()
    q_y = agg["cross_dir_entropy"].median()

    fig = px.scatter(
        agg,
        x="pct_new_pis",
        y="cross_dir_entropy",
        size="accel_size",
        color="growth_sustain",
        color_continuous_scale="RdYlGn",
        hover_name="label",
        hover_data={
            "total_awards":     True,
            "growth_rate":      ":.2f",
            "award_gini":       ":.2f",
            "citation_accel":   ":.3f",
            "accel_size":       False,
            "pct_new_pis":      ":.2f",
            "cross_dir_entropy":":.2f",
        },
        title="Breakthrough Candidate Space (2015–2024 mean)<br>"
              "<sup>Top-right = high novelty + cross-domain | "
              "Bubble size = |acceleration| | Color = growth sustainability (green=durable)</sup>",
        labels={
            "pct_new_pis":       "% New PIs (novelty)",
            "cross_dir_entropy": "Cross-Directorate Entropy",
            "growth_sustain":    "Sustainability",
        },
        size_max=40,
    )

    # Quadrant lines + labels
    for x_val, y_val, text, ax, ay in [
        (q_x, agg["cross_dir_entropy"].max() * 0.95,
         "← Established field | Emerging field →", 0, -20),
        (agg["pct_new_pis"].max() * 0.95, q_y,
         "Cross-domain ↑", -30, 0),
    ]:
        fig.add_annotation(x=x_val, y=y_val, text=text,
                           showarrow=False, font=dict(size=10, color="gray"))

    fig.add_vline(x=q_x, line=dict(color="gray", dash="dash", width=1))
    fig.add_hrect(y0=q_y, y1=agg["cross_dir_entropy"].max() + 0.1,
                  x0=q_x, x1=agg["pct_new_pis"].max() + 0.01,
                  fillcolor="green", opacity=0.04)
    fig.add_annotation(
        x=agg["pct_new_pis"].max() * 0.9,
        y=agg["cross_dir_entropy"].max() * 0.98,
        text="⭐ Breakthrough zone",
        showarrow=False, font=dict(size=11, color="darkgreen", family="Arial Black"),
    )

    fig.update_layout(
        height=750, width=1100,
        plot_bgcolor="white", paper_bgcolor="white",
        coloraxis_colorbar=dict(title="Growth<br>Sustainability"),
    )

    path = output_dir / "breakthrough_scatter.html"
    fig.write_html(str(path))
    print(f"    → {path.name}")


def viz_transition_sankey(
    trans: pd.DataFrame,
    stats: pd.DataFrame,
    output_dir: Path,
    year_t: int = 2018,
    top_n: int = 20,
    min_frac: float = 0.1,
) -> None:
    """
    Sankey flow diagram: how cluster membership flows between two consecutive years.
    Shows which clusters are stable, merging, splitting, or newly emerging.
    """
    print(f"  Rendering Sankey flow ({year_t} → {year_t+1}) …")

    df = trans[(trans["year_t"] == year_t) & (trans["frac_of_t"] >= min_frac)].copy()

    label_map = (stats[["cluster_id","top_terms"]]
                 .drop_duplicates("cluster_id")
                 .set_index("cluster_id")["top_terms"]
                 .apply(lambda x: _short_label(str(x).split("|")[0].strip(), 30)))

    # Limit to top_n source clusters by size
    top_src = (stats[(stats["year"] == year_t)]
               .nlargest(top_n, "size")["cluster_id"].tolist())
    df = df[df["cluster_t"].isin(top_src)]

    # Build node list: left = year_t clusters, right = year_t+1 clusters
    src_ids  = sorted(df["cluster_t"].unique())
    tgt_ids  = sorted(df["cluster_t1"].unique())
    all_ids  = src_ids + [f"_{x}" for x in tgt_ids]  # prefix to avoid collision

    node_labels = (
        [f"{label_map.get(c, str(c))} ({year_t})"   for c in src_ids] +
        [f"{label_map.get(c, str(c))} ({year_t+1})" for c in tgt_ids]
    )

    src_idx = {c: i               for i, c in enumerate(src_ids)}
    tgt_idx = {c: len(src_ids)+i  for i, c in enumerate(tgt_ids)}

    sources = [src_idx[r["cluster_t"]]  for _, r in df.iterrows()]
    targets = [tgt_idx[r["cluster_t1"]] for _, r in df.iterrows()]
    values  = df["n_shared"].tolist()

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=12, thickness=18,
            label=node_labels,
            color=["#4C78A8"] * len(src_ids) + ["#F58518"] * len(tgt_ids),
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(150,150,200,0.3)",
        ),
    ))
    fig.update_layout(
        title=f"Cluster Membership Flow: {year_t} → {year_t+1}<br>"
              f"<sup>Blue = {year_t} clusters | Orange = {year_t+1} clusters | "
              f"Flow width = shared awards</sup>",
        height=700, width=1100,
    )

    path = output_dir / "transition_sankey.html"
    fig.write_html(str(path))
    print(f"    → {path.name}")


def viz_cluster_heatmap(
    stats: pd.DataFrame,
    output_dir: Path,
    top_n: int = 50,
) -> None:
    """
    Feature heatmap: clusters (rows) × features (cols), normalised 0–1.
    Rows ranked by a composite "interestingness" score.
    """
    print("  Rendering cluster feature heatmap …")

    features = [
        "pct_new_pis", "cross_dir_entropy", "growth_rate",
        "citation_acceleration", "community_invasion",
        "growth_sustainability", "method_diversity",
        "award_gini", "inst_gini",
    ]
    available = [f for f in features if f in stats.columns]

    # Aggregate: mean over 2015–2024
    agg = (stats[stats["year"] >= 2015]
           .groupby("cluster_id")[available + ["size","top_terms"]]
           .agg({**{f: "mean" for f in available}, "size": "sum", "top_terms": "first"})
           .reset_index())

    # Composite interestingness: pct_new_pis + cross_dir_entropy + |acceleration|
    agg["score"] = (
        agg.get("pct_new_pis", 0).fillna(0) +
        agg.get("cross_dir_entropy", 0).fillna(0) +
        agg.get("citation_acceleration", pd.Series(0)).abs().fillna(0) * 0.5
    )
    agg = agg.nlargest(top_n, "score")

    # Normalise each feature 0–1
    feat_matrix = agg[available].copy()
    for col in available:
        mn, mx = feat_matrix[col].min(), feat_matrix[col].max()
        feat_matrix[col] = (feat_matrix[col] - mn) / (mx - mn + 1e-9)

    labels = agg["top_terms"].apply(lambda x: _short_label(str(x).split("|")[0].strip(), 35))

    fig = go.Figure(go.Heatmap(
        z=feat_matrix.values,
        x=[f.replace("_"," ") for f in available],
        y=labels.tolist(),
        colorscale="RdYlGn",
        hoverongaps=False,
        hovertemplate="Cluster: %{y}<br>Feature: %{x}<br>Score: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Normalised<br>score"),
    ))

    fig.update_layout(
        title=f"Top {top_n} Clusters by Interestingness — Feature Heatmap<br>"
              "<sup>Normalised 0–1 per feature | Green = high | Ranked by composite score</sup>",
        height=max(500, top_n * 16),
        width=900,
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
        xaxis=dict(tickangle=-30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=260, r=80, t=100, b=80),
    )

    path = output_dir / "cluster_heatmap.html"
    fig.write_html(str(path))
    print(f"    → {path.name}")


def viz_infrastructure_opportunity(
    stats: pd.DataFrame,
    output_dir: Path,
    min_size: int = 15,
) -> None:
    """
    The infrastructure opportunity plot — the primary output for the new framing.

    Two panels:
      Left scatter:
        x = infra_vocab         (community explicitly asking for infrastructure)
        y = scale_mismatch      (science requiring scale no single lab can provide)
        size = cross_lab_dup    (fragmented effort across many labs)
        color = theory_expt_ratio (theoretically mature but exp. sparse = high)

      Right bar chart:
        Top 20 clusters ranked by composite infra_opportunity score,
        stacked by component (vocab / scale / cross_lab / mri).

    Top-right of scatter = highest infrastructure investment priority.
    """
    print("  Rendering infrastructure opportunity plot …")

    recent = stats[stats["year"] >= 2015].copy()
    agg = (recent.groupby("cluster_id").agg(
        infra_vocab       =("infra_vocab",        "mean"),
        scale_mismatch    =("scale_mismatch",      "mean"),
        theory_expt_ratio =("theory_expt_ratio",   "mean"),
        cross_lab_dup     =("cross_lab_dup",        "mean"),
        mri_fraction      =("mri_fraction",         "mean"),
        infra_opportunity =("infra_opportunity",    "mean"),
        total_awards      =("size",                 "sum"),
        top_terms         =("top_terms",            "first"),
        pct_new_pis       =("pct_new_pis",          "mean"),
        cross_dir_entropy =("cross_dir_entropy",    "mean"),
    ).reset_index())

    agg = agg[agg["total_awards"] >= min_size].copy()
    agg["label"] = agg["top_terms"].apply(
        lambda x: _short_label(str(x).split("|")[0].strip(), 45)
    )
    # Clamp theory_expt_ratio for colour scale readability
    agg["te_clamped"] = agg["theory_expt_ratio"].clip(0, 5)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.58, 0.42],
        subplot_titles=[
            "Infrastructure Need Space<br>"
            "<sup>x=community asking for infra | y=scale beyond single lab | "
            "size=cross-lab fragmentation | color=theory maturity</sup>",
            "Top 20 Infrastructure Opportunities<br>"
            "<sup>Composite score = vocab + scale_mismatch + cross_lab + MRI</sup>",
        ],
    )

    # ── Left: scatter ──────────────────────────────────────────────────────
    scatter = go.Scatter(
        x=agg["infra_vocab"],
        y=agg["scale_mismatch"],
        mode="markers",
        marker=dict(
            size=np.sqrt(agg["cross_lab_dup"].fillna(0.1)) * 60 + 6,
            sizemode="diameter",
            color=agg["te_clamped"],
            colorscale="Plasma",
            colorbar=dict(title="Theory/Expt<br>ratio", x=0.56, len=0.8),
            cmin=0, cmax=5,
            opacity=0.8,
            line=dict(width=0.5, color="white"),
        ),
        text=agg.apply(lambda r: (
            f"<b>{r['label']}</b><br>"
            f"infra_vocab: {r['infra_vocab']:.4f}<br>"
            f"scale_mismatch: {r['scale_mismatch']:.4f}<br>"
            f"cross_lab_dup: {r['cross_lab_dup']:.3f}<br>"
            f"theory/expt ratio: {r['theory_expt_ratio']:.2f}<br>"
            f"mri_fraction: {r['mri_fraction']:.3f}<br>"
            f"total awards: {int(r['total_awards'])}<br>"
            f"cross_dir_entropy: {r['cross_dir_entropy']:.2f}<br>"
            f"infra_opportunity: {r['infra_opportunity']:.4f}"
        ), axis=1),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )
    fig.add_trace(scatter, row=1, col=1)

    # Quadrant: top-right = highest priority
    qx = agg["infra_vocab"].quantile(0.75)
    qy = agg["scale_mismatch"].quantile(0.75)
    fig.add_vline(x=qx, line=dict(color="gray", dash="dot", width=1), row=1, col=1)
    fig.add_hline(y=qy, line=dict(color="gray", dash="dot", width=1), row=1, col=1)
    fig.add_annotation(
        x=agg["infra_vocab"].max() * 0.92,
        y=agg["scale_mismatch"].max() * 0.97,
        text="🏗 Infrastructure<br>priority zone",
        showarrow=False,
        font=dict(size=10, color="darkred"),
        row=1, col=1,
    )

    # ── Right: stacked bar of top 20 ───────────────────────────────────────
    top20 = agg.nlargest(20, "infra_opportunity").sort_values("infra_opportunity")
    components = {
        "infra_vocab":    ("Community signalling",  "#2196F3"),
        "scale_mismatch": ("Scale beyond single lab","#FF9800"),
        "cross_lab_dup":  ("Cross-lab fragmentation","#9C27B0"),
        "mri_fraction":   ("MRI / facility awards",  "#4CAF50"),
    }
    weights = {"infra_vocab": 3.0, "scale_mismatch": 2.0,
               "cross_lab_dup": 1.5, "mri_fraction": 1.0}

    for col_name, (label, color) in components.items():
        fig.add_trace(go.Bar(
            y=top20["label"],
            x=top20[col_name].fillna(0) * weights[col_name],
            name=label,
            orientation="h",
            marker_color=color,
            hovertemplate=f"{label}: %{{x:.4f}}<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        title="NSF Infrastructure Investment Opportunity Analysis (2015–2024)<br>"
              "<sup>Identifying coordination failures where a PDB/GenBank-style "
              "community resource would unlock the field</sup>",
        barmode="stack",
        height=700, width=1400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(x=1.01, y=0.5, font=dict(size=10)),
        xaxis=dict(title="Community infra vocabulary density", gridcolor="#eee"),
        yaxis=dict(title="Scale mismatch score", gridcolor="#eee"),
        xaxis2=dict(title="Composite infrastructure opportunity score",
                    gridcolor="#eee"),
        yaxis2=dict(tickfont=dict(size=9)),
    )

    path = output_dir / "infrastructure_opportunity.html"
    fig.write_html(str(path))
    print(f"    → {path.name}")

    # Text summary
    print("\n─── Top 15 infrastructure investment opportunities ───")
    top15 = agg.nlargest(15, "infra_opportunity")
    for _, r in top15.iterrows():
        print(
            f"  [{int(r['cluster_id']):>4}]  "
            f"score={r['infra_opportunity']:.4f}  "
            f"vocab={r['infra_vocab']:.4f}  "
            f"scale={r['scale_mismatch']:.4f}  "
            f"xlab={r['cross_lab_dup']:.3f}  "
            f"n={int(r['total_awards'])}  "
            f"{_short_label(str(r['top_terms']).split('|')[0].strip(), 40)}"
        )


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: Trajectory tracking + visualization")
    p.add_argument("--assign",     default=str(ASSIGN_PATH))
    p.add_argument("--stats",      default=str(STATS_PATH))
    p.add_argument("--db",         default=str(_DB))
    p.add_argument("--output-dir", default=str(_OUTPUT))
    p.add_argument("--top-n",      type=int, default=40,
                   help="Clusters to show in trajectory + dashboard plots")
    p.add_argument("--sankey-year",type=int, default=2018,
                   help="Base year for Sankey transition diagram")
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    assign     = pd.read_parquet(args.assign)
    stats      = pd.read_parquet(args.stats)

    print(f"Loaded {len(assign):,} award assignments, {len(stats):,} cluster-year rows\n")

    # ── Step 1: transitions ────────────────────────────────────────────────
    trans       = compute_transitions(assign, Path(args.db))
    trans_types = classify_transitions(trans)
    trans.to_parquet(TRANS_OUT, index=False)
    print(f"Saved {TRANS_OUT.name}  ({len(trans):,} rows)\n")

    # ── Step 2: enrich stats with additional features ──────────────────────
    stats = compute_additional_features(stats, assign, Path(args.db))
    stats = stats.merge(
        trans_types[["cluster_id","year","transition_fwd","n_successors"]],
        on=["cluster_id","year"], how="left",
    )
    stats.to_parquet(ENRICH_OUT, index=False)
    print(f"Saved {ENRICH_OUT.name}  ({len(stats):,} rows)\n")

    # ── Step 3: visualizations ─────────────────────────────────────────────
    print("Generating visualizations …")
    viz_bubble_trajectories(stats, trans_types, output_dir, top_n=args.top_n)
    viz_feature_dashboard(stats, output_dir, top_n=12)
    viz_breakthrough_scatter(stats, output_dir)
    viz_transition_sankey(trans, stats, output_dir, year_t=args.sankey_year)
    viz_cluster_heatmap(stats, output_dir, top_n=50)
    viz_infrastructure_opportunity(stats, output_dir)

    print("\nPhase 2 complete. Open these in a browser:")
    for name in ["bubble_trajectories","feature_dashboard","breakthrough_scatter",
                 "transition_sankey","cluster_heatmap","infrastructure_opportunity"]:
        print(f"  {output_dir / (name + '.html')}")

    print("\nNext: run nsf_openalex_fetch.py for Phase 3 (citation data)")

    # ── Quick text summary of top breakthrough candidates ──────────────────
    recent = stats[stats["year"] >= 2018].copy()
    if "citation_acceleration" in recent.columns:
        candidates = (recent.groupby("cluster_id").agg(
            pct_new_pis      =("pct_new_pis",         "mean"),
            cross_dir_entropy=("cross_dir_entropy",    "mean"),
            acceleration     =("citation_acceleration","mean"),
            sustainability   =("growth_sustainability","mean"),
            size             =("size",                 "sum"),
            top_terms        =("top_terms",            "first"),
        ).query("size >= 20")
          .assign(score=lambda d:
              d.pct_new_pis.fillna(0) +
              d.cross_dir_entropy.fillna(0) +
              d.acceleration.fillna(0).abs() * 0.5)
          .sort_values("score", ascending=False)
          .head(15))

        print("\n─── Top 15 breakthrough candidates (2018–2024) ───")
        for _, r in candidates.iterrows():
            print(f"  [{int(r.name):>4}]  score={r.score:.2f}  "
                  f"new={r.pct_new_pis:.2f}  "
                  f"xdir={r.cross_dir_entropy:.2f}  "
                  f"accel={r.acceleration:.3f}  "
                  f"n={int(r.size)}  "
                  f"{_short_label(str(r.top_terms).split('|')[0].strip(), 40)}")


if __name__ == "__main__":
    main()
