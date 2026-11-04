"""Streamlit dashboard for rubric-scored proposals (output of pdf_batch_extract.py).

Shows at-a-glance summary statistics for a few hundred scored proposals: score
distributions per rubric dimension, proposal types, other-agency alignment and
FFRDC flags, a filterable proposal table with summary view, and QA checks on the
model's output (gatekeeper-rule violations, failed files).

Works with any rubric whose JSON has `dimension_scores` (D1_..., D2_...) and
`overall_score`; the agency/FFRDC/type/summary panels appear when those fields exist.

Usage:
    streamlit run rubric_dashboard.py
    streamlit run rubric_dashboard.py -- --results output/ai_datasets/all.jsonl \
        --descriptions solr_descriptions/descriptions --metadata ids.xlsx

All paths can also be changed in the sidebar.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---- palette (dataviz reference palette) ---------------------------------- #
# Ordinal blue ramps; lightest step stays >=2:1 against the surface.
RAMP_0_3 = ["#86b6ef", "#3987e5", "#256abf", "#184f95"]
RAMP_0_5 = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#184f95"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LEVEL_COLORS = {"none": "#a8a7a0", "partial": "#eda100", "strong": "#e34948"}
TYPE_ORDER = ["Planning", "Impact/Flagship", "Unclear"]
LEVEL_ORDER = ["none", "partial", "strong"]
DIRECTORATES = ["BIO", "CISE", "EDU", "ENG", "GEO", "MPS", "SBE", "TIP"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
DASHBOARD_VERSION = "2026-11-04 (domains + FFRDC table)"


def newest_results(root="output"):
    """Most recently written all.jsonl under ./output (any subfolder)."""
    found = sorted(Path(root).rglob("all.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if Path(root).is_dir() else []
    return str(found[0]) if found else "output/ai_datasets/all.jsonl"


def cli_defaults():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="default: newest output/**/all.jsonl")
    ap.add_argument("--descriptions", default="solr_descriptions/descriptions")
    ap.add_argument("--metadata", default="")
    args, _ = ap.parse_known_args(sys.argv[1:])
    if args.results is None:
        args.results = newest_results()
    return args


def short_dim(key: str) -> str:
    """D1_Program_Fit_and_Proposal_Type_Coherence -> 'D1 Program Fit'."""
    m = re.match(r"(D\d+)_?(.*)", key)
    if not m:
        return key
    words = [w for w in m.group(2).split("_") if w.lower() not in ("and", "of", "the")]
    return f"{m.group(1)} {' '.join(words[:2])}".strip()


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _clean_dir(v):
    """'BIO' / 'bio' / 'BIO (Biological Sciences)' -> 'BIO'; None if empty."""
    if not v:
        return None
    m = re.match(r"\s*([A-Za-z]+)", str(v))
    return m.group(1).upper() if m else None


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_results(path: str, mtime: float):
    recs = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    # Group dimension keys by their D-number, so naming drift across results
    # (e.g. runs without --schema, or an older rubric) can't create duplicate
    # columns. Label each dimension with its most common key.
    def dnum(k):
        m = re.match(r"(?i)(D\d+)", k.strip())
        return m.group(1).upper() if m else k.strip()

    key_counts = Counter(k for r in recs for k in (r.get("dimension_scores") or {}))
    canon = {}  # "D1" -> most common full key
    for k, _ in key_counts.most_common():
        canon.setdefault(dnum(k), k)
    nums = sorted(canon, key=lambda n: (_int(n[1:]) if re.fullmatch(r"D\d+", n) else 99, n))
    labels = {n: short_dim(canon[n]) if re.fullmatch(r"D\d+", n) else n for n in nums}
    dims = list(dict.fromkeys(labels.values()))  # unique, in order

    rows = []
    for r in recs:
        pid = Path(str(r.get("file", ""))).stem
        ds = r.get("dimension_scores") or {}
        by_num = {}
        for k, v in ds.items():
            by_num.setdefault(dnum(k), v)
        nonstd = sorted(k for k in ds if k != canon.get(dnum(k)))
        row = {
            "id": pid,
            "proposal_type": r.get("proposal_type") or "Unclear",
            "summary": r.get("summary", ""),
            "overall": _int(_get(r, "overall_score", "score")),
            "overall_just": _get(r, "overall_score", "justification", default=""),
            "agency_level": _get(r, "agency_alignment", "level"),
            "agencies": _get(r, "agency_alignment", "agencies", default=[]) or [],
            "agency_just": _get(r, "agency_alignment", "justification", default=""),
            "ffrdc": _get(r, "ffrdc_involvement", "involved"),
            "ffrdc_entities": _get(r, "ffrdc_involvement", "entities", default=[]) or [],
            "ffrdc_role": _get(r, "ffrdc_involvement", "role"),
            "ffrdc_just": _get(r, "ffrdc_involvement", "justification", default=""),
            "primary_dir": _clean_dir(_get(r, "domain", "primary_directorate")),
            "secondary_dirs": [],
            "field": _get(r, "domain", "field", default="") or "",
            "domain_just": _get(r, "domain", "justification", default="") or "",
            "nonstandard_keys": nonstd,
        }
        sec = _get(r, "domain", "secondary_directorates", default=[]) or []
        row["secondary_dirs"] = list(dict.fromkeys(
            d for d in map(_clean_dir, sec if isinstance(sec, list) else [sec])
            if d and d != row["primary_dir"]))
        for n in nums:
            v = by_num.get(n) if isinstance(by_num.get(n), dict) else {}
            row[labels[n]] = _int(v.get("score"))
            row[labels[n] + "__just"] = v.get("justification", "")
        rows.append(row)
    df = pd.DataFrame(rows)
    # same proposal scored twice (e.g. 123.txt and 123.pdf): keep the last
    df = df.drop_duplicates("id", keep="last").reset_index(drop=True)
    return df, dims


@st.cache_data(show_spinner=False)
def load_titles(desc_dir: str, mtime: float) -> dict:
    """Titles from solr_fetch_descriptions manifest.csv, else each file's first line."""
    d = Path(desc_dir)
    manifest = d.parent / "manifest.csv"
    titles = {}
    if manifest.exists():
        m = pd.read_csv(manifest, dtype=str).fillna("")
        titles = dict(zip(m["id"], m["title"]))
    for f in d.glob("*.txt"):
        if f.stem not in titles or not titles[f.stem]:
            with open(f, errors="replace") as fh:
                first = fh.readline().strip()
            titles[f.stem] = re.sub(r"^Proposal \d+:\s*", "", first)
    return titles


@st.cache_data(show_spinner=False)
def load_ffrdc_table(path: str, mtime: float):
    """[(canonical name, sponsor, [regex, ...])] from ffrdc_list.csv."""
    t = pd.read_csv(path, dtype=str).fillna("")
    table = []
    for _, row in t.iterrows():
        names = [row["ffrdc"]] + [a.strip() for a in row["aliases"].split(";") if a.strip()]
        pats = []
        for nm in names:
            body = r"[\s\-]+".join(map(re.escape, nm.split()))
            acronym = len(nm) <= 6 and nm.replace("-", "").isalnum() and nm.upper() == nm
            pats.append(re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])",
                                   0 if acronym else re.IGNORECASE))
        table.append((row["ffrdc"], row["sponsor"], pats))
    return table


def match_ffrdcs(entities, table):
    """Split the model's named entities into (recognized FFRDCs, unrecognized names)."""
    confirmed, unrecognized = [], []
    for e in entities or []:
        hits = [name for name, _, pats in table if any(p.search(str(e)) for p in pats)]
        if hits:
            confirmed += [h for h in hits if h not in confirmed]
        else:
            unrecognized.append(str(e).strip())
    return confirmed, unrecognized


@st.cache_data(show_spinner=False)
def load_metadata(path: str, mtime: float) -> pd.DataFrame:
    p = Path(path)
    df = (pd.read_excel(p, header=None, dtype=str) if p.suffix.lower() in (".xlsx", ".xls")
          else pd.read_csv(p, header=None, dtype=str))
    is_id = lambda v: bool(re.fullmatch(r"\d{7}(\.0+)?", str(v).strip()))
    id_col = max(df.columns, key=lambda c: df[c].map(is_id).sum())
    df = df[df[id_col].map(is_id)].copy()
    df["id"] = df[id_col].str.strip().str.split(".").str[0]
    return df.drop(columns=[id_col]).drop_duplicates("id")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _layout(fig, height=320, **kw):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8),
                      bargap=0.25, hoverlabel=dict(namelength=-1), **kw)
    try:  # rounded bar ends need plotly >= 5.19
        fig.update_traces(marker_cornerradius=4, selector=dict(type="bar"))
    except Exception:
        pass
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def chart_overall(df):
    counts = df["overall"].value_counts().reindex(range(6), fill_value=0)
    n = max(len(df), 1)
    fig = go.Figure(go.Bar(
        x=list(range(6)), y=counts.values, marker_color=RAMP_0_5,
        marker_line=dict(width=0),
        text=counts.values, textposition="outside",
        customdata=[f"{c / n:.0%}" for c in counts.values],
        hovertemplate="Overall score %{x}<br>%{y} proposals (%{customdata})<extra></extra>"))
    fig.update_xaxes(title="Overall relevance score (0–5)", tickvals=list(range(6)))
    fig.update_yaxes(title="Proposals")
    return _layout(fig, title="Overall relevance score")


def chart_dimensions(df, dims):
    fig = go.Figure()
    for s in range(4):
        pct, cnt = [], []
        for d in dims:
            col = df[d].dropna()
            c = int((col == s).sum())
            cnt.append(c)
            pct.append(100 * c / max(len(col), 1))
        fig.add_trace(go.Bar(
            y=dims, x=pct, name=f"{s}", orientation="h", marker_color=RAMP_0_3[s],
            marker_line=dict(width=2, color="rgba(0,0,0,0)"),
            customdata=cnt,
            hovertemplate="%{y}<br>score " + str(s) + ": %{customdata} proposals (%{x:.0f}%)<extra></extra>"))
    fig.update_layout(barmode="stack", legend=dict(title="Score", orientation="h", y=1.12, x=0),
                      yaxis=dict(autorange="reversed"))
    fig.update_xaxes(title="% of proposals", range=[0, 100], ticksuffix="%")
    return _layout(fig, height=60 + 42 * len(dims), title="Score distribution by dimension (0–3)")


def chart_types(df):
    counts = df["proposal_type"].value_counts()
    order = [t for t in TYPE_ORDER if t in counts] + [t for t in counts.index if t not in TYPE_ORDER]
    fig = go.Figure(go.Bar(
        x=order, y=[counts[t] for t in order],
        marker_color=[CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(order))],
        text=[counts[t] for t in order], textposition="outside",
        hovertemplate="%{x}: %{y} proposals<extra></extra>"))
    fig.update_yaxes(title="Proposals")
    return _layout(fig, title="Proposal type")


def chart_agencies(df):
    flagged = df[df["agency_level"].isin(["partial", "strong"])]
    exploded = flagged.explode("agencies").dropna(subset=["agencies"])
    if exploded.empty:
        return None
    agencies = exploded["agencies"].value_counts().index.tolist()
    fig = go.Figure()
    for lvl in ["partial", "strong"]:
        sub = exploded[exploded["agency_level"] == lvl]["agencies"].value_counts()
        fig.add_trace(go.Bar(
            x=agencies, y=[int(sub.get(a, 0)) for a in agencies], name=lvl,
            marker_color=LEVEL_COLORS[lvl],
            hovertemplate="%{x} (" + lvl + "): %{y} proposals<extra></extra>"))
    fig.update_layout(barmode="stack", legend=dict(title="Alignment", orientation="h", y=1.12, x=0))
    fig.update_yaxes(title="Proposals")
    return _layout(fig, title="Other agencies flagged")


def chart_ffrdc(df):
    exploded = df[df["ffrdc"] == True].explode("ffrdc_entities").dropna(subset=["ffrdc_entities"])  # noqa: E712
    if exploded.empty:
        return None
    counts = exploded["ffrdc_entities"].str.strip().value_counts().head(15)[::-1]
    fig = go.Figure(go.Bar(
        y=counts.index, x=counts.values, orientation="h", marker_color=CATEGORICAL[0],
        text=counts.values, textposition="outside",
        hovertemplate="%{y}: %{x} proposals<extra></extra>"))
    fig.update_xaxes(title="Proposals")
    return _layout(fig, height=80 + 26 * len(counts), title="FFRDCs named (top 15)")


def chart_directorates(df):
    prim = df["primary_dir"].value_counts()
    sec = df["secondary_dirs"].explode().dropna().value_counts()
    order = sorted(set(prim.index) | set(sec.index),
                   key=lambda d: -(prim.get(d, 0) + sec.get(d, 0)))
    if not order:
        return None
    order = order[::-1]  # largest at top of a horizontal chart
    fig = go.Figure()
    for name, counts, color in (("primary", prim, CATEGORICAL[0]), ("secondary", sec, CATEGORICAL[1])):
        fig.add_trace(go.Bar(
            y=order, x=[int(counts.get(d, 0)) for d in order], name=name, orientation="h",
            marker_color=color, marker_line=dict(width=2, color="rgba(0,0,0,0)"),
            hovertemplate="%{y} as " + name + ": %{x} proposals<extra></extra>"))
    fig.update_layout(barmode="stack", legend=dict(title="Directorate is", orientation="h", y=1.12, x=0))
    fig.update_xaxes(title="Proposals")
    return _layout(fig, height=90 + 34 * len(order), title="Scientific domain by NSF directorate")


def chart_pairs(df):
    pairs = []
    for p, secs in zip(df["primary_dir"], df["secondary_dirs"]):
        for s2 in secs:
            if p:
                pairs.append(" + ".join(sorted((p, s2))))
    if not pairs:
        return None
    counts = pd.Series(pairs).value_counts().head(12)[::-1]
    fig = go.Figure(go.Bar(
        y=counts.index, x=counts.values, orientation="h", marker_color=CATEGORICAL[2],
        text=counts.values, textposition="outside",
        hovertemplate="%{y}: %{x} proposals<extra></extra>"))
    fig.update_xaxes(title="Proposals")
    return _layout(fig, height=80 + 28 * len(counts), title="Most common directorate combinations")


def show(fig):
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Proposal Rubric Dashboard", layout="wide")
    defaults = cli_defaults()

    with st.sidebar:
        st.header("Data")
        res_path = st.text_input("Results (all.jsonl)", defaults.results)
        desc_path = st.text_input("Descriptions folder (optional)", defaults.descriptions)
        meta_path = st.text_input("ID spreadsheet for PI/institution (optional)", defaults.metadata)
        ffrdc_path = st.text_input("FFRDC list", str(Path(__file__).parent / "ffrdc_list.csv"))
        others = [str(p) for p in Path("output").rglob("all.jsonl")] if Path("output").is_dir() else []
        if len(others) > 1:
            st.caption("Other results files: " + ", ".join(f"`{o}`" for o in others if o != res_path))
        st.caption(f"Dashboard version {DASHBOARD_VERSION}")

    if not Path(res_path).is_file():
        st.error(f"Results file not found: `{res_path}`. Run pdf_batch_extract.py first, "
                 "or fix the path in the sidebar.")
        st.stop()

    df, dims = load_results(res_path, Path(res_path).stat().st_mtime)
    if df.empty:
        st.warning("No results in the file yet.")
        st.stop()

    titles = {}
    if desc_path and Path(desc_path).is_dir():
        titles = load_titles(desc_path, Path(desc_path).stat().st_mtime)
    df.insert(1, "title", df["id"].map(titles).fillna(""))

    extra_cols = []
    if meta_path and Path(meta_path).is_file():
        meta = load_metadata(meta_path, Path(meta_path).stat().st_mtime)
        sample = meta.iloc[0] if len(meta) else {}
        opts = ["(none)"] + [c for c in meta.columns if c != "id"]
        label = lambda c: c if c == "(none)" else f"col {c}: {str(sample.get(c, ''))[:30]}"
        guess = lambda i: opts.index(i) if i in opts else 0
        with st.sidebar:
            inst_c = st.selectbox("Institution column", opts, index=guess(3), format_func=label)
            pi_c = st.selectbox("PI column", opts, index=guess(4), format_func=label)
        for name, c in (("institution", inst_c), ("PI", pi_c)):
            if c != "(none)" and name not in df.columns:
                df = df.merge(meta[["id", c]].rename(columns={c: name}), on="id", how="left")
                extra_cols.append(name)

    has_flags = df["agency_level"].notna().any()
    has_ffrdc = df["ffrdc"].notna().any()
    has_domain = df["primary_dir"].notna().any()
    df["interdisciplinary"] = df["secondary_dirs"].map(bool) & df["primary_dir"].notna()

    # FFRDC status comes from the table, not the model: the model only names
    # organizations; anything not on ffrdc_list.csv is set aside for review.
    have_table = bool(ffrdc_path) and Path(ffrdc_path).is_file()
    df["ffrdc_named"] = df["ffrdc_entities"]
    if have_table:
        table = load_ffrdc_table(ffrdc_path, Path(ffrdc_path).stat().st_mtime)
        split = df["ffrdc_named"].map(lambda ents: match_ffrdcs(ents, table))
        df["ffrdc_entities"] = split.map(lambda t: t[0])
        df["ffrdc_unrecognized"] = split.map(lambda t: t[1])
        df["ffrdc"] = df["ffrdc_entities"].map(bool)
    else:
        df["ffrdc_unrecognized"] = [[] for _ in range(len(df))]
        if has_ffrdc:
            st.sidebar.warning("FFRDC list not found; showing the model's FFRDC calls unchecked.")

    # ---- filters ---------------------------------------------------------- #
    with st.sidebar:
        st.header("Filters")
        lo, hi = st.slider("Overall score", 0, 5, (0, 5))
        types = st.multiselect("Proposal type", sorted(df["proposal_type"].unique()))
        levels = st.multiselect("Other-agency alignment", LEVEL_ORDER) if has_flags else []
        ffrdc_only = st.checkbox("Only proposals with FFRDC involvement") if has_ffrdc else False
        dir_pick = st.multiselect("Directorate (primary or secondary)", DIRECTORATES) if has_domain else []
        inter_only = st.checkbox("Only interdisciplinary proposals") if has_domain else False
        query = st.text_input("Search title / summary / justifications")

    f = df[df["overall"].between(lo, hi) | df["overall"].isna() & (lo == 0)]
    if types:
        f = f[f["proposal_type"].isin(types)]
    if levels:
        f = f[f["agency_level"].isin(levels)]
    if dir_pick:
        f = f[f["primary_dir"].isin(dir_pick) | f["secondary_dirs"].map(lambda l: bool(set(l) & set(dir_pick)))]
    if inter_only:
        f = f[f["interdisciplinary"]]
    if ffrdc_only:
        f = f[f["ffrdc"] == True]  # noqa: E712
    if query:
        text_cols = ["title", "summary", "overall_just", "agency_just", "ffrdc_just", "field",
                     "domain_just"] + \
                    [d + "__just" for d in dims] + extra_cols
        hay = f[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        f = f[hay.str.contains(query, case=False, regex=False)]

    st.title("Proposal rubric dashboard")
    import datetime as _dt
    updated = _dt.datetime.fromtimestamp(Path(res_path).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    st.caption(f"{len(f)} of {len(df)} proposals shown · `{res_path}` (updated {updated})")

    # ---- KPI row ---------------------------------------------------------- #
    n = max(len(f), 1)
    k = st.columns(7)
    k[0].metric("Proposals", len(f))
    k[1].metric("Mean overall", f"{f['overall'].mean():.2f}" if f["overall"].notna().any() else "–")
    k[2].metric("Highly relevant (≥4)", f"{(f['overall'] >= 4).sum()}",
                f"{(f['overall'] >= 4).sum() / n:.0%}", delta_color="off")
    d1 = dims[0] if dims else None
    k[3].metric("Out of scope (D1=0)", f"{(f[d1] == 0).sum()}" if d1 else "–",
                f"{(f[d1] == 0).sum() / n:.0%}" if d1 else None, delta_color="off")
    if has_flags:
        k[4].metric("Other agency: strong", f"{(f['agency_level'] == 'strong').sum()}",
                    f"+{(f['agency_level'] == 'partial').sum()} partial", delta_color="off")
    if has_ffrdc:
        k[6].metric("FFRDC involved", f"{(f['ffrdc'] == True).sum()}",  # noqa: E712
                    f"{(f['ffrdc'] == True).sum() / n:.0%}", delta_color="off")  # noqa: E712

    if has_domain:
        k[5].metric("Interdisciplinary", f"{f['interdisciplinary'].sum()}",
                    f"{f['interdisciplinary'].sum() / n:.0%}", delta_color="off")

    tabs = st.tabs(["Overview", "Domains", "Flags", "Proposals", "QA"])

    # ---- Overview --------------------------------------------------------- #
    with tabs[0]:
        c1, c2 = st.columns([3, 2])
        with c1:
            show(chart_overall(f))
        with c2:
            show(chart_types(f))
        if dims:
            show(chart_dimensions(f, dims))
            st.subheader("Mean dimension score by proposal type")
            means = f.groupby("proposal_type")[dims + ["overall"]].mean().round(2)
            means.insert(0, "n", f.groupby("proposal_type").size())
            st.dataframe(means, use_container_width=True)

    # ---- Domains ---------------------------------------------------------- #
    with tabs[1]:
        if not has_domain:
            st.info("These results have no domain fields. Re-run with the updated r05 rubric and schema.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                show(chart_directorates(f))
            with c2:
                show(chart_pairs(f))
            st.subheader("Scores by primary directorate")
            g = f[f["primary_dir"].notna()].groupby("primary_dir")
            by_dir = g[["overall"] + dims].mean().round(2)
            by_dir.insert(0, "interdisciplinary", (g["interdisciplinary"].mean() * 100).round().astype(int)
                          .astype(str) + "%")
            by_dir.insert(0, "n", g.size())
            st.dataframe(by_dir.sort_values("n", ascending=False), use_container_width=True)
            st.subheader("Most common fields")
            fields = f["field"].str.strip().str.lower().replace("", pd.NA).dropna()
            if len(fields):
                st.dataframe(fields.value_counts().head(25).rename_axis("field").reset_index(name="proposals"),
                             hide_index=True, use_container_width=True)

    # ---- Flags ------------------------------------------------------------ #
    with tabs[2]:
        if not (has_flags or has_ffrdc):
            st.info("These results have no agency/FFRDC fields. Re-run with the r05 rubric.")
        c1, c2 = st.columns(2)
        with c1:
            if has_flags:
                show(chart_agencies(f))
                flagged = f[f["agency_level"].isin(["partial", "strong"])].sort_values(
                    "agency_level", key=lambda s: s.map({"strong": 0, "partial": 1}))
                st.markdown(f"**{len(flagged)} proposals flagged for another agency**")
                st.dataframe(flagged[["id", "title", "agency_level", "agencies", "overall", "agency_just"]],
                             hide_index=True, use_container_width=True,
                             column_config={"agency_just": st.column_config.TextColumn("why", width="large")})
        with c2:
            if has_ffrdc:
                show(chart_ffrdc(f))
                ff = f[f["ffrdc"] == True]  # noqa: E712
                st.markdown(f"**{len(ff)} proposals with FFRDC involvement**")
                st.dataframe(ff[["id", "title", "ffrdc_entities", "ffrdc_role", "overall", "ffrdc_just"]],
                             hide_index=True, use_container_width=True,
                             column_config={"ffrdc_just": st.column_config.TextColumn("why", width="large")})
                if have_table:
                    un = f[["id", "ffrdc_unrecognized"]].explode("ffrdc_unrecognized").dropna()
                    if len(un):
                        st.markdown(f"**Named by the model but not on the FFRDC list** "
                                    f"({un['id'].nunique()} proposals, not counted above)")
                        st.caption("Usually computing resources or programs such as NAIRR or state "
                                   "supercomputing centers. If one really is an FFRDC, add it to "
                                   "ffrdc_list.csv.")
                        st.dataframe(un.groupby("ffrdc_unrecognized")["id"]
                                     .agg(proposals="count", ids=lambda s: ", ".join(s))
                                     .sort_values("proposals", ascending=False).reset_index()
                                     .rename(columns={"ffrdc_unrecognized": "name"}),
                                     hide_index=True, use_container_width=True)

    # ---- Proposals -------------------------------------------------------- #
    with tabs[3]:
        cols = ["id", "title"] + extra_cols + ["proposal_type"]
        if has_domain:
            cols += ["primary_dir", "secondary_dirs", "field"]
        cols += ["overall"] + dims
        if has_flags:
            cols += ["agency_level", "agencies"]
        if has_ffrdc:
            cols += ["ffrdc_entities"]
        cols = list(dict.fromkeys(c for c in cols if c in f.columns))
        table = f[cols].sort_values("overall", ascending=False)
        cfg = {"overall": st.column_config.NumberColumn("overall", format="%d"),
               "title": st.column_config.TextColumn(width="large")}
        st.caption("Click a row to see its summary below.")
        try:  # row selection needs streamlit >= 1.35
            event = st.dataframe(table, hide_index=True, use_container_width=True, column_config=cfg,
                                 height=420, on_select="rerun", selection_mode="single-row",
                                 key="proposal_table")
            sel_rows = list(event.selection.rows) if event is not None else []
        except TypeError:
            st.dataframe(table, hide_index=True, use_container_width=True, column_config=cfg, height=420)
            sel_rows = []
        export = f.copy()
        for c in ("agencies", "ffrdc_entities", "ffrdc_named", "ffrdc_unrecognized", "secondary_dirs"):
            export[c] = export[c].map(lambda v: "; ".join(v) if isinstance(v, list) else v)
        st.download_button("Download filtered results (CSV)", export.to_csv(index=False),
                           "rubric_results_filtered.csv", "text/csv")

        st.subheader("Proposal summary")
        if len(table):
            ids = table["id"].tolist()
            labels = {i: f"{i} · {t[:80]} (overall {o})"
                      for i, t, o in zip(table["id"], table["title"], table["overall"])}
            idx = sel_rows[0] if sel_rows and sel_rows[0] < len(ids) else 0
            pick = st.selectbox("Proposal", ids, index=idx, format_func=labels.get)
            r = f.set_index("id").loc[pick]
            st.markdown(f"### {r['title'] or pick}")
            meta_line = " · ".join(str(r[c]) for c in extra_cols if pd.notna(r[c]))
            dom = ""
            if has_domain and r["primary_dir"]:
                dom = " · " + " + ".join([r["primary_dir"]] + list(r["secondary_dirs"]))
                dom += f" ({r['field']})" if r["field"] else ""
            st.caption(f"{pick} · {r['proposal_type']} · overall {r['overall']}{dom}"
                       + (f" · {meta_line}" if meta_line else ""))
            st.write(r["summary"] or "_No summary in this result._")
            with st.expander("Scores and justifications", expanded=False):
                detail = pd.DataFrame(
                    [{"dimension": d, "score": r[d], "justification": r[d + "__just"]} for d in dims]
                    + [{"dimension": "Overall (0–5)", "score": r["overall"],
                        "justification": r["overall_just"]}])
                st.dataframe(detail, hide_index=True, use_container_width=True,
                             column_config={"justification": st.column_config.TextColumn(width="large")})
                if has_domain and r["domain_just"]:
                    st.markdown(f"**Domain:** {r['domain_just']}")
                if has_flags:
                    st.markdown(f"**Other-agency alignment:** {r['agency_level']} "
                                f"{', '.join(r['agencies']) if r['agencies'] else ''} — {r['agency_just']}")
                if has_ffrdc:
                    ents = ", ".join(r["ffrdc_entities"]) if r["ffrdc_entities"] else "none"
                    st.markdown(f"**FFRDC:** {ents} ({r['ffrdc_role']}) — {r['ffrdc_just']}")
                    if r["ffrdc_unrecognized"]:
                        st.caption("Named by the model but not on the FFRDC list: "
                                   + ", ".join(r["ffrdc_unrecognized"]))

    # ---- QA --------------------------------------------------------------- #
    with tabs[4]:
        rwr_file = Path(desc_path).parent / "collaborative_rwr.csv" if desc_path else None
        if rwr_file and rwr_file.is_file():
            rwr = pd.read_csv(rwr_file, dtype=str).fillna("")
            if len(rwr):
                st.subheader(f"Collaborative proposals: {len(rwr)} return-without-review candidates")
                st.caption("Found by solr_fetch_descriptions.py and not scored (collaborative "
                           "proposals aren't allowed in this solicitation). Confirm each before acting.")
                st.dataframe(rwr, hide_index=True, use_container_width=True,
                             column_config={"title": st.column_config.TextColumn(width="large")})
        st.subheader("Output checks")
        st.markdown("Checks on the model's output. Review these before relying on the numbers.")
        issues = []
        if dims:
            d2 = dims[1] if len(dims) > 1 else None
            for _, r in df.iterrows():
                if r[dims[0]] == 0 and (r["overall"] or 0) > 1:
                    issues.append((r["id"], "D1 = 0 but overall > 1 (gatekeeper rule)"))
                if d2 and r[d2] == 0 and (r["overall"] or 0) > 2:
                    issues.append((r["id"], "D2 = 0 but overall > 2 (essential rule)"))
                if re.match(r"(?i)\s*collaborative\s+research\s*:", str(r["title"])):
                    issues.append((r["id"], "scored, but title says Collaborative Research "
                                            "(return-without-review candidate)"))
                if has_domain and not r["primary_dir"]:
                    issues.append((r["id"], "missing primary directorate"))
                if any(pd.isna(r[d]) for d in dims) or pd.isna(r["overall"]):
                    issues.append((r["id"], "missing score(s)"))
                if has_flags and r["agency_level"] == "none" and r["agencies"]:
                    issues.append((r["id"], "agency level 'none' but agencies listed"))
        for _, r in df.iterrows():
            if r["nonstandard_keys"]:
                issues.append((r["id"], "nonstandard dimension names (run without --schema, "
                                        "or older rubric?)", ", ".join(r["nonstandard_keys"])))
        failed = sorted(p.name.replace(".raw.txt", "") for p in Path(res_path).parent.glob("*.raw.txt"))
        for pid in failed:
            issues.append((pid, "model output could not be parsed (see .raw.txt)"))
        if desc_path and Path(desc_path).is_dir():
            scored = set(df["id"])
            not_scored = sorted(p.stem for p in Path(desc_path).glob("*.txt") if p.stem not in scored)
            for pid in not_scored:
                issues.append((pid, "description downloaded but not scored yet"))
        if issues:
            qa = pd.DataFrame([i if len(i) == 3 else (*i, "") for i in issues],
                              columns=["id", "issue", "detail"])
            st.dataframe(qa.groupby("issue").size().rename("count").reset_index(),
                         hide_index=True, use_container_width=True)
            st.dataframe(qa, hide_index=True, use_container_width=True)
        else:
            st.success("No issues found.")


main()
