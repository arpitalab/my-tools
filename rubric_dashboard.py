"""Streamlit dashboard for rubric-scored proposals (output of pdf_batch_extract.py).

Shows at-a-glance summary statistics for a few hundred scored proposals: score
distributions per rubric dimension, proposal types, other-agency alignment and
FFRDC flags, a filterable proposal table with drill-down, and QA checks on the
model's output (gatekeeper-rule violations, failed files).

Works with any rubric whose JSON has `dimension_scores` (D1_..., D2_...) and
`overall_score`; the agency/FFRDC/type/summary panels appear when those fields exist.

Usage:
    streamlit run rubric_dashboard.py
    streamlit run rubric_dashboard.py -- --results output/pdf_json/all.jsonl \
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


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def cli_defaults():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="output/pdf_json/all.jsonl")
    ap.add_argument("--descriptions", default="solr_descriptions/descriptions")
    ap.add_argument("--metadata", default="")
    args, _ = ap.parse_known_args(sys.argv[1:])
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
            "nonstandard_keys": nonstd,
        }
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

    # ---- filters ---------------------------------------------------------- #
    with st.sidebar:
        st.header("Filters")
        lo, hi = st.slider("Overall score", 0, 5, (0, 5))
        types = st.multiselect("Proposal type", sorted(df["proposal_type"].unique()))
        levels = st.multiselect("Other-agency alignment", LEVEL_ORDER) if has_flags else []
        ffrdc_only = st.checkbox("Only proposals with FFRDC involvement") if has_ffrdc else False
        query = st.text_input("Search title / summary / justifications")

    f = df[df["overall"].between(lo, hi) | df["overall"].isna() & (lo == 0)]
    if types:
        f = f[f["proposal_type"].isin(types)]
    if levels:
        f = f[f["agency_level"].isin(levels)]
    if ffrdc_only:
        f = f[f["ffrdc"] == True]  # noqa: E712
    if query:
        text_cols = ["title", "summary", "overall_just", "agency_just", "ffrdc_just"] + \
                    [d + "__just" for d in dims] + extra_cols
        hay = f[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        f = f[hay.str.contains(query, case=False, regex=False)]

    st.title("Proposal rubric dashboard")
    st.caption(f"{len(f)} of {len(df)} proposals shown · `{res_path}`")

    # ---- KPI row ---------------------------------------------------------- #
    n = max(len(f), 1)
    k = st.columns(6)
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
        k[5].metric("FFRDC involved", f"{(f['ffrdc'] == True).sum()}",  # noqa: E712
                    f"{(f['ffrdc'] == True).sum() / n:.0%}", delta_color="off")  # noqa: E712

    tabs = st.tabs(["Overview", "Flags", "Proposals", "QA"])

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

    # ---- Flags ------------------------------------------------------------ #
    with tabs[1]:
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

    # ---- Proposals -------------------------------------------------------- #
    with tabs[2]:
        cols = ["id", "title"] + extra_cols + ["proposal_type", "overall"] + dims
        if has_flags:
            cols += ["agency_level", "agencies"]
        if has_ffrdc:
            cols += ["ffrdc_entities"]
        cols = list(dict.fromkeys(c for c in cols if c in f.columns))
        table = f[cols].sort_values("overall", ascending=False)
        cfg = {"overall": st.column_config.NumberColumn("overall", format="%d"),
               "title": st.column_config.TextColumn(width="large")}
        st.dataframe(table, hide_index=True, use_container_width=True, column_config=cfg, height=420)
        export = f.copy()
        for c in ("agencies", "ffrdc_entities"):
            export[c] = export[c].map(lambda v: "; ".join(v) if isinstance(v, list) else v)
        st.download_button("Download filtered results (CSV)", export.to_csv(index=False),
                           "rubric_results_filtered.csv", "text/csv")

        st.subheader("Proposal detail")
        if len(table):
            labels = {i: f"{i} · {t[:80]} (overall {o})"
                      for i, t, o in zip(table["id"], table["title"], table["overall"])}
            pick = st.selectbox("Proposal", table["id"].tolist(), format_func=labels.get)
            r = f.set_index("id").loc[pick]
            st.markdown(f"### {r['title'] or pick}")
            meta_line = " · ".join(str(r[c]) for c in extra_cols if pd.notna(r[c]))
            st.caption(f"{pick} · {r['proposal_type']}" + (f" · {meta_line}" if meta_line else ""))
            if r["summary"]:
                st.write(r["summary"])
            detail = pd.DataFrame(
                [{"dimension": d, "score": r[d], "justification": r[d + "__just"]} for d in dims]
                + [{"dimension": "Overall (0–5)", "score": r["overall"], "justification": r["overall_just"]}])
            st.dataframe(detail, hide_index=True, use_container_width=True,
                         column_config={"justification": st.column_config.TextColumn(width="large")})
            if has_flags:
                st.markdown(f"**Other-agency alignment:** {r['agency_level']} "
                            f"{', '.join(r['agencies']) if r['agencies'] else ''} — {r['agency_just']}")
            if has_ffrdc:
                ents = ", ".join(r["ffrdc_entities"]) if r["ffrdc_entities"] else "none"
                st.markdown(f"**FFRDC:** {ents} ({r['ffrdc_role']}) — {r['ffrdc_just']}")
            txt = Path(desc_path) / f"{pick}.txt" if desc_path else None
            if txt and txt.is_file():
                with st.expander("Project description text"):
                    st.text(txt.read_text(errors="replace"))

    # ---- QA --------------------------------------------------------------- #
    with tabs[3]:
        st.markdown("Checks on the model's output. Review these before relying on the numbers.")
        issues = []
        if dims:
            d2 = dims[1] if len(dims) > 1 else None
            for _, r in df.iterrows():
                if r[dims[0]] == 0 and (r["overall"] or 0) > 1:
                    issues.append((r["id"], "D1 = 0 but overall > 1 (gatekeeper rule)"))
                if d2 and r[d2] == 0 and (r["overall"] or 0) > 2:
                    issues.append((r["id"], "D2 = 0 but overall > 2 (essential rule)"))
                if any(pd.isna(r[d]) for d in dims) or pd.isna(r["overall"]):
                    issues.append((r["id"], "missing score(s)"))
                if has_flags and r["agency_level"] == "none" and r["agencies"]:
                    issues.append((r["id"], "agency level 'none' but agencies listed"))
                if has_ffrdc and r["ffrdc"] is False and r["ffrdc_entities"]:
                    issues.append((r["id"], "FFRDC 'not involved' but entities listed"))
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
