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
import zlib
from collections import Counter, defaultdict
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
PLAN_ORDER = ["planning-focused", "mixed", "mostly preliminary research"]
PLAN_COLORS = {"planning-focused": "#a8a7a0", "mixed": "#eda100", "mostly preliminary research": "#e34948"}
# Solicitation limits for Planning proposals (NSF 26-512)
PLAN_MAX_DOLLARS = 200_000
PLAN_MAX_MONTHS = 24
PLAN_PREFIX_RE = re.compile(r"^\s*planning\s*:", re.I)        # declared Planning
PLAN_PREFIX_EXACT = re.compile(r"^Planning: AIDatasets: ")    # required form


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
DASHBOARD_VERSION = "2026-11-11 (other binning)"


def newest_results(root="output"):
    """Most recently written all.jsonl under ./output (any subfolder)."""
    found = sorted(Path(root).rglob("all.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if Path(root).is_dir() else []
    return str(found[0]) if found else "output/ai_datasets/all.jsonl"


RUBRIC_DIRS = (".", "~/Downloads/ai_datasets_screening", "~/Downloads")


def find_rubric(results_path: str) -> str:
    """The rubric matching the results folder's version (e.g. 'r08' in
    output/ai_datasets_r08), else the newest _best-rNN-rubric.md found."""
    found = {}
    for d in RUBRIC_DIRS:
        d = Path(d).expanduser()
        if d.is_dir():
            for f in d.glob("_best-r*-rubric.md"):
                m = re.search(r"-r(\d+)-", f.name)
                if m:
                    found.setdefault(int(m.group(1)), f)
    if not found:
        return ""
    m = re.search(r"r(\d+)", Path(results_path).parent.name)
    if m and int(m.group(1)) in found:
        return str(found[int(m.group(1))])
    return str(found[max(found)])


def cli_defaults():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="default: newest output/**/all.jsonl")
    ap.add_argument("--descriptions", default="solr_descriptions/descriptions")
    ap.add_argument("--metadata", default="")
    ap.add_argument("--other-binning", default="", help="another directorate binning (.xlsx/.csv) to show alongside")
    ap.add_argument("--rubric", default=None, help="rubric .md to show (default: matched to the results folder)")
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


# Division (or old directorate) codes the model sometimes gives instead of a directorate
DIVISION_TO_DIR = {
    **dict.fromkeys(["DBI", "DEB", "IOS", "MCB", "EF"], "BIO"),
    **dict.fromkeys(["CCF", "CNS", "IIS", "OAC"], "CISE"),
    **dict.fromkeys(["EHR", "DGE", "DRL", "DUE", "EES", "HRD"], "EDU"),
    **dict.fromkeys(["CBET", "CMMI", "ECCS", "EEC", "EFMA"], "ENG"),
    **dict.fromkeys(["AGS", "EAR", "OCE", "OPP", "RISE"], "GEO"),
    **dict.fromkeys(["AST", "CHE", "DMR", "DMS", "PHY", "OMA"], "MPS"),
    **dict.fromkeys(["BCS", "SES", "NCSES", "SMA"], "SBE"),
    **dict.fromkeys(["ITE", "TF"], "TIP"),
}


def _clean_dir(v):
    """'BIO' / 'bio' / 'BIO (Biological Sciences)' -> 'BIO'; a division code
    ('AST', 'DMR', 'OAC', ...) -> its directorate; None if empty."""
    if not v:
        return None
    m = re.match(r"\s*([A-Za-z]+)", str(v))
    if not m:
        return None
    code = m.group(1).upper()
    return DIVISION_TO_DIR.get(code, code)


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
            "plan_level": _get(r, "planning_scope", "level"),
            "plan_research": _get(r, "planning_scope", "research_activities", default=[]) or [],
            "plan_path": _get(r, "planning_scope", "future_proposal_path"),
            "plan_just": _get(r, "planning_scope", "justification", default="") or "",
        }
        if row["plan_level"] == "not applicable":
            row["plan_level"] = None
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
    titles = {}
    for name in ("pdf_manifest.csv", "manifest.csv"):  # SOLR titles win
        manifest = d.parent / name
        if manifest.exists():
            m = pd.read_csv(manifest, dtype=str).fillna("")
            if "title" in m:
                titles.update({i: t for i, t in zip(m["id"], m["title"]) if i and t})
    for f in d.glob("*.txt"):
        if f.stem not in titles or not titles[f.stem]:
            with open(f, errors="replace") as fh:
                first = fh.readline().strip()
            titles[f.stem] = re.sub(r"^Proposal \d+:\s*", "", first)
    return titles


@st.cache_data(show_spinner=False)
def find_duplicates(desc_dir: str, ids: tuple, mtime: float, min_overlap: float = 0.5) -> pd.DataFrame:
    """Pairs of proposals (different IDs) whose description texts largely
    overlap: resubmissions, the same PDF filed under two IDs, etc. Compares
    sampled 8-word shingles, so it ignores titles and formatting."""
    sigs = {}
    for pid in ids:
        f = Path(desc_dir) / f"{pid}.txt"
        if not f.is_file():
            continue
        body = f.read_text(errors="replace").split("\n", 1)[-1]  # skip "Proposal <id>: <title>"
        w = re.findall(r"[a-z0-9]+", body.lower())
        sh = {h for h in (zlib.crc32(" ".join(w[i:i + 8]).encode()) for i in range(len(w) - 7))
              if h % 8 == 0}
        if len(sh) >= 30:
            sigs[pid] = sh
    index = defaultdict(list)
    for pid, sh in sigs.items():
        for h in sh:
            index[h].append(pid)
    shared = Counter()
    for pids in index.values():
        if 1 < len(pids) <= 20:  # skip boilerplate many proposals share
            for i, a in enumerate(pids):
                for b in pids[i + 1:]:
                    shared[(a, b) if a < b else (b, a)] += 1
    rows = []
    for (a, b), n in shared.items():
        ov = n / min(len(sigs[a]), len(sigs[b]))
        if ov >= min_overlap:
            rows.append({"id_a": a, "id_b": b, "text overlap": round(100 * ov)})
    return pd.DataFrame(rows, columns=["id_a", "id_b", "text overlap"])


def duplicate_groups(pairs: pd.DataFrame) -> list[list[str]]:
    parent = {}
    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x
    for a, b in zip(pairs["id_a"], pairs["id_b"]):
        parent[root(a)] = root(b)
    groups = defaultdict(list)
    for x in parent:
        groups[root(x)].append(x)
    return [sorted(g) for g in groups.values()]


# Rule-based overall: (dimension index, score, highest overall allowed).
# D1 (program fit) and D2 (AI-readiness) are must-haves, so a weak score on
# either caps the overall no matter how well assembled the rest is.
RULE_CAPS = ((0, 0, 1), (0, 1, 2), (1, 0, 2), (1, 1, 3))
RULE_WEIGHTS = (2, 2, 2, 1, 1, 1)  # D1-D3 count double


def rule_overall(df: pd.DataFrame, dims: list, weights: dict, caps: bool = True) -> pd.Series:
    """Deterministic 0-5 overall from the stored D1-D6 scores: weighted mean
    rescaled to 0-5 and rounded half up, then the must-have caps."""
    total = sum(weights.values())

    def one(r):
        vals = [r[d] for d in dims]
        if not total or any(pd.isna(v) for v in vals):
            return pd.NA
        s = int(5 * sum(weights[d] * v for d, v in zip(dims, vals)) / (3 * total) + 0.5)
        if caps:
            for i, score, top in RULE_CAPS:
                if i < len(dims) and vals[i] == score:
                    s = min(s, top)
        return s
    return df.apply(one, axis=1).astype("Int64") if len(df) else pd.Series(dtype="Int64")


def _money(v):
    m = re.search(r"\d[\d,]*(\.\d+)?", str(v or ""))
    return float(m.group(0).replace(",", "")) if m else None


@st.cache_data(show_spinner=False)
def load_other_binning(path: str, mtime: float) -> pd.DataFrame:
    """Another directorate binning (e.g. a Copilot sheet) -> id, other_dir, other_all,
    other_confidence. Columns are found the same way as in directorate_compare.py."""
    import contextlib
    import io
    here = str(Path(__file__).resolve().parent)  # streamlit doesn't always put it on the path
    if here not in sys.path:
        sys.path.insert(0, here)
    from directorate_compare import load_theirs
    with contextlib.redirect_stdout(io.StringIO()):
        rows = load_theirs([path], None, None, None, None, None, False)
    return pd.DataFrame([{"id": pid, "other_dir": (e["their_dirs"] or [""])[0],
                          "other_all": "; ".join(e["their_dirs"]), "other_confidence": e["their_confidence"]}
                         for pid, e in rows.items()],
                        columns=["id", "other_dir", "other_all", "other_confidence"])


@st.cache_data(show_spinner=False)
def load_manifest_meta(desc_dir: str, mtime: float) -> pd.DataFrame:
    """PI / institution / requested amount / months / concurrence / pages from
    manifest.csv (SOLR) and pdf_manifest.csv (pdf_to_descriptions.py)."""
    parts = []
    for name in ("manifest.csv", "pdf_manifest.csv"):
        f = Path(desc_dir).parent / name
        if f.exists():
            part = pd.read_csv(f, dtype=str).fillna("")
            if "status" in part:  # pdf_text: SOLR metadata for a proposal whose text came from a PDF
                part = part[part["status"].isin(["ok", "pdf_text"])]
            parts.append(part[part["id"] != ""].drop_duplicates("id").set_index("id"))
    if not parts:
        return pd.DataFrame(columns=["id"])
    # field by field: SOLR's value where it has one, else the PDF sheet's
    m = parts[0].replace("", pd.NA)
    for part in parts[1:]:
        m = m.combine_first(part.replace("", pd.NA))
    m = m.fillna("").reset_index()
    out = pd.DataFrame({"id": m["id"]})
    if "pi_name" in m:
        out["PI"] = m["pi_name"]
    if "inst" in m:
        out["institution"] = m["inst"]
    if "requested_amount" in m:
        out["requested $"] = m["requested_amount"].map(_money)
    if "rqst_mnth_cnt" in m:
        out["months"] = m["rqst_mnth_cnt"].map(_money)
    if "concurrence_found" in m:
        out["concurrence_found"] = m["concurrence_found"]
    if "pages" in m:
        out["pages"] = m["pages"].map(_money)
    # drop columns SOLR never filled
    keep = ["id"] + [c for c in out.columns if c != "id"
                     and out[c].replace("", pd.NA).notna().any()]
    return out[keep]


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


def chart_planning(df):
    counts = df["plan_level"].value_counts()
    order = [p for p in PLAN_ORDER if p in counts]
    if not order:
        return None
    n = int(counts[order].sum())
    fig = go.Figure(go.Bar(
        x=order, y=[int(counts[p]) for p in order], marker_color=[PLAN_COLORS[p] for p in order],
        text=[int(counts[p]) for p in order], textposition="outside",
        customdata=[f"{counts[p] / n:.0%}" for p in order],
        hovertemplate="%{x}: %{y} proposals (%{customdata})<extra></extra>"))
    fig.update_yaxes(title="Planning proposals")
    return _layout(fig, title="Planning vs. preliminary research")


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
        meta_path = st.text_input("ID spreadsheet for PI/institution (optional)", defaults.metadata,
                                  help="Usually not needed: PI, institution and $ from the sheet given "
                                       "to pdf_to_descriptions.py are already in pdf_manifest.csv.")
        if meta_path and not Path(meta_path).is_file():
            st.warning(f"ID spreadsheet not found: `{meta_path}` (relative paths are from "
                       f"`{Path.cwd()}`)")
        rubric_path = st.text_input("Rubric (for the Rubric tab)", defaults.rubric or find_rubric(res_path),
                                    help="Found automatically from the results folder's version "
                                         "(r08 -> _best-r08-rubric.md) in this folder, "
                                         "~/Downloads/ai_datasets_screening or ~/Downloads.")
        other_path = st.text_input("Other directorate binning (optional)", defaults.other_binning,
                                   help="An .xlsx/.csv with proposal IDs and a directorate column (e.g. the "
                                        "Copilot exercise). Shown next to ours in Proposals and Domains.")
        if other_path and not Path(other_path).expanduser().is_file():
            st.warning(f"Other binning not found: `{other_path}`")
        ffrdc_path = st.text_input("FFRDC list", str(Path(__file__).parent / "ffrdc_list.csv"))
        others = [str(p) for p in Path("output").rglob("all.jsonl")] if Path("output").is_dir() else []
        if len(others) > 1:
            st.caption("Other results files: " + ", ".join(f"`{o}`" for o in others if o != res_path))
        st.caption(f"Dashboard version {DASHBOARD_VERSION}")
        st.caption("Started with: `" + (" ".join(sys.argv[1:]) or "(no arguments)") + "`")

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
    money_cols = []
    if desc_path and Path(desc_path).is_dir():
        mtimes = [f.stat().st_mtime for f in (Path(desc_path).parent / "manifest.csv",
                                              Path(desc_path).parent / "pdf_manifest.csv") if f.exists()]
        mm = load_manifest_meta(desc_path, max(mtimes, default=0))
        if len(mm.columns) > 1:
            df = df.merge(mm, on="id", how="left")
            extra_cols += [c for c in ("PI", "institution") if c in mm.columns]
            money_cols = [c for c in ("requested $", "months", "pages") if c in mm.columns]
    if "concurrence_found" not in df.columns:
        df["concurrence_found"] = ""
    df["concurrence_found"] = df["concurrence_found"].fillna("")
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

    df_all = df.copy()
    # same proposal under two IDs (resubmission, PDF filed twice, ...)
    dup_pairs, dup_hidden = pd.DataFrame(columns=["id_a", "id_b", "text overlap"]), []
    if desc_path and Path(desc_path).is_dir():
        dup_pairs = find_duplicates(desc_path, tuple(sorted(df["id"])),
                                    Path(desc_path).stat().st_mtime)
    if len(dup_pairs):
        groups = duplicate_groups(dup_pairs)
        with st.sidebar:
            hide = st.checkbox(f"Hide duplicate copies ({sum(len(g) - 1 for g in groups)})", value=True,
                               help="Same description text under different proposal IDs. Keeps the "
                                    "SOLR copy, else the highest (newest) ID; all are listed in the QA tab.")
        if hide:
            solr_ids = set()
            man = Path(desc_path).parent / "manifest.csv"
            if man.is_file():
                m = pd.read_csv(man, dtype=str).fillna("")
                solr_ids = set(m.loc[m.get("status", "ok") == "ok", "id"])
            # keep the SOLR copy if there is one (fuller metadata), else the newest ID
            keep = {max(g, key=lambda i: (i in solr_ids, i)) for g in groups}
            dup_hidden = [i for g in groups for i in g if i not in keep]
            df = df[~df["id"].isin(dup_hidden)].reset_index(drop=True)

    has_flags = df["agency_level"].notna().any()
    has_ffrdc = df["ffrdc"].notna().any()
    has_domain = df["primary_dir"].notna().any()
    has_plan = df["plan_level"].notna().any()

    # ---- which overall score to use -------------------------------------- #
    df["overall_model"] = df["overall"]
    df["overall_rule"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    basis = "Model's"
    if len(dims) >= 2:
        with st.sidebar:
            st.header("Overall score")
            basis = st.radio("Used in charts, filters and tables", ["Model's", "Rule-based"],
                             horizontal=True,
                             help="Model's: the overall score Gemma assigned. Rule-based: computed "
                                  "here from the stored D1-D6 scores (weights and caps below), "
                                  "so it can be changed without rescoring.")
            with st.expander("Rule-based weights and caps"):
                weights = {d: st.number_input(d, 0, 5, RULE_WEIGHTS[i] if i < len(RULE_WEIGHTS) else 1,
                                              key=f"w_{i}") for i, d in enumerate(dims)}
                caps = st.checkbox("Must-have caps", value=True,
                                   help="D1 = 0 → overall ≤ 1; D1 = 1 → ≤ 2; "
                                        "D2 = 0 → ≤ 2; D2 = 1 → ≤ 3")
                st.caption("Overall = weighted mean of D1-D6, rescaled to 0-5 and rounded, "
                           "then capped.")
        df["overall_rule"] = rule_overall(df, dims, weights, caps)
        if basis == "Rule-based":
            df["overall"] = df["overall_rule"]
    other_col = "overall_rule" if basis == "Model's" else "overall_model"

    df["declared"] = df["title"].map(lambda t: "Planning" if PLAN_PREFIX_RE.match(str(t))
                                     else ("" if not str(t).strip() else "Impact/Flagship"))
    df["is_planning"] = (df["declared"] == "Planning") | (df["proposal_type"] == "Planning")
    df["interdisciplinary"] = df["secondary_dirs"].map(bool) & df["primary_dir"].notna()
    op = Path(other_path).expanduser() if other_path else None
    has_other = bool(op and op.is_file())
    if has_other:
        other = load_other_binning(str(op), op.stat().st_mtime)
        df = df.merge(other, on="id", how="left")
        for c in ("other_dir", "other_all", "other_confidence"):
            df[c] = df[c].fillna("")

        def _agree(r):
            if not r["other_dir"] or not r["primary_dir"]:
                return ""
            if r["other_dir"] == r["primary_dir"]:
                return "same primary"
            ours = {r["primary_dir"], *r["secondary_dirs"]}
            return "theirs among ours" if set(r["other_all"].split("; ")) & ours else "no overlap"
        df["binning"] = df.apply(_agree, axis=1)

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
        plan_pick = st.multiselect("Planning scope", PLAN_ORDER) if has_plan else []
        bin_pick = st.multiselect("Other binning vs ours", ["same primary", "theirs among ours", "no overlap"],
                                  help="'theirs among ours': their directorate is one of our secondaries") \
            if has_other else []
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
    if plan_pick:
        f = f[f["plan_level"].isin(plan_pick)]
    if bin_pick:
        f = f[f["binning"].isin(bin_pick)]
    if ffrdc_only:
        f = f[f["ffrdc"] == True]  # noqa: E712
    if query:
        text_cols = ["title", "summary", "overall_just", "agency_just", "ffrdc_just", "field",
                     "domain_just", "plan_just"] + \
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

    tabs = st.tabs(["Overview", "Domains", "Flags", "Planning", "Proposals", "QA", "Rubric"])

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
        if len(dims) >= 2 and f["overall_rule"].notna().any():
            st.subheader("Model's overall vs rule-based overall")
            ct = pd.crosstab(f["overall_model"].rename("model's overall ↓"),
                             f["overall_rule"].rename("rule-based →"))
            st.dataframe(ct, use_container_width=True)
            both = f[f["overall_model"].notna() & f["overall_rule"].notna()]
            gap = (both["overall_model"] - both["overall_rule"]).abs()
            weak = both[((both[dims[0]] <= 1) | (both[dims[1]] <= 1)) & (both["overall_model"] >= 3)]
            st.caption(f"{(gap == 0).sum()} of {len(both)} agree exactly; {(gap >= 2).sum()} differ by "
                       f"2 or more. {len(weak)} have D1 or D2 ≤ 1 but a model overall ≥ 3 (well "
                       "assembled but weak on a must-have); they're listed in the QA tab.")

    # ---- Domains ---------------------------------------------------------- #
    with tabs[1]:
        if not has_domain:
            st.info("These results have no domain fields. Re-run with the updated r05 rubric and schema.")
        else:
            if has_other:
                b = f["binning"][f["binning"] != ""]
                if len(b):
                    st.caption(f"Other binning, {len(b)} proposals in both: same primary "
                               f"{(b == 'same primary').mean():.0%}; their directorate among ours "
                               f"{(b != 'no overlap').mean():.0%}. Use the 'Other binning vs ours' filter "
                               "to list the differences.")
                    ct = pd.crosstab(f.loc[b.index, "other_dir"].rename("theirs"),
                                     f.loc[b.index, "primary_dir"].rename("ours"))
                    with st.expander("Their primary (rows) x our primary (columns)"):
                        st.dataframe(ct, use_container_width=True)
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

    # ---- Planning --------------------------------------------------------- #
    with tabs[3]:
        p = f[f["is_planning"]]
        st.caption(f"{len(p)} planning proposals in view (title starts with \"Planning:\" or the "
                   "model judged the content to be Planning). Flags here are for review only; "
                   "confirm each before returning a proposal without review.")
        if not has_plan:
            st.info("These results have no planning_scope field. Re-run with the r06 rubric and schema "
                    "to see planning vs. preliminary research.")
        else:
            c1, c2 = st.columns([2, 3])
            with c1:
                show(chart_planning(p))
            with c2:
                path_counts = p["plan_path"].value_counts().reindex(["clear", "vague", "absent"], fill_value=0)
                st.markdown("**Says what future Impact/Flagship proposal it prepares**")
                st.dataframe(path_counts.rename_axis("future proposal path").reset_index(name="proposals"),
                             hide_index=True, use_container_width=True)
            concern = p[p["plan_level"].isin(["mostly preliminary research", "mixed"])].sort_values(
                "plan_level", key=lambda s: s.map({"mostly preliminary research": 0, "mixed": 1}))
            st.markdown(f"**{len(concern)} planning proposals with preliminary research** "
                        "(\"mostly preliminary research\" = likely out of scope for Planning)")
            st.dataframe(concern[["id", "title", "plan_level", "plan_research", "plan_path", "overall",
                                  "plan_just"]],
                         hide_index=True, use_container_width=True,
                         column_config={"plan_just": st.column_config.TextColumn("why", width="large"),
                                        "plan_research": st.column_config.ListColumn("research activities"),
                                        "title": st.column_config.TextColumn(width="medium")})

        st.subheader("Planning compliance checks")
        checks = []
        for _, r in f.iterrows():
            t = str(r["title"])
            if r["declared"] == "Planning" and r["proposal_type"] == "Impact/Flagship":
                checks.append((r["id"], t, "title says Planning, content reads as Impact/Flagship", ""))
            if r["declared"] == "Impact/Flagship" and r["proposal_type"] == "Planning":
                checks.append((r["id"], t, "content reads as Planning, title has no \"Planning:\" prefix", ""))
            if not r["is_planning"]:
                continue
            if r["declared"] == "Planning" and not PLAN_PREFIX_EXACT.match(t):
                checks.append((r["id"], t, "title prefix is not exactly \"Planning: AIDatasets:\"", t[:40]))
            amt = r.get("requested $")
            if pd.notna(amt) and amt is not None and amt > PLAN_MAX_DOLLARS:
                checks.append((r["id"], t, f"requests more than ${PLAN_MAX_DOLLARS:,}", f"${amt:,.0f}"))
            mo = r.get("months")
            if pd.notna(mo) and mo is not None and mo > PLAN_MAX_MONTHS:
                checks.append((r["id"], t, f"longer than {PLAN_MAX_MONTHS} months", f"{mo:.0f} months"))
            if r["concurrence_found"] == "no":
                checks.append((r["id"], t, "no concurrence email found in SOLR documents (verify)", ""))
            if r["plan_path"] == "absent":
                checks.append((r["id"], t, "doesn't say what future proposal it prepares", ""))
        if not money_cols:
            st.caption("Budget and duration checks need manifest.csv from the updated "
                       "solr_fetch_descriptions.py (re-run it; it only fills in the metadata).")
        if checks:
            ch = pd.DataFrame(checks, columns=["id", "title", "issue", "detail"])
            st.dataframe(ch.groupby("issue").size().rename("count").reset_index(),
                         hide_index=True, use_container_width=True)
            st.dataframe(ch, hide_index=True, use_container_width=True,
                         column_config={"title": st.column_config.TextColumn(width="large")})
        else:
            st.success("No planning compliance issues found.")
        st.caption(f"Limits checked: ${PLAN_MAX_DOLLARS:,} total, {PLAN_MAX_MONTHS} months, title prefix, "
                   "concurrence email. The 8-page limit isn't checked (SOLR text has no page count).")

    # ---- Proposals -------------------------------------------------------- #
    with tabs[4]:
        cols = ["id", "title"] + extra_cols + money_cols + ["proposal_type"]
        if has_plan:
            cols += ["plan_level"]
        if has_domain:
            cols += ["primary_dir", "secondary_dirs", "field"]
        if has_other:
            cols += ["other_dir", "binning"]
        cols += ["overall"] + ([other_col] if len(dims) >= 2 else []) + dims
        if has_flags:
            cols += ["agency_level", "agencies"]
        if has_ffrdc:
            cols += ["ffrdc_entities"]
        cols = list(dict.fromkeys(c for c in cols if c in f.columns))
        table = f[cols].sort_values("overall", ascending=False)
        cfg = {"overall": st.column_config.NumberColumn(
                   "overall" if len(dims) < 2 else f"overall ({basis.lower()})", format="%d"),
               "overall_rule": st.column_config.NumberColumn("rule-based", format="%d"),
               "overall_model": st.column_config.NumberColumn("model's overall", format="%d"),
               "requested $": st.column_config.NumberColumn("requested $", format="$%d"),
               "plan_level": st.column_config.TextColumn("planning scope"),
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
        for c in ("agencies", "ffrdc_entities", "ffrdc_named", "ffrdc_unrecognized", "secondary_dirs",
                  "plan_research"):
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
            if has_other and r["other_dir"]:
                dom += f" · other binning: {r['other_all']}" + (
                    f" ({r['other_confidence']})" if r["other_confidence"] else "")
            st.caption(f"{pick} · {r['proposal_type']} · overall {r['overall']}{dom}"
                       + (f" · {meta_line}" if meta_line else ""))
            st.write(r["summary"] or "_No summary in this result._")
            with st.expander("Scores and justifications", expanded=False):
                detail = pd.DataFrame(
                    [{"dimension": d, "score": r[d], "justification": r[d + "__just"]} for d in dims]
                    + [{"dimension": "Overall (0–5), model's", "score": r["overall_model"],
                        "justification": r["overall_just"]}]
                    + ([{"dimension": "Overall (0–5), rule-based", "score": r["overall_rule"],
                         "justification": "weighted mean of the scores above, with must-have caps "
                                          "(sidebar)"}] if len(dims) >= 2 else []))
                st.dataframe(detail, hide_index=True, use_container_width=True,
                             column_config={"justification": st.column_config.TextColumn(width="large")})
                if has_domain and r["domain_just"]:
                    st.markdown(f"**Domain:** {r['domain_just']}")
                if has_flags:
                    st.markdown(f"**Other-agency alignment:** {r['agency_level']} "
                                f"{', '.join(r['agencies']) if r['agencies'] else ''} — {r['agency_just']}")
                if has_plan and r["plan_level"]:
                    acts = "; ".join(r["plan_research"]) if r["plan_research"] else ""
                    st.markdown(f"**Planning scope:** {r['plan_level']} (future proposal path: "
                                f"{r['plan_path']}) — {r['plan_just']}" + (f" Research: {acts}" if acts else ""))
                if has_ffrdc:
                    ents = ", ".join(r["ffrdc_entities"]) if r["ffrdc_entities"] else "none"
                    st.markdown(f"**FFRDC:** {ents} ({r['ffrdc_role']}) — {r['ffrdc_just']}")
                    if r["ffrdc_unrecognized"]:
                        st.caption("Named by the model but not on the FFRDC list: "
                                   + ", ".join(r["ffrdc_unrecognized"]))

    # ---- QA --------------------------------------------------------------- #
    with tabs[5]:
        rwr_files = [Path(desc_path).parent / n for n in ("collaborative_rwr.csv",
                                                          "collaborative_rwr_pdf.csv")] if desc_path else []
        rwr_parts = [pd.read_csv(f, dtype=str).fillna("") for f in rwr_files if f.is_file()]
        if rwr_parts:
            rwr = pd.concat(rwr_parts, ignore_index=True).drop_duplicates("id")
            if len(rwr):
                st.subheader(f"Collaborative proposals: {len(rwr)} return-without-review candidates")
                st.caption("Found by solr_fetch_descriptions.py and not scored (collaborative "
                           "proposals aren't allowed in this solicitation). Confirm each before acting.")
                st.dataframe(rwr, hide_index=True, use_container_width=True,
                             column_config={"title": st.column_config.TextColumn(width="large")})
        if len(dup_pairs):
            info = df_all.set_index("id")
            show_cols = [c for c in ("title", "PI", "institution") if c in info.columns]
            d = dup_pairs.copy()
            for side in ("a", "b"):
                for c in show_cols:
                    d[f"{c}_{side}"] = d[f"id_{side}"].map(info[c])
            st.subheader(f"Possible duplicates: {len(d)} pair(s) with largely the same description text")
            st.caption("Different proposal IDs whose descriptions overlap heavily (resubmission, "
                       "same PDF filed twice, ...). Check in eJacket which one is live. "
                       + (f"Hidden from the other tabs: {', '.join(dup_hidden)} (sidebar checkbox)."
                          if dup_hidden else "All copies are shown in the other tabs."))
            st.dataframe(d.sort_values("text overlap", ascending=False), hide_index=True,
                         use_container_width=True)
        st.subheader("Output checks")
        st.markdown("Checks on the model's output. Review these before relying on the numbers.")
        issues = []
        if dims:
            d2 = dims[1] if len(dims) > 1 else None
            for _, r in df.iterrows():
                mo = r["overall_model"] if pd.notna(r["overall_model"]) else 0
                if r[dims[0]] == 0 and mo > 1:
                    issues.append((r["id"], "D1 = 0 but overall > 1 (gatekeeper rule)"))
                if d2 and r[d2] == 0 and mo > 2:
                    issues.append((r["id"], "D2 = 0 but overall > 2 (essential rule)"))
                if re.match(r"(?i)\s*collaborative\s+research\s*:", str(r["title"])):
                    issues.append((r["id"], "scored, but title says Collaborative Research "
                                            "(return-without-review candidate)"))
                if has_domain and not r["primary_dir"]:
                    issues.append((r["id"], "missing primary directorate"))
                if (r[dims[0]] == 1 and mo > 2) or (d2 and r[d2] == 1 and mo > 3):
                    issues.append((r["id"], "D1 or D2 = 1 but model overall above the must-have cap",
                                   f"D1={r[dims[0]]}, D2={r[d2] if d2 else '–'}, overall={mo}"))
                if any(pd.isna(r[d]) for d in dims) or pd.isna(r["overall_model"]):
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
        pdf_man = Path(desc_path).parent / "pdf_manifest.csv" if desc_path else None
        if pdf_man and pdf_man.is_file():
            pm = pd.read_csv(pdf_man, dtype=str).fillna("")
            for _, r in pm[~pm["status"].isin(["ok", "has_solr_text"])].iterrows():
                issues.append((r["id"] or "?", f"PDF not converted: {r['status']}", r["file"]))
        if desc_path and Path(desc_path).is_dir():
            scored = set(df_all["id"])
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

    # ---- Rubric ----------------------------------------------------------- #
    with tabs[6]:
        rp = Path(rubric_path).expanduser() if rubric_path else None
        if not rp or not rp.is_file():
            st.info("No rubric file found. Put its path in the sidebar (e.g. "
                    "`~/Downloads/ai_datasets_screening/_best-r08-rubric.md`).")
        else:
            rv = re.search(r"-r(\d+)-", rp.name)
            fv = re.search(r"r(\d+)", Path(res_path).parent.name)
            if rv and fv and rv.group(1) != fv.group(1):
                st.warning(f"The results folder looks like r{fv.group(1)} but this rubric is "
                           f"r{rv.group(1)}.")
            st.caption(f"`{rp}`")
            text = rp.read_text()
            find = st.text_input("Find in rubric", key="rubric_find")
            if find:
                hits = [ln for ln in text.splitlines() if find.lower() in ln.lower()]
                st.caption(f"{len(hits)} line(s) mention '{find}'")
                for ln in hits:
                    st.markdown("- " + ln.lstrip("-# ").replace("$", "\\$"))
                st.divider()
            st.markdown(text.replace("$", "\\$"))


main()
