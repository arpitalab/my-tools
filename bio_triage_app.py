"""BIO triage: flag each BIO-relevant proposal in the final pool as
    1 = BIO primary interest (accept), 2 = BIO secondary interest, x = not ours.

Starts from final_pool.csv (written by rubric_dashboard.py, Final pool tab), adds the
proposal type and overall score from the r08 results and the provisional call from the
bio-role pass, and pre-fills a recommended flag that you can override. Planning and
Impact/Flagship proposals are on separate tabs.

    streamlit run bio_triage_app.py
    streamlit run bio_triage_app.py -- --pool output/ai_datasets_r08/final_pool.csv \
        --results output/ai_datasets_r08/all.jsonl --bio-role output/bio_role_r02/all.jsonl

Every edit is saved straight away to bio_flags.csv next to final_pool.csv (local;
only overrides and comments are stored, so the rest follow the recommendation). Export
gives every row with its final flag, as CSV or Excel with one sheet per track.
"""

import argparse
import datetime as dt
import io
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

FLAGS = ["1", "2", "x"]
FLAG_HELP = "1 = BIO primary interest (accept) · 2 = BIO secondary interest · x = not ours"
PLAN_PREFIX_RE = re.compile(r"^\s*planning\s*:", re.I)
# words that suggest human subjects; the bio-role rubric should catch these but doesn't always
HUMAN_RE = re.compile(r"(?i)\b(human|humans|child|children|infant|infants|patient|patients|clinical|"
                      r"speech|language acquisition|participants|cohort|hospital|medical)\b")
APP_VERSION = "2026-11-23c (autosave, view stays put)"


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="output/ai_datasets_r08/final_pool.csv")
    ap.add_argument("--results", default="output/ai_datasets_r08/all.jsonl")
    ap.add_argument("--bio-role", default=None, help="default: newest output/bio_role*/all.jsonl")
    a, _ = ap.parse_known_args()
    if a.bio_role is None:
        found = sorted(Path("output").glob("bio_role*/all.jsonl"), key=lambda p: p.stat().st_mtime) \
            if Path("output").is_dir() else []
        a.bio_role = str(found[-1]) if found else ""
    return a


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p)
    return default if d is None else d


def _jsonl(path):
    for line in Path(path).read_text().splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _join(v):
    return "; ".join(map(str, v)) if isinstance(v, list) else (v or "")


@st.cache_data(show_spinner=False)
def load_results(path, mtime):
    rows = [{"id": Path(str(r.get("file", ""))).stem,
             "model_type": r.get("proposal_type") or "",
             "overall": _get(r, "overall_score", "score"),
             "field": _get(r, "domain", "field", default="")} for r in _jsonl(path)]
    return pd.DataFrame(rows, columns=["id", "model_type", "overall", "field"]).drop_duplicates("id", keep="last")


@st.cache_data(show_spinner=False)
def load_bio_role(path, mtime):
    rows = [{"id": Path(str(r.get("file", ""))).stem,
             "prov_call": _get(r, "provisional_call", "call", default=""),
             "prov_summary": _get(r, "provisional_call", "summary", default=""),
             "bio_role": _get(r, "bio_role", "level", default=""),
             "bio_theme": _get(r, "bio_theme", "theme", default=""),
             "beneficiary": _get(r, "beneficiary", "primary", default=""),
             "human_clinical": _get(r, "human_clinical", "level", default=""),
             "bio_questions": _join(r.get("biological_questions")),
             "organisms": _join(_get(r, "biological_data", "organisms_or_systems"))} for r in _jsonl(path)]
    cols = ["id", "prov_call", "prov_summary", "bio_role", "bio_theme", "beneficiary", "human_clinical",
            "bio_questions", "organisms"]
    return pd.DataFrame(rows, columns=cols).drop_duplicates("id", keep="last")


def norm_call(c: str) -> str:
    """The model sometimes misspells the enum (e.g. 'handoff_biomed')."""
    c = (c or "").lower()
    if "biomed" in c or "hand" in c:
        return "hand_off_biomedical"
    if "keep" in c:
        return "keep_bio"
    if "use_case" in c or "use case" in c:
        return "bio_as_use_case"
    return "unclear" if c else ""


def has_bio(s: str) -> bool:
    return bool(re.search(r"\bBIO\b", s or ""))


def recommend(r) -> tuple[str, str]:
    """(flag, why). Rules are applied in order; the first that matches wins."""
    final, basis, call = r["final_directorate"], r["basis"], r["prov_call"]
    human = r["human_clinical"]
    words = HUMAN_RE.findall(" ".join(str(r.get(c, "")) for c in ("title", "organisms", "bio_questions")))
    if basis == "decided":
        if final == "BIO":
            return "1", "you chose BIO"
        return "x", f"you chose {final or 'not BIO'}"
    if final in ("Hand off", "Out of scope"):
        return "x", final.lower()
    if call == "hand_off_biomedical" or human == "central":
        return "x", "biomedical / clinical (bio-role pass)"
    if final != "BIO":
        bio_secondary = has_bio(r["ours"]) or has_bio(r["theirs"])
        if bio_secondary and call in ("keep_bio", "unclear", ""):
            return "2", f"lead {final or '?'}, BIO also named; biology looks core" if call == "keep_bio" \
                else f"lead {final or '?'}, BIO also named"
        return "x", f"lead {final or '?'}" + ("; biology is a use case" if call == "bio_as_use_case" else "")
    # final directorate is BIO
    if human == "incidental" or words:
        return "2", "human subjects? (" + ", ".join(sorted({w.lower() for w in words})[:3]) + ")" \
            if words else "human data incidental: check"
    if call == "keep_bio":
        return "1", "BIO, biology is core" + (" (both binnings)" if has_bio(r["theirs"]) and
                                                 r["ours"].startswith("BIO") else "")
    if call == "bio_as_use_case":
        return "2", "BIO, but biology is a use case"
    return "2", "BIO, no clear provisional call" if call else "BIO, no bio-role result"


def track(r) -> str:
    if PLAN_PREFIX_RE.match(str(r["title"])):
        return "Planning"
    t = r.get("model_type") or ""
    return t if t in ("Planning", "Impact/Flagship") else ("Impact/Flagship" if str(r["title"]).strip()
                                                           else "Unclear")


def load_flags(path: Path) -> pd.DataFrame:
    if path.is_file():
        f = pd.read_csv(path, dtype=str).fillna("")
        return f.reindex(columns=["id", "my_flag", "comment", "saved"], fill_value="").drop_duplicates("id", keep="last")
    return pd.DataFrame(columns=["id", "my_flag", "comment", "saved"])


def write_flags(path: Path, flags: pd.DataFrame):
    """Atomic write, plus a backup of the previous file once per session."""
    if path.is_file() and not st.session_state.get("_backed_up"):
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        (path.parent / f"bio_flags.backup-{stamp}.csv").write_bytes(path.read_bytes())
        st.session_state["_backed_up"] = True
    tmp = path.with_suffix(".tmp")
    flags.to_csv(tmp, index=False)
    tmp.replace(path)


def autosave(key: str, ids: list, rec: dict, path: Path):
    """on_change of a table: write its edits to bio_flags.csv straight away.
    Streamlit keeps a table's edits as {row position: {column: value}}; `ids` maps
    positions back to proposal IDs for the rows that table showed."""
    edits = (st.session_state.get(key) or {}).get("edited_rows", {})
    cur = load_flags(path).set_index("id")
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = 0
    for pos, ch in edits.items():
        pid = ids[int(pos)]
        mine, com = (cur.loc[pid, "my_flag"], cur.loc[pid, "comment"]) if pid in cur.index else ("", "")
        if "flag" in ch:
            mine = "" if (ch["flag"] or rec[pid]) == rec[pid] else ch["flag"]
        if "comment" in ch:
            com = ch["comment"] or ""
        cur.loc[pid, ["my_flag", "comment", "saved"]] = [mine, com, now]
        n += 1
    cur = cur.fillna("")
    cur = cur[(cur["my_flag"] != "") | (cur["comment"] != "")]
    write_flags(path, cur.reset_index())
    st.session_state["_last_save"] = f"{now[-8:]} ({len(cur)} proposals with a flag override or comment)"


SHOW = ["flag", "recommended", "why", "comment", "id", "title", "PI", "institution", "final_directorate",
        "basis", "ours", "theirs", "prov_call", "bio_role", "bio_theme", "beneficiary", "human_clinical",
        "overall", "organisms", "bio_questions", "prov_summary", "note", "field"]


def main():
    st.set_page_config(page_title="BIO triage", layout="wide")
    a = cli()
    with st.sidebar:
        st.header("Data")
        pool_path = st.text_input("Final pool (from the dashboard)", a.pool)
        res_path = st.text_input("r08 results (for proposal type, overall)", a.results)
        br_path = st.text_input("Bio-role results", a.bio_role)
        st.caption(f"App version {APP_VERSION}")
    pp = Path(pool_path).expanduser()
    if not pp.is_file():
        st.error(f"`{pool_path}` not found. In rubric_dashboard.py → Final pool, click "
                 "'Write final_pool.csv next to the results' first.")
        st.stop()
    pool = pd.read_csv(pp, dtype=str).fillna("")
    for c in ("title", "final_directorate", "basis", "ours", "theirs", "note", "prov_call"):
        if c not in pool.columns:
            pool[c] = ""

    rp = Path(res_path).expanduser()
    if rp.is_file():
        pool = pool.merge(load_results(str(rp), rp.stat().st_mtime), on="id", how="left")
    bp = Path(br_path).expanduser() if br_path else None
    if bp and bp.is_file():
        br = load_bio_role(str(bp), bp.stat().st_mtime)
        pool = pool.drop(columns=[c for c in br.columns if c != "id" and c in pool.columns]).merge(br, on="id", how="left")
    else:
        st.sidebar.warning("No bio-role results: recommendations use the directorates only.")
    for c in [c for c in SHOW if c not in ("flag", "recommended", "why", "comment")] + ["model_type"]:
        if c not in pool.columns:
            pool[c] = ""
    pool = pool.fillna("")
    pool["prov_call"] = pool["prov_call"].map(norm_call)

    # BIO-relevant rows only: BIO in the final answer, our binning, theirs, or a bio-role result
    rel = (pool["final_directorate"] == "BIO") | pool["ours"].map(has_bio) | pool["theirs"].map(has_bio) | \
        pool["prov_call"].ne("")
    with st.sidebar:
        st.header("Rows")
        everything = st.checkbox("Include proposals with no BIO involvement", value=False)
    d = (pool if everything else pool[rel]).copy()
    rec = d.apply(recommend, axis=1)
    d["recommended"], d["why"] = rec.map(lambda t: t[0]), rec.map(lambda t: t[1])
    d["track"] = d.apply(track, axis=1)

    flags_path = pp.parent / "bio_flags.csv"
    saved = load_flags(flags_path)
    d = d.merge(saved, on="id", how="left").fillna("")
    d["flag"] = d["my_flag"].where(d["my_flag"].isin(FLAGS), d["recommended"])
    d["overall"] = pd.to_numeric(d["overall"], errors="coerce")
    order = {"1": 0, "2": 1, "x": 2}
    d = d.sort_values(["recommended", "overall"], key=lambda s: s.map(order) if s.name == "recommended" else -s,
                      na_position="last").reset_index(drop=True)

    st.title("BIO triage")
    st.caption(f"{FLAG_HELP}. `flag` starts at the recommendation; change it where you disagree and add a "
               f"comment if useful. **Every edit is saved as soon as you leave the cell** (Enter, Tab or "
               f"click elsewhere) to `{flags_path}`.")
    with st.sidebar:
        st.header("Filter")
        show_flag = st.multiselect("Current flag", FLAGS)
        only_over = st.checkbox("Only ones I've overridden")
        q = st.text_input("Search title / PI / notes")
    f = d
    if show_flag:
        f = f[f["flag"].isin(show_flag)]
    if only_over:
        f = f[f["my_flag"].isin(FLAGS) & (f["my_flag"] != f["recommended"])]
    if q:
        hay = f[["title", "PI", "institution", "note", "comment", "why", "organisms", "bio_questions",
                 "prov_summary", "field"]].astype(str).agg(" ".join, axis=1)
        f = f[hay.str.contains(q, case=False, regex=False)]

    tracks = [t for t in ("Planning", "Impact/Flagship", "Unclear") if (d["track"] == t).any()]
    tabs = st.tabs(tracks + ["Summary"])
    # Each table is drawn from a copy of its rows frozen when the view was opened, so
    # an autosave doesn't change the table's data (which would redraw it and jump back
    # to the top). Changing a filter opens a new view built from the saved file; the
    # old views are dropped so their stale edits can't be written again.
    filt_key = f"{sorted(show_flag)}_{only_over}_{q}_{everything}_{flags_path}"
    if st.session_state.get("_view") != filt_key:
        for k in [k for k in st.session_state if str(k).startswith(("ed_", "base_"))]:
            del st.session_state[k]
        st.session_state["_view"] = filt_key
    rec_map = d.set_index("id")["recommended"].to_dict()
    for t, tab in zip(tracks, tabs):
        with tab:
            w = f[f["track"] == t]
            all_t = d[d["track"] == t]
            k = st.columns(4)
            for i, fl in enumerate(FLAGS):
                k[i].metric({"1": "1 · primary", "2": "2 · secondary", "x": "x · not ours"}[fl],
                            int((all_t["flag"] == fl).sum()))
            k[3].metric("Overridden by you", int((all_t["my_flag"].isin(FLAGS) &
                                                 (all_t["my_flag"] != all_t["recommended"])).sum()))
            st.caption(f"{len(w)} of {len(all_t)} {t} proposals match the filters. Rows stay put while you "
                       "edit; change a filter (or reload the page) to re-sort and re-filter.")
            key = f"ed_{t}"
            if "base_" + key not in st.session_state:
                st.session_state["base_" + key] = w[SHOW].reset_index(drop=True)
            base = st.session_state["base_" + key]
            st.data_editor(
                base, key=key, hide_index=True, use_container_width=True,
                on_change=autosave, args=(key, base["id"].tolist(), rec_map, flags_path),
                height=min(700, 38 * (len(base) + 1)),
                disabled=[c for c in SHOW if c not in ("flag", "comment")],
                column_config={
                    "flag": st.column_config.SelectboxColumn("flag", options=FLAGS, required=True, width="small",
                                                             help=FLAG_HELP),
                    "recommended": st.column_config.TextColumn("rec.", width="small"),
                    "why": st.column_config.TextColumn("why recommended", width="medium"),
                    "comment": st.column_config.TextColumn("comment", width="medium"),
                    "title": st.column_config.TextColumn(width="large"),
                    "overall": st.column_config.NumberColumn("overall", format="%d"),
                    "final_directorate": st.column_config.TextColumn("final dir"),
                    "prov_call": st.column_config.TextColumn("provisional"),
                    "bio_questions": st.column_config.TextColumn("bio questions", width="large"),
                    "prov_summary": st.column_config.TextColumn("provisional why", width="large"),
                })
    with tabs[-1]:
        st.subheader("Flags by track")
        st.dataframe(pd.crosstab(d["track"], d["flag"], margins=True, margins_name="total"),
                     use_container_width=True)
        st.subheader("Your flag vs the recommendation")
        st.dataframe(pd.crosstab(d["recommended"].rename("recommended ↓"), d["flag"].rename("flag →")),
                     use_container_width=True)
        st.subheader("Why recommended")
        st.dataframe(d.groupby(["recommended", "why"]).size().rename("n").reset_index()
                     .sort_values(["recommended", "n"], ascending=[True, False]),
                     hide_index=True, use_container_width=True)

    with st.sidebar:
        st.header("Saving and export")
        if st.session_state.get("_last_save"):
            st.success("Saved " + st.session_state["_last_save"])
        else:
            st.caption(f"Edits save automatically. {len(saved)} overrides/comments on file.")
        st.caption("A backup of the previous bio_flags.csv is kept once per session "
                   "(bio_flags.backup-<time>.csv).")
        export = d[["track", "flag", "recommended", "why", "comment"] +
                   [c for c in SHOW if c not in ("flag", "recommended", "why", "comment")]]
        st.download_button("Download all (CSV)", export.to_csv(index=False), "bio_triage.csv", "text/csv")
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf) as xw:
                for t in tracks:
                    export[export["track"] == t].to_excel(xw, sheet_name=t.replace("/", "-"), index=False)
            st.download_button("Download all (Excel, one sheet per track)", buf.getvalue(), "bio_triage.xlsx")
        except ImportError:
            st.caption("Excel export needs openpyxl; CSV works without it.")


main()
