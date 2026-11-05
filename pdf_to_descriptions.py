"""Turn a folder of project-description PDFs into scoring-ready text files.

For proposals that SOLR doesn't have yet: PDFs named with the proposal ID (any
filename containing one 7-digit ID, e.g. 2703928.pdf or PD_2703928_v2.pdf) are
converted to descriptions/<id>.txt in the same folder solr_fetch_descriptions.py
uses, so pdf_batch_extract.py and rubric_dashboard.py treat them like SOLR text.

Titles, PI, institution and requested amount come from your ID spreadsheet
(--metadata). Without titles, collaborative proposals can't be detected.

Usage:
    python pdf_to_descriptions.py pd_pdfs/ --metadata ids.xlsx
    python pdf_to_descriptions.py pd_pdfs/ --metadata ids.xlsx --title-col "Proposal Title"

    # then score everything (SOLR text + converted PDFs) in one go
    python pdf_batch_extract.py solr_descriptions/descriptions --out output/ai_datasets_r06 ...

Keep the PDFs in their own folder (not inside descriptions/), so the scorer
doesn't see both 123.pdf and 123.txt.

Re-running is safe: PDFs already converted are skipped (--force redoes them),
and an ID that already has SOLR text keeps it.

Output (--out, default ./solr_descriptions):
    descriptions/<id>.txt          one per proposal, first line "Proposal <id>: <title>"
    collaborative/<id>.txt         collaborative proposals, set aside (see --collab)
    collaborative_rwr_pdf.csv      collaborative proposals from these PDFs
    pdf_manifest.csv               id, file, title, status, pages, chars, PI, institution, ...

Needs: pip install pymupdf pandas openpyxl   (pypdf works as a fallback)
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ID_IN_NAME = re.compile(r"(?<!\d)(\d{7})(?!\d)")
_COLLAB_RE = re.compile(r"(?i)^collaborative\s+research\s*:\s*")
MIN_CHARS = 500  # less than this from a whole project description = scanned/empty PDF

# header words used to guess spreadsheet columns when --*-col isn't given
GUESS = {
    "title": r"title",
    "pi_name": r"\bpi\b|principal|investigator",
    "inst": r"inst|organi[sz]ation|awardee|university",
    "requested_amount": r"amount|request|budget|\$",
}


def read_pdf(path: Path) -> tuple[str, int]:
    try:
        import fitz  # pymupdf

        with fitz.open(path) as doc:
            return "\n".join(p.get_text() for p in doc), len(doc)
    except ImportError:
        from pypdf import PdfReader

        r = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in r.pages), len(r.pages)


def load_metadata(path: Path, cols: dict) -> tuple[dict, dict]:
    """{id: {title, pi_name, inst, requested_amount}} and the columns used."""
    import pandas as pd

    df = (pd.read_excel(path, dtype=str) if path.suffix.lower() in (".xlsx", ".xls")
          else pd.read_csv(path, dtype=str)).fillna("")
    is_id = lambda v: bool(re.fullmatch(r"\d{7}(\.0+)?", str(v).strip()))
    id_col = max(df.columns, key=lambda c: df[c].map(is_id).sum())
    if not df[id_col].map(is_id).any():
        sys.exit(f"no 7-digit proposal IDs found in {path}")

    used = {}
    for key, pat in GUESS.items():
        given = cols.get(key)
        if given:
            if given in df.columns:
                used[key] = given
            elif given.isdigit() and int(given) < len(df.columns):
                used[key] = df.columns[int(given)]
            else:
                sys.exit(f"--{key.replace('_name', '').replace('requested_', '')}-col {given!r} "
                         f"not found; columns are: {', '.join(map(str, df.columns))}")
        else:
            hits = [c for c in df.columns if c != id_col and re.search(pat, str(c), re.I)]
            if hits:
                used[key] = hits[0]
    meta = {}
    for _, row in df[df[id_col].map(is_id)].iterrows():
        pid = str(row[id_col]).strip().split(".")[0]
        meta[pid] = {k: str(row[c]).strip() for k, c in used.items()}
    return meta, used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf_dir", help="folder of project-description PDFs (searched recursively)")
    ap.add_argument("--out", default="solr_descriptions")
    ap.add_argument("--metadata", help="ID spreadsheet (.xlsx/.csv) with titles, PI, institution")
    ap.add_argument("--title-col", help="title column (header name or 0-based index); default: guess")
    ap.add_argument("--pi-col", help="PI column; default: guess")
    ap.add_argument("--inst-col", help="institution column; default: guess")
    ap.add_argument("--amount-col", help="requested amount column; default: guess")
    ap.add_argument("--collab", choices=["rwr", "dedup", "keep"], default="rwr",
                    help="collaborative proposals: rwr = set every copy aside (default); "
                         "dedup = score one copy per project; keep = score all")
    ap.add_argument("--force", action="store_true", help="re-convert PDFs already converted")
    args = ap.parse_args()

    out = Path(args.out)
    desc_dir, collab_dir = out / "descriptions", out / "collaborative"
    desc_dir.mkdir(parents=True, exist_ok=True)
    collab_dir.mkdir(exist_ok=True)
    manifest_path = out / "pdf_manifest.csv"

    meta, used = {}, {}
    if args.metadata:
        meta, used = load_metadata(Path(args.metadata), {
            "title": args.title_col, "pi_name": args.pi_col, "inst": args.inst_col,
            "requested_amount": args.amount_col})
        print(f"metadata: {len(meta)} IDs from {args.metadata}; columns used: "
              + (", ".join(f"{k} = {c!r}" for k, c in used.items()) or "none recognized"))
    if "title" not in used:
        print("! no titles: collaborative proposals and \"Planning:\" title prefixes can't be "
              "checked. Give --metadata (and --title-col if the header isn't 'title').",
              file=sys.stderr)

    # IDs this script converted before (vs. .txt files that came from SOLR)
    prev = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            prev = {r["id"]: r for r in csv.DictReader(f) if r["status"] == "ok"}

    pdfs = sorted(p for p in Path(args.pdf_dir).rglob("*") if p.suffix.lower() == ".pdf")
    if not pdfs:
        sys.exit(f"no PDFs found in {args.pdf_dir}")
    print(f"{len(pdfs)} PDFs in {args.pdf_dir}")

    rows, seen = {}, {}
    for i, pdf in enumerate(pdfs, 1):
        found = sorted(set(ID_IN_NAME.findall(pdf.stem)))
        if len(found) != 1:
            status = "no_id_in_name" if not found else "several_ids_in_name"
            rows[pdf.name] = {"id": "", "file": pdf.name, "status": status}
            continue
        pid = found[0]
        m = meta.get(pid, {})
        rec = {"id": pid, "file": pdf.name, "title": m.get("title", ""),
               "pi_name": m.get("pi_name", ""), "inst": m.get("inst", ""),
               "requested_amount": m.get("requested_amount", "")}
        if pid in seen:
            rows[pdf.name] = {**rec, "status": f"duplicate_of_{seen[pid]}"}
            continue
        seen[pid] = pdf.name
        have = (desc_dir / f"{pid}.txt").exists() or (collab_dir / f"{pid}.txt").exists()
        if have and prev.get(pid, {}).get("status") != "ok":
            rows[pdf.name] = {**rec, "status": "has_solr_text"}  # not ours: keep SOLR's
            continue
        if have and not args.force and prev.get(pid, {}).get("status") == "ok":
            rows[pdf.name] = {**prev[pid], **{k: v for k, v in rec.items() if v}}
            continue
        try:
            text, pages = read_pdf(pdf)
        except Exception as e:
            rows[pdf.name] = {**rec, "status": "unreadable", "detail": str(e)[:200]}
            continue
        text = text.strip()
        rec.update(pages=pages, chars=len(text))
        if len(text) < MIN_CHARS:
            rows[pdf.name] = {**rec, "status": "no_text"}  # scanned? needs OCR
            continue
        (desc_dir / f"{pid}.txt").write_text(f"Proposal {pid}: {rec['title']}\n\n{text}\n")
        rows[pdf.name] = {**rec, "status": "ok"}
        if i % 100 == 0:
            print(f"  {i}/{len(pdfs)}")

    # ---- collaborative proposals (same rule as solr_fetch_descriptions.py) ---
    ok = [r for r in rows.values() if r["status"] == "ok"]
    solr_titles = {}
    if (out / "manifest.csv").exists():
        with open(out / "manifest.csv") as f:
            solr_titles = {r["id"]: r.get("title", "") for r in csv.DictReader(f)
                           if r.get("status") == "ok"}
    groups: dict[str, list[dict]] = {}
    for r in ok:
        if r["title"]:
            groups.setdefault(_COLLAB_RE.sub("", r["title"]).strip().lower(), []).append(r)
    solr_by_key: dict[str, list[str]] = {}
    for sid, t in solr_titles.items():
        if t:
            solr_by_key.setdefault(_COLLAB_RE.sub("", t).strip().lower(), []).append(sid)
    rwr = []
    for r in ok:
        r["collaborative"], r["collab_group"] = "", ""
    for key, g in groups.items():
        lead = max(g, key=lambda r: int(r.get("chars") or 0))
        partners_solr = [s for s in solr_by_key.get(key, []) if s not in {r["id"] for r in g}]
        for r in g:
            prefixed = bool(_COLLAB_RE.match(r["title"]))
            others = [x["id"] for x in g if x is not r] + partners_solr
            if not (prefixed or others):
                continue
            r["collaborative"], r["collab_group"] = "yes", lead["id"] if len(g) > 1 else ""
            reasons = (["'Collaborative Research:' title"] if prefixed else []) + \
                      ([f"same title as {', '.join(others)}"] if others else [])
            rwr.append({"id": r["id"], "title": r["title"], "collab_group": r["collab_group"],
                        "reason": "; ".join(reasons)})
            dst = desc_dir if args.collab == "keep" or (args.collab == "dedup" and r is lead) \
                else collab_dir
            for src in (desc_dir, collab_dir):
                f = src / f"{r['id']}.txt"
                if f.exists() and src != dst:
                    f.replace(dst / f.name)
    # an ID that is no longer collaborative (e.g. title fixed) goes back to descriptions/
    for r in ok:
        if not r.get("collaborative") and (collab_dir / f"{r['id']}.txt").exists():
            (collab_dir / f"{r['id']}.txt").replace(desc_dir / f"{r['id']}.txt")

    cols = ["id", "file", "title", "status", "pages", "chars", "collaborative", "collab_group",
            "pi_name", "inst", "requested_amount", "detail"]
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows.values())
    with open(out / "collaborative_rwr_pdf.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "collab_group", "reason"])
        w.writeheader()
        w.writerows(rwr)

    st = Counter(r["status"] if not r["status"].startswith("duplicate_of") else "duplicate"
                 for r in rows.values())
    why = {"ok": "converted (or already converted)",
           "has_solr_text": "SOLR text already downloaded; kept that",
           "no_text": "almost no extractable text (scanned? needs OCR)",
           "no_id_in_name": "no 7-digit ID in the filename",
           "several_ids_in_name": "more than one 7-digit ID in the filename",
           "duplicate": "another PDF has the same ID",
           "unreadable": "PDF couldn't be opened"}
    print("\nPDFs by result:")
    for k, n in st.most_common():
        print(f"    {n:5d}  {k:20s} {why.get(k, '')}")
    no_meta = [r["id"] for r in ok if r["id"] not in meta] if meta else []
    if no_meta:
        print(f"! {len(no_meta)} converted IDs aren't in the metadata spreadsheet "
              f"(e.g. {', '.join(no_meta[:5])}); they have no title/PI", file=sys.stderr)
    if rwr:
        verb = {"rwr": "set aside, not scored", "dedup": "one copy per project kept",
                "keep": "all kept for scoring"}[args.collab]
        print(f"{len(rwr)} collaborative proposal(s) found ({verb}); "
              f"listed in {out / 'collaborative_rwr_pdf.csv'}")
    n_desc = len(list(desc_dir.glob("*.txt")))
    print(f"{n_desc} descriptions ready to score in {desc_dir}; details in {manifest_path}")


if __name__ == "__main__":
    main()
