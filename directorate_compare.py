"""Compare another directorate binning (e.g. an Excel sheet from a Copilot exercise)
with the directorates in our rubric scoring.

    # see what it finds in each sheet (columns, which ones it picked)
    python directorate_compare.py theirs.xlsx --dry-run

    # compare (several files/sheets are merged on proposal ID)
    python directorate_compare.py theirs1.xlsx theirs2.xlsx \
        --ours output/ai_datasets_r08 output/ai_datasets_r07 --out output/dir_compare

Columns are found automatically: the ID column is the one with the most 7-digit
proposal IDs; the directorate column is one named "directorate", else the one whose
values most often look like directorate codes/names; every column with "keyword"
(or "term"/"topic") in its name is kept (e.g. PI keywords and Copilot keywords);
columns named "division" and "confidence" are carried along, and agreement is
reported by confidence level. Override with --id-col / --dir-col / --kw-col
(repeatable) / --conf-col, and pick a sheet with --sheet (default: every sheet).

Their cell may list several directorates ("CISE; MPS"): the first is taken as
their primary. Ours: domain.primary_directorate and secondary_directorates from
the scored .json files; with several --ours folders, the first that has a
proposal wins (e.g. r08, then r07 for proposals r08 hasn't reached).

Writes to --out: joined.csv (every matched proposal), disagreements.csv, and
confusion.csv (their primary x our primary). Keep it local: it has titles.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

DIRS = ["BIO", "CISE", "EDU", "ENG", "GEO", "MPS", "SBE", "TIP"]
# directorate names and old names -> code (lower-case keys, matched as substrings)
NAMES = {
    "biological": "BIO", "biology": "BIO",
    "computer and information": "CISE", "computing": "CISE",
    "stem education": "EDU", "education and human": "EDU", "education": "EDU",
    "engineering": "ENG",
    "geosciences": "GEO", "geoscience": "GEO",
    "mathematical and physical": "MPS", "physical sciences": "MPS",
    "social, behavioral": "SBE", "social behavioral": "SBE", "social": "SBE",
    "technology, innovation": "TIP", "technology innovation": "TIP", "partnerships": "TIP",
}
# division codes -> directorate (kept in step with rubric_dashboard.DIVISION_TO_DIR)
DIVISIONS = {
    **dict.fromkeys(["DBI", "DEB", "IOS", "MCB", "EF"], "BIO"),
    **dict.fromkeys(["CCF", "CNS", "IIS", "OAC"], "CISE"),
    **dict.fromkeys(["EHR", "DGE", "DRL", "DUE", "EES", "HRD"], "EDU"),
    **dict.fromkeys(["CBET", "CMMI", "ECCS", "EEC", "EFMA"], "ENG"),
    **dict.fromkeys(["AGS", "EAR", "OCE", "OPP", "RISE"], "GEO"),
    **dict.fromkeys(["AST", "CHE", "DMR", "DMS", "PHY", "OMA"], "MPS"),
    **dict.fromkeys(["BCS", "SES", "NCSES", "SMA"], "SBE"),
    **dict.fromkeys(["ITE", "TF"], "TIP"),
    "EHR": "EDU",
}


def to_dirs(v) -> list[str]:
    """'CISE; MPS' / 'Mathematical and Physical Sciences (MPS)' / 'AST' -> ['CISE', 'MPS'] ..."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    out = []
    for part in re.split(r"[;,/|\n]+|\band\b(?=\s+[A-Z]{3,5}\b)", str(v)):
        s = part.strip()
        if not s:
            continue
        codes = [c for c in re.findall(r"\b[A-Z]{2,5}\b", s) if c in DIRS or c in DIVISIONS]
        if codes:
            d = codes[0] if codes[0] in DIRS else DIVISIONS[codes[0]]
        else:
            low = s.lower()
            d = next((code for name, code in NAMES.items() if name in low), None)
        if d and d not in out:
            out.append(d)
    return out


def clean_id(v):
    s = str(v).strip()
    if re.fullmatch(r"\d{7}\.0+", s):
        s = s.split(".")[0]
    return s if re.fullmatch(r"\d{7}", s) else None


def pick_cols(df, id_col, dir_col, kw_cols, conf_col):
    def score(col, f):
        vals = df[col].dropna().astype(str).head(500)
        return sum(bool(f(v)) for v in vals)
    named = lambda pat: [c for c in df.columns if re.search(pat, str(c), re.I)]
    idc = id_col or max(df.columns, key=lambda c: score(c, clean_id))
    if score(idc, clean_id) == 0:
        return None, None, [], None, None
    rest = [c for c in df.columns if c != idc]
    dc = dir_col or next(iter(named(r"directorate")), None)
    if dc is None and rest:
        dc = max(rest, key=lambda c: score(c, to_dirs))
        if score(dc, to_dirs) == 0:
            dc = None
    kcs = kw_cols or named(r"keyword|term|topic")
    divc = next((c for c in named(r"division") if c != dc), None)
    cc = conf_col or next(iter(named(r"confiden")), None)
    return idc, dc, kcs, divc, cc


def load_theirs(paths, sheet, id_col, dir_col, kw_cols, conf_col, dry):
    rows = {}
    for path in paths:
        sheets = pd.read_excel(path, sheet_name=sheet if sheet else None, dtype=str) \
            if Path(path).suffix.lower() in (".xlsx", ".xls") else {"csv": pd.read_csv(path, dtype=str)}
        if isinstance(sheets, pd.DataFrame):
            sheets = {sheet: sheets}
        for name, df in sheets.items():
            idc, dc, kcs, divc, cc = pick_cols(df, id_col, dir_col, kw_cols, conf_col)
            print(f"{Path(path).name} [{name}]: {len(df)} rows; columns: {list(df.columns)}")
            print(f"    using id={idc!r} directorate={dc!r} keywords={kcs!r} division={divc!r} "
                  f"confidence={cc!r}")
            if dry or idc is None:
                if idc is None:
                    print("    (no column of 7-digit IDs; skipped)")
                continue
            if dc is not None:
                vals = Counter(v for x in df[dc] for v in (to_dirs(x)[:1] or ["(unrecognized)"]))
                print("    their primaries: " + ", ".join(f"{k} {n}" for k, n in vals.most_common()))
            for _, r in df.iterrows():
                pid = clean_id(r[idc])
                if not pid:
                    continue
                e = rows.setdefault(pid, {"their_dirs": [], "their_raw": "", "their_division": "",
                                          "their_confidence": "", "keywords": {}})
                val = lambda c: "" if c is None or pd.isna(r[c]) else str(r[c]).strip()
                if dc is not None and not e["their_dirs"]:
                    e["their_dirs"] = to_dirs(r[dc])
                    e["their_raw"] = val(dc)
                for kc in kcs:
                    if val(kc):
                        e["keywords"].setdefault(str(kc), val(kc))
                e["their_division"] = e["their_division"] or val(divc)
                e["their_confidence"] = e["their_confidence"] or val(cc)
    return rows


def load_ours(folders):
    ours = {}
    for folder in folders:
        n = 0
        for f in Path(folder).glob("*.json"):
            if f.stem in ours:
                continue
            try:
                r = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            dom = r.get("domain") or {}
            prim = to_dirs(dom.get("primary_directorate"))
            sec = [d for x in (dom.get("secondary_directorates") or []) for d in to_dirs(x)]
            if not prim:
                continue
            ours[f.stem] = {"our_primary": prim[0], "our_secondary": [d for d in sec if d != prim[0]],
                            "our_field": dom.get("field", ""), "title": r.get("title", ""),
                            "our_run": Path(folder).name}
            n += 1
        print(f"ours: {n} proposals from {folder}")
    return ours


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return None if pe == 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("theirs", nargs="+", help="their .xlsx/.csv file(s)")
    ap.add_argument("--ours", nargs="+", default=["output/ai_datasets_r08", "output/ai_datasets_r07"])
    ap.add_argument("--sheet", help="sheet name (default: all sheets)")
    ap.add_argument("--id-col")
    ap.add_argument("--dir-col")
    ap.add_argument("--kw-col", action="append", help="keyword column (repeatable)")
    ap.add_argument("--conf-col")
    ap.add_argument("--out", default="output/dir_compare")
    ap.add_argument("--dry-run", action="store_true", help="only show which columns would be used")
    args = ap.parse_args()

    theirs = load_theirs(args.theirs, args.sheet, args.id_col, args.dir_col, args.kw_col, args.conf_col,
                         args.dry_run)
    if args.dry_run:
        return
    ours = load_ours([f for f in args.ours if Path(f).is_dir()])
    both = sorted(set(theirs) & set(ours))
    print(f"\ntheirs: {len(theirs)} IDs; ours: {len(ours)}; in both: {len(both)}"
          f" (theirs only: {len(set(theirs) - set(ours))}, ours only: {len(set(ours) - set(theirs))})")
    rows = []
    for pid in both:
        t, o = theirs[pid], ours[pid]
        tp = t["their_dirs"][0] if t["their_dirs"] else ""
        rows.append({"id": pid, "title": o["title"], "their_primary": tp,
                     "their_all": "; ".join(t["their_dirs"]), "their_raw": t["their_raw"],
                     "their_division": t["their_division"], "their_confidence": t["their_confidence"],
                     "our_primary": o["our_primary"], "our_secondary": "; ".join(o["our_secondary"]),
                     "same_primary": tp == o["our_primary"],
                     "overlap": bool(set(t["their_dirs"]) & ({o["our_primary"]} | set(o["our_secondary"]))),
                     "our_field": o["our_field"], **t["keywords"], "our_run": o["our_run"]})
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no proposals in both; check the ID column (--dry-run) and the --ours folders")
    lab = df[df.their_primary != ""]
    k = kappa(list(lab.their_primary), list(lab.our_primary))
    print(f"\n{len(lab)} with a recognizable directorate on their side")
    print(f"same primary:            {lab.same_primary.mean():.0%}   (Cohen's kappa {k:.2f})" if k is not None else "")
    print(f"theirs among ours (primary or secondary): {lab.overlap.mean():.0%}")
    if lab.their_confidence.ne("").any():
        c = lab.their_confidence
        num = pd.to_numeric(c.str.rstrip("%"), errors="coerce")
        if num.notna().mean() > 0.8 and num.nunique() > 5:  # numeric: split into thirds
            c = pd.qcut(num, 3, labels=["low third", "middle third", "high third"], duplicates="drop")
        by = lab.groupby(c.astype(str), observed=True).agg(n=("same_primary", "size"),
                                                          same_primary=("same_primary", "mean"),
                                                          overlap=("overlap", "mean"))
        print("\nagreement by their confidence:")
        for lv, r in by.iterrows():
            print(f"  {lv:<14} n={int(r.n):<5} same primary {r.same_primary:.0%}   theirs among ours {r.overlap:.0%}")
    conf = pd.crosstab(lab.their_primary.rename("theirs"), lab.our_primary.rename("ours"),
                       margins=True, margins_name="total")
    print("\ntheir primary (rows) x our primary (columns):")
    print(conf.to_string())
    print("\nwhere they differ most (theirs -> ours):")
    for (a, b), n in Counter(zip(lab.their_primary, lab.our_primary)).most_common():
        if a != b:
            print(f"  {a} -> {b}: {n}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "joined.csv", index=False)
    lab[~lab.same_primary].to_csv(out / "disagreements.csv", index=False)
    conf.to_csv(out / "confusion.csv")
    print(f"\nwrote {out}/joined.csv, disagreements.csv, confusion.csv (has titles; keep local)")


if __name__ == "__main__":
    main()
