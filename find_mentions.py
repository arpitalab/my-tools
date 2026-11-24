"""Find every mention of named resources (default: NEON and GBIF) in the proposal
descriptions. Plain pattern matching, no LLM.

    python find_mentions.py                                   # NEON + GBIF
    python find_mentions.py --term "iDigBio=(?i)\\bidigbio\\b"  # add more (label=regex, case-sensitive
                                                              # unless (?i)), repeatable
    python find_mentions.py --join output/ai_datasets_r08/final_pool.csv   # add final directorate etc.

Matching:
  NEON  "NEON" in capitals only, so the element neon (and "Neon" at the start of a
        sentence) isn't counted, or the full name "National Ecological Observatory
        Network" in any case.
  GBIF  "GBIF" in any case, or "Global Biodiversity Information Facility".
Line breaks and hyphenation inside the full names are allowed. Check the snippets:
an acronym can still mean something else.

Writes, in --out (local; it quotes proposal text):
  mentions.csv   one row per mention: id, title, term, snippet (about 200 characters
                 either side), position
  proposals.csv  one row per proposal with any mention: id, title, a count per term,
                 the first snippet per term, and any --join columns
"""

import argparse
import csv
import re
from pathlib import Path

DEFAULT_TERMS = {
    "NEON": r"\bNEON\b|(?i:national\s+ecological\s+observ-?\s*atory\s+network)",
    "GBIF": r"(?i:\bGBIF\b|global\s+biodiversity\s+infor-?\s*mation\s+facility)",
}


def load_titles(desc_dir: Path) -> dict:
    titles = {}
    for name in ("pdf_manifest.csv", "manifest.csv"):
        m = desc_dir.parent / name
        if m.is_file():
            with open(m, newline="", errors="replace") as fh:
                for r in csv.DictReader(fh):
                    if r.get("id") and r.get("title"):
                        titles[r["id"]] = r["title"]
    return titles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("descriptions", nargs="?", default="solr_descriptions/descriptions",
                    help="folder of <id>.txt descriptions")
    ap.add_argument("--term", action="append", default=[], help='extra term as "label=regex"; repeatable')
    ap.add_argument("--only", action="store_true", help="use only the --term patterns, not NEON/GBIF")
    ap.add_argument("--context", type=int, default=200, help="characters of context either side")
    ap.add_argument("--join", help="CSV with an id column (e.g. final_pool.csv) whose columns are added")
    ap.add_argument("--out", default="output/mentions")
    args = ap.parse_args()

    terms = {} if args.only else dict(DEFAULT_TERMS)
    for t in args.term:
        label, _, rx = t.partition("=")
        if not rx:
            ap.error(f'--term needs "label=regex", got {t!r}')
        terms[label.strip()] = rx
    pats = {k: re.compile(v) for k, v in terms.items()}

    d = Path(args.descriptions)
    files = sorted(d.glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt files in {d}")
    titles = load_titles(d)

    mentions, per = [], {}
    for f in files:
        raw = f.read_text(errors="replace")
        text = re.sub(r"\s+", " ", raw)
        title = titles.get(f.stem) or re.sub(r"^Proposal \d+:\s*", "", raw.split("\n", 1)[0].strip())
        for label, pat in pats.items():
            for m in pat.finditer(text):
                a, b = max(0, m.start() - args.context), min(len(text), m.end() + args.context)
                snip = ("…" if a else "") + text[a:m.start()] + "[[" + m.group(0) + "]]" + text[m.end():b] + \
                       ("…" if b < len(text) else "")
                mentions.append({"id": f.stem, "title": title, "term": label, "match": m.group(0),
                                 "position": m.start(), "snippet": snip})
                p = per.setdefault(f.stem, {"id": f.stem, "title": title,
                                            **{f"n_{k}": 0 for k in terms}, **{f"first_{k}": "" for k in terms}})
                p[f"n_{label}"] += 1
                if not p[f"first_{label}"]:
                    p[f"first_{label}"] = snip

    join_cols = []
    if args.join:
        with open(args.join, newline="", errors="replace") as fh:
            jr = {r["id"]: r for r in csv.DictReader(fh)}
        join_cols = [c for c in (next(iter(jr.values()), {}) or {}) if c not in ("id", "title")]
        for p in per.values():
            for c in join_cols:
                p[c] = jr.get(p["id"], {}).get(c, "")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mentions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "title", "term", "match", "position", "snippet"])
        w.writeheader()
        w.writerows(mentions)
    cols = ["id", "title"] + [f"n_{k}" for k in terms] + join_cols + [f"first_{k}" for k in terms]
    rows = sorted(per.values(), key=lambda p: -sum(p[f"n_{k}"] for k in terms))
    with open(out / "proposals.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"{len(files)} descriptions searched")
    for k in terms:
        n_prop = sum(1 for p in per.values() if p[f"n_{k}"])
        n_hits = sum(p[f"n_{k}"] for p in per.values())
        print(f"  {k}: {n_prop} proposals, {n_hits} mentions")
    both = [p for p in per.values() if all(p[f"n_{k}"] for k in terms)] if len(terms) > 1 else []
    if both:
        print(f"  all terms: {len(both)} proposals")
    print(f"wrote {out / 'mentions.csv'} and {out / 'proposals.csv'} (quotes proposal text; keep local)")


if __name__ == "__main__":
    main()
