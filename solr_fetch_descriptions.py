"""Download proposal project descriptions from NSF SOLR in resumable batches.

Converts script_get_solr_data.ipynb into a CLI. Each description is written to
its own text file (no Excel 32,767-char truncation), so the output folder can be
fed straight into pdf_batch_extract.py.

Needs the NSF network/VPN (SOLR is internal) and: pip install pysolr pandas openpyxl

Usage:
    # IDs from a spreadsheet (ID column auto-detected), a .txt/.csv, or the command line
    python solr_fetch_descriptions.py kdittmar_Planning_AI_Dataset_test.xlsx
    python solr_fetch_descriptions.py ids.txt --out solr_descriptions --batch-size 20
    python solr_fetch_descriptions.py --ids 2703928 2703259

    # then analyze them
    python pdf_batch_extract.py solr_descriptions/descriptions

Re-running is safe: IDs already downloaded are skipped and failed/missing ones
are retried, so an interrupted or timed-out run picks up where it left off
(use --force to re-download everything).

Output (--out, default ./solr_descriptions):
    descriptions/<id>.txt     one per proposal (lead copy of collaboratives)
    collab_partners/<id>.txt  other institutions' copies of collaborative proposals
    manifest.csv              id, title, status, chars, collab group
    missing_ids.txt           IDs SOLR didn't return or that have no description
    meta.jsonl                per-ID fetch log (used for resume + dedup)
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import pysolr

SOLR_BASE = "http://dis-checker-a01.ad.nsf.gov/solr"
ID_RE = re.compile(r"^\d{7}$")
_COLLAB_RE = re.compile(r"(?i)^collaborative\s+research\s*:\s*")  # same as nsf_solr_agent


# --------------------------------------------------------------------------- #
# Reading IDs
# --------------------------------------------------------------------------- #
def _clean_id(v) -> str | None:
    s = str(v).strip()
    if re.fullmatch(r"\d{7}\.0+", s):  # Excel floats: 2703928.0
        s = s.split(".")[0]
    return s if ID_RE.match(s) else None


def read_ids(path: Path, id_col: str | None) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in (".txt", ""):
        ids = [_clean_id(line) for line in path.read_text().splitlines()]
        return [i for i in ids if i]

    import pandas as pd

    df = (pd.read_excel(path, header=None, dtype=str) if suffix in (".xlsx", ".xls")
          else pd.read_csv(path, header=None, dtype=str))
    if id_col is not None:
        col = df[int(id_col)] if id_col.isdigit() else df.set_axis(df.iloc[0], axis=1)[id_col][1:]
    else:  # auto-detect: the column with the most 7-digit IDs
        col = max((df[c] for c in df.columns),
                  key=lambda s: s.map(lambda v: _clean_id(v) is not None).sum())
    ids = [_clean_id(v) for v in col]
    return [i for i in ids if i]


# --------------------------------------------------------------------------- #
# SOLR
# --------------------------------------------------------------------------- #
def fetch_batch(solr, ids, fields, retries, backoff):
    """Fetch one batch; retry with backoff, then split the batch in half."""
    query = "id:(" + " OR ".join(ids) + ")"
    for attempt in range(1, retries + 1):
        try:
            return list(solr.search(query, fl=fields, rows=len(ids)))
        except (pysolr.SolrError, OSError) as e:  # timeouts/connection errors
            wait = backoff * 2 ** (attempt - 1)
            print(f"    ! batch of {len(ids)} failed (attempt {attempt}/{retries}): "
                  f"{str(e)[:150]}; retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    if len(ids) == 1:
        print(f"    ! giving up on {ids[0]}", file=sys.stderr)
        return []
    mid = len(ids) // 2
    print(f"    ! splitting batch {len(ids)} -> {mid} + {len(ids) - mid}", file=sys.stderr)
    return (fetch_batch(solr, ids[:mid], fields, retries, backoff)
            + fetch_batch(solr, ids[mid:], fields, retries, backoff))


def as_text(v) -> str:
    if isinstance(v, list):
        return "\n\n".join(map(str, v))
    return "" if v is None else str(v)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_meta(meta_path: Path) -> dict:
    meta = {}
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                meta[rec["id"]] = rec
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id_file", nargs="?", help=".xlsx/.csv/.txt with proposal IDs")
    ap.add_argument("--ids", nargs="+", help="IDs on the command line instead of a file")
    ap.add_argument("--id-col", help="ID column (0-based index or header name); default: auto-detect")
    ap.add_argument("--out", default="solr_descriptions")
    ap.add_argument("--field", default="description", help="SOLR text field to download")
    ap.add_argument("--database", default="proposals")
    ap.add_argument("--batch-size", type=int, default=25, help="IDs per SOLR request")
    ap.add_argument("--timeout", type=int, default=180, help="seconds per SOLR request")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=10, help="first retry wait (s), doubles each time")
    ap.add_argument("--pause", type=float, default=1, help="seconds between batches")
    ap.add_argument("--no-dedup", action="store_true", help="keep every collaborative copy in descriptions/")
    ap.add_argument("--force", action="store_true", help="re-download IDs already fetched")
    args = ap.parse_args()

    if args.ids:
        ids = [i for i in map(_clean_id, args.ids) if i]
    elif args.id_file:
        ids = read_ids(Path(args.id_file), args.id_col)
    else:
        ap.error("give an ID file or --ids")
    ids = list(dict.fromkeys(ids))  # dedupe, keep order
    if not ids:
        sys.exit("no 7-digit proposal IDs found")

    out = Path(args.out)
    desc_dir, partner_dir = out / "descriptions", out / "collab_partners"
    desc_dir.mkdir(parents=True, exist_ok=True)
    partner_dir.mkdir(exist_ok=True)
    meta_path = out / "meta.jsonl"
    meta = {} if args.force else load_meta(meta_path)

    # retry anything not yet fetched OK (incl. failed batches / not-yet-indexed),
    # and anything logged as OK whose .txt has since been deleted
    have = lambda i: (desc_dir / f"{i}.txt").exists() or (partner_dir / f"{i}.txt").exists()
    todo = [i for i in ids if meta.get(i, {}).get("status") != "ok" or not have(i)]
    print(f"{len(ids)} IDs; {len(ids) - len(todo)} already fetched; {len(todo)} to fetch "
          f"in batches of {args.batch_size}")

    solr = pysolr.Solr(f"{SOLR_BASE}/{args.database}/", timeout=args.timeout)
    fields = f"id,title,{args.field}"
    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size

    with open(meta_path, "a") as mf:
        for b in range(n_batches):
            batch = todo[b * args.batch_size:(b + 1) * args.batch_size]
            t0 = time.time()
            try:
                docs = fetch_batch(solr, batch, fields, args.retries, args.backoff)
            except KeyboardInterrupt:
                print("\ninterrupted; re-run to resume")
                break
            found = {str(d.get("id")): d for d in docs}
            for pid in batch:
                d = found.get(pid)
                if d is None:
                    rec = {"id": pid, "status": "not_found", "title": "", "chars": 0}
                else:
                    text = as_text(d.get(args.field)).strip()
                    title = as_text(d.get("title")).strip()
                    rec = {"id": pid, "title": title, "chars": len(text),
                           "status": "ok" if text else "no_description"}
                    if text:
                        (desc_dir / f"{pid}.txt").write_text(
                            f"Proposal {pid}: {title}\n\n{text}\n")
                meta[pid] = rec
                mf.write(json.dumps(rec) + "\n")
            mf.flush()
            ok = sum(1 for p in batch if meta[p]["status"] == "ok")
            print(f"[batch {b + 1}/{n_batches}] {ok}/{len(batch)} with text "
                  f"({time.time() - t0:.0f}s)")
            if b + 1 < n_batches:
                time.sleep(args.pause)

    # ---- collaborative dedup: keep the longest copy per normalized title ----
    rows = [meta[i] for i in ids if i in meta]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r["status"] == "ok":
            key = _COLLAB_RE.sub("", r["title"]).strip().lower() or r["id"]
            groups.setdefault(key, []).append(r)
    for g in groups.values():
        lead = max(g, key=lambda r: r["chars"])
        for r in g:
            r["collab_group"] = lead["id"] if len(g) > 1 else ""
            primary = args.no_dedup or r is lead
            src, dst = ((partner_dir, desc_dir) if primary else (desc_dir, partner_dir))
            f = src / f"{r['id']}.txt"
            if f.exists():
                f.replace(dst / f.name)

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "status", "chars", "collab_group"],
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({"collab_group": "", **r})

    missing = [r["id"] for r in rows if r["status"] != "ok"]
    (out / "missing_ids.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    n_desc = len(list(desc_dir.glob("*.txt")))
    n_partner = len(list(partner_dir.glob("*.txt")))
    print(f"\ndone: {n_desc} descriptions in {desc_dir}"
          + (f", {n_partner} collaborative partner copies set aside" if n_partner else "")
          + (f", {len(missing)} missing (see missing_ids.txt)" if missing else ""))
    not_run = len(ids) - len(rows)
    if not_run:
        print(f"{not_run} IDs not attempted yet; re-run to continue")


if __name__ == "__main__":
    main()
