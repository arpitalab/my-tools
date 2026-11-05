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
(use --force to re-download everything). IDs downloaded by an older version
without PI/institution/amount get just that metadata filled in.

Metadata (--meta-fields) is written to manifest.csv for rubric_dashboard.py.
Defaults are the SOLR field names used in nsf_proposal_extract.py/solr_mcp.py;
a field SOLR doesn't have is simply left blank. The concurrence check looks
for the words "concurrence" / "concur" in the --concurrence-fields text (the
uploaded Program Officer Concurrence Email should land there); it is a hint,
not proof: verify in eJacket/Research.gov.

Why an ID has no text (status in manifest.csv / missing_ids.txt):
    error           the SOLR request failed (timeout etc.); re-run to retry
    not_found       SOLR answered but has no proposal with this ID (not loaded
                    yet? wrong ID?)
    no_description  SOLR has the proposal but its description field is empty
                    (text not extracted yet?)
Run with --diagnose to look up missing IDs one at a time and list which
fields SOLR does hold for them, plus how recent SOLR's latest load is.

Output (--out, default ./solr_descriptions):
    descriptions/<id>.txt     one per proposal, ready to score
    collaborative/<id>.txt    collaborative proposals, set aside (see --collab)
    collaborative_rwr.csv     collaborative proposals to consider returning without review
    manifest.csv              id, title, status, chars, collaborative, collab group, plus
                              PI, institution, requested amount/months, concurrence check
    missing_ids.txt           IDs without text, with the reason (see below)
    meta.jsonl                per-ID fetch log (used for resume + dedup)
"""

import argparse
import csv
from collections import Counter
import json
import re
import sys
import time
from pathlib import Path

import pysolr

SOLR_BASE = "http://dis-checker-a01.ad.nsf.gov/solr"
ID_RE = re.compile(r"^\d{7}$")
_COLLAB_RE = re.compile(r"(?i)^collaborative\s+research\s*:\s*")  # same as nsf_solr_agent
META_FIELDS = ("pi_name,inst,inst_state,requested_amount,rqst_mnth_cnt,lead_id,lead_proposal,"
               "received,load_date,description_length,status,program_announcement,managing_program")
RESERVED = {"id", "title", "status", "chars", "collaborative", "collab_group"}  # this script's own keys


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
def fetch_batch(solr, ids, fields, retries, backoff, failed=None, extra_q=""):
    """Fetch one batch; retry with backoff, then split the batch in half.

    IDs whose request still fails are added to `failed` (so they are logged as
    'error', not mistaken for proposals SOLR doesn't have)."""
    failed = [] if failed is None else failed
    query = "id:(" + " OR ".join(ids) + ")" + (f" AND ({extra_q})" if extra_q else "")
    for attempt in range(1, retries + 1):
        try:
            return list(solr.search(query, fl=fields, rows=len(ids)))
        except (pysolr.SolrError, OSError) as e:  # timeouts/connection errors
            wait = backoff * 2 ** (attempt - 1)
            print(f"    ! batch of {len(ids)} failed (attempt {attempt}/{retries}): "
                  f"{str(e)[:150]}; retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    if len(ids) == 1:
        print(f"    ! giving up on {ids[0]} (logged as 'error'; re-run to retry)", file=sys.stderr)
        failed.append(ids[0])
        return []
    mid = len(ids) // 2
    print(f"    ! splitting batch {len(ids)} -> {mid} + {len(ids) - mid}", file=sys.stderr)
    return (fetch_batch(solr, ids[:mid], fields, retries, backoff, failed, extra_q)
            + fetch_batch(solr, ids[mid:], fields, retries, backoff, failed, extra_q))


def concurrence_ids(solr, ids, concur_fields, retries, backoff) -> set | None:
    """IDs whose concurrence fields mention 'concurrence'/'concur', found by a SOLR
    query (only IDs come back, not the documents' text). None if the query failed."""
    if not concur_fields:
        return None
    q = " OR ".join(f"{f}:(concurrence OR concur OR concurred)" for f in concur_fields)
    failed = []
    docs = fetch_batch(solr, ids, "id", retries, backoff, failed, extra_q=q)
    return None if failed else {str(d.get("id")) for d in docs}


def as_text(v) -> str:
    if isinstance(v, list):
        return "\n\n".join(map(str, v))
    return "" if v is None else str(v)


def flat(v) -> str:
    """Short metadata value: lists joined with '; ', duplicates dropped."""
    if isinstance(v, list):
        return "; ".join(dict.fromkeys(str(x).strip() for x in v if str(x).strip()))
    return "" if v is None else str(v).strip()


def meta_key(field: str) -> str:
    return f"solr_{field}" if field in RESERVED else field


def meta_from_doc(d: dict, meta_fields: list[str], concur: set | None) -> dict:
    # SOLR fields named like this script's own keys (e.g. SOLR's "status") get a solr_ prefix
    m = {meta_key(f): flat(d.get(f)) for f in meta_fields}
    if concur is not None:
        m["concurrence_found"] = "yes" if str(d.get("id")) in concur else "no"
    m["meta_ok"] = True
    return m


def diagnose(solr, ids, meta_fields, timeout_note=""):
    """Look up IDs one at a time and show what SOLR holds for each."""
    print(f"\n=== diagnosing {len(ids)} ID(s) one at a time ===")
    try:
        latest = list(solr.search("*:*", fl="load_date", sort="load_date desc", rows=1))
        print(f"latest load_date in SOLR: {latest[0].get('load_date') if latest else '(none)'}"
              "  (proposals submitted after this aren't in SOLR yet)")
    except Exception as e:
        print(f"couldn't read SOLR's latest load_date: {str(e)[:150]}")
    for pid in ids:
        try:
            docs = list(solr.search(f"id:{pid}", fl="*", rows=1))
        except Exception as e:
            print(f"{pid}: request FAILED ({str(e)[:150]}) -> SOLR/network problem, not missing text")
            continue
        if not docs:
            print(f"{pid}: SOLR has no proposal with this ID (not loaded yet, or wrong ID)")
            continue
        d = docs[0]
        sizes = {k: len(as_text(v)) for k, v in d.items()}
        text_fields = sorted((k for k, n in sizes.items() if n > 500), key=lambda k: -sizes[k])
        print(f"{pid}: {flat(d.get('title'))[:70]}")
        print(f"    received={flat(d.get('received'))} load_date={flat(d.get('load_date'))} "
              f"status={flat(d.get('status'))[:40]}")
        print(f"    description: {sizes.get('description', 0)} chars returned; SOLR's "
              f"description_length={flat(d.get('description_length')) or '(not set)'}")
        print("    text fields present: " + (", ".join(f"{k} ({sizes[k]:,})" for k in text_fields)
                                              or "none longer than 500 chars"))


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
    ap.add_argument("--collab", choices=["rwr", "dedup", "keep"], default="rwr",
                    help="collaborative proposals: rwr = set every copy aside and list them as "
                         "return-without-review candidates (default; for solicitations that don't "
                         "allow collaboratives); dedup = score one copy per project; keep = score all")
    ap.add_argument("--meta-fields", default=META_FIELDS,
                    help="comma-separated SOLR metadata fields saved to manifest.csv")
    ap.add_argument("--concurrence-fields", default="additional_documents,supplementary",
                    help="SOLR text fields searched for the PO concurrence email ('' to skip)")
    ap.add_argument("--force", action="store_true", help="re-download IDs already fetched")
    ap.add_argument("--diagnose", type=int, nargs="?", const=10, default=0, metavar="N",
                    help="look up N (default 10) missing IDs one at a time and report what SOLR "
                         "holds for them; doesn't download anything")
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
    desc_dir, collab_dir = out / "descriptions", out / "collaborative"
    desc_dir.mkdir(parents=True, exist_ok=True)
    collab_dir.mkdir(exist_ok=True)
    old_dir = out / "collab_partners"  # name used by earlier versions
    if old_dir.is_dir():
        for f in old_dir.glob("*.txt"):
            f.replace(collab_dir / f.name)
        old_dir.rmdir()
    meta_path = out / "meta.jsonl"
    meta = {} if args.force else load_meta(meta_path)

    # retry anything not yet fetched OK (incl. failed batches / not-yet-indexed),
    # and anything logged as OK whose .txt has since been deleted
    have = lambda i: (desc_dir / f"{i}.txt").exists() or (collab_dir / f"{i}.txt").exists()
    todo = [i for i in ids if meta.get(i, {}).get("status") != "ok" or not have(i)]
    print(f"{len(ids)} IDs; {len(ids) - len(todo)} already fetched; {len(todo)} to fetch "
          f"in batches of {args.batch_size}")

    meta_fields = [x.strip() for x in args.meta_fields.split(",") if x.strip()]
    concur_fields = [x.strip() for x in args.concurrence_fields.split(",") if x.strip()]
    # downloaded by an older version: fetch only the metadata, not the description again
    meta_todo = [i for i in ids if i not in todo and not meta[i].get("meta_ok")]
    if meta_todo:
        print(f"{len(meta_todo)} already-downloaded IDs need PI/institution/amount metadata")

    solr = pysolr.Solr(f"{SOLR_BASE}/{args.database}/", timeout=args.timeout)
    if args.diagnose:
        missing = [i for i in todo if meta.get(i, {}).get("status") != "ok"]
        if not missing:
            sys.exit("nothing to diagnose: every ID has text")
        by = {}
        for i in missing:
            by.setdefault(meta.get(i, {}).get("status", "not tried yet"), []).append(i)
        print("IDs without text, by logged reason: "
              + ", ".join(f"{k} {len(v)}" for k, v in by.items()))
        # sample a few of each reason
        per = max(1, args.diagnose // len(by))
        diagnose(solr, [i for v in by.values() for i in v[:per]][:args.diagnose], meta_fields)
        return
    fields = ",".join(dict.fromkeys(["id", "title", args.field] + meta_fields))
    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size

    with open(meta_path, "a") as mf:
        for b in range(n_batches):
            batch = todo[b * args.batch_size:(b + 1) * args.batch_size]
            t0 = time.time()
            failed = []
            try:
                docs = fetch_batch(solr, batch, fields, args.retries, args.backoff, failed)
                concur = concurrence_ids(solr, [str(d.get("id")) for d in docs], concur_fields,
                                         args.retries, args.backoff) if docs else None
            except KeyboardInterrupt:
                print("\ninterrupted; re-run to resume")
                break
            found = {str(d.get("id")): d for d in docs}
            for pid in batch:
                d = found.get(pid)
                if d is None:
                    rec = {"id": pid, "status": "error" if pid in failed else "not_found",
                           "title": "", "chars": 0}
                else:
                    text = as_text(d.get(args.field)).strip()
                    title = as_text(d.get("title")).strip()
                    rec = {"id": pid, "title": title, "chars": len(text),
                           "status": "ok" if text else "no_description",
                           **meta_from_doc(d, meta_fields, concur)}
                    if text:
                        (desc_dir / f"{pid}.txt").write_text(
                            f"Proposal {pid}: {title}\n\n{text}\n")
                meta[pid] = rec
                mf.write(json.dumps(rec) + "\n")
            mf.flush()
            st = Counter(meta[p]["status"] for p in batch)
            print(f"[batch {b + 1}/{n_batches}] {st['ok']}/{len(batch)} with text"
                  + "".join(f", {st[k]} {k}" for k in ("not_found", "no_description", "error") if st[k])
                  + f" ({time.time() - t0:.0f}s)")
            if b + 1 < n_batches:
                time.sleep(args.pause)

        meta_only = ",".join(dict.fromkeys(["id"] + meta_fields))
        for b in range(0, len(meta_todo), args.batch_size):
            batch = meta_todo[b:b + args.batch_size]
            try:
                docs = fetch_batch(solr, batch, meta_only, args.retries, args.backoff)
                concur = concurrence_ids(solr, [str(d.get("id")) for d in docs], concur_fields,
                                         args.retries, args.backoff) if docs else None
            except KeyboardInterrupt:
                print("\ninterrupted; re-run to resume")
                break
            for d in docs:
                pid = str(d.get("id"))
                if pid in meta:
                    meta[pid].update(meta_from_doc(d, meta_fields, concur))
                    mf.write(json.dumps(meta[pid]) + "\n")
            mf.flush()
            print(f"[metadata {b // args.batch_size + 1}/{(len(meta_todo) + args.batch_size - 1) // args.batch_size}] "
                  f"{len(docs)}/{len(batch)}")

    # ---- collaborative proposals ------------------------------------------
    # Collaborative = title starts with "Collaborative Research:" (NSF's required
    # prefix for separately submitted collaboratives) or shares its normalized
    # title with another ID in this set.
    rows = [meta[i] for i in ids if i in meta]
    # SOLR lead_id: a collaborative non-lead points at its lead's ID. Only trusted
    # when it is a 7-digit ID different from the proposal's own.
    lead_of = {r["id"]: r.get("lead_id", "").split(";")[0].strip() for r in rows}
    lead_of = {i: l for i, l in lead_of.items() if ID_RE.match(l) and l != i}
    leads = set(lead_of.values())
    if rows and len(lead_of) > 0.5 * len(rows):
        print(f"! warning: {len(lead_of)} of {len(rows)} proposals have a lead_id different from "
              "their own ID; check whether SOLR's lead_id means what this script assumes",
              file=sys.stderr)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r["status"] == "ok":
            key = _COLLAB_RE.sub("", r["title"]).strip().lower() or r["id"]
            groups.setdefault(key, []).append(r)
    rwr = []
    for g in groups.values():
        lead = max(g, key=lambda r: r["chars"])
        for r in g:
            prefixed = bool(_COLLAB_RE.match(r["title"]))
            by_lead = r["id"] in lead_of or r["id"] in leads
            is_collab = prefixed or len(g) > 1 or by_lead
            r["collaborative"] = "yes" if is_collab else ""
            r["collab_group"] = (lead["id"] if len(g) > 1 else
                                 lead_of.get(r["id"]) or (r["id"] if r["id"] in leads else ""))
            if is_collab:
                reasons = (["'Collaborative Research:' title"] if prefixed else []) + \
                          ([f"same title as {', '.join(x['id'] for x in g if x is not r)}"] if len(g) > 1 else []) + \
                          ([f"SOLR lead_id {lead_of[r['id']]}"] if r["id"] in lead_of else []) + \
                          (["SOLR lead of " + ", ".join(i for i, l in lead_of.items() if l == r["id"])]
                           if r["id"] in leads else [])
                rwr.append({"id": r["id"], "title": r["title"], "collab_group": r["collab_group"],
                            "reason": "; ".join(reasons)})
            if args.collab == "keep" or not is_collab:
                dst = desc_dir
            elif args.collab == "dedup":
                dst = desc_dir if r is lead else collab_dir
            else:  # rwr
                dst = collab_dir
            for src in (desc_dir, collab_dir):
                f = src / f"{r['id']}.txt"
                if f.exists() and src != dst:
                    f.replace(dst / f.name)

    with open(out / "manifest.csv", "w", newline="") as f:
        cols = ["id", "title", "status", "chars", "collaborative", "collab_group"] + \
               [meta_key(c) for c in meta_fields if c not in ("id", "title")] + \
               (["concurrence_found"] if concur_fields else [])
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({"collaborative": "", "collab_group": "", **r})
    with open(out / "collaborative_rwr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "collab_group", "reason"])
        w.writeheader()
        w.writerows(rwr)

    missing = [r["id"] for r in rows if r["status"] != "ok"]
    (out / "missing_ids.txt").write_text("".join(f"{r['id']}\t{r['status']}\n" for r in rows
                                                 if r["status"] != "ok"))
    n_desc = len(list(desc_dir.glob("*.txt")))
    n_set_aside = len(list(collab_dir.glob("*.txt")))
    print(f"\ndone: {n_desc} descriptions ready to score in {desc_dir}"
          + (f", {len(missing)} missing (see missing_ids.txt)" if missing else ""))
    if rwr:
        verb = {"rwr": "set aside, not scored", "dedup": "one copy per project kept for scoring",
                "keep": "all kept for scoring"}[args.collab]
        print(f"{len(rwr)} collaborative proposal(s) found ({verb}); "
              f"listed in {out / 'collaborative_rwr.csv'}")
    not_run = len(ids) - len(rows)
    if not_run:
        print(f"{not_run} IDs not attempted yet; re-run to continue")
    if missing:
        st = Counter(r["status"] for r in rows if r["status"] != "ok")
        why = {"error": "SOLR request failed (timeout/network); re-run to retry",
               "not_found": "SOLR has no proposal with that ID (not loaded yet, or wrong ID)",
               "no_description": "SOLR has the proposal but no description text"}
        print("no text, by reason:")
        for k, n in st.most_common():
            print(f"    {n:5d}  {k:15s} {why.get(k, '')}")
        print("for details on a sample: python solr_fetch_descriptions.py <same ids> --diagnose")


if __name__ == "__main__":
    main()
