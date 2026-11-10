"""Master list of people on a set of proposals (PI, co-PIs, and any other person
fields you name), for conflict-of-interest checks. Standalone: nothing to do with
scoring.

Pulls from SOLR (NSF network/VPN) for each proposal ID and, for collaborative
proposals, for every institution's copy (found through lead_id), since each copy
has its own PI and co-PIs.

    # what fields does SOLR hold? (names and sizes only, no content)
    python coi_personnel.py --list-fields 2705307

    # the list (IDs from a .txt/.xlsx/.csv, same as solr_fetch_descriptions.py)
    python coi_personnel.py ids.xlsx --out output/coi

    # once you know a senior-personnel field, add it (repeatable)
    python coi_personnel.py ids.xlsx --person-field <field_name>

Writes, in --out (keep it local; it holds names):
    personnel.csv  one row per person per proposal copy:
                   proposal_id, lead_id, title, inst, role, name, name_key, field
    people.csv     one row per distinct person (by name_key): names as written,
                   number of proposals, proposal IDs, institutions

name_key is "last|first-initial" in lower case without accents, e.g. "smith|j",
for matching against reviewer lists. It merges different people with the same
last name and initial, and splits one person written two different ways; check
matches by hand. A co-PI's institution isn't in SOLR separately, so `inst` is the
submitting institution of that copy.
"""

import argparse
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pysolr

from solr_fetch_descriptions import SOLR_BASE, fetch_batch, flat, read_ids

BASE_FIELDS = ["id", "title", "inst", "lead_id", "pi_name", "pi_all"]


def split_names(v) -> list[str]:
    """SOLR values may be a list or one string with ';' between names."""
    items = v if isinstance(v, list) else [v]
    out = []
    for x in items:
        if x is None:
            continue
        out += [p.strip() for p in re.split(r"\s*;\s*|\n", str(x)) if p.strip()]
    return out


def name_key(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(dr|prof|phd|jr|sr|ii|iii|iv)\b\.?", " ", s)
    if "," in s:  # "Smith, Jane M"
        last, first = s.split(",", 1)
    else:  # "Jane M Smith"
        parts = s.split()
        last, first = (parts[-1], " ".join(parts[:-1])) if parts else ("", "")
    last = re.sub(r"[^a-z]", "", last)
    first = re.sub(r"[^a-z ]", "", first).strip()
    return f"{last}|{first[:1]}" if last else ""


def list_fields(solr, pid: str):
    docs = list(solr.search(f"id:{pid}", fl="*", rows=1))
    if not docs:
        sys.exit(f"SOLR has no proposal {pid}")
    for k in sorted(docs[0]):
        v = docs[0][k]
        n = len(v) if isinstance(v, list) else 1
        print(f"{k:<32} {len(str(v)):>8} chars" + (f"  ({n} values)" if isinstance(v, list) else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="?", help="file of proposal IDs (.txt, .xlsx, .csv)")
    ap.add_argument("--id-col", help="column with the IDs (default: auto-detect)")
    ap.add_argument("--out", default="output/coi")
    ap.add_argument("--core", default="proposals")
    ap.add_argument("--person-field", action="append", default=[],
                    help="extra SOLR field holding names (e.g. senior personnel); repeatable")
    ap.add_argument("--no-collab", action="store_true",
                    help="don't add the other institutions' copies of collaborative proposals")
    ap.add_argument("--list-fields", metavar="ID", help="print the field names SOLR holds for one ID and exit")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=5.0)
    args = ap.parse_args()

    solr = pysolr.Solr(f"{SOLR_BASE}/{args.core}/", timeout=60)
    if args.list_fields:
        list_fields(solr, args.list_fields)
        return
    if not args.ids:
        ap.error("give a file of proposal IDs (or --list-fields ID)")

    ids = list(dict.fromkeys(read_ids(Path(args.ids), args.id_col)))
    print(f"{len(ids)} proposal IDs from {args.ids}")
    fields = ",".join(BASE_FIELDS + [f for f in args.person_field if f not in BASE_FIELDS])
    chunks = [ids[i:i + args.batch] for i in range(0, len(ids), args.batch)]

    docs, failed = {}, []
    for ch in chunks:
        for d in fetch_batch(solr, ch, fields, args.retries, args.backoff, failed):
            docs[str(d.get("id"))] = d
    missing = [i for i in ids if i not in docs and i not in failed]
    if not args.no_collab:  # every copy whose lead is one of these proposals
        leads = sorted({flat(d.get("lead_id")) or pid for pid, d in docs.items()})
        before = len(docs)
        for i in range(0, len(leads), args.batch):
            ch = leads[i:i + args.batch]
            q = "lead_id:(" + " OR ".join(ch) + ")"
            try:
                for d in solr.search(q, fl=fields, rows=10 * len(ch)):
                    docs.setdefault(str(d.get("id")), d)
            except (pysolr.SolrError, OSError) as e:
                print(f"  ! collaborative lookup failed for {len(ch)} leads: {str(e)[:150]}", file=sys.stderr)
        print(f"{len(docs) - before} other institutions' copies of collaborative proposals added")

    rows = []
    for pid, d in sorted(docs.items()):
        base = {"proposal_id": pid, "lead_id": flat(d.get("lead_id")) or pid,
                "title": flat(d.get("title")), "inst": flat(d.get("inst"))}
        pi = split_names(d.get("pi_name"))
        pi_keys = {name_key(n) for n in pi}
        seen = set()
        for n in pi:
            rows.append({**base, "role": "PI", "name": n, "name_key": name_key(n), "field": "pi_name"})
            seen.add(name_key(n))
        for n in split_names(d.get("pi_all")):
            k = name_key(n)
            if k in pi_keys or k in seen:
                continue
            seen.add(k)
            rows.append({**base, "role": "co-PI", "name": n, "name_key": k, "field": "pi_all"})
        for f in args.person_field:
            for n in split_names(d.get(f)):
                k = name_key(n)
                if k in seen:
                    continue
                seen.add(k)
                rows.append({**base, "role": "other", "name": n, "name_key": k, "field": f})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["proposal_id", "lead_id", "title", "inst", "role", "name", "name_key", "field"]
    with open(out / "personnel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    people = defaultdict(lambda: {"names": set(), "props": set(), "insts": set(), "roles": set()})
    for r in rows:
        p = people[r["name_key"] or r["name"].lower()]
        p["names"].add(r["name"])
        p["props"].add(r["lead_id"])
        p["insts"].add(r["inst"])
        p["roles"].add(r["role"])
    with open(out / "people.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name_key", "names", "n_proposals", "roles", "proposal_ids", "institutions"])
        for k, p in sorted(people.items(), key=lambda kv: (-len(kv[1]["props"]), kv[0])):
            w.writerow([k, "; ".join(sorted(p["names"])), len(p["props"]), "; ".join(sorted(p["roles"])),
                        "; ".join(sorted(p["props"])), "; ".join(sorted(i for i in p["insts"] if i))])

    by_role = defaultdict(int)
    for r in rows:
        by_role[r["role"]] += 1
    print(f"{len(rows)} person-proposal rows (" + ", ".join(f"{k}: {v}" for k, v in by_role.items())
          + f"); {len(people)} distinct people; {sum(len(p['props']) > 1 for p in people.values())} "
          "on more than one proposal")
    if missing:
        print(f"{len(missing)} IDs not in SOLR: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    if failed:
        print(f"{len(failed)} IDs failed (network); re-run to retry")
    print(f"wrote {out / 'personnel.csv'} and {out / 'people.csv'} (contains names; keep local)")


if __name__ == "__main__":
    main()
