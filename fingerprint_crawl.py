"""
fingerprint_crawl.py — Phase 1: crawl OpenAlex for all PI paper abstracts.

Fetches paper texts (abstracts via abstract_inverted_index, title fallback)
for every unique Principal Investigator in the NSF awards DB. Works for ALL
NSF directorates — no PubMed dependency. Fully resumable: already-crawled
investigators are skipped on restart.

Schema added to DB:
    researcher_papers      — one row per paper per investigator
    researcher_crawl_log   — one row per investigator (crawl status/metadata)

Usage:
    python fingerprint_crawl.py --year 2024
    python fingerprint_crawl.py --all-years
    python fingerprint_crawl.py --year 2024 --workers 4 --delay 0.1
    python fingerprint_crawl.py --year 2024 --dry-run   # count only
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH   = "./output/nsf_awards.db"
OA_BASE   = "https://api.openalex.org"
OA_EMAIL  = "spine.calcium@gmail.com"
PAPER_YEARS  = 5
MAX_PAPERS   = 30
DEFAULT_DELAY = 0.15   # seconds between requests per worker


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS researcher_papers (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    investigator_key  TEXT    NOT NULL,
    full_name         TEXT,
    openalex_id       TEXT,
    orcid             TEXT,
    paper_title       TEXT,
    paper_text        TEXT,        -- abstract if available, else title
    has_abstract      INTEGER,     -- 1 = full abstract, 0 = title only
    pub_year          INTEGER,
    topics            TEXT,        -- JSON array of OpenAlex topic strings
    venue             TEXT,
    crawled_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_rp_key
    ON researcher_papers(investigator_key);

CREATE TABLE IF NOT EXISTS researcher_crawl_log (
    investigator_key  TEXT PRIMARY KEY,
    full_name         TEXT,
    inst_name         TEXT,
    openalex_id       TEXT,
    orcid             TEXT,
    oa_works_count    INTEGER,
    oa_cited_by       INTEGER,
    n_papers_saved    INTEGER,
    status            TEXT,   -- 'ok' | 'oa_not_found' | 'error'
    error_msg         TEXT,
    crawled_at        TEXT
);
"""


def _apply_schema(conn: sqlite3.Connection) -> None:
    for stmt in _SCHEMA.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s)
    conn.commit()


# ---------------------------------------------------------------------------
# Investigator key (mirrors nsf_app.py / nsf_panel_builder.py)
# ---------------------------------------------------------------------------

def _inv_key(full_name: str, email: str) -> str | None:
    em = (email or "").strip().lower()
    if em:
        return f"e:{em}"
    name = (full_name or "").strip().lower()
    parts = [p.strip(".,") for p in name.split() if len(p.strip(".,")) > 1]
    return f"n:{' '.join(parts)}" if parts else None


# ---------------------------------------------------------------------------
# Collect unique PIs from DB
# ---------------------------------------------------------------------------

def collect_investigators(
    conn: sqlite3.Connection,
    years: list[int] | None,
) -> list[dict]:
    """
    Return one record per unique investigator key.
    Aggregates all NSF award_ids for each person.
    """
    year_clause = ""
    params: list = []
    if years:
        ph = ",".join("?" * len(years))
        year_clause = f"AND a.source_year IN ({ph})"
        params = list(years)

    rows = conn.execute(
        f"""
        SELECT ai.full_name, ai.email,
               i.name  AS inst_name,
               a.award_id
        FROM award_investigator ai
        JOIN  award       a ON a.id  = ai.award_id
        LEFT JOIN institution i ON i.id = a.institution_id
        WHERE ai.role_code = 'Principal Investigator'
          AND ai.full_name IS NOT NULL
          AND LENGTH(TRIM(ai.full_name)) > 3
          {year_clause}
        ORDER BY ai.full_name
        """,
        params,
    ).fetchall()

    # Group by investigator key
    by_key: dict[str, dict] = {}
    for r in rows:
        key = _inv_key(r["full_name"], r["email"] or "")
        if key is None:
            continue
        if key not in by_key:
            by_key[key] = {
                "key":       key,
                "full_name": r["full_name"],
                "email":     r["email"] or "",
                "inst_name": r["inst_name"] or "",
                "award_ids": [],
            }
        by_key[key]["award_ids"].append(r["award_id"])

    return list(by_key.values())


# ---------------------------------------------------------------------------
# OpenAlex helpers
# ---------------------------------------------------------------------------

def _oa_get(path: str, params: dict, delay: float) -> dict:
    time.sleep(delay)
    resp = requests.get(
        f"{OA_BASE}/{path}",
        params=params,
        headers={"User-Agent": f"mailto:{OA_EMAIL}"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _reconstruct_abstract(inv_idx: dict | None) -> str:
    if not inv_idx:
        return ""
    pos: dict[int, str] = {}
    for word, locs in inv_idx.items():
        for p in locs:
            pos[p] = word
    return " ".join(pos[i] for i in sorted(pos))


def resolve_author(name: str, inst: str, delay: float) -> dict | None:
    """Search OpenAlex by name; validate top candidates against institution."""
    try:
        data = _oa_get("authors", {"search": name, "per_page": 8}, delay)
        results = data.get("results", [])
    except Exception:
        return None

    if not results:
        return None

    # Reject implausibly prolific matches (wrong-person signal)
    candidates = [r for r in results if r.get("works_count", 0) < 600] or results

    if inst:
        inst_words = {w.lower() for w in inst.split() if len(w) > 3}
        for r in candidates:
            for oa_inst in (r.get("last_known_institutions") or []):
                oa_words = set((oa_inst.get("display_name") or "").lower().split())
                if inst_words & oa_words:
                    return r

    return candidates[0]


def fetch_papers(oa_id: str, delay: float) -> list[dict]:
    """Fetch recent works for an OpenAlex author ID."""
    cutoff = datetime.now(timezone.utc).year - PAPER_YEARS
    try:
        data = _oa_get(
            "works",
            {
                "filter":  f"authorships.author.id:{oa_id},"
                           f"publication_year:>{cutoff}",
                "per_page": MAX_PAPERS,
                "select":  "ids,title,publication_year,"
                           "abstract_inverted_index,topics,primary_location",
            },
            delay,
        )
    except Exception:
        return []

    papers = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        has_abs  = len(abstract) > 80
        text     = abstract if has_abs else (w.get("title") or "")
        if not text:
            continue
        topics = [t["display_name"] for t in (w.get("topics") or [])[:6]]
        venue  = ((w.get("primary_location") or {})
                  .get("source") or {}).get("display_name", "")
        papers.append({
            "title":        w.get("title") or "",
            "text":         text,
            "has_abstract": int(has_abs),
            "pub_year":     w.get("publication_year"),
            "topics":       topics,
            "venue":        venue,
        })
    return papers


# ---------------------------------------------------------------------------
# Crawl one investigator
# ---------------------------------------------------------------------------

def crawl_one(person: dict, delay: float) -> dict:
    """
    Resolve author on OpenAlex, fetch papers, return result dict.
    Does NOT write to DB — caller handles that for thread safety.
    """
    name = person["full_name"]
    inst = person["inst_name"]
    now  = datetime.now(timezone.utc).isoformat()

    oa = resolve_author(name, inst, delay)
    if oa is None:
        return {
            "key": person["key"], "full_name": name, "inst_name": inst,
            "openalex_id": None, "orcid": None,
            "oa_works_count": 0, "oa_cited_by": 0,
            "papers": [], "status": "oa_not_found", "error_msg": None,
            "crawled_at": now,
        }

    oa_id   = oa["id"].split("/")[-1]
    orcid   = oa.get("orcid") or ""
    oa_cnt  = oa.get("works_count", 0)
    oa_cite = oa.get("cited_by_count", 0)

    try:
        papers = fetch_papers(oa_id, delay)
        status = "ok"
        err    = None
    except Exception as e:
        papers = []
        status = "error"
        err    = str(e)

    return {
        "key": person["key"], "full_name": name, "inst_name": inst,
        "openalex_id": oa_id, "orcid": orcid,
        "oa_works_count": oa_cnt, "oa_cited_by": oa_cite,
        "papers": papers, "status": status, "error_msg": err,
        "crawled_at": now,
    }


# ---------------------------------------------------------------------------
# Batch write helpers
# ---------------------------------------------------------------------------

def _write_result(conn: sqlite3.Connection, result: dict) -> None:
    key = result["key"]
    conn.execute(
        """INSERT OR REPLACE INTO researcher_crawl_log
           (investigator_key, full_name, inst_name, openalex_id, orcid,
            oa_works_count, oa_cited_by, n_papers_saved, status, error_msg,
            crawled_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (key, result["full_name"], result["inst_name"],
         result["openalex_id"], result["orcid"],
         result["oa_works_count"], result["oa_cited_by"],
         len(result["papers"]), result["status"], result["error_msg"],
         result["crawled_at"]),
    )
    for p in result["papers"]:
        conn.execute(
            """INSERT INTO researcher_papers
               (investigator_key, full_name, openalex_id, orcid,
                paper_title, paper_text, has_abstract, pub_year,
                topics, venue, crawled_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (key, result["full_name"], result["openalex_id"], result["orcid"],
             p["title"], p["text"], p["has_abstract"], p["pub_year"],
             json.dumps(p["topics"]), p["venue"], result["crawled_at"]),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crawl OpenAlex for NSF PI paper abstracts")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--year",      type=int, nargs="+",
                     help="One or more source years, e.g. --year 2024")
    grp.add_argument("--all-years", action="store_true",
                     help="Crawl all years (74k+ unique PIs — hours)")
    p.add_argument("--db",       default=DB_PATH)
    p.add_argument("--workers",  type=int, default=1,
                   help="Parallel OpenAlex workers (default 1; max ~4 before rate-limit risk)")
    p.add_argument("--delay",    type=float, default=DEFAULT_DELAY,
                   help="Seconds between API requests per worker (default 0.15)")
    p.add_argument("--batch",    type=int, default=200,
                   help="DB commit every N investigators (default 200)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Count investigators and exit without crawling")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _apply_schema(conn)

    years = None if args.all_years else args.year
    print(f"Collecting investigators from DB (years={years or 'all'}) …", end=" ", flush=True)
    investigators = collect_investigators(conn, years)
    print(f"{len(investigators):,} unique PIs found.")

    # Skip already-crawled
    done = {r["investigator_key"] for r in
            conn.execute("SELECT investigator_key FROM researcher_crawl_log").fetchall()}
    todo = [p for p in investigators if p["key"] not in done]
    print(f"Already crawled: {len(done):,}  |  Remaining: {len(todo):,}")

    if args.dry_run or not todo:
        est_hrs = len(todo) * 2 * args.delay / 3600
        print(f"Estimated API time: {est_hrs:.1f} hrs at {args.delay}s delay "
              f"with {args.workers} worker(s)")
        conn.close()
        return

    # ── Crawl ────────────────────────────────────────────────────────────────
    ok = oa_miss = errors = 0
    batch: list[dict] = []

    def _flush(batch: list[dict]) -> None:
        with conn:
            for result in batch:
                _write_result(conn, result)

    print(f"\nCrawling {len(todo):,} investigators "
          f"({args.workers} worker(s), {args.delay}s delay) …\n")

    if args.workers == 1:
        # Sequential — simpler, easier to debug
        with tqdm(total=len(todo), unit="PI") as pbar:
            for person in todo:
                result = crawl_one(person, args.delay)
                batch.append(result)
                if result["status"] == "ok":
                    ok += 1
                elif result["status"] == "oa_not_found":
                    oa_miss += 1
                else:
                    errors += 1
                pbar.set_postfix(ok=ok, miss=oa_miss, err=errors)
                pbar.update(1)
                if len(batch) >= args.batch:
                    _flush(batch)
                    batch.clear()
    else:
        # Parallel — multiply delay per worker to stay within OA rate limit
        worker_delay = args.delay * args.workers
        with tqdm(total=len(todo), unit="PI") as pbar:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(crawl_one, p, worker_delay): p for p in todo}
                for fut in as_completed(futs):
                    result = fut.result()
                    batch.append(result)
                    if result["status"] == "ok":
                        ok += 1
                    elif result["status"] == "oa_not_found":
                        oa_miss += 1
                    else:
                        errors += 1
                    pbar.set_postfix(ok=ok, miss=oa_miss, err=errors)
                    pbar.update(1)
                    if len(batch) >= args.batch:
                        _flush(batch)
                        batch.clear()

    if batch:
        _flush(batch)

    total_papers = conn.execute(
        "SELECT COUNT(*) FROM researcher_papers"
    ).fetchone()[0]
    conn.close()

    print(f"\nDone.  OK={ok:,}  OA-miss={oa_miss:,}  Errors={errors:,}")
    print(f"Total paper rows in DB: {total_papers:,}")


if __name__ == "__main__":
    main()
