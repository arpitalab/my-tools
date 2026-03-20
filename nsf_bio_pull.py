"""
nsf_bio_pull.py — Pull all BIO directorate proposals from SOLR into a local SQLite DB.

Fetches awarded + declined + pending proposals from directorate:BIO, stores them
in output/bio/nsf_bio.db with the same award table schema as nsf_awards.db so the
existing embedding and BM25 pipelines work unchanged.

After this script completes, run:
    python nsf_embeddings.py \\
        --db output/bio/nsf_bio.db \\
        --output-dir output/bio \\
        --model allenai/specter2 \\
        --suffix bio_specter2

    python nsf_bm25_index.py \\
        --db output/bio/nsf_bio.db \\
        --output-dir output/bio

Usage:
    python nsf_bio_pull.py                  # pull all BIO proposals
    python nsf_bio_pull.py --batch 50       # SOLR batch size (default 50)
    python nsf_bio_pull.py --sleep 1.5      # seconds between batches (default 1.0)
    python nsf_bio_pull.py --reset          # wipe and restart from scratch
    python nsf_bio_pull.py --status awarded # pull only one status (for testing)

Requirements:
    pip install pysolr tqdm
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import pysolr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOLR_URL  = "http://dis-checker-a01.ad.nsf.gov/solr/proposals/"
TIMEOUT   = 60
OUT_DIR   = Path(__file__).parent / "output" / "bio"
DB_PATH   = OUT_DIR / "nsf_bio.db"

# SOLR fields to retrieve
FIELDS = "id,title,summary,description,status,pi_name,inst,received_year,division,panel_id"

# BIO status values — pull all of them
ALL_STATUSES = [
    "Proposal has been awarded",
    "Pending, PM recommends award",
    "Recommended for award, DDConcurred",
    "Decline, DDConcurred",
    "Pending, PM recommends decline",
    "Pending, Review Package Produced",
    "Pending, Assigned to PM",
]

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS award (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    award_id            TEXT    UNIQUE NOT NULL,
    title               TEXT,
    abstract_narration  TEXT,   -- mapped from SOLR summary
    por_text            TEXT,   -- mapped from SOLR description (fallback)
    status              TEXT,
    pi_name             TEXT,
    inst                TEXT,
    received_year       INTEGER,
    division            TEXT,
    panel_id            TEXT
);

CREATE INDEX IF NOT EXISTS idx_award_id    ON award (award_id);
CREATE INDEX IF NOT EXISTS idx_status      ON award (status);
CREATE INDEX IF NOT EXISTS idx_year        ON award (received_year);

CREATE TABLE IF NOT EXISTS pull_progress (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    last_start    INTEGER NOT NULL DEFAULT 0,
    total_fetched INTEGER NOT NULL DEFAULT 0,
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO pull_progress (id, last_start, total_fetched)
VALUES (1, 0, 0);
"""


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_progress(conn: sqlite3.Connection) -> tuple[int, int]:
    row = conn.execute("SELECT last_start, total_fetched FROM pull_progress WHERE id=1").fetchone()
    return row[0], row[1]


def save_progress(conn: sqlite3.Connection, last_start: int, total_fetched: int) -> None:
    conn.execute(
        "UPDATE pull_progress SET last_start=?, total_fetched=?, updated_at=datetime('now') WHERE id=1",
        (last_start, total_fetched),
    )
    conn.commit()


def insert_batch(conn: sqlite3.Connection, docs: list[dict]) -> int:
    """Insert docs, skip duplicates. Returns number of new rows inserted."""
    rows = []
    for doc in docs:
        summary     = doc.get("summary", "")
        description = doc.get("description", "")
        # Flatten lists (SOLR sometimes returns multi-value fields as lists)
        if isinstance(summary, list):
            summary = " ".join(summary)
        if isinstance(description, list):
            description = " ".join(description)
        rows.append((
            str(doc.get("id", "")),
            doc.get("title", ""),
            summary or None,
            description or None,
            doc.get("status", ""),
            doc.get("pi_name", "") if not isinstance(doc.get("pi_name"), list)
                else doc["pi_name"][0] if doc["pi_name"] else "",
            doc.get("inst", ""),
            doc.get("received_year") or None,
            doc.get("division", ""),
            doc.get("panel_id", ""),
        ))
    cursor = conn.executemany(
        """INSERT OR IGNORE INTO award
           (award_id, title, abstract_narration, por_text, status,
            pi_name, inst, received_year, division, panel_id)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Pull loop
# ---------------------------------------------------------------------------

def pull(batch: int = 50, sleep: float = 1.0, status_filter: str = "") -> None:
    conn  = open_db(DB_PATH)
    solr  = pysolr.Solr(SOLR_URL, timeout=TIMEOUT)
    start, total = get_progress(conn)

    # Build query
    if status_filter:
        query = f'directorate:BIO AND status:"{status_filter}"'
    else:
        status_clause = " OR ".join(f'"{s}"' for s in ALL_STATUSES)
        query = f"directorate:BIO AND status:({status_clause})"

    # Get total hit count
    try:
        probe = solr.search(query, **{"rows": 0})
        total_hits = probe.hits
    except Exception as e:
        print(f"SOLR error (check VPN): {e}")
        conn.close()
        return

    print(f"SOLR query: {query}")
    print(f"Total matching proposals: {total_hits:,}")
    print(f"Resuming from start={start}, already fetched={total:,}")
    print(f"Output: {DB_PATH}\n")

    pbar = tqdm(total=total_hits, initial=start, unit="proposals")

    current = start
    while current < total_hits:
        try:
            results = solr.search(query, **{
                "fl":   FIELDS,
                "rows": batch,
                "start": current,
                "sort": "id asc",   # stable ordering for resumability
            })
        except Exception as e:
            print(f"\nSOLR error at start={current}: {e}")
            print("Retrying in 10 seconds …")
            time.sleep(10)
            continue

        docs = list(results)
        if not docs:
            break

        inserted = insert_batch(conn, docs)
        current += len(docs)
        total   += inserted

        save_progress(conn, current, total)
        pbar.update(len(docs))
        pbar.set_postfix({"new": inserted, "total_db": total})

        time.sleep(sleep)

    pbar.close()
    conn.close()

    print(f"\nDone. {total:,} proposals in {DB_PATH}")
    print("\nNext steps:")
    print(f"  python nsf_embeddings.py --db {DB_PATH} --output-dir {OUT_DIR} "
          f"--model allenai/specter2 --suffix bio_specter2")
    print(f"  python nsf_bm25_index.py --db {DB_PATH} --output-dir {OUT_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pull BIO proposals from SOLR into nsf_bio.db")
    p.add_argument("--batch",  type=int,   default=50,
                   help="SOLR batch size (default 50)")
    p.add_argument("--sleep",  type=float, default=1.0,
                   help="Seconds to sleep between batches (default 1.0)")
    p.add_argument("--reset",  action="store_true",
                   help="Delete existing DB and start from scratch")
    p.add_argument("--status", default="",
                   help="Pull only one status value (for testing, e.g. 'Proposal has been awarded')")
    args = p.parse_args()

    if args.reset and DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Deleted {DB_PATH}")

    pull(batch=args.batch, sleep=args.sleep, status_filter=args.status)
