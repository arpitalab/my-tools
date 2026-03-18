"""
nsf_json_ingest.py — Download NSF awards (2010–2020) via the public API and
insert into the existing nsf_awards.db SQLite database.

The 2021–2024 data is already present (XML-sourced). INSERT OR IGNORE on the
award.award_id UNIQUE constraint silently skips any accidental overlaps.

Usage:
    python nsf_json_ingest.py \\
        --db      ./output/nsf_awards.db \\
        --years   2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 \\
        --progress ./output/ingest_progress.json
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import sqlite3
import time
from datetime import datetime

import requests
from tqdm import tqdm

# Reuse dimension-table upsert logic from XML parser
from nsf_xml_parser import NSFXMLParser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "http://api.nsf.gov/services/v1/awards.json"
RECORDS_PER_PAGE = 25
REQUEST_DELAY = 0.5  # seconds between requests


# ---------------------------------------------------------------------------
# PE/PR code normalization
# ---------------------------------------------------------------------------

def normalize_pe(code: str) -> str:
    """Extend 4-digit codes to 6-digit by appending '00'."""
    code = code.strip().upper()
    if len(code) == 4:
        return code + "00"
    return code


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_date(raw: str | None) -> str | None:
    """Convert MM/DD/YYYY or YYYY-MM-DD to YYYY-MM-DD; return None if blank."""
    if not raw:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


def _parse_amount(raw) -> float | None:
    if raw is None:
        return None
    try:
        return float(str(raw).replace(",", ""))
    except (ValueError, TypeError):
        return None


def _year_from_date(raw: str | None) -> int | None:
    """Extract 4-digit year from MM/DD/YYYY."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), "%m/%d/%Y").year
    except ValueError:
        pass
    try:
        return int(raw.strip()[:4])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Investigator parsing
# ---------------------------------------------------------------------------

_COPI_RE = re.compile(r"^([^(]+?)\s*(?:\(([^)]*)\))?$")


def _parse_investigators(award: dict) -> list[dict]:
    """
    Returns a list of investigator dicts compatible with award_investigator schema.
    The JSON API provides the PI directly and co-PIs as a list of strings.
    """
    investigators = []

    # Primary PI
    pi_first = (award.get("piFirstName") or "").strip()
    pi_last  = (award.get("piLastName")  or "").strip()
    pi_mid   = (award.get("piMiddeInitial") or "").strip()  # NSF typo: 'piMiddeInitial'
    pi_email = (award.get("piEmail") or "").strip()
    if pi_first or pi_last:
        full = " ".join(p for p in [pi_first, pi_mid, pi_last] if p)
        investigators.append({
            "nsf_id": "",
            "full_name": full or None,
            "first_name": pi_first or None,
            "last_name": pi_last or None,
            "mid_init": pi_mid or None,
            "suffix": None,
            "email": pi_email or None,
            "role_code": "Principal Investigator",
            "start_date": None,
            "end_date": None,
        })

    # Co-PIs
    for copi_raw in (award.get("coPDPI") or []):
        if not isinstance(copi_raw, str):
            continue
        m = _COPI_RE.match(copi_raw.strip())
        if not m:
            continue
        name_part = m.group(1).strip()
        email_part = (m.group(2) or "").strip()
        # "Lastname, Firstname" or just "Firstname Lastname"
        if "," in name_part:
            parts = name_part.split(",", 1)
            last = parts[0].strip()
            first = parts[1].strip()
        else:
            words = name_part.split()
            first = words[0] if words else ""
            last  = " ".join(words[1:]) if len(words) > 1 else ""
        full = f"{first} {last}".strip() if first or last else name_part
        investigators.append({
            "nsf_id": "",
            "full_name": full or None,
            "first_name": first or None,
            "last_name": last or None,
            "mid_init": None,
            "suffix": None,
            "email": email_part or None,
            "role_code": "Co-Principal Investigator",
            "start_date": None,
            "end_date": None,
        })

    return investigators


# ---------------------------------------------------------------------------
# API fetching
# ---------------------------------------------------------------------------

def _fetch_page(session: requests.Session, params: dict) -> list[dict]:
    """Fetch one page from the NSF API; return list of award dicts."""
    try:
        resp = session.get(API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", {}).get("award", []) or []
    except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
        print(f"  API error: {exc}")
        return []


def fetch_month(
    session: requests.Session,
    year: int,
    month: int,
    delay: float = REQUEST_DELAY,
) -> list[dict]:
    """
    Fetch all awards for a given YYYY-MM via paginated API calls.
    Returns a flat list of raw award dicts.
    """
    last_day = calendar.monthrange(year, month)[1]
    date_start = f"{month:02d}/01/{year}"
    date_end   = f"{month:02d}/{last_day:02d}/{year}"

    all_awards: list[dict] = []
    offset = 1

    while True:
        params = {
            "dateStart": date_start,
            "dateEnd":   date_end,
            "rpp":       RECORDS_PER_PAGE,
            "offset":    offset,
            "printFields": (
                "id,title,abstractText,transType,cfdaNumber,"
                "estimatedTotalAmt,fundsObligatedAmt,"
                "startDate,expDate,date,"
                "initAmendmentDate,latestAmendmentDate,"
                "dirAbbr,divAbbr,"
                "ueiNumber,awardeeName,"
                "awardeeCity,awardeeStateCode,awardeeZipCode,awardeeCountryCode,"
                "poName,poEmail,poPhone,"
                "progEleCode,progRefCode,"
                "piFirstName,piLastName,piMiddeInitial,piEmail,pdPIName,"
                "coPDPI"
            ),
        }
        time.sleep(delay)
        page = _fetch_page(session, params)
        if not page:
            break
        all_awards.extend(page)
        if len(page) < RECORDS_PER_PAGE:
            break
        offset += RECORDS_PER_PAGE

    return all_awards


# ---------------------------------------------------------------------------
# DB insertion
# ---------------------------------------------------------------------------

def _insert_awards(
    parser: NSFXMLParser,
    conn: sqlite3.Connection,
    awards: list[dict],
) -> tuple[int, int]:
    """Insert a batch of raw API award dicts into the DB. Returns (inserted, skipped)."""
    inserted = skipped = 0
    with conn:
        for award in awards:
            row_id = _insert_single(parser, conn, award)
            if row_id is None:
                skipped += 1
            else:
                inserted += 1
    return inserted, skipped


def _insert_single(
    parser: NSFXMLParser,
    conn: sqlite3.Connection,
    award: dict,
) -> int | None:
    """Map one API award dict to DB rows. Returns award.id (rowid) or None if duplicate."""
    award_id_str = (award.get("id") or "").strip()
    if not award_id_str:
        return None

    # Directorate / division FKs
    dir_abbr = (award.get("dirAbbr") or "").strip() or None
    div_abbr = (award.get("divAbbr") or "").strip() or None

    dir_id = parser._upsert_directorate(conn, dir_abbr, None) if dir_abbr else None
    div_id = parser._upsert_division(conn, div_abbr, None, dir_id) if div_abbr else None

    # Institution
    uei = (award.get("ueiNumber") or "").strip() or None
    inst_dict = {
        "org_uei_num": uei,
        "name": (award.get("awardeeName") or "").strip() or None,
        "legal_business_name": None,
        "city": (award.get("awardeeCity") or "").strip() or None,
        "state_code": (award.get("awardeeStateCode") or "").strip() or None,
        "state_name": None,
        "zip_code": (award.get("awardeeZipCode") or "").strip() or None,
        "country_name": (award.get("awardeeCountryCode") or "").strip() or None,
        "congress_district": None,
        "parent_uei_num": None,
    }
    inst_id = parser._upsert_institution(conn, inst_dict) if uei else None

    # Dates / amounts
    effective_date    = _parse_date(award.get("startDate"))
    expiration_date   = _parse_date(award.get("expDate"))
    min_amd           = _parse_date(award.get("initAmendmentDate"))
    max_amd           = _parse_date(award.get("latestAmendmentDate"))
    total_intended    = _parse_amount(award.get("estimatedTotalAmt"))
    award_amount      = _parse_amount(award.get("fundsObligatedAmt"))
    source_year       = _year_from_date(award.get("date"))

    cur = conn.execute(
        """INSERT OR IGNORE INTO award (
            award_id, tran_type, cfda_num, title,
            effective_date, expiration_date,
            total_intended_amount, award_amount,
            min_amd_letter_date, max_amd_letter_date,
            abstract_narration,
            directorate_id, division_id, institution_id,
            po_name, po_email, po_phone,
            source_year, source_file
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            award_id_str,
            (award.get("transType") or "").strip() or None,
            (award.get("cfdaNumber") or "").strip() or None,
            (award.get("title") or "").strip() or None,
            effective_date, expiration_date,
            total_intended, award_amount,
            min_amd, max_amd,
            (award.get("abstractText") or "").strip() or None,
            dir_id, div_id, inst_id,
            (award.get("poName")  or "").strip() or None,
            (award.get("poEmail") or "").strip() or None,
            (award.get("poPhone") or "").strip() or None,
            source_year,
            "api",  # source_file marker
        ),
    )
    if cur.rowcount == 0:
        return None

    row_id = cur.lastrowid

    # Investigators
    for inv in _parse_investigators(award):
        conn.execute(
            """INSERT INTO award_investigator
               (award_id, nsf_id, full_name, first_name, last_name,
                mid_init, suffix, email, role_code, start_date, end_date)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row_id, inv["nsf_id"], inv["full_name"],
                inv["first_name"], inv["last_name"], inv["mid_init"],
                inv["suffix"], inv["email"], inv["role_code"],
                inv["start_date"], inv["end_date"],
            ),
        )

    # Program elements (comma-separated in the JSON)
    raw_pe = (award.get("progEleCode") or "").strip()
    if raw_pe:
        for code in raw_pe.split(","):
            code = code.strip()
            if not code:
                continue
            code = normalize_pe(code)
            pe_id = parser._upsert_program_element(conn, code, None)
            conn.execute(
                "INSERT OR IGNORE INTO award_program_element (award_id, program_element_id) VALUES (?,?)",
                (row_id, pe_id),
            )

    # Program references
    raw_pr = (award.get("progRefCode") or "").strip()
    if raw_pr:
        for code in raw_pr.split(","):
            code = code.strip()
            if not code:
                continue
            code = normalize_pe(code)
            pr_id = parser._upsert_program_reference(conn, code, None)
            conn.execute(
                "INSERT OR IGNORE INTO award_program_reference (award_id, program_reference_id) VALUES (?,?)",
                (row_id, pr_id),
            )

    return row_id


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def _load_progress(path: str) -> set[str]:
    """Return set of completed 'YYYY-MM' strings."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
        return set(data.get("completed", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def _save_progress(path: str, completed: set[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"completed": sorted(completed)}, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(db_path: str, years: list[int], progress_path: str) -> None:
    # Set up DB connection + schema
    parser = NSFXMLParser(xml_dir="", db_path=db_path)
    conn = parser._connect()
    parser._apply_schema(conn)

    # Pre-populate caches from existing DB to avoid redundant SELECTs
    for abbr, row_id in conn.execute("SELECT abbreviation, id FROM directorate"):
        parser._dir_cache[abbr] = row_id
    for abbr, dir_id, row_id in conn.execute("SELECT abbreviation, directorate_id, id FROM division"):
        parser._div_cache[(abbr, dir_id)] = row_id
    for uei, row_id in conn.execute("SELECT org_uei_num, id FROM institution WHERE org_uei_num IS NOT NULL"):
        parser._inst_cache[uei] = row_id
    for code, row_id in conn.execute("SELECT code, id FROM program_element"):
        parser._pe_cache[code] = row_id
    for code, row_id in conn.execute("SELECT code, id FROM program_reference"):
        parser._pr_cache[code] = row_id

    completed = _load_progress(progress_path)

    # Build work list: all (year, month) pairs not yet completed
    work = [
        (y, m)
        for y in sorted(years)
        for m in range(1, 13)
        if f"{y}-{m:02d}" not in completed
    ]

    total_inserted = total_skipped = 0
    session = requests.Session()
    session.headers.update({"User-Agent": "nsf-json-ingest/1.0"})

    print(f"Fetching {len(work)} month(s) across {len(years)} year(s) ...")
    with tqdm(total=len(work), desc="Months", unit="mo") as pbar:
        for year, month in work:
            chunk_key = f"{year}-{month:02d}"
            awards = fetch_month(session, year, month)
            inserted, skipped = _insert_awards(parser, conn, awards)
            total_inserted += inserted
            total_skipped  += skipped
            completed.add(chunk_key)
            _save_progress(progress_path, completed)
            pbar.set_postfix(ins=total_inserted, skip=total_skipped)
            pbar.update(1)

    conn.close()
    print(f"\nDone. Inserted: {total_inserted}  Skipped (duplicate): {total_skipped}")


def parse_args():
    p = argparse.ArgumentParser(description="Ingest NSF awards (2010–2020) via JSON API")
    p.add_argument("--db",       default="./output/nsf_awards.db")
    p.add_argument("--years",    nargs="+", type=int,
                   default=list(range(2010, 2021)),
                   help="Calendar years to download")
    p.add_argument("--progress", default="./output/ingest_progress.json",
                   help="JSON file for resume state")
    p.add_argument("--delay",    type=float, default=REQUEST_DELAY,
                   help="Seconds between API requests (default: 0.5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    REQUEST_DELAY = args.delay  # allow CLI override
    run(args.db, args.years, args.progress)
