"""
nsf_xml_parser.py — NSF XML ingestion front-end.

Parses NSF award XML files into a normalized SQLite database and provides
a query adapter for the BERTopic pipeline in abstract_classifier_2.py.

Usage:
    python nsf_xml_parser.py \
        --xml-dir /Users/sraghava/downloaded_xmls \
        --db ./output/nsf_awards.db \
        --years 2021_xml 2022_xml 2023_xml 2024_xml \
        --batch-size 500
"""

from __future__ import annotations

import os
import sqlite3
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS directorate (
    id           INTEGER PRIMARY KEY,
    abbreviation TEXT NOT NULL UNIQUE,
    long_name    TEXT
);

CREATE TABLE IF NOT EXISTS division (
    id             INTEGER PRIMARY KEY,
    abbreviation   TEXT NOT NULL,
    long_name      TEXT,
    directorate_id INTEGER REFERENCES directorate(id),
    UNIQUE(abbreviation, directorate_id)
);

CREATE TABLE IF NOT EXISTS institution (
    id                  INTEGER PRIMARY KEY,
    org_uei_num         TEXT UNIQUE,
    name                TEXT,
    legal_business_name TEXT,
    city                TEXT,
    state_code          TEXT,
    state_name          TEXT,
    zip_code            TEXT,
    country_name        TEXT,
    congress_district   TEXT,
    parent_uei_num      TEXT
);

CREATE TABLE IF NOT EXISTS program_element (
    id   INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    text TEXT
);

CREATE TABLE IF NOT EXISTS program_reference (
    id   INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    text TEXT
);

CREATE TABLE IF NOT EXISTS award (
    id                    INTEGER PRIMARY KEY,
    award_id              TEXT NOT NULL UNIQUE,
    agency                TEXT,
    tran_type             TEXT,
    cfda_num              TEXT,
    award_instrument      TEXT,
    title                 TEXT,
    effective_date        TEXT,
    expiration_date       TEXT,
    total_intended_amount REAL,
    award_amount          REAL,
    min_amd_letter_date   TEXT,
    max_amd_letter_date   TEXT,
    arra_amount           REAL,
    fund_oblg_total       REAL,
    abstract_narration    TEXT,
    por_content           TEXT,
    por_text              TEXT,
    org_code              TEXT,
    directorate_id        INTEGER REFERENCES directorate(id),
    division_id           INTEGER REFERENCES division(id),
    institution_id        INTEGER REFERENCES institution(id),
    po_name               TEXT,
    po_email              TEXT,
    po_phone              TEXT,
    source_year           INTEGER,
    source_file           TEXT,
    parsed_at             TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS award_investigator (
    id         INTEGER PRIMARY KEY,
    award_id   INTEGER NOT NULL REFERENCES award(id),
    nsf_id     TEXT,
    full_name  TEXT,
    first_name TEXT,
    last_name  TEXT,
    mid_init   TEXT,
    suffix     TEXT,
    email      TEXT,
    role_code  TEXT,
    start_date TEXT,
    end_date   TEXT
);

CREATE TABLE IF NOT EXISTS award_program_element (
    award_id           INTEGER NOT NULL REFERENCES award(id),
    program_element_id INTEGER NOT NULL REFERENCES program_element(id),
    PRIMARY KEY (award_id, program_element_id)
);

CREATE TABLE IF NOT EXISTS award_program_reference (
    award_id             INTEGER NOT NULL REFERENCES award(id),
    program_reference_id INTEGER NOT NULL REFERENCES program_reference(id),
    PRIMARY KEY (award_id, program_reference_id)
);

CREATE TABLE IF NOT EXISTS award_fund (
    id           INTEGER PRIMARY KEY,
    award_id     INTEGER NOT NULL REFERENCES award(id),
    fund_code    TEXT,
    fund_name    TEXT,
    fund_symb_id TEXT,
    oblg_year    INTEGER,
    oblg_amount  REAL
);

CREATE TABLE IF NOT EXISTS performance_institution (
    id                INTEGER PRIMARY KEY,
    award_id          INTEGER NOT NULL UNIQUE REFERENCES award(id),
    name              TEXT,
    city              TEXT,
    state_code        TEXT,
    state_name        TEXT,
    zip_code          TEXT,
    country_code      TEXT,
    country_name      TEXT,
    congress_district TEXT
);

CREATE INDEX IF NOT EXISTS idx_award_directorate ON award(directorate_id);
CREATE INDEX IF NOT EXISTS idx_award_division    ON award(division_id);
CREATE INDEX IF NOT EXISTS idx_award_institution ON award(institution_id);
CREATE INDEX IF NOT EXISTS idx_award_year        ON award(source_year);
CREATE INDEX IF NOT EXISTS idx_award_cfda        ON award(cfda_num);
CREATE INDEX IF NOT EXISTS idx_inv_nsf_id        ON award_investigator(nsf_id);
CREATE INDEX IF NOT EXISTS idx_ape_element       ON award_program_element(program_element_id);
CREATE INDEX IF NOT EXISTS idx_apr_reference     ON award_program_reference(program_reference_id);
CREATE INDEX IF NOT EXISTS idx_inst_state        ON institution(state_code);
"""


# ---------------------------------------------------------------------------
# NSFXMLParser
# ---------------------------------------------------------------------------

class NSFXMLParser:
    """Parses NSF award XML files and writes to a normalized SQLite database."""

    def __init__(self, xml_dir: str, db_path: str, batch_size: int = 500):
        self.xml_dir = xml_dir
        self.db_path = db_path
        self.batch_size = batch_size
        # In-memory caches for dimension tables to avoid repeated SELECTs
        self._dir_cache: dict[str, int] = {}
        self._div_cache: dict[tuple, int] = {}
        self._inst_cache: dict[str, int] = {}
        self._pe_cache: dict[str, int] = {}
        self._pr_cache: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, years: list[str]) -> None:
        conn = self._connect()
        self._apply_schema(conn)

        file_pairs = list(self._iter_xml_files(years))
        total = len(file_pairs)
        print(f"Found {total} XML files across {len(years)} year(s).")

        batch: list[dict] = []
        inserted_total = skipped_total = error_total = 0

        with tqdm(total=total, desc="Parsing XMLs", unit="file") as pbar:
            for path, year in file_pairs:
                record = self._parse_xml_file(path, year)
                if record is None:
                    error_total += 1
                    pbar.update(1)
                    continue
                batch.append(record)
                if len(batch) >= self.batch_size:
                    ins, skip = self._flush_batch(conn, batch)
                    inserted_total += ins
                    skipped_total += skip
                    batch.clear()
                pbar.update(1)

        if batch:
            ins, skip = self._flush_batch(conn, batch)
            inserted_total += ins
            skipped_total += skip

        conn.close()
        print(
            f"\nDone. Inserted: {inserted_total}  Skipped (duplicate): {skipped_total}"
            f"  Errors: {error_total}"
        )

    # ------------------------------------------------------------------
    # File iteration
    # ------------------------------------------------------------------

    def _iter_xml_files(self, years: list[str]):
        for year_dir in years:
            dir_path = os.path.join(self.xml_dir, year_dir)
            if not os.path.isdir(dir_path):
                print(f"Warning: directory not found: {dir_path}")
                continue
            # Extract integer year from folder name like '2021_xml'
            try:
                year_int = int(year_dir.split("_")[0])
            except ValueError:
                year_int = None
            for fname in os.listdir(dir_path):
                if fname.lower().endswith(".xml"):
                    yield os.path.join(dir_path, fname), year_int

    # ------------------------------------------------------------------
    # XML parsing
    # ------------------------------------------------------------------

    def _parse_xml_file(self, path: str, year: int | None) -> dict | None:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            award_el = root.find("Award")
            if award_el is None:
                return None
            return self._extract_award(award_el, year, path)
        except ET.ParseError as exc:
            print(f"XML parse error [{path}]: {exc}")
            return None
        except Exception as exc:
            print(f"Unexpected error [{path}]: {exc}")
            return None

    def _extract_award(self, a: ET.Element, year: int | None, path: str) -> dict:
        txt = self._text  # shorthand

        # --- Core fields ---
        award_id = txt(a, "AwardID")
        title = txt(a, "AwardTitle")
        agency = txt(a, "AGENCY")
        tran_type = txt(a, "TRAN_TYPE")
        cfda_num = txt(a, "CFDA_NUM")
        abstract = txt(a, "AbstractNarration")  # may be None

        # --- Dates ---
        effective_date = self._parse_date(txt(a, "AwardEffectiveDate"))
        expiration_date = self._parse_date(txt(a, "AwardExpirationDate"))
        min_amd = self._parse_date(txt(a, "MinAmdLetterDate"))
        max_amd = self._parse_date(txt(a, "MaxAmdLetterDate"))

        # --- Amounts ---
        total_intended = self._parse_amount(txt(a, "AwardTotalIntnAmount"))
        award_amount = self._parse_amount(txt(a, "AwardAmount"))
        arra_amount = self._parse_amount(txt(a, "ARRAAmount"))

        # --- Instrument ---
        instr_el = a.find("AwardInstrument/Value")
        award_instrument = instr_el.text.strip() if instr_el is not None and instr_el.text else None

        # --- Organization ---
        org_el = a.find("Organization")
        org_code = txt(org_el, "Code") if org_el is not None else None
        dir_abbr = self._text_path(a, "Organization/Directorate/Abbreviation")
        dir_name = self._text_path(a, "Organization/Directorate/LongName")
        div_abbr = self._text_path(a, "Organization/Division/Abbreviation")
        div_name = self._text_path(a, "Organization/Division/LongName")

        # --- Program Officer ---
        po_name = self._text_path(a, "ProgramOfficer/SignBlockName")
        po_email = self._text_path(a, "ProgramOfficer/PO_EMAI")
        po_phone = self._text_path(a, "ProgramOfficer/PO_PHON")

        # --- Institution ---
        inst_el = a.find("Institution")
        institution = self._extract_institution(inst_el) if inst_el is not None else {}

        # --- Performance institution ---
        perf_el = a.find("Performance_Institution")
        perf_inst = self._extract_perf_institution(perf_el) if perf_el is not None else {}

        # --- Investigators ---
        investigators = [self._extract_investigator(el) for el in a.findall("Investigator")]

        # --- Program elements ---
        program_elements = []
        for el in a.findall("ProgramElement"):
            code = txt(el, "Code")
            text = txt(el, "Text")
            if code:
                program_elements.append({"code": code, "text": text})

        # --- Program references ---
        program_references = []
        for el in a.findall("ProgramReference"):
            code = txt(el, "Code")
            text = txt(el, "Text")
            if code:
                program_references.append({"code": code, "text": text})

        # --- Funds ---
        funds = []
        for el in a.findall("Fund"):
            funds.append({
                "code": txt(el, "Code"),
                "name": txt(el, "Name"),
                "symb_id": txt(el, "FUND_SYMB_ID"),
            })

        # --- FUND_OBLG ---
        fund_oblg_entries = []
        fund_oblg_total = 0.0
        for el in a.findall("FUND_OBLG"):
            raw = el.text.strip() if el.text else None
            yr, amt = self._parse_fund_oblg(raw)
            if amt is not None:
                fund_oblg_total += amt
                fund_oblg_entries.append({"year": yr, "amount": amt})

        # --- DRE / POR ---
        por_content = txt(a, "DRECONTENT")
        por_text = txt(a, "POR_COPY_TXT")

        return {
            "award_id": award_id,
            "agency": agency,
            "tran_type": tran_type,
            "cfda_num": cfda_num,
            "award_instrument": award_instrument,
            "title": title,
            "effective_date": effective_date,
            "expiration_date": expiration_date,
            "total_intended_amount": total_intended,
            "award_amount": award_amount,
            "min_amd_letter_date": min_amd,
            "max_amd_letter_date": max_amd,
            "arra_amount": arra_amount,
            "fund_oblg_total": fund_oblg_total if fund_oblg_total else None,
            "abstract_narration": abstract,
            "por_content": por_content,
            "por_text": por_text,
            "org_code": org_code,
            "dir_abbr": dir_abbr,
            "dir_name": dir_name,
            "div_abbr": div_abbr,
            "div_name": div_name,
            "po_name": po_name,
            "po_email": po_email,
            "po_phone": po_phone,
            "institution": institution,
            "perf_institution": perf_inst,
            "investigators": investigators,
            "program_elements": program_elements,
            "program_references": program_references,
            "funds_with_oblg": self._merge_funds_oblg(funds, fund_oblg_entries),
            "source_year": year,
            "source_file": os.path.relpath(path, self.xml_dir),
        }

    def _extract_institution(self, el: ET.Element) -> dict:
        txt = self._text
        return {
            "org_uei_num": txt(el, "ORG_UEI_NUM"),
            "name": txt(el, "Name"),
            "legal_business_name": txt(el, "ORG_LGL_BUS_NAME"),
            "city": txt(el, "CityName"),
            "state_code": txt(el, "StateCode"),
            "state_name": txt(el, "StateName"),
            "zip_code": txt(el, "ZipCode"),
            "country_name": txt(el, "CountryName"),
            "congress_district": txt(el, "CONGRESSDISTRICT"),
            "parent_uei_num": txt(el, "ORG_PRNT_UEI_NUM"),
        }

    def _extract_perf_institution(self, el: ET.Element) -> dict:
        txt = self._text
        return {
            "name": txt(el, "Name"),
            "city": txt(el, "CityName"),
            "state_code": txt(el, "StateCode"),
            "state_name": txt(el, "StateName"),
            "zip_code": txt(el, "ZipCode"),
            "country_code": txt(el, "CountryCode"),
            "country_name": txt(el, "CountryName"),
            "congress_district": txt(el, "CONGRESSDISTRICT"),
        }

    def _extract_investigator(self, el: ET.Element) -> dict:
        txt = self._text
        return {
            "nsf_id": txt(el, "NSF_ID"),
            "full_name": txt(el, "PI_FULL_NAME"),
            "first_name": txt(el, "FirstName"),
            "last_name": txt(el, "LastName"),
            "mid_init": txt(el, "PI_MID_INIT"),
            "suffix": txt(el, "PI_SUFX_NAME"),
            "email": txt(el, "EmailAddress"),
            "role_code": txt(el, "RoleCode"),
            "start_date": self._parse_date(txt(el, "StartDate")),
            "end_date": self._parse_date(txt(el, "EndDate")),
        }

    def _merge_funds_oblg(self, funds: list[dict], oblg_entries: list[dict]) -> list[dict]:
        """
        Funds come from <Fund> elements (code/name/symb_id).
        FUND_OBLG entries carry year + amount.
        We zip them positionally (NSF XMLs always list them in the same order).
        If counts differ, we emit whichever list is longer.
        """
        n = max(len(funds), len(oblg_entries))
        merged = []
        for i in range(n):
            fund = funds[i] if i < len(funds) else {}
            oblg = oblg_entries[i] if i < len(oblg_entries) else {}
            merged.append({
                "fund_code": fund.get("code"),
                "fund_name": fund.get("name"),
                "fund_symb_id": fund.get("symb_id"),
                "oblg_year": oblg.get("year"),
                "oblg_amount": oblg.get("amount"),
            })
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text(el: ET.Element | None, tag: str) -> str | None:
        if el is None:
            return None
        child = el.find(tag)
        if child is None or child.text is None:
            return None
        val = child.text.strip()
        return val if val else None

    @staticmethod
    def _text_path(root: ET.Element, path: str) -> str | None:
        el = root.find(path)
        if el is None or el.text is None:
            return None
        val = el.text.strip()
        return val if val else None

    @staticmethod
    def _parse_date(raw: str | None) -> str | None:
        if not raw:
            return None
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return raw  # return as-is if unparseable

    @staticmethod
    def _parse_amount(raw: str | None) -> float | None:
        if not raw:
            return None
        try:
            return float(raw.replace(",", ""))
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _parse_fund_oblg(raw: str | None) -> tuple[int | None, float | None]:
        if not raw or "~" not in raw:
            return None, None
        parts = raw.split("~", 1)
        try:
            yr = int(parts[0])
        except ValueError:
            yr = None
        try:
            amt = float(parts[1])
        except ValueError:
            amt = None
        return yr, amt

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _apply_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(SCHEMA_SQL)
        conn.commit()

    def _upsert_directorate(self, conn: sqlite3.Connection, abbr: str, long_name: str | None) -> int:
        if abbr in self._dir_cache:
            return self._dir_cache[abbr]
        conn.execute(
            "INSERT OR IGNORE INTO directorate (abbreviation, long_name) VALUES (?, ?)",
            (abbr, long_name),
        )
        row = conn.execute("SELECT id FROM directorate WHERE abbreviation=?", (abbr,)).fetchone()
        row_id = row[0]
        self._dir_cache[abbr] = row_id
        return row_id

    def _upsert_division(
        self, conn: sqlite3.Connection, abbr: str, long_name: str | None, dir_id: int | None
    ) -> int:
        key = (abbr, dir_id)
        if key in self._div_cache:
            return self._div_cache[key]
        conn.execute(
            "INSERT OR IGNORE INTO division (abbreviation, long_name, directorate_id) VALUES (?, ?, ?)",
            (abbr, long_name, dir_id),
        )
        row = conn.execute(
            "SELECT id FROM division WHERE abbreviation=? AND directorate_id IS ?", (abbr, dir_id)
        ).fetchone()
        row_id = row[0]
        self._div_cache[key] = row_id
        return row_id

    def _upsert_institution(self, conn: sqlite3.Connection, inst: dict) -> int | None:
        uei = inst.get("org_uei_num")
        if not uei:
            return None
        if uei in self._inst_cache:
            return self._inst_cache[uei]
        conn.execute(
            """INSERT OR IGNORE INTO institution
               (org_uei_num, name, legal_business_name, city, state_code, state_name,
                zip_code, country_name, congress_district, parent_uei_num)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                uei,
                inst.get("name"),
                inst.get("legal_business_name"),
                inst.get("city"),
                inst.get("state_code"),
                inst.get("state_name"),
                inst.get("zip_code"),
                inst.get("country_name"),
                inst.get("congress_district"),
                inst.get("parent_uei_num"),
            ),
        )
        row = conn.execute("SELECT id FROM institution WHERE org_uei_num=?", (uei,)).fetchone()
        row_id = row[0]
        self._inst_cache[uei] = row_id
        return row_id

    def _upsert_program_element(self, conn: sqlite3.Connection, code: str, text: str | None) -> int:
        if code in self._pe_cache:
            return self._pe_cache[code]
        conn.execute(
            "INSERT OR IGNORE INTO program_element (code, text) VALUES (?, ?)", (code, text)
        )
        row = conn.execute("SELECT id FROM program_element WHERE code=?", (code,)).fetchone()
        row_id = row[0]
        self._pe_cache[code] = row_id
        return row_id

    def _upsert_program_reference(self, conn: sqlite3.Connection, code: str, text: str | None) -> int:
        if code in self._pr_cache:
            return self._pr_cache[code]
        conn.execute(
            "INSERT OR IGNORE INTO program_reference (code, text) VALUES (?, ?)", (code, text)
        )
        row = conn.execute("SELECT id FROM program_reference WHERE code=?", (code,)).fetchone()
        row_id = row[0]
        self._pr_cache[code] = row_id
        return row_id

    # ------------------------------------------------------------------
    # Batch write
    # ------------------------------------------------------------------

    def _flush_batch(self, conn: sqlite3.Connection, batch: list[dict]) -> tuple[int, int]:
        inserted = skipped = 0
        with conn:  # single transaction
            for record in batch:
                row_id = self._insert_award(conn, record)
                if row_id is None:
                    skipped += 1
                else:
                    inserted += 1
                    self._insert_investigators(conn, row_id, record["investigators"])
                    self._insert_junctions(conn, row_id, record)
                    self._insert_perf_institution(conn, row_id, record["perf_institution"])
                    self._insert_funds(conn, row_id, record["funds_with_oblg"])
        return inserted, skipped

    def _insert_award(self, conn: sqlite3.Connection, r: dict) -> int | None:
        if not r.get("award_id"):
            return None

        # Resolve FK ids
        dir_id = (
            self._upsert_directorate(conn, r["dir_abbr"], r.get("dir_name"))
            if r.get("dir_abbr")
            else None
        )
        div_id = (
            self._upsert_division(conn, r["div_abbr"], r.get("div_name"), dir_id)
            if r.get("div_abbr")
            else None
        )
        inst_id = self._upsert_institution(conn, r["institution"]) if r.get("institution") else None

        cur = conn.execute(
            """INSERT OR IGNORE INTO award (
                award_id, agency, tran_type, cfda_num, award_instrument, title,
                effective_date, expiration_date, total_intended_amount, award_amount,
                min_amd_letter_date, max_amd_letter_date, arra_amount, fund_oblg_total,
                abstract_narration, por_content, por_text, org_code,
                directorate_id, division_id, institution_id,
                po_name, po_email, po_phone, source_year, source_file
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                r["award_id"], r.get("agency"), r.get("tran_type"), r.get("cfda_num"),
                r.get("award_instrument"), r.get("title"),
                r.get("effective_date"), r.get("expiration_date"),
                r.get("total_intended_amount"), r.get("award_amount"),
                r.get("min_amd_letter_date"), r.get("max_amd_letter_date"),
                r.get("arra_amount"), r.get("fund_oblg_total"),
                r.get("abstract_narration"), r.get("por_content"), r.get("por_text"),
                r.get("org_code"), dir_id, div_id, inst_id,
                r.get("po_name"), r.get("po_email"), r.get("po_phone"),
                r.get("source_year"), r.get("source_file"),
            ),
        )
        if cur.rowcount == 0:
            return None  # already existed
        return cur.lastrowid

    def _insert_investigators(self, conn: sqlite3.Connection, award_row_id: int, investigators: list) -> None:
        for inv in investigators:
            conn.execute(
                """INSERT INTO award_investigator
                   (award_id, nsf_id, full_name, first_name, last_name,
                    mid_init, suffix, email, role_code, start_date, end_date)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    award_row_id, inv.get("nsf_id"), inv.get("full_name"),
                    inv.get("first_name"), inv.get("last_name"), inv.get("mid_init"),
                    inv.get("suffix"), inv.get("email"), inv.get("role_code"),
                    inv.get("start_date"), inv.get("end_date"),
                ),
            )

    def _insert_junctions(self, conn: sqlite3.Connection, award_row_id: int, record: dict) -> None:
        for pe in record.get("program_elements", []):
            pe_id = self._upsert_program_element(conn, pe["code"], pe.get("text"))
            conn.execute(
                "INSERT OR IGNORE INTO award_program_element (award_id, program_element_id) VALUES (?,?)",
                (award_row_id, pe_id),
            )
        for pr in record.get("program_references", []):
            pr_id = self._upsert_program_reference(conn, pr["code"], pr.get("text"))
            conn.execute(
                "INSERT OR IGNORE INTO award_program_reference (award_id, program_reference_id) VALUES (?,?)",
                (award_row_id, pr_id),
            )

    def _insert_perf_institution(self, conn: sqlite3.Connection, award_row_id: int, perf: dict) -> None:
        if not perf:
            return
        conn.execute(
            """INSERT OR IGNORE INTO performance_institution
               (award_id, name, city, state_code, state_name, zip_code,
                country_code, country_name, congress_district)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                award_row_id, perf.get("name"), perf.get("city"), perf.get("state_code"),
                perf.get("state_name"), perf.get("zip_code"), perf.get("country_code"),
                perf.get("country_name"), perf.get("congress_district"),
            ),
        )

    def _insert_funds(self, conn: sqlite3.Connection, award_row_id: int, funds: list) -> None:
        for f in funds:
            conn.execute(
                """INSERT INTO award_fund
                   (award_id, fund_code, fund_name, fund_symb_id, oblg_year, oblg_amount)
                   VALUES (?,?,?,?,?,?)""",
                (
                    award_row_id, f.get("fund_code"), f.get("fund_name"),
                    f.get("fund_symb_id"), f.get("oblg_year"), f.get("oblg_amount"),
                ),
            )


# ---------------------------------------------------------------------------
# NSFAwardDB — read-only query adapter
# ---------------------------------------------------------------------------

class NSFAwardDB:
    """Read-only query adapter over the nsf_awards.db SQLite database."""

    def __init__(self, db_path: str):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DB not found: {db_path}")
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def get_pipeline_records(
        self,
        year: int | None = None,
        require_abstract: bool = True,
    ) -> list[dict]:
        """
        Returns a list of dicts for the BERTopic pipeline.
        Fields: award_id, abstract, directorate, title,
                division, program_element_codes, program_reference_codes
        If require_abstract=True, only awards with a usable text
        (abstract → por_text → title fallback) are returned.
        """
        conn = self._connect()
        year_filter = "AND a.source_year = ?" if year else ""
        params = [year] if year else []

        rows = conn.execute(
            f"""
            SELECT
                a.id            AS row_id,
                a.award_id,
                a.abstract_narration,
                a.por_text,
                a.title,
                d.abbreviation  AS directorate,
                v.abbreviation  AS division
            FROM award a
            LEFT JOIN directorate d ON d.id = a.directorate_id
            LEFT JOIN division    v ON v.id = a.division_id
            WHERE 1=1 {year_filter}
            """,
            params,
        ).fetchall()

        # Collect PE/PR codes per award in bulk
        pe_map = self._bulk_pe_codes(conn)
        pr_map = self._bulk_pr_codes(conn)

        records = []
        for row in rows:
            abstract = row["abstract_narration"] or row["por_text"] or row["title"]
            if require_abstract and not abstract:
                continue
            records.append(
                {
                    "award_id": row["award_id"],
                    "abstract": abstract,
                    "directorate": row["directorate"] or "",
                    "title": row["title"] or "",
                    "division": row["division"] or "",
                    "program_element_codes": pe_map.get(row["row_id"], []),
                    "program_reference_codes": pr_map.get(row["row_id"], []),
                }
            )
        conn.close()
        return records

    def get_label_records(self) -> list[dict]:
        """
        Returns enriched records with all three label tiers for KG classification.
        Fields: award_id, title, directorate, division, division_name,
                program_elements (list of {code, text}), program_references (list of {code, text}),
                cfda_num, award_instrument, state_code, country_name, source_year
        """
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT
                a.id            AS row_id,
                a.award_id,
                a.title,
                a.cfda_num,
                a.award_instrument,
                a.source_year,
                d.abbreviation  AS directorate,
                v.abbreviation  AS division,
                v.long_name     AS division_name,
                i.state_code,
                i.country_name
            FROM award a
            LEFT JOIN directorate d ON d.id = a.directorate_id
            LEFT JOIN division    v ON v.id = a.division_id
            LEFT JOIN institution i ON i.id = a.institution_id
            """
        ).fetchall()

        pe_full_map = self._bulk_pe_full(conn)
        pr_full_map = self._bulk_pr_full(conn)

        records = []
        for row in rows:
            records.append(
                {
                    "award_id": row["award_id"],
                    "title": row["title"],
                    "directorate": row["directorate"],
                    "division": row["division"],
                    "division_name": row["division_name"],
                    "program_elements": pe_full_map.get(row["row_id"], []),
                    "program_references": pr_full_map.get(row["row_id"], []),
                    "cfda_num": row["cfda_num"],
                    "award_instrument": row["award_instrument"],
                    "state_code": row["state_code"],
                    "country_name": row["country_name"],
                    "source_year": row["source_year"],
                }
            )
        conn.close()
        return records

    # ------------------------------------------------------------------
    # Bulk helpers (avoids N+1 queries)
    # ------------------------------------------------------------------

    def _bulk_pe_codes(self, conn: sqlite3.Connection) -> dict[int, list[str]]:
        rows = conn.execute(
            """SELECT ape.award_id, pe.code
               FROM award_program_element ape
               JOIN program_element pe ON pe.id = ape.program_element_id"""
        ).fetchall()
        result: dict[int, list[str]] = {}
        for r in rows:
            result.setdefault(r[0], []).append(r[1])
        return result

    def _bulk_pr_codes(self, conn: sqlite3.Connection) -> dict[int, list[str]]:
        rows = conn.execute(
            """SELECT apr.award_id, pr.code
               FROM award_program_reference apr
               JOIN program_reference pr ON pr.id = apr.program_reference_id"""
        ).fetchall()
        result: dict[int, list[str]] = {}
        for r in rows:
            result.setdefault(r[0], []).append(r[1])
        return result

    def _bulk_pe_full(self, conn: sqlite3.Connection) -> dict[int, list[dict]]:
        rows = conn.execute(
            """SELECT ape.award_id, pe.code, pe.text
               FROM award_program_element ape
               JOIN program_element pe ON pe.id = ape.program_element_id"""
        ).fetchall()
        result: dict[int, list[dict]] = {}
        for r in rows:
            result.setdefault(r[0], []).append({"code": r[1], "text": r[2]})
        return result

    def _bulk_pr_full(self, conn: sqlite3.Connection) -> dict[int, list[dict]]:
        rows = conn.execute(
            """SELECT apr.award_id, pr.code, pr.text
               FROM award_program_reference apr
               JOIN program_reference pr ON pr.id = apr.program_reference_id"""
        ).fetchall()
        result: dict[int, list[dict]] = {}
        for r in rows:
            result.setdefault(r[0], []).append({"code": r[1], "text": r[2]})
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Parse NSF XML awards into SQLite DB")
    parser.add_argument(
        "--xml-dir",
        default="/Users/sraghava/downloaded_xmls",
        help="Root directory containing year subdirectories",
    )
    parser.add_argument(
        "--db",
        default="./output/nsf_awards.db",
        help="Output SQLite database path",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2021_xml", "2022_xml", "2023_xml", "2024_xml"],
        help="Year subdirectory names to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of records per transaction batch",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parser = NSFXMLParser(
        xml_dir=args.xml_dir,
        db_path=args.db,
        batch_size=args.batch_size,
    )
    parser.run(args.years)
