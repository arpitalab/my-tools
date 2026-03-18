"""
nsf_json_files_ingest.py — Ingest locally-downloaded NSF JSON award files into the DB.

These files use a different schema than the API response (individual per-award JSON
with verbose field names). Designed for supplementing years where the API returned
incomplete data (e.g. 2018).

Usage:
    python nsf_json_files_ingest.py \
        --db     ./output/nsf_awards.db \
        --dir    /Users/sraghava/Downloads/2018 \
        --year   2018
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3

from tqdm import tqdm

from nsf_xml_parser import NSFXMLParser


# ---------------------------------------------------------------------------
# Code normalization
# ---------------------------------------------------------------------------

def normalize_pe(code: str) -> str:
    """Extend 4-digit codes to 6-digit by appending '00'."""
    code = code.strip().upper()
    if len(code) == 4:
        return code + "00"
    return code


# ---------------------------------------------------------------------------
# Field mapping helpers
# ---------------------------------------------------------------------------

def _str(val) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_investigators(pi_list: list) -> list[dict]:
    result = []
    for p in (pi_list or []):
        result.append({
            "nsf_id":     _str(p.get("nsf_id")),
            "full_name":  _str(p.get("pi_full_name")),
            "first_name": _str(p.get("pi_first_name")),
            "last_name":  _str(p.get("pi_last_name")),
            "mid_init":   _str(p.get("pi_mid_init")),
            "suffix":     _str(p.get("pi_sufx_name")),
            "email":      _str(p.get("pi_email_addr")),
            "role_code":  _str(p.get("pi_role")),
            "start_date": _str(p.get("pi_start_date")),
            "end_date":   _str(p.get("pi_end_date")),
        })
    return result


def _parse_institution(inst: dict | None) -> dict:
    if not inst:
        return {}
    return {
        "org_uei_num":         _str(inst.get("org_uei_num")),
        "name":                _str(inst.get("inst_name")),
        "legal_business_name": _str(inst.get("org_lgl_bus_name")),
        "city":                _str(inst.get("inst_city_name")),
        "state_code":          _str(inst.get("inst_state_code")),
        "state_name":          _str(inst.get("inst_state_name")),
        "zip_code":            _str(inst.get("inst_zip_code")),
        "country_name":        _str(inst.get("inst_country_name")),
        "congress_district":   _str(inst.get("cong_dist_code")),
        "parent_uei_num":      _str(inst.get("org_prnt_uei_num")),
    }


def _parse_perf_inst(perf: dict | None) -> dict:
    if not perf:
        return {}
    return {
        "name":              _str(perf.get("perf_inst_name")),
        "city":              _str(perf.get("perf_city_name")),
        "state_code":        _str(perf.get("perf_st_code")),
        "state_name":        _str(perf.get("perf_st_name")),
        "zip_code":          _str(perf.get("perf_zip_code")),
        "country_code":      _str(perf.get("perf_ctry_code")),
        "country_name":      _str(perf.get("perf_ctry_name")),
        "congress_district": _str(perf.get("perf_cong_dist")),
    }


# ---------------------------------------------------------------------------
# Single-award insert
# ---------------------------------------------------------------------------

def _insert_award(parser: NSFXMLParser, conn: sqlite3.Connection, d: dict, source_year: int) -> int | None:
    award_id = _str(d.get("awd_id"))
    if not award_id:
        return None

    # Directorate / division
    dir_abbr   = _str(d.get("dir_abbr"))
    dir_name   = _str(d.get("org_dir_long_name"))
    div_abbr   = _str(d.get("div_abbr"))
    div_name   = _str(d.get("org_div_long_name"))
    dir_id = parser._upsert_directorate(conn, dir_abbr, dir_name) if dir_abbr else None
    div_id = parser._upsert_division(conn, div_abbr, div_name, dir_id) if div_abbr else None

    # Institution
    inst_dict = _parse_institution(d.get("inst") if isinstance(d.get("inst"), dict) else None)
    inst_id = parser._upsert_institution(conn, inst_dict) if inst_dict.get("org_uei_num") else None

    # POR content
    por = d.get("por") or {}
    por_content = _str(por.get("por_cntn"))
    por_text    = _str(por.get("por_txt_cntn"))

    cur = conn.execute(
        """INSERT OR IGNORE INTO award (
            award_id, agency, tran_type, cfda_num, award_instrument, title,
            effective_date, expiration_date,
            total_intended_amount, award_amount,
            min_amd_letter_date, max_amd_letter_date,
            arra_amount, abstract_narration,
            por_content, por_text, org_code,
            directorate_id, division_id, institution_id,
            po_name, po_email, po_phone,
            source_year, source_file
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            award_id,
            _str(d.get("agcy_id")),
            _str(d.get("tran_type")),
            _str(d.get("cfda_num")),
            _str(d.get("awd_istr_txt")),
            _str(d.get("awd_titl_txt")),
            _str(d.get("awd_eff_date")),
            _str(d.get("awd_exp_date")),
            _float(d.get("tot_intn_awd_amt")),
            _float(d.get("awd_amount")),
            _str(d.get("awd_min_amd_letter_date")),
            _str(d.get("awd_max_amd_letter_date")),
            _float(d.get("awd_arra_amount")),
            _str(d.get("awd_abstract_narration")),
            por_content, por_text,
            _str(d.get("org_code")),
            dir_id, div_id, inst_id,
            _str(d.get("po_sign_block_name")),
            _str(d.get("po_email")),
            _str(d.get("po_phone")),
            source_year,
            "local_json",
        ),
    )
    if cur.rowcount == 0:
        return None

    row_id = cur.lastrowid

    # Investigators
    for inv in _parse_investigators(d.get("pi") or []):
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

    # Program elements
    for pe in (d.get("pgm_ele") or []):
        code = _str(pe.get("pgm_ele_code"))
        if not code:
            continue
        code = normalize_pe(code)
        text = _str(pe.get("pgm_ele_name"))
        pe_id = parser._upsert_program_element(conn, code, text)
        conn.execute(
            "INSERT OR IGNORE INTO award_program_element (award_id, program_element_id) VALUES (?,?)",
            (row_id, pe_id),
        )

    # Program references
    for pr in (d.get("pgm_ref") or []):
        code = _str(pr.get("pgm_ref_code"))
        if not code:
            continue
        code = normalize_pe(code)
        text = _str(pr.get("pgm_ref_txt"))
        pr_id = parser._upsert_program_reference(conn, code, text)
        conn.execute(
            "INSERT OR IGNORE INTO award_program_reference (award_id, program_reference_id) VALUES (?,?)",
            (row_id, pr_id),
        )

    # Performance institution
    perf = _parse_perf_inst(d.get("perf_inst") if isinstance(d.get("perf_inst"), dict) else None)
    if perf:
        conn.execute(
            """INSERT OR IGNORE INTO performance_institution
               (award_id, name, city, state_code, state_name, zip_code,
                country_code, country_name, congress_district)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                row_id, perf.get("name"), perf.get("city"), perf.get("state_code"),
                perf.get("state_name"), perf.get("zip_code"), perf.get("country_code"),
                perf.get("country_name"), perf.get("congress_district"),
            ),
        )

    # Funds (app_fund + oblg_fy zipped positionally, matching XML parser logic)
    app_funds = d.get("app_fund") or []
    oblg_fys  = d.get("oblg_fy")  or []
    n = max(len(app_funds), len(oblg_fys))
    for i in range(n):
        af  = app_funds[i] if i < len(app_funds) else {}
        obl = oblg_fys[i]  if i < len(oblg_fys)  else {}
        conn.execute(
            """INSERT INTO award_fund
               (award_id, fund_code, fund_name, fund_symb_id, oblg_year, oblg_amount)
               VALUES (?,?,?,?,?,?)""",
            (
                row_id,
                _str(af.get("fund_code")),
                _str(af.get("fund_name")),
                _str(af.get("fund_symb_id")),
                obl.get("fund_oblg_fiscal_yr"),
                _float(obl.get("fund_oblg_amt")),
            ),
        )

    return row_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(db_path: str, json_dir: str, source_year: int, batch_size: int = 500) -> None:
    parser = NSFXMLParser(xml_dir="", db_path=db_path)
    conn = parser._connect()
    parser._apply_schema(conn)

    # Pre-populate caches
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

    files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]
    print(f"Found {len(files)} JSON files in {json_dir}")

    inserted = skipped = errors = 0
    batch_awards = []

    def flush(batch):
        nonlocal inserted, skipped
        with conn:
            for award_dict in batch:
                row_id = _insert_award(parser, conn, award_dict, source_year)
                if row_id is None:
                    skipped += 1
                else:
                    inserted += 1

    with tqdm(total=len(files), desc=f"Ingesting {source_year}", unit="file") as pbar:
        for fname in files:
            path = os.path.join(json_dir, fname)
            try:
                with open(path) as f:
                    d = json.load(f)
                batch_awards.append(d)
                if len(batch_awards) >= batch_size:
                    flush(batch_awards)
                    batch_awards.clear()
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  Error [{fname}]: {exc}")
                errors += 1
            pbar.update(1)

    if batch_awards:
        flush(batch_awards)

    conn.close()
    print(f"\nDone. Inserted: {inserted}  Skipped (duplicate): {skipped}  Errors: {errors}")

    # Quick dedup check
    conn2 = sqlite3.connect(db_path)
    dups = conn2.execute(
        "SELECT award_id, COUNT(*) c FROM award GROUP BY award_id HAVING c > 1"
    ).fetchall()
    total_2018 = conn2.execute(
        f"SELECT COUNT(*) FROM award WHERE source_year = {source_year}"
    ).fetchone()[0]
    conn2.close()
    print(f"source_year={source_year} total: {total_2018}")
    if dups:
        print(f"WARNING: {len(dups)} duplicate award_ids found — investigate.")
    else:
        print("No duplicate award_ids. DB is clean.")


def parse_args():
    p = argparse.ArgumentParser(description="Ingest local NSF JSON award files into DB")
    p.add_argument("--db",   default="./output/nsf_awards.db")
    p.add_argument("--dir",  required=True, help="Directory of per-award JSON files")
    p.add_argument("--year", type=int, required=True, help="Source year integer (e.g. 2018)")
    p.add_argument("--batch-size", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.db, args.dir, args.year, args.batch_size)
