"""
nsf_pe_migrate.py — Normalize 4-digit PE/PR codes to 6-digit in existing DB.

NSF extended program element and reference codes from 4→6 digits in April 2024
by appending '00'. This script normalizes all existing 4-digit codes before new
data arrives, resolving UNIQUE constraint conflicts via FK retargeting (merge)
or simple renames.

Usage:
    python nsf_pe_migrate.py --db ./output/nsf_awards.db
"""

from __future__ import annotations

import argparse
import sqlite3


def migrate_codes(conn: sqlite3.Connection, table: str, junction: str, fk_col: str) -> tuple[int, int]:
    """
    Normalize 4-digit codes in `table` to 6-digit.

    For each 4-digit code "XXXX":
      - If "XXXX00" already exists: merge (retarget all FKs, delete 4-digit row)
      - Else: simple rename (UPDATE code = "XXXX00")

    Returns (renames, collisions).
    """
    renames = collisions = 0

    four_digit_rows = conn.execute(
        f"SELECT id, code FROM {table} WHERE length(code) = 4"
    ).fetchall()

    for row_id, code in four_digit_rows:
        six_code = code.upper() + "00"

        # Check if 6-digit version already exists
        existing = conn.execute(
            f"SELECT id FROM {table} WHERE code = ?", (six_code,)
        ).fetchone()

        if existing:
            # Collision: merge FKs from 4-digit row → 6-digit row
            six_id = existing[0]
            conn.execute(
                f"""UPDATE OR IGNORE {junction}
                    SET {fk_col} = ?
                    WHERE {fk_col} = ?""",
                (six_id, row_id),
            )
            # Delete any remaining FKs that hit the OR IGNORE (already linked to six_id)
            conn.execute(
                f"DELETE FROM {junction} WHERE {fk_col} = ?", (row_id,)
            )
            conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
            collisions += 1
        else:
            # Simple rename
            conn.execute(
                f"UPDATE {table} SET code = ? WHERE id = ?", (six_code, row_id)
            )
            renames += 1

    return renames, collisions


def run(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")

    print("Starting PE/PR code migration (4-digit → 6-digit) ...")

    with conn:  # single transaction
        pe_renames, pe_collisions = migrate_codes(
            conn,
            table="program_element",
            junction="award_program_element",
            fk_col="program_element_id",
        )
        pr_renames, pr_collisions = migrate_codes(
            conn,
            table="program_reference",
            junction="award_program_reference",
            fk_col="program_reference_id",
        )

    # Verify
    bad_pe = conn.execute(
        "SELECT COUNT(*) FROM program_element WHERE length(code) != 6"
    ).fetchone()[0]
    bad_pr = conn.execute(
        "SELECT COUNT(*) FROM program_reference WHERE length(code) != 6"
    ).fetchone()[0]

    conn.close()

    print(f"program_element  : {pe_renames} renames, {pe_collisions} merges")
    print(f"program_reference: {pr_renames} renames, {pr_collisions} merges")
    print(f"Remaining non-6-digit PE codes: {bad_pe}  (should be 0)")
    print(f"Remaining non-6-digit PR codes: {bad_pr}  (should be 0)")

    if bad_pe > 0 or bad_pr > 0:
        print("WARNING: Some codes were not normalized — inspect manually.")
    else:
        print("Migration complete. All codes are 6 digits.")


def parse_args():
    p = argparse.ArgumentParser(description="Normalize PE/PR codes from 4-digit to 6-digit")
    p.add_argument("--db", default="./output/nsf_awards.db", help="SQLite DB path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.db)
