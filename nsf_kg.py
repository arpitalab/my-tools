"""
nsf_kg.py — Build and serialize a NetworkX Knowledge Graph from the NSF awards DB.

Usage:
    python nsf_kg.py \
        --db ./output/nsf_awards.db \
        --output-dir ./output

Outputs:
    nsf_kg.pkl      — NetworkX MultiDiGraph (pickle, HIGHEST_PROTOCOL)
    ontology.json   — Human-readable PE→DIV→DIR hierarchy
"""
from __future__ import annotations

import os
import json
import pickle
import sqlite3
import argparse
from collections import defaultdict

import networkx as nx


# ---------------------------------------------------------------------------
# Node ID helpers
# ---------------------------------------------------------------------------

def dir_key(abbr: str) -> str:
    return f"DIR:{abbr}"

def div_key(abbr: str, dir_abbr: str) -> str:
    return f"DIV:{abbr}:{dir_abbr}"

def pe_key(code: str) -> str:
    return f"PE:{code}"

def pr_key(code: str) -> str:
    return f"PR:{code}"

def awd_key(award_id: str) -> str:
    return f"AWD:{award_id}"

def inv_key(nsf_id: str) -> str:
    return f"INV:{nsf_id}"

def inv_name_key(full_name: str) -> str:
    return f"INV:NAME:{full_name}"

def inst_key(uei: str) -> str:
    return f"INST:{uei}"


# ---------------------------------------------------------------------------
# NSFKnowledgeGraph
# ---------------------------------------------------------------------------

class NSFKnowledgeGraph:

    def __init__(self, db_path: str, output_dir: str):
        self.db_path = db_path
        self.output_dir = output_dir
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        # Lookup maps built during loading (int DB id → node key string)
        self._dir_id_to_key: dict[int, str] = {}
        self._div_id_to_key: dict[int, str] = {}
        self._inst_id_to_key: dict[int, str] = {}
        self._pe_id_to_key: dict[int, str] = {}
        self._pr_id_to_key: dict[int, str] = {}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self) -> nx.MultiDiGraph:
        print("Building NSF Knowledge Graph …")
        print("  Loading taxonomy (directorates / divisions) …")
        self._load_taxonomy()
        print("  Loading programs (PE / PR) …")
        self._load_programs()
        print("  Loading institutions …")
        self._load_institutions()
        print("  Loading investigators …")
        self._load_investigators()
        print("  Loading awards …")
        self._load_awards()
        print("  Building award edges …")
        self._build_award_edges()
        print("  Building investigator edges …")
        self._build_investigator_edges()
        print("  Materializing ontology (PE→DIV inference) …")
        self._materialize_ontology()
        print("  Writing outputs …")
        self._write_outputs()
        print("  Running demo queries …")
        self._demo_queries()
        print(f"\nGraph: {self.G.number_of_nodes():,} nodes, {self.G.number_of_edges():,} edges")
        return self.G

    # ------------------------------------------------------------------
    # Step: taxonomy
    # ------------------------------------------------------------------

    def _load_taxonomy(self) -> None:
        conn = self._connect()
        dirs = conn.execute("SELECT id, abbreviation, long_name FROM directorate").fetchall()
        for row in dirs:
            key = dir_key(row["abbreviation"])
            self.G.add_node(key, type="directorate",
                            abbreviation=row["abbreviation"],
                            long_name=row["long_name"] or "")
            self._dir_id_to_key[row["id"]] = key

        divs = conn.execute(
            "SELECT d.id, d.abbreviation, d.long_name, d.directorate_id, dr.abbreviation AS dir_abbr "
            "FROM division d JOIN directorate dr ON dr.id = d.directorate_id"
        ).fetchall()
        for row in divs:
            key = div_key(row["abbreviation"], row["dir_abbr"])
            self.G.add_node(key, type="division",
                            abbreviation=row["abbreviation"],
                            long_name=row["long_name"] or "",
                            dir_abbr=row["dir_abbr"])
            self._div_id_to_key[row["id"]] = key
            # DIR → DIV edge
            dir_node = self._dir_id_to_key.get(row["directorate_id"])
            if dir_node:
                self.G.add_edge(dir_node, key, rel="CONTAINS")

        conn.close()

    # ------------------------------------------------------------------
    # Step: programs
    # ------------------------------------------------------------------

    def _load_programs(self) -> None:
        conn = self._connect()
        for row in conn.execute("SELECT id, code, text FROM program_element").fetchall():
            key = pe_key(row["code"])
            self.G.add_node(key, type="program_element",
                            code=row["code"], text=row["text"] or "")
            self._pe_id_to_key[row["id"]] = key

        for row in conn.execute("SELECT id, code, text FROM program_reference").fetchall():
            key = pr_key(row["code"])
            self.G.add_node(key, type="program_reference",
                            code=row["code"], text=row["text"] or "")
            self._pr_id_to_key[row["id"]] = key

        conn.close()

    # ------------------------------------------------------------------
    # Step: institutions
    # ------------------------------------------------------------------

    def _load_institutions(self) -> None:
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, org_uei_num, name, city, state_code, country_name FROM institution"
        ).fetchall()
        for row in rows:
            if not row["org_uei_num"]:
                continue
            key = inst_key(row["org_uei_num"])
            self.G.add_node(key, type="institution",
                            org_uei_num=row["org_uei_num"],
                            name=row["name"] or "",
                            city=row["city"] or "",
                            state_code=row["state_code"] or "",
                            country_name=row["country_name"] or "")
            self._inst_id_to_key[row["id"]] = key

        conn.close()

    # ------------------------------------------------------------------
    # Step: investigators (deduplicated)
    # ------------------------------------------------------------------

    def _load_investigators(self) -> None:
        conn = self._connect()
        # Non-empty nsf_id: deduplicate by nsf_id
        rows = conn.execute(
            """
            SELECT
                nsf_id,
                MAX(full_name)  AS full_name,
                MAX(first_name) AS first_name,
                MAX(last_name)  AS last_name,
                MAX(email)      AS email,
                COUNT(DISTINCT award_id) AS award_count
            FROM award_investigator
            WHERE nsf_id IS NOT NULL AND nsf_id != ''
            GROUP BY nsf_id
            """
        ).fetchall()
        for row in rows:
            key = inv_key(row["nsf_id"])
            self.G.add_node(key, type="investigator",
                            nsf_id=row["nsf_id"],
                            full_name=row["full_name"] or "",
                            first_name=row["first_name"] or "",
                            last_name=row["last_name"] or "",
                            email=row["email"] or "",
                            award_count=row["award_count"])

        # Empty nsf_id rows: deduplicate by full_name
        rows2 = conn.execute(
            """
            SELECT
                full_name,
                MAX(first_name) AS first_name,
                MAX(last_name)  AS last_name,
                MAX(email)      AS email,
                COUNT(DISTINCT award_id) AS award_count
            FROM award_investigator
            WHERE (nsf_id IS NULL OR nsf_id = '')
              AND full_name IS NOT NULL AND full_name != ''
            GROUP BY full_name
            """
        ).fetchall()
        for row in rows2:
            key = inv_name_key(row["full_name"])
            self.G.add_node(key, type="investigator",
                            nsf_id="",
                            full_name=row["full_name"],
                            first_name=row["first_name"] or "",
                            last_name=row["last_name"] or "",
                            email=row["email"] or "",
                            award_count=row["award_count"])

        conn.close()

    # ------------------------------------------------------------------
    # Step: awards (chunked)
    # ------------------------------------------------------------------

    def _load_awards(self) -> None:
        conn = self._connect()
        chunk_size = 5000
        offset = 0
        total = 0
        while True:
            rows = conn.execute(
                """
                SELECT
                    a.id AS row_id,
                    a.award_id,
                    a.title,
                    a.source_year,
                    a.award_amount,
                    a.award_instrument,
                    a.cfda_num,
                    a.abstract_narration,
                    a.directorate_id,
                    a.division_id,
                    a.institution_id
                FROM award a
                ORDER BY a.id
                LIMIT ? OFFSET ?
                """,
                (chunk_size, offset),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                key = awd_key(row["award_id"])
                abstract_snippet = (row["abstract_narration"] or "")[:500]
                self.G.add_node(
                    key,
                    type="award",
                    award_id=row["award_id"],
                    title=row["title"] or "",
                    year=row["source_year"],
                    amount=row["award_amount"],
                    instrument=row["award_instrument"] or "",
                    cfda=row["cfda_num"] or "",
                    abstract_snippet=abstract_snippet,
                    dir_id=row["directorate_id"],
                    div_id=row["division_id"],
                    inst_id=row["institution_id"],
                )
            total += len(rows)
            offset += chunk_size
            print(f"    loaded {total:,} awards …", end="\r")
        print(f"    loaded {total:,} awards total.")
        conn.close()

    # ------------------------------------------------------------------
    # Step: award edges
    # ------------------------------------------------------------------

    def _build_award_edges(self) -> None:
        conn = self._connect()

        # --- FUNDED_BY (AWD→DIR) and CLASSIFIED_UNDER (AWD→DIV) + HOSTED_AT (AWD→INST)
        rows = conn.execute(
            "SELECT award_id, directorate_id, division_id, institution_id FROM award"
        ).fetchall()
        for row in rows:
            awd = awd_key(row["award_id"])
            if row["directorate_id"] and row["directorate_id"] in self._dir_id_to_key:
                self.G.add_edge(awd, self._dir_id_to_key[row["directorate_id"]], rel="FUNDED_BY")
            if row["division_id"] and row["division_id"] in self._div_id_to_key:
                self.G.add_edge(awd, self._div_id_to_key[row["division_id"]], rel="CLASSIFIED_UNDER")
            if row["institution_id"] and row["institution_id"] in self._inst_id_to_key:
                self.G.add_edge(awd, self._inst_id_to_key[row["institution_id"]], rel="HOSTED_AT")

        # --- TAGGED_WITH (AWD→PE) from award_program_element
        pe_rows = conn.execute(
            """
            SELECT a.award_id, ape.program_element_id
            FROM award_program_element ape
            JOIN award a ON a.id = ape.award_id
            """
        ).fetchall()
        for row in pe_rows:
            pe_node = self._pe_id_to_key.get(row["program_element_id"])
            if pe_node:
                self.G.add_edge(awd_key(row["award_id"]), pe_node, rel="TAGGED_WITH")

        # --- FLAGGED_WITH (AWD→PR) from award_program_reference
        pr_rows = conn.execute(
            """
            SELECT a.award_id, apr.program_reference_id
            FROM award_program_reference apr
            JOIN award a ON a.id = apr.award_id
            """
        ).fetchall()
        for row in pr_rows:
            pr_node = self._pr_id_to_key.get(row["program_reference_id"])
            if pr_node:
                self.G.add_edge(awd_key(row["award_id"]), pr_node, rel="FLAGGED_WITH")

        conn.close()

    # ------------------------------------------------------------------
    # Step: investigator edges
    # ------------------------------------------------------------------

    def _build_investigator_edges(self) -> None:
        conn = self._connect()

        role_map = {
            "Principal Investigator": "HAS_PI",
            "Co-Principal Investigator": "HAS_CO_PI",
            "Former Principal Investigator": "HAS_FORMER_PI",
            "Former Co-Principal Investigator": "HAS_FORMER_CO_PI",
        }

        inv_rows = conn.execute(
            """
            SELECT ai.award_id AS award_row_id,
                   a.award_id  AS award_str,
                   ai.nsf_id, ai.full_name, ai.role_code,
                   a.institution_id
            FROM award_investigator ai
            JOIN award a ON a.id = ai.award_id
            """
        ).fetchall()

        # Track INV→INST co-occurrence counts for AFFILIATED_WITH edge
        inv_inst_count: dict[tuple[str, str], int] = defaultdict(int)

        for row in inv_rows:
            awd = awd_key(row["award_str"])
            nsf_id = row["nsf_id"] or ""
            inv = inv_key(nsf_id) if nsf_id else inv_name_key(row["full_name"] or "")

            # Skip if investigator node wasn't created (e.g. empty full_name too)
            if not self.G.has_node(inv):
                continue

            rel = role_map.get(row["role_code"], "HAS_CO_PI")
            self.G.add_edge(awd, inv, rel=rel)

            # Track inv→inst for AFFILIATED_WITH
            inst_node = self._inst_id_to_key.get(row["institution_id"])
            if inst_node:
                inv_inst_count[(inv, inst_node)] += 1

        # Build AFFILIATED_WITH edges; mark highest-count as primary
        inv_inst_totals: dict[str, int] = defaultdict(int)
        for (inv, inst), cnt in inv_inst_count.items():
            inv_inst_totals[inv] = max(inv_inst_totals[inv], cnt)

        for (inv, inst), cnt in inv_inst_count.items():
            is_primary = (cnt == inv_inst_totals[inv])
            self.G.add_edge(inv, inst, rel="AFFILIATED_WITH", count=cnt, primary=is_primary)

        conn.close()

    # ------------------------------------------------------------------
    # Step: ontology materialization
    # ------------------------------------------------------------------

    def _materialize_ontology(self) -> None:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT
                v.id AS div_id,
                v.abbreviation AS div_abbr,
                dr.abbreviation AS dir_abbr,
                pe.code AS pe_code,
                COUNT(*) AS weight
            FROM award a
            JOIN award_program_element ape ON ape.award_id = a.id
            JOIN program_element pe ON pe.id = ape.program_element_id
            JOIN division v ON v.id = a.division_id
            JOIN directorate dr ON dr.id = v.directorate_id
            GROUP BY v.id, pe.id
            ORDER BY pe.code, weight DESC
            """
        ).fetchall()
        conn.close()

        # Group by PE → list of (div_id, div_abbr, dir_abbr, weight)
        pe_divs: dict[str, list[tuple]] = defaultdict(list)
        for row in rows:
            pe_divs[row["pe_code"]].append(
                (row["div_id"], row["div_abbr"], row["dir_abbr"], row["weight"])
            )

        for pe_code, div_list in pe_divs.items():
            # Already sorted by weight DESC from SQL
            pe_node = pe_key(pe_code)
            if not self.G.has_node(pe_node):
                continue
            primary_dir_abbr = None
            for rank, (div_id, div_abbr, dir_abbr, weight) in enumerate(div_list[:3]):
                div_node = self._div_id_to_key.get(div_id)
                if not div_node:
                    continue
                is_primary = (rank == 0)
                self.G.add_edge(
                    pe_node, div_node,
                    rel="PRIMARILY_IN",
                    weight=int(weight),
                    rank=rank,
                    is_primary=is_primary,
                )
                if is_primary:
                    primary_dir_abbr = dir_abbr

            if primary_dir_abbr:
                dir_node = dir_key(primary_dir_abbr)
                if self.G.has_node(dir_node):
                    self.G.add_edge(pe_node, dir_node, rel="CLASSIFIED_UNDER_DIR")

    # ------------------------------------------------------------------
    # Step: write outputs
    # ------------------------------------------------------------------

    def _write_outputs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        # Pickle graph (atomic write)
        pkl_path = os.path.join(self.output_dir, "nsf_kg.pkl")
        tmp_path = pkl_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(self.G, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, pkl_path)
        size_mb = os.path.getsize(pkl_path) / 1024 / 1024
        print(f"  Saved {pkl_path}  ({size_mb:.1f} MB)")

        # ontology.json
        ontology = self._build_ontology_json()
        ont_path = os.path.join(self.output_dir, "ontology.json")
        with open(ont_path, "w") as f:
            json.dump(ontology, f, indent=2)
        print(f"  Saved {ont_path}")

    def _build_ontology_json(self) -> dict:
        """
        Structure: {dir_abbr: {div_abbr: [{pe_code, pe_text, primary_count, spans_n_divisions}]}}
        """
        ontology: dict = {}

        for pe_node in self.G.nodes:
            if not pe_node.startswith("PE:"):
                continue
            pe_data = self.G.nodes[pe_node]
            # Find primary DIV edge (rank=0)
            primary_div = None
            primary_weight = 0
            for _, nbr, edata in self.G.out_edges(pe_node, data=True):
                if edata.get("rel") == "PRIMARILY_IN" and edata.get("rank") == 0:
                    primary_div = nbr
                    primary_weight = edata.get("weight", 0)
                    break
            if primary_div is None:
                continue

            div_data = self.G.nodes[primary_div]
            dir_abbr = div_data.get("dir_abbr", "UNKNOWN")
            div_abbr = div_data.get("abbreviation", "UNKNOWN")

            # Count how many divisions this PE spans
            spans = sum(
                1 for _, _, ed in self.G.out_edges(pe_node, data=True)
                if ed.get("rel") == "PRIMARILY_IN"
            )

            entry = {
                "pe_code": pe_data.get("code", ""),
                "pe_text": pe_data.get("text", ""),
                "primary_count": primary_weight,
                "spans_n_divisions": spans,
            }
            ontology.setdefault(dir_abbr, {}).setdefault(div_abbr, []).append(entry)

        # Sort PE entries by primary_count desc
        for dir_abbr in ontology:
            for div_abbr in ontology[dir_abbr]:
                ontology[dir_abbr][div_abbr].sort(key=lambda x: -x["primary_count"])

        return ontology

    # ------------------------------------------------------------------
    # Demo queries
    # ------------------------------------------------------------------

    def _demo_queries(self) -> None:
        G = self.G
        print("\n--- Demo queries ---")

        # 1. Children of DIR:CSE
        cse = dir_key("CSE")
        if G.has_node(cse):
            children = [v for _, v, d in G.out_edges(cse, data=True) if d.get("rel") == "CONTAINS"]
            print(f"1. Divisions under CSE ({len(children)}): {[G.nodes[c].get('abbreviation') for c in children[:8]]}")
        else:
            print("1. DIR:CSE not found.")

        # 2. Top-5 institutions by total award_amount
        inst_totals: dict[str, float] = defaultdict(float)
        for u, v, d in G.edges(data=True):
            if d.get("rel") == "HOSTED_AT" and G.has_node(u) and G.has_node(v):
                amt = G.nodes[u].get("amount") or 0.0
                inst_totals[v] += amt
        top_insts = sorted(inst_totals.items(), key=lambda x: -x[1])[:5]
        print("2. Top-5 institutions by award amount:")
        for inst_node, total in top_insts:
            name = G.nodes[inst_node].get("name", inst_node)
            print(f"     ${total/1e6:.1f}M  {name}")

        # 3. Top-5 PEs by number of divisions spanned
        pe_span: dict[str, int] = {}
        for node in G.nodes:
            if not node.startswith("PE:"):
                continue
            span = sum(1 for _, _, ed in G.out_edges(node, data=True) if ed.get("rel") == "PRIMARILY_IN")
            if span > 0:
                pe_span[node] = span
        top_pes = sorted(pe_span.items(), key=lambda x: -x[1])[:5]
        print("3. Top-5 PEs by division span:")
        for pe_node, span in top_pes:
            text = G.nodes[pe_node].get("text", "")[:50]
            print(f"     {pe_node}  spans={span}  {text}")

        # 4. Top-5 investigators by award_count
        inv_nodes = [(n, d["award_count"]) for n, d in G.nodes(data=True) if d.get("type") == "investigator"]
        inv_nodes.sort(key=lambda x: -x[1])
        print("4. Top-5 investigators by award_count:")
        for inv_node, cnt in inv_nodes[:5]:
            name = G.nodes[inv_node].get("full_name", inv_node)
            print(f"     {name}  ({cnt} awards)")

        # 5. Shortest undirected path DIR:CSE ↔ DIR:BIO
        cse_key = dir_key("CSE")
        bio_key = dir_key("BIO")
        if G.has_node(cse_key) and G.has_node(bio_key):
            try:
                path = nx.shortest_path(G.to_undirected(), cse_key, bio_key)
                print(f"5. Shortest path CSE→BIO (length {len(path)-1}): {' → '.join(path[:6])} …")
            except nx.NetworkXNoPath:
                print("5. No path between DIR:CSE and DIR:BIO.")
        else:
            print("5. DIR:CSE or DIR:BIO not in graph.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build NSF Knowledge Graph from SQLite DB")
    p.add_argument("--db", default="./output/nsf_awards.db")
    p.add_argument("--output-dir", default="./output")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"DB: {args.db}")
    kg = NSFKnowledgeGraph(db_path=args.db, output_dir=args.output_dir)
    kg.build()
    print("\nDone.")


if __name__ == "__main__":
    main()
