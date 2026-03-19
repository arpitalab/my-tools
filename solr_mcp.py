"""
solr_mcp.py — MCP server for querying the NSF SOLR database.

Server: http://dis-checker-a01.ad.nsf.gov/solr/{database}/
Requires NSF VPN access.

Available databases (cores):
    proposals   — full NSF proposal text + metadata (default)
    (others discoverable via list_cores())

Tools:
    prop_fields     — list known proposal field groups (start here)
    search          — keyword / field / range queries (Lucene syntax)
    query_with_ids  — batch-fetch proposals by NSF proposal ID
    facet           — top-N value counts for any field
    timeseries      — per-year counts or sums (trend analysis)
    stats           — min/max/mean/sum on numeric fields
    get_document    — retrieve one document by id
    schema          — discover all indexed fields in a core
    list_cores      — list available SOLR cores

Setup:
    pip install mcp pysolr requests

Configure in ~/.claude/claude_desktop_config.json:
    {
      "mcpServers": {
        "solr": {
          "command": "conda",
          "args": ["run", "-n", "thellmbook", "python",
                   "/Users/sraghava/Desktop/my_llm_explore/solr_mcp.py"],
          "env": {
            "SOLR_URL":  "http://dis-checker-a01.ad.nsf.gov/solr",
            "SOLR_CORE": "proposals"
          }
        }
      }
    }

Environment variables:
    SOLR_URL     base URL of SOLR instance  (default: http://dis-checker-a01.ad.nsf.gov/solr)
    SOLR_CORE    default core to query      (default: proposals)
    SOLR_TIMEOUT request timeout in seconds (default: 30)
"""
from __future__ import annotations

import json
import os

import pysolr
import requests
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOLR_URL  = os.environ.get("SOLR_URL",  "http://dis-checker-a01.ad.nsf.gov/solr")
SOLR_CORE = os.environ.get("SOLR_CORE", "proposals")
TIMEOUT   = int(os.environ.get("SOLR_TIMEOUT", "30"))

mcp = FastMCP(
    "solr-nsf",
    instructions=(
        "Query NSF proposal data stored in SOLR at dis-checker-a01.ad.nsf.gov. "
        "Requires NSF VPN. "
        "Start with prop_fields() to see available field groups, then search() or "
        "query_with_ids() to retrieve proposals. Use facet/timeseries/stats for "
        "portfolio and trend analysis. Full proposal text is available in fields: "
        "summary, description, bio, data_management, facilities, mentoring, "
        "references, supplementary, budget."
    ),
)

# ---------------------------------------------------------------------------
# Known proposal field groups (from util_lucern_solr.py)
# ---------------------------------------------------------------------------

_PROP_FIELDS: dict[str, str] = {
    "fi_pid": "id",
    "fi_typ": "lead_id, lead_proposal",
    "fi_dat": "received, received_year, load_date, pm_rcom_date, received_to_rcom_days",
    "fi_stu": "status",
    "fi_tle": "title",
    "fi_sum": "summary",
    "fi_des": "description, description_length",
    "fi_bio": "bio",
    "fi_dmp": "data_management",
    "fi_fac": "facilities",
    "fi_mem": "mentoring",
    "fi_ref": "references",
    "fi_sup": "supplementary",
    "fi_add": "additional_documents",
    "fi_cap": "support",
    "fi_bju": "budget",
    "fi_uni": "directorate, division, managing_program, managing_program_code",
    "fi_req": "natr_rqst_code, rqst_mnth_cnt, requested_amount",
    "fi_sub": "program_announcement, program_director, prop_po_name",
    "fi_coi": "proposal_coi_flag",
    "fi_awd": "award_amount, award_date, awd_exp_date, awd_istr_code, awd_po_name, "
              "dd_rcom_date, funding_program, funding_program_code, funding_program_count",
    "fi_pia": "pi_id, inst_attr, pi_degree, pi_degree_year, pi_race, pi_disability, "
              "pi_ethnicity, pi_gender, pi_project_role",
    "fi_pin": "pi_all, pi_email, pi_name, inst, inst_state, pi_inst, pi_city, pi_state",
    "fi_snr": "senior_name, senior_inst, senior_title",
    "fi_sgr": "suggested_reviewers",
    "fi_pan": "panel_id",
    "fi_pnl": "panel_name, panel_org_code, panel_count, panel_end_date, "
              "panel_reviewers, panel_start_date",
    "fi_rwa": "reviewer_id, reviewer_count, reviewer_disability, reviewer_gender, "
              "reviewer_ethnicity, reviewer_race, reviewer_status",
    "fi_rwn": "reviewer_all, reviewer_name, reviewer_department, reviewer_email, reviewer_inst",
    "fi_cor": "collaborators",
    "fi_int": "foreign_colb_country, foreign_colb_country_code, foreign_country, "
              "foreign_country_code, foreign_country_code_cover, "
              "foreign_country_code_cover_first, foreign_country_code_first, "
              "foreign_country_cover, foreign_funding_cover, foreign_funding_implications, "
              "foreign_non_colb_country, foreign_non_colb_country_code, "
              "intl_actv_flag, nsf_fund_trav_intl_flag",
    "fi_oth": "obj_clas_code, pdf_location, prc_code, summary_length, "
              "deviation_authorization, corpus",
}

_STATUS_AWARDED  = {'Proposal has been awarded', 'Pending, PM recommends award',
                    'Recommended for award, DDConcurred'}
_STATUS_DECLINED = {'Decline, DDConcurred', 'Pending, PM recommends decline'}
_STATUS_PENDING  = {'Pending, Review Package Produced', 'Pending, Assigned to PM'}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _solr(core: str) -> pysolr.Solr:
    return pysolr.Solr(f"{SOLR_URL}/{core}/", timeout=TIMEOUT)


def _admin_get(path: str, params: dict) -> dict:
    """Thin requests wrapper for SOLR admin endpoints pysolr doesn't cover."""
    r = requests.get(f"{SOLR_URL}/{path}", params={**params, "wt": "json"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _fmt_doc(doc: dict, fields: list[str] | None = None) -> str:
    items = doc.items() if not fields else ((k, doc[k]) for k in fields if k in doc)
    lines = []
    for k, v in items:
        if k.startswith("_"):
            continue
        if isinstance(v, list):
            v = "; ".join(str(x) for x in v[:8])
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _bar(value: float, max_value: float, width: int = 28) -> str:
    filled = min(int(value / max(max_value, 1e-9) * width), width)
    return "█" * filled + "░" * (width - filled)


def _normalize_status(status: str) -> str:
    if status in _STATUS_AWARDED:
        return "awarded"
    if status in _STATUS_DECLINED:
        return "declined"
    if status in _STATUS_PENDING:
        return "pending"
    return "other"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def prop_fields(group: str | None = None) -> str:
    """List the known proposal field groups for the NSF proposals database.

    Returns field names grouped by category so you can pick the right fields
    for search() or query_with_ids() without guessing.

    group: optional group code to show only one group. Examples:
           'fi_pin'  → PI name/email/institution fields
           'fi_awd'  → award amount/date/program fields
           'fi_des'  → full project description text
           'fi_sum'  → project summary text
           'fi_pnl'  → panel info and reviewer names

    Leave blank to see all groups.
    """
    group_labels = {
        "fi_pid": "Proposal ID",
        "fi_typ": "Lead proposal linkage",
        "fi_dat": "Dates & timeline",
        "fi_stu": "Status",
        "fi_tle": "Title",
        "fi_sum": "Project summary (full text)",
        "fi_des": "Project description (full text)",
        "fi_bio": "PI biosketch (full text)",
        "fi_dmp": "Data management plan (full text)",
        "fi_fac": "Facilities (full text)",
        "fi_mem": "Mentoring plan (full text)",
        "fi_ref": "References cited (full text)",
        "fi_sup": "Supplementary docs (full text)",
        "fi_add": "Additional documents",
        "fi_cap": "Current & pending support",
        "fi_bju": "Budget justification (full text)",
        "fi_uni": "Directorate / division / program",
        "fi_req": "Requested amount & duration",
        "fi_sub": "Program announcement & PO",
        "fi_coi": "COI flag",
        "fi_awd": "Award outcome & funding",
        "fi_pia": "PI demographics",
        "fi_pin": "PI name / email / institution",
        "fi_snr": "Senior personnel",
        "fi_sgr": "Suggested reviewers",
        "fi_pan": "Panel ID",
        "fi_pnl": "Panel details & reviewers",
        "fi_rwa": "Reviewer attributes",
        "fi_rwn": "Reviewer names & institutions",
        "fi_cor": "Collaborators & affiliations",
        "fi_int": "International activities",
        "fi_oth": "Miscellaneous / corpus",
    }

    if group:
        if group not in _PROP_FIELDS:
            return f"Unknown group '{group}'. Valid codes: {', '.join(_PROP_FIELDS)}"
        return f"{group}  [{group_labels.get(group, '')}]\n  {_PROP_FIELDS[group]}"

    lines = ["Proposal field groups (pass field names to search() fl parameter):\n"]
    for code, fields_str in _PROP_FIELDS.items():
        label = group_labels.get(code, "")
        lines.append(f"  {code}  {label}")
        lines.append(f"        {fields_str}\n")
    return "\n".join(lines)


@mcp.tool()
def search(
    query: str,
    fields: str | None = None,
    filters: list[str] | None = None,
    rows: int = 10,
    sort: str | None = None,
    core: str | None = None,
) -> str:
    """Search the NSF SOLR database using Lucene query syntax.

    query:   Lucene query string. Examples:
               'pi_name:"Jane Smith"'
               'description:(quantum AND computing)'
               'received_year:[2018 TO 2024] AND directorate:BIO'
               'summary:(climate change) AND award_amount:[500000 TO *]'
               '*:*'  (all documents)

    fields:  comma-separated field names to return.
             Use prop_fields() to find valid names.
             Examples:
               'id,title,pi_name,award_amount,status'
               'id,title,summary,description'  (includes full text)

    filters: additional filter queries (cached separately by SOLR for speed).
             Examples: ['directorate:CSE', 'received_year:[2020 TO 2024]']

    rows:    number of results to return (max 200).

    sort:    sort expression. Examples: 'award_amount desc', 'received_year asc'.

    core:    SOLR core name. Default: 'proposals'.
             Use list_cores() to see all available cores.
    """
    core = core or SOLR_CORE
    rows = min(rows, 200)

    kwargs: dict = {"rows": rows}
    if fields:
        kwargs["fl"] = fields
    if filters:
        kwargs["fq"] = filters
    if sort:
        kwargs["sort"] = sort

    try:
        results = _solr(core).search(query, **kwargs)
    except pysolr.SolrError as e:
        return f"SOLR error (are you on NSF VPN?): {e}"
    except Exception as e:
        return f"Connection error: {e}"

    total = results.hits
    docs  = list(results)

    if not docs:
        return f"No results for: {query}"

    field_list = [f.strip() for f in fields.split(",")] if fields else None
    lines = [f"Found {total:,} total documents (showing {len(docs)}):\n"]
    for i, doc in enumerate(docs, 1):
        title = doc.get("title", doc.get("id", "(no title)"))
        status_raw = doc.get("status", "")
        status_tag = f"  [{_normalize_status(status_raw)}]" if status_raw else ""
        lines.append(f"[{i}] {title}{status_tag}")
        lines.append(_fmt_doc(doc, field_list))
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def query_with_ids(
    id_list: list[str],
    fields: str | None = None,
    batch_size: int = 50,
    core: str | None = None,
) -> str:
    """Fetch proposals by NSF proposal/award ID, in efficient batches.

    Use this when you have a known list of IDs (e.g. from your local SQLite DB)
    and want to retrieve full proposal text or metadata from SOLR.

    id_list:    list of NSF proposal IDs as strings.
    fields:     comma-separated fields to return (see prop_fields()).
                Default returns all stored fields.
    batch_size: IDs per SOLR request (default 50, max 100).
    core:       SOLR core name. Default: 'proposals'.
    """
    core       = core or SOLR_CORE
    batch_size = min(batch_size, 100)
    id_list    = [str(i) for i in id_list]
    all_docs   = []

    kwargs: dict = {"rows": batch_size}
    if fields:
        kwargs["fl"] = fields

    try:
        solr = _solr(core)
        for i in range(0, len(id_list), batch_size):
            batch = id_list[i : i + batch_size]
            query = "id:(" + " OR ".join(batch) + ")"
            results = solr.search(query, **kwargs)
            all_docs.extend(results)
    except pysolr.SolrError as e:
        return f"SOLR error (are you on NSF VPN?): {e}"
    except Exception as e:
        return f"Connection error: {e}"

    found_ids   = {str(d.get("id", "")) for d in all_docs}
    missing_ids = set(id_list) - found_ids

    field_list = [f.strip() for f in fields.split(",")] if fields else None
    lines = [f"Retrieved {len(all_docs):,} of {len(id_list):,} requested IDs.\n"]

    for doc in all_docs:
        title = doc.get("title", doc.get("id", "(no title)"))
        status_raw = doc.get("status", "")
        status_tag = f"  [{_normalize_status(status_raw)}]" if status_raw else ""
        lines.append(f"ID {doc.get('id')}  {title}{status_tag}")
        lines.append(_fmt_doc(doc, field_list))
        lines.append("")

    if missing_ids:
        lines.append(f"Not found in SOLR ({len(missing_ids)}): {', '.join(sorted(missing_ids))}")

    return "\n".join(lines)


@mcp.tool()
def facet(
    query: str,
    facet_field: str,
    filters: list[str] | None = None,
    limit: int = 20,
    min_count: int = 1,
    core: str | None = None,
) -> str:
    """Count the top values for a field across all matching documents.

    Use for: award counts by directorate, proposals per institution, funding by
    program, country distribution, reviewer gender breakdown, status breakdown, etc.

    query:       Lucene query (same as search)
    facet_field: field to count values for. Examples:
                 'directorate', 'division', 'managing_program', 'inst',
                 'inst_state', 'funding_program', 'pi_gender', 'pi_race',
                 'foreign_colb_country', 'corpus'
    limit:       how many top values to return
    min_count:   omit values with fewer than this many documents
    """
    core = core or SOLR_CORE

    kwargs = {
        "rows":           0,
        "facet":          "true",
        "facet.field":    facet_field,
        "facet.limit":    limit,
        "facet.mincount": min_count,
    }
    if filters:
        kwargs["fq"] = filters

    try:
        results = _solr(core).search(query, **kwargs)
    except pysolr.SolrError as e:
        return f"SOLR error (are you on NSF VPN?): {e}"
    except Exception as e:
        return f"Connection error: {e}"

    total = results.hits
    raw   = results.facets.get("facet_fields", {}).get(facet_field, [])
    pairs = list(zip(raw[::2], raw[1::2]))

    if not pairs:
        return f"No facet results for '{facet_field}' (query matched {total:,} docs)."

    max_count = pairs[0][1]
    lines = [f"Top {len(pairs)} values for '{facet_field}' across {total:,} documents:\n"]
    for value, count in pairs:
        lines.append(f"  {str(value):<42} {count:>7,}  {_bar(count, max_count, 20)}")

    return "\n".join(lines)


@mcp.tool()
def timeseries(
    query: str,
    year_field: str = "received_year",
    value_field: str | None = None,
    year_start: int = 2010,
    year_end: int = 2024,
    filters: list[str] | None = None,
    core: str | None = None,
) -> str:
    """Get per-year counts or sums — the primary tool for trend analysis.

    year_field:  integer year field to group by.
                 For proposals use 'received_year'. For awards use 'award_date' year.

    value_field: if provided, sum this numeric field per year.
                 Examples: 'award_amount', 'requested_amount'.
                 If None, just counts proposals per year.

    Examples:
      timeseries('pi_name:"Jane Smith"', value_field='award_amount')
        → Jane's total awarded funding by year

      timeseries('directorate:BIO', value_field='award_amount', year_start=2015)
        → BIO directorate funding trend since 2015

      timeseries('description:(machine learning)')
        → how many proposals mention ML per year
    """
    core = core or SOLR_CORE

    year_facet: dict = {
        "type":  "range",
        "field": year_field,
        "start": year_start,
        "end":   year_end + 1,
        "gap":   1,
    }
    if value_field:
        year_facet["facet"] = {"total": f"sum({value_field})"}

    kwargs: dict = {
        "rows":       0,
        "json.facet": json.dumps({"years": year_facet}),
    }
    if filters:
        kwargs["fq"] = filters

    try:
        results = _solr(core).search(query, **kwargs)
    except pysolr.SolrError as e:
        return f"SOLR error (are you on NSF VPN?): {e}"
    except Exception as e:
        return f"Connection error: {e}"

    raw     = results.raw_response
    buckets = raw.get("facets", {}).get("years", {}).get("buckets", [])
    buckets = [b for b in buckets if year_start <= b["val"] <= year_end]

    if not buckets:
        return "No timeseries data found for this query and year range."

    if value_field:
        values  = [(b["val"], b.get("total") or 0.0, b.get("count", 0)) for b in buckets]
        max_val = max(v for _, v, _ in values) or 1
        lines   = [f"Sum of '{value_field}' per year (total docs: {results.hits:,}):\n"]
        for yr, total, count in values:
            lines.append(f"  {yr}  {total:>16,.0f}  (n={count:>6,})  {_bar(total, max_val)}")
    else:
        values  = [(b["val"], b.get("count", 0)) for b in buckets]
        max_val = max(c for _, c in values) or 1
        lines   = [f"Proposal count per '{year_field}' (total docs: {results.hits:,}):\n"]
        for yr, count in values:
            lines.append(f"  {yr}  {count:>8,}  {_bar(count, max_val)}")

    return "\n".join(lines)


@mcp.tool()
def stats(
    query: str,
    stat_fields: list[str],
    filters: list[str] | None = None,
    core: str | None = None,
) -> str:
    """Compute descriptive statistics on numeric fields across matching documents.

    stat_fields: list of numeric fields. Examples:
                 ['award_amount', 'requested_amount', 'received_to_rcom_days',
                  'description_length', 'summary_length', 'funding_program_count']

    Use for: total funding to a PI, mean review time by directorate,
             distribution of requested vs awarded amounts, etc.
    """
    core = core or SOLR_CORE

    kwargs: dict = {
        "rows":        0,
        "stats":       "true",
        "stats.field": stat_fields,
    }
    if filters:
        kwargs["fq"] = filters

    try:
        results = _solr(core).search(query, **kwargs)
    except pysolr.SolrError as e:
        return f"SOLR error (are you on NSF VPN?): {e}"
    except Exception as e:
        return f"Connection error: {e}"

    field_stats = results.raw_response.get("stats", {}).get("stats_fields", {})
    lines = [f"Statistics across {results.hits:,} matching documents:\n"]
    for field, s in field_stats.items():
        if s is None:
            lines.append(f"  {field}: no data\n")
            continue
        lines.append(f"  {field}:")
        lines.append(f"    count  : {s.get('count', 0):>14,}")
        lines.append(f"    sum    : {s.get('sum', 0):>14,.2f}")
        lines.append(f"    mean   : {s.get('mean', 0):>14,.2f}")
        lines.append(f"    min    : {s.get('min', 0):>14,.2f}")
        lines.append(f"    max    : {s.get('max', 0):>14,.2f}")
        lines.append(f"    stddev : {s.get('stddev', 0):>14,.2f}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def get_document(
    doc_id: str,
    fields: str | None = None,
    core: str | None = None,
) -> str:
    """Retrieve a single proposal by its NSF proposal ID.

    Returns all stored fields by default, including full proposal text sections
    (summary, description, bio, budget, etc.).

    fields: optional comma-separated field list to limit the response size.
            Example: 'id,title,summary,description,award_amount'
    """
    core   = core or SOLR_CORE
    kwargs = {"rows": 1}
    if fields:
        kwargs["fl"] = fields

    try:
        results = _solr(core).search(f'id:"{doc_id}"', **kwargs)
    except pysolr.SolrError as e:
        return f"SOLR error (are you on NSF VPN?): {e}"
    except Exception as e:
        return f"Connection error: {e}"

    docs = list(results)
    if not docs:
        return f"No document found with id: {doc_id}"

    field_list = [f.strip() for f in fields.split(",")] if fields else None
    return f"Document {doc_id}:\n" + _fmt_doc(docs[0], field_list)


@mcp.tool()
def schema(
    core: str | None = None,
    filter_prefix: str | None = None,
) -> str:
    """List all indexed fields in the SOLR schema for a given core.

    For the proposals core the known fields are already in prop_fields() —
    call schema() only when you need to verify the live schema or explore
    a non-proposals core.

    filter_prefix: show only fields starting with this string.
                   Examples: 'pi_', 'award_', 'foreign_', 'reviewer_'
    """
    core = core or SOLR_CORE

    try:
        data = _admin_get(f"{core}/schema/fields", {})
    except Exception as e:
        return f"Schema error (are you on NSF VPN?): {e}"

    fields = data.get("fields", [])
    if filter_prefix:
        fields = [f for f in fields if f["name"].startswith(filter_prefix)]

    if not fields:
        return f"No fields found{' with prefix ' + filter_prefix if filter_prefix else ''}."

    lines = [f"Fields in core '{core}' ({len(fields)} total):\n",
             f"  {'name':<40} {'type':<22} flags\n",
             "  " + "-" * 70]
    for f in sorted(fields, key=lambda x: x["name"]):
        flags = "  ".join(x for x in [
            "stored"      if f.get("stored",      True)  else "",
            "indexed"     if f.get("indexed",      True)  else "",
            "multiValued" if f.get("multiValued",  False) else "",
            "docValues"   if f.get("docValues",    False) else "",
        ] if x)
        lines.append(f"  {f['name']:<40} {f.get('type', '?'):<22} {flags}")

    return "\n".join(lines)


@mcp.tool()
def list_cores() -> str:
    """List all available SOLR cores and their document counts.

    Use this to confirm which cores exist (proposals, and potentially
    WoS/Scopus publication cores) before querying.
    """
    try:
        data = _admin_get("admin/cores", {"action": "STATUS"})
    except Exception as e:
        return f"Error listing cores (are you on NSF VPN?): {e}"

    cores = data.get("status", {})
    if not cores:
        return "No cores found."

    lines = [f"Available SOLR cores ({len(cores)}):\n"]
    for name, info in sorted(cores.items()):
        idx     = info.get("index", {})
        n_docs  = idx.get("numDocs", "?")
        size_mb = idx.get("sizeInBytes", 0) // 1024 // 1024
        lines.append(f"  {name:<30} {n_docs:>10,} docs   {size_mb:>6} MB")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
