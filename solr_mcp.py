"""
solr_mcp.py — MCP server exposing SOLR query tools for NSF/WoS/Scopus data.

Exposes six tools to any MCP-compatible client (Claude Code, Claude Desktop,
or any local LLM agent):
    search        — keyword / field / range queries
    facet         — top-N value counts for any field
    timeseries    — per-year counts or sums (trend analysis)
    stats         — min/max/mean/sum on numeric fields
    get_document  — retrieve one document by id
    schema        — discover available fields
    list_cores    — list available SOLR cores

Setup:
    pip install mcp requests

Configure in ~/.claude/claude_desktop_config.json:
    {
      "mcpServers": {
        "solr": {
          "command": "conda",
          "args": ["run", "-n", "thellmbook", "python",
                   "/Users/sraghava/Desktop/my_llm_explore/solr_mcp.py"],
          "env": {
            "SOLR_URL":  "http://localhost:8983/solr",
            "SOLR_CORE": "nsf_proposals"
          }
        }
      }
    }

Environment variables:
    SOLR_URL     base URL of SOLR instance  (default: http://localhost:8983/solr)
    SOLR_CORE    default core to query      (default: nsf_proposals)
    SOLR_TIMEOUT request timeout in seconds (default: 30)
"""
from __future__ import annotations

import json
import os

import requests
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOLR_URL  = os.environ.get("SOLR_URL",  "http://localhost:8983/solr")
SOLR_CORE = os.environ.get("SOLR_CORE", "nsf_proposals")
TIMEOUT   = int(os.environ.get("SOLR_TIMEOUT", "30"))

mcp = FastMCP(
    "solr-nsf",
    instructions=(
        "Query NSF proposals, Web of Science, and Scopus data stored in SOLR. "
        "Call list_cores() first to see available cores, then schema() to discover "
        "fields, then use search/facet/timeseries to build investigator profiles, "
        "trend reports, and portfolio analyses."
    ),
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(core: str, handler: str, params: dict) -> dict:
    url = f"{SOLR_URL}/{core}/{handler}"
    r = requests.get(url, params={**params, "wt": "json"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _fmt_doc(doc: dict, fields: list[str] | None = None) -> str:
    """Render a SOLR document as indented key: value lines."""
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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search(
    query: str,
    filters: list[str] | None = None,
    fields: list[str] | None = None,
    rows: int = 10,
    sort: str | None = None,
    core: str | None = None,
) -> str:
    """Search SOLR using Lucene query syntax.

    query:   Lucene query string.
             Examples:
               'pi_name:"Jane Smith"'
               'abstract:(quantum AND computing)'
               'year:[2018 TO 2024] AND directorate:CSE'
               '*:*'  (all documents)

    filters: additional filter queries — applied after main query but cached separately.
             Examples: ['source:NSF', 'directorate:BIO', 'award_amount:[100000 TO *]']

    fields:  which fields to return. If None, returns all stored fields.
             Example: ['id', 'pi_name', 'title', 'year', 'award_amount']

    rows:    number of results (max 100).

    sort:    sort expression. Examples: 'citation_count desc', 'year asc'.

    core:    SOLR core name. Defaults to SOLR_CORE env variable.
             Use list_cores() to see what cores are available.
    """
    core = core or SOLR_CORE
    rows = min(rows, 100)

    params: dict = {"q": query, "rows": rows}
    if filters:
        params["fq"] = filters
    if fields:
        params["fl"] = ",".join(fields)
    if sort:
        params["sort"] = sort

    try:
        data = _get(core, "select", params)
    except requests.HTTPError as e:
        return f"SOLR HTTP error {e.response.status_code}: {e.response.text[:400]}"
    except Exception as e:
        return f"SOLR error: {e}"

    resp  = data["response"]
    total = resp["numFound"]
    docs  = resp["docs"]

    if not docs:
        return f"No results found for query: {query}"

    lines = [f"Found {total:,} total documents (showing {len(docs)}):\n"]
    for i, doc in enumerate(docs, 1):
        title = doc.get("title", doc.get("id", "(no title)"))
        lines.append(f"[{i}] {title}")
        lines.append(_fmt_doc(doc, fields))
        lines.append("")

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

    Use this for: top collaborators, top journals, funding by institution,
    directorate breakdown, country distribution, PE code frequency, etc.

    query:       which documents to facet over (Lucene syntax, same as search)
    facet_field: field whose values you want to count
                 Examples: 'co_investigators', 'journal', 'institution',
                           'directorate', 'country', 'pe_code'
    limit:       how many top values to return
    min_count:   omit values with fewer than this many documents
    """
    core = core or SOLR_CORE

    params = {
        "q":              query,
        "rows":           0,
        "facet":          "true",
        "facet.field":    facet_field,
        "facet.limit":    limit,
        "facet.mincount": min_count,
    }
    if filters:
        params["fq"] = filters

    try:
        data = _get(core, "select", params)
    except Exception as e:
        return f"SOLR error: {e}"

    total  = data["response"]["numFound"]
    raw    = data["facet_counts"]["facet_fields"].get(facet_field, [])
    pairs  = list(zip(raw[::2], raw[1::2]))   # [val, count, val, count] → pairs

    if not pairs:
        return f"No facet results for field '{facet_field}' (query matched {total:,} docs)."

    max_count = pairs[0][1]
    lines = [f"Top {len(pairs)} values for '{facet_field}' across {total:,} documents:\n"]
    for value, count in pairs:
        lines.append(f"  {str(value):<42} {count:>7,}  {_bar(count, max_count, 20)}")

    return "\n".join(lines)


@mcp.tool()
def timeseries(
    query: str,
    year_field: str = "year",
    value_field: str | None = None,
    year_start: int = 2010,
    year_end: int = 2024,
    filters: list[str] | None = None,
    core: str | None = None,
) -> str:
    """Get per-year counts or sums — the primary tool for trend analysis.

    year_field:  integer year field to group by.
                 Common values: 'year', 'pub_year', 'award_year'

    value_field: if provided, sum this numeric field per year.
                 Examples: 'award_amount' (total funding per year),
                           'citation_count' (total citations per year).
                 If None, just counts documents per year.

    Examples:
      timeseries('pi_name:"Jane Smith"', value_field='citation_count')
        → Jane's citation trajectory

      timeseries('directorate:BIO', value_field='award_amount', year_start=2015)
        → BIO directorate funding trend since 2015

      timeseries('abstract:(machine learning)', year_field='year')
        → how many ML proposals per year
    """
    core = core or SOLR_CORE

    # Use JSON facet API — handles both count and sum cleanly
    year_facet: dict = {
        "type":  "range",
        "field": year_field,
        "start": year_start,
        "end":   year_end + 1,
        "gap":   1,
    }
    if value_field:
        year_facet["facet"] = {"total": f"sum({value_field})"}

    params: dict = {
        "q":           query,
        "rows":        0,
        "json.facet":  json.dumps({"years": year_facet}),
    }
    if filters:
        params["fq"] = filters

    try:
        data = _get(core, "select", params)
    except Exception as e:
        return f"SOLR error: {e}"

    buckets = data.get("facets", {}).get("years", {}).get("buckets", [])
    buckets = [b for b in buckets if year_start <= b["val"] <= year_end]

    if not buckets:
        return "No timeseries data found for this query and year range."

    if value_field:
        values   = [(b["val"], b.get("total") or 0.0, b.get("count", 0)) for b in buckets]
        max_val  = max(v for _, v, _ in values) or 1
        label    = f"Sum of '{value_field}'"
        lines    = [f"{label} per year (query matched {data['response']['numFound']:,} docs):\n"]
        for yr, total, count in values:
            lines.append(
                f"  {yr}  {total:>16,.1f}  (n={count:>6,})  {_bar(total, max_val)}"
            )
    else:
        values   = [(b["val"], b.get("count", 0)) for b in buckets]
        max_val  = max(c for _, c in values) or 1
        lines    = [f"Document count per year ('{year_field}'):\n"]
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

    stat_fields: list of numeric fields.
                 Examples: ['award_amount', 'citation_count', 'h_index']

    Use for: total funding awarded to a PI, mean citation count of a research area,
             h-index distribution of a reviewer pool, etc.
    """
    core = core or SOLR_CORE

    params: dict = {
        "q":           query,
        "rows":        0,
        "stats":       "true",
        "stats.field": stat_fields,
    }
    if filters:
        params["fq"] = filters

    try:
        data = _get(core, "select", params)
    except Exception as e:
        return f"SOLR error: {e}"

    total       = data["response"]["numFound"]
    field_stats = data["stats"]["stats_fields"]

    lines = [f"Statistics across {total:,} matching documents:\n"]
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
    core: str | None = None,
) -> str:
    """Retrieve a single document by its SOLR id.

    Use this after search() identifies an interesting document and you want
    all of its fields — full proposal text, complete metadata, etc.
    """
    core = core or SOLR_CORE

    try:
        data = _get(core, "select", {"q": f'id:"{doc_id}"', "rows": 1})
    except Exception as e:
        return f"SOLR error: {e}"

    docs = data["response"]["docs"]
    if not docs:
        return f"No document found with id: {doc_id}"

    return f"Document {doc_id}:\n" + _fmt_doc(docs[0])


@mcp.tool()
def schema(
    core: str | None = None,
    filter_prefix: str | None = None,
) -> str:
    """List all fields in the SOLR schema.

    Always call this first in a new session to discover what fields are available
    before constructing queries.

    filter_prefix: only show fields whose name starts with this string.
                   Examples: 'award', 'pub', 'author', 'citation'
    """
    core = core or SOLR_CORE

    try:
        data = _get(core, "schema/fields", {})
    except Exception as e:
        return f"SOLR schema error: {e}"

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
        lines.append(f"  {f['name']:<40} {f.get('type','?'):<22} {flags}")

    return "\n".join(lines)


@mcp.tool()
def list_cores() -> str:
    """List all available SOLR cores and their document counts.

    Use this to find which core holds NSF proposals vs. WoS vs. Scopus,
    or whether everything is in a single unified core.
    """
    try:
        r = requests.get(
            f"{SOLR_URL}/admin/cores",
            params={"action": "STATUS", "wt": "json"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return f"Error listing cores: {e}"

    cores = data.get("status", {})
    if not cores:
        return "No cores found (or SOLR is not reachable)."

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
