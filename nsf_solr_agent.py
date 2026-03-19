"""
nsf_solr_agent.py — Ollama agent over the NSF SOLR database.

Uses a simple manual tool-calling loop — no LangChain agent framework,
no version compatibility issues. Just langchain-ollama + pysolr.

Usage:
    python nsf_solr_agent.py
    python nsf_solr_agent.py --model phi4 --verbose

Requirements:
    pip install langchain-ollama pysolr
"""
from __future__ import annotations

import argparse
import json

import pysolr
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOLR_URL = "http://dis-checker-a01.ad.nsf.gov/solr/proposals/"
TIMEOUT  = 30

FIELD_GROUPS = {
    "fi_pid": "id",
    "fi_tle": "title",
    "fi_sum": "summary",
    "fi_des": "description",
    "fi_bio": "bio",
    "fi_dmp": "data_management",
    "fi_bju": "budget",
    "fi_pin": "pi_name, pi_email, inst, inst_state",
    "fi_awd": "award_amount, award_date, funding_program",
    "fi_uni": "directorate, division, managing_program",
    "fi_dat": "received, received_year, status",
    "fi_pnl": "panel_name, panel_reviewers, panel_start_date",
    "fi_snr": "senior_name, senior_inst",
    "fi_int": "foreign_colb_country, intl_actv_flag",
}


def _solr() -> pysolr.Solr:
    return pysolr.Solr(SOLR_URL, timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_proposals(query: str, fields: str = "id,title,summary", rows: int = 5) -> str:
    """Search NSF proposals using Lucene query syntax.
    query: Lucene string e.g. 'pi_name:"Jane Smith"', 'summary:(quantum computing)',
           'directorate:BIO AND received_year:[2020 TO 2024]'
    fields: comma-separated fields — id, title, summary, description, pi_name,
            inst, award_amount, directorate, received_year, status
    rows: number of results (max 50)
    """
    try:
        results = _solr().search(query, **{"fl": fields, "rows": min(rows, 50)})
        if not results.hits:
            return f"No results for: {query}"
        lines = [f"Found {results.hits:,} proposals (showing {len(list(results))}):\n"]
        for doc in results:
            lines.append(f"ID: {doc.get('id','?')}  |  {doc.get('title','(no title)')}")
            for f in fields.split(","):
                f = f.strip()
                if f not in ("id","title") and f in doc:
                    val = str(doc[f])
                    if len(val) > 400:
                        val = val[:400] + "…"
                    lines.append(f"  {f}: {val}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


@tool
def get_proposal(proposal_id: str,
                 fields: str = "id,title,summary,description,pi_name,inst,award_amount,directorate,received_year,status") -> str:
    """Retrieve a single NSF proposal by ID with full text.
    proposal_id: NSF proposal ID e.g. '2535312'
    fields: comma-separated fields to return
    """
    try:
        results = _solr().search(f'id:{proposal_id}', **{"fl": fields, "rows": 1})
        docs = list(results)
        if not docs:
            return f"No proposal found: {proposal_id}"
        doc   = docs[0]
        lines = [f"Proposal {proposal_id}\n"]
        for f in fields.split(","):
            f = f.strip()
            if f in doc:
                lines.append(f"{'─'*40}\n{f.upper()}:\n{doc[f]}")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


@tool
def fetch_proposals_by_ids(id_list: str,
                           fields: str = "id,title,summary,pi_name,award_amount") -> str:
    """Fetch multiple proposals by comma-separated IDs.
    id_list: e.g. '2535312,2535313,2535314'
    fields: fields to return for each proposal
    """
    ids = [i.strip() for i in id_list.split(",") if i.strip()]
    if not ids:
        return "No IDs provided."
    try:
        query   = "id:(" + " OR ".join(ids) + ")"
        results = _solr().search(query, **{"fl": fields, "rows": len(ids)})
        docs    = list(results)
        if not docs:
            return "None of the IDs were found."
        found   = {str(d.get("id")): d for d in docs}
        missing = [i for i in ids if i not in found]
        lines   = [f"Retrieved {len(docs)} of {len(ids)} proposals.\n"]
        for doc in docs:
            lines.append(f"ID {doc.get('id')}  |  {doc.get('title','?')}")
            for f in fields.split(","):
                f = f.strip()
                if f not in ("id","title") and f in doc:
                    val = str(doc[f])
                    lines.append(f"  {f}: {val[:300] + '…' if len(val)>300 else val}")
            lines.append("")
        if missing:
            lines.append(f"Not found: {', '.join(missing)}")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


@tool
def facet_proposals(query: str, facet_field: str, limit: int = 15) -> str:
    """Count top values for a field across matching proposals.
    query: Lucene query selecting which proposals to count over
    facet_field: field to break down — directorate, division, inst,
                 inst_state, status, received_year, funding_program
    limit: number of top values to show
    """
    try:
        results = _solr().search(query, **{
            "rows": 0,
            "facet": "true",
            "facet.field": facet_field,
            "facet.limit": limit,
            "facet.mincount": 1,
        })
        raw   = results.facets.get("facet_fields", {}).get(facet_field, [])
        pairs = list(zip(raw[::2], raw[1::2]))
        if not pairs:
            return f"No facet data for '{facet_field}' (matched {results.hits:,} proposals)"
        max_c = pairs[0][1]
        lines = [f"Top {len(pairs)} values for '{facet_field}' across {results.hits:,} proposals:\n"]
        for val, count in pairs:
            bar = "█" * int(count / max(max_c, 1) * 25)
            lines.append(f"  {str(val):<40} {count:>7,}  {bar}")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


@tool
def proposal_fields(group: str = "") -> str:
    """List available NSF proposal field groups, or details of one group.
    group: optional code e.g. 'fi_pin', 'fi_awd'. Leave empty to list all.
    """
    if group:
        if group not in FIELD_GROUPS:
            return f"Unknown group '{group}'. Valid: {', '.join(FIELD_GROUPS)}"
        return f"{group}: {FIELD_GROUPS[group]}"
    lines = ["Field groups (use field names in search/get_proposal):\n"]
    for code, fields in FIELD_GROUPS.items():
        lines.append(f"  {code:<8}  {fields}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = [search_proposals, get_proposal, fetch_proposals_by_ids,
         facet_proposals, proposal_fields]

TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM = SystemMessage(content="""You are an expert assistant for exploring NSF research
proposals stored in SOLR. Use the available tools to search and retrieve proposal data.

Key fields: summary (project summary), description (full narrative), pi_name, inst,
award_amount, directorate (BIO/CSE/ENG/GEO/MPS/SBE/EDU/TIP), received_year, status.

Always use tools to answer questions — do not guess proposal content.""")


# ---------------------------------------------------------------------------
# Simple tool-calling loop (no AgentExecutor needed)
# ---------------------------------------------------------------------------

def run_repl(model: str = "phi4", verbose: bool = False) -> None:
    llm      = ChatOllama(model=model, temperature=0).bind_tools(TOOLS)
    history  = []

    print(f"NSF SOLR Agent  (model={model}  solr={SOLR_URL})")
    print("Type 'quit' to exit, 'tools' to list tools.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "tools":
            for t in TOOLS:
                print(f"  {t.name}: {t.description.splitlines()[0]}")
            continue

        history.append(HumanMessage(content=user_input))

        # Agentic loop: keep calling LLM until no more tool calls
        for _ in range(6):   # max iterations
            response = llm.invoke([SYSTEM] + history[-20:])

            if not response.tool_calls:
                # Final answer
                print(f"\nAssistant: {response.content}\n")
                history.append(AIMessage(content=response.content))
                break

            # Execute tool calls
            history.append(response)
            for tc in response.tool_calls:
                if verbose:
                    print(f"  [tool] {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:120]})")
                fn     = TOOL_MAP.get(tc["name"])
                result = fn.invoke(tc["args"]) if fn else f"Unknown tool: {tc['name']}"
                if verbose:
                    print(f"  [result] {str(result)[:200]}")
                history.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            print("(reached max iterations)\n")

        # Keep history bounded
        if len(history) > 40:
            history = history[-40:]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NSF SOLR agent (Ollama)")
    p.add_argument("--model",   default="phi4")
    p.add_argument("--verbose", action="store_true")
    args = parse_args() if False else p.parse_args()
    run_repl(model=args.model, verbose=args.verbose)
