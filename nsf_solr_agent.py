"""
nsf_solr_agent.py — LangChain + Ollama agent over the NSF SOLR database.

Mirrors the solr_mcp.py tools but runs locally with phi4 (or any ollama model).
No Claude / no internet required — just NSF VPN + ollama running.

Usage:
    python nsf_solr_agent.py
    python nsf_solr_agent.py --model phi4 --verbose

Requirements:
    pip install langchain langchain-ollama pysolr
"""
from __future__ import annotations

import argparse
import json

import pysolr
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOLR_URL = "http://dis-checker-a01.ad.nsf.gov/solr/proposals/"
TIMEOUT  = 30

# Known field groups (from util_lucern_solr.py)
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

    Args:
        query: Lucene query string. Examples:
               'pi_name:"Jane Smith"'
               'summary:(quantum computing)'
               'directorate:BIO AND received_year:[2020 TO 2024]'
               'description:(machine learning) AND award_amount:[500000 TO *]'
        fields: comma-separated fields to return. Common fields:
                id, title, summary, description, pi_name, inst,
                award_amount, directorate, received_year, status
        rows: number of results to return (default 5, max 50)
    """
    try:
        results = _solr().search(query, **{"fl": fields, "rows": min(rows, 50)})
        if not results.hits:
            return f"No results found for: {query}"

        lines = [f"Found {results.hits:,} total proposals (showing {len(list(results))}):\n"]
        for doc in results:
            lines.append(f"ID: {doc.get('id','?')}  |  {doc.get('title','(no title)')}")
            for f in fields.split(","):
                f = f.strip()
                if f not in ("id", "title") and f in doc:
                    val = doc[f]
                    if isinstance(val, str) and len(val) > 400:
                        val = val[:400] + "…"
                    lines.append(f"  {f}: {val}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


@tool
def get_proposal(proposal_id: str, fields: str = "id,title,summary,description,pi_name,inst,award_amount,directorate,received_year,status") -> str:
    """Retrieve a single NSF proposal by its ID with full text.

    Args:
        proposal_id: NSF proposal/award ID (e.g. '2535312')
        fields: comma-separated fields to return. Use 'summary' for the
                project summary and 'description' for the full narrative.
    """
    try:
        results = _solr().search(f'id:{proposal_id}', **{"fl": fields, "rows": 1})
        docs = list(results)
        if not docs:
            return f"No proposal found with id: {proposal_id}"

        doc   = docs[0]
        lines = [f"Proposal {proposal_id}\n"]
        for f in fields.split(","):
            f = f.strip()
            if f in doc:
                val = doc[f]
                lines.append(f"{'─'*40}")
                lines.append(f"{f.upper()}:")
                lines.append(str(val))
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


@tool
def fetch_proposals_by_ids(id_list: str, fields: str = "id,title,summary,pi_name,award_amount") -> str:
    """Fetch multiple proposals by a comma-separated list of IDs.

    Args:
        id_list: comma-separated proposal IDs e.g. '2535312,2535313,2535314'
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
            return "None of the provided IDs were found."

        found  = {str(d.get("id")): d for d in docs}
        missing = [i for i in ids if i not in found]
        lines  = [f"Retrieved {len(docs)} of {len(ids)} proposals.\n"]

        for doc in docs:
            lines.append(f"ID {doc.get('id')}  |  {doc.get('title','?')}")
            for f in fields.split(","):
                f = f.strip()
                if f not in ("id","title") and f in doc:
                    val = str(doc[f])
                    if len(val) > 300:
                        val = val[:300] + "…"
                    lines.append(f"  {f}: {val}")
            lines.append("")

        if missing:
            lines.append(f"Not found: {', '.join(missing)}")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


@tool
def facet_proposals(query: str, facet_field: str, limit: int = 15) -> str:
    """Count the top values for a field across matching proposals.

    Useful for: breakdown by directorate, division, institution, status,
    PI gender, country, funding program, year, etc.

    Args:
        query: which proposals to facet over (Lucene syntax, same as search)
        facet_field: field to count. Examples:
                     directorate, division, inst, inst_state,
                     status, received_year, funding_program, pi_gender
        limit: how many top values to show (default 15)
    """
    try:
        results = _solr().search(query, **{
            "rows":           0,
            "facet":          "true",
            "facet.field":    facet_field,
            "facet.limit":    limit,
            "facet.mincount": 1,
        })
        raw   = results.facets.get("facet_fields", {}).get(facet_field, [])
        pairs = list(zip(raw[::2], raw[1::2]))

        if not pairs:
            return f"No facet data for field '{facet_field}' (matched {results.hits:,} proposals)"

        max_c = pairs[0][1]
        lines = [f"Top {len(pairs)} values for '{facet_field}' across {results.hits:,} proposals:\n"]
        for val, count in pairs:
            bar = "█" * int(count / max(max_c, 1) * 25)
            lines.append(f"  {str(val):<40} {count:>7,}  {bar}")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


@tool
def proposal_field_groups(group: str = "") -> str:
    """List available field groups for NSF proposals, or details of one group.

    Args:
        group: optional group code (e.g. 'fi_pin', 'fi_awd', 'fi_pnl').
               Leave empty to list all groups.
    """
    if group:
        if group not in FIELD_GROUPS:
            return f"Unknown group '{group}'. Valid: {', '.join(FIELD_GROUPS)}"
        return f"{group}: {FIELD_GROUPS[group]}"

    lines = ["Available field groups (use field names in search/get_proposal):\n"]
    for code, fields in FIELD_GROUPS.items():
        lines.append(f"  {code:<8}  {fields}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

TOOLS = [
    search_proposals,
    get_proposal,
    fetch_proposals_by_ids,
    facet_proposals,
    proposal_field_groups,
]

SYSTEM_PROMPT = """You are an expert assistant for exploring NSF research proposals
stored in a SOLR database. You have access to the full text of NSF proposals including
summaries, project descriptions, PI information, and funding details.

Available tools:
- search_proposals: search using Lucene queries
- get_proposal: retrieve a single proposal by ID with full text
- fetch_proposals_by_ids: batch retrieve multiple proposals
- facet_proposals: count breakdowns by field (directorate, year, institution, etc.)
- proposal_field_groups: list available fields

Tips:
- Use 'summary' for the project summary, 'description' for the full narrative
- Filter by directorate: BIO, CSE, ENG, GEO, MPS, SBE, EDU, TIP
- Filter by year: received_year:[2020 TO 2024]
- Filter by amount: award_amount:[500000 TO *]
- Always check VPN if you get connection errors
"""


def build_agent(model: str = "phi4", verbose: bool = False) -> AgentExecutor:
    llm = ChatOllama(model=model, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=verbose,
        max_iterations=6,
        handle_parsing_errors=True,
    )


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def run_repl(model: str = "phi4", verbose: bool = False) -> None:
    print(f"NSF SOLR Agent  (model: {model})")
    print(f"SOLR: {SOLR_URL}")
    print("Type 'quit' to exit, 'tools' to list available tools.\n")

    executor     = build_agent(model, verbose)
    chat_history = []

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

        try:
            response = executor.invoke({
                "input":        user_input,
                "chat_history": chat_history,
            })
            answer = response["output"]
            print(f"\nAssistant: {answer}\n")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

            # Keep history bounded
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

        except Exception as e:
            print(f"Error: {e}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="NSF SOLR agent (LangChain + Ollama)")
    p.add_argument("--model",   default="phi4",
                   help="Ollama model name (default: phi4)")
    p.add_argument("--verbose", action="store_true",
                   help="Show tool calls and reasoning steps")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_repl(model=args.model, verbose=args.verbose)
