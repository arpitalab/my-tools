"""
nsf_solr_agent.py — Ollama agent over the NSF SOLR database.

Default: uses ollama native tool-calling (llama3.1, llama3.2, qwen2.5, mistral-nemo).
Fallback: --react flag for models without tool support (phi4, older models).

Usage:
    python nsf_solr_agent.py                               # llama3.2 + native tools
    python nsf_solr_agent.py --model llama3.1              # larger context window
    python nsf_solr_agent.py --model phi4 --react          # ReAct fallback
    python nsf_solr_agent.py --out panel_report.md         # save session to file

Example queries:
    facet all proposals by panel_name for directorate:BIO AND received_year:2024
    get all proposals in panel_name:"BIO-2024-P01" with their titles and PIs
    write a summary report of the top 10 funded CSE proposals in 2023

Requirements:
    pip install ollama pysolr
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import ollama
import pysolr

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
    "fi_pnl": "panel_id, panel_name, panel_reviewers, panel_start_date",
    "fi_snr": "senior_name, senior_inst",
    "fi_int": "foreign_colb_country, intl_actv_flag",
}


def _solr() -> pysolr.Solr:
    return pysolr.Solr(SOLR_URL, timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_proposals(query: str, fields: str = "id,title,summary", rows: int = 5) -> str:
    try:
        results = _solr().search(query, **{"fl": fields, "rows": min(int(rows), 50)})
        if not results.hits:
            return f"No results for: {query}"
        lines = [f"Found {results.hits:,} proposals (showing {len(list(results))}):\n"]
        for doc in results:
            lines.append(f"ID: {doc.get('id','?')}  |  {doc.get('title','(no title)')}")
            for f in fields.split(","):
                f = f.strip()
                if f not in ("id", "title") and f in doc:
                    val = str(doc[f])
                    lines.append(f"  {f}: {val[:400] + '…' if len(val) > 400 else val}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


def get_proposal(proposal_id: str,
                 fields: str = "id,title,summary,description,pi_name,inst,award_amount,directorate,received_year,status") -> str:
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


def fetch_proposals_by_ids(id_list: str,
                           fields: str = "id,title,summary,pi_name,award_amount") -> str:
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
                if f not in ("id", "title") and f in doc:
                    val = str(doc[f])
                    lines.append(f"  {f}: {val[:300] + '…' if len(val) > 300 else val}")
            lines.append("")
        if missing:
            lines.append(f"Not found: {', '.join(missing)}")
        return "\n".join(lines)
    except Exception as e:
        return f"SOLR error (check VPN): {e}"


def facet_proposals(query: str, facet_field: str, limit: int = 15) -> str:
    try:
        results = _solr().search(query, **{
            "rows": 0,
            "facet": "true",
            "facet.field": facet_field,
            "facet.limit": int(limit),
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


def proposal_fields(group: str = "") -> str:
    if group:
        if group not in FIELD_GROUPS:
            return f"Unknown group '{group}'. Valid: {', '.join(FIELD_GROUPS)}"
        return f"{group}: {FIELD_GROUPS[group]}"
    lines = ["Field groups (use field names in search/get_proposal):\n"]
    for code, flds in FIELD_GROUPS.items():
        lines.append(f"  {code:<8}  {flds}")
    return "\n".join(lines)


TOOL_MAP = {
    "search_proposals":       search_proposals,
    "get_proposal":           get_proposal,
    "fetch_proposals_by_ids": fetch_proposals_by_ids,
    "facet_proposals":        facet_proposals,
    "proposal_fields":        proposal_fields,
}

# ---------------------------------------------------------------------------
# Ollama native tool definitions
# ---------------------------------------------------------------------------

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_proposals",
            "description": "Search NSF proposals using Lucene query syntax.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":  {"type": "string",  "description": "Lucene query, e.g. 'pi_name:\"Jane Smith\"' or 'summary:(quantum computing)'"},
                    "fields": {"type": "string",  "description": "comma-separated fields: id,title,summary,description,pi_name,inst,award_amount,directorate,received_year,status"},
                    "rows":   {"type": "integer", "description": "number of results (max 50)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_proposal",
            "description": "Retrieve a single NSF proposal by ID with full text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "string", "description": "NSF proposal ID, e.g. '2535312'"},
                    "fields":      {"type": "string", "description": "comma-separated fields to return"},
                },
                "required": ["proposal_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_proposals_by_ids",
            "description": "Fetch multiple proposals by comma-separated IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id_list": {"type": "string", "description": "comma-separated IDs e.g. '2535312,2535313'"},
                    "fields":  {"type": "string", "description": "fields to return for each proposal"},
                },
                "required": ["id_list"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "facet_proposals",
            "description": "Count top values for a field across matching proposals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "Lucene query selecting proposals to count"},
                    "facet_field": {"type": "string",  "description": "field to break down: directorate, division, inst, inst_state, status, received_year, funding_program"},
                    "limit":       {"type": "integer", "description": "number of top values to show"},
                },
                "required": ["query", "facet_field"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "proposal_fields",
            "description": "List available NSF proposal field groups.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group": {"type": "string", "description": "optional group code e.g. fi_pin, fi_awd"},
                },
            },
        },
    },
]

SYSTEM_MSG = {
    "role": "system",
    "content": """You are an expert assistant for exploring NSF research proposals in SOLR.
Use tools to answer every question — never guess proposal content.

FIELD REFERENCE:
  id, title, summary, description
  pi_name, pi_email, inst, inst_state
  award_amount, award_date, funding_program
  directorate, division, managing_program
  received_year, status
  panel_id, panel_name, panel_reviewers, panel_start_date

LUCENE QUERY EXAMPLES:
  All proposals in a panel:   panel_id:P260135
  Panel name contains string: panel_name:*keyword*
  By PI:                      pi_name:"Jane Smith"
  By keyword in summary:      summary:(quantum computing)
  By directorate + year:      directorate:BIO AND received_year:2024
  By award range:             award_amount:[500000 TO *]
  Awarded only:               status:Awarded

COMMON PATTERNS:
  - "list proposals in panel X"  → search_proposals(query="panel_id:X", fields="id,title,pi_name,status", rows=50)
  - "all funded proposals in X directorate" → search_proposals(query="directorate:X AND status:Awarded", fields="id,title,pi_name,award_amount", rows=50)
  - "breakdown by institution" → facet_proposals(query="...", facet_field="inst")
  - "what panels exist in BIO 2024" → facet_proposals(query="directorate:BIO AND received_year:2024", facet_field="panel_id", limit=50)

If a query returns no results, try wildcard: panel_name:*P260135* instead of panel_name:P260135
""",
}


# ---------------------------------------------------------------------------
# Native tool-calling loop
# ---------------------------------------------------------------------------

def _run_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Unknown tool '{name}'"
    try:
        return fn(**args)
    except TypeError as e:
        return f"Bad arguments for {name}: {e}"


def run_native(model: str, verbose: bool, outfile=None) -> None:
    history: list[dict] = []

    def emit(text: str) -> None:
        print(text)
        if outfile:
            outfile.write(text + "\n")
            outfile.flush()

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
            for t in OLLAMA_TOOLS:
                print(f"  {t['function']['name']}: {t['function']['description']}")
            continue

        if outfile:
            outfile.write(f"\n## You: {user_input}\n\n")
            outfile.flush()

        history.append({"role": "user", "content": user_input})
        messages = [SYSTEM_MSG] + history[-30:]

        for _ in range(16):
            resp    = ollama.chat(model=model, messages=messages, tools=OLLAMA_TOOLS)
            msg     = resp["message"]
            calls   = msg.get("tool_calls") or []

            if not calls:
                answer = msg.get("content", "").strip()
                emit(f"\nAssistant: {answer}\n")
                history.append({"role": "assistant", "content": answer})
                break

            messages.append(msg)
            for tc in calls:
                fn_info = tc.get("function", tc)
                name    = fn_info["name"]
                args    = fn_info.get("arguments", {})
                if verbose:
                    print(f"  [tool] {name}({json.dumps(args)[:120]})")
                result = _run_tool(name, args)
                if verbose:
                    print(f"  [result] {result[:200]}")
                messages.append({"role": "tool", "content": result})
        else:
            print("(reached max steps)\n")

        if len(history) > 40:
            history = history[-40:]


# ---------------------------------------------------------------------------
# ReAct fallback (for models without native tool support, e.g. phi4)
# ---------------------------------------------------------------------------

REACT_SYSTEM = f"""You are an expert assistant for NSF research proposals.
You have these tools:

search_proposals(query, fields="id,title,summary", rows=5)
get_proposal(proposal_id, fields="id,title,summary,description,pi_name,inst,award_amount,directorate,received_year,status")
fetch_proposals_by_ids(id_list, fields="id,title,summary,pi_name,award_amount")
facet_proposals(query, facet_field, limit=15)
proposal_fields(group="")

To call a tool output EXACTLY (no extra text before Action:):
Action: <tool_name>
Action Input: {{"key": "value"}}

After the Observation write either another Action or:
Final Answer: <your answer>
"""


def run_react(model: str, verbose: bool) -> None:
    history: list[dict] = []

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
            for name in TOOL_MAP:
                print(f"  {name}")
            continue

        history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": REACT_SYSTEM}] + history[-20:]

        for step in range(8):
            resp    = ollama.chat(model=model, messages=messages)
            content = resp["message"]["content"]
            if verbose:
                print(f"\n  [step {step+1}] {content[:400]}")

            fa = re.search(r"Final Answer:\s*(.*)", content, re.DOTALL)
            am = re.search(r"Action:\s*(\w+)", content)
            im = re.search(r"Action Input:\s*(\{.*?\})", content, re.DOTALL)

            if fa or not am:
                answer = fa.group(1).strip() if fa else content.strip()
                print(f"\nAssistant: {answer}\n")
                history.append({"role": "assistant", "content": answer})
                break

            name = am.group(1).strip()
            args = {}
            if im:
                try:
                    args = json.loads(im.group(1))
                except json.JSONDecodeError:
                    pass

            if verbose:
                print(f"  [tool] {name}({json.dumps(args)[:120]})")
            result = _run_tool(name, args)
            if verbose:
                print(f"  [obs]  {result[:200]}")

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation: {result}\n\nContinue."})
        else:
            print("(reached max steps)\n")

        if len(history) > 40:
            history = history[-40:]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_repl(model: str = "llama3.2", verbose: bool = False,
             react: bool = False, out: str | None = None) -> None:
    if not verbose:
        sys.stderr = open(os.devnull, "w")

    mode = "react" if react else "native"
    print(f"NSF SOLR Agent  (model={model}  mode={mode}  solr={SOLR_URL})")
    if out:
        print(f"Saving to: {out}")
    print("Type 'quit' to exit, 'tools' to list tools.\n")

    outfile = open(out, "w") if out else None
    if outfile:
        outfile.write(f"# NSF SOLR Session  model={model}\n\n")

    try:
        if react:
            run_react(model, verbose)
        else:
            run_native(model, verbose, outfile=outfile)
    finally:
        if outfile:
            outfile.close()
            print(f"\nSession saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NSF SOLR agent (Ollama)")
    p.add_argument("--model",   default="llama3.2",
                   help="Ollama model (default: llama3.2). Use llama3.1 for larger context.")
    p.add_argument("--verbose", action="store_true",
                   help="Show tool calls and results")
    p.add_argument("--react",   action="store_true",
                   help="Use ReAct text loop (for phi4 and models without native tool support)")
    p.add_argument("--out",     default=None, metavar="FILE",
                   help="Save session to a markdown file e.g. --out panel_report.md")
    args = p.parse_args()
    run_repl(model=args.model, verbose=args.verbose, react=args.react, out=args.out)
