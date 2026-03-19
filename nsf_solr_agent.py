"""
nsf_solr_agent.py — Ollama ReAct agent over the NSF SOLR database.

Uses a ReAct (Reason + Act) text loop — works with ANY Ollama model
including phi4, llama3, mistral etc. No native tool-calling required.

Usage:
    python nsf_solr_agent.py
    python nsf_solr_agent.py --model phi4 --verbose

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
    "fi_pnl": "panel_name, panel_reviewers, panel_start_date",
    "fi_snr": "senior_name, senior_inst",
    "fi_int": "foreign_colb_country, intl_actv_flag",
}


def _solr() -> pysolr.Solr:
    return pysolr.Solr(SOLR_URL, timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# Tools (plain functions — no decorator needed)
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
                    if len(val) > 400:
                        val = val[:400] + "…"
                    lines.append(f"  {f}: {val}")
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


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_proposals":      search_proposals,
    "get_proposal":          get_proposal,
    "fetch_proposals_by_ids": fetch_proposals_by_ids,
    "facet_proposals":       facet_proposals,
    "proposal_fields":       proposal_fields,
}

TOOL_DOCS = """
You have access to these tools. To call a tool, output EXACTLY:

Action: <tool_name>
Action Input: <JSON object with arguments>

Then stop. The result will be given back as:
Observation: <result>

After receiving an Observation you may call another tool or give a final answer.
When you have enough information write:
Final Answer: <your answer>

Available tools:

search_proposals(query, fields="id,title,summary", rows=5)
  Search NSF proposals using Lucene syntax.
  query examples: 'pi_name:"Jane Smith"', 'summary:(quantum computing)',
                  'directorate:BIO AND received_year:[2020 TO 2024]'
  fields: id, title, summary, description, pi_name, inst, award_amount,
          directorate, received_year, status

get_proposal(proposal_id, fields="id,title,summary,description,pi_name,inst,award_amount,directorate,received_year,status")
  Retrieve a single proposal by ID (e.g. "2535312") with full text.

fetch_proposals_by_ids(id_list, fields="id,title,summary,pi_name,award_amount")
  Fetch multiple proposals by comma-separated IDs.
  id_list example: "2535312,2535313"

facet_proposals(query, facet_field, limit=15)
  Count top values for a field. facet_field: directorate, division, inst,
  inst_state, status, received_year, funding_program

proposal_fields(group="")
  List available field groups. group: fi_pin, fi_awd, fi_pnl etc.
"""

SYSTEM_PROMPT = f"""You are an expert assistant for exploring NSF research proposals in SOLR.
{TOOL_DOCS}
Rules:
- ALWAYS use a tool to answer questions about proposals — never guess.
- Output Action/Action Input on separate lines with no extra text between them.
- Action Input must be valid JSON.
- After the final Observation, write Final Answer.
"""


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------

def _call_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Unknown tool '{name}'. Available: {', '.join(TOOL_MAP)}"
    try:
        return fn(**args)
    except TypeError as e:
        return f"Bad arguments for {name}: {e}"


def _parse_action(text: str):
    """Return (tool_name, args_dict) or None if no action found."""
    action_m = re.search(r"Action:\s*(\w+)", text)
    input_m  = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if not action_m:
        return None
    tool_name = action_m.group(1).strip()
    if input_m:
        try:
            args = json.loads(input_m.group(1))
        except json.JSONDecodeError:
            args = {}
    else:
        args = {}
    return tool_name, args


def run_repl(model: str = "phi4", verbose: bool = False) -> None:
    # Suppress ollama server noise (Metal GPU messages, timing lines)
    if not verbose:
        sys.stderr = open(os.devnull, "w")

    print(f"NSF SOLR Agent  (model={model}  solr={SOLR_URL})")
    print("Type 'quit' to exit, 'tools' to list tools.\n")

    history: list[dict] = []   # ollama message format

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

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-20:]

        for step in range(8):
            resp    = ollama.chat(model=model, messages=messages)
            content = resp["message"]["content"]

            if verbose:
                print(f"\n  [step {step+1}] {content[:300]}")

            # Check for Final Answer
            fa_match = re.search(r"Final Answer:\s*(.*)", content, re.DOTALL)
            action   = _parse_action(content)

            if action is None or fa_match:
                # Extract final answer text
                answer = fa_match.group(1).strip() if fa_match else content.strip()
                print(f"\nAssistant: {answer}\n")
                history.append({"role": "assistant", "content": answer})
                break

            tool_name, args = action
            if verbose:
                print(f"  [tool] {tool_name}({json.dumps(args)[:120]})")

            observation = _call_tool(tool_name, args)
            if verbose:
                print(f"  [obs]  {observation[:200]}")

            # Append assistant turn + observation to running messages
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user",
                             "content": f"Observation: {observation}\n\nContinue."})
        else:
            print("(reached max steps)\n")

        # Keep history bounded
        if len(history) > 40:
            history = history[-40:]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NSF SOLR ReAct agent (Ollama)")
    p.add_argument("--model",   default="phi4")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    run_repl(model=args.model, verbose=args.verbose)
