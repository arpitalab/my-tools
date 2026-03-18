"""
nsf_agent.py — Local LLM agent that queries SOLR via the MCP server.

The agent connects to a local Ollama instance, discovers the SOLR tools
automatically from solr_mcp.py, then runs a tool-calling loop until it
has enough information to produce a final report.

Requirements:
    pip install openai mcp
    brew install ollama && ollama serve
    ollama pull phi4          # recommended: strong reasoning, 9 GB
    # or: ollama pull qwen2.5:14b / llama3.1:8b

Usage:
    python nsf_agent.py "Generate a research profile for PI Jane Smith"
    python nsf_agent.py "Funding trends in quantum computing 2015–2024"
    python nsf_agent.py "Top institutions receiving BIO directorate grants"
    python nsf_agent.py          # interactive mode (prompts for query)

Environment variables (override defaults):
    OLLAMA_BASE   Ollama API base URL  (default: http://localhost:11434/v1)
    OLLAMA_MODEL  Model to use         (default: phi4)
    SOLR_URL      SOLR base URL        (default: http://localhost:8983/solr)
    SOLR_CORE     Default SOLR core    (default: nsf_proposals)
    AGENT_MAX_STEPS  Max tool-call rounds before forcing a final answer (default: 15)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE  = os.environ.get("OLLAMA_BASE",  "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi4")
SOLR_URL     = os.environ.get("SOLR_URL",     "http://localhost:8983/solr")
SOLR_CORE    = os.environ.get("SOLR_CORE",    "nsf_proposals")
MAX_STEPS    = int(os.environ.get("AGENT_MAX_STEPS", "15"))

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert scientific portfolio analyst with access to a SOLR database
containing NSF proposals, Web of Science publications, and Scopus records.

Your job: answer the user's question by querying SOLR iteratively until you
have enough evidence to write a thorough, grounded report.

Workflow:
1. Call list_cores() and schema() to understand the data available.
2. Plan your query strategy before executing it.
3. Use multiple focused queries rather than one broad one.
4. Cross-reference NSF data with WoS/Scopus where relevant.
5. When you have sufficient evidence, write the final report — do NOT
   ask follow-up questions or say "I need more data". Produce the report.

Report format:
- Use clear section headers (## Section Name)
- Include specific numbers, years, and amounts from the data
- Note data gaps honestly (e.g. "no WoS records found")
- Keep it concise but complete — a program officer should be able to act on it
"""

# ---------------------------------------------------------------------------
# MCP → OpenAI tool schema conversion
# ---------------------------------------------------------------------------

def _mcp_to_openai_tools(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name":        t.name,
                "description": t.description or "",
                "parameters":  t.inputSchema or {"type": "object", "properties": {}},
            },
        }
        for t in mcp_tools
    ]


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def run_agent(query: str, verbose: bool = True) -> str:
    """
    Start the SOLR MCP server, connect Ollama, run the tool-calling loop,
    and return the final report text.
    """
    llm = OpenAI(base_url=OLLAMA_BASE, api_key="ollama")

    server_params = StdioServerParameters(
        command="conda",
        args=[
            "run", "--no-capture-output", "-n", "thellmbook",
            "python", str(_HERE / "solr_mcp.py"),
        ],
        env={
            "SOLR_URL":     SOLR_URL,
            "SOLR_CORE":    SOLR_CORE,
            "SOLR_TIMEOUT": "30",
        },
    )

    if verbose:
        print(f"Model : {OLLAMA_MODEL}  ({OLLAMA_BASE})")
        print(f"SOLR  : {SOLR_URL}  core={SOLR_CORE}")
        print(f"Query : {query}\n")
        print("─" * 60)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools_resp = await session.list_tools()
            openai_tools   = _mcp_to_openai_tools(mcp_tools_resp.tools)

            if verbose:
                names = [t["function"]["name"] for t in openai_tools]
                print(f"Tools available: {', '.join(names)}\n")

            messages: list[dict] = [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": query},
            ]

            for step in range(1, MAX_STEPS + 1):
                if verbose:
                    print(f"[step {step}] Calling {OLLAMA_MODEL} …", end=" ", flush=True)

                response = llm.chat.completions.create(
                    model=OLLAMA_MODEL,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    temperature=0.1,
                )

                msg = response.choices[0].message

                # Append assistant message (preserve tool_calls if present)
                assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    assistant_entry["tool_calls"] = [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(assistant_entry)

                # ── No tool calls → final answer ──────────────────────────
                if not msg.tool_calls:
                    if verbose:
                        print("done (final answer)\n")
                    return msg.content or "(no response)"

                # ── Execute each tool call ─────────────────────────────────
                if verbose:
                    names_called = [tc.function.name for tc in msg.tool_calls]
                    print(f"calling {', '.join(names_called)}")

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    if verbose:
                        args_preview = json.dumps(fn_args)[:120]
                        print(f"  → {fn_name}({args_preview})")

                    try:
                        result     = await session.call_tool(fn_name, fn_args)
                        content    = "\n".join(
                            c.text for c in result.content if hasattr(c, "text")
                        ) or "(empty result)"
                    except Exception as e:
                        content = f"Tool error: {e}"

                    if verbose:
                        preview = content[:200].replace("\n", " ")
                        print(f"     ↳ {preview} …")

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      content,
                    })

            # Reached MAX_STEPS — ask for a summary of what was found
            if verbose:
                print(f"\n[max steps reached — requesting summary]")
            messages.append({
                "role":    "user",
                "content": (
                    "You have reached the maximum number of tool calls. "
                    "Write the best report you can from the information gathered so far."
                ),
            })
            final = llm.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                temperature=0.1,
            )
            return final.choices[0].message.content or "(no response)"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("NSF SOLR Agent  (Ctrl-C to quit)")
        print("Model:", OLLAMA_MODEL, "  SOLR:", SOLR_URL)
        print()
        query = input("Query: ").strip()
        if not query:
            print("No query provided.")
            return

    report = asyncio.run(run_agent(query, verbose=True))

    print("\n" + "═" * 60)
    print("REPORT")
    print("═" * 60)
    print(report)
    print("═" * 60)


if __name__ == "__main__":
    main()
