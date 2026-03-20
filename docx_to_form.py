"""
docx_to_form.py — Convert a DOCX instruction document into a self-contained HTML form.

Extracts text from the DOCX with python-docx (clean, no unicode artifacts), then
generates the form with either the Claude API or a local Ollama model.

Usage:
    python docx_to_form.py path/to/instructions.docx            # auto-detect backend
    python docx_to_form.py path/to/instructions.docx --backend claude
    python docx_to_form.py path/to/instructions.docx --backend ollama
    python docx_to_form.py path/to/instructions.docx --out my_form.html
    python docx_to_form.py path/to/instructions.docx --model llama3.1

Backend selection (--backend auto, the default):
    Uses Claude if ANTHROPIC_API_KEY is set, otherwise falls back to Ollama.

Requirements:
    pip install python-docx anthropic ollama
    # For Ollama backend: ollama pull llama3.1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert web developer. Your only job is to output
complete, self-contained HTML files — no explanations, no markdown fences,
just raw HTML starting with <!DOCTYPE html>.

Rules for every form you generate:
- All CSS and JavaScript must be inline — no external dependencies whatsoever
- Match input types to content: text fields, textareas, radio buttons, checkboxes,
  dropdowns as appropriate
- Where the document says "select appropriate option" or lists alternatives separated
  by OR or AND, use radio buttons or a dropdown to choose between them
- For boilerplate paragraphs with variable parts (shown in brackets, italics, or
  described as variable / highlighted), render the full paragraph with the variable
  parts as editable inline inputs or clearly marked [PLACEHOLDER] spans
- Live preview: as the user fills in fields the assembled output text updates in
  real time in a readonly textarea
- One copy-to-clipboard button per output section
- Clean, professional styling suitable for a US federal agency
- Must work completely offline — no CDN, no external fonts, no external scripts"""

FORM_PROMPT = """Convert the following document into a self-contained interactive
HTML form. The document contains instructions and boilerplate language for writing
administrative notes.

Build a form that:
1. Has an input field for every variable piece of information
2. Uses radio buttons or dropdowns wherever the document gives alternative options
3. Assembles the final text in a live-updating preview pane
4. Has copy-to-clipboard buttons on each output section

Output ONLY the raw HTML. Do not include any explanation or markdown.

DOCUMENT:

{text}"""


# ---------------------------------------------------------------------------
# DOCX → plain text  (python-docx — clean, no unicode artifacts)
# ---------------------------------------------------------------------------

def extract_text(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        print("Error: python-docx not installed. Run: pip install python-docx",
              file=sys.stderr)
        sys.exit(1)

    print(f"Reading {path.name} …", file=sys.stderr)
    doc   = Document(str(path))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text  = "\n\n".join(paras)
    print(f"  {len(paras)} paragraphs, {len(text):,} characters", file=sys.stderr)
    return text


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    """Remove markdown code fences and trim to the DOCTYPE if present."""
    html = raw.strip()
    if "```" in html:
        lines, kept, in_fence = html.splitlines(), [], False
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence:
                kept.append(line)
        html = "\n".join(kept).strip()
    if "<!DOCTYPE" in html and not html.lstrip().startswith("<!"):
        html = html[html.index("<!DOCTYPE"):]
    return html


def generate_claude(text: str, model: str) -> str:
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic not installed. Run: pip install anthropic",
              file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()
    print(f"Generating form with Claude ({model}) …", file=sys.stderr)

    parts: list[str] = []
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user",
                   "content": FORM_PROMPT.format(text=text)}],
    ) as stream:
        for chunk in stream.text_stream:
            parts.append(chunk)
            if sum(len(p) for p in parts) % 300 < len(chunk):
                print(".", end="", flush=True)

    print(file=sys.stderr)
    return _strip_fences("".join(parts))


def generate_ollama(text: str, model: str) -> str:
    try:
        import ollama
    except ImportError:
        print("Error: ollama not installed. Run: pip install ollama", file=sys.stderr)
        sys.exit(1)

    print(f"Generating form with Ollama ({model}) …", file=sys.stderr)
    print("This may take several minutes.\n", file=sys.stderr)

    parts: list[str] = []
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": FORM_PROMPT.format(text=text)},
        ],
        stream=True,
        options={"num_predict": 16384, "temperature": 0.2},
    )
    for chunk in stream:
        piece = chunk["message"]["content"]
        parts.append(piece)
        if sum(len(p) for p in parts) % 300 < len(piece):
            print(".", end="", flush=True)

    print(file=sys.stderr)
    return _strip_fences("".join(parts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert a DOCX instruction document to an HTML form"
    )
    p.add_argument("docx", help="Path to the DOCX file")
    p.add_argument("--out", default="",
                   help="Output HTML path (default: <docx_stem>_form.html)")
    p.add_argument("--backend", choices=["auto", "claude", "ollama"], default="auto",
                   help="LLM backend (default: claude if ANTHROPIC_API_KEY set, else ollama)")
    p.add_argument("--model", default="",
                   help="Model override (default: claude-opus-4-6 or llama3.1)")
    args = p.parse_args()

    docx_path = Path(args.docx)
    if not docx_path.exists():
        print(f"Error: file not found: {docx_path}", file=sys.stderr)
        sys.exit(1)

    out_path = (Path(args.out) if args.out
                else docx_path.with_name(docx_path.stem + "_form.html"))

    # Resolve backend
    backend = args.backend
    if backend == "auto":
        backend = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "ollama"
    print(f"Backend: {backend}", file=sys.stderr)

    text = extract_text(docx_path)

    if backend == "claude":
        model = args.model or "claude-opus-4-6"
        html  = generate_claude(text, model)
    else:
        model = args.model or "llama3.1"
        html  = generate_ollama(text, model)

    if not html.lstrip().startswith("<!"):
        print("\nWarning: output does not look like valid HTML — saving anyway.",
              file=sys.stderr)

    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
