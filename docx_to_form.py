"""
docx_to_form.py — Convert a DOCX instruction document into a self-contained HTML form.

Uses docling to parse the DOCX → markdown, then a local Ollama model to generate
the interactive HTML form.

Usage:
    python docx_to_form.py path/to/instructions.docx
    python docx_to_form.py path/to/instructions.docx --out my_form.html
    python docx_to_form.py path/to/instructions.docx --model llama3.1

Requirements:
    pip install ollama docling
    ollama pull llama3.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ollama

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
- Where the document says "select appropriate option" or lists alternatives with OR/AND,
  use radio buttons or a dropdown so the user picks one
- For boilerplate paragraphs with variable parts (shown in brackets or described as
  variable), render the full paragraph text in a textarea and make the variable
  parts editable inline inputs or clearly marked [PLACEHOLDER] text
- Live preview: update a readonly output textarea in real time as fields are filled
- One copy-to-clipboard button per output section
- Clean, professional styling suitable for a US federal agency
- The form must work completely offline"""

FORM_PROMPT = """Convert the following document into a self-contained interactive
HTML form. The document contains instructions and boilerplate language for writing
administrative notes. Build a form that:

1. Has input fields for every variable piece of information
2. Offers radio buttons or dropdowns wherever the document gives alternative options
3. Assembles the final text in a live preview pane that updates as the user types
4. Has a copy button for each output section

Output ONLY the complete HTML file. Do not explain anything.

DOCUMENT:

{markdown}"""


# ---------------------------------------------------------------------------
# DOCX → markdown
# ---------------------------------------------------------------------------

def convert_docx(path: Path) -> str:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        print("Error: docling not installed. Run: pip install docling", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {path.name} with docling …", file=sys.stderr)
    result   = DocumentConverter().convert(str(path))
    markdown = result.document.export_to_markdown()
    print(f"  {len(markdown):,} chars of markdown", file=sys.stderr)
    return markdown


# ---------------------------------------------------------------------------
# Generate form via Ollama (streaming)
# ---------------------------------------------------------------------------

def generate_form(markdown: str, model: str) -> str:
    print(f"Generating form with {model} (streaming) …", file=sys.stderr)
    print("This may take several minutes for a large document.\n", file=sys.stderr)

    parts: list[str] = []
    in_html = False  # skip any preamble text before the HTML starts

    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": FORM_PROMPT.format(markdown=markdown)},
        ],
        stream=True,
        options={
            "num_predict": 16384,   # generous output budget for a full HTML page
            "temperature": 0.2,     # low temperature — we want deterministic HTML
        },
    )

    for chunk in stream:
        text = chunk["message"]["content"]
        parts.append(text)

        # Show a progress dot every ~200 chars so the user knows it's working
        if sum(len(p) for p in parts) % 200 < len(text):
            print(".", end="", flush=True)

    print(file=sys.stderr)

    html = "".join(parts).strip()

    # Strip markdown code fences if the model added them despite instructions
    if "```" in html:
        lines   = html.splitlines()
        kept    = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence:
                kept.append(line)
        html = "\n".join(kept).strip()

    # If the model prefixed with prose before the DOCTYPE, trim it
    if "<!DOCTYPE" in html and not html.startswith("<!"):
        html = html[html.index("<!DOCTYPE"):]

    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert a DOCX instruction document to an HTML form using Ollama"
    )
    p.add_argument("docx",  help="Path to the DOCX file")
    p.add_argument("--out", default="",
                   help="Output HTML path (default: <docx_stem>_form.html)")
    p.add_argument("--model", default="llama3.1",
                   help="Ollama model to use (default: llama3.1)")
    args = p.parse_args()

    docx_path = Path(args.docx)
    if not docx_path.exists():
        print(f"Error: file not found: {docx_path}", file=sys.stderr)
        sys.exit(1)

    out_path = (Path(args.out) if args.out
                else docx_path.with_name(docx_path.stem + "_form.html"))

    markdown = convert_docx(docx_path)
    html     = generate_form(markdown, model=args.model)

    if not html.strip().startswith("<!"):
        print("\nWarning: output does not look like HTML — saving anyway.", file=sys.stderr)
        print("Try running again or use --model llama3.1:70b for better results.", file=sys.stderr)

    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
