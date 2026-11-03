"""Batch PDF -> structured JSON via a local Ollama model.

Each PDF is processed independently (one model call per PDF, never all PDFs in
one prompt). Long PDFs are summarized chunk-by-chunk first, then the notes are
converted to JSON following the instructions in an .md file.

Usage:
    python pdf_batch_extract.py --instructions gemma_instructions.md \
        --model gemma4:26b --out ./output/pdf_json  a.pdf b.pdf dir_of_pdfs/

    # optional: enforce a JSON schema (Ollama structured outputs)
    python pdf_batch_extract.py --instructions inst.md --schema schema.json ...

    # AI Datasets screening: SOLR descriptions + rubric + solicitation + schema
    python pdf_batch_extract.py solr_descriptions/descriptions \
        --instructions _best-r05-rubric.md --context "desc-AI datasets.txt" \
        --schema _best-r05-schema.json

Re-running resumes: files that already have a .json are skipped (--force redoes them).

Outputs: <out>/<pdf_stem>.json per PDF, <out>/all.jsonl combined, and
<out>/<pdf_stem>.raw.txt for any PDF whose output could not be parsed.

Requires: pip install pymupdf   (or pypdf as a fallback)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
THINK = False  # set by --think


# --------------------------------------------------------------------------- #
# PDF text extraction
# --------------------------------------------------------------------------- #
def extract_text(pdf_path: Path) -> str:
    if pdf_path.suffix.lower() in (".txt", ".md"):  # e.g. solr_fetch_descriptions.py output
        return pdf_path.read_text(errors="replace")
    try:
        import fitz  # pymupdf

        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except ImportError:
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(pdf_path).pages)


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries into chunks of at most ~max_chars."""
    paras = re.split(r"\n\s*\n", text)
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 > max_chars and cur:
            chunks.append(cur)
            cur = ""
        while len(p) > max_chars:  # very long paragraph
            chunks.append(p[:max_chars])
            p = p[max_chars:]
        cur += p + "\n\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
def chat(model, messages, *, num_ctx, num_predict, fmt=None, timeout=900):
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0.0},
        # Gemma 4 is a thinking model; with thinking on, Ollama can route the
        # whole reply into message.thinking and leave message.content empty.
        "think": THINK,
    }
    if fmt is not None:
        body["format"] = fmt  # "json" or a JSON-schema dict
    r = requests.post(OLLAMA_URL, json=body, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    msg = data.get("message", {})
    content = msg.get("content", "")
    if data.get("done_reason") == "length":
        print("    ! output hit the token limit (truncated) — raise --num-predict/--num-ctx",
              file=sys.stderr)
    if not content.strip():
        thinking = msg.get("thinking", "")
        print(f"    ! empty content from model (done_reason={data.get('done_reason')!r}, "
              f"thinking={len(thinking)} chars, eval_count={data.get('eval_count')})",
              file=sys.stderr)
        if thinking.strip():
            content = thinking  # fall back; parse_json will pull out the {...}
    return content


def parse_json(s: str):
    """json.loads with light repair: strip fences, take outermost {...}."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    s2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip())
    start, end = s2.find("{"), s2.rfind("}")
    if start != -1 and end > start:
        return json.loads(s2[start : end + 1])  # may still raise
    raise json.JSONDecodeError("no JSON object found", s, 0)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def summarize_chunks(model, chunks, args):
    notes = []
    for i, ch in enumerate(chunks, 1):
        print(f"    summarizing chunk {i}/{len(chunks)}")
        notes.append(chat(
            model,
            [
                {"role": "system", "content": (
                    "You extract dense, factual notes from a section of a document. "
                    "Keep all specifics (names, numbers, dates, methods, aims, results). "
                    "Plain text bullet points, no JSON.")},
                {"role": "user", "content": f"Section {i} of {len(chunks)}:\n\n{ch}"},
            ],
            num_ctx=args.num_ctx, num_predict=2048,
        ))
    return "\n\n".join(f"[Section {i}]\n{n}" for i, n in enumerate(notes, 1))


def process_pdf(pdf: Path, instructions: str, schema, args):
    text = extract_text(pdf)
    if not text.strip():
        raise ValueError("no extractable text (scanned PDF? needs OCR)")

    # Direct pass if it fits, otherwise map-reduce into notes first.
    if len(text) <= args.chunk_chars:
        material, label = text, "DOCUMENT TEXT"
    else:
        chunks = chunk_text(text, args.chunk_chars)
        material, label = summarize_chunks(args.model, chunks, args), "DOCUMENT NOTES"

    messages = [
        {"role": "system", "content": (
            instructions
            + "\n\nRespond ONLY with a single raw JSON object starting with '{'. "
              "No markdown fences, no commentary.")},
        {"role": "user", "content": f"{label} ({pdf.name}):\n\n{material}"},
    ]
    fmt = schema if schema is not None else "json"

    raw = ""
    for attempt in range(1, args.retries + 2):
        raw = chat(args.model, messages, num_ctx=args.num_ctx,
                   num_predict=args.num_predict, fmt=fmt)
        try:
            return parse_json(raw), raw
        except json.JSONDecodeError as e:
            print(f"    ! invalid JSON (attempt {attempt}): {e}", file=sys.stderr)
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"That was not valid JSON ({e}). Return the complete, valid JSON "
                    "object only. Keep string values concise.")},
            ]
    return None, raw


def collect_pdfs(paths):
    out = []
    for p in map(Path, paths):
        out += (sorted(f for f in p.rglob("*") if f.suffix.lower() in (".pdf", ".txt"))
                if p.is_dir() else [p])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="*", default=["pdfs"],
                    help="PDF/.txt files and/or directories (default: ./pdfs, searched recursively)")
    ap.add_argument("--instructions", default="instructions.md", help=".md file with the extraction instructions")
    ap.add_argument("--schema", help="optional JSON schema file to enforce output structure")
    ap.add_argument("--context", nargs="+", default=[],
                    help="extra reference files appended to the instructions (e.g. the solicitation)")
    ap.add_argument("--force", action="store_true", help="redo files that already have a .json")
    ap.add_argument("--model", default="gemma4:26b", help="Ollama model tag (see `ollama list`)")
    ap.add_argument("--out", default="./output/pdf_json")
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--num-predict", type=int, default=4096, help="max output tokens for the JSON")
    ap.add_argument("--chunk-chars", type=int, default=60000,
                    help="PDFs longer than this are summarized in chunks first (~4 chars/token)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--think", action="store_true", help="enable model thinking (off by default)")
    args = ap.parse_args()
    global THINK
    THINK = args.think

    instructions = Path(args.instructions).read_text()
    for c in args.context:
        instructions += f"\n\n---\n\n## Reference: {Path(c).name}\n\n{Path(c).read_text()}"
    schema = json.loads(Path(args.schema).read_text()) if args.schema else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pdfs = collect_pdfs(args.pdfs)
    if not pdfs:
        sys.exit(f"no .pdf or .txt files found in {args.pdfs}")
    ok = failed = skipped = 0
    t_start = time.time()
    for i, pdf in enumerate(pdfs, 1):
        dest = out / f"{pdf.stem}.json"
        if dest.exists() and not args.force:  # resume: already done
            skipped += 1
            continue
        done_new = ok + failed
        eta = ""
        if done_new:
            per = (time.time() - t_start) / done_new
            eta = f"  (~{per:.0f}s each, ~{per * (len(pdfs) - i + 1) / 3600:.1f}h left)"
        print(f"[{i}/{len(pdfs)}] {pdf.name}{eta}")
        try:
            result, raw = process_pdf(pdf, instructions, schema, args)
        except KeyboardInterrupt:
            print("\ninterrupted; re-run to resume")
            break
        except Exception as e:  # keep going on bad files
            print(f"    ! failed: {e}", file=sys.stderr)
            failed += 1
            continue
        if result is None:
            (out / f"{pdf.stem}.raw.txt").write_text(raw)
            print(f"    ! gave up; raw output saved to {pdf.stem}.raw.txt", file=sys.stderr)
            failed += 1
            continue
        dest.write_text(json.dumps({"file": pdf.name, **result}, indent=2))
        (out / f"{pdf.stem}.raw.txt").unlink(missing_ok=True)
        ok += 1

    # rebuild all.jsonl from every per-file result (includes earlier runs)
    n = 0
    with open(out / "all.jsonl", "w") as jl:
        for f in sorted(out.glob("*.json")):
            rec = json.loads(f.read_text())
            rec.setdefault("file", f.stem)
            jl.write(json.dumps(rec) + "\n")
            n += 1
    print(f"done: {ok} new, {skipped} already done, {failed} failed; "
          f"{n} results in {out / 'all.jsonl'}")


if __name__ == "__main__":
    main()
