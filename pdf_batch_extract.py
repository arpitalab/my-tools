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
all.jsonl holds only the results for the files given in this run.

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
# Schema checking (in case Ollama doesn't enforce the schema, e.g. < 0.5)
# --------------------------------------------------------------------------- #
def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", k.lower())


def repair_keys(obj, schema):
    """Rename near-miss keys to the schema's names, in place.

    A key not in the schema is renamed to a missing expected key when they share
    a D-number prefix (D1_Program_Fit_and_Proposal_DType_... -> the D1 key), are
    identical after normalizing case and punctuation, or when it is the only
    unexpected key, exactly one required key is missing, and its value has that
    key's required fields.
    """
    if not isinstance(obj, dict) or not isinstance(schema, dict):
        return obj
    props = schema.get("properties") or {}
    if props:
        for k in [k for k in obj if k not in props]:
            missing = [e for e in props if e not in obj]
            dk = re.match(r"(?i)(d\d+)(?:_|$)", k)
            match = [e for e in missing if _norm_key(e) == _norm_key(k)]
            if not match and dk:
                match = [e for e in missing if re.match(rf"(?i){dk.group(1)}(?:_|$)", e)]
            if len(match) == 1:
                obj[match[0]] = obj.pop(k)
        # one missing + one unexpected key with the right shape (e.g. the model
        # wrote "scale": {"score":..,"justification":..} instead of the D4 key)
        missing = [e for e in props if e not in obj and e in schema.get("required", [])]
        extra = [k for k in obj if k not in props]
        if len(missing) == 1 and len(extra) == 1:
            want = props[missing[0]]
            v = obj[extra[0]]
            if want.get("type") == "object" and isinstance(v, dict) and \
                    set(want.get("required", [])) <= set(v):
                obj[missing[0]] = obj.pop(extra[0])
            elif "enum" in want and not isinstance(v, (dict, list)) and v in want["enum"]:
                obj[missing[0]] = obj.pop(extra[0])  # e.g. "future_proposal": "clear"
        for k, sub in props.items():
            if k in obj:
                repair_keys(obj[k], sub)
    if schema.get("type") == "array" and isinstance(obj, list):
        for item in obj:
            repair_keys(item, schema.get("items") or {})
    return obj


def schema_problems(obj, schema, path="") -> list[str]:
    """Missing/unexpected keys, wrong types and out-of-range enum values."""
    probs = []
    t = schema.get("type")
    where = path or "top level"
    if "enum" in schema and obj not in schema["enum"]:
        probs.append(f"{where}: {obj!r} not one of {schema['enum']}")
    if t == "object":
        if not isinstance(obj, dict):
            return [f"{where}: expected an object"]
        props = schema.get("properties") or {}
        for k in schema.get("required", []):
            if k not in obj:
                probs.append(f"{where}: missing key \"{k}\"")
        if props:
            for k in obj:
                if k not in props:
                    probs.append(f"{where}: unexpected key \"{k}\"")
        for k, sub in props.items():
            if k in obj:
                probs += schema_problems(obj[k], sub, f"{path}.{k}" if path else k)
    elif t == "array":
        if not isinstance(obj, list):
            return [f"{where}: expected a list"]
        for i, item in enumerate(obj):
            probs += schema_problems(item, schema.get("items") or {}, f"{where}[{i}]")
    elif t == "integer" and not (isinstance(obj, int) and not isinstance(obj, bool)):
        probs.append(f"{where}: expected an integer")
    elif t == "boolean" and not isinstance(obj, bool):
        probs.append(f"{where}: expected true/false")
    elif t == "string" and not isinstance(obj, str):
        probs.append(f"{where}: expected a string")
    return probs


def ollama_version():
    try:
        v = requests.get(OLLAMA_URL.replace("/api/chat", "/api/version"), timeout=5).json()["version"]
        return v, tuple(int(x) for x in re.findall(r"\d+", v)[:3])
    except Exception:
        return None, None


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

    raw, last_ok = "", None
    for attempt in range(1, args.retries + 2):
        raw = chat(args.model, messages, num_ctx=args.num_ctx,
                   num_predict=args.num_predict, fmt=fmt)
        try:
            result = parse_json(raw)
        except json.JSONDecodeError as e:
            print(f"    ! invalid JSON (attempt {attempt}): {e}", file=sys.stderr)
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"That was not valid JSON ({e}). Return the complete, valid JSON "
                    "object only. Keep string values concise.")},
            ]
            continue
        if schema is None:
            return result, raw
        repair_keys(result, schema)
        probs = schema_problems(result, schema)
        if not probs:
            return result, raw
        last_ok = result
        print(f"    ! output doesn't match the schema (attempt {attempt}): "
              f"{'; '.join(probs[:4])}{' ...' if len(probs) > 4 else ''}", file=sys.stderr)
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                "That JSON doesn't match the required format:\n- " + "\n- ".join(probs[:15])
                + "\nReturn the complete JSON object again using EXACTLY the key names and "
                  "allowed values in the required output format.")},
        ]
    if last_ok is not None:
        print("    ! saving the last reply anyway even though it doesn't fully match the "
              "schema; check this proposal by hand", file=sys.stderr)
        return last_ok, raw
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
    ver, ver_t = ollama_version()
    if ver is None:
        print(f"! can't reach Ollama at {OLLAMA_URL}; is it running?", file=sys.stderr)
    else:
        print(f"Ollama {ver}, model {args.model}" + (f", schema {args.schema}" if schema else ""))
        if schema is not None and ver_t < (0, 5, 0):
            print("! this Ollama is older than 0.5 and won't enforce --schema; the script will "
                  "still check each reply and retry, but upgrading Ollama is more reliable",
                  file=sys.stderr)
    if schema is None:
        print("! no --schema given: the model may invent or misspell field names", file=sys.stderr)
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

    # rebuild all.jsonl from the results for THIS run's input files only, so
    # leftovers from earlier runs in the same --out folder don't mix in
    n = 0
    wanted = {p.stem for p in pdfs}
    with open(out / "all.jsonl", "w") as jl:
        for stem in sorted(wanted):
            f = out / f"{stem}.json"
            if f.exists():
                rec = json.loads(f.read_text())
                rec.setdefault("file", stem)
                jl.write(json.dumps(rec) + "\n")
                n += 1
    others = sorted(f.name for f in out.glob("*.json") if f.stem not in wanted)
    print(f"done: {ok} new, {skipped} already done, {failed} failed; "
          f"{n} results in {out / 'all.jsonl'}")
    if others:
        print(f"note: {len(others)} other .json file(s) in {out} from earlier runs were left out "
              f"of all.jsonl (e.g. {', '.join(others[:3])}). Use a separate --out per project.")


if __name__ == "__main__":
    main()
