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
        --instructions _best-r07-rubric.md --context "desc-AI datasets.txt" \
        --schema _best-r07-schema.json --out output/ai_datasets_r07

    # noise check: re-score a sample with sampling on (see noise_check.py)
    python pdf_batch_extract.py solr_descriptions/descriptions --ids sample_ids.txt \
        --temperature 0.7 --seed 1 --out output/r07_rerun1 ...same rubric options...

Re-running resumes: files that already have a .json are skipped (--force redoes them).
all.jsonl holds only the results for the files given in this run.

Large runs (e.g. 1,400 proposals): every proposal is its own model call. The
script works through them in batches of --batch-size, refreshing all.jsonl and
failed.txt after each batch, so the dashboard can follow progress during the
run. --limit N stops after N new proposals (e.g. one session a night). If
Ollama stops responding the script waits and retries, and stops after
--max-consecutive-failures failures in a row so it doesn't burn through the queue.
--workers N sends N proposals at once; only faster if Ollama is started with
OLLAMA_NUM_PARALLEL>=N and the GPU has memory for N contexts.

Outputs: <out>/<pdf_stem>.json per PDF, <out>/all.jsonl combined,
<out>/<pdf_stem>.raw.txt for any PDF whose output could not be parsed, and
<out>/failed.txt listing failures from the latest run.

Requires: pip install pymupdf   (or pypdf as a fallback)
"""

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
THINK = False  # set by --think; None = don't send the option (models without a thinking mode)
TEMPERATURE = 0.0  # set by --temperature
SEED = None        # set by --seed
WORKERS = 1    # set by --workers; >1 prefixes log lines with the proposal name
_print_lock = threading.Lock()
_stats = threading.local()  # per-proposal token/time totals, reset in run_one


def _add_stats(data: dict):
    for k in ("prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"):
        setattr(_stats, k, getattr(_stats, k, 0) + (data.get(k) or 0))


def stats_line() -> str:
    """'in 18.2K tok 35s · out 1.3K tok 41s (32 tok/s)' from Ollama's timings."""
    g = lambda k: getattr(_stats, k, 0)
    if not g("eval_count"):
        return ""
    in_s, out_s = g("prompt_eval_duration") / 1e9, g("eval_duration") / 1e9
    rate = f" ({g('eval_count') / out_s:.0f} tok/s)" if out_s else ""
    return (f"in {g('prompt_eval_count') / 1000:.1f}K tok {in_s:.0f}s · "
            f"out {g('eval_count') / 1000:.1f}K tok {out_s:.0f}s{rate}")


def warn(msg: str):
    """Warning line; tagged with the proposal being processed when running in parallel."""
    tag = f"[{threading.current_thread().name}] " if WORKERS > 1 else ""
    with _print_lock:
        print(f"{tag}{msg}", file=sys.stderr)


def write_atomic(path: Path, text: str):
    """Write via a temp file so a crash can't leave a truncated .json behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


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
        # JSON replies are streamed so generation can be stopped at the closing brace
        "stream": fmt is not None,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": TEMPERATURE},
    }
    if SEED is not None:
        body["options"]["seed"] = SEED
    # Gemma 4 is a thinking model; with thinking on, Ollama can route the
    # whole reply into message.thinking and leave message.content empty.
    if THINK is not None:
        body["think"] = THINK
    if fmt is not None:
        body["format"] = fmt  # "json" or a JSON-schema dict
    waits = [30, 60, 120, 300]  # Ollama down/restarting or overloaded
    for attempt in range(len(waits) + 1):
        try:
            r = requests.post(OLLAMA_URL, json=body, timeout=timeout, stream=body["stream"])
            if r.status_code not in (500, 502, 503) or attempt == len(waits):
                break
            problem = f"HTTP {r.status_code}: {r.text[:150]}"
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == len(waits):
                raise RuntimeError(f"Ollama not responding after {len(waits)} retries: {e}") from e
            problem = f"{type(e).__name__}"
        warn(f"    ! Ollama {problem}; retrying in {waits[attempt]}s")
        time.sleep(waits[attempt])
    if r.status_code == 400 and "think" in body and "think" in r.text.lower():
        body.pop("think")  # this model has no thinking option; ask again without it
        r = requests.post(OLLAMA_URL, json=body, timeout=timeout, stream=body["stream"])
    if not r.ok:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
    data = _read_stream(r) if body["stream"] else r.json()
    _add_stats(data)
    msg = data.get("message", {})
    content = msg.get("content", "")
    if data.get("done_reason") == "length":
        warn("    ! output hit the token limit (truncated) — raise --num-predict/--num-ctx")
    if not content.strip():
        thinking = msg.get("thinking", "")
        warn(f"    ! empty content from model (done_reason={data.get('done_reason')!r}, "
             f"thinking={len(thinking)} chars, eval_count={data.get('eval_count')})")
        if thinking.strip():
            content = thinking  # fall back; parse_json will pull out the {...}
    return content


def _read_stream(r) -> dict:
    """Collect a streamed /api/chat reply into one dict like the non-streamed one,
    closing the connection (which stops Ollama) once the JSON object is complete."""
    content, thinking, end, last = [], [], _ObjectEnd(), {}
    t0, n = time.time(), 0
    try:
        for line in r.iter_lines():
            if not line:
                continue
            last = json.loads(line)
            m = last.get("message") or {}
            if m.get("thinking"):
                thinking.append(m["thinking"])
            if m.get("content"):
                content.append(m["content"])
                end.feed(m["content"])
                n += 1
            if last.get("done") or end.done:
                break
    finally:
        r.close()
    data = dict(last)
    data["message"] = {"content": "".join(content), "thinking": "".join(thinking)}
    if end.done and not last.get("done"):  # stopped early: Ollama sent no final stats
        data["done_reason"] = "stop"
        data.setdefault("eval_count", n)
        data.setdefault("eval_duration", int((time.time() - t0) * 1e9))
    return data


def parse_json(s: str):
    """The first complete {...} in the reply. Strips fences and ignores anything the
    model wrote after the object (gemma sometimes keeps going past the closing brace)."""
    s2 = re.sub(r"^```(?:json)?\s*", "", s.strip())
    start = s2.find("{")
    if start == -1:
        raise json.JSONDecodeError("no JSON object found", s, 0)
    obj, _ = json.JSONDecoder().raw_decode(s2, start)  # may still raise
    return obj


class _ObjectEnd:
    """Feed streamed text; .done turns True once the first top-level {...} closes."""

    def __init__(self):
        self.depth, self.in_str, self.esc, self.started, self.done = 0, False, False, False, False

    def feed(self, text: str):
        for ch in text:
            if self.done:
                return
            if self.in_str:
                if self.esc:
                    self.esc = False
                elif ch == "\\":
                    self.esc = True
                elif ch == '"':
                    self.in_str = False
            elif ch == '"' and self.started:
                self.in_str = True
            elif ch == "{":
                self.depth += 1
                self.started = True
            elif ch == "}" and self.started:
                self.depth -= 1
                if self.depth == 0:
                    self.done = True


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
        warn(f"    summarizing chunk {i}/{len(chunks)} (long document)")
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
            warn(f"    ! invalid JSON (attempt {attempt}): {e}")
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
        warn(f"    ! output doesn't match the schema (attempt {attempt}): "
             f"{'; '.join(probs[:4])}{' ...' if len(probs) > 4 else ''}")
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                "That JSON doesn't match the required format:\n- " + "\n- ".join(probs[:15])
                + "\nReturn the complete JSON object again using EXACTLY the key names and "
                  "allowed values in the required output format.")},
        ]
    if last_ok is not None:
        warn("    ! saving the last reply anyway even though it doesn't fully match the "
             "schema; check this proposal by hand")
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
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature (default 0 = repeatable; e.g. 0.7 for a noise check)")
    ap.add_argument("--seed", type=int, default=None, help="sampling seed (with --temperature > 0)")
    ap.add_argument("--ids", help="file of proposal IDs (one per line); score only those inputs")
    ap.add_argument("--batch-size", type=int, default=25,
                    help="refresh all.jsonl and failed.txt after every N proposals")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new proposals (0 = all)")
    ap.add_argument("--workers", type=int, default=1,
                    help="proposals sent to Ollama at once (needs OLLAMA_NUM_PARALLEL>=N)")
    ap.add_argument("--max-consecutive-failures", type=int, default=5,
                    help="stop the run after this many failures in a row (Ollama down?)")
    args = ap.parse_args()
    global THINK, WORKERS, TEMPERATURE, SEED
    THINK = args.think
    TEMPERATURE, SEED = args.temperature, args.seed
    WORKERS = max(1, args.workers)

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
    for tmp in out.glob("*.json.tmp"):  # left by a crash mid-write
        tmp.unlink()

    pdfs = collect_pdfs(args.pdfs)
    if not pdfs:
        sys.exit(f"no .pdf or .txt files found in {args.pdfs}")
    if args.ids:
        want_ids = {ln.split()[0] for ln in Path(args.ids).read_text().splitlines() if ln.strip()}
        pdfs = [p for p in pdfs if p.stem in want_ids]
        missing = want_ids - {p.stem for p in pdfs}
        print(f"--ids: {len(pdfs)} of {len(want_ids)} listed IDs found in the inputs"
              + (f" (missing e.g. {', '.join(sorted(missing)[:5])})" if missing else ""))
    wanted = {p.stem for p in pdfs}
    def scored(p):  # a .json that won't parse (e.g. from an old crash) counts as not scored
        try:
            json.loads((out / f"{p.stem}.json").read_text())
            return True
        except (OSError, json.JSONDecodeError):
            return False

    todo = [p for p in pdfs if args.force or not scored(p)]
    skipped = len(pdfs) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    n_batches = (len(todo) + args.batch_size - 1) // max(args.batch_size, 1)
    print(f"{len(pdfs)} inputs: {skipped} already scored, {len(todo)} to score in "
          f"{n_batches} batch(es) of {args.batch_size}"
          + (f", {WORKERS} at a time" if WORKERS > 1 else "")
          + (f" (--limit {args.limit})" if args.limit else ""))

    def run_one(pdf):
        threading.current_thread().name = pdf.stem
        _stats.__dict__.clear()
        result, raw = process_pdf(pdf, instructions, schema, args)
        if result is None:
            (out / f"{pdf.stem}.raw.txt").write_text(raw)
            raise ValueError(f"gave up; raw output saved to {pdf.stem}.raw.txt")
        write_atomic(out / f"{pdf.stem}.json", json.dumps({"file": pdf.name, **result}, indent=2))
        (out / f"{pdf.stem}.raw.txt").unlink(missing_ok=True)
        return stats_line()

    ok, failures, streak = 0, {}, 0
    t_start, stop = time.time(), None
    try:
        for b in range(n_batches):
            batch = todo[b * args.batch_size:(b + 1) * args.batch_size]
            print(f"--- batch {b + 1}/{n_batches} ({len(batch)} proposals)")
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futs = {pool.submit(run_one, pdf): pdf for pdf in batch}
                for fut in as_completed(futs):
                    pdf = futs[fut]
                    try:
                        timing = fut.result()
                        ok += 1
                        streak = 0
                        failures.pop(pdf.stem, None)
                        status = "ok" + (f" [{timing}]" if timing else "")
                    except Exception as e:  # keep going on bad files
                        failures[pdf.stem] = str(e)[:300]
                        streak += 1
                        status = f"FAILED: {str(e)[:150]}"
                    done = ok + len(failures)
                    per = (time.time() - t_start) / done
                    with _print_lock:
                        print(f"[{done}/{len(todo)}] {pdf.name} {status}  "
                              f"(~{per:.0f}s each, ~{per * (len(todo) - done) / 3600:.1f}h left)")
                    if streak >= args.max_consecutive_failures:
                        stop = (f"{streak} failures in a row; stopping (is Ollama running? "
                                f"`ollama ps`). Re-run to resume.")
                        for f2 in futs:
                            f2.cancel()
                        break
            write_all_jsonl(out, wanted, failures)
            if stop:
                print(stop, file=sys.stderr)
                break
    except KeyboardInterrupt:
        print("\ninterrupted; re-run to resume (proposals already scored are kept)")

    n = write_all_jsonl(out, wanted, failures)
    remaining = sum(1 for p in pdfs if not (out / f"{p.stem}.json").exists())
    print(f"done: {ok} new, {skipped} already done, {len(failures)} failed; "
          f"{n} results in {out / 'all.jsonl'}"
          + (f"; {remaining} still to score (re-run to continue)" if remaining else ""))
    if failures:
        print(f"failed proposals listed in {out / 'failed.txt'}; re-run to retry them")
    others = sorted(f.name for f in out.glob("*.json") if f.stem not in wanted)
    if others:
        print(f"note: {len(others)} other .json file(s) in {out} from earlier runs were left out "
              f"of all.jsonl (e.g. {', '.join(others[:3])}). Use a separate --out per project.")


def write_all_jsonl(out: Path, wanted: set, failures: dict) -> int:
    """Rebuild all.jsonl from the results for THIS run's input files only, so
    leftovers from earlier runs in the same --out folder don't mix in."""
    lines = []
    for stem in sorted(wanted):
        f = out / f"{stem}.json"
        if f.exists():
            try:
                rec = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            rec.setdefault("file", stem)
            lines.append(json.dumps(rec))
    write_atomic(out / "all.jsonl", "\n".join(lines) + ("\n" if lines else ""))
    write_atomic(out / "failed.txt", "".join(f"{k}\t{v}\n" for k, v in sorted(failures.items())))
    return len(lines)


if __name__ == "__main__":
    main()
