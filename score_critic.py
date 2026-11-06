"""Critic pass: a model tries to refute the stored rubric scores.

For a sample of scored proposals, a model (the same one or a different one)
gets the rubric's scoring section, the proposal text and the stored scores with
their justifications. For every dimension it must look for evidence that the
score is too high AND evidence that it is too low, quote the proposal verbatim,
and give a verdict: supported / too high / too low.

Every quote is then checked against the proposal text in code (exact, close, or
not found), so invented evidence is caught whatever the model. A dispute counts
as "credible" only if it rests on a verified quote, or, for "too high", on a
named missing element (absence can't be quoted).

Usage:
    # critic run (any Ollama model; one model at a time on a 36 GB machine)
    python score_critic.py run output/ai_datasets_r07 solr_descriptions/descriptions \\
        --rubric _best-r07-rubric.md --ids sample_ids.txt --model gemma4:26b \\
        --out output/critic_gemma
    python score_critic.py run ... --model mistral-small3.1 --out output/critic_mistral

    # combine two critics (and the noise check) into one dispute list
    python score_critic.py combine output/critic_gemma output/critic_mistral \\
        --noise output/r07_rerun1/noise_report.csv

Outputs (run): <out>/<id>.json per proposal (critic reply + quote checks),
<out>/critic_summary.csv (one row per proposal and dimension), and a summary
printed at the end. Re-running resumes; --force redoes proposals.
Outputs (combine): disputes.csv next to the first critic folder.
"""

import argparse
import csv
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pdf_batch_extract as pbe  # Ollama client, JSON parsing, schema checks

VERDICTS = ["supported", "too high", "too low"]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def rubric_scoring_section(path: Path) -> str:
    """Part A (dimensions, definitions) and Part B (overall rules) of the rubric;
    the screening flags and the scorer's output format are left out."""
    text = path.read_text()
    a = re.search(r"^##\s*Part A", text, re.M)
    end = re.search(r"^##\s*Part [CD]", text, re.M)
    if not a:
        sys.exit(f"can't find '## Part A' in {path}")
    return text[a.start(): end.start() if end and end.start() > a.start() else len(text)].strip()


def load_scored(results: Path) -> dict:
    out = {}
    for f in sorted(results.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(r, dict) and r.get("dimension_scores"):
            out[Path(str(r.get("file", f.stem))).stem] = r
    return out


def critic_schema(dim_keys: list) -> dict:
    dim = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": VERDICTS},
            "suggested_score": {"type": "integer", "enum": [0, 1, 2, 3]},
            "quotes_for_lower": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "quotes_for_higher": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "missing": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "suggested_score", "quotes_for_lower", "quotes_for_higher",
                     "missing", "reason"],
    }
    overall = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": VERDICTS},
            "suggested_score": {"type": "integer", "enum": [0, 1, 2, 3, 4, 5]},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "suggested_score", "reason"],
    }
    return {
        "type": "object",
        "properties": {
            "dimensions": {"type": "object", "properties": {k: dim for k in dim_keys},
                           "required": list(dim_keys)},
            "overall": overall,
        },
        "required": ["dimensions", "overall"],
    }


SYSTEM = """You are a senior NSF program director auditing another reviewer's rubric scores for one proposal. Your job is to test each stored score against the proposal text and the rubric anchors below, not to rescore from scratch.

For EVERY dimension:
1. Look for evidence that the stored score is TOO HIGH (anchor requirements the proposal does not meet) and evidence that it is TOO LOW (proposal text meeting a higher anchor). Do both, every time.
2. Quote the proposal VERBATIM: copy up to 30 consecutive words exactly as they appear in the PROPOSAL TEXT. Never paraphrase inside a quote and never quote the rubric or the stored justification. Use quotes_for_lower for text supporting a lower score and quotes_for_higher for text supporting a higher score (0-3 quotes each).
3. If the case for a lower score is that a required element is absent, name the element in "missing" (e.g. "no sustainment plan beyond the award"); absence can't be quoted. Otherwise leave "missing" empty.
4. Verdict: "supported" unless the evidence clearly meets a different anchor. Disagreeing by one point needs specific evidence; do not change a score for writing quality or polish. suggested_score equals the stored score when the verdict is "supported".
5. reason: one or two sentences (<=40 words) naming the anchor you compared against.

Then judge the overall score the same way, applying the rubric's Part B rules (including the caps on D1 and D2) to the dimension scores you consider correct.

RUBRIC (scoring section):

{rubric}

Respond ONLY with a single raw JSON object starting with '{{'. No markdown fences, no commentary."""


def reshape(cand: dict, dim_keys: list) -> dict:
    """Accept the common near-misses: dimensions at the top level or under the
    scorer's own name ('dimension_scores'), 'overall_score' for 'overall', and
    'score' for 'suggested_score'."""
    if not isinstance(cand, dict):
        return cand
    if not isinstance(cand.get("dimensions"), dict):
        for alt in ("dimension_scores", "dimension_audit", "audit", "scores"):
            if isinstance(cand.get(alt), dict):
                cand["dimensions"] = cand.pop(alt)
                break
        else:
            top = {k: cand.pop(k) for k in list(cand)
                   if re.match(r"(?i)d\d+", k) and isinstance(cand[k], dict)}
            if top:
                cand["dimensions"] = top
    if not isinstance(cand.get("overall"), dict):
        for alt in ("overall_score", "overall_audit"):
            if isinstance(cand.get(alt), dict):
                cand["overall"] = cand.pop(alt)
                break
    dims = cand.get("dimensions") if isinstance(cand.get("dimensions"), dict) else {}
    for part in list(dims.values()) + [cand.get("overall")]:
        if not isinstance(part, dict):
            continue
        if "suggested_score" not in part and "score" in part:
            part["suggested_score"] = part.pop("score")
        v = part.get("suggested_score")
        if isinstance(v, str) and v.strip().isdigit():
            part["suggested_score"] = int(v)
        if isinstance(part.get("verdict"), str):
            part["verdict"] = part["verdict"].strip().lower().replace("_", " ")
        for k in ("missing", "reason"):
            if k in part and part[k] is None:
                part[k] = ""
        for k in ("quotes_for_lower", "quotes_for_higher"):
            if k in part and part[k] is None:
                part[k] = []
            elif isinstance(part.get(k), str):
                part[k] = [part[k]] if part[k].strip() else []
            if isinstance(part.get(k), list):
                part[k] = [q for q in part[k] if isinstance(q, str) and q.strip()][:3]
        if part is not cand.get("overall"):
            part.setdefault("missing", "")
            part.setdefault("quotes_for_lower", [])
            part.setdefault("quotes_for_higher", [])
    return cand


def skeleton(dim_keys: list) -> str:
    d = ('{"verdict": "supported | too high | too low", "suggested_score": 0-3, '
         '"quotes_for_lower": ["..."], "quotes_for_higher": ["..."], "missing": "", "reason": "..."}')
    inner = ",\n    ".join(f'"{k}": {d}' for k in dim_keys)
    return ('{\n  "dimensions": {\n    ' + inner + '\n  },\n  "overall": {"verdict": "supported | '
            'too high | too low", "suggested_score": 0-5, "reason": "..."}\n}')


def user_message(pid: str, text: str, scored: dict) -> str:
    dims = scored.get("dimension_scores") or {}
    lines = [f"PROPOSAL TEXT ({pid}):", "", text, "", "STORED SCORES TO AUDIT:",
             f"proposal_type: {scored.get('proposal_type')}"]
    for k, v in dims.items():
        v = v if isinstance(v, dict) else {}
        lines.append(f"- {k}: {v.get('score')}  (justification: {v.get('justification', '')})")
    ov = scored.get("overall_score") or {}
    lines.append(f"- overall (0-5): {ov.get('score')}  (justification: {ov.get('justification', '')})")
    lines += ["", "Audit every dimension and the overall score. Reply in exactly this layout, "
              "with all six dimensions inside \"dimensions\":", skeleton(list(dims))]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Quote verification
# --------------------------------------------------------------------------- #
def _words(s: str) -> list:
    return re.findall(r"[a-z0-9]+", s.lower())


class QuoteChecker:
    def __init__(self, text: str, n: int = 4):
        self.n = n
        self.words = _words(text)
        self.joined = " " + " ".join(self.words) + " "
        self.grams = {tuple(self.words[i:i + n]) for i in range(len(self.words) - n + 1)}

    def check(self, quote: str) -> str:
        """'exact' (word-for-word, ignoring case/punctuation), 'close' (most
        4-word runs present, e.g. a PDF line break or a dropped word), or 'no'."""
        q = _words(quote)
        if not q:
            return "no"
        if " " + " ".join(q) + " " in self.joined:
            return "exact"
        if len(q) < self.n + 1:
            return "no"
        grams = [tuple(q[i:i + self.n]) for i in range(len(q) - self.n + 1)]
        return "close" if sum(g in self.grams for g in grams) / len(grams) >= 0.6 else "no"


def assess(dim: dict, stored, checker: QuoteChecker) -> dict:
    """Quote checks and whether the dispute is backed by evidence."""
    lower = [{"quote": q, "found": checker.check(q)} for q in dim.get("quotes_for_lower") or []]
    higher = [{"quote": q, "found": checker.check(q)} for q in dim.get("quotes_for_higher") or []]
    verdict = dim.get("verdict")
    ok = lambda qs: sum(q["found"] != "no" for q in qs)
    if verdict == "too high":
        basis = ("quote" if ok(lower) else "absence" if (dim.get("missing") or "").strip()
                 else "unverified" if lower else "none")
    elif verdict == "too low":
        basis = "quote" if ok(higher) else ("unverified" if higher else "none")
    else:
        basis = ""
    sugg = dim.get("suggested_score")
    if verdict == "supported":
        sugg = stored
    delta = (sugg - stored) if isinstance(sugg, int) and isinstance(stored, int) else None
    if delta is not None and ((verdict == "too high" and delta >= 0) or (verdict == "too low" and delta <= 0)):
        basis = "inconsistent"  # e.g. "too high" on a 0, or suggested score not lower
    return {"quotes_for_lower": lower, "quotes_for_higher": higher, "basis": basis,
            "credible": verdict in ("too high", "too low") and basis in ("quote", "absence"),
            "suggested_score": sugg, "delta": delta}


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def critique(pid, scored, desc_dir: Path, system: str, args):
    f = desc_dir / f"{pid}.txt"
    if not f.is_file():
        f = next((p for p in desc_dir.rglob(f"{pid}.*") if p.suffix.lower() in (".txt", ".pdf")), None)
        if f is None:
            raise FileNotFoundError(f"no description for {pid} in {desc_dir}")
    text = pbe.extract_text(f).strip()
    truncated = len(text) > args.max_chars
    if truncated:
        pbe.warn(f"    ! {pid}: text is {len(text):,} chars; using the first {args.max_chars:,}")
        text = text[:args.max_chars]
    dim_keys = list((scored.get("dimension_scores") or {}).keys())
    schema = critic_schema(dim_keys)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_message(pid, text, scored)}]
    result, raw = None, ""
    for attempt in range(1, args.retries + 2):
        raw = pbe.chat(args.model, messages, num_ctx=args.num_ctx, num_predict=args.num_predict,
                       fmt=schema)
        try:
            cand = pbe.parse_json(raw)
        except json.JSONDecodeError as e:
            pbe.warn(f"    ! invalid JSON (attempt {attempt}): {e}")
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": "That was not valid JSON. Return the complete JSON object only."}]
            continue
        reshape(cand, dim_keys)
        pbe.repair_keys(cand, schema)
        probs = pbe.schema_problems(cand, schema)
        result = cand
        if not probs:
            break
        pbe.warn(f"    ! reply doesn't match the format (attempt {attempt}): {'; '.join(probs[:3])}")
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": "That JSON doesn't match the required format:\n- "
                      + "\n- ".join(probs[:10]) + "\nReturn the complete JSON object again with exactly "
                      "these keys and allowed values."}]
    if result is None:
        raise ValueError("no usable reply: " + raw[:200])
    checker = QuoteChecker(text)
    stored = {k: (v or {}).get("score") for k, v in scored["dimension_scores"].items()}
    dims = result.get("dimensions") or {}
    checked = {k: {**(dims.get(k) or {}), **assess(dims.get(k) or {}, stored.get(k), checker),
                   "stored_score": stored.get(k)} for k in dim_keys}
    ov = result.get("overall") or {}
    ov_stored = (scored.get("overall_score") or {}).get("score")
    ov_sugg = ov_stored if ov.get("verdict") == "supported" else ov.get("suggested_score")
    return {"id": pid, "model": args.model, "truncated": truncated, "dimensions": checked,
            "overall": {**ov, "stored_score": ov_stored, "suggested_score": ov_sugg,
                        "delta": (ov_sugg - ov_stored) if isinstance(ov_sugg, int)
                        and isinstance(ov_stored, int) else None}}


def summary_rows(rec: dict) -> list:
    rows = []
    for k, d in rec["dimensions"].items():
        qs = d["quotes_for_lower"] + d["quotes_for_higher"]
        rows.append({"id": rec["id"], "dimension": pbe_dnum(k), "stored": d.get("stored_score"),
                     "verdict": d.get("verdict"), "suggested": d.get("suggested_score"),
                     "delta": d.get("delta"), "credible": d.get("credible"), "basis": d.get("basis"),
                     "quotes": len(qs), "quotes_found": sum(q["found"] != "no" for q in qs),
                     "missing": d.get("missing", ""), "reason": d.get("reason", "")})
    ov = rec["overall"]
    rows.append({"id": rec["id"], "dimension": "overall", "stored": ov.get("stored_score"),
                 "verdict": ov.get("verdict"), "suggested": ov.get("suggested_score"),
                 "delta": ov.get("delta"), "credible": "", "basis": "", "quotes": "", "quotes_found": "",
                 "missing": "", "reason": ov.get("reason", "")})
    return rows


def pbe_dnum(k: str) -> str:
    m = re.match(r"(?i)(D\d+)", k)
    return m.group(1).upper() if m else k


def print_summary(rows: list, model: str):
    dims = [r for r in rows if r["dimension"] != "overall"]
    if not dims:
        return
    print(f"\ncritic {model}: {len({r['id'] for r in rows})} proposals")
    print(f"{'score':<8} {'n':>4} {'supported':>10} {'too high':>9} {'too low':>8} {'credible':>9}")
    by = defaultdict(list)
    for r in rows:
        by[r["dimension"]].append(r)
    for d in sorted(by, key=lambda d: (d == "overall", int(d[1:]) if d[1:].isdigit() else 99)):
        g = by[d]
        c = Counter(r["verdict"] for r in g)
        cred = sum(r["credible"] is True for r in g)
        print(f"{d:<8} {len(g):>4} {c['supported']:>10} {c['too high']:>9} {c['too low']:>8} "
              f"{cred if d != 'overall' else '':>9}")
    nq = sum(r["quotes"] for r in dims)
    nf = sum(r["quotes_found"] for r in dims)
    if nq:
        print(f"quotes found in the proposal text: {nf} of {nq} ({nf / nq:.0%}); the rest were "
              "paraphrased or invented and don't count as evidence")


def cmd_run(args):
    pbe.THINK = None if args.think == "omit" else (args.think == "on" if args.think in ("on", "off")
                                                   else args.think)
    pbe.WORKERS = max(1, args.workers)
    scored = load_scored(Path(args.results))
    if not scored:
        sys.exit(f"no scored .json files in {args.results}")
    ids = sorted(scored)
    if args.ids:
        want = [ln.split()[0] for ln in Path(args.ids).read_text().splitlines() if ln.strip()]
        missing = [i for i in want if i not in scored]
        if missing:
            print(f"! {len(missing)} listed IDs aren't scored in {args.results} (e.g. {', '.join(missing[:5])})",
                  file=sys.stderr)
        ids = [i for i in want if i in scored]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    def done(pid):  # a saved reply with no usable verdicts (e.g. wrong layout) is redone
        try:
            rec = json.loads((out / f"{pid}.json").read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return any(d.get("verdict") in VERDICTS for d in (rec.get("dimensions") or {}).values())
    todo = [i for i in ids if args.force or not done(i)]
    if args.limit:
        todo = todo[:args.limit]
    ver, _ = pbe.ollama_version()
    if ver is None:
        sys.exit(f"can't reach Ollama at {pbe.OLLAMA_URL}; is it running?")
    print(f"Ollama {ver}, critic model {args.model}: {len(ids)} proposals, "
          f"{len(ids) - len(todo)} already done, {len(todo)} to do")
    system = SYSTEM.format(rubric=rubric_scoring_section(Path(args.rubric)))

    def one(pid):
        threading.current_thread().name = pid
        pbe._stats.__dict__.clear()
        rec = critique(pid, scored[pid], Path(args.descriptions), system, args)
        pbe.write_atomic(out / f"{pid}.json", json.dumps(rec, indent=2))
        return rec, pbe.stats_line()

    t0, n_ok, n_fail = time.time(), 0, 0
    try:
        with ThreadPoolExecutor(max_workers=pbe.WORKERS) as pool:
            futs = {pool.submit(one, pid): pid for pid in todo}
            for fut in as_completed(futs):
                pid = futs[fut]
                try:
                    rec, timing = fut.result()
                    n_ok += 1
                    disputes = [f"{pbe_dnum(k)} {d['verdict']}" for k, d in rec["dimensions"].items()
                                if d.get("credible")]
                    status = "ok" + (f" · disputes: {', '.join(disputes)}" if disputes else " · all supported")
                    status += f" [{timing}]" if timing else ""
                except Exception as e:  # keep going
                    n_fail += 1
                    status = f"FAILED: {str(e)[:150]}"
                per = (time.time() - t0) / (n_ok + n_fail)
                with pbe._print_lock:
                    print(f"[{n_ok + n_fail}/{len(todo)}] {pid} {status}  "
                          f"(~{per:.0f}s each, ~{per * (len(todo) - n_ok - n_fail) / 3600:.1f}h left)")
    except KeyboardInterrupt:
        print("\ninterrupted; re-run to resume")
    rows = []
    for pid in ids:
        if done(pid):
            rows += summary_rows(json.loads((out / f"{pid}.json").read_text()))
    if rows:
        with open(out / "critic_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print_summary(rows, args.model)
        print(f"wrote {out / 'critic_summary.csv'}" + (f"; {n_fail} failed (re-run to retry)" if n_fail else ""))


# --------------------------------------------------------------------------- #
# combine
# --------------------------------------------------------------------------- #
def cmd_combine(args):
    critics = {}
    for folder in args.critics:
        f = Path(folder) / "critic_summary.csv"
        if not f.is_file():
            sys.exit(f"{f} not found; run `score_critic.py run ... --out {folder}` first")
        with open(f) as fh:
            critics[Path(folder).name] = {(r["id"], r["dimension"]): r for r in csv.DictReader(fh)}
    noise = {}
    if args.noise:
        with open(args.noise) as fh:
            for r in csv.DictReader(fh):
                d = "overall" if r["score"] == "overall" else r["score"]
                noise.setdefault(d, []).append(float(r["exact_pct"]))
        noise = {d: sum(v) / len(v) for d, v in noise.items()}
    names = list(critics)
    keys = sorted(set.intersection(*(set(c) for c in critics.values())))
    out_rows = []
    for key in keys:
        rs = [critics[n][key] for n in names]
        if key[1] == "overall":
            continue
        cred = [r for r in rs if r["credible"] == "True"]
        if not cred:
            continue
        dirs = {r["verdict"] for r in cred}
        row = {"id": key[0], "dimension": key[1], "stored": rs[0]["stored"],
               "critics_disputing": len(cred), "of": len(rs),
               "agree_on_direction": len(dirs) == 1 and len(cred) == len(rs),
               "rerun_exact_pct": round(noise[key[1]]) if key[1] in noise else ""}
        for n, r in zip(names, rs):
            row[f"{n}_verdict"] = r["verdict"] if r["credible"] == "True" else "supported/unbacked"
            row[f"{n}_suggested"] = r["suggested"]
            row[f"{n}_reason"] = (r["missing"] + " · " if r["missing"] else "") + r["reason"]
        out_rows.append(row)
    out_rows.sort(key=lambda r: (not r["agree_on_direction"], -r["critics_disputing"], r["id"], r["dimension"]))
    dest = Path(args.critics[0]) / "disputes.csv"
    if out_rows:
        with open(dest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
    both = [r for r in out_rows if r["agree_on_direction"]]
    print(f"{len({k[0] for k in keys})} proposals in all {len(names)} critics; "
          f"{len(out_rows)} backed disputes, {len(both)} where every critic disputes in the same direction")
    by_dim = Counter(r["dimension"] for r in both)
    if by_dim:
        print("same-direction disputes by dimension: " + ", ".join(
            f"{d}: {n}" + (f" (reruns agree {round(noise[d])}%)" if d in noise else "")
            for d, n in sorted(by_dim.items())))
    if out_rows:
        print(f"wrote {dest}; start with the rows where agree_on_direction is True")
    if noise:
        print("a dimension whose reruns agree < ~70% of the time is noisy: disputes there are weak evidence")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="critique stored scores with one model")
    r.add_argument("results", help="folder of scored .json files (pdf_batch_extract.py --out)")
    r.add_argument("descriptions", help="folder of the proposal .txt files that were scored")
    r.add_argument("--rubric", required=True, help="the rubric .md used for scoring (Parts A and B are sent)")
    r.add_argument("--ids", help="file of proposal IDs (e.g. from noise_check.py sample)")
    r.add_argument("--model", default="gemma4:26b")
    r.add_argument("--out", required=True)
    r.add_argument("--think", default="auto",
                   help="thinking: off / on / low / medium / high / omit. auto = off for gemma, "
                        "low for gpt-oss, omit for others")
    r.add_argument("--num-ctx", type=int, default=32768)
    r.add_argument("--num-predict", type=int, default=4096)
    r.add_argument("--max-chars", type=int, default=90000,
                   help="longest proposal text sent (~4 chars per token)")
    r.add_argument("--retries", type=int, default=2)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--force", action="store_true")
    c = sub.add_parser("combine", help="merge critics into one dispute list")
    c.add_argument("critics", nargs="+", help="critic output folders")
    c.add_argument("--noise", help="noise_report.csv from noise_check.py compare")
    args = ap.parse_args()
    if args.cmd == "run" and args.think == "auto":
        m = args.model.lower()
        args.think = "off" if m.startswith("gemma") else "low" if m.startswith("gpt-oss") else "omit"
    {"run": cmd_run, "combine": cmd_combine}[args.cmd](args)


if __name__ == "__main__":
    main()
