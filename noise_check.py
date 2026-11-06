"""How much do rubric scores move when nothing changes? (test-retest noise)

Re-scores a sample of proposals with sampling turned on and compares the runs
dimension by dimension. A critic's disagreement on a dimension only means
something if it's larger than this run-to-run noise.

Steps:
    # 1. pick a sample from the main run (spread across overall scores)
    python noise_check.py sample output/ai_datasets_r07 -n 50 --seed 1 > sample_ids.txt

    # 2. re-score the sample twice with sampling on (same rubric options as the main run)
    for k in 1 2; do
      python pdf_batch_extract.py solr_descriptions/descriptions --ids sample_ids.txt \\
          --temperature 0.7 --seed $k --out output/r07_rerun$k \\
          --instructions _best-r07-rubric.md --context "desc-AI datasets.txt" \\
          --schema _best-r07-schema.json
    done

    # 3. compare (first folder is the reference; every pair is compared)
    python noise_check.py compare output/ai_datasets_r07 output/r07_rerun1 output/r07_rerun2

Writes noise_report.csv (per dimension and pair) and noise_by_proposal.csv
(every score in every run) next to the first rerun folder, unless --out is given.
score_critic.py combine can read noise_report.csv.

The main run uses temperature 0, so its reruns at 0 would be near-identical;
two reruns at 0.7 measure the spread you'd see from sampling.
"""

import argparse
import csv
import itertools
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


def dnum(k: str) -> str:
    m = re.match(r"(?i)(D\d+)", k.strip())
    return m.group(1).upper() if m else k


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load_run(folder: Path) -> dict:
    """{id: {"D1": 2, ..., "overall": 3, "type": "Planning"}} from <folder>/*.json."""
    out = {}
    for f in sorted(folder.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(r, dict) or "dimension_scores" not in r:
            continue
        row = {dnum(k): _int((v or {}).get("score")) for k, v in (r.get("dimension_scores") or {}).items()
               if isinstance(v, dict)}
        row["overall"] = _int((r.get("overall_score") or {}).get("score"))
        row["type"] = r.get("proposal_type")
        out[Path(str(r.get("file", f.stem))).stem] = row
    return out


def weighted_kappa(a: list, b: list, lo: int, hi: int) -> float | None:
    """Quadratic-weighted Cohen's kappa for two raters on an lo..hi scale."""
    k = hi - lo + 1
    n = len(a)
    if n == 0:
        return None
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x - lo][y - lo] += 1
    ra = [sum(row) for row in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = lambda i, j: (i - j) ** 2 / (k - 1) ** 2
    num = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return None if den == 0 else 1 - num / den


def cmd_sample(args):
    run = load_run(Path(args.results))
    if not run:
        sys.exit(f"no scored .json files in {args.results}")
    groups = defaultdict(list)
    for pid, r in run.items():
        groups[r["overall"]].append(pid)
    rng = random.Random(args.seed)
    for g in groups.values():
        g.sort()
        rng.shuffle(g)
    # round-robin across overall-score levels, so the extremes are represented
    # instead of the sample being mostly 2s and 3s
    pick, levels = [], sorted(groups, key=lambda v: (v is None, v))
    while len(pick) < min(args.n, len(run)):
        for lv in levels:
            if groups[lv] and len(pick) < args.n:
                pick.append(groups[lv].pop())
    counts = defaultdict(int)
    for pid in pick:
        counts[run[pid]["overall"]] += 1
    print(f"{len(pick)} of {len(run)} proposals; by overall score: "
          + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0]))),
          file=sys.stderr)
    for pid in sorted(pick):
        print(pid)


def cmd_compare(args):
    folders = [Path(f) for f in args.runs]
    runs = {str(f): load_run(f) for f in folders}
    for f, r in runs.items():
        if not r:
            sys.exit(f"no scored .json files in {f}")
    ref = runs[str(folders[0])]
    others = [runs[str(f)] for f in folders[1:]]
    ids = sorted(set(ref) & set.intersection(*(set(o) for o in others))) if others else []
    if not ids:
        sys.exit("no proposals scored in all the runs given")
    dims = sorted({k for pid in ids for k in ref[pid] if re.fullmatch(r"D\d+", k)},
                  key=lambda d: int(d[1:])) + ["overall"]
    out_dir = Path(args.out) if args.out else folders[1]
    rows = []
    print(f"{len(ids)} proposals scored in all {len(folders)} runs\n")
    print(f"{'pair':<44} {'score':<8} {'n':>3} {'exact':>6} {'±1':>6} {'mean|Δ|':>8} {'kappa':>6}")
    for (fa, a), (fb, b) in itertools.combinations(list(runs.items()), 2):
        label = f"{Path(fa).name} vs {Path(fb).name}"
        for d in dims:
            pairs = [(a[i].get(d), b[i].get(d)) for i in ids
                     if a[i].get(d) is not None and b[i].get(d) is not None]
            if not pairs:
                continue
            xa, xb = zip(*pairs)
            diffs = [abs(x - y) for x, y in pairs]
            kap = weighted_kappa(list(xa), list(xb), 0, 5 if d == "overall" else 3)
            row = {"run_a": fa, "run_b": fb, "score": d, "n": len(pairs),
                   "exact_pct": round(100 * sum(x == 0 for x in diffs) / len(pairs)),
                   "within1_pct": round(100 * sum(x <= 1 for x in diffs) / len(pairs)),
                   "mean_abs_diff": round(sum(diffs) / len(pairs), 2),
                   "weighted_kappa": None if kap is None else round(kap, 2)}
            rows.append(row)
            print(f"{label[:44]:<44} {d:<8} {row['n']:>3} {row['exact_pct']:>5}% {row['within1_pct']:>5}% "
                  f"{row['mean_abs_diff']:>8} {row['weighted_kappa'] if kap is not None else '–':>6}")
        types = [(a[i]["type"], b[i]["type"]) for i in ids]
        same = sum(x == y for x, y in types)
        print(f"{label[:44]:<44} {'type':<8} {len(types):>3} {round(100 * same / len(types)):>5}%")
        print()
    with open(out_dir / "noise_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "noise_by_proposal.csv", "w", newline="") as f:
        w = csv.writer(f)
        names = [Path(x).name for x in runs]
        w.writerow(["id", "score"] + names + ["spread"])
        for pid in ids:
            for d in dims + ["type"]:
                vals = [runs[x][pid].get(d) for x in runs]
                nums = [v for v in vals if isinstance(v, int)]
                spread = (max(nums) - min(nums)) if d != "type" and nums else ("" if d != "type"
                                                                              else int(len(set(vals)) > 1))
                w.writerow([pid, d] + vals + [spread])
    unstable = defaultdict(int)
    for pid in ids:
        for d in dims:
            vals = [runs[x][pid].get(d) for x in runs if runs[x][pid].get(d) is not None]
            if vals and max(vals) - min(vals) >= 2:
                unstable[d] += 1
    if unstable:
        print("scores that moved by 2+ points between runs: "
              + ", ".join(f"{d}: {n}" for d, n in unstable.items()))
    print(f"wrote {out_dir / 'noise_report.csv'} and {out_dir / 'noise_by_proposal.csv'}")
    print("reading it: kappa ≥ 0.8 is very stable, 0.6-0.8 fair, < 0.6 noisy. A critic's "
          "disputes on a noisy dimension are weak evidence.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="pick proposal IDs from a scored run")
    s.add_argument("results", help="folder of scored .json files (pdf_batch_extract.py --out)")
    s.add_argument("-n", type=int, default=50)
    s.add_argument("--seed", type=int, default=1)
    c = sub.add_parser("compare", help="compare two or more scored runs")
    c.add_argument("runs", nargs="+", help="scored folders; the first is the reference")
    c.add_argument("--out", help="folder for the CSVs (default: the second run's folder)")
    args = ap.parse_args()
    if args.cmd == "compare" and len(args.runs) < 2:
        sys.exit("compare needs at least two folders")
    {"sample": cmd_sample, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
