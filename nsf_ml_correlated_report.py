"""
NSF Funding Trend Report: Machine Learning / AI × Strongly Correlated Electron Systems
(Extended: includes topological materials + quantum computing overlap)
Queries the local NSF SQLite database and writes a Markdown report + CSV data.
"""

import sqlite3
import re
from collections import defaultdict
from pathlib import Path

DB = Path("/Users/sraghava/Desktop/my_llm_explore/output/nsf_awards.db")
OUT_DIR = Path("/Users/sraghava/Desktop/my_llm_explore/output")
OUT_DIR.mkdir(exist_ok=True)

# ── keyword sets ──────────────────────────────────────────────────────────────
ML_AI_TERMS = [
    # core ML
    "machine learning", "deep learning", "neural network", "neural networks",
    "artificial intelligence", r"\bai\b", "reinforcement learning",
    "graph neural", "transformer model", "large language model",
    "generative model", "data-driven", "data driven",
    "random forest", "support vector machine", "convolutional neural",
    "bayesian optimization", "gaussian process",
    # broader ML / data science
    "generative adversarial", "variational autoencoder", "autoencoder",
    "transfer learning", "active learning", "unsupervised learning",
    "supervised learning", "semi-supervised", "foundation model",
    "diffusion model", "score-based model", "normalizing flow",
    "physics-informed neural", "physics informed neural",
    "neural operator", "deep neural", "attention mechanism",
    "language model", "natural language processing",
]

SCE_TERMS = [
    # strongly correlated electrons core
    "strongly correlated", "correlated electron", "mott insulator",
    "mott transition", "kondo", "heavy fermion", "hubbard model",
    "quantum spin liquid", "frustrated magnet",
    "charge density wave", "spin density wave", "colossal magnetoresistance",
    "cuprate", "high.tc superconductor", "unconventional superconductor",
    "strongly interacting electron", "correlated oxide",
    "quantum phase transition", "quantum criticality",
    "many-body", "many body", "dynamical mean field",
    r"\bdmft\b", "slave boson", "quantum impurity",
    "anderson impurity", "kondo lattice", "spin-orbit coupling",
    "excitonic insulator", "wigner crystal", "fractional quantum hall",
    # topological materials (extended)
    "topological insulator", "topological semimetal",
    "topological superconductor", "topological material",
    "topological phase", "topological order",
    "weyl semimetal", "dirac semimetal", "weyl node",
    "axion insulator", "magnetic topological", "moire",
    "van der waals heterostructure", "twisted bilayer",
    "quantum anomalous hall", "quantum spin hall",
    "majorana fermion", "majorana mode", "berry phase",
    "chern insulator", "topological band",
    # quantum computing / quantum materials overlap
    "quantum computing", "quantum simulation", "quantum simulator",
    "quantum annealing", "variational quantum", r"\bvqe\b",
    "quantum circuit", "qubit", "quantum error correction",
    "quantum advantage", "quantum algorithm",
    "tensor network", "matrix product state", r"\bdmrg\b",
    "quantum monte carlo", "diagrammatic monte carlo",
    "quantum many-body", "quantum matter", "quantum material",
]

def build_like_clauses(terms, field="abstract_narration"):
    clauses = [f"LOWER({field}) LIKE '%{t.replace(chr(92),'').replace('%','%%')}%'" for t in terms]
    return "(" + " OR ".join(clauses) + ")"

def regex_match(text, terms):
    if not text:
        return False
    text_lower = text.lower()
    for t in terms:
        if re.search(t, text_lower):
            return True
    return False

def classify_award(title, abstract):
    combined = (title or "") + " " + (abstract or "")
    has_ml  = regex_match(combined, ML_AI_TERMS)
    has_sce = regex_match(combined, SCE_TERMS)
    return has_ml, has_sce

# sub-category term sets for tagging
SCE_CORE = [
    "strongly correlated", "correlated electron", "mott insulator", "mott transition",
    "kondo", "heavy fermion", "hubbard model", "quantum spin liquid", "frustrated magnet",
    "charge density wave", "spin density wave", "colossal magnetoresistance",
    "cuprate", "high.tc superconductor", "unconventional superconductor",
    "strongly interacting electron", "correlated oxide",
    "dynamical mean field", r"\bdmft\b", "quantum impurity", "anderson impurity",
    "kondo lattice", "excitonic insulator", "wigner crystal", "fractional quantum hall",
]
TOPO = [
    "topological insulator", "topological semimetal", "topological superconductor",
    "topological material", "topological phase", "topological order",
    "weyl semimetal", "dirac semimetal", "weyl node", "axion insulator",
    "magnetic topological", "moire", "van der waals heterostructure", "twisted bilayer",
    "quantum anomalous hall", "quantum spin hall", "majorana fermion", "majorana mode",
    "berry phase", "chern insulator", "topological band",
]
QCOMP = [
    "quantum computing", "quantum simulation", "quantum simulator",
    "quantum annealing", "variational quantum", r"\bvqe\b",
    "quantum circuit", "qubit", "quantum error correction",
    "quantum advantage", "quantum algorithm",
    "tensor network", "matrix product state", r"\bdmrg\b",
    "quantum monte carlo", "diagrammatic monte carlo",
    "quantum many-body", "quantum matter", "quantum material",
    "many-body", "many body",
]

def tag_award(title, abstract):
    combined = (title or "") + " " + (abstract or "")
    tags = []
    if regex_match(combined, SCE_CORE): tags.append("SCE-core")
    if regex_match(combined, TOPO):     tags.append("Topological")
    if regex_match(combined, QCOMP):    tags.append("Quantum-computing")
    return tags

# ── pull candidates from DB ───────────────────────────────────────────────────
conn = sqlite3.connect(DB)

# expanded SQL filter covering all three sub-categories
ml_like_sql = build_like_clauses(
    ["machine learning","deep learning","neural network","artificial intelligence",
     " ai ","data-driven","data driven","reinforcement learning","generative model",
     "autoencoder","transfer learning","active learning","foundation model",
     "diffusion model","neural operator","physics-informed neural","physics informed neural",
     "language model","natural language processing","bayesian optimization","gaussian process"],
    "abstract_narration")

sce_like_sql = build_like_clauses(
    ["strongly correlated","correlated electron","mott insulator","mott transition",
     "kondo","heavy fermion","hubbard model","quantum spin liquid","cuprate",
     "topological insulator","topological semimetal","topological superconductor",
     "topological material","topological phase","topological order",
     "weyl semimetal","dirac semimetal","quantum anomalous hall","quantum spin hall",
     "majorana","moire","twisted bilayer","van der waals heterostructure",
     "charge density wave","quantum phase transition","quantum criticality",
     "many-body","many body","tensor network","matrix product state","dmrg",
     "quantum monte carlo","quantum computing","quantum simulation","qubit",
     "variational quantum","quantum material","quantum matter",
     "dynamical mean field","dmft"],
    "abstract_narration")

ml_title_sql = build_like_clauses(
    ["machine learning","deep learning","neural network","artificial intelligence",
     "data-driven","data driven"],
    "title")
sce_title_sql = build_like_clauses(
    ["strongly correlated","correlated electron","mott insulator","heavy fermion",
     "quantum spin liquid","topological insulator","topological semimetal",
     "topological superconductor","weyl semimetal","dirac semimetal",
     "majorana","quantum computing","qubit","tensor network","dmrg",
     "quantum material","quantum many-body","moire","twisted bilayer"],
    "title")

query = f"""
SELECT a.award_id, a.title, a.source_year, a.effective_date,
       a.total_intended_amount, a.award_amount,
       a.abstract_narration,
       d.abbreviation  AS directorate,
       dv.abbreviation AS division,
       i.name          AS institution,
       i.state_name    AS state
FROM award a
LEFT JOIN directorate  d  ON a.directorate_id  = d.id
LEFT JOIN division     dv ON a.division_id     = dv.id
LEFT JOIN institution  i  ON a.institution_id  = i.id
WHERE (
    (({ml_like_sql})  AND ({sce_like_sql}))
 OR (({ml_title_sql}) AND ({sce_like_sql}))
 OR (({sce_title_sql}) AND ({ml_like_sql}))
)
  AND a.source_year IS NOT NULL
  AND a.source_year >= 2000
"""

rows = conn.execute(query).fetchall()
conn.close()

cols = ["award_id","title","source_year","effective_date",
        "total_intended_amount","award_amount","abstract",
        "directorate","division","institution","state"]

candidates = [dict(zip(cols, r)) for r in rows]
print(f"Candidates after SQL filter: {len(candidates)}")

# Python-level regex refinement + tagging
awards = []
for r in candidates:
    has_ml, has_sce = classify_award(r["title"], r["abstract"])
    if has_ml and has_sce:
        r["tags"] = "|".join(tag_award(r["title"], r["abstract"]))
        awards.append(r)

print(f"Awards confirmed (ML+AI × SCE/Topo/QC): {len(awards)}")

# counts per sub-category
for tag in ["SCE-core","Topological","Quantum-computing"]:
    n = sum(1 for r in awards if tag in r.get("tags",""))
    print(f"  {tag}: {n}")

# ── aggregate statistics ──────────────────────────────────────────────────────
def safe_amount(v):
    try:
        return float(v) if v else 0.0
    except:
        return 0.0

SUB_TAGS = ["SCE-core", "Topological", "Quantum-computing"]

by_year  = defaultdict(lambda: {"count": 0, "amount": 0.0,
                                 "SCE-core": 0, "Topological": 0, "Quantum-computing": 0})
by_dir   = defaultdict(lambda: {"count": 0, "amount": 0.0})
by_div   = defaultdict(lambda: {"count": 0, "amount": 0.0})
by_inst  = defaultdict(lambda: {"count": 0, "amount": 0.0})
by_state = defaultdict(lambda: {"count": 0, "amount": 0.0})
by_tag   = defaultdict(lambda: {"count": 0, "amount": 0.0})

for r in awards:
    yr  = int(r["source_year"])
    amt = safe_amount(r["total_intended_amount"])
    by_year[yr]["count"]  += 1
    by_year[yr]["amount"] += amt
    for t in SUB_TAGS:
        if t in r.get("tags", ""):
            by_year[yr][t] += 1
            by_tag[t]["count"]  += 1
            by_tag[t]["amount"] += amt
    d = r["directorate"] or "Unknown"
    by_dir[d]["count"]  += 1
    by_dir[d]["amount"] += amt
    dv = r["division"] or "Unknown"
    by_div[dv]["count"]  += 1
    by_div[dv]["amount"] += amt
    inst = r["institution"] or "Unknown"
    by_inst[inst]["count"]  += 1
    by_inst[inst]["amount"] += amt
    st = r["state"] or "Unknown"
    by_state[st]["count"]  += 1
    by_state[st]["amount"] += amt

years_sorted = sorted(by_year.keys())
total_count  = sum(v["count"]  for v in by_year.values())
total_amount = sum(v["amount"] for v in by_year.values())

# ── write CSV ─────────────────────────────────────────────────────────────────
import csv

csv_path = OUT_DIR / "ml_sce_awards.csv"
csv_fields = [c for c in cols if c != "abstract"] + ["tags"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    for r in sorted(awards, key=lambda x: x["source_year"]):
        writer.writerow({k: v for k, v in r.items() if k != "abstract"})

trend_csv = OUT_DIR / "ml_sce_yearly_trend.csv"
with open(trend_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["year","count","total_amount_usd","SCE-core","Topological","Quantum-computing"])
    for yr in years_sorted:
        writer.writerow([yr, by_year[yr]["count"], f"{by_year[yr]['amount']:.0f}",
                         by_year[yr]["SCE-core"], by_year[yr]["Topological"],
                         by_year[yr]["Quantum-computing"]])

# ── build Markdown report ─────────────────────────────────────────────────────
def fmt_m(v):
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    return f"${v/1e3:.0f}K"

top_inst = sorted(by_inst.items(), key=lambda x: -x[1]["count"])[:15]
top_div  = sorted(by_div.items(),  key=lambda x: -x[1]["count"])[:10]
top_dir  = sorted(by_dir.items(),  key=lambda x: -x[1]["count"])[:8]
top_state= sorted(by_state.items(),key=lambda x: -x[1]["count"])[:10]

# period groupings
def group(yrng):
    c = sum(by_year[y]["count"]  for y in yrng if y in by_year)
    a = sum(by_year[y]["amount"] for y in yrng if y in by_year)
    return c, a

p1 = group(range(2000, 2010))
p2 = group(range(2010, 2017))
p3 = group(range(2017, 2021))
p4 = group(range(2021, 2026))

# YoY table
yoy_rows = []
for i, yr in enumerate(years_sorted):
    c  = by_year[yr]["count"]
    am = by_year[yr]["amount"]
    if i > 0:
        prev_c  = by_year[years_sorted[i-1]]["count"]
        prev_am = by_year[years_sorted[i-1]]["amount"]
        dc = f"{(c-prev_c)/prev_c*100:+.0f}%" if prev_c else "—"
        da = f"{(am-prev_am)/prev_am*100:+.0f}%" if prev_am else "—"
    else:
        dc = da = "—"
    yoy_rows.append((yr, c, fmt_m(am), dc, da))

md = []
md.append("# NSF Funding Trend Report")
md.append("## Machine Learning / AI × Quantum Materials")
md.append("### (Strongly Correlated Electrons · Topological Materials · Quantum Computing overlap)")
md.append(f"\n_Generated: 2026-03-16 | Source: NSF Awards Database (~162k awards)_\n")
md.append("---\n")

md.append("## Executive Summary\n")
md.append(f"- **Total awards identified:** {total_count}")
md.append(f"- **Total funding:** {fmt_m(total_amount)}")
md.append(f"- **Year range:** {years_sorted[0]}–{years_sorted[-1]}")
md.append(f"- **Avg award size:** {fmt_m(total_amount/total_count) if total_count else '$0'}")
md.append(f"- **Unique institutions:** {len(by_inst)}")
md.append(f"- **Unique divisions:** {len(by_div)}")
md.append("")

md.append("### Awards by Sub-Category (awards can appear in multiple)\n")
md.append("| Sub-Category | Awards | Total Funding | Description |")
md.append("|--------------|--------|--------------|-------------|")
md.append(f"| SCE-core | {by_tag['SCE-core']['count']} | {fmt_m(by_tag['SCE-core']['amount'])} | Strongly correlated, Mott, Kondo, Hubbard, spin liquids, cuprates |")
md.append(f"| Topological | {by_tag['Topological']['count']} | {fmt_m(by_tag['Topological']['amount'])} | Topological insulators/semimetals, Weyl/Dirac, Majorana, moiré |")
md.append(f"| Quantum-computing | {by_tag['Quantum-computing']['count']} | {fmt_m(by_tag['Quantum-computing']['amount'])} | QC algorithms, tensor networks, DMRG, quantum Monte Carlo, qubits |")
md.append("")

md.append("### Funding by Era\n")
md.append("| Period | Awards | Total Funding | Notes |")
md.append("|--------|--------|--------------|-------|")
md.append(f"| 2000–2009 | {p1[0]} | {fmt_m(p1[1])} | Pre-deep-learning era |")
md.append(f"| 2010–2016 | {p2[0]} | {fmt_m(p2[1])} | Early ML renaissance |")
md.append(f"| 2017–2020 | {p3[0]} | {fmt_m(p3[1])} | Graph NNs, DFT+ML, topo-ML boom |")
md.append(f"| 2021–2025 | {p4[0]} | {fmt_m(p4[1])} | Foundation models, QC-ML convergence |")
md.append("")

md.append("---\n")
md.append("## Year-over-Year Trend\n")
md.append("| Year | Awards | Total Funding | ΔAwards | ΔFunding | SCE-core | Topo | QC |")
md.append("|------|--------|--------------|---------|----------|----------|------|----|")
for i, yr in enumerate(years_sorted):
    c  = by_year[yr]["count"]
    am = by_year[yr]["amount"]
    sc = by_year[yr]["SCE-core"]
    tp = by_year[yr]["Topological"]
    qc = by_year[yr]["Quantum-computing"]
    if i > 0:
        prev_c  = by_year[years_sorted[i-1]]["count"]
        prev_am = by_year[years_sorted[i-1]]["amount"]
        dc = f"{(c-prev_c)/prev_c*100:+.0f}%" if prev_c else "—"
        da = f"{(am-prev_am)/prev_am*100:+.0f}%" if prev_am else "—"
    else:
        dc = da = "—"
    md.append(f"| {yr} | {c} | {fmt_m(am)} | {dc} | {da} | {sc} | {tp} | {qc} |")
md.append("")

md.append("---\n")
md.append("## By NSF Directorate\n")
md.append("| Directorate | Awards | Total Funding |")
md.append("|-------------|--------|--------------|")
for name, v in top_dir:
    md.append(f"| {name} | {v['count']} | {fmt_m(v['amount'])} |")
md.append("")

md.append("---\n")
md.append("## By NSF Division\n")
md.append("| Division | Awards | Total Funding |")
md.append("|----------|--------|--------------|")
for name, v in top_div:
    md.append(f"| {name} | {v['count']} | {fmt_m(v['amount'])} |")
md.append("")

md.append("---\n")
md.append("## Top 15 Institutions\n")
md.append("| Institution | Awards | Total Funding |")
md.append("|-------------|--------|--------------|")
for name, v in top_inst:
    md.append(f"| {name} | {v['count']} | {fmt_m(v['amount'])} |")
md.append("")

md.append("---\n")
md.append("## Top 10 States\n")
md.append("| State | Awards | Total Funding |")
md.append("|-------|--------|--------------|")
for name, v in top_state:
    md.append(f"| {name} | {v['count']} | {fmt_m(v['amount'])} |")
md.append("")

md.append("---\n")
md.append("## Representative Award Titles\n")
from random import seed, sample
seed(42)

def sample_titles(yr_range=None, tag=None, n=6):
    pool = awards
    if yr_range:
        pool = [r for r in pool if int(r["source_year"]) in yr_range]
    if tag:
        pool = [r for r in pool if tag in r.get("tags","")]
    return sample(pool, min(n, len(pool)))

def fmt_row(r):
    amt  = fmt_m(safe_amount(r["total_intended_amount"]))
    tags = r.get("tags","")
    return f"- [{r['award_id']}] ({r['source_year']}, {amt}, `{tags}`) _{r['title']}_"

md.append("### 2010–2016 (Early ML × quantum materials)")
for r in sample_titles(yr_range=range(2010,2017)):
    md.append(fmt_row(r))
md.append("")

md.append("### 2017–2020 (Graph NNs, DFT+ML, topo-ML)")
for r in sample_titles(yr_range=range(2017,2021)):
    md.append(fmt_row(r))
md.append("")

md.append("### 2021–2025 (Foundation models & QC-ML convergence)")
for r in sample_titles(yr_range=range(2021,2026), n=10):
    md.append(fmt_row(r))
md.append("")

md.append("### SCE-core highlights (all years)")
for r in sample_titles(tag="SCE-core", n=6):
    md.append(fmt_row(r))
md.append("")

md.append("### Topological materials highlights (all years)")
for r in sample_titles(tag="Topological", n=6):
    md.append(fmt_row(r))
md.append("")

md.append("### Quantum-computing overlap highlights (all years)")
for r in sample_titles(tag="Quantum-computing", n=6):
    md.append(fmt_row(r))
md.append("")

md.append("---\n")
md.append("## Keyword Taxonomy Used\n")
md.append("**ML/AI terms:**  \n" + ", ".join(ML_AI_TERMS))
md.append("\n**SCE-core terms:**  \n" + ", ".join(SCE_CORE))
md.append("\n**Topological terms:**  \n" + ", ".join(TOPO))
md.append("\n**Quantum-computing terms:**  \n" + ", ".join(QCOMP))
md.append("\n_Awards must match ≥1 ML/AI term AND ≥1 term from any SCE sub-category in title+abstract._\n")

report_path = OUT_DIR / "ml_sce_funding_report.md"
report_path.write_text("\n".join(md))
print(f"\nReport  → {report_path}")
print(f"CSV     → {csv_path}")
print(f"Trend   → {trend_csv}")
