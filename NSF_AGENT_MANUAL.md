# NSF Agent Manual

Local Ollama agent (`nsf_solr_agent.py`) with access to two data sources:

| Source | Contents | Requires |
|--------|----------|---------|
| **SOLR** | All proposals — awarded, declined, pending; full text, panel & reviewer data | NSF VPN |
| **SQLite** | 162k awarded grants 2010–2024; semantic search, PI profiles, SQL analytics | Nothing |

---

## Starting the Agent

```bash
# Default — llama3.2, native tool-calling
python nsf_solr_agent.py

# Larger context window (better for long reports)
python nsf_solr_agent.py --model llama3.1

# Save session to markdown
python nsf_solr_agent.py --out my_report.md

# Show tool calls as they happen
python nsf_solr_agent.py --verbose

# Custom index directory (if output/ is elsewhere)
python nsf_solr_agent.py --output /path/to/output

# phi4 or other models without native tool support
python nsf_solr_agent.py --model phi4 --react
```

Type `tools` to list available tools. Type `quit` or press `Ctrl+C` to exit.

---

## SOLR Queries

### Status values

The `status` field stores full strings — **always quote them**.

| Intent | Query |
|--------|-------|
| Awarded | `status:"Proposal has been awarded"` |
| PM recommends award | `status:"Pending, PM recommends award"` |
| Concurred for award | `status:"Recommended for award, DDConcurred"` |
| Declined | `status:"Decline, DDConcurred"` |
| PM recommends decline | `status:"Pending, PM recommends decline"` |
| In review | `status:"Pending, Review Package Produced"` |
| Assigned to PM | `status:"Pending, Assigned to PM"` |

`status:Awarded` returns **nothing** — the field value is never a single word.

To match all awarded statuses at once:
```
status:("Proposal has been awarded" OR "Pending, PM recommends award" OR "Recommended for award, DDConcurred")
```

### Panel queries

Panels are identified by `panel_id` (e.g. `P260135`), not `panel_name`.

```
# All proposals in a panel
panel_id:P260135

# Awarded proposals in a panel
panel_id:P260135 AND status:"Proposal has been awarded"

# Declined proposals in a panel
panel_id:P260135 AND status:"Decline, DDConcurred"

# Find panel IDs for a directorate/year
→ use facet_proposals with facet_field="panel_id"
```

### Common Lucene patterns

```
# By PI (exact name match)
pi_name:"Jane Smith"

# Keyword in summary
summary:(quantum computing)

# Keyword in full narrative
description:(coordination failure infrastructure)

# Directorate + year
directorate:BIO AND received_year:2024

# Award amount range
award_amount:[500000 TO *]

# Institution
inst:"University of Michigan"

# Gender / demographics
pi_gender:F AND directorate:ENG AND received_year:2023
```

### Key SOLR fields

| Group | Fields |
|-------|--------|
| Identity | `id`, `title` |
| Text | `summary`, `description`, `bio`, `data_management` |
| PI | `pi_name`, `pi_all`, `pi_email`, `inst`, `inst_state`, `pi_gender`, `pi_race`, `pi_ethnicity` |
| Funding | `award_amount`, `award_date`, `requested_amount`, `funding_program` |
| Organization | `directorate`, `division`, `managing_program` |
| Dates | `received`, `received_year`, `status` |
| Panel | `panel_id`, `panel_name`, `panel_reviewers`, `panel_start_date`, `panel_end_date` |
| Reviewer | `reviewer_name`, `reviewer_inst`, `reviewer_gender` |

Use `proposal_fields` tool (or type `proposal_fields fi_pnl`) for the full field group listing.

### Example prompts

```
list all proposals in panel P260135 with titles and PI names
what is the award rate for panel P260135?
facet proposals by panel_id for directorate:BIO AND received_year:2024
search for proposals on infrastructure for genomics databases
get full summary and description for proposal 2535312
show me proposals by pi_name:"Jane Smith" with award amounts
```

---

## SQLite Tools

The local database covers **awarded grants only** (2010–2024). No VPN needed.

### sql_query — flexible aggregations

```sql
-- Funding by directorate in 2023
SELECT d.abbreviation, COUNT(*) n, SUM(a.award_amount)/1e6 AS total_m
FROM award a JOIN directorate d ON d.id = a.directorate_id
WHERE a.source_year = 2023
GROUP BY d.abbreviation ORDER BY total_m DESC

-- Top institutions by award count
SELECT i.name, i.state_code, COUNT(*) n
FROM award a JOIN institution i ON i.id = a.institution_id
GROUP BY i.name ORDER BY n DESC LIMIT 20

-- Award rate by year (requires SOLR for declined count — use for trend shape)
SELECT source_year, COUNT(*) n_awards, SUM(award_amount)/1e6 total_m
FROM award GROUP BY source_year ORDER BY source_year

-- Top PIs by total funding
SELECT ai.full_name, COUNT(*) n_awards, SUM(a.award_amount) total
FROM award_investigator ai JOIN award a ON a.id = ai.award_id
WHERE ai.role_code = 'Principal Investigator'
GROUP BY ai.full_name ORDER BY total DESC LIMIT 20
```

Key tables: `award`, `directorate`, `division`, `institution`, `award_investigator`,
`program_element`, `award_program_element`, `researcher_fingerprint`, `researcher_papers`.

### semantic_search — concept discovery

Finds awards by meaning, not keywords. Uses SPECTER2 embeddings.

```
find awards about shared infrastructure for extremophile genomics
find proposals similar to: "coordination failure in protein structure databases"
semantic search for quantum error correction in BIO directorate 2020–2024
```

### hybrid_search — precise topic matching

Combines SPECTER2 + TF-IDF domain concepts + BM25. Better than semantic alone for specific areas.

```
hybrid search for "data repository standards proteomics" in CSE
```

### get_award — full award record

```
get award 2535312
```

Returns abstract, investigators (PI + Co-PIs), institution, program element codes.

### get_researcher — PI profile

```
get researcher profile for Jane Smith
```

Returns NSF funding history, publications (if fingerprint built), top research topics.
Falls back to raw award list if fingerprint not yet computed.

### db_schema — explore the database

```
show me the database schema
```

---

## Two-Step Workflow: Semantic Discovery → SOLR Full Text

The most powerful pattern — use SQLite to find relevant award IDs by concept,
then fetch full proposal text (including declined proposals) from SOLR.

```
Step 1: semantic search for "protein structure coordination infrastructure"
Step 2: fetch full summaries from SOLR for those IDs
```

Or in one prompt:
```
find the top 10 awards related to shared genomics infrastructure,
then pull their full SOLR summaries
```

---

## CSV → Panel Builder

Reads proposal IDs from any CSV, fetches summaries from SOLR, and writes
a formatted file for `nsf_panel_builder.py`.

```
convert /path/to/panel_proposals.csv to panel input

# Specify column and use full narrative instead of summary
convert /path/to/panel.csv to panel input using column "Prop_ID" and text field description
```

The ID column is auto-detected (looks for columns named `id`, `prop`, `award`, `number`).
Output is written to `<csv_name>.panel_input.txt` unless `out_path` is specified.

**Output format** (one block per proposal):
```
Proposal Title
Full summary text...
---
Next Proposal Title
Full summary text...
---
```

This file is pasted directly into the panel builder app.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| SOLR returns nothing | Check NSF VPN is connected |
| `status:Awarded` returns 0 results | Use `status:"Proposal has been awarded"` |
| Semantic search fails | Check `output/embeddings_specter2.npy` exists; use `--output` flag if indices are elsewhere |
| `phi4 does not support tools` | Add `--react` flag |
| Agent loops without answering | Use `--verbose` to see which tool is being called |
| CSV tool can't find ID column | Specify explicitly: `id_column="YourColumnName"` |
