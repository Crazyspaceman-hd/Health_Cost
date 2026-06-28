# Synthetic Health Cost Growth Target Analytics

A portfolio project that simulates the analytics workflow of a **state health
cost growth target program** — the kind of program that collects annual data
from insurers, hospitals, and provider organizations and evaluates healthcare
cost growth against a per-capita benchmark (here, **3.4%**).

The project demonstrates the full analyst workflow on **100% synthetic data**:
data generation, data-quality validation, claims/enrollment analysis, cost &
utilization metrics (PMPM, year-over-year growth, growth-vs-target), reporting
automation, and stakeholder-ready summaries.

> **No real PHI, no real claims, no real diagnoses.** Every record is randomly
> generated from documented assumptions in `config.yaml`. "Condition groups"
> are broad synthetic buckets, never real diagnosis codes. This project makes
> **no claim of government or program experience** — it is a simulation.

## Why synthetic data?

Real health-claims data is protected (PHI/HIPAA) and cannot be used in a public
portfolio. Generating synthetic data with a *known, seeded* economic story has a
second benefit: because the "right answers" are designed in, the analysis and
tests can be verified against ground truth.

## Project status — built in phases

- [x] **Phase 1 — Synthetic data generation** *(complete)*
- [x] **Phase 2 — Data-quality validation engine** *(complete)*
- [x] **Phase 3 — Cost & utilization metrics (PMPM, YoY, target comparison)** *(complete)*
- [x] **Phase 4 — Reporting output tables** *(complete — emitted by Phases 2–3)*
- [x] **Phase 5 — Visualizations** *(complete)*
- [x] **Phase 6 — Executive summary + pipeline + README** *(complete)*
- [x] **Phase 7 — Streamlit dashboard** *(complete)*

**The whole project is built.** Run everything with one command:

```bash
python -m src.pipeline                 # batch: generate -> validate -> db -> metrics -> charts -> report
streamlit run src/dashboard.py         # interactive dashboard over the same engines
```

## Data model (six source tables)

| Table | Grain | Key fields |
|---|---|---|
| `dim_payer` | one row per payer | payer_id, payer_type, base_pmpm, annual_trend |
| `dim_provider_org` | one row per org | provider_org_id, region, org_type |
| `ref_service_category` | one row per category | service_category, mean_allowed, extra_trend |
| `ref_condition_group` | reference | condition_group (broad synthetic buckets) |
| `fact_enrollment` | **member-month** | member_id, payer_id, provider_org_id, enrollment_month, age_band, region, line_of_business |
| `fact_claims` | **claim** | claim_id, member_id, payer_id, provider_org_id, service_date, service_category, allowed_amount, paid_amount, member_cost_share, claim_status, received_date |

### Seeded economic story (the "ground truth")

- **3 lines of business**: Medicaid, Medicare Advantage, Commercial.
- **PAY001** and **PAY004** are seeded to grow **above** the 3.4% target;
  **PAY003** **below**. **PAY002** is the designated *data-quality* payer — its
  2023 submission is deliberately incomplete to exercise the validation engine.
- **Pharmacy** and **behavioral health** are seeded with above-average cost
  trend, so they emerge as identifiable cost drivers.
- ~2% of members are high-cost claimants driving a disproportionate share of
  spend. Underlying trend is measured with **high-cost truncation** at $50k —
  standard cost-growth methodology.
- A controlled set of **data-quality defects** (duplicates, negative amounts,
  paid>allowed, orphan claims, missing fields, invalid dates, enrollment gaps,
  an incomplete submission) is injected so validation has real findings.

All parameters live in **`config.yaml`** and are fully documented there.

## How to run

```bash
pip install -r requirements.txt

# Generate the six synthetic source tables into data/raw/
python -m src.generate_data

# Run the 13 data-quality checks -> outputs/validation_summary.csv
#                                   outputs/data_quality_issues.csv
python -m src.validate

# Run the test suite
python -m pytest -q
```

(Or, with `make`: `make data`, `make validate`, `make test`, `make all`.)

### Validation engine (Phase 2)

13 documented checks run against the source tables, each emitting a severity,
an issue count, the affected records, and a recommended follow-up — modeled on
how a data-submission program reviews each annual filing:

| Severity | Checks |
|---|---|
| CRITICAL | duplicate claim IDs · negative amounts · paid > allowed |
| HIGH | missing payer ID · orphan claims · invalid dates · claims outside enrollment · payer/provider completeness · YoY volume change (incomplete submission) |
| MEDIUM | missing provider ID · missing service category · enrollment gaps |
| LOW | high-cost outlier claims (informational; drives truncation) |

Output: `outputs/validation_summary.csv` (one row per check) and
`outputs/data_quality_issues.csv` (one row per affected record).

### Metrics engine (Phase 3)

Computed on an **analytic claims** base (raw claims with the Phase 2 defects
removed). PMPM is reported both **raw** and **high-cost truncated** at $50k:

- member months · total medical expense (allowed) · total paid · cost share
- PMPM, year-over-year PMPM growth, and growth **vs the 3.4% target**
- the same growth cut by **payer**, **provider org**, **line of business**, and
  **service category** (with utilization per 1,000 member-months)
- a **price-vs-utilization decomposition** (PMPM = utilization × unit price) that
  separates cost growth driven by *more services* from growth driven by *higher
  cost per service*
- high-cost member / claim spend **concentration**

Metrics are implemented twice — in **pandas** (`src/metrics.py`) and in **SQL**
against a local **SQLite** database (`src/load_db.py`, `src/sql/*.sql`) — and a
test cross-checks that both agree. Output tables: `payer_cost_growth_summary`,
`provider_cost_growth_summary`, `line_of_business_summary`,
`service_category_trends`, `high_cost_concentration`, `executive_summary_metrics`.

**Illustrative findings on the generated data** (truncated PMPM):
PAY001 (+6.9% CAGR) and PAY004 (+4.7%) exceed the 3.4% target; PAY003 (+2.2%) is
below. Across the two complete years, **behavioral health (+9.1%)** and
**pharmacy (+8.2%)** are the top cost drivers. PAY002's 2023 filing is flagged
incomplete (DQ013), so its trend is excluded from conclusions.

### Visualizations (Phase 5)

`python -m src.visualize` renders eleven dashboard-ready charts to
`outputs/figures/`: PMPM by year, growth vs target, payer & provider cost
growth, a grouped service-category-by-provider breakdown for the highest-growth
providers, service-category growth (drivers), a **price-vs-utilization**
decomposition, utilization per 1,000 member-months,
a high-cost-member **Lorenz concentration curve** (with Gini), data-quality
issues by severity, a **found-vs-resolved** chart showing issues are handled
before analysis, and a payer × provider cost-growth heatmap.
The cost-growth charts use the **completeness-adjusted** base (the validation-
flagged incomplete payer-year is excluded, with an on-chart footnote), while the
data-quality chart reflects all issues in the raw submissions.

![Cost growth by payer](outputs/figures/payer_cost_growth.png)

### Executive summary & pipeline (Phase 6)

`python -m src.report` generates `reports/executive_summary.md` — a plain-English,
health-policy-style writeup that reads its numbers off the *actual* computed
tables (nothing hard-coded), covering what was analyzed, data-quality findings,
who exceeded the target, the cost drivers, limitations, and next steps.
`python -m src.pipeline` runs the entire workflow end to end in one process.

A note on methodology baked into the report: an incomplete payer submission is
**excluded from pooled trends across all years** (a composition-consistent
panel) and reported as *"not assessed"* at the payer level, rather than being
classified against the target on too few complete years — the way a real
cost-growth program handles a filing it has deemed incomplete.

### Interactive dashboard (Phase 7)

`streamlit run src/dashboard.py` launches an interactive dashboard built on the
**same** metrics, validation, and visualization engines as the batch pipeline
(no duplicated logic), with headline KPIs and tabs for overview, payers,
provider organizations, service categories, data quality, and the live
executive summary. A sidebar toggle switches between the raw and
completeness-adjusted views. If the raw tables aren't present, the app
generates them in-memory on first load.

## Repo structure

```
src/            generation, (coming) validation, metrics, reporting, viz
data/raw/       generated source tables (git-ignored, reproducible)
data/processed/ cleaned/joined analytic tables
outputs/        summary CSVs and figures
reports/        generated executive summary
tests/          pytest suite
config.yaml     all tunable parameters + the seeded story
```

## Skills demonstrated

Synthetic data engineering · data validation / QA · claims & enrollment
analytics · PMPM and cost-growth metrics · year-over-year trend analysis ·
high-cost truncation methodology · reproducible pipelines · pytest · Python /
pandas / NumPy.

## Limitations

This is a **simulation for skill demonstration**, not a model of any real
population, payer, or program. Magnitudes are illustrative. It is not a
risk-adjustment, actuarial, or attribution methodology.
