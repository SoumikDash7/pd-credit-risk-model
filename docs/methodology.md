# PD Credit Risk Model — Methodology & Decision Log

This document tracks key modeling decisions as they are made, with the
reasoning behind each — the same way a model development document would
be written for independent validation review at a bank.

---

## 1. Dataset

**Source:** Lending Club accepted loans, 2007–2018 (Kaggle mirror:
`wordsforthewise/lending-club`, file `accepted_2007_to_2018Q4.csv.gz`)

**Why this dataset over a pre-cleaned alternative (e.g. "Give Me Some
Credit"):** The goal of this project is to demonstrate the full PD
model development lifecycle, including data preprocessing and missing
value treatment. Pre-cleaned competition datasets (10–12 columns, no
raw dates/categoricals) skip that work. Lending Club's raw data has
151 columns, mixed types, genuine structural and random missingness,
and requires the analyst to define the target variable from
`loan_status` rather than being handed a clean binary label — all of
which mirrors a real bank model development exercise more closely.

**Scale:** 2,260,701 rows × 151 columns raw.

**Technical note:** the full file could not be loaded into memory on
this machine in one `pd.read_csv()` call (`pandas.errors.ParserError:
... out of memory`). All exploratory passes are therefore done via
chunked reads (`chunksize=200_000`), accumulating aggregate statistics
(null counts, value counts, groupby sums) across chunks rather than
materializing the full dataframe. This will also inform the
preprocessing pipeline design — later stages need to either continue
chunking, downcast dtypes aggressively, or load a reduced column set.

---

## 2. Missingness structure

Full map saved at `reports/missingness_map.csv`. Key finding: **most
severe missingness is structural, not random.**

| Missingness band | Example columns | Interpretation |
|---|---|---|
| 94–100% | `member_id`, `hardship_*`, `settlement_*`, `sec_app_*`, `*_joint` | Only populated when a specific condition applies (hardship plan entered, debt settlement reached, joint application). Missing = "condition not applicable," which is itself informative — not a data quality defect. |
| 38–84% | `mths_since_last_record`, `il_util`, `open_il_*`, `all_util`, etc. | Lending Club began collecting these fields partway through its history. Missingness here is largely a **vintage/reporting-availability effect**, to be verified against `issue_d` before deciding on imputation. |
| 0–13% | `emp_title`, `emp_length`, `mths_since_recent_inq`, various `num_*`/`mo_sin_*` tradeline fields | Genuine borrower-level gaps. These will need real imputation treatment (and will be the main "missing value handling" showcase of this project). |
| ~0% | `loan_amnt`, `term`, `int_rate`, `grade`, `installment`, `annual_inc`, `dti`, FICO fields, `purpose` | Core origination fields, essentially complete. |

**Planned treatment (to be finalized in the feature engineering
stage):** structural-missing columns → either drop (if post-origination/
leakage risk) or convert to indicator flags; vintage-driven missing →
check availability by `issue_d` before imputing; borrower-level gaps →
standard imputation (median/mode or model-based) with missingness
indicator flags retained where missingness itself may be predictive.

---

## 3. Target variable definition

Raw `loan_status` distribution (full population, `n = 2,260,701`):

| Status | Count | % |
|---|---|---|
| Fully Paid | 1,076,751 | 47.63% |
| Current | 878,317 | 38.85% |
| Charged Off | 268,559 | 11.88% |
| Late (31-120 days) | 21,467 | 0.95% |
| In Grace Period | 8,436 | 0.37% |
| Late (16-30 days) | 4,349 | 0.19% |
| Does not meet credit policy: Fully Paid | 1,988 | 0.09% |
| Does not meet credit policy: Charged Off | 761 | 0.03% |
| Default | 40 | 0.00% |
| NaN | 33 | 0.00% |

**Mapping used:**
- **Good (target = 0):** `Fully Paid`
- **Bad (target = 1):** `Charged Off`, `Late (31-120 days)`, `Default`
- **Excluded:**
  - `Current` — censored outcome; the loan has not finished its term,
    so its eventual good/bad status is unknown. Labeling these as
    "good" would be incorrect by construction, not just noisy.
  - `In Grace Period`, `Late (16-30 days)` — early-stage delinquency
    that could still cure or progress; forcing a binary label here
    would be guessing an unresolved outcome.
  - `Does not meet the credit policy: *` — originated under a
    different/earlier underwriting policy than the rest of the
    dataset. Including these would mix two non-comparable
    populations into one model, violating the principle of modeling
    on a consistent population.
  - `NaN` — dropped (33 rows, negligible).

---

## 4. Sample window — vintage maturity filter (avoiding survivorship bias)

**Problem:** Simply excluding `Current` loans and using all remaining
resolved loans introduces **survivorship/selection bias**. Recent
vintages (e.g. 2017–2018) that appear "Fully Paid" are disproportionately
short-duration loans that happened to resolve quickly; most 2017–2018
originations — especially 60-month loans — are still `Current` and
would be excluded, skewing the resolved sample away from a
representative cross-section of recent originations.

**Fix:** Restrict the modeling population to loans whose **maturity
date** (`issue_d + term`) falls before the data's effective snapshot
date, so every included loan has had the full opportunity to reach a
final outcome.

- Snapshot date determined empirically from the data: max
  `last_pymnt_d` = 2019-03, max `last_credit_pull_d` = 2019-04. Used
  **2019-01-01** as the effective snapshot with a buffer for reporting
  lag.
- Resulting cutoffs: **36-month loans** must be issued on or before
  **Jan 2016**; **60-month loans** must be issued on or before **Jan
  2014**.

This is the standard approach for defining a PD model's development
sample window at a real institution — pick a window where outcomes
are fully observed, rather than an arbitrary recent cutoff.

---

## 5. Build log

| Date | Step | Script | Output |
|---|---|---|---|
| TBD | Raw inspection (shape, columns) | `notebooks/01_raw_inspection.py` | 2,260,701 rows × 151 cols |
| TBD | Missingness map | `notebooks/02_missingness_and_target.py` | `reports/missingness_map.csv` |
| TBD | Target status counts | `notebooks/03_target_definition.py` | `reports/loan_status_counts.csv` |
| TBD | Vintage × term distribution | `notebooks/04_vintage_check.py` | `reports/vintage_by_term.csv` |
| TBD | Snapshot date check | `notebooks/05_snapshot_date.py` | max last_pymnt_d / last_credit_pull_d |
| TBD | Target + maturity filter construction | `notebooks/06_build_target.py` | `data/processed/modeling_sample_ids_target.csv` |

*(Fill in actual dates as you go — this table is the running audit
trail of the project.)*

---

## Next steps

- Feature engineering: parse raw string fields (`term`, `emp_length`,
  `revol_util`, dates) into usable numeric/categorical features
- **Leakage audit**: identify and exclude columns only known
  post-origination (`total_pymnt`, `recoveries`, `last_pymnt_d`,
  hardship/settlement fields, etc.) — a PD model must only use
  information available *at the time of origination*
- Missing value treatment per the plan in Section 2
- Train/test split strategy (likely time-based, given vintage
  structure)
- WoE/IV binning, model development, validation suite (AUC, KS, Gini,
  PSI), SHAP interpretation

---

## Update — Target construction run (Section 3 & 4 results)

Running `notebooks/06_build_target.py`:

- Rows excluded as immature vintage: **1,561,567**
- Rows excluded as ambiguous/censored status (among mature loans): **2,902**
- **Final modeling sample: 696,232 loans**
- **Bad rate: 14.82%** (103,165 bad / 696,232 total)

Note: this bad rate is meaningfully lower than the ~21.6% naive estimate
from excluding only `Current` without a maturity filter. This is
consistent with the survivorship bias hypothesis in Section 4 — the
naive resolved-loan sample over-represented loans that failed quickly,
inflating the apparent bad rate. The maturity-filtered sample is
considered the more reliable, representative estimate of true
portfolio-level default risk.

---

## 6. Leakage audit and feature eligibility

A PD model may only use information available at loan origination.
All 151 raw columns were classified (`notebooks/07_leakage_audit.py`,
output in `reports/column_classification.csv`):

| Category | Columns | Treatment |
|---|---|---|
| Origination candidates | 100 | Eligible, subject to in-sample availability check |
| Post-origination leakage | 38 | Excluded (payment history, balances, recoveries, hardship and settlement fields, last-pull dates) |
| ID / target | 5 | Excluded |
| High-cardinality text | 4 | `emp_title`, `title`, `desc`, `zip_code`: special handling or exclusion |
| Lender-assigned | 3 | `grade`, `sub_grade`, `int_rate`: Lending Club's own risk output, so kept out of the primary model and reserved as a benchmark |
| Time reference | 1 | `issue_d`: used for time-based splitting, not as a predictor |

Notable: `last_fico_range_high/low` look like ordinary FICO scores but
are pulled after origination, so they are excluded; only
`fico_range_low/high` at application are valid.

Open question: several candidate fields were only collected for later
vintages, so full-file missingness can overstate availability in the
maturity-filtered sample. Availability is therefore re-measured within
the modeling sample before any treatment decisions are made.
