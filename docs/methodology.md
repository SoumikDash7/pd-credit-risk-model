# PD Credit Risk Model — Methodology & Decision Log

This document tracks key modeling decisions as they are made, with the
reasoning behind each — the same way a model development document would
be written for independent validation review at a bank.

---

## Current rules at a glance

Quick reference to the rule in force for each topic and where it is documented.
Where a later section changes an earlier one, the earlier text is kept for the
audit trail and listed under "Replaces".

| Topic | Rule in force | Section | Replaces |
|---|---|---|---|
| Source data | Lending Club accepted loans 2007-2018 Q4, 2,260,701 rows x 151 columns, read in 200K-row chunks | 1 | |
| Target | Good = Fully Paid; Bad = Charged Off, Late (31-120 days), Default; Current, grace period, Late (16-30), policy-exception and blank statuses excluded | 3 | |
| Maturity filter | Loan kept only if issue date + term is on or before 2019-01-01 | 4, 5b | |
| Development population | 36-month loans issued Jan 2013 - Jan 2016: 568,694 loans, 14.10% bad | 7, 8 | The 696,232-loan maturity-filtered sample of Sections 4 and 5b (now an intermediate step) |
| Split | Train 307,025 and validation 76,757 (random, Jan 2013 - Jun 2015); out-of-time 184,912 (Jul 2015 - Jan 2016) | 9 | |
| Eligible columns | Origination-time columns only; 38 post-origination columns excluded; grade, sub_grade, int_rate kept as benchmark only | 6, 10 | |
| Missing values | Structural gaps get their own bin; revol_util, pct_tl_nvr_dlq and avg_cur_bal get a train median; no other imputation | 9, 11 | Section 2 planned treatment, Section 7 working hypothesis and Section 8 residual-gap plan (bankcard fields are structural, not median-imputed) |
| Rare categorical levels | Levels under 0.2% of train grouped to OTHER | 10 | The initial 1% rule noted in Section 10 |
| Binning | Entropy-tree bins (min size max(1,000 loans, 2% of non-missing)), monotone merge, WoE = ln(%good/%bad) | 11 | |
| Dropped after binning | initial_list_status, num_tl_120dpd_2m, addr_state | 11 | |
| Limitations and governance | Accepted-loans-only scope, label vs Basel definition, excluded loans, fair-lending screen, drift and recalibration plan | 12 | |

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

## 5. Build log (audit table)

One row per script. Dates are commit dates (not necessarily the day a script was first
run) and commits are mapped to scripts by commit message. Scripts 01-06 were first
committed under Notebooks/; the folder was renamed to lowercase in commit 04e9fef.

| Date | Commit | Script | What it does | Main output |
|---|---|---|---|---|
| 2026-10-07 | 1faf713 | notebooks/01_raw_inspection.py | Raw shape and column list | 2,260,701 rows x 151 columns |
| 2026-10-07 | 1faf713 | notebooks/02_missingness_and_target.py | Chunked missingness map | reports/missingness_map.csv |
| 2026-10-07 | 1faf713 | notebooks/03_target_definition.py | loan_status counts | reports/loan_status_counts.csv |
| 2026-10-07 | 1faf713 | notebooks/04_vintage_check.py | Loans by issue year and term | reports/vintage_by_term.csv |
| 2026-10-07 | 1faf713 | notebooks/05_snapshot_date.py | Effective snapshot date | max last_pymnt_d 2019-03; max last_credit_pull_d 2019-04 |
| 2026-10-07 | 1faf713 | notebooks/06_build_target.py | Maturity filter and target mapping | data/processed/modeling_sample_ids_target.csv (696,232 loans, 14.82% bad) |
| 2026-10-08 | f97ba09 | notebooks/07_leakage_audit.py | Column classification | reports/column_classification.csv |
| 2026-10-08 | f97ba09 | notebooks/08_sample_missingness.py | Missingness inside the modeling sample | reports/sample_missingness.csv, reports/sample_missingness_by_year.csv (both committed in 229f343) |
| 2026-10-08 | 229f343 | notebooks/09_missingness_by_year_and_bad_rate.py | Missingness by year; bad rate by year and term | console output (Section 7) |
| 2026-10-08 | 04e9fef | notebooks/10_development_sample.py | Development sample and missingness diagnostics | data/processed/development_sample_ids_target.csv (568,694 loans, 14.10% bad); reports/dev_sample_*.csv |
| 2026-10-08 | d23d97c | notebooks/11_split_and_missing_flags.py | Train / validation / out-of-time split; missing-flag signal | data/processed/development_split.csv (not tracked); reports/missing_flag_signal.csv |
| 2026-10-08 | 7e9f94a | notebooks/12_feature_engineering.py | Feature table | data/processed/features.parquet (not tracked); reports/feature_dictionary.csv, categorical_levels.csv, winsor_thresholds.csv |
| 2026-10-10 | a33ebec | notebooks/13_woe_binning.py, src/woe.py | WoE/IV binning | models/woe_spec.json; reports/woe_bins.csv, reports/iv_summary.csv |
| 2026-10-10 | (Section 12 commit) | notebooks/14_governance_checks.py | Label composition; excluded-loan breakdown and bounds | console output (Section 12) |

---

## 5b. Target construction run results (supplements Sections 3 and 4)

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


---

## 7. Missingness inside the modeling sample, and development scope

Missingness was re-measured inside the 696,232-loan modeling sample
(`notebooks/08_sample_missingness.py`, `notebooks/09_missingness_by_year_and_bad_rate.py`).

**Finding 1: availability is era-driven, not random.**
About 30 bureau-attribute columns (`tot_cur_bal`, `num_sats`,
`avg_cur_bal`, `mort_acc`, `total_rev_hi_lim`, ...) are 100% missing for
2007-2011, ~52% missing in 2012, and ~0% from 2013 onward. Averaging
over the full sample hid this (it showed as 3-10% missing). Imputing
these would fabricate values for pre-2013 loans, so the sample window
is cut instead.

**Finding 2: later-collected fields.**
Roughly 30 further columns (`sec_app_*`, `*_joint`, `open_acc_6m`,
`il_util`, `all_util`, `inq_last_12m`, ...) are >=94% missing in the
sample because they were only populated for late-2015+ loans (or joint
applications). They are dropped. Limitation: they exist only for loans
too recent to have outcomes, so the model reflects what is learnable
from mature vintages.

**Finding 3: 60-month loans.**
60-month loans default at 21-28% vs 10-15% for 36-month loans, and
appear only through 2013 because later ones have not matured. They
could never appear in an out-of-time test.

**Decision: development sample = 36-month loans issued from Jan 2013.**
568,694 loans (from the 696,232-loan modeling sample: 93,153 pre-2013
loans and 34,385 60-month 2013 loans removed). Exact bad rate and
monthly counts are recorded in Section 8 after
`notebooks/10_development_sample.py`. `term` is no longer a feature.
Scoping a PD model to a single product is standard practice.

**Population drift (noted for the PSI/validation sections):** the
36-month bad rate rises from 12.3% (2013) to 13.7% (2014) to 14.9%
(2015).

**Remaining gaps after 2013** (`mths_since_recent_inq` ~11%,
`mo_sin_old_il_acct` ~3.6%, `bc_util` family ~1%, `emp_length` 4-7%,
`num_tl_120dpd_2m` 4-6%, and the `mths_since_*` delinquency fields at
51-85%): the working hypothesis is that missing means "does not apply"
(e.g. no installment accounts, no bankcards, no delinquency ever). This
is tested in script 10 before any treatment is chosen. Imputation
statistics will be fitted on training data only.


---

## 8. Development sample and missingness diagnostics

Script: notebooks/10_development_sample.py. Outputs:
reports/dev_sample_by_month.csv, reports/dev_sample_missingness.csv.

**Development sample:** 36-month loans issued Jan 2013 - Jan 2016.
568,694 loans (row-match check against the raw file passed), 80,202
bad, **14.10% bad rate**. By issue year: 2013 100,422 loans (12.33%);
2014 162,570 (13.73%); 2015 283,087 (14.90%); Jan 2016 22,615 (14.66%).

**Candidate availability (100 candidates):** 30 are >=90% missing
(dropped), 6 are 10-90% (`mths_since_*` fields), 5 are 1-10%, 59 are
<1%.

**Near-constant / constant (dropped):** `application_type` (99.92%
Individual), `disbursement_method` (99.97% Cash), `term` (all 36
months). `initial_list_status` is balanced (f 49.3% / w 50.7%) and is
retained for now.

**Does "missing" mean "does not apply"?** Missing rate by value of a
related condition column:

| Column | Condition column | Missing if 0 | Missing if >0 | Reading |
|---|---|---|---|---|
| mths_since_last_record | pub_rec | 100.0% | 0.0% | Structural: no public record |
| mo_sin_old_il_acct | num_il_tl | 99.6% | 0.0% | Structural: no installment accounts |
| mths_since_recent_inq | inq_last_6mths | 18.6% | 0.0% | Consistent with no recent inquiry; not fully verifiable (inq_last_12m dropped) |
| mths_since_last_delinq | delinq_2yrs | 61.8% | 1.5% | Mostly structural; ~1.7K inconsistent loans |
| mths_since_last_major_derog | num_accts_ever_120_pd | 93.0% | 3.6% | Mostly structural; ~4.9K inconsistent loans |
| mths_since_recent_bc | num_bc_tl | 99.5% | 0.7% | Structural for ~1,055 loans; ~4K residual gaps |
| bc_util / percent_bc_gt_75 / bc_open_to_buy | num_bc_tl | 91.6% / 100% / 91.6% | 0.9% / 0.9% / 0.8% | Mostly not structural: only ~1,060 loans lack bankcards, about one-sixth of the gaps |

`emp_length` is missing for 91.1% of loans with a blank `emp_title`
and 0.0% of loans with a title; about 99% of its gaps coincide with
no employer information being supplied.

Not tested: `mths_since_recent_bc_dlq`, `mths_since_recent_revol_delinq`
(no clean condition column); treated analogously, to be checked at the
binning stage.

**Planned treatment:** structural gaps (`mths_since_*`,
`mo_sin_old_il_acct`) get a missing indicator and a dedicated bin, no
imputation; `emp_length` gets a "no employment info" indicator and
Unknown category; residual gaps (`bc_*` family, `num_tl_120dpd_2m`)
get median imputation fitted on training data only plus a missing
flag. Whether each flag carries signal is tested by comparing bad
rates of missing vs present groups before the plan is finalised.


---

## 9. Train / validation / out-of-time split and missing-flag signal

Script: notebooks/11_split_and_missing_flags.py (seed 42). Outputs:
reports/missing_flag_signal.csv; data/processed/development_split.csv
(not tracked, regenerated by the script).

| Split | Window | Loans | Bad | Bad rate |
|---|---|---|---|---|
| Train | Jan 2013 - Jun 2015, 80% stratified random | 307,025 | 42,370 | 13.80% |
| In-time validation | same window, 20% | 76,757 | 10,592 | 13.80% |
| Out-of-time | Jul 2015 - Jan 2016 | 184,912 | 27,240 | 14.73% |

The out-of-time bad rate is 0.93pp above development: calibration may
drift even if rank-ordering holds. Feature decisions use train rows
only; the out-of-time set stays untouched until final assessment.

**Does the missing group behave differently?** Bad rate, missing vs
present, train rows only; same-year diff compares within issue year to
remove drift (missing minus present, percentage points):

| Column | % missing | Bad % missing / present | Same-year diff |
|---|---|---|---|
| mths_since_last_record | 82.8 | 13.44 / 15.52 | -1.76 |
| mths_since_last_major_derog | 72.8 | 13.34 / 15.03 | -1.47 |
| mths_since_recent_bc_dlq | 74.3 | 13.57 / 14.45 | -0.76 |
| mths_since_last_delinq | 50.5 | 13.48 / 14.12 | -0.48 |
| mths_since_recent_revol_delinq | 65.0 | 13.59 / 14.19 | -0.45 |
| mths_since_recent_inq | 10.3 | 9.84 / 14.26 | -4.40 |
| mo_sin_old_il_acct | 3.8 | 16.48 / 13.69 | +2.85 |
| mths_since_recent_bc | 0.9 | 16.02 / 13.78 | +2.14 |
| bc_util | 1.1 | 16.27 / 13.77 | +2.42 |
| percent_bc_gt_75 | 1.0 | 16.05 / 13.78 | +2.19 |
| bc_open_to_buy | 1.0 | 16.19 / 13.78 | +2.35 |
| num_tl_120dpd_2m | 2.9 | 17.01 / 13.71 | +2.89 |
| emp_length | 5.8 | 20.28 / 13.40 | +6.79 |
| emp_title | 6.6 | 19.71 / 13.38 | +6.38 |

**Bankcard gaps are structural (supersedes the Section 8 reading).**
Conditioning on num_bc_tl explained only ~1,060 loans; conditioning on
num_bc_sats (satisfactory bankcard accounts) explains almost all:
96.5-98.7% of the 5,331 loans with num_bc_sats == 0 are missing each of
bc_util, percent_bc_gt_75, bc_open_to_buy, mths_since_recent_bc, vs
0.0-0.1% when num_bc_sats > 0. num_actv_bc_tl == 0 does not explain
them (49-56%). The gaps cluster: of 567,634 loans with a bankcard,
562,379 miss none of the four, 4,144 miss all four, 450 miss three,
661 miss one.

**Revised treatment (supersedes the residual-gap plan in Section 8):**
structural gaps (the five mths_since_* delinquency fields,
mths_since_recent_inq, mo_sin_old_il_acct, the four bankcard fields)
get their own bin at the WoE stage, no imputation; emp_length gets an
Unknown bin (its gaps are ~99% a subset of emp_title gaps, so no second
flag); median imputation fitted on train only is reserved for the
sub-1% columns. A flag for mths_since_last_record would duplicate
pub_rec == 0 and is not added. num_tl_120dpd_2m is deferred to the
binning stage: it is 0.0% missing through 2013Q3, 0.7% in 2013Q4, then
3-6.5% from 2014Q1, consistent with a data-feed change.

**Drift note:** the emp_length missing share rose from 3.7% (2013Q1)
to 7.3% (2015Q4) with a ~6.8pp higher bad rate. Rough arithmetic
(about 0.25pp) explains only about a tenth of the 2.6pp rise in bad
rate between 2013 and 2015; most of the drift lies elsewhere.
Associations are descriptive, not causal.


---

## 10. Feature engineering

Script: notebooks/12_feature_engineering.py. Outputs:
data/processed/features.parquet (not tracked), reports/feature_dictionary.csv,
reports/categorical_levels.csv, reports/winsor_thresholds.csv.

**Principle:** every data-dependent step (rare-level grouping,
winsorisation caps) is fitted on train rows only and applied unchanged to
validation and out-of-time. No imputation at this stage: NaNs stay for the
binning step.

**Kept / dropped.** Of 100 origination candidates: 30 sparse columns
(>=90% missing) dropped; term, application_type, disbursement_method
dropped as (near-)constant; funded_amnt (equal to loan_amnt for 100.0% of
loans) and funded_amnt_inv (investor-funded share, equal for 92.9%) dropped
as copies of loan_amnt or funding outcomes. zip_code, emp_title, title and
desc are excluded (high-cardinality text; zip code is also a fair-lending
concern). grade, sub_grade and int_rate are stored as bench_* columns only:
they are Lending Club's own risk output and are not predictors in the
primary model. Result: 66 features (61 numeric, 5 categorical).

**Derived features.** fico_mean (midpoint of the application FICO range;
width is 4 for 568,630 loans and 5 for 64); credit_history_months
(earliest_cr_line to issue date; no unparseable or negative values);
emp_length_yrs ('< 1 year' = 0, '10+ years' = 10; 35,181 NaNs, 6.2%, kept
for an Unknown bin); loan_to_income and payment_to_income.

**Data checks.** dti has no negative values; 7 loans exceed 100 although
the train maximum is 39.99 (outside the train range, handled by the
train-fitted cap).

**Winsorisation.** 16 amount/ratio columns capped at the train 0.1th and
99.9th percentiles (thresholds in reports/winsor_thresholds.csv). About
0.1-0.2% of train rows are clipped per column. Several columns clip more
out-of-time (dti 0.35% vs 0.20%, payment_to_income 0.42% vs 0.20%,
annual_inc 0.29% vs 0.20%, total_rev_hi_lim 0.28% vs 0.19%): mild tail
drift, to be quantified by PSI.

**Categoricals.** home_ownership, verification_status, purpose, addr_state,
initial_list_status. Levels below 0.2% of train are grouped to OTHER. The
rule is deliberately risk-blind: an initial 1% threshold pooled purposes
and states with different default rates, so it was lowered and risk-based
grouping is left to the WoE stage.

**Leakage tripwire.** Single-feature AUC on train: maximum 0.597
(fico_mean), none above 0.70; benchmark int_rate 0.654. This catches gross
leakage only; the primary control is the column-level audit in Section 6.

**Still to treat:** structural NaNs get their own bin; revol_util (0.04%)
and pct_tl_nvr_dlq (0.03%) get a train-fitted median. The credit-scale
columns (tot_hi_cred_lim, total_bc_limit, avg_cur_bal, tot_cur_bal,
total_rev_hi_lim, bc_open_to_buy, annual_inc) are expected to be
collinear; variable selection will cluster them.

Categorical result at the 0.2% threshold: purpose 14 -> 12 levels (OTHER =
0.15% of loans: wedding, renewable_energy, educational); addr_state 51 -> 47
levels (OTHER = 0.29%). small_business, car and medical (0.93-0.99% of train
each) are kept as separate levels; the initial 1% rule would have pooled them.

---

## 11. WoE / IV binning

Script: notebooks/13_woe_binning.py; module: src/woe.py (WoEBinner). Outputs:
models/woe_spec.json (cut points and WoE per bin), reports/woe_bins.csv (per-bin counts
and bad rates for train, validation and out-of-time), reports/iv_summary.csv (IV,
stability and review flags per feature).

**Method (everything fitted on train rows only).**
- Numeric: an entropy decision tree proposes up to 10 cut points; every bin holds at
  least max(1,000 loans, 2% of the non-missing rows). Adjacent bins are then merged
  (pool adjacent violators) until the bad rate is monotone in the feature; the direction
  is whichever gives the higher IV.
- Missing values: structural-missing features get a dedicated Missing bin. revol_util,
  pct_tl_nvr_dlq and avg_cur_bal get a train median instead (avg_cur_bal was added after
  the first run: its Missing bin held a handful of loans and only 2 bad).
- Categorical: levels are ordered by train bad rate and grouped by the same tree, so
  groups are contiguous in risk. Levels unseen in train receive a neutral WoE of 0.
- Convention: WoE = ln(%good / %bad), so a higher WoE is a safer bin; IV is the sum of
  (%good - %bad) x WoE. Counts are floored at 0.5 so an empty bin cannot give log(0).
  Applying the saved spec to train reproduces the fit-time IV exactly (checked for every
  feature).
- Validation and out-of-time are used only for testing: IV with the train WoE on each
  split, PSI of bin shares against train, and whether the bin order holds.

**Results (66 features).** IV bands on train (common rule of thumb): strong 0, medium 1,
weak 28, useless 37, suspicious 0.

| Feature | IV train | IV validation | IV out-of-time |
|---|---|---|---|
| fico_mean | 0.139 | 0.139 | 0.157 |
| annual_inc | 0.081 | 0.074 | 0.069 |
| tot_hi_cred_lim | 0.080 | 0.077 | 0.077 |
| bc_open_to_buy | 0.079 | 0.084 | 0.084 |
| avg_cur_bal | 0.076 | 0.074 | 0.072 |
| total_bc_limit | 0.073 | 0.076 | 0.080 |
| acc_open_past_24mths | 0.071 | 0.073 | 0.093 |
| tot_cur_bal | 0.066 | 0.062 | 0.061 |
| payment_to_income | 0.064 | 0.062 | 0.060 |
| total_rev_hi_lim | 0.060 | 0.058 | 0.067 |

- Stability: for all of the top 25, out-of-time IV is 0.85-1.34 times train IV; the
  recent-activity features (acc_open_past_24mths, num_tl_op_past_12m, inq_last_6mths,
  mths_since_recent_inq) gain about 30% IV out-of-time. The monotone constraint costs at
  most 5.1% of IV among the top 25. PSI(out-of-time) is below 0.10 for every feature
  except initial_list_status (0.277); the largest in the top 25 is 0.027.
- Review flags: the first run flagged 6 features for an out-of-time bin-order reversal
  (annual_inc, mo_sin_old_rev_tl_op, percent_bc_gt_75, mo_sin_old_il_acct,
  credit_history_months, num_actv_rev_tl) because any reversal counted as broken. With
  reversals now counted only beyond 2 standard errors of the difference, one remains:
  percent_bc_gt_75 (IV 0.032 train, 0.031 out-of-time, PSI 0.027); it is resolved at variable selection. In the out-of-time set its 75.75-82.1 bin (3,632 loans) shows 19.52% bad against 17.09% for the top bin, a reversal of about 3.5 standard errors; in train the two top bins are indistinguishable (16.36% vs 16.40%), so if the feature is kept they are merged, at negligible IV cost.
- Missing bins behave as hypothesised: mths_since_recent_inq Missing (no recent inquiry)
  9.84% bad vs 17.05% for an inquiry within 1.5 months, continuing the ordering of the
  non-missing bins; emp_length_yrs Missing (no employer information) 20.28% bad, WoE
  -0.46, on 17,707 train loans; num_tl_120dpd_2m Missing 17.01% bad but IV 0.002;
  mths_since_last_record IV 0.005.
- Calibration drift is visible in the bins: the out-of-time bad rate is higher than train
  in every fico_mean bin (about +1.8pp in the two riskiest, +0.1pp in the safest) while
  bin shares barely move (PSI 0.005). Rank ordering holds; the level of default rates
  rose.
- purpose: raw train bad rates run from 11.5% (car) to 22.9% (small_business, 2,987
  loans), with moving 20.4% and house 16.9%. The 2% minimum bin size merges house,
  moving and small_business into one bin of 6,160 loans at 20.94%, which understates
  small_business by 1.9pp and overstates house by 4.0pp (2.0% of train loans). Accepted:
  purpose IV is 0.021 and the group still carries the risk ordering.
- addr_state: 47 levels grouped into 10 risk-ordered bins (9.85% to 17.21% bad), IV 0.016.

**Decisions.**
- Excluded from variable selection: initial_list_status (PSI 0.277 out-of-time vs 0.0 for
  the random validation split, so the shift is time-driven; IV 0.006; it describes how the
  loan was listed, not the borrower), num_tl_120dpd_2m (IV 0.002; missingness tied to a
  2014 data-feed change) and addr_state (IV 0.016 is below the 0.02 line and geography is a
  possible proxy for protected characteristics; see Section 12).
- Features under IV 0.02 are candidates to drop at selection. 0.02 is a soft line:
  emp_length_yrs (0.0197) and purpose (0.0208) go forward because their effect sits in
  small, interpretable groups.
- Next: correlation clustering on the WoE features (the credit-size family tot_hi_cred_lim,
  total_bc_limit, avg_cur_bal, tot_cur_bal, total_rev_hi_lim, bc_open_to_buy, annual_inc is
  expected to collapse), then the first logistic-regression scorecard.

---

## 12. Limitations and governance

Script: notebooks/14_governance_checks.py (console output only). This section records what
the development sample can and cannot support, and the rules for monitoring it.

### 12.1 Population: accepted loans only (no reject inference)

The data contain only loans that Lending Club accepted and funded; rejected applicants have
no repayment outcome. The model therefore describes the accepted population, and its PDs are
valid only for applicants who would have been accepted under the 2013-2015 policy. Applying
it to the full applicant pool would extrapolate beyond the data. Reject inference was not
performed: the rejected-applications file carries only a handful of application fields, so
an extension would be weak and is out of scope. The sample is further limited to 36-month
loans issued Jan 2013 - Jan 2016 (Sections 7 and 8).

### 12.2 Default definition and horizon

Composition of the bad label in the development sample (568,694 loans, 80,202 bad):

| Status | Loans | % of bad | % of all loans |
|---|---|---|---|
| Charged Off | 80,058 | 99.82 | 14.08 |
| Late (31-120 days) | 144 | 0.18 | 0.03 |
| Default | 0 | 0 | 0 |

The label is in effect "charged off". Late (31-120) loans appear only at the end of the
window (0.00% of loans issued in 2013-2014, 0.02% in 2015, 0.37% in Jan 2016, where some
loans were still delinquent at the snapshot), so the part of the label below 90 days past
due is immaterial here.

Gaps to the Basel definition (default = 90+ days past due, or unlikely to pay):
- A charge-off occurs after a longer delinquency than 90 days (the exact timing should be
  verified against Lending Club's servicing policy before a figure is quoted). Loans that
  reached 90+ days past due and later cured are labelled good. Against a 90+ days-past-due
  definition the observed bad rate is therefore probably understated; rank-ordering is
  likely affected less than the level of PD.
- This cannot be re-labelled: the file has no monthly payment history.
- Horizon: the target is default at any point in the full 36-month life (a cumulative PD),
  not the 12-month PD used for Basel or IFRS 9 stage 1. Converting would need default-timing
  data that this snapshot does not contain.
- The 2013-2016 vintages were observed in a benign credit environment: the PDs are
  point-in-time for that window, and are neither through-the-cycle nor downturn estimates.

### 12.3 Loans excluded for an ambiguous status

2,902 mature loans were excluded in Section 5b:

| Status | Loans | Inside development window |
|---|---|---|
| Does not meet the credit policy: Fully Paid | 1,988 | 0 |
| Does not meet the credit policy: Charged Off | 761 | 0 |
| Current | 126 | 120 |
| In Grace Period | 17 | 16 |
| Late (16-30 days) | 10 | 10 |
| Total | 2,902 | 146 |

The 2,749 policy-exception loans (94.7%) were all issued in 2007-2010, outside the 2013+
development window. Only 146 loans (0.03% of the development sample) are ambiguous-status
loans inside it. Their outcome cannot be followed (the file is a snapshot), but the effect is
bounded: the development bad rate of 14.10% would be 14.10% if all 146 were good and 14.12%
if all were bad. On the earlier 696,232-loan sample the bounds were 14.76% to 15.17% (base
14.82%). Excluding them is immaterial.

### 12.4 Fair lending

- Excluded: zip_code (fine geography, Section 10) and addr_state (Section 11: IV 0.016 is
  below the 0.02 line, and state is a possible geographic proxy for protected
  characteristics).
- Retained for testing: emp_length_yrs (IV 0.0197). Employment length can correlate with age
  and employment status (for example, retirees report no current employer), and its Missing
  bin (no employer information, 17,707 train loans, 20.28% bad) is the largest
  missing-group effect in Section 9 (+6.8pp within the same issue year). That link is a
  hypothesis: the data has no age field to test it.
- Plan: when the scorecard is built, report AUC and KS with and without emp_length_yrs; if
  the loss is negligible, drop it.
- Limitation: the dataset has no protected-class attributes (race, sex, age), so a
  disparate-impact test cannot be run; in practice it would use collected or proxy-inferred
  attributes. Other credit-capacity variables (income, credit limits) can also act as
  proxies; this is documented, not tested.

### 12.5 Drift response

Observed: the out-of-time bad rate is 14.73% against 13.80% in train and validation (+0.93pp,
+6.7% relative, about 9 standard errors, so not sampling noise). Feature PSI is below 0.10
for every feature except initial_list_status (excluded), so this is mainly calibration drift,
not a change in applicant mix: in every fico_mean bin the out-of-time bad rate is higher than
in train (about +1.8pp in the two riskiest bins, +0.1pp in the safest) while bin shares
barely move.

Response plan:
1. Assess discrimination (AUC, KS, Gini) and calibration (observed/expected overall and by
   score decile, calibration intercept and slope) separately, on validation and out-of-time.
2. Tolerance (initial, to be justified): overall observed/expected within 0.95-1.05. The
   standard error of observed/expected on the out-of-time set is about 0.6% relative, so
   +-5% is well outside sampling noise. It was set knowing the 0.93pp base-rate gap, before
   the final scorecard's observed/expected has been seen.
3. If outside the band: recalibrate with an intercept-only shift in log-odds (slope kept
   unless the calibration slope departs materially from 1). To keep a clean test, fit the
   shift on Jul-Sep 2015 (73,569 loans) and test on Oct 2015 - Jan 2016 (111,343 loans);
   report both.
4. Monitoring after deployment: feature and score PSI (0.10 = watch, 0.25 = act; common
   conventions), observed/expected and Gini by quarter. Gini-drop and re-development
   triggers are set once the scorecard's out-of-time Gini is known.

### 12.6 Other limitations

- One lender, one product (36-month loans), one country and a three-year origination window;
  no downturn period.
- About 30 bureau fields were only populated for recent loans and cannot be used (Section 7).
- The purpose grouping merges house, moving and small_business into one bin (Section 11).
