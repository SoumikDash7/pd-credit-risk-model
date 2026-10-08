# Probability of Default (PD) Credit Risk Model

End-to-end retail credit risk PD model built from raw Lending Club
loan data (2007–2018) — covering target definition, sample window
design, missing data treatment, feature engineering, model
development, and independent validation, following the standard
PD model development lifecycle used in bank model risk/validation
functions.

**Full reasoning behind every modeling decision is documented in
[`docs/methodology.md`](docs/methodology.md)** — this project is
built to be read, not just run.

## Why this dataset

Rather than using a pre-cleaned competition dataset, this project
uses Lending Club's raw 151-column loan-level data specifically to
practice the full preprocessing lifecycle: messy types, genuine
missingness, and a target variable that has to be constructed
from `loan_status` rather than handed to you as a clean label.

## Status

🚧 Work in progress — currently at: target variable + sample window
construction. See [`docs/methodology.md`](docs/methodology.md) for
the live decision log.

| Stage | Status |
|---|---|
| Dataset sourcing & raw inspection | ✅ Done |
| Missingness analysis | ✅ Done |
| Target variable definition | ✅ Done |
| Survivorship-bias check & sample window | ✅ Done |
| Feature engineering & leakage audit | 🔲 Next |
| Missing value treatment | 🔲 Planned |
| WoE/IV binning | 🔲 Planned |
| Model development (Logistic Regression, RF, XGBoost) | 🔲 Planned |
| Independent validation (AUC, KS, Gini, PSI, calibration) | 🔲 Planned |
| SHAP interpretability | 🔲 Planned |
| Governance-style validation report | 🔲 Planned |

## Key decisions so far

- **Sample window**: restricted to loans whose maturity date falls
  before the data's effective snapshot (~Jan 2019), to avoid
  survivorship bias from including recently-issued loans that
  haven't had time to resolve. Full reasoning in methodology doc.
- **Target**: `Fully Paid` = good; `Charged Off` / `Late (31-120
  days)` / `Default` = bad; ambiguous/censored/policy-exception
  statuses excluded.
- **Result**: 696,232-loan modeling sample, 14.82% bad rate.

## Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · matplotlib/seaborn

## Project structure
├── data/
│ ├── raw/ # not tracked — see data/raw/README.md to reproduce
│ └── processed/ # derived datasets (target labels, feature sets)
├── notebooks/ # exploration & pipeline scripts, numbered in build order
├── reports/ # generated analysis outputs (missingness map, etc.)
├── docs/
│ └── methodology.md # full decision log with reasoning
└── requirements.txt

## Reproducing this project

```bash
git clone https://github.com/SoumikDash7/pd-credit-risk-model.git
cd pd-credit-risk-model
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash; use venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

Then download the raw dataset per `data/raw/README.md` and run the
notebooks in order.
