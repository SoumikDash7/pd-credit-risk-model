# Validation Report: PD Scorecard for 36-Month Unsecured Loans (Version 1)

| Item | Detail |
|---|---|
| Model | Logistic-regression probability-of-default (PD) scorecard on 16 WoE-binned features, with a points scale |
| Version | 1, frozen |
| Prepared by | Soumik (the model developer) |
| Date | 2026-10-15; repository state 2032190 |
| Nature of this report | **A developer's self-assessment.** It follows the structure of a validation report but is not independent. A real deployment would need review by a separate validation team |
| Detail | The decision log (docs/methodology.md) holds every rule, expectation and result; section numbers below refer to it |

## 1. Summary and overall assessment

The scorecard ranks borrowers stably across train, validation and out-of-time samples, but its calibration drifts with the loan vintage. Observed defaults run below the model's predictions for 2013 loans and above them from 2015. An intercept shift corrects the average for only a few months, and several segments stay under-predicted. Discrimination is modest, which is what application-time data gives once the lender's own grade and rate are excluded.

| Area | Assessment | Basis |
|---|---|---|
| Conceptual soundness | Satisfactory | All 16 coefficients are negative and stay negative in all 200 bootstrap resamples; no loan's PD falls under any adverse shock; no leakage found (highest single-feature AUC 0.597) |
| Data and sample | Satisfactory with limitations | Maturity-filtered, time-split sample; accepted loans only; the label is almost entirely charge-offs |
| Discrimination | Satisfactory (modest) | AUC 0.671 / 0.671 / 0.679 on train / validation / out-of-time; the riskiest decile carries 2.03 times the average bad rate on validation |
| Calibration | **Needs attention** | Observed/expected 1.001 on validation but 1.086 out-of-time before recalibration; 1.044 on the Oct 2015 to Jan 2016 test window after an intercept shift; 5 of 7 segment misfits persist |
| Stability | Satisfactory | Score PSI at most 0.024; feature PSI at most 0.021 out-of-time (final 16 features) |
| Benchmarking | Satisfactory | The best challenger is +0.0055 AUC ahead out-of-time; a random forest is not better |
| Implementation | Satisfactory | The points card re-scores all 261,669 held-out loans exactly without the modelling code |

**Overall (developer's view).** Version 1 is acceptable for its stated illustrative purpose, ranking 36-month loans and estimating their whole-life default probability for the 2013-2016 accepted population, provided that the calibration findings (F1 and F2 in Section 6) are managed through the monitoring and re-fit rules of Section 7. It is not ready for production use without independent review and the items in Section 8.

## 2. Purpose, scope and intended use

- **Purpose.** Rank unsecured consumer loans by default risk at application and estimate the probability that a loan is charged off over its 36-month life.
- **In scope.** 36-month loans of the kind originated by Lending Club between 2013 and early 2016, scored with information available at origination.
- **Out of scope.** 60-month loans; applicants who were rejected; a 12-month (regulatory-style) PD; loss given default and exposure at default; pricing; any period after early 2016.

## 3. Data and sample

| Item | Detail |
|---|---|
| Source | Lending Club accepted loans, 2007-2018 Q4 (2,260,701 loans, 151 columns), Kaggle mirror |
| Target | Bad = Charged Off, Late (31-120 days) or Default (99.8% of bad loans are charge-offs); good = Fully Paid; current, grace-period, 16-30 days late, policy-exception and blank statuses excluded |
| Maturity filter | A loan is kept only if issue date plus term falls on or before 2019-01-01, so every loan had time to reach an outcome |
| Development sample | 36-month loans issued Jan 2013 to Jan 2016: 568,694 loans, 14.10% bad |
| Train | 307,025 loans, 13.80% bad (Jan 2013 to Jun 2015, random 80%) |
| Validation | 76,757 loans, 13.80% bad (same window, random 20%) |
| Out-of-time | 184,912 loans, 14.73% bad (Jul 2015 to Jan 2016) |

- 2,749 of the 2,902 excluded loans are policy-exception loans from 2007-2010. Only 146 ambiguous loans fall in the development window, and they cannot move the bad rate beyond 14.10% to 14.12% (Section 12.3).
- Missingness is structural: about 30 bureau fields exist only for later loans (the window starts in 2013 for that reason), and the "months since" fields are missing when the event never happened. Structural gaps get their own bin and are not imputed (Sections 7 to 9).

## 4. Methodology

1. **Leakage audit.** All 151 columns were classified; 38 post-origination columns are excluded; the lender's grade, sub-grade and interest rate are benchmark-only (Section 6).
2. **Features.** 100 origination candidates became 66 engineered features, 19 after variable selection and 16 in the final model. Payment-to-income was removed because instalment divided by loan amount tracks the interest rate (correlation 0.9989); employment length was dropped for fair-lending reasons; revol_bal and mort_acc were dropped by a pre-set leave-one-out rule (Sections 10, 13 and 15).
3. **Binning.** Entropy-tree bins (each at least 2% of non-missing loans and 1,000 loans) merged until the bad rate is monotone, with a separate Missing bin; WoE = ln(good/bad) (Section 11).
4. **Model.** Unregularised logistic regression on the WoE features, fitted on train (Section 14).
5. **Calibration.** An intercept shift of 0.0689 log-odds was fitted on Jul-Sep 2015 and tested on Oct 2015 to Jan 2016 against a pre-set band of 0.95 to 1.05 (Section 15).
6. **Points.** 600 points at 50:1 good-to-bad odds, 20 points to double the odds, one-decimal points (Section 21).

Rules and expectations were written before each result and are recorded with their outcomes, including the ones that missed.

## 5. Results

### 5.1 Conceptual soundness

- All 16 coefficients are negative, as expected for WoE = ln(good/bad) and bad = 1; the largest p-value is 0.00018.
- Coefficient stability over 200 bootstrap refits (WoE bins held fixed): every coefficient is negative in all 200 resamples; the largest relative standard deviation is 26% (bc_util).
- Under one-standard-deviation adverse shocks to each numeric feature, no loan's PD falls (Section 20.1).
- The leakage tripwire (single-feature AUC) found nothing above 0.60 against a 0.70 alarm line (Section 10).

### 5.2 Discrimination

| Sample | Loans | Bad rate | AUC | Gini | KS |
|---|---|---|---|---|---|
| Train | 307,025 | 13.80% | 0.6707 | 0.3415 | 0.2478 |
| Validation | 76,757 | 13.80% | 0.6708 | 0.3416 | 0.2464 |
| Out-of-time | 184,912 | 14.73% | 0.6788 | 0.3576 | 0.2558 |

- The observed bad rate rises at every score decile on validation and out-of-time; the riskiest decile carries 2.03 times (validation) and 2.11 times (out-of-time) the average bad rate.
- Policy illustration, rejecting the riskiest 20% of validation applicants: 36.0% of bad loans are caught, precision is 24.8%, and the bad rate among approved loans falls from 13.8% to 11.0%. At the same PD threshold out-of-time the rejection share is 19.5%, recall 36.0% and the approved bad rate 11.7% against 14.7% overall (Section 19).
- The lender's own interest rate, which is not a model input, scores AUC 0.656 on validation and 0.683 out-of-time; the scorecard is 0.014 ahead on validation and 0.004 behind out-of-time.

### 5.3 Calibration and drift

| Test | Observed/expected | Verdict against the 0.95 to 1.05 band |
|---|---|---|
| Validation | 1.001 | Within |
| Out-of-time, before recalibration | 1.086 | Outside |
| Out-of-time test window (Oct 2015 to Jan 2016), before recalibration | 1.105 | Outside |
| Same window, after the intercept shift fitted on Jul-Sep 2015 | 1.044 | Within, close to the upper edge |

- **Vintage trend.** Before any shift, observed/expected runs from 0.91 (2013Q4) to 1.13 (2016Q1). The score distribution barely moves (score PSI at most 0.024), so this is calibration drift and not a change in applicant mix. After the shift, 2015Q4 reads 1.039 (inside the band) and 2016Q1 reads 1.066 (outside) (Sections 15 and 19).
- **Shelf life of a shift.** Refitting the shift on shorter or longer windows leaves December 2015 and January 2016 above 1.05 in every case; a fitted shift stays in band for about three months (Section 20.5).
- **Segments.** Seven segments flagged on validation and replicated on train (the top income decile, the lowest bankcard open-to-buy bin, the lowest and highest bankcard utilisation bins, and three loan purposes) are under-predicted by 11% to 30% on validation. 5 of the 7 persist out-of-time by a pre-set rule, and all remain under-predicted after the shift (O/E 1.08 to 1.29 in the test window). 1 of 2 control segments also persisted, so the persistence rule is lenient (Sections 18 and 19).

### 5.4 Stability

- Score PSI against train by issue quarter: at most 0.024 (0.10 = watch, 0.25 = act).
- Feature PSI for the 16 final features: at most 0.000 on validation (mths_since_recent_inq) and 0.021 out-of-time (bc_util).
- AUC by issue quarter on held-out loans: 0.655 to 0.682, with no deterioration over time.

### 5.5 Challenger models

| Challenger | Delta AUC, validation | Delta AUC, out-of-time |
|---|---|---|
| XGBoost with monotone constraints | +0.0078 | +0.0055 |
| Random forest | -0.0007 | -0.0009 |

XGBoost without constraints is +0.0112 AUC ahead out-of-time, but its extra signal sits in shapes with no economic rationale (for example a rise in risk in the top income decile), so it is not used. Capacity-ratio candidates did not close the gap (Sections 16 to 18).

### 5.6 Sensitivity and stress

- Four adverse shocks together (FICO -20 points, DTI +5 points, income -10%, bankcard utilisation +10 points) raise mean PD by 3.06 percentage points and the share rejected at a fixed threshold from 20.0% to 33.4% (Section 20.2). No policy response or macro scenario is modelled.
- The deployment drift of 0.1008 log-odds raises mean PD by 1.18 percentage points, close to the effect of every borrower losing 20 FICO points (1.20).

### 5.7 Implementation checks

- The exact score maps back to the model PD to rounding precision; integer points were rejected (+0.31 percentage points of bias in mean PD), so one-decimal points are used.
- The points card (models/scorecard_points.json), read back from disk, re-scores all 261,669 validation and out-of-time loans exactly without the WoE module.
- Odds double every 19.5 points on validation against a design value of 20.

## 6. Findings and limitations

Severity: **High** limits use unless remediated; **Medium** should be remediated or monitored; **Low** is for documentation.

| # | Finding | Severity | Recommended action | Log |
|---|---|---|---|---|
| F1 | Calibration drifts with the vintage; the intercept shift corrects the average for only about three months and 2016Q1 falls outside the band | High | Monitor observed/expected monthly and refit the shift on a rolling basis (Section 7); the deployment shift of 0.1008 is untested | 14, 15, 19, 20 |
| F2 | Segments under-predicted in-time and out-of-time: top income decile, lowest bankcard open-to-buy, extremes of bankcard utilisation, three loan purposes | Medium | Segment-level monitoring and overlays, or finer purpose bins and a non-monotone treatment once an economic rationale exists (remediation options in Section 18.4) | 18, 19 |
| F3 | The label is mostly charge-offs (later than 90 days past due) over the whole 36-month life, not a 12-month PD; the observed bad rate is probably understated against a Basel-style definition | Medium | Do not use the output as a regulatory 12-month PD; re-label from payment history if it can be obtained | 12.2 |
| F4 | Accepted loans only; no reject inference | Medium | Valid for applicants resembling the accepted population; extend with rejected-applicant data before any wider use | 12.1 |
| F5 | The data hold no protected-class attributes, so disparate impact could not be tested; zip code, state and employment length were removed as precautions | Medium | Obtain collected or proxy-inferred attributes and test before any production use | 12.4 |
| F6 | Bin artefacts: round-number income bins, pooled loan purposes, home ownership OTHER grouped with MORTGAGE | Low | Review bin boundaries at the next re-development | 11, 21.3 |
| F7 | The out-of-time set supplied the recalibration shift, so it is not untouched data | Low | Reserve a later period as the next test set | 15.4 |
| F8 | Modest discrimination and a benign 2013-2016 credit period; no downturn observed | Low | Treat PDs as point-in-time; run macro-scenario stress before any reliance in a downturn | 12.6 |

## 7. Recommended monitoring and re-fit triggers

Monitoring thresholds in the first and third rows follow the conventions set in Section 12.5. The other thresholds are the developer's proposals and need approval and calibration before use.

| Metric | Frequency | Watch | Action |
|---|---|---|---|
| Observed/expected, whole portfolio | Monthly | Outside 0.95 to 1.05 | Refit the intercept shift on the latest window if outside for two consecutive months (proposal); open a re-development review if outside 0.90 to 1.10 (proposal) |
| Observed/expected, flagged segments (Section 6, F2) | Quarterly | Relative to the portfolio above 1.15 (proposal) | Review a segment overlay |
| Score PSI against the development sample | Monthly | 0.10 | 0.25: investigate and consider re-development |
| AUC, rolling quarter with at least 20,000 loans | Quarterly | Below 0.65 (proposal) | Below 0.64: re-development review (proposal). Held-out quarters ranged from 0.655 to 0.682, the lowest on only 6,720 loans |
| Feature PSI, 16 features | Quarterly | 0.10 | 0.25 |

## 8. Not covered

- An independent validation by a team other than the developer.
- Reject inference, a 12-month PD, loss given default and exposure at default.
- A fair-lending test on protected-class data.
- Macroeconomic stress scenarios and any period after early 2016.
- Production implementation, data-pipeline controls and governance approvals.
