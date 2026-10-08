import pandas as pd

FILE = "data/raw/accepted_2007_to_2018Q4.csv.gz"
cols = list(pd.read_csv(FILE, compression="gzip", nrows=0).columns)

ID_OR_TARGET = ["id", "member_id", "url", "loan_status", "policy_code"]

TIME_REFERENCE = ["issue_d"]  # used to split train/test by vintage, not as a predictor

LEAKAGE_POST_ORIGINATION = [
    # payment history and balances, recorded as the loan performs
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "pymnt_plan",
    # hardship programme, only exists after the borrower struggles
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date",
    "hardship_end_date", "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    # debt settlement, happens after serious delinquency
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage",
    "settlement_term",
]

LENDER_ASSIGNED = ["grade", "sub_grade", "int_rate"]  # LC's own risk assessment, decision deferred

HIGH_CARDINALITY_TEXT = ["emp_title", "title", "desc", "zip_code"]

groups = {
    "id_or_target": ID_OR_TARGET,
    "time_reference": TIME_REFERENCE,
    "leakage_post_origination": LEAKAGE_POST_ORIGINATION,
    "lender_assigned_review": LENDER_ASSIGNED,
    "high_cardinality_text": HIGH_CARDINALITY_TEXT,
}

# catch typos: every listed name must exist in the file
for g, names in groups.items():
    missing = [n for n in names if n not in cols]
    assert not missing, f"{g}: not in file -> {missing}"

assigned = {n: g for g, names in groups.items() for n in names}
rows = [{"column": c, "category": assigned.get(c, "origination_candidate")} for c in cols]
out = pd.DataFrame(rows)

print(out["category"].value_counts())
print("\nOrigination candidates (eyeball these):")
print(out.loc[out["category"] == "origination_candidate", "column"].tolist())

out.to_csv("reports/column_classification.csv", index=False)
print("\nSaved reports/column_classification.csv")
