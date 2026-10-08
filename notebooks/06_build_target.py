import pandas as pd

FILE = "data/raw/accepted_2007_to_2018Q4.csv.gz"
SNAPSHOT = pd.Timestamp("2019-01-01")  # buffered snapshot date

# --- Target mapping ---
GOOD_STATUSES = ["Fully Paid"]
BAD_STATUSES = ["Charged Off", "Late (31-120 days)", "Default"]
# everything else (Current, In Grace Period, Late 16-30, policy-exception statuses, NaN) is excluded

chunksize = 200000
kept_chunks = []
total_rows = 0
excluded_immature = 0
excluded_status = 0

cols = ["id", "issue_d", "term", "loan_status"]  # minimal cols for this filtering pass

reader = pd.read_csv(FILE, compression="gzip", low_memory=False, usecols=cols, chunksize=chunksize)

for i, chunk in enumerate(reader):
    total_rows += len(chunk)

    chunk["issue_d_parsed"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
    chunk["term_months"] = chunk["term"].str.extract(r"(\d+)").astype(float)
    chunk["maturity_date"] = chunk["issue_d_parsed"] + pd.to_timedelta(chunk["term_months"] * 30.44, unit="D")

    mature_mask = chunk["maturity_date"] <= SNAPSHOT
    excluded_immature += (~mature_mask).sum()

    chunk_mature = chunk[mature_mask].copy()

    status_mask = chunk_mature["loan_status"].isin(GOOD_STATUSES + BAD_STATUSES)
    excluded_status += (~status_mask).sum()

    chunk_final = chunk_mature[status_mask].copy()
    chunk_final["target"] = chunk_final["loan_status"].isin(BAD_STATUSES).astype(int)

    kept_chunks.append(chunk_final[["id", "issue_d", "term", "loan_status", "target"]])
    print(f"Processed chunk {i+1}, cumulative rows: {total_rows:,}, kept so far: {sum(len(c) for c in kept_chunks):,}")

result = pd.concat(kept_chunks, ignore_index=True)

print(f"\nTotal rows processed: {total_rows:,}")
print(f"Excluded (immature vintage): {excluded_immature:,}")
print(f"Excluded (ambiguous/censored status, among mature): {excluded_status:,}")
print(f"Final modeling sample: {len(result):,}")
print(f"\nBad rate: {result['target'].mean():.4f} ({result['target'].sum():,} bad / {len(result):,} total)")

result.to_csv("data/processed/modeling_sample_ids_target.csv", index=False)
print("\nSaved id+target mapping to data/processed/modeling_sample_ids_target.csv")
