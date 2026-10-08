import pandas as pd

FILE = "data/raw/accepted_2007_to_2018Q4.csv.gz"

cls = pd.read_csv("reports/column_classification.csv")
candidates = cls.loc[cls["category"] == "origination_candidate", "column"].tolist()

ids = pd.read_csv("data/processed/modeling_sample_ids_target.csv", usecols=["id"], dtype=str)["id"]
id_set = set(ids)
print(f"Modeling sample ids: {len(ids):,}")

usecols = ["id", "issue_d"] + candidates
reader = pd.read_csv(FILE, compression="gzip", usecols=usecols,
                     dtype={"id": str}, low_memory=False, chunksize=200000)

nonnull_by_year = None
rows_by_year = None
matched = 0

for i, chunk in enumerate(reader):
    chunk = chunk[chunk["id"].isin(id_set)].copy()
    if chunk.empty:
        print(f"Chunk {i+1}: no sample rows")
        continue
    matched += len(chunk)
    chunk["issue_year"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y").dt.year
    nn = chunk[candidates].notnull().groupby(chunk["issue_year"]).sum()
    n = chunk.groupby("issue_year").size()
    nonnull_by_year = nn if nonnull_by_year is None else nonnull_by_year.add(nn, fill_value=0)
    rows_by_year = n if rows_by_year is None else rows_by_year.add(n, fill_value=0)
    print(f"Chunk {i+1}: matched so far {matched:,}")

status = "OK" if matched == len(ids) else "MISMATCH - investigate before trusting results"
print(f"\nRow match check: {matched:,} matched vs {len(ids):,} expected -> {status}")

print("\nSample rows by issue year:")
print(rows_by_year.astype(int))

total_rows = rows_by_year.sum()
overall_missing = ((1 - nonnull_by_year.sum() / total_rows) * 100).sort_values(ascending=False)

print("\nCandidate columns by in-sample missingness:")
print(f">=99.9% missing : {(overall_missing >= 99.9).sum()}")
print(f"50-99.9%        : {((overall_missing >= 50) & (overall_missing < 99.9)).sum()}")
print(f"10-50%          : {((overall_missing >= 10) & (overall_missing < 50)).sum()}")
print(f"1-10%           : {((overall_missing >= 1) & (overall_missing < 10)).sum()}")
print(f"<1%             : {(overall_missing < 1).sum()}")

print("\nColumns >=10% missing in sample:")
print(overall_missing[overall_missing >= 10].round(1).to_string())

miss_by_year = ((1 - nonnull_by_year.div(rows_by_year, axis=0)) * 100).round(2).T
miss_by_year.to_csv("reports/sample_missingness_by_year.csv")
overall_missing.round(2).to_csv("reports/sample_missingness.csv", header=["pct_missing"])
print("\nSaved reports/sample_missingness.csv and reports/sample_missingness_by_year.csv")
