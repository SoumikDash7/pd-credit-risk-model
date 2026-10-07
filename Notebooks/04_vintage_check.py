import pandas as pd

FILE = "data/raw/accepted_2007_to_2018Q4.csv.gz"

chunksize = 200000
vintage_counts = None
total_rows = 0

reader = pd.read_csv(FILE, compression="gzip", low_memory=False,
                      usecols=["issue_d", "term", "loan_status"], chunksize=chunksize)

for i, chunk in enumerate(reader):
    total_rows += len(chunk)
    chunk["issue_year"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y").dt.year
    counts = chunk.groupby(["issue_year", "term"]).size()
    vintage_counts = counts if vintage_counts is None else vintage_counts.add(counts, fill_value=0)
    print(f"Processed chunk {i+1}, cumulative rows: {total_rows:,}")

print("\n=== Loans by issue year and term ===")
print(vintage_counts.unstack(fill_value=0))

vintage_counts.unstack(fill_value=0).to_csv("reports/vintage_by_term.csv")
print("\nSaved to reports/vintage_by_term.csv")
