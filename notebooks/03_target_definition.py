import pandas as pd

FILE = "data/raw/accepted_2007_to_2018Q4.csv.gz"

chunksize = 200000
status_counts = None
total_rows = 0

reader = pd.read_csv(FILE, compression="gzip", low_memory=False,
                      usecols=["loan_status"], chunksize=chunksize)

for i, chunk in enumerate(reader):
    total_rows += len(chunk)
    counts = chunk["loan_status"].value_counts(dropna=False)
    status_counts = counts if status_counts is None else status_counts.add(counts, fill_value=0)
    print(f"Processed chunk {i+1}, cumulative rows: {total_rows:,}")

print(f"\nTotal rows: {total_rows:,}")
print("\n=== loan_status value counts ===")
print(status_counts.sort_values(ascending=False))

print("\n=== loan_status as % of total ===")
print((status_counts.sort_values(ascending=False) / total_rows * 100).round(2))

status_counts.sort_values(ascending=False).to_csv("reports/loan_status_counts.csv")
print("\nSaved to reports/loan_status_counts.csv")
