import pandas as pd

FILE = "data/raw/accepted_2007_to_2018Q4.csv.gz"

# --- Step 1: chunked missingness count (doesn't require full df in memory) ---
chunksize = 200000
null_counts = None
total_rows = 0

reader = pd.read_csv(FILE, compression="gzip", low_memory=False, chunksize=chunksize)

for i, chunk in enumerate(reader):
    total_rows += len(chunk)
    chunk_nulls = chunk.isnull().sum()
    null_counts = chunk_nulls if null_counts is None else null_counts.add(chunk_nulls, fill_value=0)
    print(f"Processed chunk {i+1}, cumulative rows: {total_rows:,}")

missing_pct = (null_counts / total_rows * 100).sort_values(ascending=False)
missing_summary = missing_pct[missing_pct > 0]

print(f"\nColumns with ANY missing values: {len(missing_summary)} / {len(null_counts)}")
print("\nTop 30 most-missing columns:")
print(missing_summary.head(30).round(2))

missing_summary.to_csv("reports/missingness_map.csv")
print("\nSaved full missingness map to reports/missingness_map.csv")
