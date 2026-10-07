import pandas as pd
import gzip

FILE = "data/raw/accepted_2007_to_2018Q4.csv.gz"

# Peek at columns and dtypes from a small sample
df_sample = pd.read_csv(FILE, compression="gzip", low_memory=False, nrows=5)
print("Total columns:", df_sample.shape[1])
print()
print("Column names:")
print(list(df_sample.columns))
print()

# Full row count without loading everything into memory
with gzip.open(FILE, "rt", encoding="utf-8", errors="replace") as f:
    row_count = sum(1 for _ in f) - 1  # minus header
print(f"Total rows: {row_count:,}")