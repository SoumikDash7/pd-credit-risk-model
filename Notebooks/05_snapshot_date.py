import pandas as pd

FILE = "data/raw/accepted_2007_to_2018Q4.csv.gz"

chunksize = 200000
max_last_pymnt = None
max_last_credit_pull = None

reader = pd.read_csv(FILE, compression="gzip", low_memory=False,
                      usecols=["last_pymnt_d", "last_credit_pull_d"], chunksize=chunksize)

for i, chunk in enumerate(reader):
    lp = pd.to_datetime(chunk["last_pymnt_d"], format="%b-%Y", errors="coerce").max()
    lc = pd.to_datetime(chunk["last_credit_pull_d"], format="%b-%Y", errors="coerce").max()
    max_last_pymnt = lp if max_last_pymnt is None or (pd.notna(lp) and lp > max_last_pymnt) else max_last_pymnt
    max_last_credit_pull = lc if max_last_credit_pull is None or (pd.notna(lc) and lc > max_last_credit_pull) else max_last_credit_pull
    print(f"Processed chunk {i+1}")

print(f"\nMax last_pymnt_d: {max_last_pymnt}")
print(f"Max last_credit_pull_d: {max_last_credit_pull}")
