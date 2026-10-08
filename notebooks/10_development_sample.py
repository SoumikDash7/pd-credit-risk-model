import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)

FILE = "data/raw/accepted_2007_to_2018Q4.csv.gz"

# ---------- 1. Development sample ----------
base = pd.read_csv("data/processed/modeling_sample_ids_target.csv", dtype={"id": str})
base["issue_dt"] = pd.to_datetime(base["issue_d"], format="%b-%Y")
dev = base[base["term"].str.contains("36") & (base["issue_dt"] >= "2013-01-01")].copy()

print(f"Modeling sample (before scoping): {len(base):,}")
print(f"Development sample (36m, issued >= Jan 2013): {len(dev):,}")
print(f"Bad rate: {dev['target'].mean()*100:.2f}%  ({dev['target'].sum():,} bad)")

dev["issue_year"] = dev["issue_dt"].dt.year
print("\nBy issue year:")
print(dev.groupby("issue_year")["target"].agg(n="size", bad_rate="mean")
      .assign(bad_rate=lambda d: (d["bad_rate"] * 100).round(2)).to_string())

monthly = dev.groupby(dev["issue_dt"].dt.to_period("M"))["target"].agg(n="size", bad_rate="mean")
monthly["cum_pct"] = (monthly["n"].cumsum() / monthly["n"].sum() * 100).round(1)
monthly["bad_rate"] = (monthly["bad_rate"] * 100).round(2)
print("\nBy issue month (to choose the train / out-of-time split):")
print(monthly.to_string())

dev[["id", "issue_d", "term", "loan_status", "target"]].to_csv(
    "data/processed/development_sample_ids_target.csv", index=False)
monthly.to_csv("reports/dev_sample_by_month.csv")
id_set = set(dev["id"])

# ---------- 2 & 3. Pull candidate columns for the development sample ----------
cls = pd.read_csv("reports/column_classification.csv")
candidates = cls.loc[cls["category"] == "origination_candidate", "column"].tolist()
usecols = ["id", "emp_title"] + candidates

nonnull = None
matched = 0
frames = []
cat_counts = {c: None for c in ["application_type", "disbursement_method", "initial_list_status"]}

CHECK_COLS = ["num_il_tl", "mo_sin_old_il_acct", "num_bc_tl", "bc_util", "percent_bc_gt_75",
              "bc_open_to_buy", "mths_since_recent_bc", "inq_last_6mths", "mths_since_recent_inq",
              "pub_rec", "mths_since_last_record", "delinq_2yrs", "mths_since_last_delinq",
              "num_accts_ever_120_pd", "mths_since_last_major_derog", "emp_length"]

reader = pd.read_csv(FILE, compression="gzip", usecols=usecols,
                     dtype={"id": str}, low_memory=False, chunksize=200000)
for i, ch in enumerate(reader):
    ch = ch[ch["id"].isin(id_set)]
    if ch.empty:
        continue
    matched += len(ch)
    nn = ch[candidates].notnull().sum()
    nonnull = nn if nonnull is None else nonnull.add(nn)
    for c in cat_counts:
        vc = ch[c].value_counts(dropna=False)
        cat_counts[c] = vc if cat_counts[c] is None else cat_counts[c].add(vc, fill_value=0)
    sub = ch[CHECK_COLS].copy()
    sub["emp_title_present"] = ch["emp_title"].notnull()
    frames.append(sub)
    print(f"Chunk {i+1}: matched so far {matched:,}")

status = "OK" if matched == len(dev) else "MISMATCH - investigate"
print(f"\nRow match check: {matched:,} vs {len(dev):,} -> {status}")

miss = ((1 - nonnull / matched) * 100).sort_values(ascending=False)
miss.round(2).to_csv("reports/dev_sample_missingness.csv", header=["pct_missing"])
print("\nCandidate columns by missingness in development sample:")
print(f">=90%  : {(miss >= 90).sum()}")
print(f"10-90% : {((miss >= 10) & (miss < 90)).sum()}")
print(f"1-10%  : {((miss >= 1) & (miss < 10)).sum()}")
print(f"<1%    : {(miss < 1).sum()}")
print("\nColumns >=1% missing:")
print(miss[miss >= 1].round(1).to_string())

print("\nLow-cardinality fields (near-constant ones are dropped):")
for c, vc in cat_counts.items():
    print(f"\n{c}:")
    print((vc / vc.sum() * 100).round(2).to_string())

# ---------- "missing means does-not-apply" checks ----------
d = pd.concat(frames, ignore_index=True)
checks = [
    ("mo_sin_old_il_acct", "num_il_tl"),
    ("mths_since_recent_bc", "num_bc_tl"),
    ("bc_util", "num_bc_tl"),
    ("percent_bc_gt_75", "num_bc_tl"),
    ("bc_open_to_buy", "num_bc_tl"),
    ("mths_since_recent_inq", "inq_last_6mths"),
    ("mths_since_last_record", "pub_rec"),
    ("mths_since_last_delinq", "delinq_2yrs"),
    ("mths_since_last_major_derog", "num_accts_ever_120_pd"),
]
rows = []
for col, cond in checks:
    valid = d[cond].notnull()
    zero = valid & (d[cond] == 0)
    pos = valid & (d[cond] > 0)
    rows.append({
        "column": col, "condition_col": cond,
        "n_zero": int(zero.sum()), "missing_pct_if_zero": round(d.loc[zero, col].isnull().mean() * 100, 1),
        "n_positive": int(pos.sum()), "missing_pct_if_positive": round(d.loc[pos, col].isnull().mean() * 100, 1),
    })
print("\n=== Does 'missing' mean 'does not apply'? ===")
print("(If so: ~100% missing when the condition column is 0, ~0% when positive)")
print(pd.DataFrame(rows).to_string(index=False))

ep = d["emp_title_present"]
print("\n=== emp_length missing vs emp_title ===")
print(f"emp_length missing when emp_title is missing : {d.loc[~ep, 'emp_length'].isnull().mean()*100:.1f}%  (n={int((~ep).sum()):,})")
print(f"emp_length missing when emp_title is present : {d.loc[ep, 'emp_length'].isnull().mean()*100:.1f}%  (n={int(ep.sum()):,})")
