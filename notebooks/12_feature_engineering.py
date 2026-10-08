import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)

FILE = "data/raw/accepted_2007_to_2018Q4.csv.gz"

# ---------- inputs from earlier steps ----------
dev = pd.read_csv("data/processed/development_sample_ids_target.csv", dtype={"id": str})
split = pd.read_csv("data/processed/development_split.csv", dtype={"id": str})
dev = dev.merge(split, on="id", how="left", validate="one_to_one")
assert dev["split"].notnull().all()
id_set = set(dev["id"])

cls = pd.read_csv("reports/column_classification.csv")
cand = cls.loc[cls["category"] == "origination_candidate", "column"].tolist()
leak = set(cls.loc[cls["category"] == "leakage_post_origination", "column"])
ids_target = set(cls.loc[cls["category"] == "id_or_target", "column"]) - {"id"}
miss_dev = pd.read_csv("reports/dev_sample_missingness.csv", index_col=0)["pct_missing"]

DROP_SPARSE = miss_dev[miss_dev >= 90].index.tolist()
DROP_CONSTANT = ["term", "application_type", "disbursement_method"]
REPLACED = ["funded_amnt", "funded_amnt_inv", "fico_range_low", "fico_range_high", "earliest_cr_line"]
CATEGORICAL = ["home_ownership", "verification_status", "purpose", "addr_state", "initial_list_status"]
BENCH = ["int_rate", "grade", "sub_grade"]

keep_raw = [c for c in cand if c not in DROP_SPARSE + DROP_CONSTANT + REPLACED]
usecols = ["id", "issue_d"] + keep_raw + REPLACED + BENCH
src = set(usecols) - {"id", "issue_d"}
assert not (src & leak), f"leakage columns in use: {src & leak}"
assert not (src & ids_target), f"id/target columns in use: {src & ids_target}"
print(f"Candidates: {len(cand)} | dropped sparse: {len(DROP_SPARSE)} | dropped constant: {len(DROP_CONSTANT)} "
      f"| replaced by derived: {len(REPLACED)} | kept as-is: {len(keep_raw)}")
print("Leakage / id assertions passed")

# ---------- pull raw columns for the development sample ----------
frames, matched = [], 0
reader = pd.read_csv(FILE, compression="gzip", usecols=usecols, dtype={"id": str},
                     low_memory=False, chunksize=200000)
for i, ch in enumerate(reader):
    ch = ch[ch["id"].isin(id_set)]
    if ch.empty:
        continue
    matched += len(ch)
    frames.append(ch)
    print(f"Chunk {i+1}: matched so far {matched:,}")
assert matched == len(dev), f"row mismatch: {matched:,} vs {len(dev):,}"
raw = pd.concat(frames, ignore_index=True)
d = raw.merge(dev[["id", "split", "target"]], on="id", how="left", validate="one_to_one")
assert len(d) == len(dev) and d["split"].notnull().all()
print(f"\nRow match check: {matched:,} vs {len(dev):,} -> OK")

issue_dt = pd.to_datetime(d["issue_d"], format="%b-%Y")
tr = d["split"] == "train"

def to_num(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    s = s.astype("string").str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce").astype("float64")

F = pd.DataFrame(index=d.index)
dictionary = {}
def reg(name, source, kind, note):
    dictionary[name] = {"feature": name, "source": source, "kind": kind, "treatment": note}

# ---------- numeric pass-through ----------
NUMERIC_RAW = [c for c in keep_raw if c not in CATEGORICAL + ["emp_length"]]
for c in NUMERIC_RAW:
    before = int(d[c].isnull().sum())
    F[c] = to_num(d[c])
    after = int(F[c].isnull().sum())
    if after != before:
        print(f"WARNING {c}: {after - before:,} non-numeric values coerced to NaN")
    reg(c, c, "numeric", "as-is")

# ---------- invalid values ----------
neg_dti, big_dti = int((F["dti"] < 0).sum()), int((F["dti"] > 100).sum())
print(f"\ndti: {neg_dti:,} negative values (set to NaN), {big_dti:,} values above 100")
F.loc[F["dti"] < 0, "dti"] = np.nan
dictionary["dti"]["treatment"] = "negative values set to NaN"

# ---------- derived features ----------
fa = to_num(d["funded_amnt"])
fai = to_num(d["funded_amnt_inv"])
print(f"funded_amnt == loan_amnt for {(fa == F['loan_amnt']).mean()*100:.1f}% of loans; "
      f"funded_amnt_inv == loan_amnt for {(fai == F['loan_amnt']).mean()*100:.1f}%")

lo, hi = to_num(d["fico_range_low"]), to_num(d["fico_range_high"])
print("fico_range_high - fico_range_low:", (hi - lo).value_counts(dropna=False).to_dict())
F["fico_mean"] = (lo + hi) / 2
reg("fico_mean", "fico_range_low, fico_range_high", "numeric", "midpoint of the application FICO range")

ecl = pd.to_datetime(d["earliest_cr_line"], format="%b-%Y", errors="coerce")
hist = (issue_dt.dt.year - ecl.dt.year) * 12 + (issue_dt.dt.month - ecl.dt.month)
n_bad_hist = int((hist < 0).sum())
print(f"credit_history_months: {int(ecl.isnull().sum()):,} unparseable dates, {n_bad_hist:,} negative values (set to NaN)")
F["credit_history_months"] = hist.where(hist >= 0).astype("float64")
reg("credit_history_months", "earliest_cr_line, issue_d", "numeric", "months from earliest credit line to issue date")

el = d["emp_length"].astype("string")
print("\nRaw emp_length values:")
print(el.value_counts(dropna=False).to_string())
yrs = pd.to_numeric(el.str.extract(r"(\d+)")[0], errors="coerce").astype("float64")
yrs = yrs.mask(el.str.contains("<", regex=False, na=False), 0.0)
F["emp_length_yrs"] = yrs
reg("emp_length_yrs", "emp_length", "numeric", "parsed to years ('< 1 year' = 0, '10+ years' = 10); NaN kept for an Unknown bin")
print("Parsed emp_length_yrs:")
print(F["emp_length_yrs"].value_counts(dropna=False).sort_index().to_string())

inc = F["annual_inc"].where(F["annual_inc"] > 0)
F["loan_to_income"] = F["loan_amnt"] / inc
F["payment_to_income"] = F["installment"] / (inc / 12)
reg("loan_to_income", "loan_amnt, annual_inc", "numeric", "loan amount / annual income (income <= 0 gives NaN)")
reg("payment_to_income", "installment, annual_inc", "numeric", "monthly instalment / monthly income (income <= 0 gives NaN)")

# ---------- categoricals (levels fitted on train only) ----------
cat_rows = []
def clean_cat(col, min_share=0.002):
    s = d[col].astype("string").str.strip().fillna("MISSING")
    share = s[tr].value_counts(normalize=True)
    keep = set(share[share >= min_share].index)
    for lvl, sh in share.items():
        cat_rows.append({"feature": col, "level": lvl, "train_share_pct": round(sh * 100, 2), "kept": lvl in keep})
    return s.where(s.isin(keep), "OTHER")

print("\n=== Categorical features ===")
for c in CATEGORICAL:
    n_raw = d[c].nunique(dropna=True)
    F[c] = clean_cat(c)
    reg(c, c, "categorical", "levels under 0.2% of train grouped to OTHER")
    print(f"{c}: {n_raw} raw levels -> {F[c].nunique()} levels, OTHER share {(F[c] == 'OTHER').mean()*100:.2f}%")
print("\nhome_ownership / verification_status / purpose levels:")
for c in ["home_ownership", "verification_status", "purpose"]:
    print(F[c].value_counts(normalize=True).mul(100).round(2).to_string())
    print()
pd.DataFrame(cat_rows).to_csv("reports/categorical_levels.csv", index=False)

# ---------- winsorise heavy-tailed amount / ratio columns (thresholds from train) ----------
WINSOR = [c for c in ["annual_inc", "revol_bal", "revol_util", "dti", "loan_to_income", "payment_to_income",
                      "tot_cur_bal", "tot_hi_cred_lim", "total_rev_hi_lim", "avg_cur_bal", "total_bal_ex_mort",
                      "total_bc_limit", "total_il_high_credit_limit", "bc_open_to_buy", "tot_coll_amt",
                      "delinq_amnt"] if c in F.columns]
rows = []
for c in WINSOR:
    x = F[c]
    lo_t, hi_t = x[tr].quantile(0.001), x[tr].quantile(0.999)
    row = {"feature": c, "train_min": x[tr].min(), "p0.1": lo_t, "p99.9": hi_t, "train_max": x[tr].max()}
    for sp in ["train", "valid", "oot"]:
        m = d["split"] == sp
        row[f"cap_pct_{sp}"] = round(float(((x[m] < lo_t) | (x[m] > hi_t)).mean() * 100), 3)
    F[c] = x.clip(lo_t, hi_t)
    note = f"winsorised at train p0.1/p99.9 [{lo_t:.4g}, {hi_t:.4g}]"
    dictionary[c]["treatment"] = (dictionary[c]["treatment"] + "; " + note) if dictionary[c]["treatment"] != "as-is" else note
    rows.append(row)
wz = pd.DataFrame(rows)
wz.to_csv("reports/winsor_thresholds.csv", index=False)
print("\n=== Winsorisation (thresholds fitted on train; cap_pct = % of rows clipped, either tail) ===")
print(wz.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# ---------- sanity checks ----------
assert set(F.columns) == set(dictionary), "every feature must be documented"
num_cols = F.select_dtypes("number").columns.tolist()
assert not np.isinf(F[num_cols]).any().any(), "infinite values present"
print(f"\nFeatures: {F.shape[1]} ({len(num_cols)} numeric, {F.shape[1] - len(num_cols)} categorical); no infinite values")

# single-feature AUC on train: leakage tripwire
y = d.loc[tr, "target"]
auc = {}
for c in num_cols:
    x = F.loc[tr, c]
    ok = x.notnull()
    if ok.sum() > 1000 and x[ok].nunique() > 1:
        a = roc_auc_score(y[ok], x[ok])
        auc[c] = max(a, 1 - a)
bench_rate = to_num(d["int_rate"])
ok = bench_rate[tr].notnull()
bench_auc = roc_auc_score(y[ok], bench_rate[tr][ok])
auc_s = pd.Series(auc).sort_values(ascending=False)
print("\n=== Single-feature AUC on train (orientation-free): top 15 ===")
print(auc_s.head(15).round(4).to_string())
print(f"\nReference: int_rate (lender-assigned, benchmark only) AUC = {max(bench_auc, 1 - bench_auc):.4f}")
print(f"Features above 0.70 (investigate for leakage): {auc_s[auc_s > 0.70].index.tolist()}")

# ---------- outputs ----------
fd = pd.DataFrame(list(dictionary.values()))
fd["pct_missing_train"] = fd["feature"].map(F.loc[tr].isnull().mean().mul(100).round(2))
fd["single_feature_auc"] = fd["feature"].map(auc_s.round(4))
fd.to_csv("reports/feature_dictionary.csv", index=False)

print("\nFeatures with missing values in train (%):")
mm = fd[fd["pct_missing_train"] > 0].sort_values("pct_missing_train", ascending=False)
print(mm[["feature", "pct_missing_train"]].head(30).to_string(index=False))

bench = pd.DataFrame({"bench_int_rate": bench_rate,
                      "bench_grade": d["grade"].astype("string"),
                      "bench_sub_grade": d["sub_grade"].astype("string")})
out = pd.concat([d[["id", "issue_d", "split", "target"]], F, bench], axis=1)
out.to_parquet("data/processed/features.parquet", index=False)
size_mb = os.path.getsize("data/processed/features.parquet") / 1e6
print(f"\nSaved data/processed/features.parquet: {out.shape[0]:,} rows x {out.shape[1]} cols ({size_mb:.0f} MB)")
print(out.groupby("split")["target"].agg(n="size", bad_rate="mean").assign(
    bad_rate=lambda t: (t["bad_rate"] * 100).round(2)).to_string())
print("Saved reports/feature_dictionary.csv, categorical_levels.csv, winsor_thresholds.csv")
