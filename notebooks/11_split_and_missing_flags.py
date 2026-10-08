import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)

FILE = "data/raw/accepted_2007_to_2018Q4.csv.gz"
OOT_START = pd.Timestamp("2015-07-01")
SEED = 42

# ---------- 1. Split assignment ----------
dev = pd.read_csv("data/processed/development_sample_ids_target.csv", dtype={"id": str})
assert dev["id"].is_unique
dev["issue_dt"] = pd.to_datetime(dev["issue_d"], format="%b-%Y")

dev["split"] = "oot"
window = dev.index[dev["issue_dt"] < OOT_START]
train_idx, valid_idx = train_test_split(
    window, test_size=0.20, stratify=dev.loc[window, "target"], random_state=SEED)
dev.loc[train_idx, "split"] = "train"
dev.loc[valid_idx, "split"] = "valid"

summary = dev.groupby("split").agg(n=("target", "size"), bad=("target", "sum"),
                                   bad_rate=("target", "mean"),
                                   first=("issue_dt", "min"), last=("issue_dt", "max"))
summary["bad_rate"] = (summary["bad_rate"] * 100).round(2)
print("=== Split summary ===")
print(summary.to_string())
dev[["id", "split"]].to_csv("data/processed/development_split.csv", index=False)
id_set = set(dev["id"])

# ---------- 2. Pull the columns needed for the missing-flag analysis ----------
FLAG_COLS = ["mths_since_last_record", "mths_since_last_delinq", "mths_since_last_major_derog",
             "mths_since_recent_bc_dlq", "mths_since_recent_revol_delinq", "mths_since_recent_inq",
             "mo_sin_old_il_acct", "mths_since_recent_bc", "bc_util", "percent_bc_gt_75",
             "bc_open_to_buy", "num_tl_120dpd_2m", "emp_length"]
HELPERS = ["num_bc_tl", "num_bc_sats", "num_actv_bc_tl"]

frames, matched = [], 0
reader = pd.read_csv(FILE, compression="gzip", usecols=["id", "emp_title"] + FLAG_COLS + HELPERS,
                     dtype={"id": str}, low_memory=False, chunksize=200000)
for i, ch in enumerate(reader):
    ch = ch[ch["id"].isin(id_set)]
    if ch.empty:
        continue
    matched += len(ch)
    sub = ch[["id"] + FLAG_COLS + HELPERS].copy()
    sub["emp_title_missing"] = ch["emp_title"].isnull()
    frames.append(sub)
    print(f"Chunk {i+1}: matched so far {matched:,}")

assert matched == len(dev), f"row mismatch: {matched:,} vs {len(dev):,}"
d = pd.concat(frames, ignore_index=True).merge(
    dev[["id", "split", "target", "issue_dt"]], on="id", how="left", validate="one_to_one")
assert d["split"].notnull().all() and len(d) == len(dev)
print(f"\nRow match check: {matched:,} vs {len(dev):,} -> OK")

miss = pd.DataFrame({c: d[c].isnull() for c in FLAG_COLS})
miss["emp_title"] = d["emp_title_missing"]

# ---------- 3. Does the missing group behave differently? (TRAIN rows only) ----------
def ztest_p(n1, b1, n0, b0):
    if min(n1, n0) < 30:
        return np.nan
    pp = (b1 + b0) / (n1 + n0)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
    return np.nan if se == 0 else 2 * norm.sf(abs((b1 / n1 - b0 / n0) / se))

t = d["split"] == "train"
yr = d["issue_dt"].dt.year
rows = []
for c in miss.columns:
    m, p = miss[c] & t, (~miss[c]) & t
    n1, n0 = int(m.sum()), int(p.sum())
    b1, b0 = int(d.loc[m, "target"].sum()), int(d.loc[p, "target"].sum())
    num = den = 0.0
    for y in sorted(yr[t].unique()):
        ym, yp = m & (yr == y), p & (yr == y)
        if ym.sum() >= 30 and yp.sum() >= 30:
            num += ym.sum() * (d.loc[ym, "target"].mean() - d.loc[yp, "target"].mean())
            den += ym.sum()
    rows.append({
        "column": c, "n_missing": n1, "pct_missing": round(n1 / t.sum() * 100, 2),
        "bad_rate_missing": round(b1 / n1 * 100, 2) if n1 else np.nan,
        "bad_rate_present": round(b0 / n0 * 100, 2),
        "diff_pp": round((b1 / n1 - b0 / n0) * 100, 2) if n1 else np.nan,
        "same_year_diff_pp": round(num / den * 100, 2) if den else np.nan,
        "p_value": ztest_p(n1, b1, n0, b0),
    })
sig = pd.DataFrame(rows)
print("\n=== Bad rate: missing vs present (train rows only) ===")
print("diff_pp = missing minus present; same_year_diff_pp compares within issue year to strip out drift")
print(sig.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
sig.to_csv("reports/missing_flag_signal.csv", index=False)

# ---------- 4. Do the residual bank-card gaps hit the same loans? (no outcomes used) ----------
bc4 = ["bc_util", "percent_bc_gt_75", "bc_open_to_buy", "mths_since_recent_bc"]
has_bc = d["num_bc_tl"] > 0
k = miss.loc[has_bc, bc4].sum(axis=1)
print(f"\n=== Loans with >=1 bankcard (n={int(has_bc.sum()):,}): how many of the 4 bc fields are missing per loan ===")
print(k.value_counts().sort_index().to_string())
anym = k > 0
print(f"\nOf the {int(anym.sum()):,} loans missing >=1 bc field, "
      f"{miss.loc[has_bc & anym.reindex(d.index, fill_value=False), 'num_tl_120dpd_2m'].mean()*100:.1f}% "
      f"are also missing num_tl_120dpd_2m")

rows = []
for col in bc4:
    for cond in ["num_bc_sats", "num_actv_bc_tl"]:
        valid = d[cond].notnull()
        z, ps = valid & (d[cond] == 0), valid & (d[cond] > 0)
        rows.append({"column": col, "condition_col": cond,
                     "n_zero": int(z.sum()), "missing_pct_if_zero": round(miss.loc[z, col].mean() * 100, 1),
                     "n_positive": int(ps.sum()), "missing_pct_if_positive": round(miss.loc[ps, col].mean() * 100, 1)})
print("\n=== Do num_bc_sats / num_actv_bc_tl explain the bc gaps? ===")
print(pd.DataFrame(rows).to_string(index=False))

# ---------- 5. Missingness by issue quarter (batch / pull-date artefacts?) ----------
q = d["issue_dt"].dt.to_period("Q")
qcols = ["num_tl_120dpd_2m", "mths_since_recent_inq", "mo_sin_old_il_acct", "bc_util", "emp_length"]
qt = (miss[qcols].groupby(q).mean() * 100).round(1)
qt.insert(0, "n", d.groupby(q).size())
print("\n=== % missing by issue quarter ===")
print(qt.to_string())
