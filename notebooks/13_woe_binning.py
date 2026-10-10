import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.woe import WoEBinner, psi  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 40)

MEDIAN_IMPUTE = ["revol_util", "pct_tl_nvr_dlq", "avg_cur_bal"]
# screening rules (analyst judgement, not standards)
IV_SUSPICIOUS, IV_COST_PCT, OOT_RATIO_MIN, PSI_HIGH, MIN_BAD = 0.5, 15.0, 0.70, 0.25, 100

df = pd.read_parquet("data/processed/features.parquet")
fd = pd.read_csv("reports/feature_dictionary.csv")
numeric = fd.loc[fd["kind"] == "numeric", "feature"].tolist()
categorical = fd.loc[fd["kind"] == "categorical", "feature"].tolist()
absent = [c for c in numeric + categorical if c not in df.columns]
assert not absent, f"features missing from parquet: {absent}"

splits = {k: df[df["split"] == k] for k in ["train", "valid", "oot"]}
tr = splits["train"]
print(f"Rows: " + ", ".join(f"{k}={len(v):,}" for k, v in splits.items()))
print(f"Features: {len(numeric)} numeric, {len(categorical)} categorical\n")

binner = WoEBinner(min_bin_share=0.02, min_bin_count=1000, max_bins=10, median_impute=MEDIAN_IMPUTE)
binner.fit(tr, tr["target"], numeric, categorical)


def split_iv(cnt, keep, woe_all):
    n, bad = cnt[0][keep], cnt[1][keep]
    pg = np.maximum(n - bad, 0.5) / (n.sum() - bad.sum())
    pb = np.maximum(bad, 0.5) / bad.sum()
    return float(((pg - pb) * woe_all[keep]).sum())


def shares(cnt, keep):
    n = cnt[0][keep]
    return n / n.sum()


def monotone(cnt, nb, direction):
    n, bad = cnt
    ok = [i for i in range(nb) if n[i] >= 30]; r = [bad[i] / n[i] for i in ok]; se = [(r[k] * (1 - r[k]) / n[ok[k]]) ** 0.5 for k in range(len(ok))]
    if len(r) < 2:
        return True
    up = all(r[j] - r[j + 1] <= 2 * (se[j] ** 2 + se[j + 1] ** 2) ** 0.5 for j in range(len(r) - 1))
    down = all(r[j + 1] - r[j] <= 2 * (se[j] ** 2 + se[j + 1] ** 2) ** 0.5 for j in range(len(r) - 1))
    return up if direction == "increasing" else down


def band(iv):
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious"


bin_rows, sum_rows = [], []
for c in numeric + categorical:
    s = binner.specs[c]
    nb = len(s["woe"])
    woe_all = np.array(list(s["woe"]) + [s["missing_woe"] if s["missing_woe"] is not None else 0.0, 0.0])
    labels = list(s["labels"]) + ["Missing", "Unseen level"]
    cnt = {k: binner.bin_counts(c, d[c], d["target"]) for k, d in splits.items()}
    keep = [i for i in range(nb + 2) if i < nb or any(cnt[k][0][i] > 0 for k in cnt)]

    iv_tr = split_iv(cnt["train"], keep, woe_all)
    assert abs(iv_tr - s["iv_train"]) < 1e-6, f"{c}: fit/transform mismatch ({iv_tr} vs {s['iv_train']})"
    iv_va, iv_oo = split_iv(cnt["valid"], keep, woe_all), split_iv(cnt["oot"], keep, woe_all)
    psi_va = psi(shares(cnt["train"], keep), shares(cnt["valid"], keep))
    psi_oo = psi(shares(cnt["train"], keep), shares(cnt["oot"], keep))
    mono_va = monotone(cnt["valid"], nb, s["direction"])
    mono_oo = monotone(cnt["oot"], nb, s["direction"])
    min_bad = min(int(cnt["train"][1][i]) for i in keep if cnt["train"][0][i] > 0)
    iv_unc = s["iv_unconstrained"]
    cost = (iv_unc - iv_tr) / iv_unc * 100 if iv_unc > 0 else 0.0
    ratio = iv_oo / iv_tr if iv_tr > 0 else np.nan

    flags = []
    if iv_tr > IV_SUSPICIOUS:
        flags.append("IV>0.5: check leakage")
    if iv_unc >= 0.02 and cost > IV_COST_PCT:
        flags.append(f"monotone costs {cost:.0f}% of IV")
    if iv_tr >= 0.02 and ratio < OOT_RATIO_MIN:
        flags.append(f"OOT IV only {ratio * 100:.0f}% of train")
    if psi_oo > PSI_HIGH:
        flags.append(f"PSI(oot)={psi_oo:.2f}")
    if iv_tr >= 0.02 and not mono_oo:
        flags.append("OOT bin order broken")
    if min_bad < MIN_BAD:
        flags.append(f"thin bin ({min_bad} bad)")

    sum_rows.append({"feature": c, "kind": s["kind"], "n_bins": nb, "missing_bin": s["missing_woe"] is not None,
                     "pct_missing_train": round(float(tr[c].isnull().mean() * 100), 2), "direction": s["direction"],
                     "iv_train": round(iv_tr, 4), "iv_unconstrained": round(iv_unc, 4), "iv_cost_pct": round(cost, 1),
                     "iv_valid": round(iv_va, 4), "iv_oot": round(iv_oo, 4), "iv_oot_ratio": round(ratio, 2),
                     "psi_valid": round(psi_va, 4), "psi_oot": round(psi_oo, 4), "mono_valid": mono_va,
                     "mono_oot": mono_oo, "min_bad_train": min_bad, "band": band(iv_tr), "flags": "; ".join(flags)})

    for i in keep:
        row = {"feature": c, "bin": i if i < nb else ("Missing" if i == nb else "Unseen"),
               "label": labels[i], "woe": round(float(woe_all[i]), 4), "bad_train": int(cnt["train"][1][i])}
        for k in ["train", "valid", "oot"]:
            n_k, b_k = cnt[k][0][i], cnt[k][1][i]
            row[f"n_{k}"] = int(n_k)
            row[f"bad_rate_{k}"] = round(float(b_k / n_k * 100), 2) if n_k > 0 else np.nan
            row[f"share_{k}"] = round(float(n_k / cnt[k][0][keep].sum() * 100), 2)
        bin_rows.append(row)

summ = pd.DataFrame(sum_rows).sort_values("iv_train", ascending=False).reset_index(drop=True)
bins = pd.DataFrame(bin_rows)
print("Fit / transform consistency check passed for all features\n")

print("=== IV bands (train, in-sample) ===")
print(summ["band"].value_counts().reindex(["strong", "medium", "weak", "useless", "suspicious"]).fillna(0).astype(int).to_string())

cols = ["feature", "kind", "n_bins", "missing_bin", "direction", "iv_train", "iv_valid", "iv_oot",
        "iv_cost_pct", "psi_oot", "mono_oot"]
print("\n=== Top 25 features by train IV ===")
print(summ[cols].head(25).to_string(index=False))

fl = summ[summ["flags"] != ""]
print(f"\n=== Flagged for review ({len(fl)} features) ===")
print(fl[["feature", "iv_train", "iv_oot", "psi_oot", "flags"]].to_string(index=False))

pp = summ[summ["psi_oot"] >= 0.10].sort_values("psi_oot", ascending=False)
print(f"\n=== Features with PSI(oot) >= 0.10 ({len(pp)}) ===")
print(pp[["feature", "psi_valid", "psi_oot", "iv_train"]].to_string(index=False))

print("\n=== Bin tables for selected features ===")
show = ["fico_mean", "dti", "emp_length_yrs", "mths_since_recent_inq", "mths_since_last_record",
        "num_tl_120dpd_2m", "purpose", "addr_state"]
bcols = ["bin", "label", "n_train", "bad_rate_train", "woe", "bad_rate_valid", "n_oot", "bad_rate_oot"]
for f in show:
    if f in binner.specs:
        print(f"\n--- {f} (IV train {summ.loc[summ.feature == f, 'iv_train'].iloc[0]}) ---")
        print(bins.loc[bins["feature"] == f, bcols].to_string(index=False))

os.makedirs("models", exist_ok=True)
binner.save("models/woe_spec.json")
bins.to_csv("reports/woe_bins.csv", index=False)
summ.to_csv("reports/iv_summary.csv", index=False)
print("\nSaved models/woe_spec.json, reports/woe_bins.csv, reports/iv_summary.csv")
