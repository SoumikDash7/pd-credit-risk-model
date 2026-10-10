import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.woe import WoEBinner  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

# ---- rules (fixed before looking at results) ----
IV_MIN = 0.02
FORCE_INCLUDE = ["emp_length_yrs"]
EXCLUDED = {
    "initial_list_status": "PSI 0.28 out-of-time, IV 0.006 (Section 11)",
    "num_tl_120dpd_2m": "IV 0.002; missingness tied to a 2014 data-feed change (Section 11)",
    "addr_state": "IV 0.016 below the line; geographic proxy (Sections 11, 12.4)", "payment_to_income": "installment embeds int_rate, a lender-assigned price (Section 13)",
}
CORR_MAX = 0.70
WATCH = 0.50
VIF_MAX = 5.0

summ = pd.read_csv("reports/iv_summary.csv")
summ["flags"] = summ["flags"].fillna("")
iv = summ.set_index("feature")["iv_train"]
kind = summ.set_index("feature")["kind"]
flags = summ.set_index("feature")["flags"]

df = pd.read_parquet("data/processed/features.parquet")
tr = df[df["split"] == "train"]
binner = WoEBinner.load("models/woe_spec.json")
print(f"Train rows: {len(tr):,} | features in iv_summary: {len(summ)}\n")

# ---- step 1: exclusions and IV gate ----
info, cand = {}, []
for f in summ["feature"]:
    if f in EXCLUDED:
        info[f] = ("excluded", EXCLUDED[f])
    elif iv[f] < IV_MIN and f not in FORCE_INCLUDE:
        info[f] = ("dropped_iv", f"IV {iv[f]:.4f} < {IV_MIN}")
    else:
        cand.append(f)

# ---- step 2: correlation of the WoE-transformed candidates (train only) ----
W = binner.transform(tr, cand)
corr = W.corr()
assert not corr.isnull().any().any(), "a candidate has constant WoE"

order = sorted(cand, key=lambda f: (bool(flags[f]), -iv[f]))  # unflagged first, then by IV
kept = []
for f in order:
    if kept:
        c = corr.loc[f, kept].abs()
        j = c.idxmax()
        if c[j] >= CORR_MAX:
            info[f] = ("dropped_corr", f"|r|={c[j]:.2f} with {j} (IV {iv[j]:.4f})")
            continue
    kept.append(f)
    info[f] = ("kept", "")

n = lambda s: sum(1 for v in info.values() if v[0] == s)
print("=== Selection funnel ===")
print(f"Features: {len(summ)}")
print(f"  excluded by earlier decisions : {n('excluded')}")
print(f"  dropped, IV below {IV_MIN}      : {n('dropped_iv')}")
print(f"  candidates after IV gate      : {len(cand)}")
print(f"  dropped for correlation >= {CORR_MAX} : {n('dropped_corr')}")
print(f"  KEPT                          : {len(kept)}\n")

dc = pd.DataFrame([{"feature": f, "iv_train": iv[f], "reason": r}
                   for f, (s, r) in info.items() if s == "dropped_corr"]).sort_values("iv_train", ascending=False)
print("=== Dropped for correlation ===")
print(dc.to_string(index=False))

# ---- step 3: VIF and pairwise check on the survivors ----
Rk = W[kept].corr().to_numpy()
vif = np.diag(np.linalg.inv(Rk))
maxr = np.abs(Rk - np.eye(len(kept))).max(axis=1)
kt = pd.DataFrame({"feature": kept, "kind": [kind[f] for f in kept], "iv_train": [iv[f] for f in kept],
                   "flags": [flags[f] for f in kept], "max_abs_r_with_kept": maxr.round(2), "vif": vif.round(2)})
print("\n=== KEPT features ===")
print(kt.to_string(index=False))
print(f"\nMax VIF among kept: {vif.max():.2f} ({'below' if vif.max() < VIF_MAX else 'ABOVE'} {VIF_MAX})")

pairs = [(kept[i], kept[j], round(float(Rk[i, j]), 2)) for i in range(len(kept)) for j in range(i + 1, len(kept))
         if abs(Rk[i, j]) >= WATCH]
print(f"\n=== Watch pairs among kept (|r| between {WATCH} and {CORR_MAX}): {len(pairs)} ===")
for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])):
    print(f"  {a} ~ {b}: {r}")

# ---- outputs ----
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)
pd.DataFrame([{"feature": f, "kind": kind[f], "iv_train": iv[f], "flags": flags[f],
               "status": info[f][0], "reason": info[f][1]} for f in summ["feature"]]
             ).sort_values(["status", "iv_train"], ascending=[True, False]).to_csv("reports/variable_selection.csv", index=False)
corr.round(3).to_csv("reports/woe_corr_train.csv")
with open("models/selected_features.json", "w", encoding="utf-8") as fh:
    json.dump({"selected": kept, "rules": {"iv_min": IV_MIN, "force_include": FORCE_INCLUDE,
               "corr_max": CORR_MAX, "vif_max": VIF_MAX, "excluded": list(EXCLUDED)}}, fh, indent=1)

try:
    dist = 1 - corr.abs().to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    leaf = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    names = [cand[i] for i in leaf]
    M = corr.loc[names, names].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ticks = [f + (" *" if f in kept else "") for f in names]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(ticks, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(ticks, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r between WoE features (train)")
    ax.set_title("Candidate features after the IV gate, clustered by correlation (* = selected)")
    fig.tight_layout()
    fig.savefig("reports/figures/woe_correlation.png", dpi=150)
    print("\nSaved reports/figures/woe_correlation.png")
except Exception as e:
    print(f"\nFigure skipped ({type(e).__name__}: {e}); selection results above are unaffected")

print("Saved reports/variable_selection.csv, reports/woe_corr_train.csv, models/selected_features.json")
