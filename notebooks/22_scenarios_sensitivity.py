import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.woe import WoEBinner  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 40)

# ---------- rules fixed before the results were seen ----------
THR_SHARE = 0.20                       # policy threshold: 80th percentile of validation PD
OE_LOW, OE_HIGH = 0.95, 1.05           # Section 12.5 band
N_BOOT, SEED = 200, 42
SIGN_STABILITY_MIN = 95.0              # each coefficient must be negative in at least this % of resamples
EXTRA_SHIFT = 0.15                     # hypothetical drift, log-odds
WINDOWS = {"Jul": ["2015-07"], "Jul-Aug": ["2015-07", "2015-08"],
           "Jul-Sep": ["2015-07", "2015-08", "2015-09"], "Jul-Oct": ["2015-07", "2015-08", "2015-09", "2015-10"]}
MONTHS = ["2015-07", "2015-08", "2015-09", "2015-10", "2015-11", "2015-12", "2016-01"]

with open("models/scorecard_final.json", encoding="utf-8") as fh:
    SC = json.load(fh)
FEATS = SC["features"]
assert len(FEATS) == 16
rc = SC["recalibration"]
assert rc["method"] == "intercept shift"
SHIFT_TESTED = float(rc["tested_parameters"]["shift_logodds"])
SHIFT_DEPLOY = float(rc["deployment_parameters_untested"]["shift_logodds"])
binner = WoEBinner.load("models/woe_spec.json")

df = pd.read_parquet("data/processed/features.parquet", columns=["issue_d", "split", "target"] + FEATS)
S = {k: df[df["split"] == k].reset_index(drop=True) for k in ["train", "valid", "oot"]}
Y = {k: v["target"].to_numpy().astype(int) for k, v in S.items()}
print("Rows: " + ", ".join(f"{k}={len(v):,}" for k, v in S.items()))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    q = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(q / (1 - q))


beta = np.array([SC["coefficients"][f] for f in FEATS])


def score(d):
    W = binner.transform(d, FEATS)
    return sigmoid(SC["intercept"] + W[FEATS].to_numpy() @ beta)


V = S["valid"]
p0 = score(V)
auc = roc_auc_score(Y["valid"], p0)
thr = float(np.quantile(p0, 1 - THR_SHARE))
assert abs(auc - 0.6708) < 2e-4, f"validation AUC {auc:.4f}, expected 0.6708"
assert abs(thr - 0.1962) < 5e-4, f"policy threshold {thr:.4f}, expected 0.1962"
assert abs(p0.mean() - 0.1379) < 5e-4, f"mean validation PD {p0.mean():.4f}, expected 0.1379"
print(f"Saved model reproduces validation AUC {auc:.4f}, mean PD {p0.mean() * 100:.2f}% and the policy threshold {thr:.4f}\n")


def summary(p):
    above = p >= thr
    return {"mean_pd_pct": float(p.mean() * 100), "share_above_thr_pct": float(above.mean() * 100),
            "approval_rate_pct": float((~above).mean() * 100), "mean_pd_approved_pct": float(p[~above].mean() * 100),
            "newly_rejected_pct": float(((p >= thr) & (p0 < thr)).mean() * 100)}


base = summary(p0)

# ================= 1. one-at-a-time shocks =================
NUMF = [f for f in FEATS if binner.specs[f]["kind"] == "numeric"]
tr = S["train"]
SD = {f: float(tr[f].std()) for f in NUMF}
LO = {f: float(tr[f].min()) for f in NUMF}
HI = {f: float(tr[f].max()) for f in NUMF}
DIR = {f: (1 if binner.specs[f]["direction"] == "increasing" else -1) for f in NUMF}

rows = []
for f in NUMF:
    delta = DIR[f] * SD[f]
    d2 = V.copy()
    d2[f] = np.clip(V[f].to_numpy(float) + delta, LO[f], HI[f])
    p1 = score(d2)
    dp = p1 - p0
    s = summary(p1)
    rows.append({"feature": f, "adverse": "up" if DIR[f] > 0 else "down", "shock_one_sd": delta,
                 "mean_dpd_pp": float(dp.mean() * 100), "mean_dpd_rel_pct": float(dp.mean() / p0.mean() * 100),
                 "rows_pd_down_pct": float((dp < -1e-12).mean() * 100), "rows_unchanged_pct": float((np.abs(dp) <= 1e-12).mean() * 100),
                 "share_above_thr_pct": s["share_above_thr_pct"], "newly_rejected_pct": s["newly_rejected_pct"]})
oat = pd.DataFrame(rows).sort_values("mean_dpd_pp", ascending=False).reset_index(drop=True)
print("=== 1. One-at-a-time shocks: one train SD in the adverse direction, clipped to the train range (validation loans) ===")
print(oat.round(4).to_string(index=False))
mono_ok = bool((oat["rows_pd_down_pct"] == 0).all() and (oat["mean_dpd_pp"] > 0).all())
print(f"\nMonotonicity check (every mean PD change positive and no loan's PD falls): {'PASS' if mono_ok else 'FAIL'}")

# ================= 2. named shocks and the joint scenario =================
def apply_named(d, keys):
    d2 = d.copy()
    if "fico" in keys:
        d2["fico_mean"] = np.clip(d2["fico_mean"].to_numpy(float) - 20.0, LO["fico_mean"], HI["fico_mean"])
    if "dti" in keys:
        d2["dti"] = np.clip(d2["dti"].to_numpy(float) + 5.0, LO["dti"], HI["dti"])
    if "inc" in keys:
        d2["annual_inc"] = np.clip(d2["annual_inc"].to_numpy(float) * 0.9, LO["annual_inc"], HI["annual_inc"])
        d2["loan_to_income"] = np.clip(d2["loan_to_income"].to_numpy(float) / 0.9, LO["loan_to_income"], HI["loan_to_income"])
    if "bcu" in keys:
        d2["bc_util"] = np.clip(d2["bc_util"].to_numpy(float) + 10.0, LO["bc_util"], HI["bc_util"])
    return d2


NAMED = {"baseline": [], "fico_mean -20 points": ["fico"], "dti +5 points": ["dti"],
         "annual_inc -10% (loan_to_income raised by 1/0.9)": ["inc"], "bc_util +10 points": ["bcu"],
         "Joint: all four": ["fico", "dti", "inc", "bcu"]}
rows = []
for name, keys in NAMED.items():
    p1 = p0 if not keys else score(apply_named(V, keys))
    s = summary(p1)
    rows.append({"scenario": name, **s, "delta_mean_pd_pp": s["mean_pd_pct"] - base["mean_pd_pct"]})
named = pd.DataFrame(rows)
print(f"\n=== 2. Named shocks (validation loans; policy threshold fixed at {thr:.4f}) ===")
print(named.round(3).to_string(index=False))

# ================= 3. drift scenarios =================
rows = []
for name, d in [("no shift", 0.0), (f"tested shift {SHIFT_TESTED:.4f}", SHIFT_TESTED),
                (f"deployment shift {SHIFT_DEPLOY:.4f}", SHIFT_DEPLOY), (f"hypothetical shift {EXTRA_SHIFT:.2f}", EXTRA_SHIFT)]:
    s = summary(sigmoid(logit(p0) + d))
    rows.append({"scenario": name, "shift_logodds": d, **s, "delta_mean_pd_pp": s["mean_pd_pct"] - base["mean_pd_pct"]})
drift = pd.DataFrame(rows)
print(f"\n=== 3. Drift scenarios: a log-odds shift on validation PDs, rejection threshold held at {thr:.4f} ===")
print(drift.round(3).to_string(index=False))

# ================= 4. coefficient stability =================
Wtr = binner.transform(S["train"], FEATS)[FEATS].to_numpy()
ytr = Y["train"].astype(float)
X1 = np.column_stack([np.ones(len(ytr)), Wtr])
ref = np.r_[SC["intercept"], beta]


def irls(X, y, w, b0, iters=30, tol=1e-9):
    b = b0.copy()
    for _ in range(iters):
        p = sigmoid(X @ b)
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        g = X.T @ (w * (y - p))
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < tol:
            break
    return b


b_full = irls(X1, ytr, np.ones(len(ytr)), np.zeros(X1.shape[1]))
assert np.abs(b_full - ref).max() < 1e-3, "IRLS does not reproduce the saved coefficients"
print(f"\nIRLS reproduces the saved coefficients (largest difference {np.abs(b_full - ref).max():.2e})")
rng = np.random.default_rng(SEED)
B = np.empty((N_BOOT, X1.shape[1]))
for i in range(N_BOOT):
    B[i] = irls(X1, ytr, rng.poisson(1.0, len(ytr)).astype(float), ref)
bs = pd.DataFrame({"feature": FEATS, "coef_saved": beta, "boot_mean": B[:, 1:].mean(0), "boot_sd": B[:, 1:].std(0, ddof=1),
                   "p2_5": np.percentile(B[:, 1:], 2.5, axis=0), "p97_5": np.percentile(B[:, 1:], 97.5, axis=0),
                   "pct_negative": (B[:, 1:] < 0).mean(0) * 100})
bs["rel_sd_pct"] = bs["boot_sd"] / bs["coef_saved"].abs() * 100
bs["sign_stable"] = bs["pct_negative"] >= SIGN_STABILITY_MIN
print(f"\n=== 4. Coefficient stability: {N_BOOT} Poisson-bootstrap refits on train (WoE bins held fixed) ===")
print(bs.sort_values("rel_sd_pct", ascending=False).round(4).to_string(index=False))
stab_ok = bool(bs["sign_stable"].all())
print(f"\nSign stability (every coefficient negative in at least {SIGN_STABILITY_MIN:.0f}% of resamples): {'PASS' if stab_ok else 'FAIL'}; "
      f"lowest: {bs.loc[bs['pct_negative'].idxmin(), 'feature']} at {bs['pct_negative'].min():.1f}%")

# ================= 5. sensitivity of the recalibration to its window =================
O = S["oot"]
Po, yo = score(O), Y["oot"]
mon = pd.to_datetime(O["issue_d"], format="%b-%Y").dt.to_period("M").astype(str).to_numpy()
rows = [{"window": "no shift (raw)", "n_cal": 0, "shift": 0.0, **{m: float(yo[mon == m].mean() / Po[mon == m].mean()) for m in MONTHS}}]
for name, wm in WINDOWS.items():
    m = np.isin(mon, wm)
    shift = float(sm.GLM(yo[m], np.ones((int(m.sum()), 1)), family=sm.families.Binomial(), offset=logit(Po[m])).fit().params[0])
    ps = sigmoid(logit(Po) + shift)
    row = {"window": name, "n_cal": int(m.sum()), "shift": shift}
    post = []
    for mm in MONTHS:
        k = mon == mm
        oe = float(yo[k].mean() / ps[k].mean())
        row[mm] = oe
        if mm not in wm:
            post.append(oe)
    row["months_after_window"] = len(post)
    row["after_window_in_band"] = int(sum(OE_LOW <= v <= OE_HIGH for v in post))
    row["after_window_max_OE"] = max(post)
    rows.append(row)
win = pd.DataFrame(rows)
assert abs(float(win.loc[win["window"] == "Jul-Sep", "shift"].iloc[0]) - SHIFT_TESTED) < 1e-3, "Jul-Sep shift differs from Section 15"
print("\n=== 5. Observed/expected by out-of-time month after an intercept shift fitted on each window ===")
print("(months inside the fitting window are in-sample: Jul = first 1 month, Jul-Aug = 2, Jul-Sep = 3, Jul-Oct = 4)")
print(win.round(3).to_string(index=False))

# ---------- outputs ----------
os.makedirs("reports/figures", exist_ok=True)
oat.to_csv("reports/scenarios_oat.csv", index=False)
named.to_csv("reports/scenarios_named.csv", index=False)
drift.to_csv("reports/scenarios_drift.csv", index=False)
bs.to_csv("reports/bootstrap_coefficients.csv", index=False)
win.to_csv("reports/recalibration_windows.csv", index=False)
try:
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    o = oat.sort_values("mean_dpd_pp")
    ax[0, 0].barh(o["feature"], o["mean_dpd_pp"], color="tab:red")
    ax[0, 0].set_xlabel("mean PD change (percentage points)")
    ax[0, 0].set_title("One train SD in the adverse direction, one feature at a time")
    ax[0, 0].tick_params(axis="y", labelsize=8)
    sc = pd.concat([named, drift[drift["scenario"] != "no shift"]], ignore_index=True)
    colors = ["grey"] + ["tab:orange"] * 4 + ["tab:red"] + ["tab:purple"] * 3
    ax[0, 1].barh(sc["scenario"], sc["mean_pd_pct"], color=colors[:len(sc)])
    ax[0, 1].invert_yaxis()
    for i, (m, s) in enumerate(zip(sc["mean_pd_pct"], sc["share_above_thr_pct"])):
        ax[0, 1].text(m, i, f" {m:.1f}% (reject {s:.0f}%)", va="center", fontsize=7)
    ax[0, 1].set_xlim(0, float(sc["mean_pd_pct"].max()) * 1.35)
    ax[0, 1].set_xlabel("mean PD (%)")
    ax[0, 1].set_title("Scenarios: mean PD and share rejected at the fixed threshold")
    ax[0, 1].tick_params(axis="y", labelsize=7)
    b2 = bs.sort_values("boot_mean")
    yy = np.arange(len(b2))
    ax[1, 0].errorbar(b2["boot_mean"], yy, xerr=[b2["boot_mean"] - b2["p2_5"], b2["p97_5"] - b2["boot_mean"]], fmt="o", ms=4, capsize=2)
    ax[1, 0].axvline(0, color="grey", lw=1)
    ax[1, 0].set_yticks(yy)
    ax[1, 0].set_yticklabels(b2["feature"], fontsize=8)
    ax[1, 0].set_xlabel("coefficient (bootstrap mean and 95% interval)")
    ax[1, 0].set_title("Coefficient stability")
    xs = np.arange(len(MONTHS))
    for _, r in win.iterrows():
        style = dict(color="grey", ls="--") if r["window"].startswith("no shift") else {}
        ax[1, 1].plot(xs, [r[m] for m in MONTHS], marker="o", label=r["window"], **style)
    ax[1, 1].axhspan(OE_LOW, OE_HIGH, color="green", alpha=0.15)
    ax[1, 1].axhline(1.0, color="grey", lw=1)
    ax[1, 1].set_xticks(xs)
    ax[1, 1].set_xticklabels(MONTHS, rotation=45)
    ax[1, 1].set_ylabel("observed / expected")
    ax[1, 1].set_title("Out-of-time O/E by month after a shift fitted on each window")
    ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("reports/figures/scenario_sensitivity.png", dpi=150)
    print("\nSaved reports/figures/scenario_sensitivity.png")
except Exception as e:
    print(f"\nFigure skipped ({type(e).__name__}: {e}); results above are unaffected")

print("Saved reports/scenarios_oat.csv, scenarios_named.csv, scenarios_drift.csv, bootstrap_coefficients.csv, recalibration_windows.csv")
