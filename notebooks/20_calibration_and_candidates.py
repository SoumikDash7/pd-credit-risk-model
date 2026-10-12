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
Z_FLAG, OE_FLAG = 3.0, 0.05                       # part 1: flag a bin if |z| > 3 AND |O/E - 1| > 5%
GAIN_MIN, CORR_MAX = 0.002, 0.70                  # part 2: qualification thresholds
CI_LOW, CI_HIGH = 0.5, 99.5                       # 99% paired-bootstrap interval (four looks)
N_BOOT, SEED = 500, 42
AUC_CHECK, AUC_TOL = 0.6708, 2e-4                 # final scorecard validation AUC (Section 15)
GAP_UNC = 0.0131                                  # unconstrained XGBoost minus scorecard, validation (Section 16)

with open("models/scorecard_final.json", encoding="utf-8") as fh:
    SC = json.load(fh)
FEATS = SC["features"]
assert len(FEATS) == 16
binner = WoEBinner.load("models/woe_spec.json")

df = pd.read_parquet("data/processed/features.parquet", columns=["split", "target", "total_bc_limit"] + FEATS)
df = df[df["split"].isin(["train", "valid"])]
S = {k: df[df["split"] == k].reset_index(drop=True) for k in ["train", "valid"]}
Y = {k: v["target"].to_numpy().astype(int) for k, v in S.items()}
print("Out-of-time rows are not loaded or used in this script.")
print("Rows: " + ", ".join(f"{k}={len(v):,}" for k, v in S.items()) + "\n")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def boot_delta(y, p1, p0, B=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i]))
    return np.percentile(d, [CI_LOW, CI_HIGH])


# ---------- the saved scorecard ----------
Wb = {k: binner.transform(S[k], FEATS) for k in S}
beta = np.array([SC["coefficients"][f] for f in FEATS])
P = {k: sigmoid(SC["intercept"] + Wb[k][FEATS].to_numpy() @ beta) for k in S}
auc_v = roc_auc_score(Y["valid"], P["valid"])
assert abs(auc_v - AUC_CHECK) < AUC_TOL, f"saved scorecard gives validation AUC {auc_v:.4f}, expected {AUC_CHECK}"
print(f"Saved scorecard reproduces validation AUC {auc_v:.4f}; observed/expected on validation "
      f"{Y['valid'].mean() / P['valid'].mean():.4f}\n")

# ================= PART 1: calibration by feature decile =================
def assign_bins(kind, series, edges):
    if kind == "numeric":
        a = series.to_numpy(dtype=float)
        return np.where(np.isnan(a), -1, np.searchsorted(edges, a, side="right"))
    return series.astype(str).to_numpy()


def bin_stats(b, y, p):
    d = pd.DataFrame({"b": b, "y": y, "p": p})
    d["v"] = d["p"] * (1 - d["p"])
    g = d.groupby("b").agg(n=("y", "size"), bad=("y", "sum"), exp_sum=("p", "sum"), var=("v", "sum"))
    g["obs"] = g["bad"] / g["n"]
    g["exp"] = g["exp_sum"] / g["n"]
    g["se"] = np.sqrt(g["var"]) / g["n"]
    g["z"] = (g["obs"] - g["exp"]) / g["se"]
    g["OE"] = g["obs"] / g["exp"]
    return g


cal_rows = []
for f in FEATS:
    kind = binner.specs[f]["kind"]
    edges = None
    if kind == "numeric":
        av = S["valid"][f].to_numpy(dtype=float)
        edges = np.unique(np.quantile(av[~np.isnan(av)], np.linspace(0, 1, 11))[1:-1])
    st = {k: bin_stats(assign_bins(kind, S[k][f], edges), Y[k], P[k]) for k in ["valid", "train"]}
    bv = assign_bins(kind, S["valid"][f], edges)
    vals = S["valid"][f].to_numpy(dtype=float) if kind == "numeric" else None
    order = sorted(st["valid"].index, key=lambda x: (x == -1, x) if kind == "numeric" else (False, str(x)))
    for key in order:
        if kind == "numeric":
            if key == -1:
                label = "Missing"
            else:
                m = bv == key
                label = f"[{vals[m].min():.4g}, {vals[m].max():.4g}]"
        else:
            label = str(key)
        gv = st["valid"].loc[key]
        has_t = key in st["train"].index
        gt = st["train"].loc[key] if has_t else None
        flag_v = bool(abs(gv["z"]) > Z_FLAG and abs(gv["OE"] - 1) > OE_FLAG)
        flag_t = bool(has_t and abs(gt["z"]) > Z_FLAG and abs(gt["OE"] - 1) > OE_FLAG)
        cal_rows.append({"feature": f, "bin": label, "n_valid": int(gv["n"]), "obs_valid": gv["obs"],
                         "exp_valid": gv["exp"], "OE_valid": gv["OE"], "se_valid": gv["se"], "z_valid": gv["z"],
                         "flag_valid": flag_v, "n_train": int(gt["n"]) if has_t else 0,
                         "OE_train": gt["OE"] if has_t else np.nan, "z_train": gt["z"] if has_t else np.nan,
                         "flag_train": flag_t})
cal = pd.DataFrame(cal_rows)
n_tests, n_flag = len(cal), int(cal["flag_valid"].sum())
print(f"=== PART 1: calibration of the final scorecard by feature decile (validation) ===")
print(f"Bins tested: {n_tests}; flagged on validation (|z| > {Z_FLAG} and |O/E - 1| > {OE_FLAG:.0%}): {n_flag}; "
      f"expected by chance at 3 standard errors: about {0.0027 * n_tests:.1f}")
if n_flag:
    show = cal.loc[cal["flag_valid"], ["feature", "bin", "n_valid", "obs_valid", "exp_valid", "OE_valid", "z_valid",
                                       "OE_train", "z_train", "flag_train"]].copy()
    show["obs_valid"] = (show["obs_valid"] * 100).round(2)
    show["exp_valid"] = (show["exp_valid"] * 100).round(2)
    print(show.round(3).to_string(index=False))
else:
    print("No bin is flagged.")
summ = cal.groupby("feature").agg(bins=("bin", "size"), flagged=("flag_valid", "sum"),
                                  max_abs_z=("z_valid", lambda s: float(np.abs(s).max())),
                                  min_OE=("OE_valid", "min"), max_OE=("OE_valid", "max")).sort_values("max_abs_z", ascending=False)
print("\nBy feature (validation):")
print(summ.round(3).to_string())
print(f"\nBins flagged on both validation and train: {int((cal['flag_valid'] & cal['flag_train']).sum())}")

# ================= PART 2: capacity-ratio candidates =================
def derive(d):
    out = pd.DataFrame(index=d.index)
    inc = d["annual_inc"].astype("float64")
    lim = d["total_bc_limit"].astype("float64")
    out["total_bc_limit"] = lim
    out["hi_cred_lim_to_income"] = d["tot_hi_cred_lim"].astype("float64") / inc
    out["bc_balance_to_income"] = (lim * d["bc_util"].astype("float64") / 100.0) / inc
    return out


D = {k: derive(S[k]) for k in S}
fe = np.quantile(S["train"]["fico_mean"].to_numpy(float), [1 / 3, 2 / 3])
le = np.quantile(S["train"]["loan_to_income"].to_numpy(float), [1 / 3, 2 / 3])
for k in S:
    a = np.searchsorted(fe, S[k]["fico_mean"].to_numpy(float), side="right")
    b = np.searchsorted(le, S[k]["loan_to_income"].to_numpy(float), side="right")
    D[k]["fico_lti_cell"] = [f"f{i}_l{j}" for i, j in zip(a, b)]
NUMC = ["total_bc_limit", "hi_cred_lim_to_income", "bc_balance_to_income"]
CATC = ["fico_lti_cell"]
CANDS = NUMC + CATC
print("\nCandidate columns with missing values in train: "
      + (", ".join(f"{c} ({D['train'][c].isnull().mean() * 100:.2f}%)" for c in NUMC if D["train"][c].isnull().any()) or "none"))

nb = WoEBinner(min_bin_share=0.02, min_bin_count=1000, max_bins=10)
nb.fit(D["train"], Y["train"], NUMC, CATC)
WA = {k: pd.concat([Wb[k][FEATS], nb.transform(D[k], CANDS)], axis=1) for k in S}


def fit_lr(cols):
    return sm.Logit(Y["train"], sm.add_constant(WA["train"][cols], has_constant="add")).fit(disp=0, maxiter=100)


def pred_valid(res, cols):
    return np.asarray(res.predict(sm.add_constant(WA["valid"][cols], has_constant="add")))


r0 = fit_lr(FEATS)
p0 = pred_valid(r0, FEATS)
a0 = roc_auc_score(Y["valid"], p0)
print(f"\n=== PART 2: candidates added one at a time to the 16-feature scorecard (validation only) ===")
print(f"Baseline validation AUC {a0:.4f}; gap to the unconstrained XGBoost {GAP_UNC}")
rows = []
for c in CANDS:
    cols = FEATS + [c]
    r = fit_lr(cols)
    pc = pred_valid(r, cols)
    a1 = roc_auc_score(Y["valid"], pc)
    lo, hi = boot_delta(Y["valid"], pc, p0)
    coef, pv = float(r.params[c]), float(r.pvalues[c])
    mc = float(WA["train"][FEATS].corrwith(WA["train"][c]).abs().max())
    crit = {"gain_ok": bool(a1 - a0 >= GAIN_MIN), "interval_ok": bool(lo > 0), "sign_ok": bool(coef < 0), "corr_ok": bool(mc < CORR_MAX)}
    rows.append({"candidate": c, "iv_train": nb.specs[c]["iv_train"], "bins": len(nb.specs[c]["woe"]),
                 "coef": coef, "p_value": pv, "auc_valid": a1, "gain": a1 - a0, "ci99_low": lo, "ci99_high": hi,
                 "max_abs_corr_with_kept": mc, "share_of_gap_pct": max(0.0, a1 - a0) / GAP_UNC * 100, **crit,
                 "qualifies": all(crit.values())})
cd = pd.DataFrame(rows)
sh = cd.copy()
sh["p_value"] = sh["p_value"].map(lambda v: f"{v:.2g}")
print(sh.round(4).to_string(index=False))
qual = cd.loc[cd["qualifies"], "candidate"].tolist()
print(f"\nCandidates meeting all four criteria: {qual or 'none'}")
for _, r in cd.iterrows():
    if not r["qualifies"]:
        failed = [k for k in ["gain_ok", "interval_ok", "sign_ok", "corr_ok"] if not r[k]]
        print(f"  {r['candidate']}: fails {', '.join(failed)}")
if len(qual) >= 2:
    cols = FEATS + qual
    r = fit_lr(cols)
    pc = pred_valid(r, cols)
    lo, hi = boot_delta(Y["valid"], pc, p0)
    print(f"\nCombined model with {qual}: validation AUC {roc_auc_score(Y['valid'], pc):.4f} "
          f"(gain {roc_auc_score(Y['valid'], pc) - a0:+.4f}, 99% interval {lo:+.4f} to {hi:+.4f}); "
          f"coefficients {', '.join(f'{c} {r.params[c]:.3f}' for c in qual)}")

# ---------- outputs ----------
os.makedirs("reports/figures", exist_ok=True)
cal.to_csv("reports/calibration_by_feature.csv", index=False)
cd.to_csv("reports/capacity_candidates.csv", index=False)
try:
    fig, axes = plt.subplots(4, 4, figsize=(15, 11))
    for ax, f in zip(axes.ravel(), FEATS):
        g = cal[cal["feature"] == f].reset_index(drop=True)
        x = np.arange(len(g))
        ok = ~g["flag_valid"].to_numpy()
        yerr = 2 * g["se_valid"].to_numpy() / g["exp_valid"].to_numpy()
        ax.errorbar(x[ok], g["OE_valid"].to_numpy()[ok], yerr=yerr[ok], fmt="o", color="tab:blue", capsize=2, ms=4)
        if (~ok).any():
            ax.errorbar(x[~ok], g["OE_valid"].to_numpy()[~ok], yerr=yerr[~ok], fmt="o", color="tab:red", capsize=2, ms=5)
        ax.axhline(1.0, color="grey", lw=1)
        ax.set_ylim(0.6, 1.4)
        ax.set_title(f, fontsize=8)
        ax.set_xticks(x)
        cat = binner.specs[f]["kind"] == "categorical"
        ax.set_xticklabels(list(g["bin"]) if cat else [str(i + 1) for i in x], rotation=90 if cat else 0, fontsize=5)
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle("Scorecard observed / expected by feature decile on validation (bars: 2 standard errors; red: flagged)", fontsize=10)
    fig.tight_layout()
    fig.savefig("reports/figures/calibration_by_feature.png", dpi=150)
    print("\nSaved reports/figures/calibration_by_feature.png")
except Exception as e:
    print(f"\nFigure skipped ({type(e).__name__}: {e}); results above are unaffected")

print("Saved reports/calibration_by_feature.csv and reports/capacity_candidates.csv")
