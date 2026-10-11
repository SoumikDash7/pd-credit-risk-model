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
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.woe import WoEBinner  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)

# ---------- rules fixed before the results were seen ----------
N_SHAP, N_INT, SEED, N_BOOT = 20000, 5000, 42, 200
REVERSAL_FLAG = 0.05                      # log-odds, adjacent-bin move against the expected direction
LOAN_MATERIAL = 0.002                     # validation AUC gain for loan_amnt to count as an explanation
CHECK_MONO, CHECK_UNC, CHECK_TOL = 0.6787, 0.6839, 1e-3   # validation AUCs from Section 16

with open("models/scorecard_final.json", encoding="utf-8") as fh:
    SC = json.load(fh)
with open("models/challenger_params.json", encoding="utf-8") as fh:
    CP = json.load(fh)
FEATS = SC["features"]
assert CP["features"] == FEATS
COLS, LEVELS, MONO = CP["columns"], CP["levels"], CP["monotone_constraints"]
XCFG, XFIXED = CP["xgboost"]["selected"], CP["xgboost"]["fixed"]
binner = WoEBinner.load("models/woe_spec.json")
CAT = [f for f in FEATS if binner.specs[f]["kind"] == "categorical"]
NUM = [f for f in FEATS if f not in CAT]

df = pd.read_parquet("data/processed/features.parquet", columns=["issue_d", "split", "target", "loan_amnt"] + FEATS)
df = df[df["split"].isin(["train", "valid"])]
S = {k: df[df["split"] == k].reset_index(drop=True) for k in ["train", "valid"]}
Y = {k: v["target"].to_numpy().astype(int) for k, v in S.items()}
print("Out-of-time rows are not loaded or used in this script.")
print("Rows: " + ", ".join(f"{k}={len(v):,}" for k, v in S.items()) + "\n")


def design(d):
    X = d[NUM].astype("float64").copy()
    for c in CAT:
        s = d[c].astype(object).to_numpy()
        for lv in LEVELS[c]:
            X[f"{c}__{lv}"] = (s == lv).astype("float64")
    return X


X = {k: design(v) for k, v in S.items()}
assert list(X["train"].columns) == COLS, "design matrix differs from the one used in script 18"


def boot_delta(y, p1, p0, B=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i]))
    return np.percentile(d, [2.5, 97.5])


# ---------- rebuild the two XGBoost models and check they match Section 16 ----------
def fit_xgb(constrained):
    kw = dict(XFIXED, **XCFG, n_jobs=-1)
    if constrained:
        kw["monotone_constraints"] = MONO
    m = xgb.XGBClassifier(**kw)
    m.fit(X["train"], Y["train"], eval_set=[(X["valid"], Y["valid"])], verbose=False)
    return m


models = {"xgb_monotone": fit_xgb(True), "xgb_unconstrained": fit_xgb(False)}
for name, chk in [("xgb_monotone", CHECK_MONO), ("xgb_unconstrained", CHECK_UNC)]:
    m = models[name]
    a = roc_auc_score(Y["valid"], m.predict_proba(X["valid"], iteration_range=(0, m.best_iteration + 1))[:, 1])
    assert abs(a - chk) < CHECK_TOL, f"{name}: validation AUC {a:.4f}, expected {chk}"
    print(f"{name}: {m.best_iteration + 1} trees, validation AUC {a:.4f} (Section 16: {chk})")

# ---------- fixed validation samples ----------
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(len(S["valid"]), N_SHAP, replace=False))
perm = rng.permutation(N_SHAP)
Xs = X["valid"].iloc[idx].reset_index(drop=True)
Xi = Xs.iloc[perm[:N_INT]].reset_index(drop=True)
print(f"SHAP sample: {len(Xs):,} validation rows; interaction sample: {len(Xi):,} rows\n")

G = np.zeros((len(COLS), len(FEATS)))
for j, c in enumerate(COLS):
    G[j, FEATS.index(c.split("__")[0])] = 1.0


def contribs(m, Xdf):
    dm = xgb.DMatrix(Xdf)
    it = (0, m.best_iteration + 1)
    b = m.get_booster()
    c = b.predict(dm, pred_contribs=True, iteration_range=it)
    margin = b.predict(dm, output_margin=True, iteration_range=it)
    assert np.abs(c.sum(axis=1) - margin).max() < 1e-3, "contributions do not add up to the margin"
    return c[:, :len(COLS)]


R = {name: contribs(m, Xs) for name, m in models.items()}   # per-column contributions (log-odds)
print("Check passed: per-row contributions plus bias equal the model margin\n")

# ---------- 1. importance on one scale ----------
mu = binner.transform(S["train"], FEATS).mean()
beta = pd.Series(SC["coefficients"])[FEATS]
Csc = (binner.transform(S["valid"].iloc[idx], FEATS) - mu) * beta
Cm, Cu = R["xgb_monotone"] @ G, R["xgb_unconstrained"] @ G
imp = pd.DataFrame({"scorecard": np.abs(Csc.to_numpy()).mean(axis=0),
                    "xgb_monotone": np.abs(Cm).mean(axis=0),
                    "xgb_unconstrained": np.abs(Cu).mean(axis=0)}, index=FEATS)
for c in list(imp.columns):
    imp[f"rank_{c}"] = imp[c].rank(ascending=False).astype(int)
imp = imp.sort_values("xgb_unconstrained", ascending=False)
print("=== Mean absolute contribution to log-odds (validation sample); one-hot columns summed per row first ===")
print(imp.round(4).to_string())
print(f"\nSum of mean absolute contributions: scorecard {imp['scorecard'].sum():.3f}, "
      f"monotone XGBoost {imp['xgb_monotone'].sum():.3f}, unconstrained XGBoost {imp['xgb_unconstrained'].sum():.3f}")

# ---------- 2. non-monotone dependence ----------
dirs = {f: (1 if binner.specs[f]["direction"] == "increasing" else -1) for f in NUM}


def profile(f, name):
    v = Xs[f].to_numpy()
    ok = ~np.isnan(v)
    codes, _ = pd.qcut(v[ok], q=10, labels=False, duplicates="drop", retbins=True)
    contrib = R[name][:, COLS.index(f)][ok]
    prof = pd.Series(contrib).groupby(codes).mean().to_numpy()
    med = pd.Series(v[ok]).groupby(codes).median().to_numpy()
    return med, prof


def rev_stats(prof, d):
    rev = np.maximum(0.0, -d * np.diff(prof))
    return float(prof.max() - prof.min()), float(rev.max()), float(rev.sum())


mrows, profs = [], {}
for f in NUM:
    med, pm = profile(f, "xgb_monotone")
    _, pu = profile(f, "xgb_unconstrained")
    rm, revm, _ = rev_stats(pm, dirs[f])
    ru, revu, tot = rev_stats(pu, dirs[f])
    profs[f] = (med, pm, pu)
    mrows.append({"feature": f, "expected": "up" if dirs[f] > 0 else "down", "bins": len(pm),
                  "range_mono": rm, "max_reversal_mono": revm, "range_unc": ru,
                  "max_reversal_unc": revu, "total_reversal_unc": tot, "flag_unc": bool(revu > REVERSAL_FLAG)})
mono_df = pd.DataFrame(mrows).sort_values("max_reversal_unc", ascending=False).reset_index(drop=True)
print(f"\n=== Dependence by feature decile (mean contribution, log-odds); reversal = move against the scorecard direction; "
      f"flag if above {REVERSAL_FLAG} ===")
print(mono_df.round(4).to_string(index=False))
fl = mono_df.loc[mono_df["flag_unc"], "feature"].tolist()
print(f"\nFlagged in the unconstrained model: {fl or 'none'}. Noise floor (largest reversal in the monotone model): "
      f"{mono_df['max_reversal_mono'].max():.4f}")
for f in fl[:5]:
    med, pm, pu = profs[f]
    print(f"\n--- {f} (expected {'up' if dirs[f] > 0 else 'down'}) ---")
    print(pd.DataFrame({"decile_median": med, "monotone": pm, "unconstrained": pu}).round(4).to_string(index=False))

# ---------- 3. pairwise interactions ----------
def pair_table(m):
    dm = xgb.DMatrix(Xi)
    I = m.get_booster().predict(dm, pred_interactions=True, iteration_range=(0, m.best_iteration + 1))
    I = I[:, :len(COLS), :len(COLS)].astype("float64")
    A = np.einsum("rab,ag,bh->rgh", I, G, G, optimize=True)
    Pm = A + A.transpose(0, 2, 1)
    main = np.abs(np.einsum("rgg->rg", A)).mean(axis=0)
    pairs = {(FEATS[g], FEATS[h]): float(np.abs(Pm[:, g, h]).mean()) for g in range(len(FEATS)) for h in range(g + 1, len(FEATS))}
    return main, pairs


main_m, pairs_m = pair_table(models["xgb_monotone"])
main_u, pairs_u = pair_table(models["xgb_unconstrained"])
pt = pd.DataFrame([{"feature_a": a, "feature_b": b, "mean_abs_monotone": pairs_m[(a, b)],
                    "mean_abs_unconstrained": pairs_u[(a, b)]} for (a, b) in pairs_u])
pt = pt.sort_values("mean_abs_unconstrained", ascending=False).reset_index(drop=True)
pt["rank_unconstrained"] = np.arange(1, len(pt) + 1)
print(f"\n=== Top 10 pairwise interactions (mean absolute interaction effect, log-odds, {len(Xi):,} validation rows) ===")
print(pt.head(10).round(4).to_string(index=False))
share_m = sum(pairs_m.values()) / (main_m.sum() + sum(pairs_m.values())) * 100
share_u = sum(pairs_u.values()) / (main_u.sum() + sum(pairs_u.values())) * 100
print(f"\nRough share of attributed effect that sits in pairwise interactions: monotone {share_m:.1f}%, unconstrained {share_u:.1f}% "
      f"(sums of mean absolute values, so approximate)")
hp = pt[((pt["feature_a"] == "annual_inc") & (pt["feature_b"] == "loan_to_income")) |
        ((pt["feature_a"] == "loan_to_income") & (pt["feature_b"] == "annual_inc"))]
if len(hp):
    print(f"Hypothesis pair annual_inc x loan_to_income: rank {int(hp['rank_unconstrained'].iloc[0])} of {len(pt)}, "
          f"monotone {hp['mean_abs_monotone'].iloc[0]:.4f}, unconstrained {hp['mean_abs_unconstrained'].iloc[0]:.4f}")

# ---------- 4. loan-size test (validation only) ----------
ext = FEATS + ["loan_amnt"]
Wl = {k: binner.transform(S[k], ext) for k in S}


def fit_lr(cols):
    return sm.Logit(Y["train"], sm.add_constant(Wl["train"][cols], has_constant="add")).fit(disp=0, maxiter=100)


def pred_lr(res, cols):
    return np.asarray(res.predict(sm.add_constant(Wl["valid"][cols], has_constant="add")))


r0, r1 = fit_lr(FEATS), fit_lr(ext)
p0, p1 = pred_lr(r0, FEATS), pred_lr(r1, ext)
a0, a1 = roc_auc_score(Y["valid"], p0), roc_auc_score(Y["valid"], p1)
lo, hi = boot_delta(Y["valid"], p1, p0)
gap_unc = roc_auc_score(Y["valid"], models["xgb_unconstrained"].predict_proba(
    X["valid"], iteration_range=(0, models["xgb_unconstrained"].best_iteration + 1))[:, 1]) - a0
cl = Wl["train"][["loan_amnt", "loan_to_income", "annual_inc"]].corr()
coef, pval = float(r1.params["loan_amnt"]), float(r1.pvalues["loan_amnt"])
supported = bool((a1 - a0) >= LOAN_MATERIAL and lo > 0 and coef < 0)
print("\n=== Loan-size test: add the WoE of loan_amnt to the 16-feature scorecard (validation only) ===")
print(f"loan_amnt coefficient {coef:.3f} (p = {pval:.2g}); WoE correlation with loan_to_income {cl.loc['loan_amnt', 'loan_to_income']:.2f}, "
      f"with annual_inc {cl.loc['loan_amnt', 'annual_inc']:.2f}")
print(f"Validation AUC: {a0:.4f} -> {a1:.4f} (gain {a1 - a0:+.4f}, 95% interval {lo:+.4f} to {hi:+.4f})")
print(f"Gap between the unconstrained XGBoost and the scorecard on validation: {gap_unc:.4f}; "
      f"share explained by loan_amnt: {max(0.0, a1 - a0) / gap_unc * 100:.0f}%")
print(f"Rule (gain >= {LOAN_MATERIAL}, interval excludes 0, negative coefficient): "
      f"{'SUPPORTED: loan size explains part of the gap' if supported else 'NOT SUPPORTED as an explanation of the gap'}")

# ---------- outputs ----------
os.makedirs("reports/figures", exist_ok=True)
imp.to_csv("reports/shap_importance.csv")
mono_df.to_csv("reports/shap_monotonicity.csv", index=False)
pt.to_csv("reports/shap_interactions.csv", index=False)
pd.DataFrame([{"coef_loan_amnt_woe": coef, "p_value": pval, "auc_valid_base": a0, "auc_valid_with_loan_amnt": a1,
               "gain": a1 - a0, "ci95_low": lo, "ci95_high": hi, "gap_unconstrained_minus_scorecard": gap_unc,
               "supported": supported}]).to_csv("reports/loan_size_test.csv", index=False)

try:
    top = imp.index.tolist()
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    y = np.arange(len(top))
    h = 0.27
    for i, (c, col) in enumerate([("scorecard", "black"), ("xgb_monotone", "tab:blue"), ("xgb_unconstrained", "tab:orange")]):
        ax.barh(y + (i - 1) * h, imp.loc[top, c], h, color=col, label=c)
    ax.set_yticks(y)
    ax.set_yticklabels(top, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean absolute contribution to log-odds (validation sample)")
    ax.set_title("Feature importance on one scale")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("reports/figures/shap_importance.png", dpi=150)
    sel = mono_df["feature"].head(6).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for a, f in zip(axes.ravel(), sel):
        med, pm, pu = profs[f]
        a.plot(med, pm, marker="o", color="tab:blue", label="monotone")
        a.plot(med, pu, marker="o", color="tab:orange", label="unconstrained")
        a.set_title(f"{f} (expected {'up' if dirs[f] > 0 else 'down'})", fontsize=9)
        a.set_xlabel("decile median", fontsize=8)
        a.set_ylabel("mean contribution (log-odds)", fontsize=8)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Dependence of the two XGBoost models, six largest reversals", fontsize=10)
    fig.tight_layout()
    fig.savefig("reports/figures/shap_profiles.png", dpi=150)
    print("\nSaved reports/figures/shap_importance.png and shap_profiles.png")
except Exception as e:
    print(f"\nFigures skipped ({type(e).__name__}: {e}); results above are unaffected")

print("Saved reports/shap_importance.csv, shap_monotonicity.csv, shap_interactions.csv, loan_size_test.csv")
