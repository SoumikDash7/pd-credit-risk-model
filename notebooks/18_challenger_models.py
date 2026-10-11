import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.woe import WoEBinner  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)

# ---------- rules fixed before the results were seen ----------
XGB_GRID = [{"max_depth": d, "min_child_weight": m} for d in (2, 3, 4) for m in (100, 500)]
XGB_FIXED = dict(learning_rate=0.05, n_estimators=2000, subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0,
                 tree_method="hist", eval_metric="logloss", early_stopping_rounds=50, n_jobs=-1, random_state=0)
RF_GRID = [{"min_samples_leaf": l, "max_features": f} for l in (300, 1000) for f in (0.3, 0.6)]
RF_FIXED = dict(n_estimators=200, max_samples=0.5, n_jobs=-1, random_state=0)
GAP_SMALL, GAP_LARGE = 0.01, 0.02       # AUC-gap materiality thresholds (own judgement, not standards)
N_BOOT, SEED = 200, 42
AUC_CHECK, AUC_TOL = 0.6708, 2e-4       # final scorecard validation AUC from Section 15

with open("models/scorecard_final.json", encoding="utf-8") as fh:
    SC = json.load(fh)
FEATS = SC["features"]
assert len(FEATS) == 16
binner = WoEBinner.load("models/woe_spec.json")
CAT = [f for f in FEATS if binner.specs[f]["kind"] == "categorical"]
NUM = [f for f in FEATS if f not in CAT]

df = pd.read_parquet("data/processed/features.parquet", columns=["issue_d", "split", "target"] + FEATS)
S = {k: df[df["split"] == k].reset_index(drop=True) for k in ["train", "valid", "oot"]}
Y = {k: v["target"].to_numpy().astype(int) for k, v in S.items()}
print("Rows: " + ", ".join(f"{k}={len(v):,}" for k, v in S.items()))

# ---------- design matrix: raw numerics (NaN kept) + one-hot categoricals (levels from train) ----------
LEVELS = {c: sorted(S["train"][c].astype(str).unique()) for c in CAT}


def design(d):
    X = d[NUM].astype("float64").copy()
    for c in CAT:
        s = d[c].astype(object).to_numpy()
        for lv in LEVELS[c]:
            X[f"{c}__{lv}"] = (s == lv).astype("float64")
    return X


X = {k: design(v) for k, v in S.items()}
COLS = list(X["train"].columns)
mono = [(1 if binner.specs[c]["direction"] == "increasing" else -1) if c in NUM else 0 for c in COLS]
MONO = "(" + ",".join(str(v) for v in mono) + ")"
print(f"Design matrix: {len(COLS)} columns ({len(NUM)} numeric, {len(COLS) - len(NUM)} one-hot)")
print("Monotone directions (+1 = predicted default rises with the feature): "
      + ", ".join(f"{c} {'+' if v > 0 else '-'}" for c, v in zip(COLS, mono) if v != 0) + "\n")


# ---------- helpers ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def ks_stat(y, p):
    d = pd.DataFrame({"p": p, "y": y})
    g = d.groupby("p")["y"].agg(["sum", "count"]).sort_index()
    bad = g["sum"].cumsum() / g["sum"].sum()
    good = (g["count"] - g["sum"]).cumsum() / (g["count"] - g["sum"]).sum()
    return float((bad - good).abs().max())


def boot_delta(y, p1, p0, B=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i]))
    return np.percentile(d, [2.5, 97.5])


def xgb_pred(m, k):
    return m.predict_proba(X[k], iteration_range=(0, m.best_iteration + 1))[:, 1]


# ---------- the scorecard, from the saved parameters ----------
SCN = "Scorecard (16 WoE features, logistic)"
Wk = {k: binner.transform(S[k], FEATS) for k in S}
beta = np.array([SC["coefficients"][f] for f in FEATS])
P = {SCN: {k: sigmoid(SC["intercept"] + Wk[k][FEATS].to_numpy() @ beta) for k in S}}
chk = roc_auc_score(Y["valid"], P[SCN]["valid"])
assert abs(chk - AUC_CHECK) < AUC_TOL, f"saved scorecard gives validation AUC {chk:.4f}, expected {AUC_CHECK}"
print(f"Saved scorecard reproduces validation AUC {chk:.4f} (Section 15: {AUC_CHECK})\n")

# ---------- XGBoost with monotone constraints: tune on validation ----------
tune, best = [], None
for cfg in XGB_GRID:
    t0 = time.time()
    m = xgb.XGBClassifier(**XGB_FIXED, **cfg, monotone_constraints=MONO)
    m.fit(X["train"], Y["train"], eval_set=[(X["valid"], Y["valid"])], verbose=False)
    pv = xgb_pred(m, "valid")
    row = {"model": "XGBoost (monotone)", **cfg, "trees": int(m.best_iteration + 1),
           "auc_valid": roc_auc_score(Y["valid"], pv), "logloss_valid": log_loss(Y["valid"], pv),
           "seconds": round(time.time() - t0)}
    tune.append(row)
    print(f"XGBoost depth {cfg['max_depth']}, min_child_weight {cfg['min_child_weight']}: {row['trees']} trees, "
          f"validation AUC {row['auc_valid']:.4f} ({row['seconds']}s)", flush=True)
    if best is None or row["auc_valid"] > best[0]:
        best = (row["auc_valid"], cfg, m)
xgb_cfg, xgb_m = best[1], best[2]
print(f"-> selected on validation AUC: {xgb_cfg}\n")

t0 = time.time()
xgb_u = xgb.XGBClassifier(**XGB_FIXED, **xgb_cfg)
xgb_u.fit(X["train"], Y["train"], eval_set=[(X["valid"], Y["valid"])], verbose=False)
print(f"XGBoost without monotone constraints (same settings): {xgb_u.best_iteration + 1} trees "
      f"({round(time.time() - t0)}s)\n", flush=True)

# ---------- random forest: tune on validation ----------
rf_best = None
for cfg in RF_GRID:
    t0 = time.time()
    rf = RandomForestClassifier(**RF_FIXED, **cfg).fit(X["train"], Y["train"])
    pv = rf.predict_proba(X["valid"])[:, 1]
    row = {"model": "Random forest", **cfg, "trees": RF_FIXED["n_estimators"],
           "auc_valid": roc_auc_score(Y["valid"], pv), "logloss_valid": log_loss(Y["valid"], np.clip(pv, 1e-6, 1 - 1e-6)),
           "seconds": round(time.time() - t0)}
    tune.append(row)
    print(f"Random forest min_samples_leaf {cfg['min_samples_leaf']}, max_features {cfg['max_features']}: "
          f"validation AUC {row['auc_valid']:.4f} ({row['seconds']}s)", flush=True)
    if rf_best is None or row["auc_valid"] > rf_best[0]:
        rf_best = (row["auc_valid"], cfg, rf)
rf_cfg, rf_m = rf_best[1], rf_best[2]
print(f"-> selected on validation AUC: {rf_cfg}\n")

XGN, XUN, RFN = "XGBoost (monotone constraints)", "XGBoost (no constraints, same settings)", "Random forest"
P[XGN] = {k: xgb_pred(xgb_m, k) for k in S}
P[XUN] = {k: xgb_pred(xgb_u, k) for k in S}
P[RFN] = {k: rf_m.predict_proba(X[k])[:, 1] for k in S}

# ---------- comparison ----------
disc, prob = [], []
for name, preds in P.items():
    d_row, p_row = {"model": name}, {"model": name}
    for k in S:
        a = roc_auc_score(Y[k], preds[k])
        d_row[f"auc_{k}"] = a
        if k != "train":
            d_row[f"gini_{k}"] = 2 * a - 1
            d_row[f"ks_{k}"] = ks_stat(Y[k], preds[k])
            p_row[f"logloss_{k}"] = log_loss(Y[k], np.clip(preds[k], 1e-6, 1 - 1e-6))
            p_row[f"brier_{k}"] = brier_score_loss(Y[k], preds[k])
            p_row[f"O/E_{k}"] = Y[k].mean() / preds[k].mean()
    disc.append(d_row)
    prob.append(p_row)
disc, prob = pd.DataFrame(disc), pd.DataFrame(prob)
print("=== Discrimination ===")
print(disc.round(4).to_string(index=False))
print("\n=== Probability quality (observed/expected is raw, with no recalibration shift applied to any model) ===")
print(prob.round(4).to_string(index=False))

# ---------- paired bootstrap: challenger minus scorecard ----------
drows = []
for name in [XGN, RFN]:
    for k in ["valid", "oot"]:
        a1, a0 = roc_auc_score(Y[k], P[name][k]), roc_auc_score(Y[k], P[SCN][k])
        lo, hi = boot_delta(Y[k], P[name][k], P[SCN][k])
        drows.append({"challenger": name, "split": k, "auc_challenger": a1, "auc_scorecard": a0,
                      "delta_auc": a1 - a0, "ci95_low": lo, "ci95_high": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)})
deltas = pd.DataFrame(drows)
print(f"\n=== AUC gap: challenger minus scorecard (paired bootstrap, {N_BOOT} resamples) ===")
print(deltas.round(4).to_string(index=False))

best_name = XGN if tune and best[0] >= rf_best[0] else RFN
gap = float(deltas.loc[deltas["challenger"] == best_name, "delta_auc"].max())
if gap < GAP_SMALL:
    verdict = "SMALL: the scorecard gives up little; no interaction search is needed"
elif gap < GAP_LARGE:
    verdict = "MODERATE: document the gap; an interaction search is optional"
else:
    verdict = "MATERIAL: investigate interactions before relying on the scorecard"
print(f"\nBest challenger on validation AUC: {best_name}. Larger of its validation and out-of-time AUC gaps: "
      f"{gap:.4f} (thresholds {GAP_SMALL} / {GAP_LARGE}) -> {verdict}")
du = roc_auc_score(Y["oot"], P[XUN]["oot"]) - roc_auc_score(Y["oot"], P[XGN]["oot"])
print(f"Cost of the monotone constraints: unconstrained minus constrained XGBoost AUC = "
      f"{roc_auc_score(Y['valid'], P[XUN]['valid']) - roc_auc_score(Y['valid'], P[XGN]['valid']):.4f} (validation), {du:.4f} (out-of-time)")

# ---------- what the boosted model uses ----------
gain = xgb_m.get_booster().get_score(importance_type="gain")
imp = {}
for col, g in gain.items():
    o = col.split("__")[0]
    imp[o] = imp.get(o, 0.0) + g
imp = pd.Series(imp)
imp = (imp / imp.sum() * 100).sort_values(ascending=False)
print("\n=== XGBoost (monotone) gain importance, % of total, aggregated by original feature ===")
print(imp.round(1).to_string())

# ---------- AUC by issue quarter on held-out loans ----------
q = pd.to_datetime(pd.concat([S["valid"]["issue_d"], S["oot"]["issue_d"]]), format="%b-%Y").dt.to_period("Q").astype(str).to_numpy()
yh = np.concatenate([Y["valid"], Y["oot"]])
qrows = []
for qq in sorted(set(q)):
    m = q == qq
    row = {"quarter": qq, "n": int(m.sum()), "bad": int(yh[m].sum())}
    for name in [SCN, XGN, RFN]:
        ph = np.concatenate([P[name]["valid"], P[name]["oot"]])
        row[name] = roc_auc_score(yh[m], ph[m]) if yh[m].sum() >= 100 and (1 - yh[m]).sum() >= 100 else np.nan
    qrows.append(row)
qt = pd.DataFrame(qrows)
print("\n=== AUC by issue quarter on held-out loans (validation rows through 2015Q2, out-of-time from 2015Q3) ===")
print(qt.round(4).to_string(index=False))

# ---------- outputs ----------
os.makedirs("reports/figures", exist_ok=True)
pd.DataFrame(tune).to_csv("reports/challenger_tuning.csv", index=False)
disc.merge(prob, on="model").to_csv("reports/challenger_metrics.csv", index=False)
deltas.to_csv("reports/challenger_deltas.csv", index=False)
qt.to_csv("reports/challenger_by_quarter.csv", index=False)
imp.round(3).to_csv("reports/challenger_xgb_importance.csv", header=["gain_pct"])
with open("models/challenger_params.json", "w", encoding="utf-8") as fh:
    json.dump({"features": FEATS, "columns": COLS, "levels": LEVELS, "monotone_constraints": MONO,
               "xgboost": {"fixed": {k: v for k, v in XGB_FIXED.items() if k != "n_jobs"}, "selected": xgb_cfg,
                           "best_n_trees": int(xgb_m.best_iteration + 1)},
               "random_forest": {"fixed": {k: v for k, v in RF_FIXED.items() if k != "n_jobs"}, "selected": rf_cfg}},
              fh, indent=1)

try:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(qt))
    for name, c in [(SCN, "black"), (XGN, "tab:blue"), (RFN, "tab:green")]:
        ax.plot(x, qt[name], marker="o", color=c, label=name)
    split_at = [i for i, v in enumerate(qt["quarter"]) if v == "2015Q3"]
    if split_at:
        ax.axvline(split_at[0] - 0.5, color="grey", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(list(qt["quarter"]), rotation=45)
    ax.set_ylabel("AUC on held-out loans")
    ax.set_title("Scorecard vs challengers by issue quarter (right of dotted line: out-of-time)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("reports/figures/challenger_by_quarter.png", dpi=150)
    print("\nSaved reports/figures/challenger_by_quarter.png")
except Exception as e:
    print(f"\nFigure skipped ({type(e).__name__}: {e}); results above are unaffected")

print("Saved reports/challenger_*.csv and models/challenger_params.json")
