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
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)

OE_LOW, OE_HIGH = 0.95, 1.05   # tolerance band from Section 12.5
N_BOOT, SEED = 200, 42
EXTRA = "payment_to_income"    # only used in ablation B

with open("models/selected_features.json", encoding="utf-8") as fh:
    FEATS = json.load(fh)["selected"]
assert len(FEATS) == 19 and "loan_to_income" in FEATS and EXTRA not in FEATS, "unexpected feature set"
binner = WoEBinner.load("models/woe_spec.json")
assert EXTRA in binner.specs

df = pd.read_parquet("data/processed/features.parquet",
                     columns=["issue_d", "split", "target", "bench_int_rate"] + FEATS + [EXTRA])
S = {k: df[df["split"] == k].reset_index(drop=True) for k in ["train", "valid", "oot"]}
Y = {k: v["target"].to_numpy().astype(int) for k, v in S.items()}
W = {k: binner.transform(v, FEATS + [EXTRA]) for k, v in S.items()}
print("Rows: " + ", ".join(f"{k}={len(v):,}" for k, v in S.items()))
print("Bad rate: " + ", ".join(f"{k}={Y[k].mean() * 100:.2f}%" for k in S) + "\n")


# ---------- helpers ----------
def ks_stat(y, p):
    d = pd.DataFrame({"p": p, "y": y})
    g = d.groupby("p")["y"].agg(["sum", "count"]).sort_index()
    bad = g["sum"].cumsum() / g["sum"].sum()
    good = (g["count"] - g["sum"]).cumsum() / (g["count"] - g["sum"]).sum()
    return float((bad - good).abs().max())


def perf(y, p):
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    auc = roc_auc_score(y, p)
    return auc, 2 * auc - 1, ks_stat(y, p)


def fit(feats):
    return sm.Logit(Y["train"], sm.add_constant(W["train"][feats], has_constant="add")).fit(disp=0, maxiter=100)


def predict(res, feats, split):
    return np.asarray(res.predict(sm.add_constant(W[split][feats], has_constant="add")))


def boot_delta(y, p1, p0, B=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i]))
    return np.percentile(d, [2.5, 97.5])


def calib(y, p):
    q = np.clip(p, 1e-12, 1 - 1e-12)
    lp = np.log(q / (1 - q))
    a, b = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params
    shift = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0])
    return {"obs": y.mean(), "pred": p.mean(), "O/E": y.mean() / p.mean(),
            "intercept": float(a), "slope": float(b), "shift_logodds": shift}


# ---------- 1. base scorecard ----------
base = fit(FEATS)
print(f"=== Base scorecard: logistic regression on {len(FEATS)} WoE features (train) ===")
print(f"Converged: {base.mle_retvals.get('converged')} | train mean predicted {np.mean(predict(base, FEATS, 'train')) * 100:.3f}% "
      f"vs observed {Y['train'].mean() * 100:.3f}% (equal at the MLE with an intercept)\n")

uni = {}
for f in FEATS:
    uni[f] = sm.Logit(Y["train"], sm.add_constant(W["train"][[f]], has_constant="add")).fit(disp=0)
iv = pd.read_csv("reports/iv_summary.csv").set_index("feature")["iv_train"]
coef = pd.DataFrame({"coef": base.params[FEATS], "std_err": base.bse[FEATS], "z": base.tvalues[FEATS],
                     "p_value": base.pvalues[FEATS]})
coef["univariate_coef"] = [float(uni[f].params[f]) for f in FEATS]
coef["iv_train"] = [float(iv[f]) for f in FEATS]
coef["sign_ok"] = coef["coef"] < 0
print(f"Intercept: {base.params['const']:.4f}")
show = coef.copy()
show["p_value"] = show["p_value"].map(lambda v: f"{v:.2g}")
print(show.round(4).to_string())
pos = coef.index[~coef["sign_ok"]].tolist()
print(f"\nSign check (WoE = ln(good/bad), target 1 = bad, so every coefficient should be negative): "
      f"{int(coef['sign_ok'].sum())} of {len(FEATS)} negative")
print(f"Coefficients with p > 0.05: {coef.index[coef['p_value'] > 0.05].tolist() or 'none'}")

if pos:
    corr = pd.read_csv("reports/woe_corr_train.csv", index_col=0)
    for f in pos:
        rest = [c for c in FEATS if c != f]
        top = corr.loc[f, rest].sort_values(key=np.abs, ascending=False).head(4)
        r_wo = fit(rest)
        a1 = roc_auc_score(Y["valid"], predict(base, FEATS, "valid"))
        a0 = roc_auc_score(Y["valid"], predict(r_wo, rest, "valid"))
        print(f"\nPositive coefficient diagnostic: {f} (multivariate {coef.loc[f, 'coef']:.3f}, "
              f"univariate {coef.loc[f, 'univariate_coef']:.3f})")
        print("  most correlated kept features (r): " + ", ".join(f"{k} {v:.2f}" for k, v in top.items()))
        print(f"  validation AUC {a1:.4f} with it, {a0:.4f} without it")

# ---------- 2. performance ----------
P = {"base": {k: predict(base, FEATS, k) for k in S}}
perf_rows = []


def add_model(name, preds):
    row = {"model": name}
    for k in S:
        auc, gini, ks = perf(Y[k], preds[k])
        row.update({f"auc_{k}": auc, f"gini_{k}": gini, f"ks_{k}": ks})
    perf_rows.append(row)


add_model("Scorecard (19 features)", P["base"])
add_model("fico_mean WoE alone", {k: np.asarray(uni["fico_mean"].predict(
    sm.add_constant(W[k][["fico_mean"]], has_constant="add"))) for k in S})
add_model("Benchmark: int_rate (Section 10)", {k: S[k]["bench_int_rate"].to_numpy(dtype=float) for k in S})

# ---------- 3. ablations ----------
no_emp = [f for f in FEATS if f != "emp_length_yrs"]
pti = [EXTRA if f == "loan_to_income" else f for f in FEATS]
variants = {"A: without emp_length_yrs": (no_emp, "emp_length_yrs excluded"),
            "B: payment_to_income instead of loan_to_income": (pti, "payment_to_income in place of loan_to_income")}
abl_rows = []
for name, (feats, note) in variants.items():
    res = fit(feats)
    P[name] = {k: predict(res, feats, k) for k in S}
    add_model(name, P[name])
    for k in ["valid", "oot"]:
        d = roc_auc_score(Y[k], P["base"][k]) - roc_auc_score(Y[k], P[name][k])
        lo, hi = boot_delta(Y[k], P["base"][k], P[name][k])
        abl_rows.append({"variant": name, "split": k, "auc_base": roc_auc_score(Y[k], P["base"][k]),
                         "auc_variant": roc_auc_score(Y[k], P[name][k]), "delta_auc_base_minus_variant": d,
                         "ci95_low": lo, "ci95_high": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)})
    if name.startswith("B"):
        print(f"\nAblation B: coefficient of {EXTRA} = {res.params[EXTRA]:.3f} (p = {res.pvalues[EXTRA]:.2g}); "
              f"loan_to_income in base = {base.params['loan_to_income']:.3f}")
perf_df = pd.DataFrame(perf_rows)
abl = pd.DataFrame(abl_rows)

print("\n=== Discrimination (AUC / Gini / KS) ===")
print(perf_df.round(4).to_string(index=False))
print("\n=== Ablations: delta AUC = base minus variant (positive = base is better); paired bootstrap, "
      f"{N_BOOT} resamples ===")
print(abl.round(4).to_string(index=False))

# ---------- 4. calibration (base scorecard) ----------
cal_rows = []
for k in S:
    c = calib(Y[k], P["base"][k])
    c["split"] = k
    c["within_band"] = bool(OE_LOW <= c["O/E"] <= OE_HIGH)
    cal_rows.append(c)
cal = pd.DataFrame(cal_rows)[["split", "obs", "pred", "O/E", "intercept", "slope", "shift_logodds", "within_band"]]
print(f"\n=== Calibration of the base scorecard (O/E band {OE_LOW}-{OE_HIGH} from Section 12.5) ===")
print(cal.round(4).to_string(index=False))

edges = np.quantile(P["base"]["train"], np.linspace(0, 1, 11))[1:-1]
dec = {}
for k in S:
    dd = np.searchsorted(edges, P["base"][k], side="right") + 1
    dec[k] = pd.DataFrame({"d": dd, "y": Y[k], "p": P["base"][k]}).groupby("d").agg(
        n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
dtab = pd.concat({k: dec[k] for k in S}, axis=1)
dtab.columns = [f"{b}_{a}" for a, b in dtab.columns]
show = dtab.copy()
for c in show.columns:
    if c.startswith(("obs", "pred")):
        show[c] = (show[c] * 100).round(2)
print("\n=== Observed vs predicted bad rate (%) by score decile (1 = lowest predicted PD; cut points from train) ===")
print(show.to_string())
rev = int((np.diff(dec["oot"]["obs"].to_numpy()) < 0).sum())
print(f"\nOut-of-time: observed bad rate falls between adjacent deciles {rev} time(s) out of 9")

m = pd.DataFrame({"month": pd.to_datetime(S["oot"]["issue_d"], format="%b-%Y").dt.to_period("M"),
                  "y": Y["oot"], "p": P["base"]["oot"]})
mt = m.groupby("month").agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
mt["O/E"] = mt["obs"] / mt["pred"]
print("\n=== Out-of-time by issue month (descriptive only; no model choice is based on it) ===")
print(mt.assign(obs=(mt["obs"] * 100).round(2), pred=(mt["pred"] * 100).round(2), **{"O/E": mt["O/E"].round(3)}).to_string())

# ---------- outputs ----------
os.makedirs("reports/figures", exist_ok=True)
coef.to_csv("reports/scorecard_coefficients.csv")
perf_df.to_csv("reports/scorecard_metrics.csv", index=False)
abl.to_csv("reports/scorecard_ablations.csv", index=False)
cal.to_csv("reports/scorecard_calibration.csv", index=False)
dtab.to_csv("reports/scorecard_deciles.csv")
with open("models/scorecard_params.json", "w", encoding="utf-8") as fh:
    json.dump({"features": FEATS, "intercept": float(base.params["const"]),
               "coefficients": {f: float(base.params[f]) for f in FEATS}, "woe_spec": "models/woe_spec.json",
               "convention": "WoE = ln(good/bad); target 1 = bad; PD = 1 / (1 + exp(-(b0 + sum(b_i * WoE_i))))"}, fh, indent=1)

try:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    lim = float(max(dec["oot"]["obs"].max(), dec["oot"]["pred"].max(), dec["valid"]["obs"].max())) * 100 * 1.08
    ax[0].plot([0, lim], [0, lim], color="grey", lw=1, ls="--", label="perfect calibration")
    for k, c in [("valid", "tab:blue"), ("oot", "tab:red")]:
        ax[0].plot(dec[k]["pred"] * 100, dec[k]["obs"] * 100, marker="o", color=c, label=k)
    ax[0].set_xlabel("mean predicted bad rate (%)")
    ax[0].set_ylabel("observed bad rate (%)")
    ax[0].set_title("Calibration by score decile")
    ax[0].legend()
    x = np.arange(len(mt))
    ax[1].bar(x, mt["O/E"].to_numpy(), color="tab:red", alpha=0.7)
    ax[1].axhspan(OE_LOW, OE_HIGH, color="green", alpha=0.15, label=f"tolerance {OE_LOW}-{OE_HIGH}")
    ax[1].axhline(1.0, color="grey", lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([str(i) for i in mt.index], rotation=45)
    ax[1].set_ylim(0.8, max(1.2, float(mt["O/E"].max()) * 1.05))
    ax[1].set_ylabel("observed / expected")
    ax[1].set_title("Out-of-time O/E by issue month")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("reports/figures/scorecard_calibration.png", dpi=150)
    print("\nSaved reports/figures/scorecard_calibration.png")
except Exception as e:
    print(f"\nFigure skipped ({type(e).__name__}: {e}); results above are unaffected")

print("Saved reports/scorecard_*.csv and models/scorecard_params.json")
