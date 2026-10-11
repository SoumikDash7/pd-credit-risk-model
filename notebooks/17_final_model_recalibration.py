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
from src.woe import WoEBinner, psi  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)

# ---------- rules fixed before the results were seen ----------
DROP_DECISION = ["emp_length_yrs"]                 # Section 15: fair-lending / proxy risk
LOO_AUC_LOSS, LOO_P = 0.0005, 0.01                 # leave-one-out: drop only if BOTH hold
SLOPE_RANGE = (0.90, 1.10)                         # calibration slope considered close to 1
OE_LOW, OE_HIGH = 0.95, 1.05                       # Section 12.5 band
CAL_MONTHS = ["2015-07", "2015-08", "2015-09"]     # shift fitted here; the remaining OOT months are the test
N_BOOT, SEED = 100, 42

with open("models/selected_features.json", encoding="utf-8") as fh:
    F19 = json.load(fh)["selected"]
assert len(F19) == 19 and all(f in F19 for f in DROP_DECISION)
F18 = [f for f in F19 if f not in DROP_DECISION]
binner = WoEBinner.load("models/woe_spec.json")

df = pd.read_parquet("data/processed/features.parquet", columns=["issue_d", "split", "target"] + F19)
S = {k: df[df["split"] == k].reset_index(drop=True) for k in ["train", "valid", "oot"]}
Y = {k: v["target"].to_numpy().astype(int) for k, v in S.items()}
W = {k: binner.transform(v, F19) for k, v in S.items()}
print("Rows: " + ", ".join(f"{k}={len(v):,}" for k, v in S.items()))


# ---------- helpers ----------
def ks_stat(y, p):
    d = pd.DataFrame({"p": p, "y": y})
    g = d.groupby("p")["y"].agg(["sum", "count"]).sort_index()
    bad = g["sum"].cumsum() / g["sum"].sum()
    good = (g["count"] - g["sum"]).cumsum() / (g["count"] - g["sum"]).sum()
    return float((bad - good).abs().max())


def perf(y, p):
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


def logit(p):
    q = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calib(y, p):
    lp = logit(p)
    a, b = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params
    shift = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0])
    return {"n": int(len(y)), "obs": float(y.mean()), "pred": float(p.mean()), "O/E": float(y.mean() / p.mean()),
            "intercept": float(a), "slope": float(b), "shift_logodds": shift}


# ---------- 1. 18-feature model and leave-one-out on validation ----------
full = fit(F18)
p18 = {k: predict(full, F18, k) for k in S}
auc_v, ks_v = roc_auc_score(Y["valid"], p18["valid"]), ks_stat(Y["valid"], p18["valid"])
print(f"\n=== {len(F18)}-feature model (emp_length_yrs dropped by decision): validation AUC {auc_v:.4f}, KS {ks_v:.4f} ===")

rows = []
for f in F18:
    feats = [c for c in F18 if c != f]
    r = fit(feats)
    pv = predict(r, feats, "valid")
    lo, hi = boot_delta(Y["valid"], p18["valid"], pv)
    loss = auc_v - roc_auc_score(Y["valid"], pv)
    rows.append({"feature": f, "coef": float(full.params[f]), "p_value": float(full.pvalues[f]),
                 "auc_loss_valid": loss, "ks_loss_valid": ks_v - ks_stat(Y["valid"], pv),
                 "ci95_low": lo, "ci95_high": hi,
                 "drop_by_rule": bool(loss < LOO_AUC_LOSS and full.pvalues[f] > LOO_P)})
loo = pd.DataFrame(rows).sort_values("auc_loss_valid").reset_index(drop=True)
show = loo.copy()
show["p_value"] = show["p_value"].map(lambda v: f"{v:.2g}")
print(f"\nLeave-one-out on validation (refit on train without each feature; {N_BOOT} bootstrap resamples). "
      f"Rule: drop if AUC loss < {LOO_AUC_LOSS} AND p > {LOO_P}")
print(show.round(4).to_string(index=False))

DROP2 = loo.loc[loo["drop_by_rule"], "feature"].tolist()
FINAL = [f for f in F18 if f not in DROP2]
print(f"\nDropped by the leave-one-out rule: {DROP2 or 'none'}")
print(f"Final feature set: {len(FINAL)} features")

# ---------- 2. final model ----------
final = fit(FINAL)
P = {"f19": {k: predict(fit(F19), F19, k) for k in S}, "f18": p18,
     "final": {k: predict(final, FINAL, k) for k in S}}
coef = pd.DataFrame({"coef": final.params[FINAL], "std_err": final.bse[FINAL], "z": final.tvalues[FINAL],
                     "p_value": final.pvalues[FINAL]})
coef["negative"] = coef["coef"] < 0
cs = coef.copy()
cs["p_value"] = cs["p_value"].map(lambda v: f"{v:.2g}")
print(f"\n=== Final model coefficients (intercept {final.params['const']:.4f}; converged: "
      f"{final.mle_retvals.get('converged')}) ===")
print(cs.round(4).to_string())
print(f"Negative coefficients: {int(coef['negative'].sum())} of {len(FINAL)}; "
      f"largest p-value {coef['p_value'].max():.3g}")

perf_rows = []
for name, preds in [("19 features (Section 14)", P["f19"]), ("18 features (emp_length_yrs dropped)", P["f18"]),
                    (f"Final ({len(FINAL)} features)", P["final"])]:
    row = {"model": name}
    for k in S:
        a, g, ks = perf(Y[k], preds[k])
        row.update({f"auc_{k}": a, f"gini_{k}": g, f"ks_{k}": ks})
    perf_rows.append(row)
perf_df = pd.DataFrame(perf_rows)
print("\n=== Discrimination ===")
print(perf_df.round(4).to_string(index=False))

edges = np.quantile(P["final"]["train"], np.linspace(0, 1, 11))[1:-1]
shares = {k: np.bincount(np.searchsorted(edges, P["final"][k], side="right"), minlength=10) / len(Y[k]) for k in S}
print(f"\nScore PSI against train (decile bins; 0.10 = watch, 0.25 = act): validation {psi(shares['train'], shares['valid']):.4f}, "
      f"out-of-time {psi(shares['train'], shares['oot']):.4f}")

# ---------- 3. recalibration (Section 12.5) ----------
p_oot, y_oot = P["final"]["oot"], Y["oot"]
mon = pd.to_datetime(S["oot"]["issue_d"], format="%b-%Y").dt.to_period("M").astype(str).to_numpy()
cal_mask = np.isin(mon, CAL_MONTHS)
test_mask = ~cal_mask
assert cal_mask.sum() == 73569 and test_mask.sum() == 111343, f"window sizes {cal_mask.sum()}, {test_mask.sum()}"

c_cal = calib(y_oot[cal_mask], p_oot[cal_mask])
slope_ok = SLOPE_RANGE[0] <= c_cal["slope"] <= SLOPE_RANGE[1]
if slope_ok:
    method = "intercept shift"
    recal = lambda p: sigmoid(logit(p) + c_cal["shift_logodds"])
else:
    method = "intercept and slope"
    recal = lambda p: sigmoid(c_cal["intercept"] + c_cal["slope"] * logit(p))
p_re = recal(p_oot)
print(f"\n=== Recalibration ({int(cal_mask.sum()):,} calibration loans Jul-Sep 2015; {int(test_mask.sum()):,} test loans Oct 2015 - Jan 2016) ===")
print(f"Calibration-window slope {c_cal['slope']:.3f} (rule: {SLOPE_RANGE[0]}-{SLOPE_RANGE[1]}) -> method: {method}")
print(f"Shift in log-odds fitted on the calibration window: {c_cal['shift_logodds']:.4f}")

auc_b, auc_a = roc_auc_score(y_oot[test_mask], p_oot[test_mask]), roc_auc_score(y_oot[test_mask], p_re[test_mask])
assert abs(auc_b - auc_a) < 1e-9, "recalibration must not change the ranking"
print(f"AUC on the test window before / after: {auc_b:.4f} / {auc_a:.4f} (unchanged, as required)")

cal_rows = []
for window, y, pb, pa in [("validation", Y["valid"], P["final"]["valid"], None),
                          ("oot_all", y_oot, p_oot, p_re),
                          ("oot_calibration (Jul-Sep 2015)", y_oot[cal_mask], p_oot[cal_mask], p_re[cal_mask]),
                          ("oot_test (Oct 2015 - Jan 2016)", y_oot[test_mask], p_oot[test_mask], p_re[test_mask])]:
    for stage, p in [("before", pb), ("after", pa)]:
        if p is not None:
            c = calib(y, p)
            c.update({"window": window, "stage": stage})
            cal_rows.append(c)
cal = pd.DataFrame(cal_rows)[["window", "stage", "n", "obs", "pred", "O/E", "intercept", "slope", "shift_logodds"]]
print("\n" + cal.round(4).to_string(index=False))

oe_after = cal.loc[(cal["window"].str.startswith("oot_test")) & (cal["stage"] == "after"), "O/E"].iloc[0]
oe_before = cal.loc[(cal["window"].str.startswith("oot_test")) & (cal["stage"] == "before"), "O/E"].iloc[0]
passed = OE_LOW <= oe_after <= OE_HIGH
print(f"\nSection 12.5 test: test-window O/E {oe_before:.3f} before -> {oe_after:.3f} after "
      f"(band {OE_LOW}-{OE_HIGH}): {'PASS' if passed else 'FAIL'}")

mm = pd.DataFrame({"month": mon, "y": y_oot, "before": p_oot, "after": p_re}).groupby("month").agg(
    n=("y", "size"), obs=("y", "mean"), pred_before=("before", "mean"), pred_after=("after", "mean"))
mm["O/E_before"] = mm["obs"] / mm["pred_before"]
mm["O/E_after"] = mm["obs"] / mm["pred_after"]
mm["window"] = ["calibration" if m in CAL_MONTHS else "test" for m in mm.index]
msh = mm.copy()
for c in ["obs", "pred_before", "pred_after"]:
    msh[c] = (msh[c] * 100).round(2)
print("\n=== Out-of-time by issue month (bad rates in %) ===")
print(msh.round(3).to_string())

dd = np.searchsorted(edges, p_oot[test_mask], side="right") + 1
dt = pd.DataFrame({"d": dd, "y": y_oot[test_mask], "b": p_oot[test_mask], "a": p_re[test_mask]}).groupby("d").agg(
    n=("y", "size"), obs=("y", "mean"), pred_before=("b", "mean"), pred_after=("a", "mean"))
dsh = dt.copy()
for c in ["obs", "pred_before", "pred_after"]:
    dsh[c] = (dsh[c] * 100).round(2)
print("\n=== Test window by score decile (decile 1 = lowest predicted PD; bad rates in %) ===")
print(dsh.to_string())

c_full = calib(y_oot, p_oot)
print(f"\nFor deployment (untested): shift refitted on all out-of-time loans = {c_full['shift_logodds']:.4f} log-odds "
      f"(intercept/slope fit: {c_full['intercept']:.4f} / {c_full['slope']:.4f})")

# ---------- outputs ----------
os.makedirs("reports/figures", exist_ok=True)
loo.to_csv("reports/loo_validation.csv", index=False)
coef.to_csv("reports/final_model_coefficients.csv")
perf_df.to_csv("reports/final_model_metrics.csv", index=False)
cal.to_csv("reports/recalibration.csv", index=False)
mm.to_csv("reports/recalibration_monthly.csv")
dt.to_csv("reports/recalibration_deciles.csv")


def params_of(c):
    return {"shift_logodds": c["shift_logodds"]} if slope_ok else {"intercept": c["intercept"], "slope": c["slope"]}


with open("models/scorecard_final.json", "w", encoding="utf-8") as fh:
    json.dump({"features": FINAL, "intercept": float(final.params["const"]),
               "coefficients": {f: float(final.params[f]) for f in FINAL}, "woe_spec": "models/woe_spec.json",
               "convention": "WoE = ln(good/bad); target 1 = bad; PD = 1 / (1 + exp(-(b0 + sum(b_i * WoE_i))))",
               "recalibration": {"method": method,
                                 "tested_design": "fitted on Jul-Sep 2015, tested on Oct 2015 - Jan 2016",
                                 "tested_parameters": params_of(c_cal), "test_O/E_before": float(oe_before),
                                 "test_O/E_after": float(oe_after), "passed_band": bool(passed),
                                 "deployment_parameters_untested": params_of(c_full)}}, fh, indent=1)

try:
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x, w = np.arange(len(mm)), 0.38
    ax[0].bar(x - w / 2, mm["O/E_before"], w, color="tab:red", alpha=0.75, label="before shift")
    ax[0].bar(x + w / 2, mm["O/E_after"], w, color="tab:blue", alpha=0.75, label="after shift")
    ax[0].axhspan(OE_LOW, OE_HIGH, color="green", alpha=0.15, label=f"tolerance {OE_LOW}-{OE_HIGH}")
    ax[0].axvline(len(CAL_MONTHS) - 0.5, color="black", ls=":", lw=1)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(list(mm.index), rotation=45)
    ax[0].set_ylim(0.85, max(1.2, float(mm["O/E_before"].max()) * 1.05))
    ax[0].set_ylabel("observed / expected")
    ax[0].set_title("Out-of-time O/E by month (left of dotted line: calibration window)")
    ax[0].legend(fontsize=8)
    lim = float(max(dt["obs"].max(), dt["pred_before"].max(), dt["pred_after"].max())) * 100 * 1.08
    ax[1].plot([0, lim], [0, lim], color="grey", lw=1, ls="--", label="perfect calibration")
    ax[1].plot(dt["pred_before"] * 100, dt["obs"] * 100, marker="o", color="tab:red", label="before shift")
    ax[1].plot(dt["pred_after"] * 100, dt["obs"] * 100, marker="o", color="tab:blue", label="after shift")
    ax[1].set_xlabel("mean predicted bad rate (%)")
    ax[1].set_ylabel("observed bad rate (%)")
    ax[1].set_title("Test window (Oct 2015 - Jan 2016) by score decile")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("reports/figures/final_calibration.png", dpi=150)
    print("\nSaved reports/figures/final_calibration.png")
except Exception as e:
    print(f"\nFigure skipped ({type(e).__name__}: {e}); results above are unaffected")

print("Saved reports/loo_validation.csv, final_model_*.csv, recalibration*.csv and models/scorecard_final.json")
