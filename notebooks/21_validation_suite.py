import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.woe import WoEBinner, psi  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 40)

# ---------- rules fixed before the results were seen ----------
REJECT_SHARE = 0.20                                  # stated cut-off: reject the riskiest 20% of validation applicants
OE_LOW, OE_HIGH = 0.95, 1.05                         # Section 12.5 band
CAL_MONTHS = ["2015-07", "2015-08", "2015-09"]       # window the shift was fitted on (= 2015Q3)
PERSIST_FRACTION = 0.5                               # a segment misfit persists if same side of 1 and at least half as large
AUC_CHECK, AUC_TOL = {"valid": 0.6708, "oot": 0.6788}, 2e-4
EXPECTED_N = {"annual_inc top decile": 8476, "bc_open_to_buy lowest bin": 7589, "bc_util lowest bin": 7592,
              "bc_util highest bin": 7689, "purpose home_improvement": 4086, "purpose major_purchase": 1319,
              "purpose small_business": 716, "control: mo_sin_rcnt_tl 0-1": 7633, "control: mo_sin_rcnt_tl 7-8": 9328}

with open("models/scorecard_final.json", encoding="utf-8") as fh:
    SC = json.load(fh)
FEATS = SC["features"]
assert len(FEATS) == 16
rc = SC["recalibration"]
assert rc["method"] == "intercept shift"
SHIFT = float(rc["tested_parameters"]["shift_logodds"])
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


def ks_stat(y, p):
    d = pd.DataFrame({"p": p, "y": y})
    g = d.groupby("p")["y"].agg(["sum", "count"]).sort_index()
    bad = g["sum"].cumsum() / g["sum"].sum()
    good = (g["count"] - g["sum"]).cumsum() / (g["count"] - g["sum"]).sum()
    return float((bad - good).abs().max())


beta = np.array([SC["coefficients"][f] for f in FEATS])
P = {k: sigmoid(SC["intercept"] + binner.transform(S[k], FEATS)[FEATS].to_numpy() @ beta) for k in S}
Ps = {k: sigmoid(logit(P[k]) + SHIFT) for k in S}      # shifted PD; used for out-of-time only
for k in ["valid", "oot"]:
    a = roc_auc_score(Y[k], P[k])
    assert abs(a - AUC_CHECK[k]) < AUC_TOL, f"{k}: AUC {a:.4f}, expected {AUC_CHECK[k]}"
mon = pd.to_datetime(S["oot"]["issue_d"], format="%b-%Y").dt.to_period("M").astype(str).to_numpy()
cal_mask = np.isin(mon, CAL_MONTHS)
test_mask = ~cal_mask
assert cal_mask.sum() == 73569 and test_mask.sum() == 111343
oe_test = Y["oot"][test_mask].mean() / Ps["oot"][test_mask].mean()
assert abs(oe_test - 1.044) < 0.0015, f"test-window O/E after shift is {oe_test:.4f}, expected 1.044"
print(f"Saved model reproduces validation AUC {roc_auc_score(Y['valid'], P['valid']):.4f} and out-of-time AUC "
      f"{roc_auc_score(Y['oot'], P['oot']):.4f}; shift {SHIFT:.4f} log-odds; test-window O/E after shift {oe_test:.3f}")
print(f"Test window (Oct 2015 - Jan 2016): {int(test_mask.sum()):,} loans; the shift was fitted on the {int(cal_mask.sum()):,} loans of Jul-Sep 2015\n")

# ================= A. lift and gain =================
edges_tr = np.quantile(P["train"], np.linspace(0, 1, 11))[1:-1]


def decile_table(y, p):
    d = np.searchsorted(edges_tr, p, side="right") + 1
    t = pd.DataFrame({"d": d, "y": y}).groupby("d").agg(n=("y", "size"), bad=("y", "sum")).reindex(range(1, 11), fill_value=0)
    overall = y.mean()
    t["bad_rate_pct"] = t["bad"] / t["n"] * 100
    t["lift"] = t["bad"] / t["n"] / overall
    r = t.loc[list(range(10, 0, -1))].copy()
    r["cum_loans_pct"] = r["n"].cumsum() / r["n"].sum() * 100
    r["cum_bad_pct"] = r["bad"].cumsum() / r["bad"].sum() * 100
    r["cum_lift"] = r["cum_bad_pct"] / r["cum_loans_pct"]
    return r


DT = {}
rows = []
print("=== A. Lift and gain by score decile (decile 10 = riskiest; cut points from train; ranking is unaffected by the shift) ===")
for k, label in [("valid", "validation"), ("oot", "out-of-time")]:
    DT[k] = decile_table(Y[k], P[k])
    print(f"\n{label}: AUC {roc_auc_score(Y[k], P[k]):.4f}, Gini {2 * roc_auc_score(Y[k], P[k]) - 1:.4f}, KS {ks_stat(Y[k], P[k]):.4f}, "
          f"overall bad rate {Y[k].mean() * 100:.2f}%")
    print(DT[k].round(2).to_string())
    x = DT[k].reset_index().assign(split=label)
    rows.append(x)
pd.concat(rows).to_csv("reports/validation_deciles.csv", index=False)

# ================= B. confusion matrix at a stated cut-off =================
thr = float(np.quantile(P["valid"], 1 - REJECT_SHARE))
print(f"\n=== B. Confusion matrix: reject if predicted PD >= {thr:.4f} (the 80th percentile of validation PD, a rejection share of "
      f"{REJECT_SHARE:.0%}; equivalent to {sigmoid(logit(thr) + SHIFT):.4f} on the shifted scale) ===")


def conf(y, reject, name):
    tp, fp = int((reject & (y == 1)).sum()), int((reject & (y == 0)).sum())
    fn, tn = int((~reject & (y == 1)).sum()), int((~reject & (y == 0)).sum())
    return {"case": name, "n": len(y), "rejected_pct": reject.mean() * 100, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision_pct": tp / (tp + fp) * 100, "recall_pct": tp / (tp + fn) * 100, "specificity_pct": tn / (tn + fp) * 100,
            "bad_rate_all_pct": y.mean() * 100, "bad_rate_approved_pct": fn / (fn + tn) * 100,
            "reduction_in_bad_rate_pct": (1 - (fn / (fn + tn)) / y.mean()) * 100}


thr_oot_share = float(np.quantile(P["oot"], 1 - REJECT_SHARE))
cm = pd.DataFrame([conf(Y["valid"], P["valid"] >= thr, "validation, threshold from validation"),
                   conf(Y["oot"], P["oot"] >= thr, "out-of-time, same PD threshold"),
                   conf(Y["oot"], P["oot"] >= thr_oot_share, "out-of-time, same 20% rejection share")])
print(cm.round(2).to_string(index=False))
print(f"(Out-of-time threshold for a 20% share: {thr_oot_share:.4f}, against {thr:.4f} on validation.)")
cm.to_csv("reports/validation_confusion.csv", index=False)

# ================= C. segment back-tests =================
def decile_edges(series):
    a = series.to_numpy(float)
    return np.unique(np.quantile(a[~np.isnan(a)], np.linspace(0, 1, 11))[1:-1])


E = {f: decile_edges(S["valid"][f]) for f in ["annual_inc", "bc_open_to_buy", "bc_util"]}


def seg_masks(d):
    ai, bo = d["annual_inc"].to_numpy(float), d["bc_open_to_buy"].to_numpy(float)
    bu, ms = d["bc_util"].to_numpy(float), d["mo_sin_rcnt_tl"].to_numpy(float)
    pur = d["purpose"].astype(str).to_numpy()
    return {"annual_inc top decile": ai >= E["annual_inc"][-1],
            "bc_open_to_buy lowest bin": bo < E["bc_open_to_buy"][0],
            "bc_util lowest bin": bu < E["bc_util"][0],
            "bc_util highest bin": bu >= E["bc_util"][-1],
            "purpose home_improvement": pur == "home_improvement",
            "purpose major_purchase": pur == "major_purchase",
            "purpose small_business": pur == "small_business",
            "control: mo_sin_rcnt_tl 0-1": (ms >= 0) & (ms <= 1),
            "control: mo_sin_rcnt_tl 7-8": (ms >= 7) & (ms <= 8)}


MV, MO = seg_masks(S["valid"]), seg_masks(S["oot"])
for name, n in EXPECTED_N.items():
    assert int(MV[name].sum()) == n, f"{name}: {int(MV[name].sum())} validation loans, script 20 had {n}"
print("\nSegment definitions reproduce the validation counts of script 20")


def seg_stats(mask, y, p):
    n = int(mask.sum())
    obs, exp = float(y[mask].mean()), float(p[mask].mean())
    se = np.sqrt(float((p[mask] * (1 - p[mask])).sum())) / n
    return n, obs, exp, obs / exp, (obs - exp) / se


yt, pt_raw, pt_sh = Y["oot"][test_mask], P["oot"][test_mask], Ps["oot"][test_mask]
oe_all_test = yt.mean() / pt_sh.mean()
rows = []
for name in ["all loans"] + list(EXPECTED_N):
    mv = np.ones(len(Y["valid"]), dtype=bool) if name == "all loans" else MV[name]
    mo = np.ones(int(test_mask.sum()), dtype=bool) if name == "all loans" else MO[name][test_mask]
    nv, ov, ev, oev, zv = seg_stats(mv, Y["valid"], P["valid"])
    nt, ot, et_raw, oe_raw, _ = seg_stats(mo, yt, pt_raw)
    _, _, et_sh, oe_sh, z_sh = seg_stats(mo, yt, pt_sh)
    rel = oe_sh / oe_all_test
    persists = bool((oev - 1) * (rel - 1) > 0 and abs(rel - 1) >= PERSIST_FRACTION * abs(oev - 1)) if name != "all loans" else np.nan
    rows.append({"segment": name, "n_valid": nv, "OE_valid": oev, "z_valid": zv, "n_test": nt, "obs_test_pct": ot * 100,
                 "OE_test_raw": oe_raw, "OE_test_shifted": oe_sh, "z_test_shifted": z_sh,
                 "relative_OE_test": rel, "persists": persists})
seg = pd.DataFrame(rows)
print(f"\n=== C. Segment back-tests on the test window (Oct 2015 - Jan 2016). Relative O/E = segment O/E after the shift divided by "
      f"the overall {oe_all_test:.3f}; persists = same side of 1 as validation and at least {PERSIST_FRACTION:.0%} as large ===")
print(seg.round(3).to_string(index=False))
n_main = int(seg.loc[seg["segment"].str.startswith(("annual", "bc_", "purpose")), "persists"].sum())
n_ctrl = int(seg.loc[seg["segment"].str.startswith("control"), "persists"].sum())
print(f"\nPersisting: {n_main} of 7 replicated segments; {n_ctrl} of 2 controls")
seg.to_csv("reports/validation_segments.csv", index=False)

# ================= D. stability by issue quarter =================
q = pd.to_datetime(pd.concat([S["valid"]["issue_d"], S["oot"]["issue_d"]]), format="%b-%Y").dt.to_period("Q").astype(str).to_numpy()
yh = np.concatenate([Y["valid"], Y["oot"]])
ph = np.concatenate([P["valid"], P["oot"]])
psh = np.concatenate([Ps["valid"], Ps["oot"]])
is_oot = np.concatenate([np.zeros(len(Y["valid"]), bool), np.ones(len(Y["oot"]), bool)])
sh_tr = np.bincount(np.searchsorted(edges_tr, P["train"], side="right"), minlength=10) / len(P["train"])
rows = []
for qq in sorted(set(q)):
    m = q == qq
    sh_q = np.bincount(np.searchsorted(edges_tr, ph[m], side="right"), minlength=10) / m.sum()
    row = {"quarter": qq, "n": int(m.sum()), "bad": int(yh[m].sum()), "obs_pct": yh[m].mean() * 100,
           "exp_raw_pct": ph[m].mean() * 100, "OE_raw": yh[m].mean() / ph[m].mean(),
           "auc": roc_auc_score(yh[m], ph[m]), "score_psi": psi(sh_tr, sh_q)}
    if is_oot[m].all():
        row["OE_shifted"] = yh[m].mean() / psh[m].mean()
        row["note"] = "calibration window (1.000 by construction)" if qq == "2015Q3" else "test"
        row["in_band"] = bool(OE_LOW <= row["OE_shifted"] <= OE_HIGH)
    rows.append(row)
qt = pd.DataFrame(rows)
print("\n=== D. Stability by issue quarter (held-out loans: validation rows through 2015Q2, out-of-time from 2015Q3) ===")
print(qt.round(4).to_string(index=False))
print(f"\nBand {OE_LOW}-{OE_HIGH} after the shift: 2015Q4 and 2016Q1 are the only quarters that test it; "
      f"score PSI is at most {qt['score_psi'].max():.4f} (0.10 = watch)")
qt.to_csv("reports/validation_by_quarter.csv", index=False)

# ---------- figure ----------
os.makedirs("reports/figures", exist_ok=True)
try:
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    for k, c, lab in [("valid", "tab:blue", "validation"), ("oot", "tab:red", "out-of-time")]:
        t = DT[k]
        ax[0, 0].plot([0] + list(t["cum_loans_pct"]), [0] + list(t["cum_bad_pct"]), marker="o", ms=4, color=c, label=lab)
    ax[0, 0].plot([0, 100], [0, 100], color="grey", ls="--", lw=1, label="random")
    ax[0, 0].set_xlabel("% of loans rejected (riskiest first)")
    ax[0, 0].set_ylabel("% of bad loans captured")
    ax[0, 0].set_title("Cumulative gain")
    ax[0, 0].legend(fontsize=8)
    w = 0.38
    xs = np.arange(10)
    ax[0, 1].bar(xs - w / 2, DT["valid"].sort_index()["lift"], w, color="tab:blue", label="validation")
    ax[0, 1].bar(xs + w / 2, DT["oot"].sort_index()["lift"], w, color="tab:red", label="out-of-time")
    ax[0, 1].axhline(1.0, color="grey", lw=1)
    ax[0, 1].set_xticks(xs)
    ax[0, 1].set_xticklabels([str(i) for i in range(1, 11)])
    ax[0, 1].set_xlabel("score decile (10 = riskiest)")
    ax[0, 1].set_ylabel("lift (decile bad rate / overall)")
    ax[0, 1].set_title("Lift by decile")
    ax[0, 1].legend(fontsize=8)
    sg = seg[seg["segment"] != "all loans"].reset_index(drop=True)
    xs = np.arange(len(sg))
    ax[1, 0].scatter(xs, sg["OE_valid"], color="tab:blue", label="validation O/E (before shift)", zorder=3)
    ax[1, 0].scatter(xs, sg["relative_OE_test"], color="tab:red", marker="s", label="test window, relative O/E after shift", zorder=3)
    for i in xs:
        ax[1, 0].plot([i, i], [sg["OE_valid"][i], sg["relative_OE_test"][i]], color="grey", lw=0.8)
    ax[1, 0].axhline(1.0, color="grey", lw=1)
    ax[1, 0].set_xticks(xs)
    ax[1, 0].set_xticklabels(list(sg["segment"]), rotation=45, ha="right", fontsize=7)
    ax[1, 0].set_title("Do the in-time misfits persist out-of-time?")
    ax[1, 0].legend(fontsize=7)
    xq = np.arange(len(qt))
    ax[1, 1].plot(xq, qt["OE_raw"], marker="o", color="grey", label="before shift")
    if "OE_shifted" in qt:
        ok = qt["OE_shifted"].notna().to_numpy()
        ax[1, 1].plot(xq[ok], qt["OE_shifted"][ok], marker="o", color="tab:blue", label="after shift (2015Q3 = calibration window)")
    ax[1, 1].axhspan(OE_LOW, OE_HIGH, color="green", alpha=0.15, label=f"band {OE_LOW}-{OE_HIGH}")
    ax[1, 1].axhline(1.0, color="grey", lw=1)
    ax[1, 1].set_xticks(xq)
    ax[1, 1].set_xticklabels(list(qt["quarter"]), rotation=45, fontsize=8)
    ax[1, 1].set_ylabel("observed / expected")
    ax[1, 1].set_title("Observed / expected by issue quarter")
    ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("reports/figures/validation_suite.png", dpi=150)
    print("\nSaved reports/figures/validation_suite.png")
except Exception as e:
    print(f"\nFigure skipped ({type(e).__name__}: {e}); results above are unaffected")

print("Saved reports/validation_deciles.csv, validation_confusion.csv, validation_segments.csv, validation_by_quarter.csv")
