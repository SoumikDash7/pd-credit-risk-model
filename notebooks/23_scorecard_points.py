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
from src.woe import WoEBinner  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 40)

# ---------- rules fixed before the results were seen ----------
BASE_SCORE, BASE_ODDS, PDO = 600.0, 50.0, 20.0     # 600 points at 50:1 good:bad, 20 points to double the odds
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)
AUC_TOL_ROUND, PD_TOL_ROUND = 0.0005, 0.1          # integer points accepted if both hold (PD tolerance in pp)
PDO_LOW, PDO_HIGH = 18.0, 22.0                     # implied points-to-double-odds on validation
AUC_CHECK = {"valid": 0.6708, "oot": 0.6788}

with open("models/scorecard_final.json", encoding="utf-8") as fh:
    SC = json.load(fh)
FEATS = SC["features"]
assert len(FEATS) == 16
rc = SC["recalibration"]
SHIFT_TESTED = float(rc["tested_parameters"]["shift_logodds"])
SHIFT_DEPLOY = float(rc["deployment_parameters_untested"]["shift_logodds"])
binner = WoEBinner.load("models/woe_spec.json")

df = pd.read_parquet("data/processed/features.parquet", columns=["issue_d", "split", "target"] + FEATS)
S = {k: df[df["split"] == k].reset_index(drop=True) for k in ["train", "valid", "oot"]}
Y = {k: v["target"].to_numpy().astype(int) for k, v in S.items()}
print("Rows: " + ", ".join(f"{k}={len(v):,}" for k, v in S.items()))
print(f"Scaling: Factor {FACTOR:.4f}, Offset {OFFSET:.4f} (score = Offset - Factor x log-odds of default; higher is safer)")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def pd_from_score(s):
    return 1 / (1 + np.exp((s - OFFSET) / FACTOR))


def ks_stat(y, p):
    d = pd.DataFrame({"p": p, "y": y})
    g = d.groupby("p")["y"].agg(["sum", "count"]).sort_index()
    bad = g["sum"].cumsum() / g["sum"].sum()
    good = (g["count"] - g["sum"]).cumsum() / (g["count"] - g["sum"]).sum()
    return float((bad - good).abs().max())


# ---------- exact model and exact score ----------
beta = np.array([SC["coefficients"][f] for f in FEATS])
IDX = {k: {f: binner.assign(f, S[k][f])[0] for f in FEATS} for k in S}
Wb = {k: binner.transform(S[k], FEATS)[FEATS].to_numpy() for k in S}
z = {k: SC["intercept"] + Wb[k] @ beta for k in S}
pd_dev = {k: sigmoid(z[k]) for k in S}
score_exact = {k: OFFSET - FACTOR * z[k] for k in S}
for k in ["valid", "oot"]:
    a = roc_auc_score(Y[k], pd_dev[k])
    assert abs(a - AUC_CHECK[k]) < 2e-4, f"{k}: AUC {a:.4f}, expected {AUC_CHECK[k]}"
    assert abs(roc_auc_score(Y[k], -score_exact[k]) - a) < 1e-9, "the score does not rank like the PD"
back = max(float(np.abs(pd_from_score(score_exact[k]) - pd_dev[k]).max()) for k in S)
assert back < 1e-9, f"score does not map back to the PD (max difference {back:.2e})"
print(f"\nCheck 1 passed: the exact score maps back to the model PD (largest difference {back:.1e}) and ranks identically")
for k in S:
    q = np.percentile(score_exact[k], [0, 1, 25, 50, 75, 99, 100])
    print(f"Exact score, {k}: min {q[0]:.1f}, p1 {q[1]:.1f}, p25 {q[2]:.1f}, median {q[3]:.1f}, p75 {q[4]:.1f}, p99 {q[5]:.1f}, max {q[6]:.1f}; mean {score_exact[k].mean():.1f}")

# ---------- points, rounded two ways ----------
def build_points(decimals):
    r = lambda x: float(np.round(x, decimals))
    base = r(OFFSET - FACTOR * SC["intercept"])
    pts = {}
    for f in FEATS:
        s = binner.specs[f]
        b = SC["coefficients"][f]
        bins = [r(-FACTOR * b * w) for w in s["woe"]]
        miss = None if s["missing_woe"] is None else r(-FACTOR * b * s["missing_woe"])
        pts[f] = (bins, miss)
    return base, pts


def score_from(decimals_pts, k):
    base, pts = decimals_pts
    total = np.full(len(Y[k]), base, dtype=float)
    for f in FEATS:
        bins, miss = pts[f]
        slots = np.array(list(bins) + [0.0 if miss is None else miss, 0.0])
        total += slots[IDX[k][f]]
    return total


rows, chosen = [], None
BUILT, SCORE = {}, {}
for dec in (0, 1):
    BUILT[dec] = build_points(dec)
    SCORE[dec] = {k: score_from(BUILT[dec], k) for k in S}
    for k in ["valid", "oot"]:
        s_d = SCORE[dec][k]
        d_pd = pd_from_score(s_d)
        a_ex, a_d = roc_auc_score(Y[k], -score_exact[k]), roc_auc_score(Y[k], -s_d)
        rows.append({"decimals": dec, "split": k, "mean_abs_score_diff": float(np.abs(s_d - score_exact[k]).mean()),
                     "max_abs_score_diff": float(np.abs(s_d - score_exact[k]).max()), "auc_exact": a_ex, "auc_rounded": a_d,
                     "auc_loss": a_ex - a_d, "ks_exact": ks_stat(Y[k], -score_exact[k]), "ks_rounded": ks_stat(Y[k], -s_d),
                     "mean_pd_diff_pp": float((d_pd.mean() - pd_dev[k].mean()) * 100),
                     "max_abs_pd_diff_pp": float(np.abs(d_pd - pd_dev[k]).max() * 100)})
rnd = pd.DataFrame(rows)
print("\n=== Check 2: rounding of points ===")
print(rnd.round(5).to_string(index=False))
ok0 = rnd[(rnd["decimals"] == 0) & (rnd["split"] == "valid")].iloc[0]
int_ok = bool(ok0["auc_loss"] < AUC_TOL_ROUND and abs(ok0["mean_pd_diff_pp"]) < PD_TOL_ROUND)
DEC = 0 if int_ok else 1
print(f"Integer points {'accepted' if int_ok else 'REJECTED (one-decimal points are used instead)'}: validation AUC loss "
      f"{ok0['auc_loss']:.5f} (limit {AUC_TOL_ROUND}), mean PD change {ok0['mean_pd_diff_pp']:+.4f}pp (limit {PD_TOL_ROUND})")
base_pts, pts = BUILT[DEC]
sc = SCORE[DEC]

# ---------- the scorecard as a self-contained card and an independent scorer ----------
card = {"scaling": {"base_score": BASE_SCORE, "base_odds_good_to_bad": BASE_ODDS, "pdo": PDO, "factor": FACTOR, "offset": OFFSET,
                    "score": "score = base_points + sum of bin points; higher is safer",
                    "pd_development": "PD = 1 / (1 + exp((score - offset) / factor))"},
        "points_decimals": DEC, "base_points": base_pts,
        "recalibration": {"shift_logodds_tested": SHIFT_TESTED, "shift_logodds_deployment_untested": SHIFT_DEPLOY,
                          "shift_in_points_tested": FACTOR * SHIFT_TESTED, "shift_in_points_deployment_untested": FACTOR * SHIFT_DEPLOY,
                          "pd_recalibrated": "PD = 1 / (1 + exp((score - offset) / factor - shift_logodds)); equivalently score + factor * shift"},
        "features": {}}
for f in FEATS:
    s = binner.specs[f]
    bins, miss = pts[f]
    card["features"][f] = {"kind": s["kind"], "cuts": list(s["cuts"]), "fill": s["fill"], "points": bins, "missing_points": miss,
                           "labels": list(s["labels"]), "level_to_bin": s.get("level_to_bin"), "unseen_points": 0.0}


def score_from_card(card, d):
    total = np.full(len(d), card["base_points"], dtype=float)
    for f, spec in card["features"].items():
        if spec["kind"] == "numeric":
            a = d[f].to_numpy(float)
            if spec["fill"] is not None:
                a = np.where(np.isnan(a), spec["fill"], a)
            idx = np.searchsorted(np.asarray(spec["cuts"], dtype=float), a, side="left")
            p = np.asarray(spec["points"], dtype=float)[idx]
            total += np.where(np.isnan(a), 0.0 if spec["missing_points"] is None else spec["missing_points"], p)
        else:
            m, p = spec["level_to_bin"], spec["points"]
            total += np.array([p[m[v]] if v in m else spec["unseen_points"] for v in d[f].astype(str).to_numpy()])
    return total


os.makedirs("models", exist_ok=True)
with open("models/scorecard_points.json", "w", encoding="utf-8") as fh:
    json.dump(card, fh, indent=1)
with open("models/scorecard_points.json", encoding="utf-8") as fh:
    card_back = json.load(fh)
for k in ["valid", "oot"]:
    sj = score_from_card(card_back, S[k])
    diff = float(np.abs(sj - sc[k]).max())
    assert diff < 1e-9, f"independent scorer differs from the points score on {k} (largest difference {diff:.3g})"
print(f"\nCheck 3 passed: models/scorecard_points.json, read back from disk, reproduces the points score of every validation "
      f"and out-of-time loan exactly ({len(S['valid']) + len(S['oot']):,} loans)")

# ---------- points table ----------
prow, srow = [], []
for f in FEATS:
    s = binner.specs[f]
    bins, miss = pts[f]
    nb = len(s["woe"])
    labels = list(s["labels"]) + ["Missing", "Unseen level"]
    woe_all = list(s["woe"]) + [s["missing_woe"] if s["missing_woe"] is not None else 0.0, 0.0]
    pall = list(bins) + [0.0 if miss is None else miss, 0.0]
    n = np.bincount(IDX["train"][f], minlength=nb + 2).astype(float)
    bad = np.bincount(IDX["train"][f], weights=Y["train"].astype(float), minlength=nb + 2)
    used = []
    for i in range(nb + 2):
        if i < nb or n[i] > 0:
            used.append(pall[i])
            prow.append({"feature": f, "bin": i if i < nb else ("Missing" if i == nb else "Unseen"), "label": labels[i],
                         "woe": round(woe_all[i], 4), "points": pall[i], "n_train": int(n[i]),
                         "share_train_pct": round(n[i] / n.sum() * 100, 2),
                         "bad_rate_train_pct": round(bad[i] / n[i] * 100, 2) if n[i] > 0 else np.nan})
    srow.append({"feature": f, "coefficient": SC["coefficients"][f], "min_points": min(used), "max_points": max(used),
                 "points_range": max(used) - min(used)})
ptab, psum = pd.DataFrame(prow), pd.DataFrame(srow)
psum["share_of_total_range_pct"] = psum["points_range"] / psum["points_range"].sum() * 100
psum = psum.sort_values("points_range", ascending=False).reset_index(drop=True)
print(f"\n=== Scorecard: base points {base_pts:g}; points per feature bin (decimals {DEC}) ===")
print(ptab[["feature", "bin", "label", "points", "n_train", "share_train_pct", "bad_rate_train_pct"]].to_string(index=False))
print("\n=== Spread of points by feature (max minus min) ===")
print(psum.round(3).to_string(index=False))

# ---------- score distribution, bands, scale check ----------
print(f"\n=== Points score ({DEC}-decimal points) ===")
for k in S:
    q = np.percentile(sc[k], [0, 1, 25, 50, 75, 99, 100])
    print(f"{k}: min {q[0]:g}, p1 {q[1]:g}, p25 {q[2]:g}, median {q[3]:g}, p75 {q[4]:g}, p99 {q[5]:g}, max {q[6]:g}; mean {sc[k].mean():.1f}, "
          f"share at or above 600: {(sc[k] >= 600).mean() * 100:.2f}%")
edges = np.quantile(sc["train"], np.linspace(0, 1, 11))[1:-1]


def band_of(s):
    return 10 - np.searchsorted(edges, s, side="right")       # 1 = highest scores (safest), 10 = lowest scores (riskiest)


brow, fits = [], {}
for k in ["train", "valid", "oot"]:
    b = band_of(sc[k])
    d = pd.DataFrame({"band": b, "y": Y[k], "s": sc[k], "p": pd_from_score(sc[k])})
    g = d.groupby("band").agg(n=("y", "size"), bad=("y", "sum"), mean_score=("s", "mean"), min_score=("s", "min"),
                              max_score=("s", "max"), pred=("p", "mean")).sort_index()
    g["obs"] = g["bad"] / g["n"]
    g["OE"] = g["obs"] / g["pred"]
    g["split"] = k
    brow.append(g.reset_index())
    sl, ic = np.polyfit(g["mean_score"].to_numpy(), np.log((1 - g["obs"]) / g["obs"]).to_numpy(), 1)
    fits[k] = (sl, ic, np.log(2) / sl, bool(np.all(np.diff(g["obs"].to_numpy()) >= 0)))
bands = pd.concat(brow, ignore_index=True)
tr_rng = bands[bands["split"] == "train"].set_index("band")
print("\n=== Check 4: ten risk bands from train score deciles (band 1 = highest scores = safest) ===")
out = []
for bnd in range(1, 11):
    v = bands[(bands["split"] == "valid") & (bands["band"] == bnd)].iloc[0]
    o = bands[(bands["split"] == "oot") & (bands["band"] == bnd)].iloc[0]
    out.append({"band": bnd, "train_score_range": f"{tr_rng.loc[bnd, 'min_score']:g}-{tr_rng.loc[bnd, 'max_score']:g}",
                "valid_n": int(v["n"]), "valid_obs_pct": v["obs"] * 100, "valid_pred_pct": v["pred"] * 100,
                "oot_n": int(o["n"]), "oot_obs_pct": o["obs"] * 100, "oot_pred_pct": o["pred"] * 100, "oot_OE": o["OE"]})
bt = pd.DataFrame(out)
print(bt.round(2).to_string(index=False))
for k, label in [("valid", "validation"), ("oot", "out-of-time")]:
    print(f"{label}: observed bad rate rises from band 1 to band 10 at every step: {'PASS' if fits[k][3] else 'FAIL (a reversal is reported above)'}")
print("\n=== Check 5: scale of the points (log-odds of good over bad against band mean score) ===")
for k, label in [("valid", "validation"), ("oot", "out-of-time")]:
    sl, ic, pdo_hat, _ = fits[k]
    note = ""
    if k == "valid":
        note = f" -> {'PASS' if PDO_LOW <= pdo_hat <= PDO_HIGH else 'FAIL'} (accepted range {PDO_LOW:g}-{PDO_HIGH:g})"
    print(f"{label}: slope {sl:.5f} per point (design {1 / FACTOR:.5f}); implied points to double the odds {pdo_hat:.2f} (design {PDO:g}){note}")

# ---------- recalibration expressed in points; PD lookup ----------
print(f"\n=== Recalibration in points ===")
print(f"Tested shift {SHIFT_TESTED:.4f} log-odds = {FACTOR * SHIFT_TESTED:.2f} points; deployment shift {SHIFT_DEPLOY:.4f} = {FACTOR * SHIFT_DEPLOY:.2f} points "
      f"(untested); a loan scoring S has the PD of a development-model loan scoring S - shift in points")
lk = pd.DataFrame({"score": np.arange(480, 621, 20)})
lk["pd_development_pct"] = pd_from_score(lk["score"].to_numpy(float)) * 100
lk["pd_tested_shift_pct"] = 1 / (1 + np.exp((lk["score"] - OFFSET) / FACTOR - SHIFT_TESTED)) * 100
lk["pd_deployment_shift_pct"] = 1 / (1 + np.exp((lk["score"] - OFFSET) / FACTOR - SHIFT_DEPLOY)) * 100
lk["odds_good_to_bad_development"] = (1 - lk["pd_development_pct"] / 100) / (lk["pd_development_pct"] / 100)
print(lk.round(2).to_string(index=False))

# ---------- outputs ----------
os.makedirs("reports/figures", exist_ok=True)
ptab.to_csv("reports/scorecard_points.csv", index=False)
psum.to_csv("reports/scorecard_points_summary.csv", index=False)
bands.to_csv("reports/score_bands.csv", index=False)
lk.to_csv("reports/score_to_pd.csv", index=False)
rnd.to_csv("reports/points_rounding_check.csv", index=False)
try:
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    ps = psum.sort_values("points_range")
    ax[0, 0].barh(ps["feature"], ps["points_range"], color="tab:blue")
    ax[0, 0].set_xlabel("points range (max - min)")
    ax[0, 0].set_title("Spread of points by feature")
    ax[0, 0].tick_params(axis="y", labelsize=8)
    lo, hi = float(min(sc["valid"].min(), sc["oot"].min())), float(max(sc["valid"].max(), sc["oot"].max()))
    for k, c in [("valid", "tab:blue"), ("oot", "tab:red")]:
        ax[0, 1].hist(sc[k], bins=np.arange(lo, hi + 2, 2), density=True, alpha=0.5, color=c, label=k)
    for e in edges:
        ax[0, 1].axvline(e, color="grey", lw=0.6, ls=":")
    ax[0, 1].set_xlabel("score (dotted lines: train decile edges)")
    ax[0, 1].set_title("Score distribution")
    ax[0, 1].legend(fontsize=8)
    w = 0.38
    xs = np.arange(1, 11)
    ax[1, 0].bar(xs - w / 2, bt["valid_obs_pct"], w, color="tab:blue", label="validation observed")
    ax[1, 0].bar(xs + w / 2, bt["oot_obs_pct"], w, color="tab:red", label="out-of-time observed")
    ax[1, 0].plot(xs, bt["valid_pred_pct"], color="black", marker="o", ms=3, label="development PD (validation)")
    ax[1, 0].set_xticks(xs)
    ax[1, 0].set_xlabel("risk band (1 = safest)")
    ax[1, 0].set_ylabel("bad rate (%)")
    ax[1, 0].set_title("Bad rate by score band")
    ax[1, 0].legend(fontsize=7)
    for k, c in [("valid", "tab:blue"), ("oot", "tab:red")]:
        g = bands[bands["split"] == k]
        ax[1, 1].scatter(g["mean_score"], np.log((1 - g["obs"]) / g["obs"]), color=c, label=f"{k} bands")
    sx = np.linspace(float(bands["mean_score"].min()) - 3, float(bands["mean_score"].max()) + 3, 50)
    ax[1, 1].plot(sx, (sx - OFFSET) / FACTOR, color="black", label="design: doubles every 20 points")
    ax[1, 1].set_xlabel("band mean score")
    ax[1, 1].set_ylabel("ln(good / bad), observed")
    ax[1, 1].set_title("Scale of the points")
    ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("reports/figures/scorecard_points.png", dpi=150)
    print("\nSaved reports/figures/scorecard_points.png")
except Exception as e:
    print(f"\nFigure skipped ({type(e).__name__}: {e}); results above are unaffected")

print("Saved models/scorecard_points.json and reports/scorecard_points.csv, scorecard_points_summary.csv, score_bands.csv, score_to_pd.csv, points_rounding_check.csv")
