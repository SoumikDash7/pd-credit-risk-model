"""Weight-of-Evidence (WoE) binning for a PD scorecard.

Numeric features: decision-tree cut points, then adjacent bins are merged until the
bad rate is monotone in the feature (direction chosen by higher IV). Missing values
get their own bin. Categorical features: levels are ordered by train bad rate and
adjacent levels are grouped by a decision tree. Everything is fitted on train rows
only; transform() applies the stored cut points / WoE values to any other data.

Convention: WoE = ln(%good / %bad), so a HIGHER WoE means a SAFER bin.
"""
import json

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

FLOOR = 0.5  # count smoothing: an empty bin never produces log(0)


def woe_iv(n, bad, total_good, total_bad):
    n = np.asarray(n, dtype=float)
    bad = np.asarray(bad, dtype=float)
    pg = np.maximum(n - bad, FLOOR) / total_good
    pb = np.maximum(bad, FLOOR) / total_bad
    woe = np.log(pg / pb)
    return woe, (pg - pb) * woe


def psi(p_ref, p_new, eps=1e-4):
    a = np.maximum(np.asarray(p_ref, dtype=float), eps)
    b = np.maximum(np.asarray(p_new, dtype=float), eps)
    return float(np.sum((b - a) * np.log(b / a)))


class WoEBinner:
    def __init__(self, min_bin_share=0.02, min_bin_count=1000, max_bins=10, median_impute=()):
        self.min_bin_share = min_bin_share
        self.min_bin_count = min_bin_count
        self.max_bins = max_bins
        self.median_impute = set(median_impute)
        self.specs = {}
        self.totals = None

    # ---------- helpers ----------
    def _min_leaf(self, n):
        return max(self.min_bin_count, int(self.min_bin_share * n))

    def _tree_cuts(self, x, y):
        min_leaf = self._min_leaf(len(x))
        if len(x) < 2 * min_leaf or np.unique(x).size < 2:
            return np.array([], dtype=float)
        tree = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=self.max_bins,
                                      min_samples_leaf=min_leaf, random_state=0)
        tree.fit(x.reshape(-1, 1), y.astype(int))
        thresholds = tree.tree_.threshold[tree.tree_.feature >= 0]
        return np.unique(thresholds)

    @staticmethod
    def _counts(idx, y, nbins):
        n = np.bincount(idx, minlength=nbins).astype(float)
        bad = np.bincount(idx, weights=y, minlength=nbins)
        return n, bad

    @staticmethod
    def _pava(n, bad, increasing):
        """Pool adjacent violators: merge neighbouring bins until bad rate is monotone."""
        groups = [[i] for i in range(len(n))]
        N, B = [float(v) for v in n], [float(v) for v in bad]
        merged = True
        while merged and len(N) > 1:
            merged = False
            for i in range(len(N) - 1):
                r0, r1 = B[i] / N[i], B[i + 1] / N[i + 1]
                if (increasing and r0 > r1) or ((not increasing) and r0 < r1):
                    N[i] += N[i + 1]
                    B[i] += B[i + 1]
                    groups[i] = groups[i] + groups[i + 1]
                    del N[i + 1], B[i + 1], groups[i + 1]
                    merged = True
                    break
        return groups

    @staticmethod
    def _numeric_labels(cuts):
        if not cuts:
            return ["all values"]
        labels = []
        for i in range(len(cuts) + 1):
            lo = "-inf" if i == 0 else f"{cuts[i - 1]:.6g}"
            hi = "inf" if i == len(cuts) else f"{cuts[i]:.6g}"
            labels.append(f"({lo}, {hi}]" if hi != "inf" else f"({lo}, inf)")
        return labels

    # ---------- fitting ----------
    def _fit_numeric(self, name, x, y, tg, tb):
        x = pd.Series(x).astype("float64")
        fill = None
        if name in self.median_impute:
            fill = float(x.median())
            x = x.fillna(fill)
        arr = x.to_numpy()
        miss = np.isnan(arr)
        xv, yv = arr[~miss], y[~miss]
        n_m, b_m = float(miss.sum()), float(y[miss].sum())
        cuts = self._tree_cuts(xv, yv)
        idx0 = np.searchsorted(cuts, xv, side="left")
        n0, b0 = self._counts(idx0, yv, len(cuts) + 1)

        def total_iv(n, b):
            if n_m > 0:
                n, b = np.append(n, n_m), np.append(b, b_m)
            return woe_iv(n, b, tg, tb)

        iv_unc = float(total_iv(n0, b0)[1].sum())
        best = None
        for inc in (True, False):
            groups = self._pava(n0, b0, inc)
            n = np.array([n0[g].sum() for g in groups])
            b = np.array([b0[g].sum() for g in groups])
            iv = float(total_iv(n, b)[1].sum())
            if best is None or iv > best[0] + 1e-12:
                best = (iv, groups, n, b, inc)
        iv, groups, n, b, inc = best
        final_cuts = [float(cuts[g[-1]]) for g in groups[:-1]]
        woe_all, iv_c = total_iv(n, b)
        nb = len(n)
        return {"kind": "numeric", "cuts": final_cuts, "fill": fill,
                "direction": "increasing" if inc else "decreasing",
                "woe": woe_all[:nb].tolist(),
                "missing_woe": float(woe_all[nb]) if n_m > 0 else None,
                "labels": self._numeric_labels(final_cuts),
                "iv_unconstrained": iv_unc, "iv_train": float(iv_c.sum())}

    def _fit_categorical(self, name, x, y, tg, tb):
        lvl = pd.Series(x).astype("string").fillna("MISSING").astype(object).to_numpy()
        df = pd.DataFrame({"lvl": lvl, "y": y})
        g = df.groupby("lvl")["y"].agg(["size", "sum"]).rename(columns={"size": "n", "sum": "bad"})
        g["rate"] = g["bad"] / g["n"]
        order = g.sort_values(["rate", "n"]).index.tolist()
        rank = {lv: i for i, lv in enumerate(order)}
        r = df["lvl"].map(rank).to_numpy(dtype=float)
        cuts = self._tree_cuts(r, y)
        level_to_bin = {lv: int(np.searchsorted(cuts, rank[lv], side="left")) for lv in order}
        nb = len(cuts) + 1
        idx = np.searchsorted(cuts, r, side="left")
        n, b = self._counts(idx, y, nb)
        woe, ivc = woe_iv(n, b, tg, tb)
        members = [[] for _ in range(nb)]
        for lv in order:
            members[level_to_bin[lv]].append(lv)
        labels = [" | ".join(m[:6]) + (f" | ... (+{len(m) - 6})" if len(m) > 6 else "") for m in members]
        return {"kind": "categorical", "cuts": [], "fill": None, "direction": "increasing",
                "level_to_bin": level_to_bin, "woe": woe.tolist(), "missing_woe": None,
                "labels": labels, "iv_unconstrained": float(ivc.sum()), "iv_train": float(ivc.sum())}

    def fit(self, X, y, numeric, categorical):
        y = np.asarray(y, dtype=float)
        tb = float(y.sum())
        tg = float(len(y) - tb)
        self.totals = (tg, tb)
        for c in numeric:
            self.specs[c] = self._fit_numeric(c, X[c], y, tg, tb)
        for c in categorical:
            self.specs[c] = self._fit_categorical(c, X[c], y, tg, tb)
        return self

    # ---------- applying ----------
    def assign(self, name, x):
        """Return (bin index, WoE) per row. Slot nb = Missing, slot nb+1 = unseen level."""
        s = self.specs[name]
        nb = len(s["woe"])
        miss_woe = s["missing_woe"] if s["missing_woe"] is not None else 0.0
        slots = np.array(list(s["woe"]) + [miss_woe, 0.0], dtype=float)
        if s["kind"] == "numeric":
            x = pd.Series(x).astype("float64")
            if s["fill"] is not None:
                x = x.fillna(s["fill"])
            arr = x.to_numpy()
            idx = np.searchsorted(np.asarray(s["cuts"], dtype=float), arr, side="left")
            idx = np.where(np.isnan(arr), nb, idx)
        else:
            lv = pd.Series(x).astype("string").fillna("MISSING").astype(object).to_numpy()
            m = s["level_to_bin"]
            idx = np.array([m.get(v, nb + 1) for v in lv])
        idx = idx.astype(int)
        return idx, slots[idx]

    def bin_counts(self, name, x, y):
        nb = len(self.specs[name]["woe"])
        idx, _ = self.assign(name, x)
        return self._counts(idx, np.asarray(y, dtype=float), nb + 2)

    def transform(self, X, features=None):
        feats = features or list(self.specs)
        return pd.DataFrame({c: self.assign(c, X[c])[1] for c in feats}, index=X.index)

    # ---------- persistence ----------
    def save(self, path):
        payload = {"params": {"min_bin_share": self.min_bin_share, "min_bin_count": self.min_bin_count,
                              "max_bins": self.max_bins, "median_impute": sorted(self.median_impute)},
                   "totals": list(self.totals), "specs": self.specs}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        obj = cls(**d["params"])
        obj.specs = d["specs"]
        obj.totals = tuple(d["totals"])
        return obj
