"""TraceQ evidence analysis on UCI SECOM.
Time-ordered split: train on earliest 70% of production runs, evaluate on the latest 30%
(simulates deploying on past data and scoring future batches). No test data used for fitting.
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

d = sys.argv[1]
X = pd.read_csv(f"{d}/secom.data", sep=r"\s+", header=None, na_values="NaN")
X.columns = [f"S{i}" for i in range(X.shape[1])]
lab = pd.read_csv(f"{d}/secom_labels.data", sep=r"\s+", header=None, names=["y", "ts"], quotechar='"')
lab["ts"] = pd.to_datetime(lab["ts"], format="%d/%m/%Y %H:%M:%S")
y = (lab["y"] == 1).astype(int).values
order = np.argsort(lab["ts"].values, kind="stable")
X, y, ts = X.iloc[order].reset_index(drop=True), y[order], lab["ts"].iloc[order].reset_index(drop=True)

out = {}
out["rows"] = int(len(X)); out["features"] = int(X.shape[1]); out["fails"] = int(y.sum())
out["fail_rate"] = float(y.mean()); out["date_min"] = str(ts.min().date()); out["date_max"] = str(ts.max().date())
out["missing_cell_pct"] = float(X.isna().mean().mean() * 100)
out["cols_gt50_missing"] = int((X.isna().mean() > 0.5).sum())
out["constant_cols"] = int((X.nunique(dropna=True) <= 1).sum())

# monthly fail rate
m = pd.DataFrame({"m": ts.dt.to_period("M").astype(str), "y": y}).groupby("m")["y"].agg(["count", "sum"])
out["monthly"] = [{"month": k, "runs": int(r["count"]), "fails": int(r["sum"])} for k, r in m.iterrows()]

n_tr = int(len(X) * 0.7)
Xtr, Xte, ytr, yte = X.iloc[:n_tr], X.iloc[n_tr:], y[:n_tr], y[n_tr:]
out["train_rows"], out["test_rows"] = n_tr, int(len(Xte))
out["train_period"] = [str(ts.iloc[0].date()), str(ts.iloc[n_tr - 1].date())]
out["test_period"] = [str(ts.iloc[n_tr].date()), str(ts.iloc[-1].date())]
out["test_fails"] = int(yte.sum()); out["test_fail_rate"] = float(yte.mean())

keep = [c for c in X.columns if Xtr[c].isna().mean() <= 0.5 and Xtr[c].nunique(dropna=True) > 1]
med = Xtr[keep].median()
Atr, Ate = Xtr[keep].fillna(med), Xte[keep].fillna(med)
sc = StandardScaler().fit(Atr)
Ztr, Zte = np.clip(sc.transform(Atr), -10, 10), np.clip(sc.transform(Ate), -10, 10)
out["features_used"] = len(keep)

def prec_at(score, yv, k):
    idx = np.argsort(-score)[:k]; return float(yv[idx].mean())
def rec_at(score, yv, k):
    idx = np.argsort(-score)[:k]; return float(yv[idx].sum() / max(yv.sum(), 1))
def rank(s):
    return pd.Series(s).rank(pct=True).values

iso = IsolationForest(n_estimators=500, random_state=7).fit(Ztr[ytr == 0])  # learn "normal" from passing runs only
s_iso = -iso.score_samples(Zte)
rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample", random_state=7, n_jobs=-1).fit(Ztr, ytr)
s_rf = rf.predict_proba(Zte)[:, 1]
lr = LogisticRegression(penalty="l1", C=0.05, solver="liblinear", class_weight="balanced").fit(Ztr, ytr)
s_lr = lr.predict_proba(Zte)[:, 1]
# TraceQ score: pre-declared equal-weight rank blend of anomaly + supervised signals
s_tq = (rank(s_iso) + rank(s_rf) + rank(s_lr)) / 3

scores = {"Isolation Forest (unsupervised)": s_iso, "Random Forest": s_rf, "L1 Logistic": s_lr, "TraceQ blend": s_tq}
base = float(yte.mean())
ks = [10, 20, 30, 50]
res = {}
for name, s in scores.items():
    res[name] = {"auc": float(roc_auc_score(yte, s)),
                 **{f"p@{k}": prec_at(s, yte, k) for k in ks},
                 **{f"r@{k}": rec_at(s, yte, k) for k in ks}}
out["models"] = res
out["base_rate_test"] = base

# bootstrap CI for TraceQ precision@20 and recall at 20% budget
rng = np.random.default_rng(7)
budget = int(round(0.2 * len(yte)))
out["budget_20pct_n"] = budget
bp, br, bl = [], [], []
for _ in range(2000):
    i = rng.integers(0, len(yte), len(yte))
    yy, ss = yte[i], s_tq[i]
    if yy.sum() == 0: continue
    bp.append(prec_at(ss, yy, 20)); br.append(rec_at(ss, yy, budget)); bl.append(prec_at(ss, yy, budget) / yy.mean())
out["tq_p20_ci"] = [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))]
out["tq_recall_budget"] = rec_at(s_tq, yte, budget)
out["tq_recall_budget_ci"] = [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]
out["tq_lift_budget"] = prec_at(s_tq, yte, budget) / base
out["tq_lift_budget_ci"] = [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))]
# random inspection at same budget catches ~20% of fails in expectation
out["random_recall_budget"] = budget / len(yte)

# capture curve (share of fails caught vs share inspected)
curve = []
o = np.argsort(-s_tq)
for frac in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    k = int(round(frac * len(yte)))
    curve.append({"inspected_pct": int(frac * 100), "caught_pct": round(100 * yte[o[:k]].sum() / yte.sum(), 1)})
out["capture_curve"] = curve

# drivers: permutation importance of RF on test, scored by average precision
pi = permutation_importance(rf, Zte, yte, scoring="average_precision", n_repeats=20, random_state=7, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=keep).sort_values(ascending=False)
out["top_drivers"] = [{"sensor": k, "importance": float(v)} for k, v in imp.head(8).items()]

# drift: features whose test-period median moved > 1 train-std
shift = np.abs(np.median(Zte, axis=0))
out["drift_features_gt1sd"] = int((shift > 1).sum())
out["drift_top"] = [{"sensor": keep[i], "median_shift_sd": float(shift[i])} for i in np.argsort(-shift)[:5]]

# example explanation for top-ranked test batch: largest |z| vs training passing runs
pass_mu, pass_sd = Ztr[ytr == 0].mean(0), Ztr[ytr == 0].std(0) + 1e-9
top = o[0]
z = (Zte[top] - pass_mu) / pass_sd
zi = np.argsort(-np.abs(z))[:3]
out["example_batch"] = {"test_index": int(top), "timestamp": str(ts.iloc[n_tr + top]), "actual_label": "fail" if yte[top] else "pass",
                        "drivers": [{"sensor": keep[i], "z": round(float(z[i]), 1)} for i in zi]}
# how many of top-20 flagged are actually fails vs passes (unusual != defect)
t20 = o[:20]
out["top20_fails"], out["top20_passes"] = int(yte[t20].sum()), int(20 - yte[t20].sum())

print(json.dumps(out, indent=2))
json.dump(out, open(sys.argv[2], "w"), indent=2)
