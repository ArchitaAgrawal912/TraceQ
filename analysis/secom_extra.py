"""Extra experiments: (1) random-split CV (common but optimistic) vs time split,
(2) walk-forward weekly retraining = TraceQ feedback loop, (3) SPC-style rule baseline."""
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

d = sys.argv[1]
X = pd.read_csv(f"{d}/secom.data", sep=r"\s+", header=None, na_values="NaN")
lab = pd.read_csv(f"{d}/secom_labels.data", sep=r"\s+", header=None, names=["y", "ts"], quotechar='"')
lab["ts"] = pd.to_datetime(lab["ts"], format="%d/%m/%Y %H:%M:%S")
y = (lab["y"] == 1).astype(int).values
o = np.argsort(lab["ts"].values, kind="stable")
X, y, ts = X.iloc[o].reset_index(drop=True), y[o], lab["ts"].iloc[o].reset_index(drop=True)

def prep(tr, te):
    keep = [c for c in tr.columns if tr[c].isna().mean() <= .5 and tr[c].nunique(dropna=True) > 1]
    med = tr[keep].median(); mu = tr[keep].fillna(med).mean(); sd = tr[keep].fillna(med).std() + 1e-9
    f = lambda a: np.clip(((a[keep].fillna(med) - mu) / sd).values, -10, 10)
    return f(tr), f(te)
def rf(): return RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample", random_state=7, n_jobs=-1)
out = {}

# (1) random stratified 5-fold CV
aucs = []
for seed in range(3):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        a, b = prep(X.iloc[tr], X.iloc[te]); m = rf().fit(a, y[tr]); aucs.append(roc_auc_score(y[te], m.predict_proba(b)[:, 1]))
out["random_cv_auc_mean"] = float(np.mean(aucs))

# (2) static vs walk-forward on the same test period (last 30%)
n_tr = int(len(X) * .7)
a, b = prep(X.iloc[:n_tr], X.iloc[n_tr:]); s_static = rf().fit(a, y[:n_tr]).predict_proba(b)[:, 1]
week = ts.dt.to_period("W").values
s_wf = np.zeros(len(X) - n_tr)
test_idx = np.arange(n_tr, len(X))
for w in pd.unique(week[n_tr:]):
    idx = test_idx[week[n_tr:] == w]
    past = np.arange(0, idx[0])  # everything labelled before this week (investigation outcomes fed back)
    a, b = prep(X.iloc[past], X.iloc[idx]); s_wf[idx - n_tr] = rf().fit(a, y[past]).predict_proba(b)[:, 1]
yte = y[n_tr:]
def cap(s, frac):
    k = int(round(frac * len(s))); return float(yte[np.argsort(-s)[:k]].sum() / yte.sum())
out["static_auc"] = float(roc_auc_score(yte, s_static)); out["walkforward_auc"] = float(roc_auc_score(yte, s_wf))
out["static_capture"] = {p: cap(s_static, p / 100) for p in [10, 20, 30]}
out["walkforward_capture"] = {p: cap(s_wf, p / 100) for p in [10, 20, 30]}

# (3) SPC-style rule: count of sensors beyond 3 sigma of training passes
a, b = prep(X.iloc[:n_tr], X.iloc[n_tr:])
mu, sd = a[y[:n_tr] == 0].mean(0), a[y[:n_tr] == 0].std(0) + 1e-9
s_spc = (np.abs((b - mu) / sd) > 3).sum(1).astype(float)
out["spc_auc"] = float(roc_auc_score(yte, s_spc)); out["spc_capture"] = {p: cap(s_spc, p / 100) for p in [10, 20, 30]}
out["spc_alarms_per_batch_median"] = float(np.median(s_spc))
out["spc_share_batches_with_alarm"] = float((s_spc > 0).mean())
print(json.dumps(out, indent=2)); json.dump(out, open(sys.argv[2], "w"), indent=2)

# bootstrap CI for static RF capture at 20% budget + timing
rng = np.random.default_rng(7); caps = []
for _ in range(2000):
    i = rng.integers(0, len(yte), len(yte)); yy, ss = yte[i], s_static[i]
    if yy.sum() == 0: continue
    k = int(round(.2 * len(yy))); caps.append(yy[np.argsort(-ss)[:k]].sum() / yy.sum())
print("RF capture@20% CI", np.percentile(caps, [2.5, 97.5]))
