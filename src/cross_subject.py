#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
cross_subject.py - leave-one-subject-out (LOSO) decoding.

For each subject: train on the other 29 people, test on the held-out one.
This is the question the project actually asks: does a pattern learned from
other people predict a new person's left vs right imagery?

Models  : L2 logistic regression and L1 logistic regression (ordinary sparsity
          baseline). ARD / horseshoe come next and will be compared against these.
Features: the same four sets as compare_features.py.
Metrics : AUC, balanced accuracy at 0.5, Brier score (calibration), per subject.

Leakage control
  * The scaler and the regularization strength are fitted on the TRAINING
    subjects only (they live inside the pipeline).
  * The inner tuning folds are grouped by subject, so the strength is chosen by
    how well a model transfers to unseen people, not to unseen trials of
    people it already saw.
  * Features are z-scored WITHIN each subject (no labels used). That removes
    person-specific baselines, but it uses the test subject's unlabeled trials,
    so it is transductive. State this in the README.

Usage:  python cross_subject.py          -> main comparison
        python cross_subject.py null     -> label-shuffle reference (L2 only)
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_features import feature_sets
from data import EVENT_ID, SUBJECTS

RIGHT = EVENT_ID["right"]              # positive class = right-hand imagery
CS = np.logspace(-3, 1, 9)
N_NULL = 5


def pooled(data):
    """Stack all subjects; z-score each feature within each subject."""
    names = list(next(iter(data.values()))[0])
    y = np.concatenate([data[s][1] for s in SUBJECTS])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in SUBJECTS])
    X = {}
    for n in names:
        blocks = []
        for s in SUBJECTS:
            B = data[s][0][n]
            sd = B.std(axis=0)
            sd[sd == 0] = 1.0
            blocks.append((B - B.mean(axis=0)) / sd)
        X[n] = np.vstack(blocks)
    return X, y, g


def make_model(kind, g_train):
    inner = list(GroupKFold(n_splits=5).split(np.zeros(len(g_train)), groups=g_train))
    if kind == "l2":
        lr = LogisticRegressionCV(Cs=CS, cv=inner, scoring="neg_log_loss",
                                  max_iter=2000)
    else:   # "l1": ordinary sparsity baseline
        lr = LogisticRegressionCV(Cs=CS, cv=inner, penalty="l1", solver="liblinear",
                                  scoring="neg_log_loss", max_iter=2000)
    return make_pipeline(StandardScaler(), lr)


def loso(X, y, g, kind):
    rows, preds = [], []
    for s in SUBJECTS:
        te = g == s
        model = make_model(kind, g[~te]).fit(X[~te], y[~te])
        p = model.predict_proba(X[te])[:, 1]          # P(right)
        yt = (y[te] == RIGHT).astype(int)
        lr = model[-1]                      # the fitted LogisticRegressionCV
        rows.append(dict(subject=s,
                         auc=roc_auc_score(yt, p),
                         bal_acc=balanced_accuracy_score(yt, p > 0.5),
                         brier=float(np.mean((p - yt) ** 2)),
                         c_chosen=float(lr.C_[0]),          # regularization picked
                         n_nonzero=int((lr.coef_ != 0).sum())))  # features kept
        preds += [dict(subject=s, y_right=int(a), p_right=float(b))
                  for a, b in zip(yt, p)]
    return pd.DataFrame(rows), pd.DataFrame(preds)


def main():
    X, y, g = pooled(feature_sets())
    results, preds = [], []
    for kind in ("l2", "l1"):
        for name, Xn in X.items():
            res, pr = loso(Xn, y, g, kind)
            res["features"], res["model"] = name, kind
            pr["features"], pr["model"] = name, kind
            results.append(res)
            preds.append(pr)
            print(f"{kind} {name:15s} median AUC {res.auc.median():.3f}", flush=True)
    df, pr = pd.concat(results), pd.concat(preds)
    df.to_csv("data/processed/cross_subject_metrics.csv", index=False)
    pr.to_csv("data/processed/cross_subject_predictions.csv", index=False)

    print("\nmodel features        median AUC (IQR)        n>0.5   bal.acc  Brier")
    for (kind, name), d in df.groupby(["model", "features"]):
        print(f"{kind:5s} {name:15s} {d.auc.median():.3f} "
              f"({d.auc.quantile(.25):.3f}-{d.auc.quantile(.75):.3f})   "
              f"{(d.auc > 0.5).sum():2d}/30   {d.bal_acc.median():.3f}   "
              f"{d.brier.median():.3f}")
    print("\nReference: Brier of always predicting 0.5 is 0.250.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    rng = np.random.default_rng(0)
    for i, ((kind, name), d) in enumerate(df.groupby(["model", "features"])):
        ax.scatter(i + rng.normal(0, 0.05, len(d)), d.auc, alpha=0.6)
        ax.hlines(d.auc.median(), i - 0.25, i + 0.25, color="k", lw=2)
    labels = [f"{k}\n{n}" for (k, n), _ in df.groupby(["model", "features"])]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylabel("held-out-subject AUC (one dot per subject)")
    ax.set_title("Leave-one-subject-out decoding (black bar = median)")
    fig.tight_layout()
    fig.savefig("cross_subject_auc.png", dpi=150)


def null():
    """Shuffle labels within each subject, rerun LOSO (L2): the chance reference."""
    X, y, g = pooled(feature_sets())
    out = {n: [] for n in X}
    for k in range(N_NULL):
        y_shuf = y.copy()
        for s in SUBJECTS:
            idx = np.where(g == s)[0]
            y_shuf[idx] = np.random.default_rng(k + 1000 * s).permutation(y[idx])
        for n, Xn in X.items():
            out[n].append(loso(Xn, y_shuf, g, "l2")[0].auc.median())
        print(f"shuffle {k + 1}/{N_NULL} done", flush=True)
    pd.DataFrame([dict(features=n, shuffle=k, median_auc=m)
                  for n, v in out.items() for k, m in enumerate(v)]
                 ).to_csv("data/processed/cross_subject_null.csv", index=False)
    print("\nfeature set      shuffled group-median AUC: mean (range)")
    for n, v in out.items():
        print(f"{n:15s}  {np.mean(v):.3f} ({min(v):.3f} to {max(v):.3f})")


if __name__ == "__main__":
    null() if len(sys.argv) > 1 and sys.argv[1] == "null" else main()