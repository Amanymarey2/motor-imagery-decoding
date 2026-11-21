#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
ard_cross_subject.py - sparse Bayesian learning (ARD) in leave-one-subject-out.

Model: Bayesian linear regression on the 0/1 label (1 = right) where every
feature weight has its OWN Gaussian prior precision, learned from the training
data by evidence maximization (sklearn's ARDRegression). Features the data do
not support get a huge precision and are pruned to exactly zero.
No inner cross-validation: the amount of shrinkage per feature comes from the
evidence, which is the difference from the L1/L2 baselines.

Same features, same per-subject z-scoring and same LOSO folds as
cross_subject.py, so the numbers are directly comparable.

Caveats :
  * A Gaussian likelihood on a binary target is a misspecified model. It is the
    same simplification the SBLEST paper makes. The logistic horseshoe will be
    the properly specified model.
  * Predictions are clipped to [0, 1] to compute a Brier score. That is a
    linear-probability approximation, NOT a calibrated probability.
  * Selection frequency across LOSO folds is optimistic: any two training sets
    share 28 of 29 subjects, so folds are not independent.

Usage:  python ard_cross_subject.py          -> LOSO comparison
        python ard_cross_subject.py null     -> label-shuffle reference
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ARDRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_features import feature_sets
from connectivity import BANDS, CHANNELS
from cross_subject import RIGHT, pooled
from data import SUBJECTS
from graphs import feature_names

N_NULL = 5


def names_per_set():
    """Readable column names, in the same order the features were stacked."""
    rows, cols = np.tril_indices(len(CHANNELS), k=-1)
    edges = [f"{CHANNELS[r]}-{CHANNELS[c]}" for r, c in zip(rows, cols)]
    bands = list(BANDS)
    return {
        "lateralization": [f"{b}_C3-C4" for b in bands],
        # (trials, 66, bands).reshape -> edge-major, band-minor
        "wpli_edges": [f"{e}_{b}" for e in edges for b in bands],
        # (trials, channels, bands).reshape -> channel-major, band-minor
        "power_12ch": [f"{ch}_{b}" for ch in CHANNELS for b in bands],
        "graph_metrics": feature_names(),
    }


def loso_ard(X, y, g):
    rows, selected = [], []
    for s in SUBJECTS:
        te = g == s
        model = make_pipeline(StandardScaler(),
                              ARDRegression(max_iter=300, tol=1e-3))
        model.fit(X[~te], (y[~te] == RIGHT).astype(float))
        pred = model.predict(X[te])                    # continuous score
        yt = (y[te] == RIGHT).astype(int)
        coef = model[-1].coef_
        selected.append(coef != 0)
        rows.append(dict(
            subject=s,
            auc=roc_auc_score(yt, pred),
            bal_acc=balanced_accuracy_score(yt, pred > 0.5),
            brier=float(np.mean((np.clip(pred, 0, 1) - yt) ** 2)),
            n_nonzero=int((coef != 0).sum()),
        ))
    return pd.DataFrame(rows), np.array(selected)


def main():
    X, y, g = pooled(feature_sets())
    names = names_per_set()
    results, freq_rows = [], []
    for name, Xn in X.items():
        res, sel = loso_ard(Xn, y, g)
        res["features"] = name
        results.append(res)
        freq = sel.mean(axis=0)                         # share of folds kept
        freq_rows += [dict(features=name, feature=f, selection_freq=float(q))
                      for f, q in zip(names[name], freq)]
        top = np.argsort(freq)[::-1][:5]
        print(f"{name:15s} median AUC {res.auc.median():.3f} | "
              f"median features kept {res.n_nonzero.median():.0f}/{Xn.shape[1]}",
              flush=True)
        print("   most often kept: " +
              ", ".join(f"{names[name][i]} ({freq[i]:.2f})" for i in top))
    df = pd.concat(results)
    df.to_csv("data/processed/ard_metrics.csv", index=False)
    pd.DataFrame(freq_rows).to_csv("data/processed/ard_selection.csv", index=False)

    try:   # side-by-side with the baselines from cross_subject.py
        base = pd.read_csv("data/processed/cross_subject_metrics.csv")
        med = base.groupby(["features", "model"]).auc.median().unstack()
    except FileNotFoundError:
        med = None
    print("\nfeatures         ARD     L2      L1      | Brier (ARD)")
    for name, d in df.groupby("features"):
        l2 = f"{med.loc[name, 'l2']:.3f}" if med is not None else "  n/a"
        l1 = f"{med.loc[name, 'l1']:.3f}" if med is not None else "  n/a"
        print(f"{name:15s}  {d.auc.median():.3f}   {l2}   {l1}   | "
              f"{d.brier.median():.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(0)
    for i, (name, d) in enumerate(df.groupby("features")):
        ax.scatter(i + rng.normal(0, 0.05, len(d)), d.auc, alpha=0.6)
        ax.hlines(d.auc.median(), i - 0.25, i + 0.25, color="k", lw=2)
    ax.set_xticks(range(df.features.nunique()))
    ax.set_xticklabels(sorted(df.features.unique()), rotation=15)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylabel("held-out-subject AUC (one dot per subject)")
    ax.set_title("ARD (sparse Bayesian) leave-one-subject-out")
    fig.tight_layout()
    fig.savefig("ard_cross_subject_auc.png", dpi=150)


def null():
    X, y, g = pooled(feature_sets())
    out = {n: [] for n in X}
    for k in range(N_NULL):
        y_shuf = y.copy()
        for s in SUBJECTS:
            idx = np.where(g == s)[0]
            y_shuf[idx] = np.random.default_rng(k + 1000 * s).permutation(y[idx])
        for n, Xn in X.items():
            out[n].append(loso_ard(Xn, y_shuf, g)[0].auc.median())
        print(f"shuffle {k + 1}/{N_NULL} done", flush=True)
    print("\nfeature set      shuffled group-median AUC: mean (range)")
    for n, v in out.items():
        print(f"{n:15s}  {np.mean(v):.3f} ({min(v):.3f} to {max(v):.3f})")


if __name__ == "__main__":
    null() if len(sys.argv) > 1 and sys.argv[1] == "null" else main()
