#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
compare_features.py - within-subject comparison of four feature sets.

Question: for ONE person at a time, which kind of feature separates imagined
left from right hand best, using the same model and the same metric?

Feature sets (all use the same trials, same order):
    lateralization : log P(C3) - log P(C4), mu and beta        ->   2 features
    power_12ch     : log band power at the 12 channels, 2 bands -> 24 features
    wpli_edges     : 66 wPLI edges x 2 bands                    -> 132 features
    graph_metrics  : strength, clustering, global efficiency    ->  50 features

Model: standardize -> L2 logistic regression, regularization strength chosen by
an INNER cross-validation. Because scaler and tuning live inside the pipeline,
they are fitted on training folds only (no leakage).
Metric: out-of-fold AUC (0.5 = chance), averaged over repeated 5-fold splits.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import SUBJECTS
from graphs import build_all, load_edges
from power_control import load_power

N_REPEATS = 5


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5,
                             scoring="neg_log_loss", max_iter=2000),
    )


def subject_auc(X, y, n_repeats=N_REPEATS):
    """Mean over repeats of the pooled out-of-fold AUC."""
    aucs = []
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        dec = cross_val_predict(make_model(), X, y, cv=cv,
                                method="decision_function")
        aucs.append(roc_auc_score(y, dec))
    return float(np.mean(aucs))


def feature_sets():
    """{subject: ({set name: X (trials, features)}, y)}"""
    power = load_power()                  # {s: (P (trials, ch, bands), y, ch_names)}
    edges = load_edges()                  # {s: (X (trials, 66, bands), y)}
    graph, graph_y, _ = build_all()       # {s: F (trials, 50)}, {s: y}
    out = {}
    for s in SUBJECTS:
        P, y, names = power[s]
        # all three sources must describe the same trials in the same order
        assert np.array_equal(y, edges[s][1]) and np.array_equal(y, graph_y[s]), \
            f"subject {s}: trial order differs between feature sources"
        i3, i4 = names.index("C3"), names.index("C4")
        out[s] = ({
            "lateralization": P[:, i3, :] - P[:, i4, :],
            "power_12ch": P.reshape(len(y), -1),
            "wpli_edges": edges[s][0].reshape(len(y), -1),
            "graph_metrics": graph[s],
        }, y)
    return out


def run():
    rows = []
    for s, (sets, y) in feature_sets().items():
        for name, X in sets.items():
            rows.append(dict(subject=s, features=name, n_features=X.shape[1],
                             auc=subject_auc(X, y)))
    return pd.DataFrame(rows)


def summarize(df):
    wide = df.pivot(index="subject", columns="features", values="auc")
    print("\nPer feature set (median over subjects):")
    for name in wide.columns:
        v = wide[name]
        print(f"  {name:15s} median AUC {v.median():.3f} | "
              f"{(v > 0.5).sum()}/{len(v)} subjects above 0.5 | "
              f"IQR {v.quantile(.25):.3f}-{v.quantile(.75):.3f}")
    print("\nPaired Wilcoxon vs lateralization (3 comparisons -> use p < 0.017):")
    for name in wide.columns.drop("lateralization"):
        _, p = wilcoxon(wide["lateralization"], wide[name])
        diff = (wide["lateralization"] - wide[name]).median()
        print(f"  lateralization vs {name:14s} median difference "
              f"{diff:+.3f}  p = {p:.4f}")
    return wide


def plot(wide):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(0)
    for i, name in enumerate(wide.columns):
        v = wide[name].to_numpy()
        ax.scatter(i + rng.normal(0, 0.05, len(v)), v, alpha=0.6)
        ax.hlines(np.median(v), i - 0.25, i + 0.25, color="k", lw=2)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(wide.columns)))
    ax.set_xticklabels(wide.columns, rotation=15)
    ax.set_ylabel("out-of-fold AUC (one dot per subject)")
    ax.set_title("Within-subject decoding by feature set (black bar = median)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df = run()
    df.to_csv("data/processed/within_subject_auc.csv", index=False)
    plot(summarize(df)).savefig("within_subject_auc.png", dpi=150)
