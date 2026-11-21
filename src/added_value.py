#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
added_value.py - does connectivity add anything BEYOND band power?

Feature sets (all trained leave-one-subject-out, same folds as before):
    power           : log band power, 12 channels x 2 bands          (24)
    power+graph     : power + graph metrics                           (74)
    power+noise50   : power + 50 columns of pure noise                (74)
    power+wpli      : power + wPLI edges                             (156)
    power+noise132  : power + 132 columns of pure noise              (156)

Why the noise controls: adding features can HURT a model just by diluting it,
even if they are uninformative. Comparing "power + connectivity" with
"power + the same number of noise columns" separates "connectivity is no
better than nothing" from "adding any columns costs accuracy".

Pre-specified primary contrasts (decided before seeing results):
    power+graph - power+noise50
    power+wpli  - power+noise132
for each of two models (L2, ARD) -> 4 tests, so use p < 0.0125.
Everything else printed is descriptive.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ard_cross_subject import loso_ard, names_per_set
from compare_features import feature_sets
from cross_subject import loso, pooled

N_POWER = 24


def build_sets():
    X, y, g = pooled(feature_sets())
    rng = np.random.default_rng(0)
    P, G, E = X["power_12ch"], X["graph_metrics"], X["wpli_edges"]
    noise = lambda k: rng.normal(size=(len(y), k))
    sets = {
        "power": P,
        "power+graph": np.hstack([P, G]),
        "power+noise50": np.hstack([P, noise(G.shape[1])]),
        "power+wpli": np.hstack([P, E]),
        "power+noise132": np.hstack([P, noise(E.shape[1])]),
    }
    return sets, y, g


def run():
    sets, y, g = build_sets()
    names = names_per_set()
    conn_names = {"power+graph": names["graph_metrics"],
                  "power+wpli": names["wpli_edges"]}
    rows, kept = [], {}
    for name, X in sets.items():
        res, _ = loso(X, y, g, "l2")
        res["model"], res["features"] = "l2", name
        rows.append(res)
        res, sel = loso_ard(X, y, g)
        res["model"], res["features"] = "ard", name
        rows.append(res)
        if name in conn_names:
            kept[name] = sel[:, N_POWER:]          # connectivity block only
        print(f"{name:15s} L2 {rows[-2].auc.median():.3f} | "
              f"ARD {rows[-1].auc.median():.3f}", flush=True)
    df = pd.concat(rows)
    df.to_csv("data/processed/added_value.csv", index=False)
    return df, kept, conn_names


def paired(df, model, a, b):
    w = df[df.model == model].pivot(index="subject", columns="features",
                                    values="auc")
    d = w[a] - w[b]
    return d.median(), int((d > 0).sum()), wilcoxon(w[a], w[b]).pvalue


def report(df, kept, conn_names):
    print("\nPRIMARY (is connectivity better than the same number of noise "
          "columns?)  p < 0.0125")
    for model in ("l2", "ard"):
        for a, b in (("power+graph", "power+noise50"),
                     ("power+wpli", "power+noise132")):
            m, n, p = paired(df, model, a, b)
            print(f"  {model:3s} {a:12s} - {b:14s} median diff {m:+.3f} "
                  f"({n}/30 subjects better)  p = {p:.4f}")
    print("\nDESCRIPTIVE (each set minus power alone)")
    for model in ("l2", "ard"):
        for a in ("power+graph", "power+noise50", "power+wpli", "power+noise132"):
            m, n, p = paired(df, model, a, "power")
            print(f"  {model:3s} {a:15s} - power  median diff {m:+.3f} "
                  f"({n}/30 better)  p = {p:.4f}")
    print("\nARD: connectivity features kept when power is available")
    for name, sel in kept.items():
        freq = sel.mean(axis=0)
        top = np.argsort(freq)[::-1][:5]
        print(f"  {name}: median {np.median(sel.sum(axis=1)):.0f} of "
              f"{sel.shape[1]} kept per fold; most frequent: " +
              ", ".join(f"{conn_names[name][i]} ({freq[i]:.2f})" for i in top))


def plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    combos = ["power+graph", "power+noise50", "power+wpli", "power+noise132"]
    rng = np.random.default_rng(0)
    for ax, model in zip(axes, ("l2", "ard")):
        w = df[df.model == model].pivot(index="subject", columns="features",
                                        values="auc")
        for i, c in enumerate(combos):
            d = (w[c] - w["power"]).to_numpy()
            ax.scatter(i + rng.normal(0, 0.05, len(d)), d, alpha=0.6)
            ax.hlines(np.median(d), i - 0.25, i + 0.25, color="k", lw=2)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(combos, rotation=20, fontsize=8)
        ax.set_title(model.upper())
    axes[0].set_ylabel("AUC minus power-alone AUC (per subject)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df, kept, conn_names = run()
    report(df, kept, conn_names)
    plot(df).savefig("added_value.png", dpi=150)
