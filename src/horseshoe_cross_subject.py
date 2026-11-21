#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
horseshoe_cross_subject.py - Bayesian logistic regression with a regularized
horseshoe prior (Piironen & Vehtari 2017), leave-one-subject-out.

Feature sets tested : graph_metrics (50) and wpli_edges (132), z-scored within
subject, same folds as before.
Two variants of each set:
    plain : the real features only      -> AUC, Brier, calibration
    noise : real features + the same number of PURE NOISE columns
            -> how many noise columns does the prior let through?

Model (non-centred):
    beta_j = z_j * tau * lambda_tilde_j,   z_j ~ N(0, 1)
    lambda_j ~ HalfCauchy(1)               local scale (per feature)
    tau ~ HalfCauchy(tau0)                 global scale; tau0 from p0 = expected
                                           number of relevant features
    c^2 ~ InvGamma(2, 2)                   slab: stops large weights being unshrunk
    y ~ Bernoulli(sigmoid(b0 + X beta))    a proper logistic likelihood

Inference: stochastic variational inference with a mean-field Gaussian guide.
CAVEAT: mean-field VI usually UNDERESTIMATES posterior uncertainty, so credible
intervals are too narrow and "interval excludes zero" over-selects. The noise
variant measures how much. Treat intervals as optimistic.

Selection rule (fixed in advance): a feature is "selected" if its 90% posterior
interval excludes zero.

Usage:  python horseshoe_cross_subject.py quick   -> 3 held-out subjects, 1000 steps
        python horseshoe_cross_subject.py         -> full leave-one-subject-out
Requires: pip install numpyro jax
"""
import sys

import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from ard_cross_subject import names_per_set
from compare_features import feature_sets
from cross_subject import RIGHT, pooled
from data import SUBJECTS

STEPS, LR, N_POST = 8000, 0.005, 500
P0 = {"power_12ch": 5, "graph_metrics": 5, "wpli_edges": 13}
SETS = ("power_12ch", "graph_metrics", "wpli_edges")   # edit to run a subset
TAG = "all" if len(SETS) == 3 else "_".join(SETS)       # tags the output files   # ~10% of the real features: a PRIOR choice
NOISE_SEED = 1
SELECT_LEVEL = 0.90


def model(X, y=None, p0=5.0):
    n, p = X.shape
    tau0 = p0 / (p - p0) * 2.0 / np.sqrt(n)    # sigma ~ 2 for a logistic model
    nu, slab = 4.0, 1.0
    b0 = numpyro.sample("b0", dist.Normal(0.0, 2.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(tau0))
    lam = numpyro.sample("lam", dist.HalfCauchy(np.ones(p)))
    c2 = numpyro.sample("c2", dist.InverseGamma(0.5 * nu, 0.5 * nu * slab ** 2))
    z = numpyro.sample("z", dist.Normal(np.zeros(p), 1.0))
    lam_tilde = (c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2)) ** 0.5
    beta = numpyro.deterministic("beta", z * tau * lam_tilde)
    numpyro.sample("y", dist.Bernoulli(logits=b0 + X @ beta), obs=y)


def fit_posterior(X_tr, y_tr, p0, key):
    guide = AutoNormal(model)
    svi = SVI(model, guide, Adam(LR), Trace_ELBO())
    res = svi.run(key, STEPS, X_tr, y_tr, float(p0), progress_bar=False)
    post = Predictive(model, guide=guide, params=res.params,
                      num_samples=N_POST, return_sites=["b0", "beta"])
    return post(random.PRNGKey(1), X_tr, None, float(p0)), np.asarray(res.losses)


def loso_hs(X, y, g, p0, n_real, subjects):
    rows, preds, selected = [], [], []
    lo_q, hi_q = (1 - SELECT_LEVEL) / 2, 1 - (1 - SELECT_LEVEL) / 2
    for i, s in enumerate(subjects):
        te = g == s
        ytr = (y[~te] == RIGHT).astype(int)
        yte = (y[te] == RIGHT).astype(int)
        post, losses = fit_posterior(X[~te], ytr, p0, random.PRNGKey(int(s)))
        b0, beta = np.asarray(post["b0"]), np.asarray(post["beta"])
        p = expit(b0[:, None] + beta @ X[te].T).mean(axis=0)   # posterior mean P(right)
        lo, hi = np.quantile(beta, [lo_q, hi_q], axis=0)
        sel = (lo > 0) | (hi < 0)
        selected.append(sel)
        rows.append(dict(subject=s, auc=roc_auc_score(yte, p),
                         bal_acc=balanced_accuracy_score(yte, p > 0.5),
                         brier=float(np.mean((p - yte) ** 2)),
                         n_sel_real=int(sel[:n_real].sum()),
                         n_sel_noise=int(sel[n_real:].sum()),
                         loss_rel_drop=float((losses[2 * len(losses) // 3] - losses[-1])
                                             / abs(losses[-1]))))
        preds += [dict(subject=s, y_right=int(a), p_right=float(b))
                  for a, b in zip(yte, p)]
        if i == 0 or rows[-1]["loss_rel_drop"] > 0.005:
            k = len(losses)
            print(f"   subject {s}: ELBO loss at 1/3, 2/3, end: "
                  f"{losses[k // 3]:.0f}, {losses[2 * k // 3]:.0f}, {losses[-1]:.0f}"
                  f" (dropped {rows[-1]['loss_rel_drop']:.1%} over the last third)",
                  flush=True)
    return pd.DataFrame(rows), pd.DataFrame(preds), np.array(selected)


def reliability(ax, preds, label, bins=8):
    q = np.quantile(preds.p_right, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.digitize(preds.p_right, q[1:-1]), 0, bins - 1)
    mp = [preds.p_right[idx == b].mean() for b in range(bins)]
    of = [preds.y_right[idx == b].mean() for b in range(bins)]
    ax.plot(mp, of, "o-", label=label)


def main(quick=False):
    global STEPS
    subjects = SUBJECTS[:3] if quick else SUBJECTS
    if quick:
        STEPS = 1000
    X, y, g = pooled(feature_sets())
    names = names_per_set()
    rng = np.random.default_rng(NOISE_SEED)
    results, plain_preds, sel_rows = [], {}, []
    for name in SETS:
        Xr = X[name]
        n_real = Xr.shape[1]
        variants = {"plain": Xr,
                    "noise": np.hstack([Xr, rng.normal(size=Xr.shape)])}
        for variant, Xv in variants.items():
            print(f"{name} / {variant}  ({Xv.shape[1]} features)", flush=True)
            res, pr, sel = loso_hs(Xv, y, g, P0[name], n_real, subjects)
            res["features"], res["variant"] = name, variant
            results.append(res)
            if variant == "plain":
                plain_preds[name] = pr
            fn = names[name] + [f"noise_{k}" for k in range(Xv.shape[1] - n_real)]
            sel_rows += [dict(features=name, variant=variant, feature=f,
                              selection_freq=float(q))
                         for f, q in zip(fn, sel.mean(axis=0))]
    df = pd.concat(results)
    df.to_csv(f"data/processed/horseshoe_metrics_{TAG}.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(f"data/processed/horseshoe_selection_{TAG}.csv", index=False)

    try:
        ref = {"l2": pd.read_csv("data/processed/cross_subject_metrics.csv"),
               "ard": pd.read_csv("data/processed/ard_metrics.csv")}
        l2 = ref["l2"][ref["l2"].model == "l2"].groupby("features").auc.median()
        ard = ref["ard"].groupby("features").auc.median()
    except FileNotFoundError:
        l2 = ard = None
    print("\nPLAIN variant: median over held-out subjects")
    print("features        horseshoe AUC   L2     ARD    | Brier  (0.250 = uninformative)")
    for name in SETS:
        d = df[(df.features == name) & (df.variant == "plain")]
        r = f"{l2[name]:.3f}   {ard[name]:.3f}" if l2 is not None else "n/a    n/a"
        print(f"{name:15s} {d.auc.median():.3f}           {r}  | {d.brier.median():.3f}")
    print("\nNOISE variant: how many columns does the prior let through?")
    for name in SETS:
        d = df[(df.features == name) & (df.variant == "noise")]
        n = X[name].shape[1]
        print(f"{name:15s} real selected {d.n_sel_real.median():.0f}/{n} "
              f"({d.n_sel_real.median() / n:.0%}) | noise selected "
              f"{d.n_sel_noise.median():.0f}/{n} ({d.n_sel_noise.median() / n:.0%}) "
              f"| AUC with noise present {d.auc.median():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    rr = np.random.default_rng(0)
    groups = list(df.groupby(["features", "variant"]))
    for i, ((f, v), d) in enumerate(groups):
        axes[0].scatter(i + rr.normal(0, 0.05, len(d)), d.auc, alpha=0.6)
        axes[0].hlines(d.auc.median(), i - 0.25, i + 0.25, color="k", lw=2)
    axes[0].set_xticks(range(len(groups)))
    axes[0].set_xticklabels([f"{f}\n{v}" for (f, v), _ in groups], fontsize=7)
    axes[0].axhline(0.5, color="gray", ls="--", lw=1)
    axes[0].set_ylabel("held-out-subject AUC")
    axes[0].set_title("Horseshoe, leave-one-subject-out")
    for name, pr in plain_preds.items():
        reliability(axes[1], pr, name)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set(xlabel="mean predicted P(right)", ylabel="observed fraction right",
                title="Reliability (pooled, quantile bins)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"horseshoe_{TAG}.png", dpi=150)


if __name__ == "__main__":
    main(quick=len(sys.argv) > 1 and sys.argv[1] == "quick")