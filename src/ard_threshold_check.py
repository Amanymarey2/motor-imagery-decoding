#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
ard_threshold_check.py - is "kept" by ARD a statement about a lenient threshold?

One fit on all 30 subjects (a diagnostic of selection behaviour, NOT a
cross-validated performance estimate). For several pruning thresholds, count
how many power columns and how many pure-noise columns survive, and how large
their coefficients are (on the standardized scale).

If noise columns survive with tiny coefficients, "kept" mostly reflects the
threshold. If they survive with coefficients as large as the power columns',
the over-selection is real.
"""
import numpy as np
from sklearn.linear_model import ARDRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from added_value import N_POWER, build_sets
from cross_subject import RIGHT


def median_abs(v):
    nz = v[v != 0]
    return float(np.median(np.abs(nz))) if len(nz) else float("nan")


if __name__ == "__main__":
    sets, y, g = build_sets()
    X, target = sets["power+noise132"], (y == RIGHT).astype(float)
    print("threshold_lambda | power kept (of 24) | noise kept (of 132) | "
          "median |coef| power | noise")
    for tl in (1e2, 1e3, 1e4, 1e5):
        model = make_pipeline(
            StandardScaler(),
            ARDRegression(max_iter=300, tol=1e-3, threshold_lambda=tl),
        ).fit(X, target)
        coef = model[-1].coef_
        cp, cn = coef[:N_POWER], coef[N_POWER:]
        print(f"{tl:15.0e} | {(cp != 0).sum():18d} | {(cn != 0).sum():19d} | "
              f"{median_abs(cp):19.4f} | {median_abs(cn):.4f}")
