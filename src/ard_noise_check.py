#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
ard_noise_check.py - does ARD keep columns that are PURE NOISE?

The noise columns are independent of everything, so a well-behaved sparse
prior should prune nearly all of them. How many survive is a direct measure
of ARD's false-positive selection on this data.

Benchmark: the noise matrix is fixed, so a few columns will correlate with the
labels by chance in the pooled data (with 132 columns, a handful). A handful
is expected. Dozens would mean ARD is over-selecting.
"""
import numpy as np

import pandas as pd
from added_value import N_POWER, build_sets
from ard_cross_subject import loso_ard

if __name__ == "__main__":
    sets, y, g = build_sets()
    rows = []
    for name in ("power+noise50", "power+noise132"):
        _, sel = loso_ard(sets[name], y, g)          # (folds, features) boolean
        noise = sel[:, N_POWER:]
        per_fold = noise.sum(axis=1)
        rows += [dict(set=name, fold=k, noise_kept=int(v), n_noise=noise.shape[1],
                      power_kept=int(sel[k, :N_POWER].sum()))
                 for k, v in enumerate(per_fold)]
        freq = noise.mean(axis=0)
        print(f"{name}: noise columns kept per fold: median {np.median(per_fold):.0f} "
              f"(min {per_fold.min()}, max {per_fold.max()}) of {noise.shape[1]}; "
              f"power columns kept: median {np.median(sel[:, :N_POWER].sum(axis=1)):.0f}"
              f" of {N_POWER}")
        print(f"   {(freq > 0.5).sum()} noise columns are kept in more than half "
              f"of the folds; {(freq == 1.0).sum()} in every fold")
    pd.DataFrame(rows).to_csv("data/processed/ard_noise_selection.csv", index=False)