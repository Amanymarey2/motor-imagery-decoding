#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
auc_null.py - what group-median AUC does each feature set give when the class
labels carry no information?

For each shuffle: permute the labels inside every subject, rerun the same
within-subject decoding, and take the median AUC over subjects. The spread of
those group medians is the reference for the real values
(graph 0.477, lateralization 0.542, power 0.487, wPLI 0.503).
"""
import numpy as np

from compare_features import feature_sets, subject_auc
from data import SUBJECTS

N_SHUFFLES = 10      # each shuffle = 30 subjects x 4 sets x 5 fits; raise if time allows


def run():
    data = feature_sets()
    names = list(next(iter(data.values()))[0])
    real = {n: [] for n in names}
    null = {n: [] for n in names}
    for s in SUBJECTS:
        sets, y = data[s]
        for n, X in sets.items():
            real[n].append(subject_auc(X, y))
    for k in range(N_SHUFFLES):
        per_set = {n: [] for n in names}
        for s in SUBJECTS:
            sets, y = data[s]
            y_shuf = np.random.default_rng(k + 1000 * s).permutation(y)
            for n, X in sets.items():
                per_set[n].append(subject_auc(X, y_shuf, n_repeats=1))
        for n in names:
            null[n].append(np.median(per_set[n]))
        print(f"shuffle {k + 1}/{N_SHUFFLES} done", flush=True)
    print("\nfeature set      real median | shuffled group medians: mean (range) | n >= real")
    for n in names:
        r, z = np.median(real[n]), np.array(null[n])
        print(f"{n:15s}  {r:.3f}       | {z.mean():.3f} ({z.min():.3f} to {z.max():.3f})"
              f" | {int((z >= r).sum())}/{N_SHUFFLES}")


if __name__ == "__main__":
    run()
