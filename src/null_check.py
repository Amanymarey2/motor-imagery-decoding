"""
null_check.py - is the split-half contrast reliability above chance?

Real labels  -> r_contrast we measured (about 0.05-0.08).
Shuffled labels -> what the same metric gives when class information is destroyed.
If real is not clearly above the shuffled values, the data show no evidence of a
reproducible left/right difference in the 66-edge wPLI vector.
"""
from pathlib import Path

import numpy as np

from connectivity import BANDS, prepare, split_half_fast as split_half, trial_wpli
from data import SUBJECTS, build_subject

CACHE = Path("data/processed/wpli_features.npz")
N_REAL_SEEDS = 5      # different random half-splits with the REAL labels
N_SHUFFLES = 500       # different label shuffles


def load_features():
    """wPLI features per subject, computed once and cached."""
    if CACHE.exists():
        z = np.load(CACHE)
        return {s: (z[f"X{s}"], z[f"y{s}"]) for s in SUBJECTS}
    feats = {}
    for s in SUBJECTS:
        ep = prepare(build_subject(s))
        feats[s] = (trial_wpli(ep), ep.events[:, 2])
    np.savez(CACHE, **{f"X{s}": X for s, (X, _) in feats.items()},
                    **{f"y{s}": y for s, (_, y) in feats.items()})
    return feats


def group_median(feats, band_idx, shuffle_seed=None, split_seed=0):
    """Median over subjects of r_contrast for one band."""
    out = []
    for s in SUBJECTS:
        X, y = feats[s]
        if shuffle_seed is not None:
            y = np.random.default_rng(shuffle_seed + 1000 * s).permutation(y)
        out.append(split_half(X[:, :, band_idx], y, seed=split_seed)[1])
    return np.median(out)


if __name__ == "__main__":
    feats = load_features()
    for b, band in enumerate(BANDS):
        real = [group_median(feats, b, None, sd) for sd in range(N_REAL_SEEDS)]
        null = [group_median(feats, b, sh, sh) for sh in range(N_SHUFFLES)]
        p = (np.sum(np.array(null) >= np.mean(real)) + 1) / (N_SHUFFLES + 1)
        print(f"{band:5s} real: mean {np.mean(real):.3f} "
              f"(range {min(real):.3f} to {max(real):.3f}) | "
              f"shuffled: mean {np.mean(null):.3f} "
              f"(range {min(null):.3f} to {max(null):.3f}) | "
              f"permutation p = {p:.3f}")