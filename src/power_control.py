#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
power_control.py - positive control: do band-POWER features carry a reproducible
left-vs-right contrast, as the ERD result suggests they should?

Part A  C3/C4 lateralization (pre-specified from physiology, nothing tuned):
        lat_trial = log10 P(C3) - log10 P(C4)   per trial, per band
        Left imagery suppresses C4 -> lat is larger on left trials.
        AUC = P(lat_left > lat_right); expected > 0.5. One value per subject.

Part B  Same split-half + label-shuffle test used for wPLI, on log power at all
        12 channels (12 features per band instead of 66 edges).
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from connectivity import BANDS, CHANNELS, prepare, split_half_fast
from data import EVENT_ID, SUBJECTS, build_subject

N_SHUFFLES = 200
LEFT, RIGHT = EVENT_ID["left"], EVENT_ID["right"]


def trial_logpower(ep):
    """Single-trial log10 band power: (n_trials, n_channels, n_bands)."""
    ps = ep.compute_psd(method="welch", fmin=8, fmax=30,
                        n_fft=256, n_overlap=128, verbose="ERROR")
    psd, freqs = ps.get_data(return_freqs=True)     # (trials, ch, freqs)
    cols = []
    for lo, hi in BANDS.values():
        m = (freqs >= lo) & (freqs <= hi)
        cols.append(np.log10(psd[:, :, m].mean(axis=2)))
    return np.stack(cols, axis=-1)


def auc(a, b):
    """P(a > b), ties counted as half (Mann-Whitney U / (n_a * n_b))."""
    return mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b))


def load_power():
    feats = {}
    for s in SUBJECTS:
        ep = prepare(build_subject(s))
        feats[s] = (trial_logpower(ep), ep.events[:, 2], ep.ch_names)
    return feats


def part_a(feats):
    rows = []
    for s, (P, y, names) in feats.items():
        i3, i4 = names.index("C3"), names.index("C4")
        for b, band in enumerate(BANDS):
            lat = P[:, i3, b] - P[:, i4, b]
            rows.append(dict(subject=s, band=band,
                             auc=auc(lat[y == LEFT], lat[y == RIGHT])))
    df = pd.DataFrame(rows)
    for band, g in df.groupby("band"):
        _, p = wilcoxon(g["auc"] - 0.5, alternative="greater")
        print(f"A  {band:5s} median AUC = {g['auc'].median():.3f}  "
              f"n AUC>0.5 = {(g['auc'] > 0.5).sum()}/{len(g)}  Wilcoxon p = {p:.4f}")
    return df


def group_r(feats, b, shuffle_seed=None, split_seed=0):
    out = []
    for s, (P, y, _) in feats.items():
        if shuffle_seed is not None:
            y = np.random.default_rng(shuffle_seed + 1000 * s).permutation(y)
        out.append(split_half_fast(P[:, :, b], y, seed=split_seed)[1])
    return np.median(out)


def part_b(feats):
    for b, band in enumerate(BANDS):
        real = [group_r(feats, b, None, sd) for sd in range(5)]
        null = [group_r(feats, b, sh, sh) for sh in range(N_SHUFFLES)]
        k = int(np.sum(np.array(null) >= np.mean(real)))
        print(f"B  {band:5s} real r_contrast mean {np.mean(real):.3f} | "
              f"shuffled mean {np.mean(null):.3f} "
              f"(range {min(null):.3f} to {max(null):.3f}) | "
              f"{k} of {N_SHUFFLES} shuffles >= real | "
              f"p = {(k + 1) / (N_SHUFFLES + 1):.3f}")


if __name__ == "__main__":
    feats = load_power()
    part_a(feats).to_csv("data/processed/power_lat_auc.csv", index=False)
    part_b(feats)
