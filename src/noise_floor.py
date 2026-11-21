#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
noise_floor.py - how large is single-trial wPLI when there is NO coupling?

Our real edges have median ~0.32 and minimum ~0.03. Single-trial wPLI is
estimated from few time-frequency samples, and such estimates sit above zero
even for unrelated signals. This script measures that floor.

Surrogate: channel c of trial t is replaced by channel c of trial (t - c).
Every channel then comes from a DIFFERENT trial, so real coupling between
channels is destroyed, while each channel keeps its own spectrum and length.
The same wPLI code is run on the surrogate, so the floor is measured with
exactly the same estimator as the real edges.

Caveat: anything time-locked to the cue is shared across trials, so it can
survive the shuffle. That makes the surrogate floor a slightly CONSERVATIVE
(possibly high) estimate of pure noise.
"""
import mne
import numpy as np
import pandas as pd

from connectivity import BANDS, prepare, trial_wpli
from data import SUBJECTS, build_subject
from graphs import load_edges


def surrogate_epochs(ep):
    data = ep.get_data()                       # (trials, channels, times)
    n_trials, n_ch, _ = data.shape
    assert n_trials >= n_ch, "need at least as many trials as channels"
    sur = np.empty_like(data)
    for c in range(n_ch):
        # distinct shifts per channel -> any two channels always come from
        # different trials
        sur[:, c, :] = np.roll(data[:, c, :], shift=c, axis=0)
    return mne.EpochsArray(sur, ep.info, tmin=ep.tmin, verbose="ERROR")


def run():
    edges = load_edges()                       # cached real wPLI edges
    rows = []
    for s in SUBJECTS:
        ep = prepare(build_subject(s))
        real = edges[s][0]                     # (trials, 66, bands)
        floor = trial_wpli(surrogate_epochs(ep))
        for b, band in enumerate(BANDS):
            r, f = real[:, :, b].ravel(), floor[:, :, b].ravel()
            p95 = np.quantile(f, 0.95)
            rows.append(dict(subject=s, band=band,
                             real_median=np.median(r),
                             floor_median=np.median(f),
                             floor_p95=p95,
                             frac_real_above_p95=float((r > p95).mean())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    df.to_csv("data/processed/wpli_noise_floor.csv", index=False)
    print(df.groupby("band")[["real_median", "floor_median",
                              "floor_p95", "frac_real_above_p95"]]
            .median().round(3))
    print("\nIf real edges were pure noise, frac_real_above_p95 would be ~0.05.")
