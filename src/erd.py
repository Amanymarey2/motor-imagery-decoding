#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
erd.py - sanity check: do we see contralateral mu/beta ERD during imagery?

Prediction:
    left-hand imagery  -> power drops over the RIGHT motor cortex (C4)
    right-hand imagery -> power drops over the LEFT  motor cortex (C3)

ERD is power change relative to each electrode's own pre-cue baseline:
    ERD% = (power - baseline_power) / baseline_power * 100     (negative = suppression)

One number per subject (the lateralization index) feeds the group test:
    lat = (ERD_C3 - ERD_C4)[left trials] - (ERD_C3 - ERD_C4)[right trials]
    Expected: lat > 0.
    Left trials : C4 more negative than C3 -> (C3 - C4) > 0
    Right trials: C3 more negative than C4 -> (C3 - C4) < 0
"""
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_morlet
from scipy import stats

from data import SUBJECTS, build_subject

CHANNELS = ["C3", "C4"]
BANDS = {"mu": (8, 13), "beta": (13, 30)}
FREQS = np.arange(8, 31, 1.0)       # 8..30 Hz in 1 Hz steps
N_CYCLES = FREQS / 2.0              # fewer cycles at low freq -> good time
                                    # resolution everywhere; common choice
# Baseline sits inside the pre-cue period, away from the epoch edge.
# At 8 Hz with 4 cycles the wavelet is ~0.8 s long, so the first ~0.4 s of the
# epoch (-1.0 to -0.6 s) is edge-distorted. Hence -0.6 to -0.1 s.
BASELINE = (-0.6, -0.1)
TASK_WINDOW = (0.5, 3.5)            # same window we will use for connectivity


def class_tfr(epochs, cls):
    """Average time-frequency power for one class, as fraction change from baseline."""
    ep = epochs[cls].copy().pick(CHANNELS)
    tfr = tfr_morlet(
        ep, freqs=FREQS, n_cycles=N_CYCLES, use_fft=True,
        return_itc=False, decim=2, average=True, verbose="ERROR",
    )
    tfr.apply_baseline(BASELINE, mode="percent", verbose="ERROR")  # fraction, not %
    return tfr


def band_curve(tfr, ch, band):
    """ERD% over time for one channel, averaged over the band's frequencies."""
    fmin, fmax = BANDS[band]
    f_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    ch_idx = tfr.ch_names.index(ch)
    return tfr.data[ch_idx, f_mask, :].mean(axis=0) * 100.0


def window_mean(curve, times):
    """Mean of a time course over the task window."""
    t_mask = (times >= TASK_WINDOW[0]) & (times <= TASK_WINDOW[1])
    return curve[t_mask].mean()


def subject_erd(subject):
    """Return (rows, curves) for one subject."""
    epochs = build_subject(subject)               # cached by data.py
    tfrs = {cls: class_tfr(epochs, cls) for cls in ("left", "right")}
    times = tfrs["left"].times

    rows, curves = [], {}
    for band in BANDS:
        win = {}
        for cls, tfr in tfrs.items():
            for ch in CHANNELS:
                curve = band_curve(tfr, ch, band)
                curves[(band, cls, ch)] = curve
                win[(cls, ch)] = window_mean(curve, times)
        lat = ((win[("left", "C3")] - win[("left", "C4")])
               - (win[("right", "C3")] - win[("right", "C4")]))
        rows.append(dict(subject=subject, band=band, lat=lat,
                         **{f"{cls}_{ch}": v for (cls, ch), v in win.items()}))
    return rows, curves, times


def run_group():
    all_rows, all_curves, times = [], [], None
    for s in SUBJECTS:
        rows, curves, times = subject_erd(s)
        all_rows += rows
        all_curves.append(curves)
    return pd.DataFrame(all_rows), all_curves, times


def summarize(df):
    """Group test: is lat > 0? Wilcoxon (no normality assumption, n is small)."""
    for band, g in df.groupby("band"):
        stat, p = stats.wilcoxon(g["lat"], alternative="greater")
        print(f"{band:5s}  median lat = {g['lat'].median():6.2f}  "
              f"n positive = {(g['lat'] > 0).sum()}/{len(g)}  "
              f"Wilcoxon one-sided p = {p:.4f}")


def plot_group(df, all_curves, times):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    # top row: group mean ERD time courses
    for ax, band in zip(axes[0], BANDS):
        for cls, ls in (("left", "-"), ("right", "--")):
            for ch, colour in (("C3", "tab:blue"), ("C4", "tab:red")):
                mean = np.mean([c[(band, cls, ch)] for c in all_curves], axis=0)
                ax.plot(times, mean, ls, color=colour, label=f"{cls} imagery, {ch}")
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvspan(*TASK_WINDOW, color="gray", alpha=0.1)
        ax.set(title=f"{band} ERD (group mean)", xlabel="time from cue (s)",
               ylabel="power change (%)")
    axes[0, 0].legend(fontsize=8)
    # bottom row: one dot per subject
    for ax, band in zip(axes[1], BANDS):
        vals = df.loc[df.band == band, "lat"].to_numpy()
        ax.scatter(np.random.default_rng(0).normal(0, 0.04, len(vals)), vals, alpha=0.7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(title=f"{band}: lateralization index per subject (expected > 0)",
               ylabel="lat (percentage points)", xticks=[])
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df, all_curves, times = run_group()
    df.to_csv("data/processed/erd_per_subject.csv", index=False)
    summarize(df)
    plot_group(df, all_curves, times).savefig("erd_check.png", dpi=150)

