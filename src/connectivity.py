"""
connectivity.py - single-trial wPLI graphs on the 12 chosen channels, plus a
split-half reliability check.

Channel choice (fixed BEFORE looking at any decoding result):
    primary sensorimotor : C5 C3 C1 Cz C2 C4 C6   (mu/beta ERD over the hand area)
    premotor / SMA region: FC3 FC1 FCz FC2 FC4    (motor planning areas)
12 channels -> 66 edges per band.

Why wPLI: it ignores zero-lag phase coupling, which is what volume conduction
produces, so it is less inflated by one source seen by many electrodes.

Why a reliability check: a single 3 s trial gives a noisy wPLI estimate. If the
edge values are not reproducible, no model can rescue them.
"""
import mne
import numpy as np
import pandas as pd
from mne_connectivity import spectral_connectivity_time
from scipy.stats import rankdata, spearmanr

from data import SUBJECTS, EVENT_ID, build_subject

CHANNELS = ["C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "FC3", "FC1", "FCz", "FC2", "FC4"]
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
FREQS = np.arange(8.0, 31.0, 1.0)
TASK_WINDOW = (0.5, 3.5)


def prepare(epochs):
    """Subset channels and crop to the task window (CSD was applied on all 64
    channels in data.py, before subsetting - the Laplacian needs neighbours)."""
    missing = set(CHANNELS) - set(epochs.ch_names)
    assert not missing, f"channels not found: {missing}"
    ep = epochs.copy().pick(CHANNELS).crop(*TASK_WINDOW)
    return ep


def trial_wpli(epochs):
    """One wPLI value per trial, per edge, per band.

    Returns array (n_trials, n_edges, n_bands), edges in lower-triangle order.
    """
    data = epochs.get_data()                       # (n_trials, n_ch, n_times)
    n_ch = data.shape[1]
    con = spectral_connectivity_time(
        data, freqs=FREQS, method="wpli", average=False,
        sfreq=epochs.info["sfreq"],
        fmin=tuple(b[0] for b in BANDS.values()),
        fmax=tuple(b[1] for b in BANDS.values()),
        faverage=True, mode="cwt_morlet", n_cycles=FREQS / 2.0,
        verbose="ERROR",
    )
    dense = con.get_data(output="dense")           # (n_trials, n_ch, n_ch, n_bands)
    # CHECK THIS on your first run: shape, and that the values live in the
    # lower triangle only (upper should be zeros).
    rows, cols = np.tril_indices(n_ch, k=-1)
    return dense[:, rows, cols, :]


def split_half(X, labels, n_splits=50, seed=0):
    """Reliability of (a) the mean edge pattern and (b) the left-vs-right
    contrast, using random stratified half splits of the trials.

    X: (n_trials, n_edges) for ONE band.  Returns median Spearman r for each.
    """
    rng = np.random.default_rng(seed)
    r_pattern, r_contrast = [], []
    for _ in range(n_splits):
        halves = {}
        for cls in (EVENT_ID["left"], EVENT_ID["right"]):
            idx = rng.permutation(np.where(labels == cls)[0])
            h = len(idx) // 2
            halves[(cls, "a")], halves[(cls, "b")] = idx[:h], idx[h:2 * h]
        L, R = EVENT_ID["left"], EVENT_ID["right"]
        mean_a = X[np.r_[halves[(L, "a")], halves[(R, "a")]]].mean(axis=0)
        mean_b = X[np.r_[halves[(L, "b")], halves[(R, "b")]]].mean(axis=0)
        diff_a = X[halves[(L, "a")]].mean(0) - X[halves[(R, "a")]].mean(0)
        diff_b = X[halves[(L, "b")]].mean(0) - X[halves[(R, "b")]].mean(0)
        r_pattern.append(spearmanr(mean_a, mean_b)[0])
        r_contrast.append(spearmanr(diff_a, diff_b)[0])
    return np.median(r_pattern), np.median(r_contrast)


def split_half_fast(X, labels, n_splits=50, seed=0):
    """Same as split_half (same random splits, same Spearman r), but all
    correlations are computed at once instead of one scipy call per split."""
    rng = np.random.default_rng(seed)
    L, R = EVENT_ID["left"], EVENT_ID["right"]
    pools = {c: np.where(labels == c)[0] for c in (L, R)}
    mean_a, mean_b, diff_a, diff_b = [], [], [], []
    for _ in range(n_splits):
        h = {}
        for c in (L, R):
            idx = rng.permutation(pools[c])
            n = len(idx) // 2
            h[(c, "a")], h[(c, "b")] = idx[:n], idx[n:2 * n]
        mean_a.append(X[np.r_[h[(L, "a")], h[(R, "a")]]].mean(axis=0))
        mean_b.append(X[np.r_[h[(L, "b")], h[(R, "b")]]].mean(axis=0))
        diff_a.append(X[h[(L, "a")]].mean(0) - X[h[(R, "a")]].mean(0))
        diff_b.append(X[h[(L, "b")]].mean(0) - X[h[(R, "b")]].mean(0))

    def row_spearman(A, B):
        A = rankdata(np.array(A), axis=1)      # Spearman = Pearson on ranks
        B = rankdata(np.array(B), axis=1)
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)
        return (A * B).sum(1) / np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))

    return (np.median(row_spearman(mean_a, mean_b)),
            np.median(row_spearman(diff_a, diff_b)))


def run_group():
    rows = []
    for s in SUBJECTS:
        ep = prepare(build_subject(s))
        feats = trial_wpli(ep)                     # (trials, edges, bands)
        labels = ep.events[:, 2]
        for b, band in enumerate(BANDS):
            r_pat, r_con = split_half(feats[:, :, b], labels)
            rows.append(dict(subject=s, band=band,
                             r_pattern=r_pat, r_contrast=r_con))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_group()
    df.to_csv("data/processed/wpli_reliability.csv", index=False)
    print(df.groupby("band")[["r_pattern", "r_contrast"]]
            .agg(["median", "min", "max"]).round(2))