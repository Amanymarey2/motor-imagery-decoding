#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
data.py - download and preprocess the EEG Motor Movement/Imagery dataset (eegmmidb).

Goal: for each subject, produce clean epochs for imagined LEFT vs RIGHT fist
(runs 4, 8, 12) and cache them to disk so later steps never re-download.

Order of operations per run, and why:
    1. read EDF            -> raw signal + annotations (T0/T1/T2)
    2. standardize names   -> the dataset uses names like 'C3..'; we need 'C3'
    3. set montage         -> 3D electrode positions; REQUIRED for the Laplacian
    4. band-pass 8-30 Hz   -> mu (8-13) + beta (13-30), where motor imagery acts
    5. surface Laplacian   -> reduces volume conduction (one source spreading
                              over many electrodes), which would fake connectivity
    6. epoch around cues   -> one segment per left/right cue

Each run is processed separately and the EPOCHS are joined at the end.
Joining raw runs first would put filter artifacts at the run boundaries.
"""
import time
from pathlib import Path

import mne
import pandas as pd
from mne.datasets import eegbci

# ----------------------------------------------------------------------------
# Configuration: every analysis choice lives here, so it can be documented.
# ----------------------------------------------------------------------------
IMAGERY_LR_RUNS = [4, 8, 12]            # imagined left vs right fist
EXCLUDED_SUBJECTS = [88, 92, 100, 104, 106, 109]   # irregular recordings
SUBJECTS = [s for s in range(1, 31) if s not in EXCLUDED_SUBJECTS]  # fixed in advance

L_FREQ, H_FREQ = 8.0, 30.0              # mu + beta
EPOCH_TMIN, EPOCH_TMAX = -1.0, 4.0      # wide on purpose: pre-cue period is
                                        # kept for the ERD check; crop to
                                        # 0.5-3.5 s later for connectivity
EVENT_ID = {"left": 1, "right": 2}      # T1 -> left, T2 -> right in these runs

OUT_DIR = Path("data/processed")


def load_run(subject, run):
    """Download (cached by MNE in ~/mne_data) and read one run."""
    # Downloads from PhysioNet sometimes time out; retry a few times
    for attempt in range(1, 4):
        try:
            path = eegbci.load_data(subject, [run], update_path=True)[0]
            break
        except Exception as err:
            if attempt == 3:
                raise
            print(f"S{subject} run {run}: download failed ({err}); retry {attempt}")
            time.sleep(5 * attempt)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eegbci.standardize(raw)
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        on_missing="ignore",
    )
    return raw


def preprocess_run(raw):
    """Filter, then apply the surface Laplacian (current source density)."""
    raw.filter(L_FREQ, H_FREQ, verbose="ERROR")
    raw = mne.preprocessing.compute_current_source_density(raw, verbose="ERROR")
    return raw


def epoch_run(raw, subject, run):
    """Cut epochs around left/right cues, skipping rest (T0)."""
    events, _ = mne.events_from_annotations(
        raw, event_id={"T1": 1, "T2": 2}, verbose="ERROR"
    )
    epochs = mne.Epochs(
        raw, events, event_id=EVENT_ID,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None,          # no baseline correction: data are band-passed
        preload=True, verbose="ERROR",
    )
    # metadata lets us track subject/run later (needed for subject-wise
    # splits and for any early-vs-late-run comparison)
    epochs.metadata = pd.DataFrame(
        {"subject": subject, "run": run, "label": epochs.events[:, 2]}
    )
    return epochs


def build_subject(subject, force=False):
    """Return cached epochs for one subject, building them if needed."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"S{subject:03d}-epo.fif"
    if out_file.exists() and not force:
        return mne.read_epochs(out_file, preload=True, verbose="ERROR")

    run_epochs = []
    for run in IMAGERY_LR_RUNS:
        raw = load_run(subject, run)
        if raw.info["sfreq"] != 160:
            raise ValueError(f"S{subject} run {run}: sfreq={raw.info['sfreq']}")
        raw = preprocess_run(raw)
        run_epochs.append(epoch_run(raw, subject, run))

    epochs = mne.concatenate_epochs(run_epochs, verbose="ERROR")
    epochs.save(out_file, overwrite=True, verbose="ERROR")
    return epochs


def build_all():
    """Process every subject; record failures instead of hiding them."""
    log = []
    for subject in SUBJECTS:
        try:
            ep = build_subject(subject)
            n_left = int((ep.events[:, 2] == EVENT_ID["left"]).sum())
            n_right = int((ep.events[:, 2] == EVENT_ID["right"]).sum())
            log.append(dict(subject=subject, status="ok",
                            n_left=n_left, n_right=n_right, note=""))
        except Exception as err:      # keep going, but write the reason down
            log.append(dict(subject=subject, status="failed",
                            n_left=0, n_right=0, note=str(err)))
    log = pd.DataFrame(log)
    log.to_csv(OUT_DIR / "subject_log.csv", index=False)
    return log


if __name__ == "__main__":
    print(build_all().to_string(index=False))


# In[ ]:




