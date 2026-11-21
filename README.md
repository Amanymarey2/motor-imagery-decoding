# Does brain connectivity add anything to motor-imagery decoding?

**A sparse Bayesian study on EEG from 30 healthy volunteers**

When you *imagine* squeezing your left or right fist, the brain activity over your motor cortex changes. A brain-computer interface (BCI) can try to read that change. This project asks a simple question with a careful answer:

> Can sparse Bayesian models, working on EEG **connectivity graphs**, decode imagined left vs right hand movement in a person they have never seen, and does connectivity tell us anything that plain **band power** does not?

![Summary figure](figures/hero_linkedin.png)

*A: weights of the Bayesian model on band power (red = predicts right-hand imagery, blue = left, black outline = selected). B: accuracy (AUC) on held-out subjects against the range you get with shuffled labels. C: how many real features and how many pure-noise columns each method selects.*

## Key findings

1. **Band power carries the signal.** Across subjects, a model trained on other people reaches a median AUC of about 0.60 (0.5 = chance) on a new person. It selects the sensorimotor channels a neuroscientist would expect, with the expected signs: left-hemisphere channels push toward "left imagery", right-hemisphere channels toward "right imagery".
2. **Connectivity adds little.** Graph metrics and wPLI edges reach about 0.54-0.55, only just above chance, and adding them to power does not beat adding the same number of columns of random noise.
3. **Single-trial wPLI is mostly estimator noise.** A surrogate test shows that most of a typical edge value is what you would get with no real coupling.
4. **The Bayesian horseshoe prior selected almost no noise** (0.1% of pure-noise columns), where ARD at its default settings selected about a quarter of them. It did not improve accuracy or calibration over ordinary L2 regression.
5. **Most of the project's value is in the controls:** shuffled labels, noise columns, a noise-floor test and paired comparisons showed that several early "differences" were noise.

All results are exploratory, from 30 healthy volunteers. Please read the [limitations](#limitations).

## Why I did this project

I am a physician moving into clinical data science, and I wanted a project that shows three things: that I understand the neuroscience behind the signal, that I can use graph theory and Bayesian modeling for a real question, and that I can validate honestly (no leakage, proper controls, negative results reported). A negative result that is well controlled is worth more than a positive one that is not.

## Key terms in plain words

| Term | What it means here |
|---|---|
| **Motor imagery** | Imagining a movement without moving. |
| **Mu (8-13 Hz) and beta (13-30 Hz) rhythms** | Brain oscillations over the motor cortex. They get weaker when you move or imagine moving. |
| **ERD (event-related desynchronization)** | That drop in mu/beta power. It is strongest on the side opposite to the hand (contralateral), so imagining the left hand drops power over the right hemisphere (C4) and vice versa. |
| **Band power** | How strong an oscillation is at each electrode. |
| **Surface Laplacian (CSD)** | A filter that sharpens the signal under each electrode and reduces how much one source spreads over many electrodes (volume conduction). |
| **wPLI (weighted phase lag index)** | A measure of how consistently two channels keep a time lag between their rhythms. It ignores zero-lag coupling, which is mostly volume conduction, not real communication. |
| **Graph / network** | Channels are *nodes*, wPLI values between pairs are *edges* (12 channels give 66 edges). |
| **Node strength, clustering, global efficiency** | Graph measures: how strongly a channel connects to the rest; whether its neighbors also connect to each other; how easily information could flow between any two nodes. |
| **Sparse Bayesian model** | A model that starts with a prior belief that most features are irrelevant and lets the data decide which ones are not. |
| **ARD and horseshoe** | Two such priors. ARD gives each feature its own learned shrinkage; the horseshoe shrinks most features hard but lets a few large ones escape. |
| **Leave-one-subject-out (LOSO)** | Train on 29 people, test on the 30th, repeat for everyone. It answers "does this work on a new person?" |
| **AUC** | Probability that the model ranks a random "right" trial above a random "left" trial. 0.5 is chance, 1.0 is perfect. |
| **Brier score** | How good the predicted probabilities are. A model that always says 50% scores 0.250. Lower is better. |

## Data

[EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (PhysioNet): 64-channel EEG, 160 Hz, recorded with BCI2000 from healthy volunteers.

- **Subjects:** 1-30. I fixed this range before looking at any result, so that subjects were not chosen by performance. None of the subjects usually excluded for irregular recordings (88, 92, 100, 104, 106, 109) fall in this range.
- **Runs:** 4, 8 and 12 (imagined left vs right fist). In these runs `T1` means left and `T2` means right. In other runs the codes mean something else, so the code asserts this mapping.
- **Timing, measured from the annotations and not assumed:** cue 4.1 s, rest 4.2 s, 15 cues per run, giving about 45 trials per subject (21-24 per class).

The data are downloaded automatically by MNE on first use. Please follow the dataset's license and citation terms on its PhysioNet page.

## Pipeline, and why each step

| # | Step | Why |
|---|---|---|
| 1 | **Band-pass 8-30 Hz, run by run** | Mu and beta are where motor imagery acts. Each run is filtered separately because filtering across the join of two recordings creates artifacts. |
| 2 | **Surface Laplacian on all 64 channels, then keep 12** | The Laplacian needs neighboring electrodes to compute its estimate, so it comes before channel selection. Electrode positions come from a standard template, not individually measured ones. |
| 3 | **Epochs from -1 to 4 s around the cue; analysis window 0.5-3.5 s** | The pre-cue period is needed as a baseline. The first 0.5 s contains a visual response to the cue, not imagery. The window ends before the cue disappears. |
| 4 | **Sanity check: do we see ERD?** (`erd.py`) | If a pipeline cannot recover a known effect, nothing downstream can be trusted. |
| 5 | **12 channels, chosen before looking at any decoding result** | C5, C3, C1, Cz, C2, C4, C6 (primary sensorimotor) and FC3, FC1, FCz, FC2, FC4 (premotor/SMA region), from the physiology. Choosing channels by test performance would leak the answer into the design. 12 channels give 66 edges per band. |
| 6 | **Features: log band power; C3-minus-C4 lateralization; single-trial wPLI graphs; graph metrics** | Power is the standard feature. wPLI was chosen over plain coherence or correlation because those are inflated by volume conduction. Weighted graph metrics avoid choosing a threshold, because tuning a threshold on the same data you test on is circular. |
| 7 | **Noise-floor test for wPLI** (`noise_floor.py`) | A single 3 s trial gives a very noisy phase estimate. Surrogate data (each channel taken from a different trial) show how large wPLI is with no real coupling. |
| 8 | **Models: L2, L1, ARD, horseshoe** | L1 and L2 are the ordinary baselines. ARD and the horseshoe are the sparse Bayesian models. |
| 9 | **Evaluation: leave-one-subject-out, everything fitted on training subjects only** | Prevents leakage. Scaling and the strength of regularization are chosen inside training folds, with inner folds also grouped by subject. Features are z-scored within each subject using no labels (this uses the test subject's unlabeled trials, so it is transductive). |
| 10 | **Controls everywhere:** shuffled labels, noise columns, paired tests | They turned out to matter most (see below). |

## Results

### 1. The pipeline recovers known physiology
Mu and beta power drop 10-30% after the cue at both C3 and C4. A lateralization index (C3-minus-C4 in left trials minus the same in right trials) is above zero across subjects (one-sided Wilcoxon, p ≈ 0.001-0.003 for the main baseline; median about 16 points in mu and 10 in beta). It was stable across baseline windows and task windows, with 20-21 of 30 subjects in the expected direction. About a third of subjects do not show the pattern, which is typical for motor imagery. The effect is clearer for right-hand imagery than left.

### 2. Single-trial wPLI sits mostly on its noise floor
The median real edge is 0.32 against 0.28-0.29 for surrogates. Only 10% (mu) and 16% (beta) of real edges exceed the surrogates' 95th percentile, where pure noise would give 5%. So there is some structure above the floor, but not much.

### 3. Cross-subject decoding (leave-one-subject-out)

| Feature set | L2 | ARD | Horseshoe | Reference with shuffled labels |
|---|---|---|---|---|
| Band power (24) | 0.602 | 0.622 | 0.605 | about 0.50 |
| C3-minus-C4 (2) | 0.582 | 0.583 | not run | about 0.51 |
| wPLI edges (132) | 0.548 | 0.515 | 0.545 | 0.48-0.49 |
| Graph metrics (50) | 0.542 | 0.566 | 0.477 | about 0.50 |

*Median AUC across held-out subjects. L1 gave exactly 0.500 for connectivity features (it shrank every weight to zero).*

Two lessons here. First, **differences between group medians of about 0.02-0.03 are noise**: adding pure-noise columns moved a median by that much. I compared models with paired per-subject tests instead, and none of the model differences were significant (for example horseshoe vs L2: p = 0.30-0.40). Second, the horseshoe's connectivity AUCs moved by up to 0.07 when I changed only optimizer settings, because a model shrunk to near-constant predictions ranks trials almost at random.

### 4. Does connectivity add anything beyond power?
No detectable benefit. Adding graph metrics or wPLI edges to power did no better than adding the same number of columns of random noise (all four pre-specified comparisons had p > 0.0125). With ARD, adding wPLI edges even did worse than adding noise.

### 5. What do the priors select? (the noise-column test)

| Method | Real power features selected | Pure-noise columns selected |
|---|---|---|
| ARD (default pruning) | 44% | 24% |
| Horseshoe | 19% | 0.1% |

ARD's selection depended strongly on a single constant it does not learn: changing the pruning threshold took it from keeping nothing to keeping 111 of 132 noise columns. In the cross-validation folds, 15 noise columns were selected in *every* fold, a reminder that being stable across folds is not evidence of importance, since training sets overlap by 28 of 29 subjects. The horseshoe was far more precise but also more conservative. On connectivity features it selected nothing, and because it does select power features, that zero is informative.

### 6. The weights make physiological sense
On power, every feature the horseshoe selected had the sign expected from contralateral ERD (left-hemisphere sites such as C3, FC3 and C1 negative; right-hemisphere sites such as C4 and C6 positive). The effects are small (a one-standard-deviation change in C4 mu power changes the log-odds by about 0.2), consistent with an AUC near 0.6.

## What I learned (in short)

- **Always include a negative control.** Pure-noise columns showed that ARD over-selects, and that "kept in every fold" means little.
- **Cross-validation has its own bias.** Within-subject decoding with shuffled labels gave AUCs of 0.44-0.46, not 0.50, so raw AUCs there cannot be compared across feature sets or with cross-subject results.
- **Compare with paired tests, not differences of medians.** Several apparent gains vanished.
- **A weak metric can hide a real effect.** My split-half reliability metric could not reliably detect even the known power effect, so its near-chance result for wPLI is inconclusive, not evidence of no information.
- **Bayesian does not automatically mean better.** Here it did not improve AUC or calibration over L2. Its benefit was cleaner selection with a measured false-positive rate.

## Limitations

- **30 healthy volunteers, one dataset, imagined fist movements only.** Nothing here transfers to patients (for example after stroke), where rhythms and lateralization can differ.
- **One connectivity measure (wPLI), broad bands (8-13, 13-30 Hz), 3 s windows, 12 channels.** A negative result here does not mean connectivity is uninformative in general. Narrower or individualized bands, longer windows or other measures could change the picture.
- **Single-trial wPLI is noisy**, as the noise-floor test shows.
- **Variational inference is approximate.** The horseshoe was fitted with mean-field variational inference, which tends to understate uncertainty. By my own criterion (loss change below 0.5% over the last third of training), only about 60-70% of folds (33% for one variant) had converged; the remaining drift was at most 2.6%.
- **Selection means "the 90% interval excludes zero in this fit"**, and marginal features can flip between fits. Selected features are sufficient for prediction, not proven drivers, and neighboring electrodes carry correlated information.
- **The noise controls are one random draw**, and shuffled-label references use only 5 shuffles (500 for the first reliability test), so they are descriptive, not exact p-values.
- **Exploratory analysis with many comparisons.** I fixed some analyses in advance (subjects, channels, primary noise-control contrasts) and report all results, but this is not a confirmatory study.
- **Template electrode positions** were used for the Laplacian, and within-subject z-scoring uses the test subject's unlabeled trials.

## Future work

- **Phase clustering (PLV)** as a sensitivity analysis. It is insensitive to time lags, a known weakness of lag-based measures (see Cohen 2015, *Effects of time lag and frequency matching on phase-based connectivity*), but it counts zero-lag coupling, which volume conduction produces, and its single-trial noise floor would need measuring first.
- Narrower or individualized frequency bands, longer windows, more channels.
- A hierarchical model with partial pooling across subjects, and MCMC instead of mean-field variational inference.
- Validation on patient data.

## Reproduce

```bash
pip install -r requirements.txt
```

Run everything from the repository root. Scripts import each other, and results are cached in `data/processed/`, so the order matters.

```bash
python src/data.py                      # download, filter, Laplacian, epoch, cache
python src/erd.py                       # ERD sanity check
python src/connectivity.py              # wPLI graphs + split-half reliability
python src/null_check.py                # label-shuffle reference for that reliability
python src/noise_floor.py               # wPLI noise floor
python src/graphs.py                    # graph metrics (runs a self-test first)
python src/power_control.py             # power positive control
python src/compare_features.py          # within-subject comparison
python src/auc_null.py                  # shuffled-label reference (within-subject)
python src/cross_subject.py             # leave-one-subject-out, L2 and L1
python src/cross_subject.py null        # shuffled-label reference
python src/ard_cross_subject.py         # ARD (add "null" for its reference)
python src/added_value.py               # does connectivity add beyond power?
python src/ard_noise_check.py           # does ARD select pure noise?
python src/ard_threshold_check.py       # sensitivity of ARD to its pruning threshold
python src/horseshoe_cross_subject.py   # horseshoe, all feature sets (slow)
python src/horseshoe_power_fit.py       # weights for the scalp maps
python src/make_figures.py              # figures/
```

The horseshoe runs are the slowest. Add `quick` as an argument to `horseshoe_cross_subject.py` for a fast smoke test.

## Repository map

| File | Purpose |
|---|---|
| `data.py` | Download, filter, Laplacian, epoching, caching |
| `erd.py` | ERD sanity check (mu/beta at C3 and C4) |
| `connectivity.py` | Single-trial wPLI graphs, split-half reliability |
| `null_check.py`, `noise_floor.py` | Controls for the wPLI features |
| `graphs.py` | Matrix rebuilding and graph metrics, with a hand-checkable self-test |
| `power_control.py` | Positive control with band power |
| `compare_features.py`, `auc_null.py` | Within-subject comparison and its shuffled-label reference |
| `cross_subject.py` | Leave-one-subject-out with L2 and L1 |
| `ard_cross_subject.py`, `ard_noise_check.py`, `ard_threshold_check.py` | ARD and its diagnostics |
| `added_value.py` | Added value of connectivity, with noise controls |
| `horseshoe_cross_subject.py`, `horseshoe_power_fit.py` | Regularized horseshoe logistic regression (NumPyro) |
| `make_figures.py` | Summary figures |

## Process note

The analysis code was developed with AI assistance (Claude, Anthropic). I ran the analyses, checked the outputs against hand calculations and controls where I could, and made and reviewed the analysis decisions described above. Any remaining errors are mine.

## Credits and license

- Dataset: Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR. *BCI2000: a general-purpose brain-computer interface (BCI) system.* IEEE Trans Biomed Eng 2004; hosted on PhysioNet: Goldberger A, et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 2000.
- Tools: MNE-Python, MNE-Connectivity, scikit-learn, NumPyro, JAX.
- Horseshoe prior: Piironen J, Vehtari A. *Sparsity information and regularization in the horseshoe and other shrinkage priors.* Electronic Journal of Statistics 2017.
- Code license: choose one before publishing (MIT is common).
- Author: Amany Marey.

*Not a medical device. Research and educational use only.*
