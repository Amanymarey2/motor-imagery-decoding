#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
make_figures.py - portfolio / LinkedIn figures from the saved result files.

Hero image (one picture, three ideas):
  A  scalp maps of the horseshoe weights for band power (mu, beta): the weights
     show the contralateral ERD pattern (left-hemisphere negative, right positive).
  B  per-subject AUC by feature set (L2, leave-one-subject-out) against the range of
     shuffled-label medians.
  C  selection behaviour: % of real power features and % of pure-noise columns
     selected by ARD (default pruning) and by the horseshoe.
Also saves each panel as a square image for a carousel.

Inputs (data/processed/), a missing file only blanks or flags its panel:
  horseshoe_power_weights.csv   <- python horseshoe_power_fit.py
  cross_subject_metrics.csv     <- python cross_subject.py
  cross_subject_null.csv        <- python cross_subject.py null
  ard_noise_selection.csv       <- python ard_noise_check.py
  horseshoe_metrics_all.csv     <- python horseshoe_cross_subject.py

Group-level results only: no individual subject is labelled.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Patch, Polygon, Rectangle

from connectivity import CHANNELS

D = Path("data/processed")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

SETS = ["power_12ch", "graph_metrics", "wpli_edges"]
LABELS = {"power_12ch": "Band power\n(24 features)",
          "graph_metrics": "Graph metrics\n(50 features)",
          "wpli_edges": "wPLI edges\n(132 features)"}
COLORS = {"power_12ch": "#1f77b4", "graph_metrics": "#ff7f0e", "wpli_edges": "#2ca02c"}
N_REAL = {"power_12ch": 24, "graph_metrics": 50, "wpli_edges": 132}
HEADLINE = "Imagined hand movement from EEG: power carries the signal, connectivity adds little"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})


def read(name):
    p = D / name
    if not p.exists():
        print(f"[missing] {p}")
        return None
    return pd.read_csv(p)


def blank(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            color="0.4")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Panel A: scalp maps of the weights
# ---------------------------------------------------------------------------
def electrode_xy():
    """2D top-down positions (nose up), radius = angle from the vertex."""
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    P = np.array([pos[c] for c in CHANNELS], dtype=float)
    i = {c: k for k, c in enumerate(CHANNELS)}
    if P[i["C4"], 0] < P[i["C3"], 0]:          # make sure right is +x
        P[:, 0] *= -1
    if P[i["FCz"], 1] < P[i["Cz"], 1]:         # make sure front is +y
        P[:, 1] *= -1
    rho = np.linalg.norm(P, axis=1)
    theta = np.arccos(P[:, 2] / rho)
    phi = np.arctan2(P[:, 1], P[:, 0])
    r = theta / (np.pi / 2)
    return np.c_[r * np.cos(phi), r * np.sin(phi)]


def draw_head(ax):
    ax.add_patch(Circle((0, 0), 1, fill=False, lw=2, color="0.35"))
    ax.add_patch(Polygon([[-0.12, 0.985], [0, 1.13], [0.12, 0.985]],
                         fill=False, lw=2, color="0.35"))
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.1, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_scalp(ax, w, band, vmax):
    draw_head(ax)
    xy = electrode_xy()
    d = w.assign(ch=w.feature.str.rsplit("_", n=1).str[0],
                 band=w.feature.str.rsplit("_", n=1).str[1])
    d = d[d.band == band].set_index("ch")
    mean = np.array([d.loc[c, "mean"] for c in CHANNELS])
    sel = np.array([bool(d.loc[c, "selected"]) for c in CHANNELS])
    ax.scatter(xy[:, 0], xy[:, 1], s=60 + 2200 * np.abs(mean), c=mean,
               cmap="RdBu_r", vmin=-vmax, vmax=vmax, zorder=3,
               edgecolors=["black" if s else "0.75" for s in sel],
               linewidths=[2.5 if s else 0.8 for s in sel])
    for (x, y), c in zip(xy, CHANNELS):
        # frontal-central labels above their markers, central ones below,
        # so labels never sit on top of a marker
        dy = 0.18 if c.startswith("FC") else -0.2
        ax.text(x, y + dy, c, ha="center", va="center", fontsize=9, color="0.2",
                zorder=4, bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                    ec="none", alpha=0.8))
    ax.set_title(f"{band} power", fontsize=13)


# ---------------------------------------------------------------------------
# Panel B: AUC against the shuffled-label range
# ---------------------------------------------------------------------------
def panel_auc(ax):
    m = read("cross_subject_metrics.csv")
    if m is None:
        return blank(ax, "cross_subject_metrics.csv missing")
    m = m[m.model == "l2"]
    null = read("cross_subject_null.csv")
    rng = np.random.default_rng(0)
    for i, s in enumerate(SETS):
        v = m[m.features == s].auc.to_numpy()
        if null is not None:
            z = null[null.features == s].median_auc
            ax.add_patch(Rectangle((i - 0.38, z.min()), 0.76, z.max() - z.min(),
                                   color="0.85", zorder=0))
        ax.scatter(i + rng.normal(0, 0.06, len(v)), v, s=30, alpha=0.65,
                   color=COLORS[s], zorder=2)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color="k", lw=3, zorder=3)
        ax.text(i, 0.99, f"median {np.median(v):.2f}", ha="center", fontsize=11)
    ax.axhline(0.5, color="0.5", ls="--", lw=1)
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([LABELS[s] for s in SETS], fontsize=10)
    ax.set_ylim(0.2, 1.03)
    ax.set_ylabel("AUC on a held-out subject (one dot each)")
    if null is not None:
        ax.legend(handles=[Patch(color="0.85",
                                 label="range of shuffled-label medians")],
                  loc="lower left", fontsize=9, frameon=False)
    else:
        ax.text(0.5, 0.03, "no chance band: run cross_subject.py null",
                transform=ax.transAxes, ha="center", fontsize=8, color="firebrick")


# ---------------------------------------------------------------------------
# Panel C: real features vs pure-noise columns selected
# ---------------------------------------------------------------------------
def selection_rates():
    ard, hs = read("ard_noise_selection.csv"), read("horseshoe_metrics_all.csv")
    if ard is None or hs is None:
        return None
    h = hs[hs.variant == "noise"].copy()
    h["n_noise"] = h.features.map(N_REAL)
    noise = {"ARD": ard.noise_kept.sum() / ard.n_noise.sum() * 100,
             "Horseshoe": h.n_sel_noise.sum() / h.n_noise.sum() * 100}
    real = None
    if "power_kept" in ard.columns:
        hp = h[h.features == "power_12ch"]
        n = N_REAL["power_12ch"]
        real = {"ARD": ard.power_kept.sum() / (n * len(ard)) * 100,
                "Horseshoe": hp.n_sel_real.sum() / (n * len(hp)) * 100}
    else:
        print("[note] ard_noise_selection.csv has no power_kept column: "
              "rerun ard_noise_check.py to show real-feature selection")
    return real, noise


def panel_noise(ax):
    rates = selection_rates()
    if rates is None:
        return blank(ax, "selection inputs missing")
    real, noise = rates
    groups = ([("Real power\nfeatures", real)] if real else []) + \
             [("Pure-noise\ncolumns", noise)]
    width, top = 0.36, 5.0
    for gi, (label, r) in enumerate(groups):
        for k, (method, color) in enumerate((("ARD", "#d62728"),
                                             ("Horseshoe", "#2ca02c"))):
            x = gi + (k - 0.5) * width
            ax.bar(x, r[method], width=width * 0.95, color=color,
                   label=method if gi == 0 else None)
            ax.text(x, r[method] + 1.0, f"{r[method]:.0f}%" if r[method] >= 1
                    else f"{r[method]:.1f}%", ha="center", fontsize=12,
                    fontweight="bold")
            top = max(top, r[method])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("% of columns selected")
    ax.set_ylim(0, top * 1.22)
    ax.legend(frameon=False, loc="upper right", fontsize=10)


# ---------------------------------------------------------------------------
# Compositions
# ---------------------------------------------------------------------------
NOISE_NOTE = ("ARD: default pruning; noise appended to the power features. "
              "Horseshoe: noise appended to each feature set.")


def hero(w):
    fig = plt.figure(figsize=(13, 7.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.2, 0.9], hspace=0.2,
                          wspace=0.32, top=0.78, bottom=0.2, left=0.04, right=0.97)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    b, c = fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])
    if w is None:
        blank(a1, "weights missing")
        blank(a2, "")
    else:
        vmax = np.abs(w["mean"]).max()
        panel_scalp(a1, w, "mu", vmax)
        panel_scalp(a2, w, "beta", vmax)
    panel_auc(b)
    panel_noise(c)
    title = dict(fontsize=12.5, fontweight="bold", va="bottom", ha="left")
    fig.text(0.04, 0.835, "A  Weights show contralateral ERD", **title)
    fig.text(0.37, 0.835, "B  Power clearly beats chance;\n    connectivity only barely",
             **title)
    fig.text(0.735, 0.835, "C  What do the priors select?", **title)
    fig.suptitle(HEADLINE, fontsize=15.5, fontweight="bold", y=0.975)
    fig.text(0.04, 0.125, "Maps: red = predicts right-hand imagery, blue = predicts left; "
             "circle size = weight size; black outline = interval excludes zero.",
             fontsize=9.5, color="0.3")
    fig.text(0.04, 0.09, NOISE_NOTE, fontsize=9.5, color="0.3")
    fig.text(0.04, 0.045, "PhysioNet EEG Motor Movement/Imagery dataset, 30 healthy "
             "volunteers, imagined left vs right fist, leave-one-subject-out. "
             "Group-level and descriptive; not for clinical use.",
             fontsize=9, color="0.4")
    return fig


def carousel(w):
    pages = {}
    fig, axes = plt.subplots(2, 1, figsize=(8, 8.4))
    if w is None:
        for ax in axes:
            blank(ax, "weights missing")
    else:
        vmax = np.abs(w["mean"]).max()
        panel_scalp(axes[0], w, "mu", vmax)
        panel_scalp(axes[1], w, "beta", vmax)
    fig.suptitle("Weights show contralateral ERD", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.02, "red = predicts right imagery, blue = predicts left; "
             "size = weight size; black outline = interval excludes zero",
             ha="center", fontsize=9.5, color="0.3")
    fig.subplots_adjust(top=0.92, bottom=0.06, hspace=0.05)
    pages["carousel_1_scalp"] = fig

    fig, ax = plt.subplots(figsize=(8, 8))
    panel_auc(ax)
    ax.set_title("Power clearly beats chance; connectivity only barely",
                 fontsize=14, fontweight="bold")
    pages["carousel_2_auc"] = fig

    fig, ax = plt.subplots(figsize=(8, 8))
    panel_noise(ax)
    ax.set_title("What do the priors select?", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.03, NOISE_NOTE, ha="center", fontsize=8.5, color="0.35",
             wrap=True)
    fig.subplots_adjust(bottom=0.14)
    pages["carousel_3_selection"] = fig
    return pages


if __name__ == "__main__":
    weights = read("horseshoe_power_weights.csv")
    hero(weights).savefig(OUT / "hero_linkedin.png", dpi=150)
    for name, fig in carousel(weights).items():
        fig.savefig(OUT / f"{name}.png", dpi=150)
    print(f"saved to {OUT}/")