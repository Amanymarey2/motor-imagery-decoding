#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
graphs.py - from single-trial wPLI edges to graph-theory features.

Pipeline in this file:
    cached wPLI edges (trials x 66 x 2 bands)
      -> symmetric 12x12 matrices (one weighted graph per trial per band)
      -> node strength (12), weighted clustering (12), global efficiency (1)
      -> feature matrix (trials x 50) with readable column names

Design decisions (put these in your README):
  * WEIGHTED graphs, no thresholding. 
  * wPLI lies in [0, 1], so weights are used as they are (not rescaled per
    graph), which keeps trials comparable with each other.
  * Strength + clustering = basic coupling and SEGREGATION; global efficiency =
    INTEGRATION. Local efficiency and betweenness are left out on purpose:
    in a 12-node subnetwork they mostly reflect electrode layout.
"""
from pathlib import Path

import numpy as np

from connectivity import BANDS, CHANNELS, prepare, trial_wpli
from data import SUBJECTS, build_subject

N_CH = len(CHANNELS)
EDGE_CACHE = Path("data/processed/wpli_features.npz")    # same file null_check.py uses
GRAPH_CACHE = Path("data/processed/graph_features.npz")


# ----------------------------------------------------------------------------
# 1. Edges: load or compute (cached)
# ----------------------------------------------------------------------------
def load_edges():
    """{subject: (edges (trials, 66, bands), labels (trials,))}"""
    if EDGE_CACHE.exists():
        z = np.load(EDGE_CACHE)
        return {s: (z[f"X{s}"], z[f"y{s}"]) for s in SUBJECTS}
    out = {}
    for s in SUBJECTS:
        ep = prepare(build_subject(s))
        out[s] = (trial_wpli(ep), ep.events[:, 2])
    np.savez(EDGE_CACHE,
             **{f"X{s}": X for s, (X, _) in out.items()},
             **{f"y{s}": y for s, (_, y) in out.items()})
    return out


# ----------------------------------------------------------------------------
# 2. Vector -> symmetric matrix
# ----------------------------------------------------------------------------
def edges_to_matrices(edges, n_ch=N_CH):
    """(trials, n_edges) -> (trials, n_ch, n_ch), symmetric, zero diagonal.
    Uses the SAME index order that trial_wpli used to build the vector."""
    rows, cols = np.tril_indices(n_ch, k=-1)
    assert edges.shape[1] == len(rows), "edge count does not match n_ch"
    W = np.zeros((edges.shape[0], n_ch, n_ch))
    W[:, rows, cols] = edges
    W[:, cols, rows] = edges
    return W


# ----------------------------------------------------------------------------
# 3. Graph metrics (batched over trials; W has shape (trials, n, n))
# ----------------------------------------------------------------------------
def strength(W):
    """Weighted degree: sum of a node's edge weights. -> (trials, n)"""
    return W.sum(axis=2)


def clustering(W):
    """Weighted clustering coefficient (Onnela et al. 2005), weights in [0, 1].

    C_i = sum_{j,h} (w_ij * w_jh * w_hi)^(1/3) / (k_i * (k_i - 1)),
    k_i = number of neighbours. Measures whether a node's neighbours are also
    connected to each other (local segregation). -> (trials, n)
    """
    Wc = np.cbrt(W)
    triangles = np.einsum("tij,tjk,tki->ti", Wc, Wc, Wc)
    k = (W > 0).sum(axis=2)
    denom = k * (k - 1)
    return np.divide(triangles, denom, out=np.zeros_like(triangles),
                     where=denom > 0)


def global_efficiency(W):
    """Mean inverse shortest-path length, with edge length = 1 / weight.
    Strong coupling = short distance. High value = information can move easily
    between any two nodes (integration). -> (trials,)"""
    T, n, _ = W.shape
    with np.errstate(divide="ignore"):
        d = np.where(W > 0, 1.0 / W, np.inf)
    d[:, np.arange(n), np.arange(n)] = 0.0
    for k in range(n):                          # Floyd-Warshall, batched
        d = np.minimum(d, d[:, :, k:k + 1] + d[:, k:k + 1, :])
    inv = np.zeros_like(d)
    off = ~np.eye(n, dtype=bool)
    inv[:, off] = 1.0 / d[:, off]               # 1/inf = 0 for unreachable pairs
    return inv.sum(axis=(1, 2)) / (n * (n - 1))


def graph_features(W):
    """One row per trial: [strength x n, clustering x n, global efficiency]."""
    return np.concatenate(
        [strength(W), clustering(W), global_efficiency(W)[:, None]], axis=1)


def feature_names():
    names = []
    for band in BANDS:
        names += [f"{band}_strength_{ch}" for ch in CHANNELS]
        names += [f"{band}_clustering_{ch}" for ch in CHANNELS]
        names += [f"{band}_global_efficiency"]
    return names


# ----------------------------------------------------------------------------
# 4. Self-test: hand-checkable answers. Read this BEFORE trusting the metrics.
# ----------------------------------------------------------------------------
def self_test():
    # Triangle (3 nodes), every edge weight 0.5.
    W = np.full((1, 3, 3), 0.5)
    W[0, np.arange(3), np.arange(3)] = 0.0
    # strength: two edges of 0.5 -> 1.0
    assert np.allclose(strength(W), 1.0)
    # clustering: sum over the 2 ordered neighbour pairs = 2 * 0.5 = 1, / (2*1) = 0.5
    assert np.allclose(clustering(W), 0.5)
    # efficiency: length 1/0.5 = 2 for every pair; direct path is shortest -> 1/2
    assert np.allclose(global_efficiency(W), 0.5)

    # Symmetry and round trip of the vector <-> matrix conversion.
    rng = np.random.default_rng(0)
    v = rng.random((4, N_CH * (N_CH - 1) // 2))
    M = edges_to_matrices(v)
    rows, cols = np.tril_indices(N_CH, k=-1)
    assert np.allclose(M, M.transpose(0, 2, 1))
    assert np.allclose(M[:, rows, cols], v)

    # A missing edge: zero weight means no path, and efficiency drops.
    W2 = W.copy()
    W2[0, 0, 1] = W2[0, 1, 0] = 0.0
    assert global_efficiency(W2)[0] < global_efficiency(W)[0]
    print("self-test passed")


# ----------------------------------------------------------------------------
# 5. Build and cache the feature matrix for every subject
# ----------------------------------------------------------------------------
def build_all():
    edges = load_edges()
    names = feature_names()
    F, Y = {}, {}
    for s, (X, y) in edges.items():
        blocks = [graph_features(edges_to_matrices(X[:, :, b]))
                  for b in range(X.shape[2])]
        F[s], Y[s] = np.concatenate(blocks, axis=1), y
        assert F[s].shape[1] == len(names)
    np.savez(GRAPH_CACHE, names=np.array(names),
             **{f"F{s}": f for s, f in F.items()},
             **{f"y{s}": y for s, y in Y.items()})
    return F, Y, names


if __name__ == "__main__":
    self_test()
    F, Y, names = build_all()
    s0 = SUBJECTS[0]
    print(f"subject {s0}: feature matrix {F[s0].shape}  (trials x features)")
    for band in BANDS:
        col = names.index(f"{band}_global_efficiency")
        vals = [F[s][:, col].mean() for s in SUBJECTS]
        print(f"{band}: mean global efficiency across subjects "
              f"{np.mean(vals):.3f} (range {min(vals):.3f} to {max(vals):.3f})")
