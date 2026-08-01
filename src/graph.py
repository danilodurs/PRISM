"""Prior-agnostic graph operations: mask construction and structural controls
(degree-preserving random rewire, fully randomized edges, sign-flip) for
isolating whether a prior's real structure matters, not just "having a
graph." Every function here operates on a generic (source, target, weight)
edge list and a shared gene universe -- nothing in this module knows whether
the edges came from a TF-target graph or a co-accessibility graph. Prior
modules (src/priors/*.py) are responsible for producing edges in this shape
before calling in; this module does the rest, identically, for any prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_EDGES_PER_SOURCE = 5


def filter_edges(edges: pd.DataFrame, genes: list[str], min_edges_per_source: int = MIN_EDGES_PER_SOURCE) -> pd.DataFrame:
    """Restrict edges to the given gene universe and drop low-degree source nodes
    (hidden units with too few linked genes to form a meaningful embedding
    coordinate).
    """
    gene_set = set(genes)
    e = edges[edges["source"].isin(gene_set) & edges["target"].isin(gene_set)].copy()
    degree = e.groupby("source").size()
    keep_sources = degree[degree >= min_edges_per_source].index
    e = e[e["source"].isin(keep_sources)].reset_index(drop=True)
    return e


def symmetrize_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """For priors whose underlying relation is undirected (e.g. co-accessibility:
    gene A and gene B share a chromatin neighborhood, no inherent direction),
    add the reverse of every edge so both endpoints are eligible to become a
    hidden unit (source), each reading its own linked neighborhood. Directed
    priors (e.g. TF->target) should not call this.
    """
    reversed_e = edges.rename(columns={"source": "target", "target": "source"})[edges.columns]
    out = pd.concat([edges, reversed_e], ignore_index=True)
    out = out.drop_duplicates(subset=["source", "target"]).reset_index(drop=True)
    return out


def edges_to_mask(edges: pd.DataFrame, genes: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a (n_genes, n_hidden_units) signed mask and sign matrix for the
    masked-linear encoder.

    Returns (mask [0/1], sign_weight [-1/0/1], hidden_unit_names) with rows
    ordered as `genes`. Unsigned priors should pass edges with weight == 1.0
    throughout, which yields sign == +1 everywhere a mask entry is nonzero.
    """
    gene_idx = {g: i for i, g in enumerate(genes)}
    hidden_units = sorted(edges["source"].unique())
    unit_idx = {u: i for i, u in enumerate(hidden_units)}

    # vectorized index mapping -- edges_to_mask is called ~20+ times across a full
    # run (real + every control seed, for both priors), and a per-row Python loop
    # here was measurably slow at the epigenomic prior's edge counts.
    e = edges[edges["target"].isin(gene_idx)]
    gi = e["target"].map(gene_idx).to_numpy()
    ui = e["source"].map(unit_idx).to_numpy()
    w = e["weight"].to_numpy(dtype=np.float64)

    mask = np.zeros((len(genes), len(hidden_units)), dtype=np.float32)
    sign = np.zeros((len(genes), len(hidden_units)), dtype=np.float32)
    mask[gi, ui] = 1.0
    sign_vals = np.sign(w)
    sign_vals[sign_vals == 0] = 1.0
    sign[gi, ui] = sign_vals
    return mask, sign, hidden_units


def degree_preserving_random(edges: pd.DataFrame, seed: int) -> pd.DataFrame:
    """C1: bipartite configuration-model shuffle. Preserves each hidden unit's
    out-degree and each gene's in-degree as closely as possible via repeated
    double-edge swaps.
    """
    rng = np.random.default_rng(seed)
    e = edges[["source", "target", "weight"]].copy()
    pairs = list(zip(e["source"], e["target"]))
    existing = set(pairs)
    n_swaps = len(pairs) * 20
    for _ in range(n_swaps):
        i, j = rng.integers(0, len(pairs), size=2)
        if i == j:
            continue
        s1, t1 = pairs[i]
        s2, t2 = pairs[j]
        if s1 == s2 or t1 == t2:
            continue
        new1, new2 = (s1, t2), (s2, t1)
        if new1 in existing or new2 in existing:
            continue
        existing.discard((s1, t1))
        existing.discard((s2, t2))
        existing.add(new1)
        existing.add(new2)
        pairs[i] = new1
        pairs[j] = new2

    out = pd.DataFrame(pairs, columns=["source", "target"])
    out["weight"] = e["weight"].sample(frac=1.0, random_state=seed).to_numpy()
    out["confidence"] = "control_degree_preserving"
    return out


def fully_randomized(edges: pd.DataFrame, genes: list[str], seed: int) -> pd.DataFrame:
    """C2: same edge count as the real graph, endpoints drawn uniformly at random
    from the hidden-unit set and the full gene universe (not degree-preserving).
    """
    rng = np.random.default_rng(seed)
    hidden_units = sorted(edges["source"].unique())
    n_edges = len(edges)
    sources = rng.choice(hidden_units, size=n_edges, replace=True)
    targets = rng.choice(genes, size=n_edges, replace=True)
    weights = edges["weight"].sample(frac=1.0, random_state=seed).to_numpy()
    out = pd.DataFrame({"source": sources, "target": targets, "weight": weights})
    out = out.drop_duplicates(subset=["source", "target"]).reset_index(drop=True)
    out["confidence"] = "control_fully_random"
    return out


def sign_flipped(edges: pd.DataFrame) -> pd.DataFrame:
    """C3: same topology as the real graph, all edge signs inverted. Only
    meaningful for signed priors -- callers must check `Prior.signed` before
    generating this control.
    """
    out = edges.copy()
    out["weight"] = -out["weight"]
    out["confidence"] = out["confidence"].astype(str) + "_sign_flipped"
    return out
