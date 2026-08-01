"""Epigenomic prior: gene-gene links derived from the ABC (Activity-by-
Contact) model's genome-wide enhancer-gene predictions (Nasser et al. 2021,
Nature Genome-wide enhancer maps link risk variants to disease genes),
restricted to PBMC/immune-relevant biosamples. Two genes are linked if the
ABC model predicts they share a regulatory element (same class + genomic
coordinates, the `name` column) in at least one selected biosample -- an
unsigned, undirected relation (co-accessibility resources like this one
carry no activator/repressor directionality), so no sign-flip control
applies here (see `Prior.signed=False`).

Chosen specifically for PBMC/immune tissue specificity over a generic
pan-tissue resource: the ABC atlas spans 131 biosamples across many tissues,
and this module restricts to 5 immune biosamples matching this harness's
five major pseudobulk cell types as closely as the public atlas allows
(CD14+ monocyte, CD19+ B cell, CD4+ helper T cell, CD8+ T cell, CD56+ NK
cell) -- a deliberately chosen subset, not an exhaustive sweep of the atlas.

This module's entire surface, per the harness's Prior interface: stream and
filter the ABC predictions to PBMC biosamples, collapse to a gene-gene edge
list, filter to the shared gene universe, and declare `signed=False`.
Everything else is shared harness code (see src/priors/tf_target.py's
docstring for the same point made about the TF-target prior).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from src.graph import filter_edges, symmetrize_edges
from src.priors.base import Prior

ABC_RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "abc_predictions_raw.txt.gz"

# Overrides the shared harness default (5): ABC's biosample-pooled edges are
# dense enough that the shared default leaves ~13.8k genes with >=5 linked
# partners -- nearly as many hidden units as the gene universe itself. That's
# both a capacity regime wildly mismatched with TF-target's ~300 hidden units
# (confounding "prior helps" with "prior gets far more embedding capacity")
# and, since this architecture stores its mask as a dense tensor, a jump from
# a ~16.5k x 300 mask to ~16.5k x 13.8k -- computationally impractical (the
# decoder alone would be a ~230M-parameter dense matrix, refit from scratch
# on every one of ~180 fold fits). A degree floor of 75 lands the hidden-unit
# count at a similar order of magnitude to TF-target's, keeping the two
# priors' encoder capacity comparable.
MIN_EDGES_PER_SOURCE = 75

PBMC_BIOSAMPLES = {
    "classical monocyte": "CD14-positive_monocyte-ENCODE",
    "B cell": "CD19-positive_B_cell-Roadmap",
    "CD4-positive, alpha-beta T cell": "CD4-positive_helper_T_cell-ENCODE",
    "CD8-positive, alpha-beta T cell": "CD8-positive_alpha-beta_T_cell-ENCODE",
    "natural killer cell": "CD56-positive_natural_killer_cells-Roadmap",
}

MIN_ABC_SCORE = 0.015  # the file's own inclusion threshold; not tightened further


def load_abc_edges(
    abc_path: Path = ABC_RAW_PATH,
    biosamples: list[str] | None = None,
    min_abc_score: float = MIN_ABC_SCORE,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Stream the genome-wide ABC predictions file in chunks, keeping only
    rows from the selected PBMC biosamples (the full file spans all 131
    biosamples and is far larger than the PBMC-relevant subset needed here),
    then collapse to gene-gene edges: any two genes linked to the same
    regulatory element in the same biosample become an edge.
    """
    biosample_set = set(biosamples if biosamples is not None else PBMC_BIOSAMPLES.values())

    kept_chunks = []
    reader = pd.read_csv(
        abc_path,
        sep="\t",
        chunksize=chunksize,
        compression="gzip",
        usecols=["name", "TargetGene", "CellType", "ABC.Score"],
    )
    for chunk in reader:
        keep = chunk["CellType"].isin(biosample_set) & (chunk["ABC.Score"] >= min_abc_score)
        if keep.any():
            kept_chunks.append(chunk.loc[keep])

    abc = pd.concat(kept_chunks, ignore_index=True)

    edges = []
    for _, grp in abc.groupby(["CellType", "name"], observed=True):
        genes_here = sorted(grp["TargetGene"].unique())
        if len(genes_here) < 2:
            continue
        edges.extend(combinations(genes_here, 2))

    edge_df = pd.DataFrame(edges, columns=["source", "target"]).drop_duplicates().reset_index(drop=True)
    edge_df["weight"] = 1.0
    edge_df["confidence"] = "abc_coaccessibility"
    return symmetrize_edges(edge_df)


def build(genes: list[str], min_edges_per_source: int = MIN_EDGES_PER_SOURCE) -> Prior:
    raw_edges = load_abc_edges()
    edges = filter_edges(raw_edges, genes, min_edges_per_source=min_edges_per_source)
    return Prior(name="epigenomic", signed=False, edges=edges)
