"""Combined TF-target + epigenomic prior: joint mask over the shared gene
panel, feeding one `PriorEncoder`, without merging edges into a shared
per-gene hidden-unit space.

Design choice -- separate channels concatenated pre-encoder, not a
gene-keyed union with provenance tags (see MEMO.md for the full
writeup):

- TF-target hidden units (TFs) and epigenomic hidden units (co-accessibility
  anchor genes) are different kinds of nodes. In the RA gene panel, 331
  genes are both a DoRothEA TF and an ABC anchor gene -- merging them into
  one shared per-gene hidden unit would require inventing a sign-conflict
  rule (does a directed repressor edge and an undirected co-accessibility
  edge for the same gene pair average? does one win?) with no principled
  answer.
- Concatenating each prior's mask as its own block lets every joint
  structural control reuse `src/graph.py`'s existing per-prior functions
  (`degree_preserving_random`, `fully_randomized`, `sign_flipped`)
  completely unmodified: call each once per source-prior block, concatenate
  the results. This is what makes "preserve each gene's degree *within each
  source prior*" fall out for free instead of requiring new graph-theory
  code.
- Sign handling: each block keeps exactly the sign behavior it has
  standalone -- TF-target's block carries its real +/- signs, epigenomic's
  block is uniformly sign=+1 (as it already is standalone). There is no
  cross-block sign interaction, because a hidden unit only ever belongs to
  one block.
- Overlapping edges (the same gene pair present in both priors' edge lists,
  e.g. a DoRothEA edge `STAT1->IL6` and an ABC edge `STAT1-IL6`) are kept in
  BOTH blocks, not merged: gene `IL6`'s mask row gets a nonzero entry from
  both the `tf_target::STAT1` hidden unit and the `epigenomic::STAT1`
  hidden unit. Overlap is preserved via duplication across channels.

Every function here works at the EDGE-LIST level (like `src/graph.py`'s
functions), not the dense-mask level -- consistent with the existing
pipeline's split of concerns: `scripts/02_build_graph.py` persists edge
lists (small parquet files), `scripts/03_run_experiment.py` builds the
actual dense mask at experiment time via `joint_mask_from_edges`. This keeps
storage cheap (sparse edge lists, not dense n_genes x n_units arrays) and
matches exactly how the standalone `tf_target`/`epigenomic` conditions are
already built and run.

Neither `src/graph.py` nor either standalone prior module (`base.py`,
`tf_target.py`, `epigenomic.py`) is modified -- this module is purely
additive composition on top of them, so both priors keep working standalone
for `main`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.graph import degree_preserving_random, edges_to_mask, fully_randomized, sign_flipped
from src.priors import epigenomic, tf_target
from src.priors.base import Prior

TF_NAME = "tf_target"
EPI_NAME = "epigenomic"


@dataclass
class CombinedPrior:
    name: str
    tf_prior: Prior
    epi_prior: Prior


def build(genes: list[str]) -> CombinedPrior:
    """Builds each standalone prior exactly as the independent-mode pipeline
    does (`src.priors.tf_target.build`, `src.priors.epigenomic.build` --
    unmodified), then wraps them for joint edge-list / mask construction.
    """
    return CombinedPrior(name="combined", tf_prior=tf_target.build(genes), epi_prior=epigenomic.build(genes))


def real_edges(combined: CombinedPrior) -> tuple[pd.DataFrame, pd.DataFrame]:
    return combined.tf_prior.edges, combined.epi_prior.edges


def c1_edges(combined: CombinedPrior, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joint C1: degree-preserving random rewire applied independently to
    each source prior's real edges (same seed for both blocks) -- preserves
    each gene's degree *within each source prior* separately, never pooling
    degree across priors.
    """
    return (
        degree_preserving_random(combined.tf_prior.edges, seed=seed),
        degree_preserving_random(combined.epi_prior.edges, seed=seed),
    )


def c2_edges(combined: CombinedPrior, seed: int, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joint C2: fully-randomized edges applied independently to each source
    prior (same seed for both blocks, edge count matched per prior).
    """
    return (
        fully_randomized(combined.tf_prior.edges, genes, seed=seed),
        fully_randomized(combined.epi_prior.edges, genes, seed=seed),
    )


def c3_edges(combined: CombinedPrior) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joint C3: only the TF-target block's signs are flipped -- epigenomic
    carries no sign to flip, the same reason it has no standalone C3.
    """
    return sign_flipped(combined.tf_prior.edges), combined.epi_prior.edges


def only_tf_real_edges(combined: CombinedPrior, seed: int, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exploratory ablation (reported outside the Bonferroni-corrected
    family, see scripts/04_significance_test.py / MEMO.md): real
    TF-target block + a C2 fully-randomized epigenomic block, isolating
    whether TF-target alone drives any combined-condition effect.
    """
    return combined.tf_prior.edges, fully_randomized(combined.epi_prior.edges, genes, seed=seed)


def only_epi_real_edges(combined: CombinedPrior, seed: int, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exploratory ablation, mirror of `only_tf_real_edges`: real epigenomic
    block + a C2 fully-randomized TF-target block.
    """
    return fully_randomized(combined.tf_prior.edges, genes, seed=seed), combined.epi_prior.edges


def joint_mask_from_edges(
    tf_edges: pd.DataFrame, epi_edges: pd.DataFrame, genes: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Builds the dense joint mask/sign/unit-names at experiment time (called
    from scripts/03_run_experiment.py, not scripts/02_build_graph.py) --
    each block via the unmodified `edges_to_mask`, concatenated column-wise,
    hidden-unit names prefixed by source prior for lossless provenance.
    """
    tf_mask, tf_sign, tf_units = edges_to_mask(tf_edges, genes)
    epi_mask, epi_sign, epi_units = edges_to_mask(epi_edges, genes)
    mask = np.hstack([tf_mask, epi_mask])
    sign = np.hstack([tf_sign, epi_sign])
    unit_names = [f"{TF_NAME}::{u}" for u in tf_units] + [f"{EPI_NAME}::{u}" for u in epi_units]
    return mask, sign, unit_names
