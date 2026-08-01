"""TF-target prior: DoRothEA TF->target regulatory graph via `decoupler`.

This is the harness's correctness check (see README/MEMO): the first prior
run through the shared harness is expected to reproduce TRACE's original
finding shape (real graph roughly matching or losing to a degree-preserving
random rewiring of itself). Divergence from that is treated as a harness bug
to find, not a new result.

This module's entire surface, per the harness's Prior interface
(src/priors/base.py): fetch DoRothEA and filter it to the shared gene
universe, and declare that it's signed (activator/repressor edges support a
sign-flip control). Everything else -- mask construction, structural
controls, encoder, CV, significance testing, permutation null -- is shared
code in src/graph.py, src/encoders.py, src/evaluate.py, src/significance.py.
"""

from __future__ import annotations

import pandas as pd

from src.graph import MIN_EDGES_PER_SOURCE, filter_edges
from src.priors.base import Prior


def load_dorothea(organism: str = "human") -> pd.DataFrame:
    import decoupler as dc

    return dc.op.dorothea(organism=organism, license="academic")


def build(genes: list[str], min_edges_per_source: int = MIN_EDGES_PER_SOURCE) -> Prior:
    dorothea = load_dorothea()
    edges = filter_edges(dorothea, genes, min_edges_per_source=min_edges_per_source)
    return Prior(name="tf_target", signed=True, edges=edges)
