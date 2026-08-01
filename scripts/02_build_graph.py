"""Build both priors' real graphs (filtered to the shared pseudobulk gene
universe) and their structural controls (degree-preserving rewire, fully
randomized, and -- for the signed TF-target prior only -- sign-flipped).
Prior-specific logic is limited to `tf_target.build`/`epigenomic.build`
(see src/priors/*.py); everything below is shared, prior-agnostic code.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.graph import degree_preserving_random, fully_randomized, sign_flipped
from src.priors import epigenomic, tf_target
from src.priors.base import Prior

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
N_CONTROL_SEEDS = 5


def build_and_save(prior: Prior, genes: list[str], summary_lines: list[str]) -> None:
    n_units = prior.edges["source"].nunique()
    summary_lines.append(f"\n## {prior.name} (signed={prior.signed})")
    summary_lines.append(f"Real graph: {len(prior.edges)} edges, {n_units} hidden units")
    print(f"[{prior.name}] real: {len(prior.edges)} edges, {n_units} hidden units, signed={prior.signed}")
    prior.edges.to_parquet(DATA_DIR / f"graph_{prior.name}_real.parquet")

    for seed in range(N_CONTROL_SEEDS):
        c1 = degree_preserving_random(prior.edges, seed=seed)
        c1.to_parquet(DATA_DIR / f"graph_{prior.name}_c1_degree_preserving_seed{seed}.parquet")

        c2 = fully_randomized(prior.edges, genes, seed=seed)
        c2.to_parquet(DATA_DIR / f"graph_{prior.name}_c2_fully_random_seed{seed}.parquet")

        if seed == 0:
            print(f"  seed {seed}: C1 {len(c1)} edges, C2 {len(c2)} edges")

    summary_lines.append(f"Controls generated per seed (n_seeds={N_CONTROL_SEEDS}): C1 degree-preserving, C2 fully random")

    if prior.signed:
        c3 = sign_flipped(prior.edges)
        c3.to_parquet(DATA_DIR / f"graph_{prior.name}_c3_sign_flipped.parquet")
        summary_lines.append("C3 sign-flipped (single version, deterministic given real graph)")
    else:
        summary_lines.append("C3 sign-flipped: not applicable (unsigned prior)")


def main() -> None:
    genes = pd.read_csv(DATA_DIR / "pseudobulk_genes.csv")["gene_symbol"].tolist()
    summary_lines = ["# Graph construction summary\n", f"Shared gene universe: {len(genes)} genes"]

    tf_prior = tf_target.build(genes)
    build_and_save(tf_prior, genes, summary_lines)

    epi_prior = epigenomic.build(genes)
    build_and_save(epi_prior, genes, summary_lines)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "graph_summary.md").write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
