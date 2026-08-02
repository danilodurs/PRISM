"""Build both priors' real graphs (filtered to the shared pseudobulk gene
universe) and their structural controls (degree-preserving rewire, fully
randomized, and -- for the signed TF-target prior only -- sign-flipped).
Prior-specific logic is limited to `tf_target.build`/`epigenomic.build`
(see src/priors/*.py); everything below is shared, prior-agnostic code.

`--prior-mode combined` (default) additionally builds the joint TF-target +
epigenomic mask and its own structural controls (`src/priors/combined.py`)
-- see that module's docstring for the design (concatenated channels, not a
merged per-gene edge union). `--dataset ra` (default) is this branch's
target; `--dataset sle` remains available for anyone who wants to reproduce
`main`'s original SLE run, writing `sle_`-prefixed outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.datasets import DATASETS, DatasetConfig
from src.graph import degree_preserving_random, fully_randomized, sign_flipped
from src.priors import combined as combined_mod
from src.priors import epigenomic, tf_target
from src.priors.base import Prior

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
N_CONTROL_SEEDS = 5


def _save_edge_pair(prefix: str, condition: str, tf_edges: pd.DataFrame, epi_edges: pd.DataFrame) -> None:
    """Persist a joint condition as two small edge-list parquet files (one
    per source-prior block) rather than a precomputed dense mask -- mirrors
    how the standalone tf_target/epigenomic conditions are stored, and
    defers the actual (mask, sign) construction to experiment time via
    `combined_mod.joint_mask_from_edges` (scripts/03_run_experiment.py),
    exactly like `edges_to_mask` is called there for the single priors.
    """
    tf_edges.to_parquet(DATA_DIR / f"{prefix}graph_combined_{condition}_tf.parquet")
    epi_edges.to_parquet(DATA_DIR / f"{prefix}graph_combined_{condition}_epi.parquet")


def build_and_save_combined(combined: combined_mod.CombinedPrior, genes: list[str], prefix: str, summary_lines: list[str]) -> None:
    """Joint real edges + joint C1/C2/C3 + the exploratory only-one-real
    ablation (reported outside the Bonferroni-corrected family, see
    scripts/04_significance_test.py). Reuses graph.py's per-prior control
    functions unmodified via src/priors/combined.py -- see that module's
    docstring for why concatenation, not a merged edge union.
    """
    tf_edges, epi_edges = combined_mod.real_edges(combined)
    n_tf_units = tf_edges["source"].nunique()
    n_epi_units = epi_edges["source"].nunique()
    summary_lines.append("\n## combined (tf_target::* + epigenomic::*)")
    summary_lines.append(f"Real joint mask: {n_tf_units + n_epi_units} hidden units ({n_tf_units} tf_target + {n_epi_units} epigenomic)")
    print(f"[combined] real: {n_tf_units + n_epi_units} hidden units ({n_tf_units} tf_target + {n_epi_units} epigenomic)")
    _save_edge_pair(prefix, "real", tf_edges, epi_edges)

    for seed in range(N_CONTROL_SEEDS):
        c1_tf, c1_epi = combined_mod.c1_edges(combined, seed=seed)
        _save_edge_pair(prefix, f"c1_degree_preserving_seed{seed}", c1_tf, c1_epi)

        c2_tf, c2_epi = combined_mod.c2_edges(combined, seed=seed, genes=genes)
        _save_edge_pair(prefix, f"c2_fully_random_seed{seed}", c2_tf, c2_epi)

        only_tf_tf, only_tf_epi = combined_mod.only_tf_real_edges(combined, seed=seed, genes=genes)
        _save_edge_pair(prefix, f"only_tf_real_seed{seed}", only_tf_tf, only_tf_epi)

        only_epi_tf, only_epi_epi = combined_mod.only_epi_real_edges(combined, seed=seed, genes=genes)
        _save_edge_pair(prefix, f"only_epi_real_seed{seed}", only_epi_tf, only_epi_epi)

        if seed == 0:
            print(f"  seed {seed}: joint C1 {len(c1_tf)}+{len(c1_epi)} edges, joint C2 {len(c2_tf)}+{len(c2_epi)} edges")

    c3_tf, c3_epi = combined_mod.c3_edges(combined)
    _save_edge_pair(prefix, "c3_sign_flipped", c3_tf, c3_epi)

    summary_lines.append(
        f"Controls generated per seed (n_seeds={N_CONTROL_SEEDS}): joint C1 degree-preserving (per source "
        "prior), joint C2 fully random (per source prior), plus exploratory only_tf_real/only_epi_real "
        "ablations (uncorrected, see MEMO.md)"
    )
    summary_lines.append("Joint C3 sign-flipped (TF-target block only, single deterministic version)")


def build_and_save(prior: Prior, genes: list[str], summary_lines: list[str], prefix: str = "") -> None:
    n_units = prior.edges["source"].nunique()
    summary_lines.append(f"\n## {prior.name} (signed={prior.signed})")
    summary_lines.append(f"Real graph: {len(prior.edges)} edges, {n_units} hidden units")
    print(f"[{prior.name}] real: {len(prior.edges)} edges, {n_units} hidden units, signed={prior.signed}")
    prior.edges.to_parquet(DATA_DIR / f"{prefix}graph_{prior.name}_real.parquet")

    for seed in range(N_CONTROL_SEEDS):
        c1 = degree_preserving_random(prior.edges, seed=seed)
        c1.to_parquet(DATA_DIR / f"{prefix}graph_{prior.name}_c1_degree_preserving_seed{seed}.parquet")

        c2 = fully_randomized(prior.edges, genes, seed=seed)
        c2.to_parquet(DATA_DIR / f"{prefix}graph_{prior.name}_c2_fully_random_seed{seed}.parquet")

        if seed == 0:
            print(f"  seed {seed}: C1 {len(c1)} edges, C2 {len(c2)} edges")

    summary_lines.append(f"Controls generated per seed (n_seeds={N_CONTROL_SEEDS}): C1 degree-preserving, C2 fully random")

    if prior.signed:
        c3 = sign_flipped(prior.edges)
        c3.to_parquet(DATA_DIR / f"{prefix}graph_{prior.name}_c3_sign_flipped.parquet")
        summary_lines.append("C3 sign-flipped (single version, deterministic given real graph)")
    else:
        summary_lines.append("C3 sign-flipped: not applicable (unsigned prior)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ra")
    parser.add_argument("--prior-mode", choices=["independent", "combined"], default="combined")
    args = parser.parse_args()
    config: DatasetConfig = DATASETS[args.dataset]
    prefix = config.file_prefix

    genes = pd.read_csv(DATA_DIR / f"{prefix}pseudobulk_genes.csv")["gene_symbol"].tolist()
    header = f"# Graph construction summary ({config.key}, prior-mode={args.prior_mode})"
    summary_lines = [f"{header}\n", f"Shared gene universe: {len(genes)} genes"]

    tf_prior = tf_target.build(genes)
    build_and_save(tf_prior, genes, summary_lines, prefix=prefix)

    epi_prior = epigenomic.build(genes)
    build_and_save(epi_prior, genes, summary_lines, prefix=prefix)

    if args.prior_mode == "combined":
        combined = combined_mod.CombinedPrior(name="combined", tf_prior=tf_prior, epi_prior=epi_prior)
        build_and_save_combined(combined, genes, prefix, summary_lines)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{prefix}graph_summary.md").write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
