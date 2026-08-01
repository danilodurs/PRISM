"""Harness evaluation, run identically for both priors: real graph vs
baseline_pca vs structural controls (vs sign-flip, TF-target only) on the
SLE-vs-normal disease-status target, plus sex/leakage/cell-type-identity
probes. Same donor-level CV splits shared across both priors (computed once,
reused for every condition of every prior) so results are directly
comparable. Nothing here branches on which prior is active except which
graph files get loaded from disk -- the encoder, CV protocol, and probes are
identical.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.encoders import PCAEncoder, PriorEncoder
from src.evaluate import (
    DEFAULT_N_JOBS,
    fit_transform_cv,
    make_splits,
    score_classification_probe,
    score_continuous_probe,
    score_multiclass_probe,
)
from src.graph import edges_to_mask

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_SPLITS = 6
N_SEEDS = 5
N_CONTROL_SEEDS = 5
PRIOR_NAMES = ["tf_target", "epigenomic"]
DISEASE_POSITIVE_LABEL = "systemic lupus erythematosus"


def load_pseudobulk():
    expr = np.load(DATA_DIR / "pseudobulk_expr.npy")
    meta = pd.read_parquet(DATA_DIR / "pseudobulk_meta.parquet")
    genes = pd.read_csv(DATA_DIR / "pseudobulk_genes.csv")["gene_symbol"].tolist()
    return expr, meta, genes


def run_condition(name: str, expr, meta, encoder_factory, splits, results: list[pd.DataFrame]) -> None:
    t0 = time.time()
    fold_emb = fit_transform_cv(expr, meta, encoder_factory, splits)
    print(f"  [{name}] {len(splits)} folds fit in {time.time() - t0:.1f}s", flush=True)

    disease = score_classification_probe(fold_emb, meta, "disease", positive_label=DISEASE_POSITIVE_LABEL)
    disease["condition"] = name
    disease["target"] = "disease_status"
    results.append(disease)

    sex = score_classification_probe(fold_emb, meta, "sex", positive_label="female")
    sex["condition"] = name
    sex["target"] = "sex_sanity_check"
    results.append(sex)

    leakage = score_continuous_probe(fold_emb, meta, "lib_size")
    leakage["condition"] = name
    leakage["target"] = "lib_size_leakage"
    results.append(leakage)

    celltype = score_multiclass_probe(fold_emb, meta, "cell_type")
    celltype["condition"] = name
    celltype["target"] = "celltype_sanity_check"
    results.append(celltype)


def run_prior(prior_name: str, expr, meta, genes, splits, results: list[pd.DataFrame]) -> None:
    print(f"--- {prior_name} ---", flush=True)
    real_edges = pd.read_parquet(DATA_DIR / f"graph_{prior_name}_real.parquet")
    mask, sign, units = edges_to_mask(real_edges, genes)
    n_components = len(units)
    print(f"[{prior_name}] embedding dimensionality (matched across conditions): {n_components}")

    run_condition(f"{prior_name}_baseline_pca", expr, meta, lambda seed: PCAEncoder(n_components, seed=seed), splits, results)
    run_condition(f"{prior_name}_real", expr, meta, lambda seed: PriorEncoder(mask, sign, seed=seed), splits, results)

    for cseed in range(N_CONTROL_SEEDS):
        c1_edges = pd.read_parquet(DATA_DIR / f"graph_{prior_name}_c1_degree_preserving_seed{cseed}.parquet")
        c1_mask, c1_sign, _ = edges_to_mask(c1_edges, genes)
        run_condition(
            f"{prior_name}_c1_degree_preserving_seed{cseed}",
            expr, meta,
            lambda seed, m=c1_mask, s=c1_sign: PriorEncoder(m, s, seed=seed),
            splits, results,
        )

        c2_edges = pd.read_parquet(DATA_DIR / f"graph_{prior_name}_c2_fully_random_seed{cseed}.parquet")
        c2_mask, c2_sign, _ = edges_to_mask(c2_edges, genes)
        run_condition(
            f"{prior_name}_c2_fully_random_seed{cseed}",
            expr, meta,
            lambda seed, m=c2_mask, s=c2_sign: PriorEncoder(m, s, seed=seed),
            splits, results,
        )

    c3_path = DATA_DIR / f"graph_{prior_name}_c3_sign_flipped.parquet"
    if c3_path.exists():
        c3_edges = pd.read_parquet(c3_path)
        c3_mask, c3_sign, _ = edges_to_mask(c3_edges, genes)
        run_condition(
            f"{prior_name}_c3_sign_flipped",
            expr, meta,
            lambda seed: PriorEncoder(c3_mask, c3_sign, seed=seed),
            splits, results,
        )


def main() -> None:
    run_t0 = time.time()
    print(f"Parallel fold fits: n_jobs={DEFAULT_N_JOBS}", flush=True)

    expr, meta, genes = load_pseudobulk()
    splits = make_splits(meta, n_splits=N_SPLITS, n_seeds=N_SEEDS, stratify_col="disease")
    print(f"{len(splits)} donor-level CV splits ({N_SEEDS} seeds x {N_SPLITS} folds), shared across both priors")

    results: list[pd.DataFrame] = []
    for prior_name in PRIOR_NAMES:
        run_prior(prior_name, expr, meta, genes, splits, results)

    main_results = pd.concat(results, ignore_index=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    main_results.to_csv(RESULTS_DIR / "tables" / "main_results.csv", index=False)
    print(main_results.groupby(["condition", "target", "metric"])["score"].agg(["mean", "std"]))
    print(f"Done in {time.time() - run_t0:.1f}s total. Results written to results/tables/main_results.csv")


if __name__ == "__main__":
    main()
