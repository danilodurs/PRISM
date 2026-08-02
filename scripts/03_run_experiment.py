"""Harness evaluation, run identically for both priors: real graph vs
baseline_pca vs structural controls (vs sign-flip, TF-target only) on the
disease-status target, plus sex/leakage/cell-type-identity probes. Same
donor-level CV splits shared across every condition (computed once, reused
throughout) so results are directly comparable. Nothing here branches on
which prior is active except which graph files get loaded from disk -- the
encoder, CV protocol, and probes are identical.

`--dataset ra` (default) is this repo's target; `--dataset sle` remains
available for anyone who wants to reproduce the original SLE/single-prior
run, writing `sle_`-prefixed outputs. `--prior-mode combined` (default)
additionally runs the joint TF-target + epigenomic condition and its
structural controls (`src/priors/combined.py`), plus -- RA only, since
SLE's pseudobulk meta carries no DAS28 column -- a descriptive
(non-CV-protocol, non-corrected) DAS28 activity-score probe.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.datasets import DATASETS, DatasetConfig
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
from src.priors import combined as combined_mod

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_SPLITS = 6
N_SEEDS = 5
N_CONTROL_SEEDS = 5
PRIOR_NAMES = ["tf_target", "epigenomic"]
DAS28_COL = "das28_crp"


def load_pseudobulk(prefix: str = ""):
    expr = np.load(DATA_DIR / f"{prefix}pseudobulk_expr.npy")
    meta = pd.read_parquet(DATA_DIR / f"{prefix}pseudobulk_meta.parquet")
    genes = pd.read_csv(DATA_DIR / f"{prefix}pseudobulk_genes.csv")["gene_symbol"].tolist()
    return expr, meta, genes


def das28_probe_score(fold_embeddings, meta: pd.DataFrame, das28_col: str = DAS28_COL) -> dict:
    """Descriptive-only, non-CV-protocol, non-corrected probe (see
    MEMO.md / results/tables/step0_validation.md): DAS28 is
    RA-only and missing for 2/18 RA donors, too few donors for the harness's
    6-fold donor-level CV to give a meaningful per-fold estimate. Instead of
    a per-fold R^2 (unstable at n~2-3 valid test donors per fold), pools
    out-of-fold predictions across all 30 (seed, fold) splits into one set
    of held-out (prediction, true) pairs and reports a single Pearson r/R^2
    over that pool -- still strictly held-out (each prediction comes from a
    Ridge fit on that fold's training profiles only), just not broken out
    per split the way the primary AUROC target is.
    """
    all_pred: list[float] = []
    all_true: list[float] = []
    for fe in fold_embeddings:
        train_meta = meta.iloc[fe.train_idx]
        test_meta = meta.iloc[fe.test_idx]
        train_y = train_meta[das28_col].to_numpy(dtype=float)
        test_y = test_meta[das28_col].to_numpy(dtype=float)
        train_valid = ~np.isnan(train_y)
        test_valid = ~np.isnan(test_y)
        if train_valid.sum() < 3 or test_valid.sum() < 1:
            continue
        reg = Ridge(alpha=1.0)
        reg.fit(fe.Z_train[train_valid], train_y[train_valid])
        pred = reg.predict(fe.Z_test[test_valid])
        all_pred.extend(pred.tolist())
        all_true.extend(test_y[test_valid].tolist())

    if len(all_true) < 5:
        return {"r2": float("nan"), "pearson_r": float("nan"), "n_pooled_predictions": len(all_true)}
    pred_arr, true_arr = np.array(all_pred), np.array(all_true)
    return {
        "r2": float(r2_score(true_arr, pred_arr)),
        "pearson_r": float(np.corrcoef(true_arr, pred_arr)[0, 1]),
        "n_pooled_predictions": len(all_true),
    }


def run_condition(
    name: str,
    expr,
    meta,
    encoder_factory,
    splits,
    results: list[pd.DataFrame],
    disease_positive_label: str,
    das28_rows: list[dict] | None = None,
) -> None:
    t0 = time.time()
    fold_emb = fit_transform_cv(expr, meta, encoder_factory, splits)
    print(f"  [{name}] {len(splits)} folds fit in {time.time() - t0:.1f}s", flush=True)

    disease = score_classification_probe(fold_emb, meta, "disease", positive_label=disease_positive_label)
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

    if das28_rows is not None and DAS28_COL in meta.columns:
        das28_result = das28_probe_score(fold_emb, meta)
        das28_result["condition"] = name
        das28_rows.append(das28_result)


def run_prior(
    prior_name: str, expr, meta, genes, splits, results: list[pd.DataFrame], prefix: str, disease_positive_label: str,
    das28_rows: list[dict] | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    print(f"--- {prior_name} ---", flush=True)
    real_edges = pd.read_parquet(DATA_DIR / f"{prefix}graph_{prior_name}_real.parquet")
    mask, sign, units = edges_to_mask(real_edges, genes)
    n_components = len(units)
    print(f"[{prior_name}] embedding dimensionality (matched across conditions): {n_components}")

    run_condition(
        f"{prior_name}_baseline_pca", expr, meta, lambda seed: PCAEncoder(n_components, seed=seed), splits, results,
        disease_positive_label, das28_rows,
    )
    run_condition(
        f"{prior_name}_real", expr, meta, lambda seed: PriorEncoder(mask, sign, seed=seed), splits, results,
        disease_positive_label, das28_rows,
    )

    for cseed in range(N_CONTROL_SEEDS):
        c1_edges = pd.read_parquet(DATA_DIR / f"{prefix}graph_{prior_name}_c1_degree_preserving_seed{cseed}.parquet")
        c1_mask, c1_sign, _ = edges_to_mask(c1_edges, genes)
        run_condition(
            f"{prior_name}_c1_degree_preserving_seed{cseed}",
            expr, meta,
            lambda seed, m=c1_mask, s=c1_sign: PriorEncoder(m, s, seed=seed),
            splits, results, disease_positive_label, das28_rows,
        )

        c2_edges = pd.read_parquet(DATA_DIR / f"{prefix}graph_{prior_name}_c2_fully_random_seed{cseed}.parquet")
        c2_mask, c2_sign, _ = edges_to_mask(c2_edges, genes)
        run_condition(
            f"{prior_name}_c2_fully_random_seed{cseed}",
            expr, meta,
            lambda seed, m=c2_mask, s=c2_sign: PriorEncoder(m, s, seed=seed),
            splits, results, disease_positive_label, das28_rows,
        )

    c3_path = DATA_DIR / f"{prefix}graph_{prior_name}_c3_sign_flipped.parquet"
    if c3_path.exists():
        c3_edges = pd.read_parquet(c3_path)
        c3_mask, c3_sign, _ = edges_to_mask(c3_edges, genes)
        run_condition(
            f"{prior_name}_c3_sign_flipped",
            expr, meta,
            lambda seed: PriorEncoder(c3_mask, c3_sign, seed=seed),
            splits, results, disease_positive_label, das28_rows,
        )

    return mask, sign, n_components


def run_combined(
    expr, meta, genes, splits, results: list[pd.DataFrame], prefix: str, disease_positive_label: str,
    das28_rows: list[dict] | None,
) -> None:
    """Combined TF-target + epigenomic condition and its joint structural
    controls (see src/priors/combined.py). Edge lists are loaded from disk
    (written by scripts/02_build_graph.py's --prior-mode combined) and the
    dense joint mask is built here at experiment time via
    `combined_mod.joint_mask_from_edges`, exactly mirroring how `run_prior`
    builds single-prior masks via `edges_to_mask`.
    """
    print("--- combined ---", flush=True)

    def load_joint(condition: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        tf_edges = pd.read_parquet(DATA_DIR / f"{prefix}graph_combined_{condition}_tf.parquet")
        epi_edges = pd.read_parquet(DATA_DIR / f"{prefix}graph_combined_{condition}_epi.parquet")
        return combined_mod.joint_mask_from_edges(tf_edges, epi_edges, genes)

    mask, sign, units = load_joint("real")
    n_components = len(units)
    print(f"[combined] embedding dimensionality (matched across conditions): {n_components}")

    run_condition(
        "combined_baseline_pca", expr, meta, lambda seed: PCAEncoder(n_components, seed=seed), splits, results,
        disease_positive_label, das28_rows,
    )
    run_condition(
        "combined_real", expr, meta, lambda seed: PriorEncoder(mask, sign, seed=seed), splits, results,
        disease_positive_label, das28_rows,
    )

    for cseed in range(N_CONTROL_SEEDS):
        c1_mask, c1_sign, _ = load_joint(f"c1_degree_preserving_seed{cseed}")
        run_condition(
            f"combined_c1_degree_preserving_seed{cseed}", expr, meta,
            lambda seed, m=c1_mask, s=c1_sign: PriorEncoder(m, s, seed=seed),
            splits, results, disease_positive_label, das28_rows,
        )

        c2_mask, c2_sign, _ = load_joint(f"c2_fully_random_seed{cseed}")
        run_condition(
            f"combined_c2_fully_random_seed{cseed}", expr, meta,
            lambda seed, m=c2_mask, s=c2_sign: PriorEncoder(m, s, seed=seed),
            splits, results, disease_positive_label, das28_rows,
        )

        # exploratory ablation, reported outside the Bonferroni-corrected family (04_significance_test.py)
        only_tf_mask, only_tf_sign, _ = load_joint(f"only_tf_real_seed{cseed}")
        run_condition(
            f"combined_only_tf_real_seed{cseed}", expr, meta,
            lambda seed, m=only_tf_mask, s=only_tf_sign: PriorEncoder(m, s, seed=seed),
            splits, results, disease_positive_label, das28_rows,
        )

        only_epi_mask, only_epi_sign, _ = load_joint(f"only_epi_real_seed{cseed}")
        run_condition(
            f"combined_only_epi_real_seed{cseed}", expr, meta,
            lambda seed, m=only_epi_mask, s=only_epi_sign: PriorEncoder(m, s, seed=seed),
            splits, results, disease_positive_label, das28_rows,
        )

    c3_mask, c3_sign, _ = load_joint("c3_sign_flipped")
    run_condition(
        "combined_c3_sign_flipped", expr, meta, lambda seed: PriorEncoder(c3_mask, c3_sign, seed=seed),
        splits, results, disease_positive_label, das28_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ra")
    parser.add_argument("--prior-mode", choices=["independent", "combined"], default="combined")
    args = parser.parse_args()
    config: DatasetConfig = DATASETS[args.dataset]
    prefix = config.file_prefix

    run_t0 = time.time()
    print(f"Parallel fold fits: n_jobs={DEFAULT_N_JOBS}", flush=True)

    expr, meta, genes = load_pseudobulk(prefix=prefix)
    splits = make_splits(meta, n_splits=N_SPLITS, n_seeds=N_SEEDS, stratify_col="disease")
    print(f"{len(splits)} donor-level CV splits ({N_SEEDS} seeds x {N_SPLITS} folds), shared across all conditions")

    results: list[pd.DataFrame] = []
    das28_rows: list[dict] | None = [] if DAS28_COL in meta.columns else None

    for prior_name in PRIOR_NAMES:
        run_prior(prior_name, expr, meta, genes, splits, results, prefix, config.disease_positive_label, das28_rows)

    if args.prior_mode == "combined":
        run_combined(expr, meta, genes, splits, results, prefix, config.disease_positive_label, das28_rows)

    main_results = pd.concat(results, ignore_index=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    main_results.to_csv(RESULTS_DIR / "tables" / f"{prefix}main_results.csv", index=False)
    print(main_results.groupby(["condition", "target", "metric"])["score"].agg(["mean", "std"]))

    if das28_rows:
        das28_df = pd.DataFrame(das28_rows)
        das28_df.to_csv(RESULTS_DIR / "tables" / f"{prefix}das28_probe.csv", index=False)
        print("\nDAS28 descriptive probe (non-CV-protocol, non-corrected, pooled out-of-fold predictions):")
        print(das28_df.to_string(index=False))

    print(f"Done in {time.time() - run_t0:.1f}s total. Results written to results/tables/{prefix}main_results.csv")


if __name__ == "__main__":
    main()
