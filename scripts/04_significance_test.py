"""Paired significance tests, label-permutation null, and the prior x
condition scorecard -- run identically for both priors via the shared
src/significance.py functions. AUROC (donor-level disease-status prediction)
is the primary tested metric, matching TRACE's precedent; accuracy is
reported in the scorecard for descriptive context but not separately
Bonferroni-corrected.

Comparisons per single prior (the "comparison family" Bonferroni-corrected
together): real vs baseline_pca, real vs pooled C1 (degree-preserving,
5 seeds), real vs pooled C2 (fully random, 5 seeds), and -- TF-target only,
since it's the only signed prior -- real vs C3 (sign-flipped). This is why
the correction denominator differs between priors (4 vs 3): it isn't
arbitrary, it's exactly the set of controls that prior has.

`--prior-mode combined` additionally tests the combined condition against a
6-comparison family (Bonferroni alpha/6): baseline_pca, joint C1, joint C2,
joint C3 (TF-block sign-flip), and the two single-prior RA results (does
combining beat the better of the two priors alone?). The exploratory
only-one-real ablations (isolating which prior drives any combined effect)
are reported separately, uncorrected -- they were explicitly scoped as
supplementary, not part of the primary inferential family (see MEMO.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.datasets import DATASETS, DatasetConfig
from src.encoders import PCAEncoder, PriorEncoder
from src.evaluate import fit_transform_cv, make_splits
from src.graph import edges_to_mask
from src.priors import combined as combined_mod
from src.significance import label_permutation_null, permutation_percentile, run_paired_tests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_SPLITS = 6
N_SEEDS = 5
N_CONTROL_SEEDS = 5
N_PERMUTATIONS = 100
PRIORS = [("tf_target", True), ("epigenomic", False)]


def comparison_family(prior_name: str, signed: bool) -> list[tuple[str, str, str | list[str]]]:
    family = [
        (f"{prior_name}: real vs baseline_pca", f"{prior_name}_real", f"{prior_name}_baseline_pca"),
        (
            f"{prior_name}: real vs C1 degree-preserving (pooled {N_CONTROL_SEEDS} seeds)",
            f"{prior_name}_real",
            [f"{prior_name}_c1_degree_preserving_seed{s}" for s in range(N_CONTROL_SEEDS)],
        ),
        (
            f"{prior_name}: real vs C2 fully random (pooled {N_CONTROL_SEEDS} seeds)",
            f"{prior_name}_real",
            [f"{prior_name}_c2_fully_random_seed{s}" for s in range(N_CONTROL_SEEDS)],
        ),
    ]
    if signed:
        family.append((f"{prior_name}: real vs C3 sign-flipped", f"{prior_name}_real", f"{prior_name}_c3_sign_flipped"))
    return family


def combined_comparison_family() -> list[tuple[str, str, str | list[str]]]:
    """6-comparison Bonferroni-corrected family: baseline, joint C1/C2/C3,
    and both RA single-prior results (does combining beat the better of the
    two priors alone on RA?).
    """
    return [
        ("combined: real vs baseline_pca", "combined_real", "combined_baseline_pca"),
        (
            f"combined: real vs joint C1 degree-preserving (pooled {N_CONTROL_SEEDS} seeds)",
            "combined_real",
            [f"combined_c1_degree_preserving_seed{s}" for s in range(N_CONTROL_SEEDS)],
        ),
        (
            f"combined: real vs joint C2 fully random (pooled {N_CONTROL_SEEDS} seeds)",
            "combined_real",
            [f"combined_c2_fully_random_seed{s}" for s in range(N_CONTROL_SEEDS)],
        ),
        ("combined: real vs joint C3 sign-flipped (TF block)", "combined_real", "combined_c3_sign_flipped"),
        ("combined: real vs tf_target_real (single prior)", "combined_real", "tf_target_real"),
        ("combined: real vs epigenomic_real (single prior)", "combined_real", "epigenomic_real"),
    ]


def combined_exploratory_family() -> list[tuple[str, str, str | list[str]]]:
    """Exploratory, uncorrected ablation comparisons -- isolates which
    source prior drives any combined-condition effect. Reported outside the
    Bonferroni-corrected family (see module docstring / MEMO.md).
    """
    return [
        (
            f"combined: real vs only_tf_real (pooled {N_CONTROL_SEEDS} seeds)",
            "combined_real",
            [f"combined_only_tf_real_seed{s}" for s in range(N_CONTROL_SEEDS)],
        ),
        (
            f"combined: real vs only_epi_real (pooled {N_CONTROL_SEEDS} seeds)",
            "combined_real",
            [f"combined_only_epi_real_seed{s}" for s in range(N_CONTROL_SEEDS)],
        ),
    ]


def _accuracy_means(d: pd.DataFrame, family: list[tuple[str, str, str | list[str]]]) -> pd.DataFrame:
    rows = []
    for label, cond_a, cond_b in family:
        acc_a = d[(d["condition"] == cond_a) & (d["metric"] == "accuracy")]["score"].mean()
        conds_b = [cond_b] if isinstance(cond_b, str) else cond_b
        acc_b = d[(d["condition"].isin(conds_b)) & (d["metric"] == "accuracy")]["score"].mean()
        rows.append({"comparison": label, "accuracy_a": acc_a, "accuracy_b": acc_b})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ra")
    parser.add_argument("--prior-mode", choices=["independent", "combined"], default="combined")
    args = parser.parse_args()
    config: DatasetConfig = DATASETS[args.dataset]
    prefix = config.file_prefix

    main_results = pd.read_csv(RESULTS_DIR / "tables" / f"{prefix}main_results.csv")
    d = main_results[main_results["target"] == "disease_status"].copy()

    expr = np.load(DATA_DIR / f"{prefix}pseudobulk_expr.npy")
    meta = pd.read_parquet(DATA_DIR / f"{prefix}pseudobulk_meta.parquet")
    genes = pd.read_csv(DATA_DIR / f"{prefix}pseudobulk_genes.csv")["gene_symbol"].tolist()
    splits = make_splits(meta, n_splits=N_SPLITS, n_seeds=N_SEEDS, stratify_col="disease")

    sig_lines = ["# Paired significance tests -- donor-level disease-status AUROC\n"]
    scorecard_frames = []
    percentiles: dict[str, dict[str, float]] = {}

    for prior_name, signed in PRIORS:
        print(f"--- {prior_name} ---", flush=True)
        family = comparison_family(prior_name, signed)
        tests = run_paired_tests(d, "auroc", family)
        tests.insert(0, "prior", prior_name)
        tests = tests.merge(_accuracy_means(d, family), on="comparison")

        sig_lines.append(f"\n## {prior_name} (Bonferroni alpha = 0.05 / {len(family)})")
        sig_lines.append(tests.to_string(index=False))
        tests.to_csv(RESULTS_DIR / "tables" / f"{prefix}significance_tests_{prior_name}.csv", index=False)
        scorecard_frames.append(tests)

        # --- permutation null: refit real + baseline_pca embeddings once, reuse across permutations ---
        real_edges = pd.read_parquet(DATA_DIR / f"{prefix}graph_{prior_name}_real.parquet")
        mask, sign_mat, units = edges_to_mask(real_edges, genes)
        n_components = len(units)

        fold_emb_pca = fit_transform_cv(expr, meta, lambda seed: PCAEncoder(n_components, seed=seed), splits)
        fold_emb_real = fit_transform_cv(expr, meta, lambda seed: PriorEncoder(mask, sign_mat, seed=seed), splits)

        null_pca = label_permutation_null(
            fold_emb_pca, meta, "disease", positive_label=config.disease_positive_label, n_permutations=N_PERMUTATIONS, seed=0
        )
        null_real = label_permutation_null(
            fold_emb_real, meta, "disease", positive_label=config.disease_positive_label, n_permutations=N_PERMUTATIONS, seed=0
        )
        null_pca.to_csv(RESULTS_DIR / "tables" / f"{prefix}permutation_null_{prior_name}_baseline_pca.csv", index=False)
        null_real.to_csv(RESULTS_DIR / "tables" / f"{prefix}permutation_null_{prior_name}_real.csv", index=False)

        mean_real_auroc = d[(d["condition"] == f"{prior_name}_real") & (d["metric"] == "auroc")]["score"].mean()
        mean_pca_auroc = d[(d["condition"] == f"{prior_name}_baseline_pca") & (d["metric"] == "auroc")]["score"].mean()
        pct_real = permutation_percentile(mean_real_auroc, null_real["mean_auroc"].to_numpy())
        pct_pca = permutation_percentile(mean_pca_auroc, null_pca["mean_auroc"].to_numpy())
        percentiles[prior_name] = {"real": pct_real, "baseline_pca": pct_pca}
        sig_lines.append(
            f"\nPermutation null (n={N_PERMUTATIONS} donor-level label permutations, same CV splits): "
            f"real AUROC {mean_real_auroc:.4f} is at the {pct_real:.1f}th percentile of its own null "
            f"(range {null_real['mean_auroc'].min():.3f}-{null_real['mean_auroc'].max():.3f}); "
            f"baseline_pca AUROC {mean_pca_auroc:.4f} is at the {pct_pca:.1f}th percentile of its null "
            f"(range {null_pca['mean_auroc'].min():.3f}-{null_pca['mean_auroc'].max():.3f})."
        )

    if args.prior_mode == "combined" and "combined_real" in d["condition"].unique():
        print("--- combined ---", flush=True)
        family = combined_comparison_family()
        tests = run_paired_tests(d, "auroc", family)
        tests.insert(0, "prior", "combined")
        tests = tests.merge(_accuracy_means(d, family), on="comparison")

        sig_lines.append(f"\n## combined (Bonferroni alpha = 0.05 / {len(family)})")
        sig_lines.append(tests.to_string(index=False))
        tests.to_csv(RESULTS_DIR / "tables" / f"{prefix}significance_tests_combined.csv", index=False)
        scorecard_frames.append(tests)

        exploratory_family = combined_exploratory_family()
        exploratory_rows = []
        for label, cond_a, cond_b in exploratory_family:
            one_off = run_paired_tests(d, "auroc", [(label, cond_a, cond_b)], alpha=0.05)
            exploratory_rows.append(one_off)
        exploratory_tests = pd.concat(exploratory_rows, ignore_index=True)
        exploratory_tests.insert(0, "prior", "combined")
        exploratory_tests = exploratory_tests.merge(_accuracy_means(d, exploratory_family), on="comparison")
        sig_lines.append(
            "\n## combined -- exploratory ablation (isolates which prior drives the effect; "
            "uncorrected, NOT part of the 6-comparison family above)"
        )
        sig_lines.append(exploratory_tests.to_string(index=False))
        exploratory_tests.to_csv(RESULTS_DIR / "tables" / f"{prefix}significance_tests_combined_exploratory.csv", index=False)

        # --- permutation null for the combined condition, same pattern as each single prior ---
        tf_edges = pd.read_parquet(DATA_DIR / f"{prefix}graph_combined_real_tf.parquet")
        epi_edges = pd.read_parquet(DATA_DIR / f"{prefix}graph_combined_real_epi.parquet")
        mask, sign_mat, units = combined_mod.joint_mask_from_edges(tf_edges, epi_edges, genes)
        n_components = len(units)

        fold_emb_pca = fit_transform_cv(expr, meta, lambda seed: PCAEncoder(n_components, seed=seed), splits)
        fold_emb_real = fit_transform_cv(expr, meta, lambda seed: PriorEncoder(mask, sign_mat, seed=seed), splits)

        null_pca = label_permutation_null(
            fold_emb_pca, meta, "disease", positive_label=config.disease_positive_label, n_permutations=N_PERMUTATIONS, seed=0
        )
        null_real = label_permutation_null(
            fold_emb_real, meta, "disease", positive_label=config.disease_positive_label, n_permutations=N_PERMUTATIONS, seed=0
        )
        null_pca.to_csv(RESULTS_DIR / "tables" / f"{prefix}permutation_null_combined_baseline_pca.csv", index=False)
        null_real.to_csv(RESULTS_DIR / "tables" / f"{prefix}permutation_null_combined_real.csv", index=False)

        mean_real_auroc = d[(d["condition"] == "combined_real") & (d["metric"] == "auroc")]["score"].mean()
        mean_pca_auroc = d[(d["condition"] == "combined_baseline_pca") & (d["metric"] == "auroc")]["score"].mean()
        pct_real = permutation_percentile(mean_real_auroc, null_real["mean_auroc"].to_numpy())
        pct_pca = permutation_percentile(mean_pca_auroc, null_pca["mean_auroc"].to_numpy())
        percentiles["combined"] = {"real": pct_real, "baseline_pca": pct_pca}
        sig_lines.append(
            f"\nPermutation null (n={N_PERMUTATIONS} donor-level label permutations, same CV splits): "
            f"real AUROC {mean_real_auroc:.4f} is at the {pct_real:.1f}th percentile of its own null "
            f"(range {null_real['mean_auroc'].min():.3f}-{null_real['mean_auroc'].max():.3f}); "
            f"baseline_pca AUROC {mean_pca_auroc:.4f} is at the {pct_pca:.1f}th percentile of its null "
            f"(range {null_pca['mean_auroc'].min():.3f}-{null_pca['mean_auroc'].max():.3f})."
        )

    scorecard = pd.concat(scorecard_frames, ignore_index=True)
    scorecard["permutation_percentile_real"] = scorecard["prior"].map(lambda p: percentiles[p]["real"])
    scorecard["permutation_percentile_baseline_pca"] = scorecard["prior"].map(lambda p: percentiles[p]["baseline_pca"])
    scorecard.to_csv(RESULTS_DIR / "tables" / f"{prefix}scorecard.csv", index=False)

    out_path = RESULTS_DIR / "tables" / f"{prefix}significance_tests.md"
    out_path.write_text("\n".join(sig_lines) + "\n")
    print("\n".join(sig_lines))
    print(f"\nWrote {out_path} and results/tables/{prefix}scorecard.csv")


if __name__ == "__main__":
    main()
