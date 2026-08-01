"""Paired significance tests, label-permutation null, and the prior x
condition scorecard -- run identically for both priors via the shared
src/significance.py functions. AUROC (donor-level disease-status prediction)
is the primary tested metric, matching TRACE's precedent; accuracy is
reported in the scorecard for descriptive context but not separately
Bonferroni-corrected.

Comparisons per prior (the "comparison family" Bonferroni-corrected
together): real vs baseline_pca, real vs pooled C1 (degree-preserving,
5 seeds), real vs pooled C2 (fully random, 5 seeds), and -- TF-target only,
since it's the only signed prior -- real vs C3 (sign-flipped). This is why
the correction denominator differs between priors (4 vs 3): it isn't
arbitrary, it's exactly the set of controls that prior has.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.encoders import PCAEncoder, PriorEncoder
from src.evaluate import fit_transform_cv, make_splits
from src.graph import edges_to_mask
from src.significance import label_permutation_null, permutation_percentile, run_paired_tests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_SPLITS = 6
N_SEEDS = 5
N_CONTROL_SEEDS = 5
N_PERMUTATIONS = 100
DISEASE_POSITIVE_LABEL = "systemic lupus erythematosus"
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


def main() -> None:
    main_results = pd.read_csv(RESULTS_DIR / "tables" / "main_results.csv")
    d = main_results[main_results["target"] == "disease_status"].copy()

    expr = np.load(DATA_DIR / "pseudobulk_expr.npy")
    meta = pd.read_parquet(DATA_DIR / "pseudobulk_meta.parquet")
    genes = pd.read_csv(DATA_DIR / "pseudobulk_genes.csv")["gene_symbol"].tolist()
    splits = make_splits(meta, n_splits=N_SPLITS, n_seeds=N_SEEDS, stratify_col="disease")

    sig_lines = ["# Paired significance tests -- donor-level disease-status AUROC\n"]
    scorecard_frames = []
    percentiles: dict[str, dict[str, float]] = {}

    for prior_name, signed in PRIORS:
        print(f"--- {prior_name} ---", flush=True)
        family = comparison_family(prior_name, signed)
        tests = run_paired_tests(d, "auroc", family)
        tests.insert(0, "prior", prior_name)

        acc_family_means = []
        for label, cond_a, cond_b in family:
            acc_a = d[(d["condition"] == cond_a) & (d["metric"] == "accuracy")]["score"].mean()
            conds_b = [cond_b] if isinstance(cond_b, str) else cond_b
            acc_b = d[(d["condition"].isin(conds_b)) & (d["metric"] == "accuracy")]["score"].mean()
            acc_family_means.append({"comparison": label, "accuracy_a": acc_a, "accuracy_b": acc_b})
        tests = tests.merge(pd.DataFrame(acc_family_means), on="comparison")

        sig_lines.append(f"\n## {prior_name} (Bonferroni alpha = 0.05 / {len(family)})")
        sig_lines.append(tests.to_string(index=False))
        tests.to_csv(RESULTS_DIR / "tables" / f"significance_tests_{prior_name}.csv", index=False)
        scorecard_frames.append(tests)

        # --- permutation null: refit real + baseline_pca embeddings once, reuse across permutations ---
        real_edges = pd.read_parquet(DATA_DIR / f"graph_{prior_name}_real.parquet")
        mask, sign_mat, units = edges_to_mask(real_edges, genes)
        n_components = len(units)

        fold_emb_pca = fit_transform_cv(expr, meta, lambda seed: PCAEncoder(n_components, seed=seed), splits)
        fold_emb_real = fit_transform_cv(expr, meta, lambda seed: PriorEncoder(mask, sign_mat, seed=seed), splits)

        null_pca = label_permutation_null(
            fold_emb_pca, meta, "disease", positive_label=DISEASE_POSITIVE_LABEL, n_permutations=N_PERMUTATIONS, seed=0
        )
        null_real = label_permutation_null(
            fold_emb_real, meta, "disease", positive_label=DISEASE_POSITIVE_LABEL, n_permutations=N_PERMUTATIONS, seed=0
        )
        null_pca.to_csv(RESULTS_DIR / "tables" / f"permutation_null_{prior_name}_baseline_pca.csv", index=False)
        null_real.to_csv(RESULTS_DIR / "tables" / f"permutation_null_{prior_name}_real.csv", index=False)

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

    scorecard = pd.concat(scorecard_frames, ignore_index=True)
    scorecard["permutation_percentile_real"] = scorecard["prior"].map(lambda p: percentiles[p]["real"])
    scorecard["permutation_percentile_baseline_pca"] = scorecard["prior"].map(lambda p: percentiles[p]["baseline_pca"])
    scorecard.to_csv(RESULTS_DIR / "tables" / "scorecard.csv", index=False)

    out_path = RESULTS_DIR / "tables" / "significance_tests.md"
    out_path.write_text("\n".join(sig_lines) + "\n")
    print("\n".join(sig_lines))
    print(f"\nWrote {out_path} and results/tables/scorecard.csv")


if __name__ == "__main__":
    main()
