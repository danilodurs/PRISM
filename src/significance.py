"""Paired significance testing and label-permutation null, shared across
every prior and every comparison family in this harness.

Every condition for a given prior is scored on the identical set of donor-
level (seed, fold) splits, so a comparison between two conditions is
naturally paired: for each split, take the score difference, then test
whether that paired difference is distinguishable from zero. Reports both
the Wilcoxon signed-rank test (primary -- nonparametric, appropriate for a
bounded score with a modest number of paired samples and no assumption of
normally-distributed differences) and a paired t-test (secondary, standard
companion statistic). Bonferroni correction is applied across whichever
comparison family is passed in, so it naturally scales from a signed
prior's larger family (includes the sign-flip control) to an unsigned
prior's smaller one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

from src.evaluate import FoldEmbeddings, score_classification_probe


def _paired_scores(d: pd.DataFrame, cond_a: str | list[str], cond_b: str | list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Align two conditions' per-(seed,fold) scores by split. `cond_a`/`cond_b`
    may each be a single condition name or a list of condition names to pool
    (e.g. several random-control seeds averaged per split before pairing).
    """
    conds_a = [cond_a] if isinstance(cond_a, str) else list(cond_a)
    conds_b = [cond_b] if isinstance(cond_b, str) else list(cond_b)
    a = d[d["condition"].isin(conds_a)].groupby(["seed", "fold"])["score"].mean()
    b = d[d["condition"].isin(conds_b)].groupby(["seed", "fold"])["score"].mean()
    idx = a.index.intersection(b.index)
    assert len(idx) > 0, f"no overlapping splits between {conds_a} and {conds_b}"
    return a.loc[idx].to_numpy(), b.loc[idx].to_numpy()


def run_paired_tests(
    results_df: pd.DataFrame,
    metric: str,
    comparison_family: list[tuple[str, str | list[str], str | list[str]]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """`comparison_family`: list of (label, cond_a, cond_b) tuples defining the
    full family of comparisons to Bonferroni-correct together. Each cond_a/
    cond_b is a condition name or list of names to pool before pairing.
    """
    d = results_df[results_df["metric"] == metric]
    corrected_alpha = alpha / len(comparison_family)

    rows = []
    for label, cond_a, cond_b in comparison_family:
        x, y = _paired_scores(d, cond_a, cond_b)
        diff = x - y
        wil = stats.wilcoxon(x, y)
        tt = stats.ttest_rel(x, y)
        rows.append(
            {
                "comparison": label,
                "n_paired_splits": len(x),
                "mean_a": x.mean(),
                "mean_b": y.mean(),
                "mean_diff": diff.mean(),
                "diff_std": diff.std(ddof=1),
                "wilcoxon_stat": wil.statistic,
                "wilcoxon_p": wil.pvalue,
                "ttest_stat": tt.statistic,
                "ttest_p": tt.pvalue,
                "bonferroni_alpha": corrected_alpha,
                "significant_wilcoxon": bool(wil.pvalue < corrected_alpha),
                "significant_ttest": bool(tt.pvalue < corrected_alpha),
            }
        )
    return pd.DataFrame(rows)


def label_permutation_null(
    fold_embeddings: list[FoldEmbeddings],
    meta: pd.DataFrame,
    label_col: str,
    positive_label: str | None = None,
    n_permutations: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Shuffle `label_col` across donors (preserving each class's donor count
    and within-donor consistency) and rescore the SAME fold embeddings against
    each permutation. Valid because every encoder in this harness is
    unsupervised (fit() never sees labels, see src/evaluate.py::fit_transform_cv)
    -- the embeddings are identical no matter how the label is permuted, so
    refitting only the downstream probe per permutation is mathematically
    equivalent to refitting the whole encoder+probe pipeline from scratch each
    time, at a fraction of the cost.
    """
    donor_labels = meta[["donor_id", label_col]].drop_duplicates().set_index("donor_id")[label_col]
    rng = np.random.default_rng(seed)

    auroc_metric = "auroc" if positive_label is not None else "auroc_macro_ovr"
    rows = []
    # this loop refits a small LogisticRegression per (permutation, fold) --
    # thousands of tiny fits. Left to its default, BLAS spins up a full
    # thread pool per fit, which oversubscribes the machine (more runnable
    # threads than cores) for no throughput benefit on matrices this small.
    # One thread per fit is faster in aggregate here.
    with threadpool_limits(limits=1):
        for perm in range(n_permutations):
            permuted = rng.permutation(donor_labels.to_numpy())
            donor_map = dict(zip(donor_labels.index, permuted))
            perm_meta = meta.copy()
            perm_meta[label_col] = perm_meta["donor_id"].map(donor_map)

            scored = score_classification_probe(fold_embeddings, perm_meta, label_col, positive_label=positive_label)
            auroc_rows = scored[scored["metric"] == auroc_metric]
            acc_rows = scored[scored["metric"] == "accuracy"]
            rows.append(
                {
                    "permutation": perm,
                    "mean_auroc": auroc_rows["score"].mean(),
                    "mean_accuracy": acc_rows["score"].mean(),
                    "n_folds_scored": len(auroc_rows),
                }
            )
    return pd.DataFrame(rows)


def permutation_percentile(observed: float, null_scores: np.ndarray) -> float:
    return float((null_scores < observed).mean() * 100)
