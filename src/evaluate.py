"""Donor-level evaluation protocol, shared across every prior tested by this
harness.

The encoder (PCA or prior-masked, real graph or control) is refit inside
every CV fold on training-fold profiles only, then applied to the held-out
fold -- fitting on all profiles up front would leak held-out donors'
expression distribution into the embedding even though no labels are
involved. Splits are generated once per (seed, fold) from metadata alone
(donor groups + target stratification) so every encoder variant, for every
prior, is scored on identical splits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

DEFAULT_N_JOBS = min(8, max(1, (os.cpu_count() or 1) - 1))


def assert_no_donor_leakage(meta: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    train_donors = set(meta.iloc[train_idx]["donor_id"])
    test_donors = set(meta.iloc[test_idx]["donor_id"])
    overlap = train_donors & test_donors
    assert not overlap, f"donor leakage across train/test: {overlap}"


@dataclass
class FoldEmbeddings:
    seed: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    Z_train: np.ndarray
    Z_test: np.ndarray


def make_splits(
    meta: pd.DataFrame,
    n_splits: int = 6,
    n_seeds: int = 5,
    stratify_col: str = "disease",
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    y = meta[stratify_col].to_numpy()
    groups = meta["donor_id"].to_numpy()
    dummy_X = np.zeros((len(meta), 1))

    splits = []
    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(sgkf.split(dummy_X, y, groups)):
            assert_no_donor_leakage(meta, train_idx, test_idx)
            splits.append((seed, fold, train_idx, test_idx))
    return splits


def _fit_one_fold(
    X: np.ndarray,
    seed: int,
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    encoder_factory: Callable[[int], object],
) -> FoldEmbeddings:
    # each fold fit runs in its own worker process; cap torch's intra-op
    # threading so it doesn't oversubscribe against the other worker processes
    # also competing for the same cores.
    import torch

    torch.set_num_threads(1)

    encoder = encoder_factory(seed)
    encoder.fit(X[train_idx])
    Z_train = encoder.transform(X[train_idx])
    Z_test = encoder.transform(X[test_idx])
    return FoldEmbeddings(seed, fold, train_idx, test_idx, Z_train, Z_test)


def fit_transform_cv(
    X: np.ndarray,
    meta: pd.DataFrame,
    encoder_factory: Callable[[int], object],
    splits: list[tuple[int, int, np.ndarray, np.ndarray]],
    n_jobs: int = DEFAULT_N_JOBS,
) -> list[FoldEmbeddings]:
    """Refit `encoder_factory(seed)` on each fold's training rows only.

    Folds are independent (no shared state, no shared RNG stream), so they're
    fit in parallel worker processes via joblib. `Parallel` returns results in
    the same order the tasks were submitted (the `splits` order), regardless
    of which worker finishes first, so downstream result collection stays
    deterministic.

    Every encoder in this harness (PCAEncoder, PriorEncoder) is unsupervised
    -- fit() never takes labels -- so fold embeddings are identical no matter
    which target column is later scored against them. This is what makes
    `src/significance.py::label_permutation_null` cheap: embeddings are fit
    once and reused across every permutation.
    """
    out = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_one_fold)(X, seed, fold, train_idx, test_idx, encoder_factory)
        for seed, fold, train_idx, test_idx in splits
    )
    return out


def score_classification_probe(
    fold_embeddings: list[FoldEmbeddings],
    meta: pd.DataFrame,
    label_col: str,
    positive_label: str | None = None,
    include_cell_type_covariate: bool = True,
) -> pd.DataFrame:
    """Logistic-regression probe per fold; predictions averaged per held-out
    donor across its cell-type profiles before scoring donor-level AUROC and
    accuracy.

    `positive_label` set -> binary probe (e.g. sex, or a binary fallback
    target); the label column is binarized to `label_col == positive_label`.
    `positive_label` None -> multiclass probe (e.g. the 3-tier SLE activity
    target): reports macro one-vs-rest AUROC (metric "auroc_macro_ovr") in
    addition to accuracy. Same donor-level aggregation and CV protocol either
    way, so the two priors and every sanity probe share this one function.
    """
    rows = []
    for fe in fold_embeddings:
        train_meta = meta.iloc[fe.train_idx]
        test_meta = meta.iloc[fe.test_idx]

        if positive_label is not None:
            y_train = (train_meta[label_col] == positive_label).astype(int).to_numpy()
            y_test = (test_meta[label_col] == positive_label).astype(int).to_numpy()
        else:
            y_train = train_meta[label_col].to_numpy()
            y_test = test_meta[label_col].to_numpy()

        if include_cell_type_covariate:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            ct_train = ohe.fit_transform(train_meta[["cell_type"]])
            ct_test = ohe.transform(test_meta[["cell_type"]])
            X_train = np.hstack([fe.Z_train, ct_train])
            X_test = np.hstack([fe.Z_test, ct_test])
        else:
            X_train, X_test = fe.Z_train, fe.Z_test

        if len(np.unique(y_train)) < 2:
            continue

        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)

        pred_df = test_meta[["donor_id"]].copy()
        pred_df["y"] = y_test
        proba_cols = [f"proba_{c}" for c in clf.classes_]
        for col, p in zip(proba_cols, proba.T):
            pred_df[col] = p
        donor_pred = pred_df.groupby("donor_id", observed=True).agg(
            **{col: (col, "mean") for col in proba_cols}, y=("y", "first")
        )
        donor_proba = donor_pred[proba_cols].to_numpy()
        donor_y = donor_pred["y"].to_numpy()
        n_test_donors = len(donor_pred)

        pred_labels = clf.classes_[np.argmax(donor_proba, axis=1)]
        acc = float((pred_labels == donor_y).mean())
        rows.append(
            {"seed": fe.seed, "fold": fe.fold, "metric": "accuracy", "score": acc, "n_test_donors": n_test_donors}
        )

        if len(clf.classes_) == 2:
            pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
            auroc = np.nan if len(np.unique(donor_y)) < 2 else roc_auc_score(donor_y, donor_proba[:, pos_idx])
            rows.append(
                {"seed": fe.seed, "fold": fe.fold, "metric": "auroc", "score": auroc, "n_test_donors": n_test_donors}
            )
        else:
            try:
                auroc = roc_auc_score(donor_y, donor_proba, multi_class="ovr", average="macro", labels=clf.classes_)
            except ValueError:
                auroc = np.nan
            rows.append(
                {
                    "seed": fe.seed,
                    "fold": fe.fold,
                    "metric": "auroc_macro_ovr",
                    "score": auroc,
                    "n_test_donors": n_test_donors,
                }
            )

    return pd.DataFrame(rows)


def score_multiclass_probe(
    fold_embeddings: list[FoldEmbeddings],
    meta: pd.DataFrame,
    label_col: str,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """kNN label-preservation probe: does embedding geometry cluster profiles by
    `label_col`? Per-profile accuracy, no donor aggregation (unlike
    score_classification_probe) -- this isn't a donor-level generalization
    claim, just a working-pipeline sanity check that the encoder captures
    *some* biologically meaningful structure (used as the cell-type-identity
    anchor, which has far more data points than any donor-level target).
    """
    rows = []
    for fe in fold_embeddings:
        y_train = meta.iloc[fe.train_idx][label_col].to_numpy()
        y_test = meta.iloc[fe.test_idx][label_col].to_numpy()

        clf = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(y_train)))
        clf.fit(fe.Z_train, y_train)
        pred = clf.predict(fe.Z_test)
        acc = float((pred == y_test).mean())
        rows.append({"seed": fe.seed, "fold": fe.fold, "metric": "accuracy", "score": acc, "n_test": len(y_test)})

    return pd.DataFrame(rows)


def score_continuous_probe(
    fold_embeddings: list[FoldEmbeddings],
    meta: pd.DataFrame,
    covariate_col: str,
) -> pd.DataFrame:
    """Negative control: how well does the embedding predict a technical covariate
    (e.g. per-profile sequencing depth)? Ridge probe, out-of-fold R^2.
    """
    rows = []
    for fe in fold_embeddings:
        y_train = meta.iloc[fe.train_idx][covariate_col].to_numpy(dtype=float)
        y_test = meta.iloc[fe.test_idx][covariate_col].to_numpy(dtype=float)

        reg = Ridge(alpha=1.0)
        reg.fit(fe.Z_train, y_train)
        pred = reg.predict(fe.Z_test)
        rows.append({"seed": fe.seed, "fold": fe.fold, "metric": "r2", "score": r2_score(y_test, pred)})

    return pd.DataFrame(rows)
