"""Census access and donor x cell_type pseudobulk construction for the SLE
PBMC dataset (Perez et al. 2022, CELLxGENE Census dataset
`218acb0f-9f2f-4f76-b90b-15a4b7c7f629`) -- the only SLE cohort in Census
(confirmed at validation time: every SLE-labeled cell in Census belongs to
this one dataset). See `results/tables/step0_validation.md` for why this
dataset's target is SLE-vs-normal disease status rather than a 3-tier
activity score: the dataset carries no SLEDAI or other composite activity
score, and its coarse `disease_state` field (flare/managed/treated) is both
severely imbalanced and, for some donors, inconsistent across cells from the
same donor_id (a per-visit label, not a stable per-donor one).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATASET_ID = "218acb0f-9f2f-4f76-b90b-15a4b7c7f629"
CENSUS_VERSION = "2025-11-08"
MIN_CELLS_PER_PROFILE = 10
MIN_DONOR_COVERAGE = 0.8  # keep a cell type only if present at threshold in >= this fraction of donors
RAW_H5AD_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_dataset.h5ad"


def load_obs(census_version: str = CENSUS_VERSION, dataset_id: str = DATASET_ID) -> pd.DataFrame:
    """Standard Census obs fields only -- used for the validation report.
    Non-standard fields specific to this submission (disease_state,
    Processing_Cohort, author_cell_type, cell_state) aren't part of Census's
    schema and are read separately, directly from the source h5ad, by
    `load_extra_obs_fields` -- see step0_validation.md for how the two are
    combined for reporting.
    """
    import cellxgene_census

    census = cellxgene_census.open_soma(census_version=census_version)
    try:
        obs = cellxgene_census.get_obs(census, "homo_sapiens", value_filter=f"dataset_id == '{dataset_id}'")
    finally:
        census.close()
    for c in obs.columns:
        if str(obs[c].dtype) == "category":
            obs[c] = obs[c].astype(str)
    return obs


def load_extra_obs_fields(h5ad_path: Path = RAW_H5AD_PATH) -> pd.DataFrame:
    """Submission-specific obs columns not preserved in Census's standardized
    schema: disease_state (flare/managed/treated/na -- the closest thing to
    an activity label this dataset has), Processing_Cohort (technical batch),
    author_cell_type and cell_state (finer-grained annotations than Census's
    own cell_type, relevant for a future ligand-receptor phase). Read
    directly from the local h5ad's obs group so row order is self-consistent
    (no join against a separately-fetched Census obs call).
    """
    import h5py

    def read_cat(obs_group, col):
        g = obs_group[col]
        cats = np.array([c.decode() for c in g["categories"][:]])
        codes = g["codes"][:]
        return pd.Categorical.from_codes(codes, categories=cats)

    with h5py.File(h5ad_path, "r") as h5:
        obs = h5["obs"]
        return pd.DataFrame(
            {
                "disease_state": read_cat(obs, "disease_state"),
                "processing_cohort": read_cat(obs, "Processing_Cohort"),
                "author_cell_type": read_cat(obs, "author_cell_type"),
                "cell_state": read_cat(obs, "cell_state"),
            }
        )


def eligible_cell_types(
    obs: pd.DataFrame,
    min_cells: int = MIN_CELLS_PER_PROFILE,
    min_donor_coverage: float = MIN_DONOR_COVERAGE,
) -> list[str]:
    """Cell types present at >= min_cells in >= min_donor_coverage fraction of donors."""
    counts = obs.groupby(["donor_id", "cell_type"], observed=True).size().unstack(fill_value=0)
    frac_donors = (counts >= min_cells).mean(axis=0)
    return frac_donors[frac_donors >= min_donor_coverage].index.tolist()


def gene_universe(dataset_genes: set[str], *prior_gene_sets: set[str]) -> list[str]:
    """The shared gene universe used for pseudobulk construction: the union of
    genes touched by every prior's raw resource, intersected with genes
    actually present in this dataset's panel. Deliberately not restricted to
    any single prior's resource -- with two priors sharing one harness,
    letting one prior's coverage define the universe would arbitrarily
    disadvantage the other.
    """
    prior_genes: set[str] = set()
    for s in prior_gene_sets:
        prior_genes |= s
    return sorted(prior_genes & dataset_genes)


def _read_categorical_or_plain(group, col: str):
    import h5py

    item = group[col]
    if isinstance(item, h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in item["categories"][:]])
        codes = item["codes"][:]
        return pd.Categorical.from_codes(codes, categories=cats)
    return item[:]


def _read_h5ad_obs(h5: "h5py.File") -> pd.DataFrame:
    obs = h5["obs"]
    cols = ["donor_id", "cell_type", "disease", "sex", "self_reported_ethnicity"]
    return pd.DataFrame({c: _read_categorical_or_plain(obs, c) for c in cols})


def load_dataset_gene_panel(h5ad_path: Path = RAW_H5AD_PATH) -> set[str]:
    """This dataset's own measured gene panel (its `raw/var/feature_name`),
    NOT Census's global cross-dataset gene panel (which is a union across
    every dataset in Census and can include genes this specific submission
    never measured). The gene universe must be intersected against what this
    dataset actually has, or a "shared universe" gene could silently end up
    as an all-zero pseudobulk column.
    """
    import h5py

    with h5py.File(h5ad_path, "r") as h5:
        names = _read_categorical_or_plain(h5["raw"]["var"], "feature_name")
    return {n.decode() if isinstance(n, bytes) else str(n) for n in names}


def build_pseudobulk_streaming(
    genes: list[str],
    cell_types: list[str],
    h5ad_path: Path = RAW_H5AD_PATH,
    min_cells: int = MIN_CELLS_PER_PROFILE,
    chunk_size: int = 100_000,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Sum raw counts per (donor, cell_type) group meeting the min-cell
    threshold, streaming the source h5ad's raw counts in row-chunks rather
    than materializing the full 1.26M-cell x 30k-gene sparse matrix in memory
    at once (unlike TRACE's much smaller RA dataset, this dataset's full raw
    matrix does not fit comfortably in memory -- a chunked, group-vectorized
    accumulation keeps peak memory bounded to one chunk's worth of cells).

    Each chunk's group sums are computed as one sparse matmul (a
    group-indicator matrix against the chunk's gene-subset matrix) rather
    than a per-row Python loop, which is what keeps this tractable at this
    scale.

    Normalization uses each cell's whole-panel raw count total (summed from
    the full un-subsetted chunk, before restricting to the gene universe)
    rather than a sum restricted to the fetched gene subset, matching TRACE's
    approach so library-size normalization isn't biased by which genes
    happen to be in this harness's gene universe. This dataset's source h5ad
    has no submitter-provided library-size column (unlike TRACE's RA
    dataset's `total_counts`) -- computed directly from raw counts instead.

    Returns (metadata, log1p-CPM matrix [profiles x genes], gene_symbols) with
    genes ordered as the input `genes` list.
    """
    import h5py

    with h5py.File(h5ad_path, "r") as h5:
        obs = _read_h5ad_obs(h5)
        feature_names = _read_categorical_or_plain(h5["raw"]["var"], "feature_name")
        feature_names = [v.decode() if isinstance(v, bytes) else str(v) for v in feature_names]
        var = pd.DataFrame({"feature_name": feature_names})

        gene_set = set(genes)
        keep_var = var["feature_name"].isin(gene_set).to_numpy()
        var_keep = var.loc[keep_var]
        first_per_symbol = ~var_keep["feature_name"].duplicated(keep="first")
        keep_col_idx = var_keep.index[first_per_symbol].to_numpy()
        kept_gene_names = var.loc[keep_col_idx, "feature_name"].tolist()
        # column position within the (subset, ordered-by-`genes`) output matrix
        kept_gene_set = set(kept_gene_names)
        name_to_out_col = {g: i for i, g in enumerate(genes) if g in kept_gene_set}
        out_cols_for_keep_idx = np.array([name_to_out_col[g] for g in kept_gene_names])
        n_out_genes = len(genes)

        keep_cell_mask = obs["cell_type"].isin(cell_types).to_numpy()
        # vectorized group assignment: factorize a combined donor||cell_type key,
        # restricted to kept cells (pd.factorize assigns code -1 to the None/NaN
        # sentinel used for dropped cells) -- a 1.26M-row Python loop here would be
        # needlessly slow next to this C-level pandas op.
        combo_key = obs["donor_id"].astype(str) + "||" + obs["cell_type"].astype(str)
        combo_key = combo_key.where(keep_cell_mask, other=None)
        codes, uniques = pd.factorize(combo_key, sort=True)
        group_idx_per_cell = codes.astype(np.int64)
        group_keys = [tuple(u.split("||")) for u in uniques]
        n_groups = len(group_keys)

        expr_accum = np.zeros((n_groups, n_out_genes), dtype=np.float64)
        n_cells_accum = np.zeros(n_groups, dtype=np.int64)
        libsize_accum = np.zeros(n_groups, dtype=np.float64)

        indptr = h5["raw"]["X"]["indptr"][:]
        data_ds = h5["raw"]["X"]["data"]
        indices_ds = h5["raw"]["X"]["indices"]
        n_cells_total, n_genes_total = h5["raw"]["X"].attrs["shape"]

        for start in range(0, n_cells_total, chunk_size):
            end = min(start + chunk_size, n_cells_total)
            chunk_group_idx = group_idx_per_cell[start:end]
            if (chunk_group_idx < 0).all():
                continue

            row_start, row_end = indptr[start], indptr[end]
            chunk_data = data_ds[row_start:row_end]
            chunk_indices = indices_ds[row_start:row_end]
            chunk_indptr = indptr[start : end + 1] - row_start
            chunk_csr = sp.csr_matrix((chunk_data, chunk_indices, chunk_indptr), shape=(end - start, n_genes_total))
            chunk_lib_size = np.asarray(chunk_csr.sum(axis=1)).ravel()  # whole-panel total, before gene subsetting
            chunk_sub = chunk_csr[:, keep_col_idx]  # (chunk_rows, n_kept_cols), still gene-subset order = kept_gene_names

            valid = chunk_group_idx >= 0
            if not valid.any():
                continue
            rows_valid = np.nonzero(valid)[0]
            grp_valid = chunk_group_idx[valid]

            indicator = sp.csr_matrix(
                (np.ones(len(rows_valid)), (grp_valid, rows_valid)),
                shape=(n_groups, end - start),
            )
            delta = indicator @ chunk_sub  # (n_groups, n_kept_cols)
            delta = np.asarray(delta.todense())
            expr_accum[:, out_cols_for_keep_idx] += delta

            n_cells_accum += np.bincount(grp_valid, minlength=n_groups)
            libsize_accum += np.bincount(grp_valid, weights=chunk_lib_size[valid], minlength=n_groups)

    keep_groups = n_cells_accum >= min_cells
    kept_group_keys = [g for g, keep in zip(group_keys, keep_groups) if keep]
    cpm = expr_accum[keep_groups] / libsize_accum[keep_groups, None] * 1e6
    log_expr = np.log1p(cpm)

    meta_rows = []
    donor_meta = obs.drop_duplicates("donor_id").set_index("donor_id")[["disease", "sex", "self_reported_ethnicity"]]
    donor_meta.index = donor_meta.index.astype(str)
    for (donor_id, cell_type), n_cells in zip(kept_group_keys, n_cells_accum[keep_groups]):
        row = donor_meta.loc[donor_id].to_dict()
        row["donor_id"] = donor_id
        row["cell_type"] = cell_type
        row["n_cells"] = int(n_cells)
        meta_rows.append(row)
    meta = pd.DataFrame(meta_rows)
    meta["lib_size"] = libsize_accum[keep_groups]

    n_donors = meta["donor_id"].nunique()
    assert n_donors > 1, "pseudobulk collapsed to <=1 donor"
    return meta, log_expr, genes
