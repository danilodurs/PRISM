"""Build the donor x cell_type pseudobulk matrix over the shared gene
universe (union of both priors' raw resources, intersected with this
dataset's gene panel -- see src/data.py::gene_universe for why neither
prior's coverage is allowed to define the universe alone).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import (
    RAW_H5AD_PATH,
    build_pseudobulk_streaming,
    eligible_cell_types,
    gene_universe,
    load_dataset_gene_panel,
    load_obs,
)
from src.priors.epigenomic import load_abc_edges
from src.priors.tf_target import load_dorothea

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"


def main() -> None:
    dataset_genes = load_dataset_gene_panel()

    dorothea = load_dorothea()
    dorothea_genes = set(dorothea["source"]) | set(dorothea["target"])

    abc_edges = load_abc_edges()
    abc_genes = set(abc_edges["source"]) | set(abc_edges["target"])

    genes = gene_universe(dataset_genes, dorothea_genes, abc_genes)
    print(f"Shared gene universe: {len(genes)} genes (DoRothEA {len(dorothea_genes)}, ABC {len(abc_genes)}, "
          f"union {len(dorothea_genes | abc_genes)}, intersected with {len(dataset_genes)}-gene dataset panel)")

    obs = load_obs()
    cts = eligible_cell_types(obs)
    print(f"Eligible cell types: {cts}")

    meta, expr, genes_out = build_pseudobulk_streaming(genes=genes, cell_types=cts, h5ad_path=RAW_H5AD_PATH)

    n_donors = meta["donor_id"].nunique()
    n_profiles = len(meta)
    print(f"Pseudobulk: {n_profiles} profiles, {n_donors} donors, {len(genes_out)} genes")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.save(DATA_DIR / "pseudobulk_expr.npy", expr)
    meta.to_parquet(DATA_DIR / "pseudobulk_meta.parquet")
    pd.DataFrame({"gene_symbol": genes_out}).to_csv(DATA_DIR / "pseudobulk_genes.csv", index=False)

    summary = [
        "# Pseudobulk construction summary\n",
        f"Gene universe: {len(genes)} genes (union of DoRothEA + ABC genes, intersected with dataset panel)",
        f"Cell types: {len(cts)} -- {cts}",
        f"Profiles: {n_profiles}",
        f"Donors: {n_donors}",
        f"\nProfiles per cell type:\n{meta['cell_type'].value_counts().to_string()}",
        f"\nProfiles per disease group:\n{meta['disease'].value_counts().to_string()}",
        f"\nCells per profile: mean={meta['n_cells'].mean():.1f} min={meta['n_cells'].min()} max={meta['n_cells'].max()}",
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "pseudobulk_summary.md").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
