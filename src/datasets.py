"""Per-dataset configuration for the harness's two target cohorts: RA (this
branch's primary/default target) and SLE (`main`'s original target, kept
available here via explicit `--dataset sle` for anyone who wants it, but not
this branch's focus). `src/data.py`'s Census/pseudobulk functions take a
`DatasetConfig` (default `RA`) so both datasets share the identical code
path, differing only in dataset id, raw-obs field mapping, and output-file
prefix.

RA's dataset id (`d18736c3-6292-4379-919a-d6d973204c87`, Binvignat et al.
2024, JCI Insight) was confirmed the only RA cohort in CELLxGENE Census
`2025-11-08` by querying every RA-labeled cell in Census -- see
`results/tables/step0_validation.md`. It is also the same dataset TRACE
originally used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CENSUS_VERSION = "2025-11-08"


@dataclass
class DatasetConfig:
    key: str  # "ra", "sle" -- used as the output-file prefix (RA: none, this branch's canonical dataset)
    dataset_id: str
    census_version: str
    raw_h5ad_path: Path
    disease_positive_label: str  # value of the `disease` column naming the positive class
    disease_display_label: str  # short form for figure axis labels, e.g. "SLE vs normal"
    n_donors_expected: int
    extra_obs_columns: dict[str, str] = field(default_factory=dict)  # local name -> h5ad obs column
    donor_meta_columns: list[str] = field(
        default_factory=lambda: ["disease", "sex", "self_reported_ethnicity"]
    )  # local names (standard + extra) carried into the pseudobulk meta table

    @property
    def file_prefix(self) -> str:
        return "" if self.key == "ra" else f"{self.key}_"


SLE = DatasetConfig(
    key="sle",
    dataset_id="218acb0f-9f2f-4f76-b90b-15a4b7c7f629",
    census_version=CENSUS_VERSION,
    raw_h5ad_path=DATA_DIR / "raw_dataset.h5ad",
    disease_positive_label="systemic lupus erythematosus",
    disease_display_label="SLE vs normal",
    n_donors_expected=261,
    extra_obs_columns={
        "disease_state": "disease_state",
        "processing_cohort": "Processing_Cohort",
        "author_cell_type": "author_cell_type",
        "cell_state": "cell_state",
    },
    donor_meta_columns=["disease", "sex", "self_reported_ethnicity"],
)

RA = DatasetConfig(
    key="ra",
    dataset_id="d18736c3-6292-4379-919a-d6d973204c87",
    census_version=CENSUS_VERSION,
    raw_h5ad_path=DATA_DIR / "ra_raw_dataset.h5ad",
    disease_positive_label="rheumatoid arthritis",
    disease_display_label="RA vs normal",
    n_donors_expected=36,
    extra_obs_columns={
        # development_stage is deliberately NOT repeated here -- it's already a
        # standard Census obs field returned by load_obs(), and duplicating it
        # would create two same-named columns after the obs/extra concat.
        "das28_crp": "MY_das28crp4",
        "das28_esr": "MY_das28esr4",
        "mtx": "MY_MTX",
        "bdmard": "MY_bDMARD",
        "batch": "batch",
        "pair_index": "pair_index_CW",
    },
    donor_meta_columns=["disease", "sex", "self_reported_ethnicity", "das28_crp", "mtx", "bdmard", "batch"],
)

DATASETS: dict[str, DatasetConfig] = {"sle": SLE, "ra": RA}
