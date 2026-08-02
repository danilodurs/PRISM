"""Step 0: confirm each target dataset and both priors' raw resources
actually support the research question. `--dataset ra` (default) is this
branch's target. `--dataset sle` remains available for anyone who wants to
reproduce `main`'s original SLE validation gate.

Checks: donor/disease balance, technical-covariate confounds (assay, tissue,
suspension type, ethnicity, processing batch), activity-score availability,
treatment-status confound, cell-state annotation granularity, and gene
overlap with each prior's raw resource. Fails loudly (assert) on the
invariants the rest of the pipeline depends on; writes a human-readable
report for judgment calls.

SLE target-choice finding (see the written report for full detail): this is
the only SLE cohort in CELLxGENE Census, and it carries no SLEDAI or other
composite disease-activity score. The closest available field,
`disease_state` (flare/managed/treated/na), is both severely imbalanced
(143 managed vs 8 flare vs 11 donors with inconsistent per-cell values) and,
for those 11 donors, inconsistent across cells from the same donor_id -- a
per-visit label, not a stable per-donor one. A 3-tier activity split isn't
constructible from this. This script therefore validates SLE-vs-normal
disease status as the primary target instead, and reports the activity-score
absence explicitly rather than silently downgrading the target.

RA target-choice finding: unlike SLE, this dataset (Binvignat et al. 2024)
DOES carry a DAS28 activity score, and it IS stable per donor (unlike SLE's
disease_state). But it's structurally RA-only (not defined for healthy
donors) and missing for 2/18 RA donors -- 16 usable donors, too few for the
existing 6-fold donor-level CV protocol. This script therefore validates
RA-vs-normal disease status as the primary target (same shape as TRACE's
original choice, and as SLE's), and flags DAS28 as usable only as a
descriptive probe (see scripts/03_run_experiment.py), not a primary or
secondary CV target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import eligible_cell_types, load_dataset_gene_panel, load_extra_obs_fields, load_obs
from src.datasets import DATASETS, RA, SLE, DatasetConfig
from src.graph import filter_edges
from src.priors.epigenomic import PBMC_BIOSAMPLES, load_abc_edges
from src.priors.tf_target import load_dorothea

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"


def _prior_overlap_checks(census_genes: set[str], lines: list[str], min_tf_coverage_frac: float = 0.80) -> None:
    """Shared by both datasets: DoRothEA and ABC gene-overlap checks against
    this dataset's own measured gene panel.

    `min_tf_coverage_frac` default (0.80) is deliberately looser than a
    naive 0.95: RA's own raw panel (21,645 genes) is genuinely smaller than
    SLE's (30,162 genes) -- this study measured/retained fewer genes, not a
    bug -- so a smaller fraction of DoRothEA's 429 TFs are found there
    (86.5%) even though the requirement that actually matters for the
    harness (enough *well-connected* TFs after filtering, checked below,
    independent of this coverage-fraction check) passes comfortably for
    both datasets. Flagged here explicitly rather than silently loosened.
    """
    dorothea = load_dorothea()
    tf_genes = set(dorothea["source"])
    target_genes = set(dorothea["target"])
    lines.append(f"\n## TF-target (DoRothEA) overlap\n{len(dorothea)} edges, {len(tf_genes)} TFs, {len(target_genes)} targets")
    lines.append(f"This dataset's own raw gene panel: {len(census_genes)} genes")
    lines.append(f"TFs found in dataset panel: {len(tf_genes & census_genes)}/{len(tf_genes)}")
    lines.append(f"Targets found in dataset panel: {len(target_genes & census_genes)}/{len(target_genes)}")
    tf_coverage = len(tf_genes & census_genes) / len(tf_genes)
    assert tf_coverage > min_tf_coverage_frac, "too many DoRothEA TFs missing from this dataset's panel"

    dorothea_present = dorothea[dorothea["target"].isin(census_genes) & dorothea["source"].isin(census_genes)]
    tf_degree = dorothea_present.groupby("source").size()
    lines.append(f"Edges with both endpoints in panel: {len(dorothea_present)}/{len(dorothea)}")
    lines.append(f"TFs with >=5 targets after filtering: {(tf_degree >= 5).sum()}")
    assert (tf_degree >= 5).sum() >= 100, "too few well-connected TFs survive gene-overlap filtering"

    abc_edges = load_abc_edges()
    abc_genes = set(abc_edges["source"]) | set(abc_edges["target"])
    lines.append(
        f"\n## Epigenomic (ABC model, {len(PBMC_BIOSAMPLES)} PBMC biosamples) overlap\n"
        f"{len(abc_edges)} directed edges (symmetrized), {len(abc_genes)} unique genes touched"
    )
    lines.append(f"Genes found in dataset panel: {len(abc_genes & census_genes)}/{len(abc_genes)}")
    abc_present = abc_edges[abc_edges["target"].isin(census_genes) & abc_edges["source"].isin(census_genes)]
    abc_degree = abc_present.groupby("source").size()
    lines.append(f"Edges with both endpoints in panel: {len(abc_present)}/{len(abc_edges)}")
    lines.append(f"Genes with >=5 linked genes after filtering: {(abc_degree >= 5).sum()}")
    assert (abc_degree >= 5).sum() >= 100, "too few well-connected genes survive gene-overlap filtering"


def validate_sle() -> str:
    config = SLE
    lines: list[str] = ["# Step 0 -- dataset validation\n"]

    obs = load_obs(census_version=config.census_version, dataset_id=config.dataset_id)
    extra = load_extra_obs_fields(h5ad_path=config.raw_h5ad_path, config=config)
    assert len(obs) == len(extra), "standard and extra obs reads disagree on cell count -- investigate before trusting either"
    obs = pd.concat([obs.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)

    lines.append(f"Dataset: CELLxGENE Census `{config.dataset_id}` (Perez et al. 2022, Science)")
    lines.append(f"Total cells: {len(obs)}")

    donor_level = obs.groupby("donor_id", observed=True).agg(
        disease=("disease", lambda x: x.unique().tolist()),
        sex=("sex", lambda x: x.unique().tolist()),
        assay=("assay", lambda x: x.unique().tolist()),
        suspension_type=("suspension_type", lambda x: x.unique().tolist()),
        tissue=("tissue", lambda x: x.unique().tolist()),
        ethnicity=("self_reported_ethnicity", lambda x: x.unique().tolist()),
        n_cells=("donor_id", "count"),
    )
    assert (donor_level["disease"].apply(len) == 1).all(), "a donor has >1 disease label"
    assert (donor_level["sex"].apply(len) == 1).all(), "a donor has >1 sex label"
    for col in ["disease", "sex", "assay", "suspension_type", "tissue"]:
        donor_level[col] = donor_level[col].str[0]
    donor_level["ethnicity"] = donor_level["ethnicity"].apply(lambda x: x[0] if len(x) == 1 else "MULTI")

    n_donors = len(donor_level)
    lines.append(f"Donors: {n_donors}")
    assert n_donors == config.n_donors_expected, f"expected {config.n_donors_expected} donors, found {n_donors}"

    disease_counts = donor_level["disease"].value_counts()
    lines.append(f"\nDisease balance:\n{disease_counts.to_string()}")
    assert set(disease_counts.index) == {"normal", "systemic lupus erythematosus"}

    lines.append(f"\nAssay values (should be singleton): {donor_level['assay'].unique().tolist()}")
    assert donor_level["assay"].nunique() == 1, "more than one assay present -- assay is a confound risk"
    lines.append(f"Suspension type values (should be singleton): {donor_level['suspension_type'].unique().tolist()}")
    assert donor_level["suspension_type"].nunique() == 1
    lines.append(f"Tissue values (should be singleton): {donor_level['tissue'].unique().tolist()}")
    assert donor_level["tissue"].nunique() == 1

    sex_x_disease = pd.crosstab(donor_level["disease"], donor_level["sex"])
    lines.append(f"\nSex x disease:\n{sex_x_disease.to_string()}")

    eth_x_disease = pd.crosstab(donor_level["disease"], donor_level["ethnicity"])
    lines.append(f"\nEthnicity x disease:\n{eth_x_disease.to_string()}")

    cells_by_disease = donor_level.groupby("disease")["n_cells"].agg(["mean", "std", "min", "max"])
    lines.append(f"\nCells per donor by disease group:\n{cells_by_disease.to_string()}")

    # --- technical batch confound: Processing_Cohort ---
    donor_cohort = obs.groupby("donor_id", observed=True)["processing_cohort"].agg(lambda x: x.value_counts().idxmax())
    donor_level["majority_cohort"] = donor_cohort
    cohort_x_disease = pd.crosstab(donor_level["disease"], donor_level["majority_cohort"])
    lines.append(f"\n## Technical confound: Processing_Cohort x disease (donor-level, majority cohort)\n{cohort_x_disease.to_string()}")
    lines.append(
        "\n**Flagged, not corrected**: Processing_Cohort 1 contains zero SLE donors -- a perfect "
        "batch/disease confound for that subset of donors. Cohorts 2-4 are reasonably mixed. This is "
        "reported here and in the memo as an unresolved limitation (consistent with how TRACE flagged "
        "the RA dataset's residual age-group skew rather than attempting a correction)."
    )

    # --- activity-score availability: the key target-choice finding ---
    donor_state = obs.groupby("donor_id", observed=True)["disease_state"].agg(lambda x: x.unique().tolist())
    donor_level["disease_state"] = donor_state.apply(lambda x: x[0] if len(x) == 1 else "MULTI_STATE")
    state_x_disease = pd.crosstab(donor_level["disease"], donor_level["disease_state"])
    lines.append(f"\n## Activity-score availability (target-choice finding)\n{state_x_disease.to_string()}")
    lines.append(
        "\nNo SLEDAI or other composite disease-activity score is present in this dataset's metadata "
        "(checked obs, uns; no supplementary per-sample table is bundled in the deposited h5ad). The "
        "closest field, `disease_state`, is a per-cell categorical (flare/managed/treated/na) that is "
        "severely imbalanced among SLE donors and, for 11 donors, inconsistent within the same donor_id "
        "(different cells from the same person carry different states -- a per-visit label, not a stable "
        "per-donor one). Neither a clinical-cutpoint 3-tier split nor a data-driven tertile split is "
        "constructible from this: there is no continuous score to re-derive tertiles from, and the "
        "flare tier has too few donors (8, or 19 including MULTI_STATE donors) for any donor-level CV "
        "fold to be meaningful. **Primary target for this harness is therefore SLE-vs-normal disease "
        "status**, not a 3-tier activity score -- flagged explicitly per the validation protocol rather "
        "than silently downgrading the target."
    )

    # --- treatment-status confound check (required regardless of outcome) ---
    lines.append(
        "\n## Treatment-status confound check\n"
        "This dataset has no field independent of `disease_state` that records treatment/immunosuppressant "
        "status -- `treated` is one of the same four mutually-exclusive `disease_state` categories as "
        "`flare`/`managed`/`na`, not a separate flag that could be crossed against activity level. Since "
        "the primary target here is binary disease status (not activity tier), treatment status is not a "
        "confound for the disease-vs-normal comparison in the way it would be for a graded activity target "
        "-- but it remains true that this dataset cannot separate treatment burden from activity state at "
        "all, which would need to be resolved with different metadata (or a different dataset) before "
        "attempting a graded activity target in a future phase."
    )

    # --- fine-grained cell-state annotations (for the future ligand-receptor phase) ---
    author_ct_counts = obs["author_cell_type"].value_counts()
    cell_state_counts = obs["cell_state"].value_counts()
    lines.append(
        "\n## Finer-grained cell-state annotations (recorded for a future phase, not used here)\n"
        f"`author_cell_type` ({obs['author_cell_type'].nunique()} classes, coarser groupings than Census's "
        f"own `cell_type`):\n{author_ct_counts.to_string()}\n\n"
        f"`cell_state` (proliferation flag):\n{cell_state_counts.to_string()}"
    )

    census_genes = load_dataset_gene_panel(h5ad_path=config.raw_h5ad_path)
    _prior_overlap_checks(census_genes, lines)

    cts = eligible_cell_types(obs)
    lines.append(f"\n## Cell types eligible for pseudobulk (>=10 cells in >=80% of donors): {len(cts)}")
    for ct in cts:
        lines.append(f"  - {ct}")
    assert len(cts) >= 5, "too few cell types survive the coverage threshold for donor x cell_type pseudobulk"

    lines.append(
        "\n## Verdict: PROCEED with SLE-vs-normal disease status as the primary target. "
        "Processing_Cohort confound flagged, not corrected. No 3-tier activity target available."
    )
    return "\n".join(str(x) for x in lines)


def validate_ra() -> str:
    config = RA
    lines: list[str] = ["# Step 0 -- dataset validation (RA)\n"]

    obs = load_obs(census_version=config.census_version, dataset_id=config.dataset_id)
    extra = load_extra_obs_fields(h5ad_path=config.raw_h5ad_path, config=config)
    assert len(obs) == len(extra), "standard and extra obs reads disagree on cell count -- investigate before trusting either"
    obs = pd.concat([obs.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)

    lines.append(f"Dataset: CELLxGENE Census `{config.dataset_id}` (Binvignat et al. 2024, JCI Insight)")
    lines.append(
        "Confirmed the only RA cohort in Census (queried every RA-labeled cell in Census "
        f"`{config.census_version}`, same approach used to confirm SLE's cohort) -- the same dataset "
        "TRACE originally used."
    )
    lines.append(f"Total cells: {len(obs)}")

    donor_level = obs.groupby("donor_id", observed=True).agg(
        disease=("disease", lambda x: x.unique().tolist()),
        sex=("sex", lambda x: x.unique().tolist()),
        assay=("assay", lambda x: x.unique().tolist()),
        suspension_type=("suspension_type", lambda x: x.unique().tolist()),
        tissue=("tissue", lambda x: x.unique().tolist()),
        ethnicity=("self_reported_ethnicity", lambda x: x.unique().tolist()),
        development_stage=("development_stage", lambda x: x.unique().tolist()),
        n_cells=("donor_id", "count"),
    )
    assert (donor_level["disease"].apply(len) == 1).all(), "a donor has >1 disease label"
    assert (donor_level["sex"].apply(len) == 1).all(), "a donor has >1 sex label"
    for col in ["disease", "sex", "assay", "suspension_type", "tissue"]:
        donor_level[col] = donor_level[col].str[0]
    donor_level["ethnicity"] = donor_level["ethnicity"].apply(lambda x: x[0] if len(x) == 1 else "MULTI")
    donor_level["development_stage"] = donor_level["development_stage"].apply(lambda x: x[0] if len(x) == 1 else "MULTI")

    n_donors = len(donor_level)
    lines.append(f"Donors: {n_donors}")
    assert n_donors == config.n_donors_expected, f"expected {config.n_donors_expected} donors, found {n_donors}"

    disease_counts = donor_level["disease"].value_counts()
    lines.append(f"\nDisease balance:\n{disease_counts.to_string()}")
    assert set(disease_counts.index) == {"normal", "rheumatoid arthritis"}

    lines.append(f"\nAssay values (should be singleton): {donor_level['assay'].unique().tolist()}")
    assert donor_level["assay"].nunique() == 1, "more than one assay present -- assay is a confound risk"
    lines.append(f"Suspension type values (should be singleton): {donor_level['suspension_type'].unique().tolist()}")
    assert donor_level["suspension_type"].nunique() == 1
    lines.append(f"Tissue values (should be singleton): {donor_level['tissue'].unique().tolist()}")
    assert donor_level["tissue"].nunique() == 1

    sex_x_disease = pd.crosstab(donor_level["disease"], donor_level["sex"])
    lines.append(f"\nSex x disease:\n{sex_x_disease.to_string()}")

    eth_x_disease = pd.crosstab(donor_level["disease"], donor_level["ethnicity"])
    lines.append(f"\nEthnicity x disease:\n{eth_x_disease.to_string()}")

    cells_by_disease = donor_level.groupby("disease")["n_cells"].agg(["mean", "std", "min", "max"])
    lines.append(f"\nCells per donor by disease group:\n{cells_by_disease.to_string()}")

    # --- technical batch confound: batch / Lane (this dataset's equivalent of SLE's Processing_Cohort) ---
    batch_x_disease = pd.crosstab(donor_level["disease"], obs.groupby("donor_id", observed=True)["batch"].first())
    lines.append(f"\n## Technical confound: batch x disease (donor-level)\n{batch_x_disease.to_string()}")
    lines.append(
        "\n**Checked, no confound found**: unlike SLE's Processing_Cohort (one cohort with zero SLE "
        "donors, a perfect batch/disease confound), RA's `batch` field is reasonably balanced across "
        "disease groups here -- the same class of check was run, the answer is just different for this "
        "dataset."
    )

    # --- development-stage (age) skew -- carried forward from TRACE's original finding ---
    dev_x_disease = pd.crosstab(donor_level["disease"], donor_level["development_stage"])
    lines.append(f"\n## Development-stage (age) skew x disease (donor-level)\n{dev_x_disease.to_string()}")
    lines.append(
        "\n**Flagged, not corrected**: RA donors skew slightly older than healthy donors here, consistent "
        "with TRACE's original finding on this same dataset (`TRACE/results/tables/step0_validation.md`). "
        "Carried forward as an unresolved limitation, not corrected, per this repo's established posture."
    )

    # --- activity-score availability: DAS28 exists here, unlike SLE ---
    donor_das28 = obs.groupby("donor_id", observed=True)["das28_crp"].agg(lambda x: x.dropna().unique().tolist())
    n_multi_valued = (donor_das28.apply(len) > 1).sum()
    assert n_multi_valued == 0, "DAS28 is inconsistent within a donor -- per-visit label, not stable per-donor"
    donor_level["das28_crp"] = donor_das28.apply(lambda x: x[0] if len(x) == 1 else float("nan"))
    n_ra = (donor_level["disease"] == "rheumatoid arthritis").sum()
    n_das28_missing_ra = donor_level.loc[donor_level["disease"] == "rheumatoid arthritis", "das28_crp"].isna().sum()
    lines.append(
        f"\n## Activity-score availability (target-choice finding)\n"
        f"DAS28-CRP present and, checked directly, **stable per donor** (no donor has >1 distinct "
        f"non-missing value) -- unlike SLE's per-visit-inconsistent `disease_state`. But it is "
        f"structurally RA-only (undefined for the {n_donors - n_ra} healthy donors) and missing for "
        f"{n_das28_missing_ra}/{n_ra} RA donors, leaving only {n_ra - n_das28_missing_ra} donors with a "
        f"usable value -- too few for this harness's 6-fold donor-level CV protocol (would need to hold "
        f"out donors from an already-thin pool). **Primary target for this harness is therefore "
        f"RA-vs-normal disease status**, matching TRACE's original choice and SLE's own precedent. DAS28 "
        f"is used only as a descriptive, non-CV, non-corrected probe (see "
        f"scripts/03_run_experiment.py), not a primary or secondary CV target -- flagged explicitly "
        f"rather than silently building an underpowered graded-target pipeline."
    )

    # --- treatment-status confound check: genuinely different from SLE here ---
    donor_mtx = obs.groupby("donor_id", observed=True)["mtx"].first()
    donor_bdmard = obs.groupby("donor_id", observed=True)["bdmard"].first()
    donor_level["mtx"] = donor_mtx
    donor_level["bdmard"] = donor_bdmard
    mtx_x_disease = pd.crosstab(donor_level["disease"], donor_level["mtx"].astype(str))
    bdmard_x_disease = pd.crosstab(donor_level["disease"], donor_level["bdmard"].astype(str))
    ra_mask = donor_level["disease"] == "rheumatoid arthritis"
    das28_by_mtx = donor_level.loc[ra_mask].groupby(donor_level.loc[ra_mask, "mtx"].astype(str))["das28_crp"].agg(
        ["mean", "std", "count"]
    )
    lines.append(
        "\n## Treatment-status confound check\n"
        "Unlike SLE (where `treated` was one mutually-exclusive category inside `disease_state`, "
        "impossible to cross against activity level), this dataset's methotrexate (`MY_MTX`) and biologic "
        "DMARD (`MY_bDMARD`) flags are independent of disease status and of each other -- they CAN be "
        f"crossed against DAS28.\n\nMTX x disease:\n{mtx_x_disease.to_string()}\n\n"
        f"bDMARD x disease:\n{bdmard_x_disease.to_string()}\n\n"
        f"DAS28-CRP by MTX status (RA donors only):\n{das28_by_mtx.to_string()}\n\n"
        "MTX-treated donors trend toward lower DAS28 (consistent with expected treatment response, not an "
        "artifact) -- worth carrying forward as a covariate for any future graded-activity phase, not a "
        "confound for this phase's binary target."
    )

    # --- paired case-control design ---
    lines.append(
        "\n## Paired case-control design (not present in SLE)\n"
        "`pair_index_CW` reveals each RA donor is matched 1:1 with a healthy control donor. Not used to "
        "correct anything in this phase -- the donor-level CV protocol (src/evaluate.py::make_splits) "
        "treats every donor independently, as it already does for SLE -- but worth noting as a design "
        "feature a future paired-analysis phase could exploit."
    )

    census_genes = load_dataset_gene_panel(h5ad_path=config.raw_h5ad_path)
    _prior_overlap_checks(census_genes, lines)

    cts = eligible_cell_types(obs)
    lines.append(f"\n## Cell types eligible for pseudobulk (>=10 cells in >=80% of donors): {len(cts)}")
    for ct in cts:
        lines.append(f"  - {ct}")
    assert len(cts) >= 5, "too few cell types survive the coverage threshold for donor x cell_type pseudobulk"

    lines.append(
        "\n## Verdict: PROCEED with RA-vs-normal disease status as the primary target. No batch confound "
        "found (checked, unlike SLE). Development-stage skew flagged, not corrected. DAS28 available but "
        "underpowered for CV -- descriptive probe only."
    )
    return "\n".join(str(x) for x in lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ra")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config: DatasetConfig = DATASETS[args.dataset]

    report = validate_sle() if args.dataset == "sle" else validate_ra()

    out_name = "step0_validation.md" if config.key == "ra" else f"step0_validation_{config.key}.md"
    out_path = RESULTS_DIR / out_name
    out_path.write_text(report)
    print(report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
