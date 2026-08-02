# Step 0 -- dataset validation (RA)

Dataset: CELLxGENE Census `d18736c3-6292-4379-919a-d6d973204c87` (Binvignat et al. 2024, JCI Insight)
Confirmed the only RA cohort in Census (queried every RA-labeled cell in Census `2025-11-08`, same approach used to confirm SLE's cohort) -- the same dataset TRACE originally used.
Total cells: 108717
Donors: 36

Disease balance:
disease
normal                  18
rheumatoid arthritis    18

Assay values (should be singleton): ["10x 3' v3"]
Suspension type values (should be singleton): ['cell']
Tissue values (should be singleton): ['blood']

Sex x disease:
sex                   female  male
disease                           
normal                    12     6
rheumatoid arthritis      12     6

Ethnicity x disease:
ethnicity             Asian  European American  Hispanic or Latin
disease                                                          
normal                    2                 15                  1
rheumatoid arthritis      2                 15                  1

Cells per donor by disease group:
                             mean          std   min   max
disease                                                   
normal                3337.777778  1800.685247  1339  9366
rheumatoid arthritis  2702.055556   718.837417  1851  4628

## Technical confound: batch x disease (donor-level)
batch                 1  2  3
disease                      
normal                6  4  8
rheumatoid arthritis  7  3  8

**Checked, no confound found**: unlike SLE's Processing_Cohort (one cohort with zero SLE donors, a perfect batch/disease confound), RA's `batch` field is reasonably balanced across disease groups here -- the same class of check was run, the answer is just different for this dataset.

## Development-stage (age) skew x disease (donor-level)
development_stage     60-79 year-old stage  adult stage  middle aged stage  young adult stage
disease                                                                                      
normal                                   4            1                  9                  4
rheumatoid arthritis                     5            6                  5                  2

**Flagged, not corrected**: RA donors skew slightly older than healthy donors here, consistent with TRACE's original finding on this same dataset (`TRACE/results/tables/step0_validation.md`). Carried forward as an unresolved limitation, not corrected, per this repo's established posture.

## Activity-score availability (target-choice finding)
DAS28-CRP present and, checked directly, **stable per donor** (no donor has >1 distinct non-missing value) -- unlike SLE's per-visit-inconsistent `disease_state`. But it is structurally RA-only (undefined for the 18 healthy donors) and missing for 2/18 RA donors, leaving only 16 donors with a usable value -- too few for this harness's 6-fold donor-level CV protocol (would need to hold out donors from an already-thin pool). **Primary target for this harness is therefore RA-vs-normal disease status**, matching TRACE's original choice and SLE's own precedent. DAS28 is used only as a descriptive, non-CV, non-corrected probe (see scripts/03_run_experiment.py), not a primary or secondary CV target -- flagged explicitly rather than silently building an underpowered graded-target pipeline.

## Treatment-status confound check
Unlike SLE (where `treated` was one mutually-exclusive category inside `disease_state`, impossible to cross against activity level), this dataset's methotrexate (`MY_MTX`) and biologic DMARD (`MY_bDMARD`) flags are independent of disease status and of each other -- they CAN be crossed against DAS28.

MTX x disease:
mtx                   0.0  1.0  nan
disease                            
normal                  0    0   18
rheumatoid arthritis   10    7    1

bDMARD x disease:
bdmard                0.0  1.0  nan
disease                            
normal                  0    0   18
rheumatoid arthritis   17    1    0

DAS28-CRP by MTX status (RA donors only):
         mean       std  count
mtx                           
0.0  3.703750  0.958793      8
1.0  2.907143  0.930873      7
nan  3.260000       NaN      1

MTX-treated donors trend toward lower DAS28 (consistent with expected treatment response, not an artifact) -- worth carrying forward as a covariate for any future graded-activity phase, not a confound for this phase's binary target.

## Paired case-control design (not present in SLE)
`pair_index_CW` reveals each RA donor is matched 1:1 with a healthy control donor. Not used to correct anything in this phase -- the donor-level CV protocol (src/evaluate.py::make_splits) treats every donor independently, as it already does for SLE -- but worth noting as a design feature a future paired-analysis phase could exploit.

## TF-target (DoRothEA) overlap
32286 edges, 429 TFs, 9228 targets
This dataset's own raw gene panel: 21645 genes
TFs found in dataset panel: 371/429
Targets found in dataset panel: 8331/9228
Edges with both endpoints in panel: 28163/32286
TFs with >=5 targets after filtering: 268

## Epigenomic (ABC model, 5 PBMC biosamples) overlap
470542 directed edges (symmetrized), 17368 unique genes touched
Genes found in dataset panel: 13704/17368
Edges with both endpoints in panel: 310932/470542
Genes with >=5 linked genes after filtering: 12762

## Cell types eligible for pseudobulk (>=10 cells in >=80% of donors): 13
  - CD8-positive, alpha-beta memory T cell
  - central memory CD4-positive, alpha-beta T cell
  - classical monocyte
  - effector memory CD4-positive, alpha-beta T cell
  - effector memory CD8-positive, alpha-beta T cell, terminally differentiated
  - gamma-delta T cell
  - memory B cell
  - myeloid dendritic cell
  - naive B cell
  - naive thymus-derived CD4-positive, alpha-beta T cell
  - naive thymus-derived CD8-positive, alpha-beta T cell
  - natural killer cell
  - plasmablast

## Verdict: PROCEED with RA-vs-normal disease status as the primary target. No batch confound found (checked, unlike SLE). Development-stage skew flagged, not corrected. DAS28 available but underpowered for CV -- descriptive probe only.