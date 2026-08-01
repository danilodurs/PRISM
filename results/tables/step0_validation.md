# Step 0 -- dataset validation

Dataset: CELLxGENE Census `218acb0f-9f2f-4f76-b90b-15a4b7c7f629` (Perez et al. 2022, Science)
Total cells: 1263676
Donors: 261

Disease balance:
disease
systemic lupus erythematosus    162
normal                           99

Assay values (should be singleton): ["10x 3' v2"]
Suspension type values (should be singleton): ['cell']
Tissue values (should be singleton): ['blood']

Sex x disease:
sex                           female  male
disease                                   
normal                            97     2
systemic lupus erythematosus     147    15

Ethnicity x disease:
ethnicity                     African American  Asian  European American  Hispanic or Latin
disease                                                                                    
normal                                       0     24                 74                  1
systemic lupus erythematosus                 3     83                 75                  1

Cells per donor by disease group:
                                     mean          std   min    max
disease                                                            
normal                        4913.313131  2610.320504   456  13543
systemic lupus erythematosus  4797.888889  1993.472338  1189  10253

## Technical confound: Processing_Cohort x disease (donor-level, majority cohort)
majority_cohort               1.0  2.0  3.0  4.0
disease                                         
normal                         33   16    6   44
systemic lupus erythematosus    0  103   19   40

**Flagged, not corrected**: Processing_Cohort 1 contains zero SLE donors -- a perfect batch/disease confound for that subset of donors. Cohorts 2-4 are reasonably mixed. This is reported here and in the memo as an unresolved limitation (consistent with how TRACE flagged the RA dataset's residual age-group skew rather than attempting a correction).

## Activity-score availability (target-choice finding)
disease_state                 MULTI_STATE  flare  managed  na
disease                                                      
normal                                  0      0        0  99
systemic lupus erythematosus           11      8      143   0

No SLEDAI or other composite disease-activity score is present in this dataset's metadata (checked obs, uns; no supplementary per-sample table is bundled in the deposited h5ad). The closest field, `disease_state`, is a per-cell categorical (flare/managed/treated/na) that is severely imbalanced among SLE donors and, for 11 donors, inconsistent within the same donor_id (different cells from the same person carry different states -- a per-visit label, not a stable per-donor one). Neither a clinical-cutpoint 3-tier split nor a data-driven tertile split is constructible from this: there is no continuous score to re-derive tertiles from, and the flare tier has too few donors (8, or 19 including MULTI_STATE donors) for any donor-level CV fold to be meaningful. **Primary target for this harness is therefore SLE-vs-normal disease status**, not a 3-tier activity score -- flagged explicitly per the validation protocol rather than silently downgrading the target.

## Treatment-status confound check
This dataset has no field independent of `disease_state` that records treatment/immunosuppressant status -- `treated` is one of the same four mutually-exclusive `disease_state` categories as `flare`/`managed`/`na`, not a separate flag that could be crossed against activity level. Since the primary target here is binary disease status (not activity tier), treatment status is not a confound for the disease-vs-normal comparison in the way it would be for a graded activity target -- but it remains true that this dataset cannot separate treatment burden from activity state at all, which would need to be resolved with different metadata (or a different dataset) before attempting a graded activity target in a future phase.

## Finer-grained cell-state annotations (recorded for a future phase, not used here)
`author_cell_type` (11 classes, coarser groupings than Census's own `cell_type`):
author_cell_type
T4        380477
cM        307429
T8        248927
B         151570
NK         92554
ncM        48800
cDC        18203
Prolif      8265
pDC         5233
PB          1411
Progen       807

`cell_state` (proliferation flag):
cell_state
na               1255411
proliferating       8265

## TF-target (DoRothEA) overlap
32286 edges, 429 TFs, 9228 targets
TFs found in dataset panel: 425/429
Targets found in dataset panel: 9116/9228
Edges with both endpoints in panel: 31968/32286
TFs with >=5 targets after filtering: 296

## Epigenomic (ABC model, 5 PBMC biosamples) overlap
470542 directed edges (symmetrized), 17368 unique genes touched
Genes found in dataset panel: 14699/17368
Edges with both endpoints in panel: 346794/470542
Genes with >=5 linked genes after filtering: 13788

## Cell types eligible for pseudobulk (>=10 cells in >=80% of donors): 8
  - B cell
  - CD4-positive, alpha-beta T cell
  - CD8-positive, alpha-beta T cell
  - classical monocyte
  - conventional dendritic cell
  - lymphocyte
  - natural killer cell
  - non-classical monocyte

## Verdict: PROCEED with SLE-vs-normal disease status as the primary target. Processing_Cohort confound flagged, not corrected. No 3-tier activity target available.