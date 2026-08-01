# PRISM: findings memo

## Research question

Do public biological priors placed over a gene panel before any learning
happens improve pseudobulk expression embeddings' ability to predict SLE
(systemic lupus erythematosus) disease status, relative to a matched
non-graph baseline and to structurally matched random controls of each
graph? Two priors are tested independently through one shared harness: a
TF-target regulatory graph (DoRothEA) and an epigenomic co-accessibility/
regulatory-proximity graph (the ABC model, restricted to PBMC-relevant
biosamples). They are not combined.

## Dataset and target

CELLxGENE Census contains exactly one SLE cohort (confirmed by querying
every SLE-labeled cell in Census, not just this dataset's own metadata):
`218acb0f-9f2f-4f76-b90b-15a4b7c7f629` (Perez et al. 2022, *Science*), 261
donors (162 SLE / 99 normal), a single assay/tissue/suspension type.

The original scope called for a 3-tier SLEDAI-based activity target. This
dataset carries no SLEDAI or other composite activity score in its obs,
uns, or any bundled supplementary table. Its closest field, `disease_state`
(flare/managed/treated/na), is a per-cell categorical that is severely
imbalanced (8 donors purely "flare") and, for 11 donors, inconsistent
across cells from the same `donor_id` -- a per-visit label, not a stable
per-donor one. Neither a clinical-cutpoint 3-tier split nor a data-driven
tertile split is constructible from it. Per the validation protocol, this
is flagged explicitly rather than silently downgraded: **the primary
target is SLE-vs-normal disease status**. See
`results/tables/step0_validation.md` for the full check.

Two confounds were checked and are worth carrying forward as caveats
rather than corrections:

- **Processing cohort.** One of four processing cohorts contains zero SLE
  donors -- a perfect batch/disease confound for that subset. Flagged, not
  corrected (the same posture TRACE took toward its RA dataset's residual
  age-group skew, rather than attempting an ad hoc correction that could
  introduce its own bias).
- **Treatment status.** `treated` is one of the same four mutually
  exclusive `disease_state` categories as `flare`/`managed`, not an
  independent flag, so it can't be crossed against activity level. For the
  binary disease-vs-normal target used here this isn't a live confound,
  but it means this dataset cannot separate treatment burden from activity
  state at all -- relevant if a future phase revisits a graded target.

## Harness correctness check: TF-target vs. TRACE

Before trusting the epigenomic result, the TF-target prior was run through
this harness first and checked against TRACE's original finding shape:
*the real graph should roughly match or lose to a degree-preserving random
rewiring* -- divergence would mean a harness bug, not a new finding.

| comparison | mean AUROC (real vs. other) | Wilcoxon p | verdict |
|---|---|---|---|
| real vs. baseline_pca | 0.959 vs. 0.984 | 3.2e-6 | real loses |
| real vs. C1 (degree-preserving) | 0.959 vs. 0.963 | 0.428 | no difference |
| real vs. C2 (fully random) | 0.959 vs. 0.929 | 1.3e-7 | real wins |
| real vs. C3 (sign-flipped) | 0.959 vs. 0.959 | 0.777 | no difference |

This matches TRACE's finding shape exactly: real ties its degree-preserving
control, loses to the unconstrained baseline, and beats full randomization
(so *some* topology matters, just not the specific TF-target edges over a
degree-matched null). The harness is trusted as a correctness check passed,
not re-litigated further.

## Epigenomic prior result

| comparison | mean AUROC (real vs. other) | Wilcoxon p | verdict |
|---|---|---|---|
| real vs. baseline_pca | 0.927 vs. 0.984 | 2e-6 | real loses |
| real vs. C1 (degree-preserving) | 0.927 vs. 0.937 | 0.013 | **real loses** |
| real vs. C2 (fully random) | 0.927 vs. 0.933 | 0.280 | no difference |

The epigenomic prior does not help. Unlike TF-target, it loses even to its
own degree-preserving rewiring, and is statistically indistinguishable
from a fully-random graph of the same size. Whatever regulatory-proximity
signal the ABC model encodes at this resolution and biosample coverage,
this harness finds no evidence it structures the embedding usefully for
this target -- topology alone (via C1) does about as much as the real
edges do, and both underperform a plain PCA baseline.

One result from the sanity probes bears on *why*: the cell-type-identity
kNN anchor (an easy structure any reasonable embedding should preserve)
scores 0.98 for baseline_pca and 0.92 for TF-target's real embedding, but
only **0.58** for epigenomic's real embedding (`results/tables/main_results.csv`,
`celltype_sanity_check` target). The epigenomic masked encoder's embedding
geometry is markedly worse at separating cell types than either baseline
or the other prior's encoder -- plausibly the same underlying weakness
driving the disease-status shortfall, not two unrelated findings. This
isn't explained further here; it's flagged as the most concrete lead for
why this specific prior underperforms, not just that it does.

## Sanity and leakage probes

- **Sex positive control** (should score well above chance): AUROC
  0.75-0.89 across conditions -- passes. Embeddings do carry a real,
  recoverable biological signal (consistent with e.g. XIST-driven
  expression differences), so the harness isn't just fitting noise.
- **Library-size negative control** (should score low; high R² would mean
  the embedding is substantially just encoding sequencing depth): R² of
  0.29-0.56 across conditions. This is **not low** -- flagged honestly as a
  real limitation rather than waved away. Some of what the encoders (both
  PCA and masked) capture is likely confounded with technical depth,
  meaning the headline AUROCs may be partly inflated by this route rather
  than purely biological/disease signal. This applies to both priors and
  the baseline about equally, so it doesn't change the *relative* ranking
  used in the paired significance tests, but it does bound how much the
  absolute AUROC numbers should be trusted as "real biology only."
- **Label-permutation null**: for both priors, both `real` and
  `baseline_pca` land at the 100th percentile of a 100-permutation
  donor-level label-shuffle null (null AUROC range roughly 0.38-0.61 in
  all four cases). The fitted models are recovering genuine, non-random
  structure -- this doesn't bear on whether the *prior* helps (that's what
  the C1/C2/C3 comparisons test), only that the pipeline as a whole isn't
  an artifact of the CV protocol itself.

## Comparison to TRACE

TRACE tested one prior (TF-target) against one dataset (RA) and found the
real graph roughly tying its degree-preserving control while losing to an
unconstrained baseline. Run through PRISM's generalized harness, TF-target
reproduces that same shape on a different disease and a different, larger
dataset -- evidence the finding is about the TF-target prior's information
content (or lack of it, relative to a degree-matched random graph) rather
than an artifact of TRACE's specific dataset or implementation. The second
prior tested here, epigenomic, does not even manage TF-target's result:
it loses to its own degree-preserving control, a strictly weaker outcome.
Across both priors and two datasets, no version of "impose this
biologically-motivated graph structure on the encoder" has yet beaten a
plain PCA baseline on donor-level disease-status prediction, and only one
of the two priors even beats naive full randomization of its own edges.

## Limitations

- Single dataset, single disease, binary target only -- generalization to
  other conditions or to a graded activity target is untested.
- The library-size leakage probe (R² 0.29-0.56) means the absolute AUROC
  values likely overstate pure biological signal; the paired comparisons
  (which hold this confound roughly constant across conditions) are more
  trustworthy than any single condition's AUROC in isolation.
- The Processing_Cohort confound (one cohort with zero SLE donors) is
  unresolved; donor-level CV can't fully separate its effect from disease
  status for that subset of donors.
- The ABC model's biosample coverage was mapped onto this harness's five
  major cell types by best available match, not an exact correspondence
  (see `src/priors/epigenomic.py`'s `PBMC_BIOSAMPLES` mapping) -- some of
  the epigenomic prior's underperformance could reflect this
  approximation rather than the ABC model's information content itself.
- No cross-prior combination or ensembling was attempted (explicitly out
  of scope for this phase, see `README.md`).

## What a ligand-receptor phase would need

A ligand-receptor prior is qualitatively different from both priors tested
here: it's inherently cross-cell-type (a ligand expressed in one cell type
acting on a receptor in another), whereas this harness's masked-linear
encoder treats each donor x cell_type pseudobulk profile independently.
Extending this harness would require, at minimum: (1) an architecture that
takes a donor's full set of cell-type profiles as joint input rather than
scoring each cell-type profile independently, (2) a redefinition of what a
"hidden unit" is (a ligand-receptor pair spanning two specific cell types,
not a single gene's neighborhood), and (3) a new structural-control design
-- the current C1/C2/C3 controls assume the mask lives entirely within one
shared gene panel; a cross-cell-type mask would need controls that
preserve *which* cell-type pairs are linked, not just node degree within a
single panel. `cell_state`/`author_cell_type`, recorded but unused in this
phase (see `step0_validation.md`), would likely matter here as the
finer-grained annotation a ligand-receptor prior would want to key off of.
None of this is scaffolded into the current `Prior` interface; it is a
genuinely separate architecture, not a third instance of this phase's
harness.
