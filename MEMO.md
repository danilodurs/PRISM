# PRISM: findings memo

## Research question

Do public biological priors placed over a gene panel before any learning
happens improve pseudobulk expression embeddings' ability to predict
rheumatoid arthritis (RA) disease status, relative to a matched non-graph
baseline and to structurally matched random controls -- and does combining
two priors (a TF-target regulatory graph and an epigenomic
co-accessibility/regulatory-proximity graph) into one joint mask beat the
better of the two priors alone?

This generalizes an earlier project (TRACE, which tested TF-target alone
against this same RA dataset) two ways at once: a second prior
(epigenomic, ABC model) tested independently through the same harness, and
a combined TF-target + epigenomic condition with its own structural
controls. The harness (`src/`) is dataset-parameterized
(`src/datasets.py`) and prior-mode-parameterized
(`src/priors/combined.py`); an SLE cohort is also supported via
`--dataset sle` for anyone who wants it, but RA is this repo's primary
target.

## Dataset and target

CELLxGENE Census (version `2025-11-08`) contains exactly one dataset with
any RA-labeled cells -- confirmed by querying every RA-labeled cell in
Census: `d18736c3-6292-4379-919a-d6d973204c87` (Binvignat et al. 2024,
*JCI Insight*), 36 donors (18 RA / 18 healthy), single
assay/tissue/suspension type. This is the same dataset TRACE originally
used. No fallback was needed. Full check:
`results/tables/step0_validation.md`.

Confounds checked:

- **Batch/processing confound**: checked `batch` and `Lane` against
  disease -- reasonably balanced both ways (e.g. batch: 6/4/8 normal vs
  7/3/8 RA). No perfect confound found.
- **Development-stage (age) skew**: RA donors skew slightly older,
  matching TRACE's own original finding on this dataset. Flagged, not
  corrected.
- **Activity-score availability**: this dataset carries a real DAS28
  activity score (`MY_das28crp4`/`MY_das28esr4`), and it's **stable per
  donor** (checked directly: no donor has more than one distinct
  non-missing value). But it's structurally RA-only (undefined for the 18
  healthy donors) and missing for 2/18 RA donors, leaving only 16 usable
  donors -- too few for this harness's 6-fold donor-level CV. Used only as
  a **descriptive, non-CV, non-corrected probe** (see below), not a
  primary or secondary CV target.
- **Treatment-status**: methotrexate (`MY_MTX`) and biologic DMARD
  (`MY_bDMARD`) flags are independent of disease status and each other.
  MTX-treated donors trend toward lower DAS28 (mean 2.91 vs 3.70),
  consistent with expected treatment response, not an artifact.
- **Paired case-control design**: `pair_index_CW` reveals each RA donor is
  matched 1:1 with a healthy control donor. Not used to correct anything
  this phase (the CV protocol treats every donor independently), but noted
  for a future paired-analysis phase.
- **Gene-panel size**: this dataset's own raw panel (21,645 genes) is
  smaller than a typical comprehensive panel -- this study measured/
  retained fewer genes. This lowers the raw DoRothEA-TF coverage fraction
  (371/429 = 86.5%), which required loosening the validation gate's
  coverage-fraction threshold from a naive 0.95 to 0.80 (documented in
  `scripts/00_validate_dataset.py`) -- flagged explicitly, not silently
  adjusted. The requirement that actually matters for the harness (enough
  *well-connected* TFs after filtering) passes comfortably regardless: 268
  TFs with >=5 targets.

Verdict: proceed with RA-vs-normal disease status as the primary target,
matching TRACE's original choice.

## Combined-mask design: concatenated channels, not a merged edge union

`src/priors/combined.py` builds the joint mask by concatenating each
prior's standalone mask as its own block, rather than merging edges into a
shared per-gene hidden-unit space. Full reasoning (also in the module's
docstring):

- TF-target hidden units (TFs) and epigenomic hidden units (co-accessibility
  anchor genes) are different kinds of nodes. In this gene panel, 331
  genes are both a DoRothEA TF and an ABC anchor gene -- merging them into
  one shared hidden unit per gene would require inventing a sign-conflict
  rule (does a directed repressor edge and an undirected co-accessibility
  edge for the same gene pair average? does one win?) with no principled
  answer.
- Concatenation lets every joint structural control reuse `src/graph.py`'s
  existing per-prior functions (`degree_preserving_random`,
  `fully_randomized`, `sign_flipped`) completely unmodified -- call each
  once per source-prior block, concatenate. This is what makes "preserve
  each gene's degree *within each source prior*" fall out for free,
  instead of requiring new graph-theory code for a merged bipartite
  structure.
- **Sign handling**: each block keeps exactly the sign behavior it has
  standalone -- TF-target's block carries real +/- signs, epigenomic's
  block is uniformly sign=+1. No cross-block sign interaction, because a
  hidden unit only ever belongs to one block.
- **Overlapping edges** (the same gene pair present in both priors' edge
  lists) are kept in BOTH blocks, not merged: a gene's mask row gets a
  nonzero entry from both its `tf_target::*` and `epigenomic::*` hidden
  units, if both exist. Overlap is preserved via duplication across
  channels, with lossless provenance (hidden-unit names are prefixed by
  source prior).

This produced a joint mask with 630 hidden units (268 tf_target + 362
epigenomic). Epigenomic's `MIN_EDGES_PER_SOURCE=75` capacity-matching
override keeps both blocks' hidden-unit counts in the same order of
magnitude, avoiding a capacity imbalance that would confound "combining
helps" with "one block just gets far more embedding capacity."

**Structural controls for the joint mask**:
- **Joint C1** (degree-preserving): `degree_preserving_random` applied
  independently to each block, 5 seeds.
- **Joint C2** (fully random): `fully_randomized` applied independently to
  each block, 5 seeds.
- **Joint C3** (sign-flip): only the TF-target block's signs flipped
  (epigenomic has no sign to flip).
- **Exploratory ablation** (kept outside the corrected family):
  `only_tf_real` (real TF-target block + a C2-randomized epigenomic
  block) and `only_epi_real` (mirror), isolating which source prior
  drives any combined effect.

## Comparison family and Bonferroni correction

The combined condition's corrected family is **6 comparisons** (alpha =
0.05/6 = 0.00833): real vs baseline_pca, real vs joint C1 (pooled 5
seeds), real vs joint C2 (pooled 5 seeds), real vs joint C3, real vs
`tf_target_real` (the single-prior result), real vs `epigenomic_real`
(ditto) -- the last two directly test "does combining beat the better
single prior alone." The two exploratory ablation comparisons are
reported separately, uncorrected.

Each single-prior family (`tf_target`, `epigenomic` alone) is a
4-comparison / 3-comparison family (the fourth being TF-target's
sign-flip control, which epigenomic -- unsigned -- doesn't have), matching
TRACE's original correction shape.

## Results

Donor-level AUROC (RA vs normal), 30 seed x fold CV splits, paired
Wilcoxon signed-rank test. Full numbers: `results/tables/scorecard.csv`,
`results/tables/significance_tests.md`.

| prior | comparison | real/a | other/b | p (Wilcoxon) | significant | verdict |
|---|---|---|---|---|---|---|
| tf_target | real vs. baseline_pca | 0.530 | 0.485 | 0.229 | no | no difference |
| tf_target | real vs. C1 degree-preserving | 0.530 | 0.489 | 0.347 | no | no difference |
| tf_target | real vs. C2 fully random | 0.530 | 0.452 | 0.040 | no (alpha=0.0125) | no difference |
| tf_target | real vs. C3 sign-flipped | 0.530 | 0.530 | NaN (exact zero diff) | no | no difference |
| epigenomic | real vs. baseline_pca | 0.530 | 0.459 | 0.065 | no | no difference |
| epigenomic | real vs. C1 degree-preserving | 0.530 | 0.426 | 0.027 | no (alpha=0.0167) | no difference (Wilcoxon); ttest p=0.016 significant |
| epigenomic | real vs. C2 fully random | 0.530 | 0.488 | 0.368 | no | no difference |
| combined | real vs. baseline_pca | 0.515 | 0.448 | 0.101 | no | no difference |
| combined | real vs. joint C1 | 0.515 | 0.430 | 0.062 | no | no difference |
| combined | real vs. joint C2 | 0.515 | 0.481 | 0.304 | no | no difference |
| combined | real vs. joint C3 (TF sign-flip) | 0.515 | 0.515 | 0.655 (exact zero diff) | no | no difference |
| combined | real vs. tf_target_real (single) | 0.515 | 0.530 | 0.684 | no | combined does not beat tf_target alone |
| combined | real vs. epigenomic_real (single) | 0.515 | 0.530 | 0.552 | no | combined does not beat epigenomic alone |
| combined (exploratory, uncorrected) | real vs. only_tf_real | 0.515 | 0.502 | 0.399 | no | no difference |
| combined (exploratory, uncorrected) | real vs. only_epi_real | 0.515 | 0.437 | 0.077 | no | no difference (closest to significance of any exploratory comparison) |

**Headline finding: nothing beats anything here, in either direction.**
Every single comparison -- baseline vs. real, real vs. every structural
control, and combined vs. both single priors -- fails to reach
significance at n=36 donors.

**The exact-zero C3 difference is a genuine, explainable architectural
property, not a bug.** For both `tf_target` standalone and `combined`'s
joint C3, the real-vs-sign-flipped AUROC difference is *exactly* zero
across all 30 splits (not just statistically indistinguishable, but
bit-identical). Mechanism: the decoder is unconstrained (a plain
`torch.nn.Linear`, no sign/mask restriction) and `tanh` is an odd
function, so a global sign flip of the encoder's fixed sign buffer can be
exactly compensated by refitting the (unconstrained) magnitude and bias
parameters to produce a mirror-image embedding -- and a downstream linear
classifier (logistic regression) is invariant to that mirroring (it just
flips its own coefficient signs to match). This dataset's small, simple
optimization landscape (409 profiles) converges to the *exact* mirror
solution consistently; a larger, noisier dataset would likely show the
same effect only approximately (near-zero, not bit-exact).

## Critical caveat: this replication is severely underpowered

Before trusting any of the above, the label-permutation null tells a
sobering story:

- `tf_target_real` AUROC 0.530 is at the **52nd percentile** of its null
  (range 0.209-0.737); `tf_target_baseline_pca` 0.485 at the **37th**.
- `epigenomic_real` 0.530 at the **53rd percentile** (range 0.209-0.790);
  `epigenomic_baseline_pca` 0.459 at the **26th**.
- `combined_real` 0.515 at the **46th percentile** (range 0.207-0.770);
  `combined_baseline_pca` 0.448 at the **27th**.

**Every condition here is statistically indistinguishable from fitting
pure label-shuffle noise.** This isn't a new finding -- it's an exact
replication of TRACE's own original caveat on this same dataset ("at n=36
donors, a follow-up label-permutation null shows no single condition here
... is distinguishable from chance"). It means the entire results table
above should be read as "no evidence of a difference, in a regime where we
also can't confirm the pipeline is fitting real signal at all." Any
interpretation of the AUROC point estimates alone (e.g. "combined tied
real tf_target") is not well-supported without this caveat attached.

## DAS28 descriptive probe

Non-CV-protocol, non-corrected (see `src/data.py`,
`scripts/03_run_experiment.py::das28_probe_score` docstrings): out-of-fold
Ridge predictions pooled across all 30 splits, restricted to profiles from
the ~16-18 RA donors with a non-missing DAS28-CRP value. Full table:
`results/tables/das28_probe.csv`.

| condition | R² | Pearson r |
|---|---|---|
| tf_target_baseline_pca | -2.00 | -0.21 |
| tf_target_real | -1.95 | -0.09 |
| epigenomic_baseline_pca | -0.64 | -0.41 |
| epigenomic_real | -0.76 | -0.33 |
| combined_baseline_pca | -0.50 | -0.48 |
| combined_real | -1.68 | -0.14 |

No condition's embedding usefully predicts DAS28 activity level -- every
R² is negative (worse than predicting the mean) and correlations are weak.
Given this is pooled across profiles from only 16-18 distinct donors
(repeated across cell types and CV seeds), the effective sample size is
far smaller than the raw `n_pooled_predictions` (~1025) suggests -- this
probe should be read as "no signal detected, in a regime too small to
detect a small-to-moderate one," not as evidence of a true null.

## Sanity and leakage probes

- **Sex positive control**: baseline_pca conditions score 0.82-0.83 AUROC;
  real/combined conditions 0.65-0.70 -- weaker but still well above
  chance. Passes.
- **Library-size negative control**: baseline_pca conditions show R²
  0.58-0.66, but the masked-encoder conditions are wildly unstable:
  `tf_target_real` R² = -0.75, `combined_real` R² = -0.45,
  `epigenomic_real` R² = +0.23. These negative R² values mean the probe
  itself is too unstable at n=36 donors to draw a clean leakage conclusion
  for the masked-encoder conditions -- flagged as a real limitation of the
  probe at this sample size, not evidence of "no leakage."
- **Cell-type-identity anchor**: baseline_pca ~0.30-0.31 accuracy (much
  lower than a large-cohort baseline would score, reflecting far fewer
  profiles per cell type at n=36 donors). `tf_target_real` scores *higher*
  than baseline (0.419 vs 0.309). `epigenomic_real` scores lower than
  baseline (0.197 vs 0.309) -- epigenomic's embedding geometry is weaker
  at cell-type separation than either baseline or TF-target's.
- **Label-permutation null**: see the dedicated section above -- this is
  the probe that matters most here, and it does not pass cleanly.

## Comparison to TRACE

TRACE (n=36, this same dataset, TF-target only) found the real graph
losing to its degree-preserving control (a significant result at the
time) while also noting the whole comparison was underpowered per its own
permutation-null check. Run through this generalized harness, `tf_target`
no longer even reaches TRACE's original significant result (real vs C1:
TRACE p=0.0011, this run p=0.347) -- consistent with TRACE's own caveat
that its one significant finding was fragile, not a contradiction of
TRACE's substantive conclusion (real graph doesn't clearly beat a
degree-matched random rewiring on this dataset, in either implementation).
The honest summary: at n=36 donors, this harness's CV protocol is not
reliably powered for any comparison tested here, independent prior or
combined -- a limitation of the dataset/design, not evidence for or
against any specific prior.

## Limitations

- **Severe underpowering at n=36** (see the critical-caveat section) is
  the dominant limitation -- it bounds every other finding in this memo.
- **DAS28's usable subgroup (16-18 donors) is smaller still**, so its null
  result should be read as inconclusive, not as evidence against a
  DAS28-activity relationship.
- **The library-size leakage probe is itself unstable** at this sample
  size for the masked-encoder conditions (negative R²) -- no confident
  statement about library-size confounding is possible for the
  real/combined conditions.
- **The exact-zero C3 result**, while explained mechanistically above, is
  a reminder that this architecture's downstream linear probe absorbs
  sign information perfectly whenever the optimizer converges to the
  exact mirror solution -- something to watch for in any future dataset
  small/simple enough for optimization to be this deterministic.
- **Does the concatenated-channel combined-mask design generalize to a
  future ligand-receptor phase?** No, not directly. A ligand-receptor
  prior is inherently cross-cell-type (a ligand in one cell type acting on
  a receptor in another), while this design -- like both standalone
  priors it combines -- still treats each donor x cell_type pseudobulk
  profile independently. The concatenation trick answers a narrower
  question ("how do you combine two same-shape, same-architecture priors
  without inventing an edge-merge rule") that doesn't carry over to
  combining a within-cell-type prior with a cross-cell-type one -- that
  would need a genuinely cross-cell-type architecture built first, before
  a combination question is even well-posed. This combined-mask design
  should be read as a one-off answer for two structurally similar priors,
  not a scaffolded general combination mechanism.
- **Development-stage (age) skew** carried forward from TRACE, flagged
  not corrected.
- **Single dataset, single disease.** An SLE cohort is supported via
  `--dataset sle` for anyone who wants a second, much larger-n
  replication target, but is not this repo's focus.

## Reproducing

```bash
python scripts/00_validate_dataset.py
python scripts/01_build_pseudobulk.py
python scripts/02_build_graph.py
python scripts/03_run_experiment.py
python scripts/04_significance_test.py
python scripts/05_make_figures.py
```

(`--dataset ra --prior-mode combined` is the default for every script; pass
`--dataset sle --prior-mode independent` to reproduce the original SLE/
single-prior run instead.)

## Contributor

Danilo Dursoniah (ddursoniah@gmail.com)
