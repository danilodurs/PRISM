# PRISM -- Prior-informed Representation and Structural Modeling

A reusable harness for testing whether public biological priors -- graphs
placed over a gene panel before any learning happens -- improve pseudobulk
expression embeddings' ability to predict a clinical target, relative to a
matched non-graph baseline and to structurally matched random controls of
the same graph. This generalizes an earlier project (TRACE), which tested a
single prior (a TF-target regulatory graph) against a single dataset. PRISM
builds the shared evaluation machinery once and tests two priors through it
independently: TF-target (DoRothEA) and epigenomic (ABC model enhancer-gene
predictions, restricted to PBMC-relevant biosamples).

**Research question**: do a TF-target regulatory graph and an epigenomic
co-accessibility/regulatory-proximity graph each improve pseudobulk
expression embeddings' ability to predict SLE (systemic lupus
erythematosus) disease status, relative to a matched non-graph baseline and
to structurally matched random controls of each graph? Each prior is tested
independently through the identical harness; they are not combined.

See `MEMO.md` for the full writeup: dataset validation and target-choice
reasoning, per-prior results against baseline/controls/permutation null,
comparison to TRACE's original TF-target finding, limitations, and what a
future ligand-receptor phase would need.

## Results

Donor-level AUROC (SLE vs. normal), 30 seed x fold CV splits, paired
Wilcoxon signed-rank test, Bonferroni-corrected within each prior's
comparison family (4 comparisons for TF-target including its sign-flip
control, 3 for epigenomic). Full numbers: `results/tables/scorecard.csv`
and `results/tables/significance_tests.md`.

| prior | comparison | real | other | p (Wilcoxon) | significant | verdict |
|---|---|---|---|---|---|---|
| tf_target | real vs. baseline_pca | 0.959 | 0.984 | 3.2e-6 | yes | real loses |
| tf_target | real vs. C1 degree-preserving | 0.959 | 0.963 | 0.428 | no | no difference |
| tf_target | real vs. C2 fully random | 0.959 | 0.929 | 1.3e-7 | yes | real wins |
| tf_target | real vs. C3 sign-flipped | 0.959 | 0.959 | 0.777 | no | no difference |
| epigenomic | real vs. baseline_pca | 0.927 | 0.984 | 2.0e-6 | yes | real loses |
| epigenomic | real vs. C1 degree-preserving | 0.927 | 0.937 | 0.013 | yes | **real loses** |
| epigenomic | real vs. C2 fully random | 0.927 | 0.933 | 0.280 | no | no difference |

**TF-target** reproduces TRACE's original finding shape exactly: the real
graph ties its degree-preserving control and loses to an unconstrained
PCA baseline -- the harness's correctness check passes. **Epigenomic**
does not even manage that: it loses to its own degree-preserving rewiring
and is statistically indistinguishable from a fully-random graph of the
same size. Neither prior beats plain PCA on this target; see `MEMO.md`
for the sanity-probe finding (a much weaker cell-type-identity signal in
epigenomic's embedding) that's the most concrete lead for why.

## The harness

Prior-specific code is limited to exactly two things, per prior:
fetching/filtering that prior's raw resource to a shared gene universe, and
declaring whether it's signed (enables a sign-flip control) or unsigned.
Everything downstream -- encoder architecture, structural controls, CV
protocol, significance testing, permutation null -- is shared code,
parameterized by which prior is active:

```
src/
  data.py          # Census access, this SLE dataset's pseudobulk construction
  graph.py         # mask builder + structural controls (prior-agnostic)
  encoders.py      # PCAEncoder (baseline), PriorEncoder (hard-sign-constrained masked linear autoencoder)
  evaluate.py       # donor-level CV, classification/multiclass/continuous probes
  significance.py  # paired significance tests, label-permutation null
  priors/
    base.py         # the Prior interface: name, signed, edges
    tf_target.py    # DoRothEA -- signed=True
    epigenomic.py   # ABC model, PBMC biosamples -- signed=False
```

As a correctness check on the harness itself, the first prior run through it
is TF-target (DoRothEA), and its result is compared against TRACE's original
finding shape (real graph roughly matching or losing to a degree-preserving
random rewiring) before the epigenomic prior is built on top of the same
machinery.

## Pipeline

```
scripts/download_raw_h5ad.py           # cache the SLE dataset's source h5ad locally
scripts/download_abc_predictions.py    # cache the genome-wide ABC predictions locally
scripts/00_validate_dataset.py         # dataset validation gate (target choice, confounds)
scripts/01_build_pseudobulk.py         # donor x cell_type pseudobulk over the shared gene universe
scripts/02_build_graph.py              # both priors' real graphs + structural controls
scripts/03_run_experiment.py           # harness evaluation: both priors x all conditions x all probes
scripts/04_significance_test.py        # paired tests, permutation null, scorecard
scripts/05_make_figures.py             # figures referenced in MEMO.md
```

Each script writes its outputs under `data/` (gitignored, regenerated from
Census/ABC on each run) or `results/` (tracked).

## Setup

```bash
conda env create -f environment.yml
conda activate prism
# or: pip install -r requirements.txt
```

## Reproducing

```bash
python scripts/download_raw_h5ad.py
python scripts/download_abc_predictions.py
python scripts/00_validate_dataset.py
python scripts/01_build_pseudobulk.py
python scripts/02_build_graph.py
python scripts/03_run_experiment.py
python scripts/04_significance_test.py
python scripts/05_make_figures.py
```

Random seeds are fixed throughout (5 CV seeds x 6 folds; 5 seeds each for the
degree-preserving and fully-random controls).

## Dataset and target

CELLxGENE Census dataset `218acb0f-9f2f-4f76-b90b-15a4b7c7f629` (Perez et
al. 2022, *Science*) -- the only SLE cohort in Census. 261 donors (162 SLE /
99 normal), single assay/tissue/suspension type. This dataset carries no
SLEDAI or other composite disease-activity score, so the primary target here
is **SLE-vs-normal disease status**, not the originally-scoped 3-tier
activity target -- see `MEMO.md` and `results/tables/step0_validation.md`
for the full reasoning, including a technical confound (one processing
cohort contains zero SLE donors) that's flagged but not corrected.

## Explicitly out of scope (this phase)

- **Ligand-receptor prior.** Requires a genuinely different, cross-cell-type
  architecture (a ligand in one cell type acting on a receptor in another),
  not a straightforward third instance of this harness's single-cell-type
  gene x gene masked-linear design. Not built, and not scaffolded for beyond
  the extension points already implicit in the `Prior` interface.
- **Any cross-cell-type architecture.** This harness's encoder treats each
  donor x cell_type pseudobulk profile independently; nothing here models
  interactions between cell types.
- **Prior combination or ensembling.** TF-target and epigenomic are each
  tested independently against their own controls; they are never combined
  into a joint model in this phase.

All three are deferred to a later phase, once this two-prior harness has
been validated as the reusable foundation.

## Contributor

Danilo Dursoniah (ddursoniah@gmail.com)
