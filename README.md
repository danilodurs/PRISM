# PRISM -- Prior-informed Representation and Structural Modeling

A reusable harness for testing whether public biological priors -- graphs
placed over a gene panel before any learning happens -- improve pseudobulk
expression embeddings' ability to predict a clinical target, relative to a
matched non-graph baseline and to structurally matched random controls of
the same graph. This generalizes an earlier project (TRACE, which tested a
single prior -- a TF-target regulatory graph -- against this same RA
dataset). PRISM builds the shared evaluation machinery once and tests two
priors through it -- TF-target (DoRothEA) and epigenomic (ABC model
enhancer-gene predictions, restricted to PBMC-relevant biosamples) -- both
independently and combined into one joint mask.

**Research question**: do a TF-target regulatory graph and an epigenomic
co-accessibility/regulatory-proximity graph each improve pseudobulk
expression embeddings' ability to predict RA (rheumatoid arthritis) disease
status, relative to a matched non-graph baseline and to structurally
matched random controls of each graph -- and does combining both priors
into a single joint mask beat the better of the two alone?

See `MEMO.md` for the full writeup: dataset validation and confound
reasoning, the combined-mask design decision and why, results against
baseline/controls/permutation null/both single priors, comparison to
TRACE's original TF-target finding, sanity probes, and limitations.

## Results

Donor-level AUROC (RA vs. normal), 30 seed x fold CV splits, paired
Wilcoxon signed-rank test. Full numbers: `results/tables/scorecard.csv` and
`results/tables/significance_tests.md`.

| prior | comparison | real | other | p (Wilcoxon) | significant | verdict |
|---|---|---|---|---|---|---|
| tf_target | real vs. baseline_pca | 0.530 | 0.485 | 0.229 | no | no difference |
| tf_target | real vs. C1 degree-preserving | 0.530 | 0.489 | 0.347 | no | no difference |
| tf_target | real vs. C2 fully random | 0.530 | 0.452 | 0.040 | no | no difference |
| tf_target | real vs. C3 sign-flipped | 0.530 | 0.530 | exact zero diff | no | no difference |
| epigenomic | real vs. baseline_pca | 0.530 | 0.459 | 0.065 | no | no difference |
| epigenomic | real vs. C1 degree-preserving | 0.530 | 0.426 | 0.027 | no (Wilcoxon) | no difference |
| epigenomic | real vs. C2 fully random | 0.530 | 0.488 | 0.368 | no | no difference |
| combined | real vs. baseline_pca | 0.515 | 0.448 | 0.101 | no | no difference |
| combined | real vs. joint C1 | 0.515 | 0.430 | 0.062 | no | no difference |
| combined | real vs. joint C2 | 0.515 | 0.481 | 0.304 | no | no difference |
| combined | real vs. joint C3 (TF sign-flip) | 0.515 | 0.515 | exact zero diff | no | no difference |
| combined | real vs. tf_target_real (single prior) | 0.515 | 0.530 | 0.684 | no | combining doesn't beat tf_target alone |
| combined | real vs. epigenomic_real (single prior) | 0.515 | 0.530 | 0.552 | no | combining doesn't beat epigenomic alone |

**Nothing beats anything here, in either direction** -- but the critical
caveat is that at n=36 donors, every condition (real, baseline, and every
control) lands within the 26th-53rd percentile of its own
label-permutation null: statistically indistinguishable from fitting pure
noise. This exactly replicates TRACE's own original power caveat on this
same dataset. See `MEMO.md` for the full picture, including why the
combined mask's sign-flip control produces an *exact* zero AUROC
difference (a genuine, explained architectural property, not a bug).

## The harness

Prior-specific code is limited to exactly two things, per prior:
fetching/filtering that prior's raw resource to a shared gene universe, and
declaring whether it's signed (enables a sign-flip control) or unsigned.
Everything downstream -- encoder architecture, structural controls, CV
protocol, significance testing, permutation null -- is shared code,
parameterized by which prior/dataset is active:

```
src/
  datasets.py      # per-dataset config (RA: default; SLE: available via --dataset sle)
  data.py          # Census access, dataset-parameterized pseudobulk construction
  graph.py         # mask builder + structural controls (prior-agnostic)
  encoders.py      # PCAEncoder (baseline), PriorEncoder (hard-sign-constrained masked linear autoencoder)
  evaluate.py       # donor-level CV, classification/multiclass/continuous probes
  significance.py  # paired significance tests, label-permutation null
  priors/
    base.py         # the Prior interface: name, signed, edges
    tf_target.py    # DoRothEA -- signed=True
    epigenomic.py   # ABC model, PBMC biosamples -- signed=False
    combined.py     # joint TF-target + epigenomic mask (concatenated channels) + joint structural controls
```

As a correctness check on the harness itself, TF-target's independent
result is compared against TRACE's original finding shape (real graph
roughly matching or losing to a degree-preserving random rewiring) before
trusting the epigenomic and combined results built on the same machinery.

## Pipeline

```
scripts/download_raw_h5ad.py           # cache a dataset's source h5ad locally (--dataset ra default)
scripts/download_abc_predictions.py    # cache the genome-wide ABC predictions locally
scripts/00_validate_dataset.py         # dataset validation gate (target choice, confounds)
scripts/01_build_pseudobulk.py         # donor x cell_type pseudobulk over the shared gene universe
scripts/02_build_graph.py              # both priors' real graphs + structural controls + joint combined mask
scripts/03_run_experiment.py           # harness evaluation: both priors + combined x all conditions x all probes
scripts/04_significance_test.py        # paired tests, permutation null, scorecard
scripts/05_make_figures.py             # figures referenced in MEMO.md
```

Every script takes `--dataset {ra,sle}` (default `ra`) and, where relevant,
`--prior-mode {combined,independent}` (default `combined`).

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

Pass `--dataset sle --prior-mode independent` to any script instead to
reproduce the original SLE/single-prior run.

Random seeds are fixed throughout (5 CV seeds x 6 folds; 5 seeds each for
the degree-preserving, fully-random, and exploratory-ablation controls).

## Dataset and target

CELLxGENE Census dataset `d18736c3-6292-4379-919a-d6d973204c87` (Binvignat
et al. 2024, *JCI Insight*) -- the only RA cohort in Census, and the same
dataset TRACE originally used. 36 donors (18 RA / 18 normal), single
assay/tissue/suspension type, paired case-control design (each RA donor
matched 1:1 with a healthy control). This dataset carries a DAS28 activity
score, stable per donor but too small a usable subgroup (16-18 donors) for
this harness's CV protocol -- used only as a descriptive probe. See
`MEMO.md` and `results/tables/step0_validation.md` for the full reasoning,
including confounds checked (none found in batch/processing; age skew
flagged not corrected).

## Combined prior

`src/priors/combined.py` builds a joint TF-target + epigenomic mask by
concatenating each prior's mask as its own block (not merging edges into a
shared per-gene hidden-unit space) -- see `MEMO.md` for the full design
rationale and the joint structural controls (degree-preserving, fully
random, sign-flip, plus an exploratory "swap only one prior to random"
ablation).

## Contributor

Danilo Dursoniah (ddursoniah@gmail.com)
