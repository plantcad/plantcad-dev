# Maize missense genomic prediction

CPU-only comparison of precomputed PlantCAD-l32, phyloP and phastCons scores for
genomic prediction. This is a new pipeline, informed by PR #29. It preserves the
supplied genome-wide G0 and population-structure covariates, and adds a missense
kernel G1. No inference models are downloaded or run.

## Layout and dependencies

This directory is an independent uv project with a checked-in `uv.lock`.
Python functions live in `src/maize_missense/` and use Polars (not pandas).
Sourceable R functions live in `src/r/`; they use SNPRelate for GDS access and
qgg for REML. The Snakemake rules and `workflow/scripts/*.R` only orchestrate
these functions. R packages are described by `workflow/envs/r.yaml`; Python's
lockfile does not lock R packages. The fit reports record the R session versions.

Run commands below from this directory:

```sh
uv sync --locked
uv run --locked snakemake --dry-run
uv run --locked snakemake prepare --jobs 1 --cores 8
uv run --locked snakemake --jobs 1 --cores 8
```

Snakemake 9 requires a recent Conda (>=24.7). The default workflow profile stores
inputs and outputs at `s3://oa-plantcad-dev/snakemake/maize_missense_prediction/v1`.
It runs one job at a time. AWS credentials or an instance role must permit reads
and writes there. S3 storage is independent of the execution backend.

For local tests or a scratch run without S3, use `--workflow-profile none`.
For an isolated S3 experiment override `--default-storage-prefix` with a fresh
prefix. Production data preparation and fitting belong on a suitably sized CPU
worker, not the shared development VM.

From the repository root, use SkyPilot on AWS:

```sh
sky launch -y -c pcad-missense-cpu -i 30 --down snakemake/maize_missense_prediction/sky/cpu.yaml
# To prepare and inspect the missense intersection before fitting:
sky exec pcad-missense-cpu snakemake/maize_missense_prediction/sky/cpu.yaml --env TARGET=prepare
```

This requests >=8 vCPUs, >=32 GiB RAM, 100 GB disk and no GPU. It terminates after
30 idle minutes; outputs are uploaded by Snakemake as rules complete. The
`sky/setup.yaml` and `sky/validate.yaml` tasks support development and validation
on the same worker. A separate Conda installation for the workflow avoids
modifying SkyPilot's base Python and native libraries.
This follows marin-dna's separation of a workflow Conda installation from
SkyPilot, with rule-specific `workflow/envs/` files, strict channel priority,
and Conda enabled in the default S3 profile. Here the isolated launcher pins
Conda 25.7.0 rather than installing the latest Miniforge.

## Dataset contract

Hugging Face revisions are pinned in `config/config.yaml`:

1. Scores: `plantcad/maize-allele-frequency-raw-data`, columns `PlantCAD`, `phyloP`,
   and `phastCons`. These are the original stored conservation scores, not the
   newer conservation tracks downloaded by PR #29.
2. Annotations: `plantcad/maize-allele-frequency`, **full** validation and test
   partitions combined. Their chromosome partition is not a phenotype validation
   split. Missense means `consequence == "missense_variant"` under the supplied
   most-severe annotation, rather than any missense consequence on any transcript.
3. Hybrid genotypes, G0, Q and phenotypes: `plantcad/_dev_maize_genomic_prediction_eval`.

First intersect missense annotations and score coordinates, requiring finite
scores from all three methods. Then lift the hybrid GDS coordinates from AGPv4
to v5, validate the reference allele against the v5 FASTA, and match all four
fields `(chrom, pos, ref, alt)`. GDS allele order is not assumed to be REF/ALT:
we identify the reference from the FASTA and record allele-order swaps. Failed
or multiple liftover mappings, allele mismatches, and duplicated target variant
keys are excluded and counted. Duplicate score keys are an error.

Original GDS SNP IDs accompany every retained variant and weight. The R reader
realigns returned genotype columns by SNP ID, never by assumed row order.
`prepare` produces the candidate table, final intersection, coverage reports,
and fifteen weight tables before any REML evaluation.

## Weights and kernels

PlantCAD uses `abs(LLR)` only. phyloP and phastCons retain their usual higher-score
ordering. Each method uses the same final variant set. For fractions 1%, 5%,
10%, 20%, and 50%, let `k = floor(n * fraction)`:

- Score above the kth-highest score: weight 1.
- Score below it: weight 0.
- Boundary tie with `t` variants and `a` variants strictly above it: weight
  `(k - a) / t` for every tied variant.

Total weight is k; the number of nonzero variants may be larger. An all-tied score
produces uniform weights. Reports record cutoff, boundary tie count and weight,
nonzero count and effective variant count. There is no jitter or random tie
breaking. Raw score atoms remain in the variant table.

G1 is a nonnegative weighted genotype cross-product. Genotypes are mean-imputed
and centered, without variance scaling by default, and G1 is normalized to mean
diagonal 1. Monomorphic/all-missing SNPs make no contribution and are reported.
This differs deliberately from PR #29's uncentered kernel: centered kernels are
invariant to which allele is counted. The construction is positive semidefinite
without `nearPD`. G0 is retained as supplied.

## Evaluation

Evaluate G0 alone and G0+G1 for each of the fifteen method/fraction combinations on
GY_adjusted, PH and DTS, using cross-panel (CP) and NAM leave-one-family-out
(LOFO) validation. Family IDs use PR #29's convention: the prefix before `E` in
the sample ID. Missing trait values are excluded before forming each trait's
folds; all methods share the same splits.

Fit qgg explicitly on each training fold and predict the held-out samples from
the fitted fixed effects and cross-kernel BLUP. Validation phenotypes do not
enter parameter fitting. This also avoids qgg's bundled cross-validation wrapper
silently failing to forward some fit settings. Some qgg releases fix their
internal tolerance to 1e-5; the default matches that value. Failed or unconverged
fits stop the rule, rather than becoming silently missing performance rows.

Outputs include per-sample held-out predictions, per-fold Pearson r and MSE,
variance components, iteration counts and package versions. The summary adds
paired differences from G0 alone. The controls below are included; no directional LLR
transform is included. All predefined settings are reported; test-fold results
must not be used to select a favorable transform and present it as an unbiased
final estimate.

A Snakemake checkpoint identifies duplicate relative weight sets by hashing SNP
IDs and weights divided by their maximum, in SNP-ID order. Equality is exact;
there is no tolerance that could collapse genuinely different weights.
Only distinct normalized kernels are built and fitted. Every requested setting
still gets labeled metric and prediction files, with `fitted_model` and
`reused_fit` provenance. `results/qc/evaluation_plan.json` records the complete
mapping, and the final report includes it.

## Missense controls

Two score-free controls use the same 152,844 matched missense variants with
finite scores in **all three models**; the supplied genome-wide G0 stays unchanged.

- **G0 + all missense:** weight 1 on every variant, using the same centered,
  mean-diagonal-normalized kernel construction as the score-based models.
- **G0 + shuffled PlantCAD weights:** one fixed-seed shuffle at each of PlantCAD's
  five fractions. Shuffling preserves the complete weight multiset, including
  zeros and fractional boundary weights. This is not a new score-based ranking.

The current configuration requests only PlantCAD and seed `20260929`: five random
kernels plus the unweighted baseline, for six additional fits. Add methods/seeds
under `controls` to extend this later. SNP IDs are sorted before applying NumPy
PCG64's permutation, making results independent of input row order. A seed uses
the same SNP permutation across settings, preserving their overlap and exact
relative-weight equivalences. The control checkpoint deduplicates only identical
normalized weight assignments; every requested setting retains its label and
fit provenance. We do **not** additionally match MAF, LD, gene or chromosome.

Existing score-based fits are reused. Controls have separate weights, kernels,
metrics and predictions under `results/controls/`, with an explicit
`results/qc/control_plan.json` included in the final report. Their training
procedure and held-out folds are unchanged. A single random shuffle is a sanity
check, not a calibrated permutation test or an estimate of random-control
variability. No seed is selected based on held-out performance.

## Result plots

The separate [methodology schematic](docs/methodology.svg) is a standalone,
editable-text SVG documenting the 2026-09-29 configuration and observed cohort
counts (rounded for readability). It focuses on the three-score comparison:
the common variant filter, tie-aware weighting, kernels and held-out evaluation.
It distinguishes cross-panel transfer (panel-wide metrics) from transfer to an
unseen NAM family (within-family metrics averaged over held-out families).
Controls and implementation details are omitted from the schematic but retained
in the analysis and documented above. This documentation snapshot is
independent of the results plot; update it if the analysis design changes.

The default target also writes one `results/plots/prediction_summary.svg`, showing
**held-out prediction accuracy (Pearson r) only**. Two rows show the cross-panel
average and NAM leave-one-family-out. Columns show an all-trait average followed
by yield, plant height and days to silking. There are no PNG, PDF or CSV plot
exports. The plotted values and detailed summaries are saved together in
`results/plots/plot_data.parquet`.

Versioned snapshots of the completed analysis are available in
[`docs/prediction_summary.svg`](docs/prediction_summary.svg) and
[`docs/plot_data.parquet`](docs/plot_data.parquet), alongside the separate
[`methodology schematic`](docs/methodology.svg). These small reporting artifacts
are committed for review and commit-pinned GitHub issue links; raw data and run
caches remain excluded. The workflow writes fresh results under `results/`.
After an intentional rerun, refresh the two report snapshots from
`results/plots/` before committing updated results.

Each fraction is a separate group of horizontally offset points: PlantCAD,
its shuffled control, phyloP and phastCons. Methods use distinct marker shapes;
shuffled points have hollow markers and stay beside their originating method.
There are no connecting lines. Fractions are categorical, equally spaced groups
(not a log axis). Matplotlib autoscales each subplot independently from its
points, displayed intervals and reference lines. Only its default margin is
used: there is no custom padding, rounding limits to ticks, forced zero origin
or sharing of limits across traits or validation rows.
The genome-wide baseline (G0) and genome-wide + all-missense baseline are
horizontal dashed/dotted reference lines, with full names in the legend.
They are not fraction categories. Their confidence intervals remain in
Parquet but are not shown, keeping the plot's reference lines uncluttered.
The Parquet has explicit `kind` and `seed` columns (seed -1 means no shuffle).
Seeds are never pooled with families; each requested seed remains its own
point series and its own table rows.

Averaging is explicit and balanced:

- For each trait, `CP/mean` is the arithmetic mean of NAM → Ames and Ames → NAM
  correlations. The two directions have equal weight, not sample-count weights.
- For each trait, `LOFO` is the equal-weight mean of within-family correlations.
- For each evaluation, `trait == "ALL_TRAITS"` is the equal-weight mean of the
  three trait means above. A trait with more usable families does not get extra
  weight. These are descriptive means of correlations, not pooled correlations
  or Fisher-z means. Cross-panel and LOFO are not combined into one number.

The Parquet retains individual directions (`CP/Ames_H`, `CP/NAM_H`), per-trait
family summaries, paired changes in correlation and paired MSE changes. MSE is
**not** averaged across traits with different units. Differences are calculated
within folds before aggregation. `aggregation`, `n_traits`, `n_directions`, and
`n_folds` record how each summary was formed; `n_folds` counts underlying folds,
not independent replicates or the averaging denominator for a macro mean.
The detailed `q25`/`q75` values describe family heterogeneity, not uncertainty.
Those fields are null for direction/trait macro means: no confidence interval
is inferred by averaging quantiles.

### Conditional uncertainty

The figure uses **capped vertical error bars, never shaded bands**. Bars are
pointwise 95% percentile intervals from 5,000 paired cluster-bootstrap draws
(PCG64 seed 20260930, configurable under `uncertainty`). Each draw recomputes the
correlations and the same equal-weight summaries; interval endpoints are not
averaged. Only the fraction-dependent score/shuffle points carry error bars;
G0 and all-missense intervals are retained in the underlying Parquet table.

Resampling uses the saved, held-out predictions, with no model refits:

- **NAM:** resample the 24 whole family blocks. Cross-panel NAM correlations
  are recalculated from all observations in the sampled blocks. For LOFO,
  average the sampled families' within-family correlations, counting repeated
  families with their sampled multiplicity. Within-family observations stay
  fixed; there is no second within-family bootstrap.
- **Ames:** resample parent blocks, keeping both tester hybrids together.
  Resample within fixed tester-pattern strata. In the saved predictions,
  these are B47-only (407 parents), PHZ51-only (230), and both (233): 870 parent
  blocks and 1,103 hybrids, after phenotype filtering. This preserves the
  observed tester composition (640 B47 and 463 PHZ51 hybrids). Parent IDs are
  the portion before `/tester`, retaining accession IDs. The raw phenotype
  table has 1,106 hybrids and 871 parents, but only samples with saved held-out
  predictions enter resampling. NAM similarly retains 1,630 predicted hybrids
  from 24 families, out of 1,640 phenotype-table rows.
- Reuse **identical block counts across every model, threshold, trait and
  validation scheme**. Compute paired `delta_r` versus G0 within each draw.
  The two named directions and three named traits are fixed conditions, not
  independently resampled studies. No Fisher-z or inverse-variance weighting.

These intervals are **conditional on frozen fits and observed cluster
distributions**. They assume exchangeability between the resampled blocks
within each stratum. They do not capture residual kinship between different
parents/families, uncertainty from training or overlapping training folds, or
generalization to new traits, panels or testers. They are not simultaneous
intervals or multiplicity-adjusted significance tests. Shuffle intervals
condition on the one fixed permutation; they do not measure variability
across shuffles. A full training-resampling analysis would need refitting.

`plot_data.parquet` contains `ci_lower`, `ci_upper`, `ci_level`, `ci_method`,
`ci_scope`, `bootstrap_seed`, `n_bootstrap` and `n_bootstrap_valid` for both `r`
and paired `delta_r`. MSE-change intervals are not estimated (null fields).
`results/uncertainty/intervals.parquet` stores the interval table separately;
`sample_units.parquet` records the exact grouping of every hybrid;
`report.json` records strata, assumptions, input SHA256 hashes and provenance.
Any degenerate replicate is excluded jointly across all comparisons and
reported; more than 1% degenerate draws causes an error, not silent filtering.

All predeclared fractions remain visible, including identical conservation
kernels; there is no best-fraction selection. Plotting rejects incomplete
or nonfinite fold results instead of silently comparing different subsets.
Plot functions live in `src/maize_missense/plots.py`, use Polars and Matplotlib
without pandas, and have local tests for pairing, balanced averaging,
validation and figure generation.

To render only the figure from an existing S3 summary, without permitting any
data-preparation or fitting rules:

```sh
uv run --locked snakemake plot_results --allowed-rules plot_results --jobs 1 --cores 1
```

To generate uncertainty and the figure from existing metrics and predictions,
while still forbidding data preparation, kernel building and model fitting:

```sh
uv run --locked snakemake plot_results --allowed-rules bootstrap_uncertainty plot_results --jobs 1 --cores 1
```

## First real-data preparation (2026-09-28)

The CPU-only `prepare` target completed on SkyPilot AWS (8 vCPUs, 32 GiB RAM)
and wrote its outputs to the default S3 prefix. It found 317,734 annotated
missense variants, all present in the raw score table. Requiring finite values
from all three methods left 286,769; matching against the 13,313,939 GDS variants
after liftover retained 152,844. No matched target keys were duplicated.
The full counts are in `results/qc/score_coverage.json` and
`results/qc/genotype_coverage.json` under that S3 prefix.

The initial weight reports reveal a material limitation of the conservation
scores in this dataset:

| Method | Maximum-score tie size | Nonzero variants at soft-top 1% |
| --- | ---: | ---: |
| PlantCAD abs(LLR) | Not relevant to this boundary | 1,528 |
| phyloP | 29,538 (score 0.695) | 29,538 |
| phastCons | 15,617 (score 1.0) | 15,617 |

Both conservation maxima contain more than 10% of the matched variants.
Consequently their 1%, 5%, and 10% soft-top weights differ only by a global
multiplier, which disappears when G1 is normalized. These settings are the
same normalized kernel for each conservation method, not independent parameter
comparisons. PlantCAD's three settings remain distinct. This is a property of
the supplied score atoms; the workflow does not invent a ranking within ties.
The evaluation grid was subsequently extended to 20% and 50% for all methods
to include settings beyond these top-score atoms. The real-data checkpoint
confirms that these 15 requested settings yield 11 distinct normalized kernels,
plus the separate G0-only baseline.

## Tests and pre-commit

```sh
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
uv run --locked ty check
uv run --locked snakefmt --check workflow
uv run --locked pytest
uv run --locked snakemake --conda-create-envs-only --jobs 1 --cores 1
# Activate the R environment created under .snakemake/conda, then:
Rscript tests/run_r.R
```

Python tests cover cutoff ties, permutation invariance, missing scores, duplicate
keys, reference/allele handling, and liftover matching. R tests use a synthetic GDS
to verify metadata extraction, shuffled SNP IDs, chunked versus direct kernels,
allele reversal and missingness. They test fold membership and held-out outcome
isolation, and compare a native qgg prediction with the mixed-model formula.

The repository's root `.pre-commit-config.yaml` has hooks restricted to this
directory for Ruff, ty and snakefmt, executed using this project's locked uv
environment. Install the existing repository hook with `pre-commit install`;
do not install a second hook that replaces the root configuration. Tests are run
explicitly, rather than downloading R dependencies on every commit.
All Python functions, including tests, have argument and return type hints;
Ruff's `ANN` rules enforce this and ty checks their use. R function contracts
are documented with parameter/return types and validated at runtime.
