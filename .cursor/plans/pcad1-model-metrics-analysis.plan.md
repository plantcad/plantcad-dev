# PCAD1 Model Metrics Analysis Plan

## Overview
Create a comprehensive analysis to compare PCAD1-l20, PCAD1-l24, PCAD1-l28, and PCAD1-l32 models using multiple metrics, with downsampling experiments to identify which metrics maintain stable model rankings.

## Prerequisites
- PCAD1-l28 predictions will be computed separately and updated tomorrow
- Use existing PCAD1-l20 and PCAD1-l24 predictions to start
- PCAD1-l32 predictions are already precomputed according to config comments

## Analysis Components

### 1. Metrics to Implement
All metrics will be computed for each model and analyzed across different sample sizes:

- **Mean AF at quantiles**: Calculate mean allele frequency for variants above various score quantiles (e.g., top 1%, 5%, 10%, 25%)
- **Correlation**: Pearson and Spearman correlation between AF and model scores
- **Odds Ratio (OR)**: Rare (AC=4) vs. Common (AF > 20%) at different score quantiles (e.g., top 30, 60, 90 variants)
- **Classification metrics**: AUROC and AUPRC for rare vs. common variant classification

### 2. Downsampling Strategy
Test metric stability by:
- Creating random subsamples at multiple sizes (e.g., 10%, 25%, 50%, 75%, 100% of data)
- Running each metric on each subsample
- Computing variance/stability of model rankings across subsamples
- Using multiple random seeds for robustness
- **Sample without replacement** for each subsample

### 3. Implementation Approach

**Snakemake-based workflow** (no Marimo notebooks):

All analysis will be implemented as Snakemake rules that generate parquet files with results.

**Update**: `workflow/rules/common.smk`

Add core metric calculation functions:
- `compute_mean_af_at_quantile(variants_df, score_col, quantiles)`: Mean AF for top quantiles
- `compute_pearson_correlation(variants_df, score_col)`: Pearson correlation between AF and model scores
- `compute_spearman_correlation(variants_df, score_col)`: Spearman correlation between AF and model scores
- `compute_odds_ratio(variants_df, score_col, n_top_variants)`: OR for rare (AC=4) vs common (AF>20%)
- `compute_auroc(variants_df, score_col)`: AUROC for rare vs common classification
- `compute_auprc(variants_df, score_col)`: AUPRC for rare vs common classification
- Helper functions for loading and merging variant + prediction data

**New file**: `workflow/rules/subsampling.smk`

Rules to generate downsampled datasets:
- `rule create_subsamples`: Generate random subsamples at various fractions (10%, 25%, 50%, 75%, 100%)
- **Sample without replacement** for each subsample
- Use multiple random seeds for robustness (e.g., 5-10 replicates per fraction)
- Output: `results/subsamples/frac_{fraction}_seed_{seed}.parquet`

**New file**: `workflow/rules/metrics.smk`

Rules to compute metrics on full and subsampled data:
- `rule compute_mean_af_quantile`: Generate `results/metrics/mean_af_quantile/{model}_{sample}.parquet`
- `rule compute_correlation`: Generate `results/metrics/correlation/{model}_{sample}.parquet`
- `rule compute_odds_ratio`: Generate `results/metrics/odds_ratio/{model}_{sample}.parquet`
- `rule compute_classification`: Generate `results/metrics/classification/{model}_{sample}.parquet`
- `rule aggregate_metrics`: Combine all metrics into summary tables for analysis

**Update**: `config/config.yaml`

Add configuration for:
- List of PCAD1 models to analyze (start with l20, l24)
- Models requiring sign flip (e.g., PCAD1 models use negative log-likelihood, need to flip to make higher = more functional)
- Subsample fractions and seeds
- Quantiles for mean AF analysis
- Thresholds for OR analysis

**Update**: `workflow/Snakefile`

Include new rule files and define `rule all` to generate all metric outputs

### 4. Analysis Workflow

1. Load full dataset with variants + PCAD1 predictions
2. **Filter to odd chromosomes only** (1, 3, 5, 7, 9) - validation set
3. Generate subsampled datasets at different fractions from odd-chrom data
4. Compute each metric for each model on each subsample
5. Aggregate results to analyze metric stability and model rankings
6. Results saved as parquet files for downstream visualization

**Notes**:
- Using odd chromosomes (validation set) for this analysis. Even chromosomes (test set) are reserved and will not be touched.
- **Analysis uses all consequences together** (not stratified by consequence type). Future work may stratify by specific consequences (e.g., missense variants only).

## Expected Outcomes
- Clear understanding of which metrics are most stable at smaller sample sizes
- Identification of minimum dataset size needed for reliable model comparison
- Ranking of PCAD1 models by their ability to predict functional variants
- Recommendations for which metrics to use in future model evaluations

## TODOs

- [x] ~~Compute PCAD1-l28 predictions using Snakemake~~ (CANCELLED - user doing separately)
- [x] ~~Create new Marimo notebook for PCAD1 model comparison analysis~~ (CANCELLED - using Snakemake instead)
- [x] Implement metric calculation functions in common.smk (mean AF, correlation, OR, AUROC/AUPRC) - **COMPLETED**
- [x] Create subsampling.smk with rules for generating subsampled datasets - **COMPLETED**
- [x] Create metrics.smk with rules for computing all metrics on subsampled data - **COMPLETED**
- [x] Update config.yaml with PCAD1 model configurations, sign flipping, and analysis parameters - **COMPLETED**
- [x] Update Snakefile to include new rule files and define targets - **COMPLETED**
- [x] Execute analysis on available PCAD1 models (l20, l24) and wait for l28 - **READY TO RUN**
- [x] Refactor binary label creation: Move to full dataset creation time in `create_complete_dataset` rule, use boolean type instead of float (1.0/0.0), and drop rows with null labels for binary classification tasks - **COMPLETED**
- [x] Optimize metric computation: Load only required columns from parquet files when computing metrics to reduce memory usage and improve performance - **COMPLETED**

## Progress
- ✅ **COMPLETED**: Implemented all metric calculation functions in common.smk:
  - `filter_analysis_chromosomes()`: Filter to configurable chromosomes (not hardcoded)
  - `compute_mean_af_at_quantile()`: Mean AF at different score quantiles
  - `compute_pearson_correlation()`: Pearson correlation between AF and scores
  - `compute_spearman_correlation()`: Spearman correlation between AF and scores
  - `compute_odds_ratio()`: OR for rare vs common variants using configurable thresholds
  - `compute_auroc()`: AUROC for rare vs common classification using configurable thresholds
  - `compute_auprc()`: AUPRC for rare vs common classification using configurable thresholds
  - **IMPROVED**: All imports moved to top of common.smk for better organization
  - **IMPROVED**: No hardcoded values - all configurable via config.yaml

- ✅ **COMPLETED**: Created subsampling.smk with rules:
  - `create_complete_dataset`: Load variants, add all PCAD1 model predictions, THEN filter to analysis chromosomes
  - `create_subsamples`: Generate random subsamples using polars built-in sampling with seeds
  - **IMPROVED**: Uses polars.sample() instead of random library, works fine for fraction=1.0
  - All sampling done without replacement using polars native functions
  - **REFACTORED**: More efficient approach - create complete dataset first, then subsample from it

- ✅ **COMPLETED**: Created metrics.smk with rules:
  - `compute_mean_af_quantile`: Mean AF at quantiles for each model/subsample
  - `compute_pearson_correlation`: Pearson correlation for each model/subsample
  - `compute_spearman_correlation`: Spearman correlation for each model/subsample
  - `compute_odds_ratio`: Odds ratio for rare vs common at different thresholds
  - `compute_auroc`: AUROC for rare vs common classification
  - `compute_auprc`: AUPRC for rare vs common classification
  - `aggregate_all_metrics`: Combine all metric results into summary tables
  - **REFACTORED**: All metric rules now work with complete dataset containing all model scores as columns

- ✅ **COMPLETED**: Updated config.yaml with PCAD1 analysis configuration:
  - PCAD1 models to analyze (l20, l24, PlantCAD as precomputed l32)
  - Sign flipping configuration for PCAD1 models
  - Subsample fractions (10%, 25%, 50%, 75%, 100%)
  - Random seeds for robustness (5 seeds)
  - Analysis quantiles for mean AF (1%, 5%, 10%, 25%)
  - Odds ratio thresholds (30, 60, 90 top variants)
  - **IMPROVED**: Analysis chromosomes configurable (not hardcoded)
  - **IMPROVED**: Rare/common thresholds configurable (AC=4, AF>20%)
  - **IMPROVED**: Proper data types - numeric values without quotes, chromosomes as strings

- ✅ **COMPLETED**: Updated Snakefile to include new rule files and targets:
  - Added subsampling.smk and metrics.smk includes
  - Added complete_dataset.parquet and aggregated_metrics.parquet as targets
  - Maintains original model prediction targets

## READY TO RUN

The complete PCAD1 model analysis framework is now implemented and ready to run!

### To execute the analysis:

```bash
cd snakemake/maize_allele_frequency_eval/eda
source .venv/bin/activate
snakemake --cores all
```

### What will be generated:

1. **Complete dataset**: `results/complete_dataset.parquet`
   - All variants from odd chromosomes with all PCAD1 model scores as columns
   - Sign flipping already applied for PCAD1 models

2. **Subsampled datasets**: `results/subsamples/frac_{fraction}_seed_{seed}.parquet`
   - 5 fractions × 5 seeds = 25 subsampled datasets
   - Plus 1 full odd-chromosome dataset
   - All contain complete model score columns

3. **Metric results**: `results/metrics/{metric_type}/{model}_{fraction}_{seed}.parquet`
   - Mean AF quantile results
   - Pearson correlation results
   - Spearman correlation results
   - Odds ratio results
   - AUROC results
   - AUPRC results

4. **Aggregated results**: `results/aggregated_metrics.parquet`
   - Combined all metric results for analysis

### Analysis scope:
- **Models**: PCAD1-l20, PCAD1-l24, PlantCAD (precomputed PCAD1-l32) - PCAD1-l28 will be added when available
- **Data**: Configurable chromosomes (default: "1", "3", "5", "7", "9") - validation set
- **Metrics**: All 6 metric types across multiple subsample sizes
- **Robustness**: 5 random seeds per subsample size
- **Thresholds**: Configurable rare (AC=4) vs common (AF>20%) thresholds

### Next steps after running:
1. Load `results/aggregated_metrics.parquet` for analysis
2. Compare metric stability across subsample sizes
3. Identify which metrics maintain consistent model rankings
4. Add PCAD1-l28 when available by updating config.yaml
