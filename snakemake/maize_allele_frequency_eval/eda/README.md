# Maize allele frequency eval exploratory data analysis

## Setup

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
source .venv/bin/activate
snakemake --cores all
```

## Evaluation Metrics

This analysis evaluates genomic conservation models by assessing how well their functional prediction scores correlate with allele frequency patterns in maize populations. The following metrics are computed across different sample sizes and random seeds to assess model performance and stability.

### Core Premise

All metrics are based on the biological hypothesis that **functionally important genomic positions are under stronger purifying selection**, resulting in **lower allele frequencies** for alternate alleles. Therefore, higher model scores (indicating more functional/conserved positions) should correlate with lower allele frequencies.

---

### 1. Pearson Correlation

**What it measures:** Linear correlation between model scores and allele frequencies.

**Computation:**
- Standard Pearson correlation coefficient between AF and model scores
- **Sign flipped** so that negative correlation becomes positive (higher is better)
- Higher values indicate better performance (functional variants have lower AF)

**Interpretation:** Measures the linear relationship strength between predicted functional importance and observed allele frequencies. A high positive value (after sign flip) indicates that variants predicted as more functional tend to have lower allele frequencies.

**Range:** -1 to 1 (after flip: higher is better, 0 = baseline)

---

### 2. Spearman Correlation

**What it measures:** Rank-based (monotonic) correlation between model scores and allele frequencies.

**Computation:**
- Standard Spearman rank correlation coefficient between AF and model scores
- **Sign flipped** so that negative correlation becomes positive (higher is better)
- Robust to outliers and non-linear relationships

**Interpretation:** Measures how well the rank ordering of model scores corresponds to the rank ordering of allele frequencies. Less sensitive to outliers than Pearson correlation.

**Range:** -1 to 1 (after flip: higher is better, 0 = baseline)

---

### 3. AUROC (Area Under ROC Curve)

**What it measures:** Binary classification performance for distinguishing rare from common variants.

**Computation:**
- **Positive class (label=True):** Rare variants (AC = 4, i.e., allele count of exactly 4)
- **Negative class (label=False):** Common variants (AF > 20%)
- Computes standard AUROC using model scores as classifier
- Variants not meeting either threshold are excluded from evaluation

**Interpretation:** Represents the probability that a randomly selected rare variant has a higher model score than a randomly selected common variant. Higher values indicate better discrimination between rare and common variants.

**Range:** 0 to 1 (0.5 = random baseline, 1.0 = perfect)

---

### 4. AUPRC (Area Under Precision-Recall Curve)

**What it measures:** Precision-recall performance for identifying rare variants.

**Computation:**
- Uses same binary labels as AUROC (rare vs common)
- Computes average precision score (equivalent to AUPRC)
- More informative than AUROC for imbalanced datasets

**Interpretation:** Summarizes the precision-recall trade-off across all thresholds. Particularly useful when the positive class (rare variants) is less frequent. Higher values indicate better performance at identifying rare functional variants.

**Range:** 0 to 1 (baseline = proportion of positive class, 1.0 = perfect)

---

### 5. Mean AF at Quantile

**What it measures:** Average allele frequency among variants with the highest model scores.

**Computation:**
- For each quantile q (e.g., 0.01, 0.1):
  - Identify the top q fraction of variants by model score
  - Calculate mean allele frequency of these variants
  - **Flip sign** (negate) so lower mean AF becomes higher metric value
- Evaluated at multiple quantiles configured in `analysis_quantiles`

**Interpretation:** Tests whether variants predicted as most functional (top scoring) actually have lower allele frequencies. A high positive value (after sign flip) indicates that the highest-scoring variants have the lowest allele frequencies on average.

**Range:** Negative values (after flip, higher is better)

**Example:** `mean_af_at_q0.01` examines the top 1% of variants by score

---

### 6. Odds Ratio

**What it measures:** Enrichment of rare variants vs common variants among top-scoring predictions.

**Computation:**
- For each quantile q (e.g., 0.01, 0.1):
  - Select top q% and bottom q% of variants by model score
  - Create 2×2 contingency table:
    ```
                  Top q%    Bottom q%
    Rare (AC=4)    a          b
    Common (AF>20%) c          d
    ```
  - Compute Fisher's exact test odds ratio: OR = (a×d) / (b×c)
- Uses one-sided test (alternative="greater")
- Evaluated at multiple quantiles configured in `or_quantiles`

**Interpretation:** An odds ratio > 1 indicates that rare variants are more likely to be found in the top-scoring group than in the bottom-scoring group. Higher values indicate stronger enrichment of rare variants among predicted functional sites.

**Range:** 0 to ∞ (1.0 = no enrichment, >1 = enrichment)

**Example:** `odds_ratio_at_q0.01` compares top 1% vs bottom 1% of variants

---

### Additional Analysis: Ordering Consistency

Beyond individual metric scores, the analysis also computes **ordering consistency** across random subsamples:

**What it measures:** How frequently models maintain their expected performance ranking across different random samples.

**Computation:**
- For each (sample_size, metric) combination:
  - Count how many seeds produce the expected model ordering (PCAD1-l20 < PCAD1-l24 < PCAD1-l28 < PlantCAD)
  - Calculate proportion of seeds with correct ordering

**Interpretation:** Indicates the stability and reliability of model comparisons. Higher values mean the relative model rankings are robust to sampling variability.
