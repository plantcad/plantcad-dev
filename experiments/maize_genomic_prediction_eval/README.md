# Maize Genomic Prediction Evaluation

This experiment evaluates genomic prediction models for hybrid yield prediction in maize, assessing how different models evaluate deleterious mutations. See [issue #22](https://github.com/plantcad/plantcad-dev/issues/22) for context.

## Overview

We use genomic prediction of hybrid yield to assess how different models can prioritize functional variants. The approach is based on the methodology described in:

- [Ramstein et al. (2022)](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02747-2)

## Setup

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

Additional requirements:
- conda

## Usage

Run the baseline model evaluation:

```bash
source .venv/bin/activate
snakemake --cores all --use-conda
```

This will:
1. Download input data from HuggingFace
2. Run the baseline REML model evaluation using only the polygenic GRM (G0)
3. Generate cross-panel (CP) and leave-one-family-out (LOFO) validation results

## Baseline Model

The baseline model (`eval_baseline`) evaluates genomic prediction using only the polygenic genomic relationship matrix (G0), without any SNP weighting or filtering. This serves as a control to compare against models that incorporate variant effect scores.

### Output

Results are saved to `results/CV_baseline.rds` and contain:
- Cross-panel (CP) validation: Predictions across NAM_H and Ames_H panels
- Leave-one-family-out (LOFO) validation: Predictions within NAM_H families
- Metrics: Correlation (r) and mean squared prediction error (MSE) for each trait and validation split

### Traits Evaluated

- `GY_adjusted`: Adjusted grain yield
- `PH`: Plant height
- `DTS`: Days to silking

## Configuration

Edit `config/config.yaml` to configure:
- HuggingFace dataset repository ID (`input_data_hf_repo_id`)

## Future Models

Additional REML models will be added to `workflow/rules/reml.smk`:
- Null models with random MAF-matched SNPs
- Score-based models using variant effect predictions (PNC, SIFT, ESM, etc.)
