# Implementation Plan and Status

## Context

This experiment implements genomic prediction evaluation for maize hybrid yield, assessing how different models evaluate deleterious mutations. See [GitHub issue #22](https://github.com/plantcad/plantcad-dev/issues/22).

Key references:
- Ramstein et al. (2022). [Genome Biology paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02747-2)
- Long et al. (2022). Utilizing evolutionary conservation to detect deleterious mutations and improve genomic prediction in cassava. *Frontiers in Plant Science*, 13, 1041925.
- Wu et al. (2023). Phylogenomic discovery of deleterious mutations facilitates hybrid potato breeding. *Cell*, 186(11), 2313-2328.e15.

## Implementation Status

### ✅ Completed

1. **Snakemake workflow structure**
   - Created `workflow/Snakefile` as main entry point
   - Created `workflow/rules/common.smk` for data download
   - Created `workflow/rules/reml.smk` for REML model evaluation
   - All Python imports in `common.smk`

2. **Data download**
   - Rule `download_input_data` downloads entire HF repository using `hf download` CLI
   - Downloads to `results/input_data/`
   - Explicitly lists all required files as outputs:
     - `results/input_data/hybrids/G.rds`
     - `results/input_data/hybrids/Q.rds`
     - `results/input_data/hybrids/AGPv4_hybrids.gds`
     - `results/input_data/NAM_H/pheno.rds`
     - `results/input_data/Ames_H/pheno.rds`
   - HF repo ID: `plantcad/_dev_maize_genomic_prediction_eval`

3. **Baseline model evaluation**
   - Refactored `eval_baseline_G0.R` to accept command-line arguments:
     - `--hybrids-dir`, `--nam-dir`, `--ames-dir`, `--output`, `--n-threads`
   - Moved to `workflow/scripts/eval_baseline_G0.R`
   - Outputs Parquet format (flattened CP and LOFO results with `validation` column)
   - Uses conda environment `workflow/envs/r-reml.yaml`
   - Threads configured via `workflow.cores`

4. **Conda environment**
   - Created `workflow/envs/r-reml.yaml` with R dependencies:
     - r-base>=4.0
     - r-data.table
     - r-qgg
     - r-matrix
     - r-optparse
     - r-arrow (for Parquet output)
   - Channels: conda-forge, bioconda, r

5. **Project setup**
   - Created `README.md` with setup and usage instructions
   - Created `.gitignore` (`.snakemake/`, `results/`)
   - Created `requirements.txt` (snakemake, huggingface_hub)
   - Created `config/config.yaml` with HF repo ID
   - Installable with `uv` (similar to `maize_allele_frequency_eda`)

6. **Configuration**
   - Minimal config: only `input_data_hf_repo_id`
   - No hardcoded paths in R scripts
   - No configuration for traits, Q variables, validation folders (hardcoded in R script for now)

## TODO / Future Work

1. **Null model evaluation** (`eval_null_random.R`)
   - Add rule to `workflow/rules/reml.smk`
   - Requires `r-snprelate` package (not yet in conda env)
   - Creates MAF-matched random GRMs at quantiles {0.9, 0.99, 0.999}
   - Generates multiple replicates (B replicates)

2. **Score-based model evaluation** (`eval_score_model.R`)
   - Add rule to `workflow/rules/reml.smk`
   - Requires `r-snprelate` package
   - Takes variant effect scores as input (PNC, SIFT, ESM, etc.)
   - Builds weighted GRMs based on score quantiles

3. **Enhanced weighting schemes** (from issue #22)
   - Continuous SNP weighting (not just binary 0/1)
   - Fixed-effect mutational load with weights (vs. current random effects)
   - Log-scale weights for ESM scores
   - Compare different weighting approaches

4. **Dependencies**
   - Resolve `r-snprelate` availability in conda/bioconda
   - May need to install via R's `install.packages()` or BiocManager if not in conda

## Key Decisions

- **Output format**: Parquet (efficient, type-preserving, works well with pandas/polars)
- **Conda path**: Use relative paths (`../envs/r-reml.yaml`) from rule files
- **Threads**: Use `workflow.cores` and pass via `--n-threads` argument
- **Conda activation**: Use CLI flags (`--use-conda`) rather than `use_conda: True` in Snakefile
- **Reference scripts**: Keep original R scripts in root directory for reference

## File Structure

```
experiments/maize_genomic_prediction_eval/
├── config/
│   └── config.yaml              # HF repo ID
├── workflow/
│   ├── Snakefile                 # Main workflow
│   ├── rules/
│   │   ├── common.smk            # Data download, Python imports
│   │   └── reml.smk              # REML model evaluation rules
│   ├── envs/
│   │   └── r-reml.yaml           # Conda environment for R
│   └── scripts/
│       └── eval_baseline_G0.R    # Baseline evaluation script
├── eval_baseline_G0.R            # Reference (original)
├── eval_null_random.R            # Reference (to be migrated)
├── eval_score_model.R            # Reference (to be migrated)
├── README.md
├── requirements.txt
└── .gitignore
```

## Usage

```bash
# Setup
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt

# Initialize conda (if needed)
eval "$(/path/to/conda/bin/conda shell.bash hook)"

# Run workflow
snakemake --cores all --use-conda
```

## Notes

- Conda and uv can coexist: uv venv for Snakemake, conda for R dependencies
- Snakemake creates isolated conda environments per rule automatically
- Original R scripts kept as reference until fully migrated
