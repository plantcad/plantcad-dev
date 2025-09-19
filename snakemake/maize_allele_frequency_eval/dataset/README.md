# Maize allele frequency eval dataset creation

Processes the data from PCad1 paper, adds Ensembl VEP consequences, and uploads to HF.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Additional requirements:
- apptainer (for running Ensembl VEP)

## Usage

```bash
source .venv/bin/activate
snakemake --cores all --software-deployment-method apptainer
```
