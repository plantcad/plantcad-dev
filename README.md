# BioLM Demo Repository

Biological Language Model data, training and inference pipeline prototypes.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management.

```bash
# Install uv
# ----------
# Move uv home to same filesystem as project repo which is useful
# on hosts with ephemeral root mounts (e.g. Lambda Labs); see:
# https://docs.astral.sh/uv/reference/installer/#changing-the-installation-path
export UV_INSTALL_DIR=`findmnt --first-only --noheading --output=target --target .`
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment
# ------------------
uv venv --python 3.12

# Install dependencies - choose CPU or GPU
# -----------------------------------------
# For CPU-only installation:
uv sync --extra cpu --group dev

# For GPU (CUDA 12.8) installation:
uv sync --extra gpu --extra mamba --group dev  # Mamba requires CUDA

# Install pre-commit hooks
uv run pre-commit install
```

## Execution

These examples demonstrate how to run a pipeline using the [Marin execution framework](https://github.com/marin-community/marin/blob/main/docs/tutorials/executor-101.md) (extracted to [Thalas](https://github.com/Open-Athena/thalas)):

```bash
# Activate the virtual environment
source .venv/bin/activate

CONFIG=src/pipelines/plantcad2/evaluation/configs/config.yaml

# Execute the evaluation pipeline (prefix is defined in config file)
python -m src.pipelines.plantcad2.evaluation.pipeline --config_path $CONFIG

# Force re-run of failed steps
python -m src.pipelines.plantcad2.evaluation.pipeline \
  --config_path $CONFIG --executor.force_run_failed

# Clear all pipeline data and start fresh (extract prefix from config)
PREFIX=$(yq -r '.executor.prefix' $CONFIG)
rm -rf $PREFIX
python -m src.pipelines.plantcad2.evaluation.pipeline --config_path $CONFIG
```

## Storage

Shared storage is currently supported via Hugging Face. See the [Hugging Face Filesystem API](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_file_system) docs for more details.

### Reading data

```python
from src.io import HfRepo
import pandas as pd

# Read openai/gsm8k test split
repo = HfRepo.from_repo_id("openai/gsm8k", type="dataset")
df = pd.read_parquet(repo.url("main/test-00000-of-00001.parquet"))

# Alternatively, use explicit factory function
from src.io import hf_repo

# Create repo reference
gsm8k_repo = hf_repo("gsm8k", entity="openai", type="dataset", internal=False)
train_df = pd.read_parquet(gsm8k_repo.url("train/train-00000-of-00001.parquet"))
```

### Writing data

Note: Writing data requires authentication. Use `huggingface-cli login` or set your HF token.

```python
from src import io

# Create repo reference (uses "plantcad" as default entity)
repo = io.hf_repo("test-dataset", type="dataset")

# Create the dataset repository on HuggingFace Hub
repo_url = io.create_on_hub(repo, private=False)

# Get filesystem instance and write data
fs = io.filesystem()
content = "This is a test data file."
with fs.open(repo.url("data.txt"), "w") as f:
    f.write(content)

# Example with internal naming convention
internal_repo = io.hf_internal_repo("test-dataset")
io.create_on_hub(internal_repo, private=False)  # Creates "plantcad/_dev_test-dataset"

with fs.open(internal_repo.url("data.txt"), "w") as f:
    f.write(content)
```
