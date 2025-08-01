# BioLM Demo

Biological Language Model demonstrations and experiments.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management.

```bash
# Install dependencies
uv venv
uv pip install --no-config --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv sync --dev --no-build-isolation

# Install pre-commit hooks
uv run pre-commit install
```

## Execution

```bash
uv run src/pipelines/plantcad2/evaluation.py run \
--model_path kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000 \
--sample_size 100
```
