# BioLM Demo Repository

Biological Language Model data, training and inference pipeline prototypes.

![Pipeline Architecture](docs/architecture.svg)

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
uv sync
uv sync --extra mamba

# Install dev tools
# -----------------
uv sync --group dev
uv run pre-commit install
```

## Execution

```bash
export METAFLOW_RUN_MAX_WORKERS=1
export PIPELINE_DIR=src/pipelines/plantcad2/evaluation

# Execute a terminal flow
uv run $PIPELINE_DIR/tasks/evolutionary_constraint/flow.py run

# Execute a terminal flow with config overrides
uv run $PIPELINE_DIR/tasks/evolutionary_constraint/flow.py run \
--overrides "tasks.evolutionary_constraint.output_dir=data/evolutionary_constraint_override"

# Execute a parent flow composing terminal flows
uv run $PIPELINE_DIR/flow.py run
```

## Debugging

Metaflow has no serial, local execution mode which makes debugging with `pdb` or `ipdb` difficult; see https://github.com/Netflix/metaflow/issues/89.  This utility below can be used to debug subprocesses, however, which provides access to a PDB debugging session through a local TCP socket.

```python
from src.debug import remote_pdb; remote_pdb(4444).set_trace()
# Connect with this in another terminal: nc 127.0.0.1 4444
```
