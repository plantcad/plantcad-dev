# PlantCAD Development Repository

This project contains [PlantCaduceus](https://doi.org/10.1101/2024.06.04.596709) (PlantCAD) evaluation and experimentation pipelines.

## Setup

Create an environment via [uv](https://docs.astral.sh/uv/):

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
uv venv

# Install dependencies - choose CPU or GPU
# -----------------------------------------
# For CPU-only installation:
uv sync --extra cpu --group dev

# For GPU (CUDA 12.8) installation:
uv sync --extra gpu --extra mamba --group dev  # Mamba requires CUDA

# Install pre-commit hooks
uv run pre-commit install
```

## Pipelines

This example shows how to use [Thalas](https://github.com/Open-Athena/thalas), Draccus, UPath, and Ray to create a simple pipeline for distributed execution:

```python
"""greeting_pipeline.py"""
import logging
import draccus
import ray
from upath import UPath
from pydantic import Field
from dataclasses import replace
from pydantic.dataclasses import dataclass
from thalas.execution import ExecutorStep, ExecutorMainConfig, output_path_of, this_output_path
from src.exec import executor_main
from src.log import initialize_logging

logger = logging.getLogger("ray")

@dataclass
class GreetingConfig:
    name: str = Field(default=None, description="The name to greet")
    output_path: str = Field(default=None, description="The path to write the greeting to")

@dataclass
class PipelineConfig:
    greeting: GreetingConfig
    executor: ExecutorMainConfig

@ray.remote # Omit to execute on ray driver instead
def greeting_step(config: GreetingConfig) -> None:
    message = f"Hello, {config.name}!"
    # Print to console
    logger.info(message)
    # Write to output path on Hugging Face
    (UPath(config.output_path) / "greeting.txt").write_text(message)

class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def greet(self) -> ExecutorStep:
        return ExecutorStep(
            name="greet",
            fn=greeting_step,
            config=replace(self.config.greeting, output_path=this_output_path()),
            description="Print a greeting"
        )

def main():
    initialize_logging()
    cfg = draccus.parse(config_class=PipelineConfig)
    pipeline = SimplePipeline(cfg)
    executor_main(cfg.executor, [pipeline.greet()], init_logging=False)

if __name__ == "__main__":
    main()
```

Run it with:

```bash
# Create a config file
cat > greeting_pipeline.yaml << 'EOF'
greeting:
  name: "World"
executor:
  # Set the HF dataset into which data from all steps will be saved
  prefix: "hf://datasets/plantcad/_dev_greeting_pipeline"
EOF

# Run the pipeline
uv run python greeting_pipeline.py \
  --config_path greeting_pipeline.yaml \
  --greeting.name "PlantCAD"
```


## Execution

### Local

This example shows how to run existing pipelines:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Path to pipeline config
CONFIG=src/pipelines/plantcad2/evaluation/configs/config.yaml

# Execute the evaluation pipeline (prefix is defined in config file)
python -m src.pipelines.plantcad2.evaluation.pipeline --config_path $CONFIG

# Force re-run of failed steps
python -m src.pipelines.plantcad2.evaluation.pipeline \
  --config_path $CONFIG --executor.force_run_failed true

# Run the pipeline without "simulation mode" (requires a GPU when false)
python -m src.pipelines.plantcad2.evaluation.pipeline \
  --config_path $CONFIG \
  --tasks.evolutionary_constraint.generate_logits.simulate_mode false \
  --executor.force_run_failed true
```

### Remote (Lambda)

This example shows how to create a Lambda cluster and run a pipeline on it:

```bash
# Create Lambda API key at https://cloud.lambda.ai/api-keys/cloud-api
# and add to ~/.lambda_cloud/credentials.json:
# echo "api_key = <key>" >> ~/.lambda_cloud/lambda_keys
sky check -v # Ensure that Lambda is detected as an available cloud

CONFIG_PATH=src/pipelines/plantcad2/evaluation/configs

# Launch a dev cluster; see:
# - https://docs.skypilot.co/en/latest/reference/cli.html
# - https://docs.skypilot.co/en/latest/reference/yaml-spec.html
sky launch -c pc-dev --num-nodes 2 --gpus "A10:1" --disk-size 100 --workdir .
# Alternatively, use the cluster YAML config:
sky launch -c pc-dev $CONFIG_PATH/cluster.sky.yaml --env HUGGING_FACE_HUB_TOKEN
# On successful completion, you will see the following:
# 📋 Useful Commands
# Cluster name: pc-dev
# ├── To log into the head VM:	ssh pc-dev
# ├── To submit a job:		sky exec pc-dev yaml_file
# ├── To stop the cluster:	sky stop pc-dev
# └── To teardown the cluster:	sky down pc-dev

# Submit a job to the cluster
# NOTE: code from the working directory is synced to the cluster
# for every `exec` and `launch` command; see:
# https://docs.skypilot.co/en/latest/examples/syncing-code-artifacts.html#sync-code-from-a-local-directory-or-a-git-repository
sky exec pc-dev $CONFIG_PATH/task.sky.yaml --env HUGGING_FACE_HUB_TOKEN

# Add arbitrary arguments to the task execution
ARGS="--executor.force_run_failed=true" sky exec pc-dev $CONFIG_PATH/task.sky.yaml \
  --env HUGGING_FACE_HUB_TOKEN --env ARGS
```

### Adding dependencies

To add new dependencies in a running cluster, you can simply run the cluster launch command again.  SkyPilot will recognize the cluster exists and then issue the same setup commands to all the nodes.  In this case, those commands include a `uv sync`.  I.e. you can do this:

```bash
uv add universal-pathlib==0.2.6
sky launch -c pc-dev $CONFIG_PATH/cluster.sky.yaml --env HUGGING_FACE_HUB_TOKEN
# ...
# └── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)
# (setup pid=4016) + uv sync --extra gpu --extra mamba
# (setup pid=5650, ip=10.19.95.95) + uv sync --extra gpu --extra mamba
# ...
# (setup pid=4016) Installed 2 packages in 0.59ms
# (setup pid=4016)  ~ plantcad-dev==0.1.0 (from file:///home/ubuntu/sky_workdir)
# (setup pid=4016)  + universal-pathlib==0.2.6s
# (setup pid=5650, ip=10.19.95.95) Installed 2 packages in 0.58ms
# (setup pid=5650, ip=10.19.95.95)  ~ plantcad-dev==0.1.0 (from file:///home/ubuntu/sky_workdir)
# (setup pid=5650, ip=10.19.95.95)  + universal-pathlib==0.2.6
```

## Ray

Both SkyPilot and Thalas require Ray, so two separate Ray clusters are running on the Lamba cluster when using remote execution.  Here are some more details on how to monitor and interact these clusters:


```bash
# View the ray dashboard port to your local machine
ssh -L 8365:localhost:8365 pc-dev -N
open http://localhost:8365

# Kill Ray processes for one of the clusters; e.g. the non-SkyPilot cluster in this case
# TODO: Find a better way to do this since Ray has no systemd (or similar) support
GCS_PORT=6479 ps aux | grep ray \
  | grep -E "(--gcs_server_port=$GCS_PORT|--gcs-address=.*:$GCS_PORT)" \
  | awk '{print $2}' | xargs kill -9
```


## Storage

Shared storage is currently supported via Hugging Face. See the [Hugging Face Filesystem API](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_file_system) docs for more details.

### Reading data

The simplest way to read remote data is through existing fsspec-compatible libraries (pandas, pyarrow, dask, xarray, etc.) or via [UPath](https://github.com/fsspec/universal_pathlib), an extension to `pathlib.Path` supporting remote file systems.

Here are a few examples:

```python
# Use fsspec compatibily libraries
import pandas as pd
df = pd.read_parquet("hf://datasets/openai/gsm8k/main/train-00000-of-00001.parquet")

# Use UPath
from upath import UPath; import gzip
path = UPath("hf://openai/gpt-oss-120b") / "config.json
# Note: HF text files are gzip compressed by default
gzip.decompress(path.read_bytes()).decode("utf-8")

# Plain text HTTP example
path = UPath("https://text.npr.org")
path.read_text(encoding="utf-8")
```

Other existing utilities like [huggingface_hub.RepoUrl](https://github.com/huggingface/huggingface_hub/blob/v0.34.4/src/huggingface_hub/hf_api.py#L536) are also useful.  This makes it straightforward to parse HF urls into their constituent parts, e.g.:

```python
from huggingface_hub import RepoUrl
url = RepoUrl("hf://openai/gpt-oss-120b")
url.repo_type, url.repo_id, url.namespace, url.repo_name
# ('model', 'openai/gpt-oss-120b', 'openai', 'gpt-oss-120b')
```

This project provides an additional utility on top of `RepoUrl` to support the separation of "internal" and "external" repositories.  This exists so that intermediate pipeline results can be stored in public HF repositories with a naming convention that separates them from final results.  Currently, that convention prefixes "_dev_" to the repository names.  Examples:

```python
from src import io
import pandas as pd

repo = io.hf_repo("plantcad/training_dataset", internal=False)
repo
# HfRepo(entity='plantcad', name='plantcad/training_dataset', type='dataset', internal=False)
repo.url()
# 'hf://datasets/plantcad/plantcad/training_dataset'

io.hf_repo("plantcad/training_dataset", internal=True).url()
# 'hf://datasets/plantcad/_dev_training_dataset'
```

### Writing data

Writing data can be done through an fsspec filesystem or via UPath.  This will require authentication, so use `huggingface-cli login` or set `HUGGING_FACE_HUB_TOKEN` in the environment to do that.  Examples:

```python
from src import io
from upath import UPath

# Create repo reference (uses "plantcad" as default entity)
repo = io.hf_repo("test-dataset", type="dataset")
repo.url()
# 'hf://datasets/plantcad/test-dataset'

# Create the dataset repository on HuggingFace Hub
io.create_on_hub(repo, private=False)

# Write with explicit filesystem instance
fs = io.filesystem()
content = "This is a test data file."
with fs.open(repo.url("data.txt"), "w") as f:
    f.write(content)

# Write via UPath with filesystem implicit in url (i.e. "hf://")
path = UPath(repo.url("data.txt"))
path.write_text(content)
```


### Deleting data

When running Thalas pipelines, it is common to need to clear the paths used for a pipeline during development, or to delete data for a specific step.  Here are some examples:

```bash
# Clear all data within the dataset (without deleting it entirely)
hf repo-files delete --repo-type dataset plantcad/_dev_pc2_eval '*'

# Clear all data for a specific step
hf repo-files delete --repo-type dataset plantcad/_dev_pc2_eval evolutionary_downsample_dataset-be132f
```
