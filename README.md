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
from src.utils.logging_utils import filter_known_warnings, initialize_logging
from src.exec import executor_main

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
            config=replace(
              self.config.greeting,
              output_path=this_output_path()
              # Note: output_path_of(self.some_step()) can
              # be used to access data from any prior step
            ),
            description="Print a greeting"
        )

def main():
    initialize_logging()
    filter_known_warnings()
    cfg = draccus.parse(config_class=PipelineConfig)
    pipeline = Pipeline(cfg)
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

## Local Execution

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

## Remote Execution (Lambda Cloud)

This example shows how to create a Lambda cluster and run a pipeline on it.

### Launch cluster

Create a new cluster with the SkyPilot [launch](https://docs.skypilot.co/en/latest/reference/cli.html#sky-launch) command.

```bash
# Clear any existing, local SkyPilot state and stop the API server
sky api stop; [ -d ~/.sky ] && rm -rf ~/.sky

# Create Lambda API key at https://cloud.lambda.ai/api-keys/cloud-api
# and add to ~/.lambda_cloud/credentials.json:
# echo "api_key = <key>" >> ~/.lambda_cloud/lambda_keys
sky check -v # Ensure that Lambda is detected as an available cloud

# Set the number of nodes to launch and manage
NUM_NODES=2

# Launch a dev cluster; see:
# - https://docs.skypilot.co/en/latest/reference/cli.html
# - https://docs.skypilot.co/en/latest/reference/yaml-spec.html
sky launch -c pc-dev --num-nodes $NUM_NODES --gpus "A10:1" --disk-size 100 --workdir .

# Alternatively, use the cluster YAML config:
CONFIG_PATH=src/pipelines/plantcad2/evaluation/configs
sky launch -c pc-dev $CONFIG_PATH/cluster.sky.yaml --num-nodes $NUM_NODES --env HUGGING_FACE_HUB_TOKEN
# On successful completion, you will see the following:
# 📋 Useful Commands
# Cluster name: pc-dev
# ├── To log into the head VM:	ssh pc-dev
# ├── To submit a job:		sky exec pc-dev yaml_file
# ├── To stop the cluster:	sky stop pc-dev
# └── To teardown the cluster:	sky down pc-dev

# View the ray dashboard for the new cluster on your local machine
# to ensure that all nodes came online as expected
ssh -L 8365:localhost:8365 pc-dev -N && open http://localhost:8365

# Alternatively, check status on head node directly:
ssh pc-dev 'source ~/sky_workdir/.venv/bin/activate && ray status --address="localhost:6479"'
```

Note that:

- Both SkyPilot and Thalas require Ray, so two separate Ray clusters are deployed via `sky launch`
- Any environment variables, e.g. `HUGGING_FACE_HUB_TOKEN`, necessary for use in Ray workers must either be set when the cluster is launched or specified on a per-task basis via the [RuntimeEnv](https://docs.ray.io/en/latest/ray-core/api/doc/ray.runtime_env.RuntimeEnv.html) for a `ray.remote` task.
- Environment variables only necessary for the driver can be set prior to running the python script for that driver instead.


### Submit job

Submit jobs to the cluster with the SkyPilot [exec](https://docs.skypilot.co/en/latest/reference/cli.html#sky-exec) command.

```bash
# Submit a job to the cluster
# - Code from the working directory is synced to the cluster
#   for every `exec` and `launch` command; see:
#   https://docs.skypilot.co/en/latest/examples/syncing-code-artifacts.html#sync-code-from-a-local-directory-or-a-git-repository
# - It is currently necessary to specify NUM_NODES to force the inclusion of
#   the head node for the task, which is essential for Thalas since it introspects
#   on the cluster through the `localhost`, i.e. it assumes execution on the head node
sky exec pc-dev $CONFIG_PATH/task.sky.yaml \
  --num-nodes $NUM_NODES --env HUGGING_FACE_HUB_TOKEN

# Add arbitrary arguments to the task execution
ARGS="--tasks.evolutionary_constraint.downsample_dataset.sample_size=1000" \
sky exec pc-dev $CONFIG_PATH/task.sky.yaml \
  --num-nodes $NUM_NODES --env HUGGING_FACE_HUB_TOKEN --env ARGS
```

### Cluster management

#### SSH access

SkyPilot configures ssh configs for all cluster hosts that follow the naming convention `<cluster_name>[-worker<worker_index>]`.  E.g.:

```bash
ssh pc-dev
# or
ssh pc-dev-worker1 # pc-dev-worker2, etc.
```

For convenience, the [fabfile.py](fabfile.py) script provides a way to run arbitrary shell commands on all hosts in the cluster based on the generated SkyPilot ssh config (`~/.sky/generated/ssh/<cluster_name>`).  E.g.:

```bash
fab exec --cmd="nvidia-smi"
# ========== pc-dev ==========
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 570.148.08             Driver Version: 570.148.08     CUDA Version: 12.8     |
# |-----------------------------------------+------------------------+----------------------+
# ...
# ========== pc-dev-worker1 ==========
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 570.148.08             Driver Version: 570.148.08     CUDA Version: 12.8     |
# |-----------------------------------------+------------------------+----------------------+
# ...
```

#### Cluster info

Fabric can also be used to fetch important cluster info that isn't available via SkyPilot (or I can't find it in the docs/API).  Examples:

```bash
# Find the Ray address for the cluster
# - The Ray address is stored in a file in the cluster's temp/working directory
#   and can be retrieved with `cat /tmp/ray-pc-dev/ray_current_cluster` (that's what the fabric command does)
# - The SkyPilot command for this, `sky status --ip pc-dev`, could be better
#   if it returned the private IP rather than the public IP
# - The name RAY_GCS_ADDRESS is used so as not to clobber the RAY_ADDRESS
#   environment variable set for SkyPilot's internal ray cluster
export RAY_GCS_ADDRESS=$(fab ray-address)
echo $RAY_GCS_ADDRESS
# 10.19.109.245:6479

# Find all nodes in the cluster
fab cluster-nodes
# TODO: show output
```

#### Reinitialization

To reinitialize the existing Ray clusters (for both SkyPilot and Thalas), you can use the following:

```bash
# Stop Ray cluster-wide:
fab exec --cmd="source ~/sky_workdir/.venv/bin/activate && ray stop"

# Restart Ray cluster-wide
sky launch ... # i.e. same cluster initialization command as before
```

The `ray stop` command will stop both Ray clusters (it seems to blindly kill any processes associated with Ray), but issuing another `sky launch` command will bring them both back up, which is useful for debugging and experimenting with cluster configurations.

#### Adding dependencies

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

#### Caching dependencies

Wheels for `mamba-ssm` and `causal-conv1d` are often not available or simply fail to work once installed (depending on the platform).  This project forces builds from source for them, which is very slow.  These are steps previously taken to speed that up:

1. Find wheels on a Lambda host where wheels built correctly:

```bash
find ~/.cache/uv/ | grep mamba | grep whl
# /home/ubuntu/.cache/uv/sdists-v9/git/287e017c5a750cba/95d8aba8a8c75aed/mamba_ssm-2.2.4-cp312-cp312-linux_x86_64.whl
find ~/.cache/uv/ | grep causal | grep whl
# /home/ubuntu/.cache/uv/sdists-v9/git/287e017c5a750cba/95d8aba8a8c75aed/causal_conv1d-1.5.0.post8-cp312-cp312-linux_x86_64.whl
```

2. Copy the wheels to a release in a repo like https://github.com/Open-Athena/python-wheels/releases/tag/v0.0.1
3. Update `pyproject.toml` to use the new wheels if applicable, with fallback to source build

## Storage

Shared storage is currently supported via Hugging Face. See the [Hugging Face Filesystem API](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_file_system) docs for more details.

### Reading data

The simplest way to read remote data is through existing fsspec-compatible libraries (pandas, pyarrow, dask, xarray, etc.) or via [UPath](https://github.com/fsspec/universal_pathlib), an extension to `pathlib.Path` supporting remote file systems.

Here are a few examples:

```python
# Use fsspec-compatible libraries
import pandas as pd
df = pd.read_parquet("hf://datasets/openai/gsm8k/main/train-00000-of-00001.parquet")

# Use UPath
from upath import UPath; import gzip
path = UPath("hf://openai/gpt-oss-120b") / "config.json"
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

This project provides additional utilities on top of `RepoUrl` to support the separation of "internal" and "external" repositories.  This exists so that intermediate pipeline results can be stored in public HF repositories with a naming convention that separates them from final results.  Currently, that convention prefixes "_dev_" to the repository names.  Examples:

```python
from src.io.hf import hf_repo
import pandas as pd

repo = hf_repo(name="training_dataset", internal=False)
repo
# HfRepo(entity='plantcad', name='training_dataset', type='dataset', internal=False)
repo.to_url()
# 'hf://datasets/plantcad/training_dataset'

hf_repo(name="training_dataset", internal=True).to_url()
# 'hf://datasets/plantcad/_dev_training_dataset'
```

### Writing data

Writing data can be done through an fsspec filesystem or via UPath.  This will require authentication, so use `huggingface-cli login` or set `HUGGING_FACE_HUB_TOKEN` in the environment to do that.  Examples:

```python
from src.io.hf import HfPath, initialize_hf_path
from upath import UPath

# Create repo reference (uses "plantcad" as default entity)
path = HfPath.from_url("hf://datasets/plantcad/test-dataset")
path.to_url()
# 'hf://datasets/plantcad/test-dataset'

# Create the repository for the path on Hugging Face if it doesn't exist
initialize_hf_path(path)

# Write via UPath
content = "This is a test data file."
path.join("data.txt").write_text(content)

# Write with finer control through filesystem
fs = path.to_upath().fs
with fs.open(path.join("data.txt").to_url(), "w") as f:
    f.write(content)
```

### Deleting data

When running Thalas pipelines, it is common to need to clear the paths used for a pipeline during development, or to delete data for a specific step.  Here are some examples:

```bash
# Clear all data within the dataset (without deleting it entirely)
hf repo-files delete --repo-type dataset plantcad/_dev_pc2_eval '*'

# Clear all data for a specific step
hf repo-files delete --repo-type dataset plantcad/_dev_pc2_eval evolutionary_downsample_dataset-be132f
```

## AWS S3

Commands to use S3 on remote hosts:

```
# On local host
# Install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
aws configure
aws sso login --profile default
aws configure export-credentials --profile default --format env | source /dev/stdin

# Launch a cluster with AWS access
sky launch -c pc-dev configs/skypilot/cluster.sky.yaml --num-nodes 1 \
  --env HUGGING_FACE_HUB_TOKEN \
  --env AWS_ACCESS_KEY_ID \
  --env AWS_SECRET_ACCESS_KEY \
  --env AWS_SESSION_TOKEN
```

TODO: configure service accounts

## SkyPilot

### Environment variables

SkyPilot sets different environment variables about cluster nodes for the `setup` and `run` sections of YAML configs used typically by the `launch` and `exec` CLI commands (respectively).  See:

- [environment-variables.html#environment-variables-for-setup](https://docs.skypilot.co/en/v0.5.0/running-jobs/environment-variables.html#environment-variables-for-setup)
- [environment-variables.html#environment-variables-for-run](https://docs.skypilot.co/en/v0.5.0/running-jobs/environment-variables.html#environment-variables-for-run)

In this context, it is worth noting that the `SKYPILOT_NODE_IPS` list available to `run` does not contain all cluster hosts.  It contains only those allocated for the task according to `num_nodes`.  This means that the Ray head node cannot be determined from this variable if all nodes are not used by the task.  If a subset of nodes is used, then `SKYPILOT_NODE_RANK` may be 0 on a worker node (depending on what nodes were chosen).  This contradicts the behavior documented at https://docs.skypilot.co/en/v0.5.0/reference/yaml-spec.html, which clearly implies that the head node should always be included: "Number of nodes (optional; defaults to 1) to launch including the head node".

By contrast, `SKYPILOT_SETUP_NODE_IPS` contains all cluster hosts, so the Ray head node can always be determined from this variable.

## Development

Commands for development:

- Tests: `uv run pytest`
- Linting: `uv run pre-commit run --all-files`
- Type checking: `uv run pyrefly check --summarize-errors`
