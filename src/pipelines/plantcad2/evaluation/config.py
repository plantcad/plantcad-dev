"""Configuration models for PlantCAD2 evaluation pipeline."""

from pydantic import Field
from pydantic.dataclasses import dataclass
from typing import Any


@dataclass
class BaseStepConfig:
    """Base configuration for pipeline steps."""

    input_path: Any = Field(default=None, description="Input path for pipeline data")
    output_path: Any = Field(default=None, description="Output path for pipeline data")


@dataclass
class DownsampleDatasetConfig(BaseStepConfig):
    """Configuration for dataset downsampling step."""

    dataset_split: str = Field(
        default="valid",
        description="The name of the input dataset, either 'valid' or 'test'",
    )
    dataset_subdir: str = Field(
        default="Evolutionary_constraint",
        description="Subdirectory within the HF dataset repo for this task",
    )
    sample_size: int | None = Field(
        default=None,
        description="Number of samples to downsample to (None for full dataset)",
    )
    repo_id: str = Field(
        default="kuleshov-group/cross-species-single-nucleotide-annotation",
        description="HuggingFace repository ID for the dataset",
    )


@dataclass
class GenerateLogitsConfig(BaseStepConfig):
    """Configuration for logits generation step."""

    model_path: str = Field(
        default="kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000",
        description="The path of pre-trained model",
    )
    device: str = Field(default="cuda:0", description="The device to run the model")
    batch_size: int = Field(default=128, description="The batch size for the model")
    token_idx: int = Field(default=255, description="The index of the nucleotide")
    simulation_mode: bool = Field(
        default=True,
        description="Whether to use fake random logits for testing (simulation_mode=True) or real model inference",
    )


@dataclass
class GenerateScoresConfig(BaseStepConfig):
    """Configuration for scores generation step."""

    token_idx: int = Field(default=255, description="The index of the nucleotide")


@dataclass
class ComputeRocConfig(BaseStepConfig):
    """Configuration for ROC computation step."""

    # This step only needs the pipeline data from previous steps
    ...


@dataclass
class EvolutionaryConstraintStepsConfig:
    """Configuration for evolutionary constraint pipeline steps."""

    downsample_dataset: DownsampleDatasetConfig
    generate_logits: GenerateLogitsConfig
    generate_scores: GenerateScoresConfig
    compute_roc: ComputeRocConfig


@dataclass
class TasksConfig:
    evolutionary_constraint: EvolutionaryConstraintStepsConfig


@dataclass
class PipelineConfig:
    tasks: TasksConfig
    input_path: Any = Field(default=None, description="Input path for pipeline data")
    output_path: Any = Field(default=None, description="Output path for pipeline data")


@dataclass
class SlurmConfig:
    """Configuration for SLURM execution."""

    timeout_min: int = Field(default=5, description="Timeout in minutes")
    partition: str = Field(default="gg", description="SLURM partition")
    nodes: int = Field(default=1, description="Number of nodes")
    tasks_per_node: int = Field(default=1, description="Tasks per node")
