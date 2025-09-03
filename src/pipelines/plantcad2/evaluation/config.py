"""Configuration models for PlantCAD2 evaluation pipeline."""

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from typing import Any
from typing_extensions import Self
from thalas.execution import ExecutorMainConfig

from src.utils.pipeline import BaseStepConfig


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
    dataset_path: str = Field(
        default="kuleshov-group/cross-species-single-nucleotide-annotation",
        description="HuggingFace repository ID for the dataset",
    )


@dataclass
class GenerateLogitsConfig(BaseStepConfig):
    """Configuration for logits generation step."""

    model_path: str = Field(
        default="kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000",
        description="The path of the pre-trained model to use",
    )

    device: str = Field(default="cuda:0", description="The device to run the model")
    batch_size: int = Field(
        default=128, description="The batch size for the model", gt=0
    )
    token_idx: int = Field(default=255, description="The index of the nucleotide", ge=0)
    simulation_mode: bool = Field(
        default=True,
        description="Whether to use fake random logits for testing (simulation_mode=True) or real model inference",
    )
    num_workers: int = Field(
        default=1, description="The number of workers to use for Ray"
    )


@dataclass
class GenerateScoresConfig(BaseStepConfig):
    """Configuration for scores generation step."""

    dataset_input_path: Any = Field(
        default=None, description="Output path from dataset downsampling step"
    )


@dataclass
class ComputeRocConfig(BaseStepConfig):
    """Configuration for ROC computation step."""

    ...


@dataclass
class EvolutionaryConstraintConfig:
    """Configuration for evolutionary constraint pipeline steps."""

    downsample_dataset: DownsampleDatasetConfig
    generate_logits: GenerateLogitsConfig
    generate_scores: GenerateScoresConfig
    compute_roc: ComputeRocConfig


@dataclass
class TasksConfig:
    evolutionary_constraint: EvolutionaryConstraintConfig


@dataclass
class ModelConfig:
    """Configuration for a model."""

    path: str = Field(description="Path or identifier for the model")
    max_sequence_length: int = Field(
        description="Maximum sequence length the model can handle"
    )


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    path: str = Field(description="Path or identifier for the dataset")
    sequence_length: int = Field(description="Expected sequence length for the dataset")


@dataclass
class PipelineConfig:
    tasks: TasksConfig
    datasets: list[DatasetConfig]
    models: list[ModelConfig]
    executor: ExecutorMainConfig

    def get_dataset_config(self, path: str) -> DatasetConfig:
        """Get dataset configuration by path."""
        for dataset in self.datasets:
            if dataset.path == path:
                return dataset
        raise ValueError(
            f"Dataset {path!r} not found in datasets config; "
            f"available datasets: {[d.path for d in self.datasets]}"
        )

    def get_model_config(self, path: str) -> ModelConfig:
        """Get model configuration by path."""
        for model in self.models:
            if model.path == path:
                return model
        raise ValueError(
            f"Model {path!r} not found in models config; "
            f"available models: {[m.path for m in self.models]}"
        )

    @model_validator(mode="after")
    def check_dataset_path(self) -> Self:
        """Check that the dataset path is present in the datasets config."""
        dataset_path = (
            self.tasks.evolutionary_constraint.downsample_dataset.dataset_path
        )
        self.get_dataset_config(dataset_path)  # Will raise ValueError if not found
        return self

    @model_validator(mode="after")
    def check_model_path(self) -> Self:
        """Check that the model path is present in the models config."""
        model_path = self.tasks.evolutionary_constraint.generate_logits.model_path
        self.get_model_config(model_path)  # Will raise ValueError if not found
        return self

    @model_validator(mode="after")
    def check_token_idx(self) -> Self:
        """Check that the token index is within both dataset sequence length and model max sequence length."""
        dataset_path = (
            self.tasks.evolutionary_constraint.downsample_dataset.dataset_path
        )
        model_path = self.tasks.evolutionary_constraint.generate_logits.model_path
        token_idx = self.tasks.evolutionary_constraint.generate_logits.token_idx

        # Get dataset and model configs
        dataset_config = self.get_dataset_config(dataset_path)
        model_config = self.get_model_config(model_path)

        # Check against dataset sequence length
        if token_idx >= dataset_config.sequence_length:
            raise ValueError(
                f"Token index {token_idx} is out of range for dataset {dataset_path!r}; "
                f"must be less than dataset sequence length of {dataset_config.sequence_length}"
            )

        # Check against model max sequence length
        if token_idx >= model_config.max_sequence_length:
            raise ValueError(
                f"Token index {token_idx} is out of range for model {model_path!r}; "
                f"must be less than model max sequence length of {model_config.max_sequence_length}"
            )
        return self


@dataclass
class SlurmConfig:
    """Configuration for SLURM execution."""

    timeout_min: int = Field(default=5, description="Timeout in minutes")
    partition: str = Field(default="gg", description="SLURM partition")
    nodes: int = Field(default=1, description="Number of nodes")
    tasks_per_node: int = Field(default=1, description="Tasks per node")
