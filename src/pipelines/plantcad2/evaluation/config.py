from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from thalas.execution import ExecutorMainConfig
from typing_extensions import Self

from src.utils.pipeline_utils import BaseStepConfig


class ModelType(StrEnum):
    """Model architecture type."""

    mlm = "mlm"
    clm = "clm"


class MotifInferenceMode(StrEnum):
    """Inference mode for motif prediction tasks (CLM only)."""

    fwd_only = "fwd_only"
    rc_only = "rc_only"
    fwd_rc_avg = "fwd_rc_avg"


@dataclass(kw_only=True)
class ModelConfig:
    """Model specification with architecture type."""

    path: str = Field(..., description="Model identifier")
    type: ModelType = Field(..., description="Model architecture type")
    context_length: int = Field(..., gt=0, description="Model context length in tokens")
    subfolder: str = Field(default="", description="Subfolder within model repo")
    name: str = Field(
        default="",
        description="Model display name (defaults to `path`)",
    )
    motif_inference_mode: MotifInferenceMode = Field(
        default=MotifInferenceMode.fwd_rc_avg,
        description="Inference mode for motif tasks (only relevant for CLM models)",
    )

    @model_validator(mode="before")
    @classmethod
    def set_default_name(cls, data: dict) -> dict:
        if not data.get("name"):
            data["name"] = data["path"]
        return data


# Use kw_only to support required and optional fields in any order
@dataclass(kw_only=True)
class TaskConfig(BaseStepConfig):
    """Base task execution config with common fields (all required)."""

    repo_id: str = Field(..., description="Hugging Face dataset repository identifier")
    task: str = Field(..., description="Dataset configuration name")
    split: str = Field(..., description="Dataset split name")
    seq_column: str = Field(..., description="Sequence column name")
    seq_length: int = Field(..., gt=0, description="Sequence length in dataset")
    label_column: str = Field(..., description="Label column name")
    num_workers: int | None = Field(
        ..., description="Number of Ray workers to launch (defaults to available GPUs)"
    )
    model_path: str = Field(..., description="Model identifier")
    model_type: ModelType = Field(
        ..., description="Model architecture type (mlm or clm)"
    )
    model_subfolder: str = Field(default="", description="Subfolder within model repo")
    model_context_length: int = Field(..., gt=0, description="Model context length")
    model_name: str = Field(..., description="Model name for identification")
    model_motif_inference_mode: MotifInferenceMode = Field(
        default=MotifInferenceMode.fwd_rc_avg,
        description="Inference mode for motif tasks (only relevant for CLM models)",
    )
    device: str = Field(..., description="Device for model execution")
    batch_size: int = Field(..., gt=0, description="Inference batch size")
    sample_rate: float | None = Field(default=None, description="Dataset sample rate")
    sample_max_size: int | None = Field(
        default=None, description="Maximum dataset size"
    )
    sample_seed: int = Field(default=0, description="Random seed for sampling")


@dataclass(kw_only=True)
class MultiMaskTaskConfig(TaskConfig):
    """Base configuration for tasks with multiple masked positions."""

    mask_token_indexes: list[int] = Field(
        default_factory=lambda: [4094, 4095, 4096],
        description="Indices to mask per sequence",
    )
    motif_len: int = Field(default=3, gt=0, description="Number of masked tokens")

    @model_validator(mode="after")
    def check_lengths(self) -> Self:
        if len(self.mask_token_indexes) != self.motif_len:
            raise ValueError("mask_token_indexes count must equal motif_len")
        return self


@dataclass(kw_only=True)
class EvoConsTaskConfig(MultiMaskTaskConfig):
    """Evolutionary constraint task configuration (single mask position)."""

    @model_validator(mode="after")
    def check_length(self) -> Self:
        if self.motif_len != 1:
            raise ValueError("EvoConsTaskConfig requires motif_len=1")
        return self

    @property
    def mask_token_index(self) -> int:
        """Single mask token position."""
        assert len(self.mask_token_indexes) == 1, (
            "EvoConsTaskConfig requires motif_len=1"
        )
        return self.mask_token_indexes[0]


@dataclass(kw_only=True)
class MotifTaskConfig(MultiMaskTaskConfig):
    """Motif recovery task configuration."""

    pass


@dataclass(kw_only=True)
class StructuralVariantTaskConfig(TaskConfig):
    """Structural variant effect prediction task configuration."""

    flanking: int = Field(default=5, gt=0, description="Flanking window size")


@dataclass(kw_only=True)
class CoreNonCoreTaskConfig(MultiMaskTaskConfig):
    """Core/noncore classification task configuration."""

    pass


@dataclass(kw_only=True)
class ComputeConfig:
    """Compute resource configuration."""

    device: str = Field(default="cuda", description="Device for model execution")
    batch_size: int = Field(default=128, gt=0, description="Inference batch size")
    num_workers: int | None = Field(
        default=None,
        description="Number of Ray workers to launch (defaults to available GPUs)",
    )


@dataclass(kw_only=True)
class SampleConfig:
    """Dataset sampling configuration."""

    rate: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description="Sample rate applied first (e.g., 0.1 for 10% of data)",
    )
    max_size: int | None = Field(
        default=None,
        gt=0,
        description="Maximum dataset size applied after rate (caps final size)",
    )
    seed: int = Field(default=0, description="Random seed for sampling")


@dataclass(kw_only=True)
class SplitConfig:
    """Specification of a task name and split combination with common defaults."""

    task: str = Field(..., description="Task configuration name")
    split: str = Field(..., description="Dataset split name")
    repo_id: str = Field(
        default="plantcad/PlantCAD2_zero_shot_tasks",
        description="Hugging Face dataset repository identifier",
    )
    seq_column: str = Field(default="sequence", description="Sequence column name")
    seq_length: int = Field(
        default=8192, gt=0, description="Sequence length in dataset"
    )
    label_column: str = Field(default="label", description="Label column name")
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific config overrides (e.g., mask_token_indexes, motif_len)",
    )


@dataclass(kw_only=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    executor: ExecutorMainConfig = Field(..., description="Executor configuration")
    models: list[ModelConfig] = Field(
        default_factory=list,
        description="Models to evaluate with their architecture types",
    )
    splits: list[SplitConfig] = Field(
        default_factory=list,
        description="Dataset splits to evaluate (task+split combinations)",
    )
    tasks: list[TaskConfig] = Field(
        default_factory=list,
        description="Task configurations (built from splits or specified directly)",
    )
    compute: ComputeConfig = Field(
        default_factory=ComputeConfig, description="Compute resource configuration"
    )
    sampling: SampleConfig | None = Field(
        default=None, description="Dataset sampling configuration"
    )
