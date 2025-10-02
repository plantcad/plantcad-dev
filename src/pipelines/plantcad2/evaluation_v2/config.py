"""Configuration models for the evaluation_v2 pipeline."""

from __future__ import annotations

from typing import Sequence

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from thalas.execution import ExecutorMainConfig
from typing_extensions import Self

from src.utils.pipeline_utils import BaseStepConfig


@dataclass
class BaseZeroShotConfig(BaseStepConfig):
    repo_id: str = Field(description="Hugging Face dataset repository identifier")
    task: str = Field(description="Dataset configuration name")
    split: str = Field(default="valid", description="Dataset split to evaluate")
    seq_column: str = Field(default="sequence", description="Sequence column name")


@dataclass
class DistributedZeroShotConfig(BaseZeroShotConfig):
    num_workers: int | None = Field(
        default=None,
        ge=1,
        description="Number of Ray workers to launch (defaults to available GPUs)",
    )
    gpus_per_worker: float = Field(
        default=1.0, gt=0, description="Number of GPUs to reserve per Ray worker"
    )


@dataclass
class MaskedLMConfig(DistributedZeroShotConfig):
    model: str = Field(description="Masked language model identifier")
    device: str = Field(default="cuda:0", description="Device for model execution")
    batch_size: int = Field(default=128, gt=0, description="Inference batch size")


@dataclass
class EvoConsProbsConfig(MaskedLMConfig):
    token_idx: int = Field(default=255, ge=0, description="Masked position index")


@dataclass
class EvoConsEvalConfig(BaseZeroShotConfig):
    label_column: str = Field(default="label", description="Label column name")
    token_idx: int = Field(default=255, ge=0, description="Masked position index")


@dataclass
class EvoConsTaskConfig:
    compute_probs: EvoConsProbsConfig
    evaluate: EvoConsEvalConfig


@dataclass
class MotifProbsConfig(MaskedLMConfig):
    mask_idx: Sequence[int] = Field(
        default=(255, 256, 257), description="Indices to mask per sequence"
    )
    motif_len: int = Field(default=3, gt=0, description="Number of masked tokens")

    @model_validator(mode="after")
    def check_lengths(self) -> Self:
        if len(tuple(self.mask_idx)) != self.motif_len:
            raise ValueError("mask_idx count must equal motif_len")
        return self


@dataclass
class MotifEvalConfig(BaseZeroShotConfig):
    mask_idx: Sequence[int] = Field(
        default=(255, 256, 257), description="Indices to mask per sequence"
    )
    motif_len: int = Field(default=3, gt=0, description="Number of masked tokens")


@dataclass
class MotifTaskConfig:
    compute_probs: MotifProbsConfig
    evaluate: MotifEvalConfig


@dataclass
class StructuralVariantScoreConfig(MaskedLMConfig):
    flanking: int = Field(default=5, gt=0, description="Flanking window size")


@dataclass
class StructuralVariantEvalConfig(BaseZeroShotConfig):
    label_column: str = Field(default="label", description="Label column name")


@dataclass
class StructuralVariantTaskConfig:
    compute_scores: StructuralVariantScoreConfig
    evaluate: StructuralVariantEvalConfig


@dataclass
class CoreNonCoreScoreConfig(MaskedLMConfig):
    mask_idx: Sequence[int] = Field(
        default=(255, 256, 257), description="Indices to mask per sequence"
    )
    motif_len: int = Field(default=3, gt=0, description="Number of masked tokens")
    label_column: str = Field(default="label", description="Label column name")

    @model_validator(mode="after")
    def check_lengths(self) -> Self:
        if len(tuple(self.mask_idx)) != self.motif_len:
            raise ValueError("mask_idx count must equal motif_len")
        return self


@dataclass
class CoreNonCoreEvalConfig(BaseZeroShotConfig):
    label_column: str = Field(default="label", description="Label column name")
    mask_idx: Sequence[int] = Field(
        default=(255, 256, 257), description="Indices to mask per sequence"
    )
    motif_len: int = Field(default=3, gt=0, description="Number of masked tokens")


@dataclass
class CoreNonCoreTaskConfig:
    compute_scores: CoreNonCoreScoreConfig
    evaluate: CoreNonCoreEvalConfig


@dataclass
class TasksConfig:
    evo_cons: EvoConsTaskConfig
    motif: MotifTaskConfig
    sv_effect: StructuralVariantTaskConfig
    core_noncore: CoreNonCoreTaskConfig


@dataclass
class PipelineConfig:
    tasks: TasksConfig
    executor: ExecutorMainConfig
