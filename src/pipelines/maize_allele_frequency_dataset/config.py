"""Configuration models for maize allele frequency dataset pipeline."""

from pydantic import Field
from pydantic.dataclasses import dataclass
from thalas.execution import ExecutorMainConfig

from src.utils.pipeline_utils import BaseStepConfig


@dataclass
class ProcessVariantsConfig(BaseStepConfig):
    """Configuration for processing raw variants."""

    raw_data_path: str = Field(default=..., description="Path to raw data")


@dataclass
class EnsemblVEPAnnotationConfig(BaseStepConfig):
    """Configuration for Ensembl VEP annotation (download cache + make input + run VEP + process output)."""

    cache_url: str = Field(default=..., description="URL to download VEP cache")
    cache_version: int = Field(default=..., description="VEP cache version")
    species: str = Field(default=..., description="Species name for VEP")
    distance: int = Field(
        default=1000,
        description="Distance up/downstream between a variant and a transcript for which Ensembl VEP will assign upstream_gene_variant or downstream_gene_variant consequences",
    )
    docker_image: str = Field(
        default="ensemblorg/ensembl-vep:release_115.1",
        description="Docker image for VEP",
    )


@dataclass
class GroupConsequencesConfig(BaseStepConfig):
    """Configuration for grouping consequences."""

    groups: dict[str, list[str]] = Field(
        default=..., description="Consequence grouping rules"
    )
    include: list[str] = Field(
        default=..., description="List of consequence types to include"
    )


@dataclass
class AddGenomeRepeatsConfig(BaseStepConfig):
    """Configuration for downloading genome and adding repeat annotations."""

    genome_url: str = Field(default=..., description="URL to download genome FASTA")


@dataclass
class CreateAndPublishDatasetConfig(BaseStepConfig):
    """Configuration for creating dataset configs, README, and uploading to HuggingFace."""

    split_chroms: dict[str, list[str]] = Field(
        default=..., description="Chromosomes for each split"
    )
    max_n: dict[str, int] = Field(
        default=..., description="Maximum number of variants per sample size"
    )
    include: list[str] = Field(
        default=..., description="List of consequence types to include"
    )
    groups: dict[str, list[str]] = Field(
        default=..., description="Consequence grouping rules"
    )
    default_config: str = Field(default=..., description="Default configuration name")
    output_hf_path: str = Field(default=..., description="HuggingFace output path")
    seed: int = Field(default=42, description="Random seed for subsampling")


@dataclass
class MaizeAlleleFrequencyDatasetConfig:
    """Configuration for maize allele frequency dataset pipeline."""

    process_variants: ProcessVariantsConfig
    ensembl_vep_annotation: EnsemblVEPAnnotationConfig
    group_consequences: GroupConsequencesConfig
    add_genome_repeats: AddGenomeRepeatsConfig
    create_and_publish_dataset: CreateAndPublishDatasetConfig
    executor: ExecutorMainConfig
