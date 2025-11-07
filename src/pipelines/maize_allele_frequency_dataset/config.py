"""Configuration models for maize allele frequency dataset pipeline."""

from pydantic import Field
from pydantic.dataclasses import dataclass
from thalas.execution import ExecutorMainConfig

from src.utils.pipeline_utils import BaseStepConfig


@dataclass(kw_only=True)
class ProcessVariantsConfig(BaseStepConfig):
    """Configuration for processing raw variants."""

    raw_data_path: str = Field(description="Path to raw data")


@dataclass(kw_only=True)
class EnsemblVEPAnnotationConfig(BaseStepConfig):
    """Configuration for Ensembl VEP annotation (download cache + make input + run VEP + process output)."""

    cache_url: str = Field(description="URL to download VEP cache")
    cache_version: int = Field(description="VEP cache version")
    species: str = Field(description="Species name for VEP")
    docker_image: str = Field(description="Docker image for VEP")
    distance: int = Field(
        default=1000,
        description="Distance up/downstream between a variant and a transcript for which Ensembl VEP will assign upstream_gene_variant or downstream_gene_variant consequences",
    )


@dataclass(kw_only=True)
class GroupConsequencesConfig(BaseStepConfig):
    """Configuration for grouping consequences."""

    groups: dict[str, list[str]] = Field(description="Consequence grouping rules")


@dataclass(kw_only=True)
class AddGenomeRepeatsConfig(BaseStepConfig):
    """Configuration for downloading genome and adding repeat annotations."""

    genome_url: str = Field(description="URL to download genome FASTA")


@dataclass(kw_only=True)
class CreateAndPublishDatasetConfig(BaseStepConfig):
    """Configuration for creating dataset configs, README, and uploading to HuggingFace."""

    split_chroms: dict[str, list[str]] = Field(description="Chromosomes for each split")
    max_n: dict[str, int] = Field(
        description="Maximum number of variants per sample size"
    )
    include: list[str] = Field(description="List of consequence types to include")
    groups: dict[str, list[str]] = Field(description="Consequence grouping rules")
    default_config: str = Field(description="Default configuration name")
    output_hf_path: str = Field(description="HuggingFace output path")
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
