"""Configuration models for PlantCAD2 evaluation pipeline."""

from pydantic import BaseModel, Field


class SlurmConfig(BaseModel):
    """Configuration for SLURM execution."""

    timeout_min: int = Field(default=5, description="Timeout in minutes")
    partition: str = Field(default="gg", description="SLURM partition")
    nodes: int = Field(default=1, description="Number of nodes")
    tasks_per_node: int = Field(default=1, description="Tasks per node")


class EvolutionaryConstraintConfig(BaseModel):
    """Configuration for evolutionary constraint evaluation."""

    model_path: str = Field(
        default="kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000",
        description="The path of pre-trained model",
    )
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
    device: str = Field(default="cuda:0", description="The device to run the model")
    batch_size: int = Field(default=128, description="The batch size for the model")
    token_idx: int = Field(default=255, description="The index of the nucleotide")
    repo_id: str = Field(
        default="kuleshov-group/cross-species-single-nucleotide-annotation",
        description="HuggingFace repository ID for the dataset",
    )
    output_dir: str = Field(
        default="data/evolutionary_constraint", description="The directory of output"
    )
    execution_mode: str = Field(
        default="local", description="Execution mode: 'local' or 'slurm'"
    )
    slurm_config: SlurmConfig = Field(
        default_factory=SlurmConfig, description="SLURM configuration"
    )


class SpliceAcceptorConfig(BaseModel):
    """Configuration for splice acceptor evaluation."""

    model_path: str = Field(
        default="kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000",
        description="The path of pre-trained model",
    )
    dataset_split: str = Field(
        default="valid",
        description="The name of the input dataset, either 'train', 'valid', or 'test'",
    )
    dataset_subdir: str = Field(
        default="Acceptor",
        description="Subdirectory within the HF dataset repo for this task",
    )
    sample_size: int | None = Field(
        default=None,
        description="Number of samples to downsample to (None for full dataset)",
    )
    device: str = Field(default="cuda:0", description="The device to run the model")
    batch_size: int = Field(default=128, description="The batch size for the model")
    token_idx: int = Field(default=255, description="The index of the nucleotide")
    repo_id: str = Field(
        default="kuleshov-group/cross-species-single-nucleotide-annotation",
        description="HuggingFace repository ID for the dataset",
    )
    output_dir: str = Field(
        default="data/splice_acceptor", description="The directory of output"
    )
    execution_mode: str = Field(
        default="local", description="Execution mode: 'local' or 'slurm'"
    )
    slurm_config: SlurmConfig = Field(
        default_factory=SlurmConfig, description="SLURM configuration"
    )


class TasksConfig(BaseModel):
    evolutionary_constraint: EvolutionaryConstraintConfig
    splice_acceptor: SpliceAcceptorConfig


class FlowConfig(BaseModel):
    tasks: TasksConfig
