import logging
import requests
import pandas as pd
import submitit
from pathlib import Path
from dataclasses import dataclass
from metaflow import FlowSpec, step, Parameter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Species:
    id: str
    gff_url: str
    gff_path: str


SPECIES = [
    Species(
        id="carabica",
        gff_url="https://drive.google.com/uc?export=download&id=1kXZjMRYCo724eFIAl7tjC5848Z2Z8jH6",
        gff_path="carabica.gene.gff3.gz",
    ),
    Species(
        id="pvulgaris",
        gff_url="https://drive.google.com/uc?export=download&id=1iC0nIBJVS9BTZW1XDijh9x1de74vJSq9",
        gff_path="pvulgaris.gene.gff3.gz",
    ),
]


@dataclass
class SlurmConfig:
    timeout_min: int = 5
    partition: str = "gg"
    nodes: int = 1
    tasks_per_node: int = 1


SLURM_CONFIG = SlurmConfig()


def calculate_chromosome_stats(df: pd.DataFrame) -> pd.Series:
    """Calculate chromosome sequence statistics from a GFF DataFrame."""
    return (
        df.groupby("seq_id")
        .size()
        .pipe(lambda s: s[s.index.to_series().str.lower().str.startswith("chr")])
    )


def compute_stats_task(dataset_path: str) -> pd.Series:
    """Compute statistics for a portion of the dataset in a SLURM job."""
    job_env = submitit.JobEnvironment()
    rank, world_size = job_env.global_rank, job_env.num_tasks
    print(f"Computing stats on rank {rank} of {world_size}")

    df = pd.read_parquet(dataset_path)
    # Select partition for this rank
    df_partition = df.iloc[rank::world_size]
    return calculate_chromosome_stats(df_partition)


def compute_dataset_stats(
    dataset_path: str,
    log_path: str,
    environment: str,
    nodes: int,
    slurm_config: SlurmConfig = SLURM_CONFIG,
) -> pd.Series:
    """Compute dataset statistics using specified execution environment."""
    if environment == "local":
        # Local execution - compute directly
        logger.info("Running statistics computation locally")
        df = pd.read_parquet(dataset_path)
        return calculate_chromosome_stats(df)

    # Distributed execution using submitit
    executor = submitit.AutoExecutor(
        folder=log_path,
        cluster=environment,
    )

    executor_params = {
        "timeout_min": slurm_config.timeout_min,
        "slurm_partition": slurm_config.partition,
        "tasks_per_node": slurm_config.tasks_per_node,
        "nodes": nodes,
    }
    executor.update_parameters(**executor_params)

    logger.info(f"Submitting {environment} job with parameters: {executor_params}")
    job = executor.submit(compute_stats_task, dataset_path)
    logger.info(f"Submitted job {job.job_id}, awaiting completion...")

    results = job.results()
    # Sum results across all ranks
    return pd.concat(results, axis=1).sum(axis=1)


class GFFPipelineFlow(FlowSpec):
    """A Metaflow pipeline that downloads and processes GFF files for multiple species."""

    output_path = Parameter(
        "output_path", help="Output directory for processed data", default="data"
    )

    environment = Parameter(
        "environment",
        help="Execution environment for compute-intensive steps (local or slurm)",
        default="local",
    )

    @step
    def start(self):
        """Initialize the pipeline with species configuration."""
        self.species_list = SPECIES

        # Setup directories
        self.output_dir = Path(self.output_path).resolve()
        self.log_dir = self.output_dir / "logs"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting pipeline with {len(self.species_list)} species")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"SLURM log directory: {self.log_dir}")

        self.next(self.download_gff_files, foreach="species_list")

    @step
    def download_gff_files(self):
        """Download GFF file for each species in parallel."""
        species = self.input

        logger.info(f"Downloading GFF file for species: {species.id}")

        # Create output path for this species
        file_path = self.output_dir / f"{species.id}.gene.gff3.gz"

        # Download the file
        try:
            response = requests.get(species.gff_url)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Successfully downloaded {species.id} to {file_path}")

            # Store the species info and file path for next step
            self.species_info = species
            self.gff_file_path = str(file_path)

        except Exception as e:
            logger.error(f"Failed to download {species.id}: {e}")
            raise

        self.next(self.join_downloads)

    @step
    def join_downloads(self, inputs):
        """Join the parallel download steps."""
        self.downloaded_files = []
        # see https://docs.metaflow.org/metaflow/basics#data-flow-through-the-graph
        self.merge_artifacts(inputs, include=["output_dir", "log_dir"])

        for input_step in inputs:
            self.downloaded_files.append(
                {
                    "species": input_step.species_info,
                    "file_path": input_step.gff_file_path,
                }
            )

        logger.info(f"Joined {len(self.downloaded_files)} downloaded files")
        self.next(self.process_gff_dataset)

    def _read_gff(self, file_path: str) -> pd.DataFrame:
        """Helper function to read GFF file."""
        GFF3_SPEC = [
            ("seq_id", str),
            ("source", str),
            ("type", str),
            ("start", int),
            ("end", int),
            ("score", float),
            ("strand", str),
            ("phase", pd.Int8Dtype()),
            ("attributes", str),
        ]
        GFF3_COLUMNS = [col for col, _ in GFF3_SPEC]
        GFF3_DTYPES = dict(GFF3_SPEC)

        return pd.read_csv(
            file_path,
            sep="\t",
            comment="#",
            names=GFF3_COLUMNS,
            na_values=".",
            dtype=GFF3_DTYPES,
        )

    @step
    def process_gff_dataset(self):
        """Combine all GFF files into a single dataset."""
        logger.info("Processing GFF dataset from downloaded files")

        dataframes = []
        for file_info in self.downloaded_files:
            df = self._read_gff(file_info["file_path"])
            df = df.assign(species_id=file_info["species"].id)
            dataframes.append(df)

        # Combine all dataframes
        self.gff_dataset = pd.concat(dataframes, ignore_index=True)

        # Save combined dataset
        self.dataset_path = self.output_dir / "gff_dataset.parquet"
        self.gff_dataset.to_parquet(self.dataset_path)

        logger.info(f"Created combined dataset with {len(self.gff_dataset)} rows")
        logger.info(f"Saved dataset to: {self.dataset_path}")

        self.next(self.compute_stats)

    @step
    def compute_stats(self):
        """Compute statistics on the GFF dataset using specified execution environment."""
        logger.info(
            f"Computing GFF dataset statistics using {self.environment} environment"
        )

        # Compute statistics using the specified environment
        self.stats = compute_dataset_stats(
            str(self.dataset_path),
            str(self.log_dir),
            self.environment,
            nodes=2,
        )

        # Read the dataset for additional statistics
        df = pd.read_parquet(self.dataset_path)
        self.row_count = len(df)
        self.species_count = df["species_id"].nunique()
        self.chromosome_count = len(self.stats)

        logger.info("Dataset statistics:")
        logger.info(f"  Total rows: {self.row_count}")
        logger.info(f"  Species count: {self.species_count}")
        logger.info(f"  Chromosome sequences: {self.chromosome_count}")

        self.next(self.end)

    @step
    def end(self):
        """Final step of the pipeline."""
        logger.info("Pipeline completed successfully!")
        logger.info(f"Final dataset saved to: {str(self.dataset_path)}")
        logger.info(f"Processed {self.row_count} total records")

        # Print top chromosome counts
        if len(self.stats) > 0:
            logger.info("Top 5 chromosome sequence counts:")
            for seq_id, count in self.stats.nlargest(5).items():
                logger.info(f"  {seq_id}: {count}")


if __name__ == "__main__":
    GFFPipelineFlow()
