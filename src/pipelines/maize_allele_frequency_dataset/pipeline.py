"""Maize allele frequency dataset pipeline using thalas execution pattern."""

import logging
import multiprocessing
import numpy as np
import os
import shutil
import urllib.request
from dataclasses import replace
from pathlib import Path
import subprocess
from typing import Any, Callable

import docker
import polars as pl
from jinja2 import Template
from biofoundation.data import Genome
from upath import UPath
from thalas.execution import ExecutorStep, output_path_of, this_output_path

from huggingface_hub import HfApi
from src.pipelines.maize_allele_frequency_dataset.config import (
    MaizeAlleleFrequencyDatasetConfig,
    ProcessVariantsConfig,
    EnsemblVEPAnnotationConfig,
    GroupConsequencesConfig,
    AddGenomeRepeatsConfig,
    CreateAndPublishDatasetConfig,
)

logger = logging.getLogger("ray")

COORDINATES = ["chrom", "pos", "ref", "alt"]
NUCLEOTIDES = list("ACGT")

VARIANTS_FILENAME = "variants.parquet"
ANNOTATED_VARIANTS_FILENAME = "variants.annot.parquet"
GROUPED_VARIANTS_FILENAME = "variants.annot.grouped.parquet"
REPEAT_ANNOTATED_FILENAME = "variants.annot.grouped.add_is_repeat.parquet"


def process_variants(config: ProcessVariantsConfig) -> None:
    """Process raw variants data and compute allele frequencies."""
    logger.info(f"Starting process_variants step; {config=}")

    output_path = UPath(config.output_path)

    variants = (
        pl.read_parquet(
            config.raw_data_path,
            columns=["CHR", "POS", "REF", "ALT", "REF_FREQ", "ALT_FREQ"],
        )
        .rename(
            {
                "CHR": "chrom",
                "POS": "pos",
                "REF": "ref",
                "ALT": "alt",
                "REF_FREQ": "ref_count",
                "ALT_FREQ": "alt_count",
            }
        )
        .with_columns(
            pl.col("chrom").cast(str),
            pl.col("alt_count").alias("AC"),
            (pl.col("ref_count") + pl.col("alt_count")).alias("AN"),
        )
        .drop(["ref_count", "alt_count"])
        .with_columns((pl.col("AC") / pl.col("AN")).alias("AF"))
        .with_columns(
            pl.when(pl.col("AF") < 0.5)
            .then(pl.col("AF"))
            .otherwise(1 - pl.col("AF"))
            .alias("MAF")
        )
        .sort(COORDINATES)
    )

    variants_path = output_path / VARIANTS_FILENAME
    variants.write_parquet(variants_path)


def ensembl_vep_annotation(config: EnsemblVEPAnnotationConfig) -> None:
    """Download VEP cache, prepare input, run VEP, and process output."""
    logger.info(f"Starting ensembl_vep_annotation step; {config=}")

    variants_path = UPath(config.input_path) / VARIANTS_FILENAME
    output_path = UPath(config.output_path)

    logger.info("Downloading and extracting VEP cache...")
    cache_dir = output_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_filename = os.path.basename(config.cache_url)
    cache_path = cache_dir / cache_filename

    logger.info(f"Downloading VEP cache from {config.cache_url}")
    urllib.request.urlretrieve(config.cache_url, cache_path)

    logger.info(f"Extracting VEP cache to {cache_dir}")
    subprocess.run(["tar", "xzf", str(cache_path), "-C", str(cache_dir)], check=True)
    cache_path.unlink()

    logger.info("Preparing VEP input file...")
    df = pl.read_parquet(variants_path)
    df = df.with_columns(
        start=pl.col("pos"),
        end=pl.col("pos"),
        allele=pl.col("ref") + pl.lit("/") + pl.col("alt"),
        strand=pl.lit("+"),
    )

    vep_input_path = output_path / "variants.ensembl_vep.input.tsv"
    df.select(["chrom", "start", "end", "allele", "strand"]).write_csv(
        vep_input_path,
        separator="\t",
        include_header=False,
    )

    logger.info("Running Ensembl VEP...")
    vep_output_path = output_path / "variants.ensembl_vep.output.tsv.gz"

    client = docker.from_env()

    volumes = {
        str(output_path): {"bind": "/data", "mode": "rw"},
        str(cache_dir): {"bind": "/cache", "mode": "ro"},
    }

    # Run container as current user to avoid permission issues
    user = f"{os.getuid()}:{os.getgid()}"

    command = [
        "vep",
        "-i",
        f"/data/{vep_input_path.name}",
        "-o",
        f"/data/{vep_output_path.name}",
        "--fork",
        str(multiprocessing.cpu_count()),
        "--cache",
        "--dir_cache",
        "/cache",
        "--format",
        "ensembl",
        "--species",
        config.species,
        "--most_severe",
        "--compress_output",
        "gzip",
        "--tab",
        "--distance",
        str(config.distance),
        "--offline",
        "--cache_version",
        str(config.cache_version),
        "--no_stats",
    ]

    logger.info(f"Running VEP with command: {' '.join(command)}")
    client.containers.run(
        config.docker_image,
        command=command,
        volumes=volumes,
        user=user,
        remove=True,
    )
    logger.info("VEP completed successfully")

    logger.info("Processing VEP output...")
    V = pl.read_parquet(variants_path)

    V2 = pl.read_csv(
        vep_output_path,
        separator="\t",
        has_header=False,
        comment_prefix="#",
        columns=[0, 6],
        new_columns=["variant", "consequence"],
    )

    V2 = V2.with_columns(
        chrom=pl.col("variant").str.split("_").list.get(0),
        pos=pl.col("variant").str.split("_").list.get(1).cast(pl.Int64),
        ref=pl.col("variant").str.split("_").list.get(2).str.split("/").list.get(0),
        alt=pl.col("variant").str.split("_").list.get(2).str.split("/").list.get(1),
    ).drop("variant")

    V = V.join(V2, on=COORDINATES, how="left", maintain_order="left")

    annotated_path = output_path / ANNOTATED_VARIANTS_FILENAME
    V.write_parquet(annotated_path)

    logger.info("Cleaning up temporary VEP files...")
    vep_input_path.unlink()
    logger.info(f"Deleted {vep_input_path}")
    vep_output_path.unlink()
    logger.info(f"Deleted {vep_output_path}")

    logger.info("Cleaning up VEP cache...")
    shutil.rmtree(cache_dir)
    logger.info(f"Deleted cache directory {cache_dir}")


def group_consequences(config: GroupConsequencesConfig) -> None:
    """Group consequences according to configuration."""
    logger.info(f"Starting group_consequences step; {config=}")

    V = pl.read_parquet(
        UPath(config.input_path) / ANNOTATED_VARIANTS_FILENAME
    ).with_columns(original_consequence=pl.col("consequence"))
    for new_c, old_cs in config.groups.items():
        V = V.with_columns(
            pl.when(pl.col("consequence").is_in(old_cs))
            .then(pl.lit(new_c))
            .otherwise(pl.col("consequence"))
            .alias("consequence")
        )
    V.write_parquet(UPath(config.output_path) / GROUPED_VARIANTS_FILENAME)


def add_genome_repeats(config: AddGenomeRepeatsConfig) -> None:
    """Download genome and add repeat annotations."""
    logger.info(f"Starting add_genome_repeats step; {config=}")

    output_path = UPath(config.output_path)

    logger.info("Downloading genome...")
    genome_path = output_path / "genome.fa.gz"
    urllib.request.urlretrieve(config.genome_url, genome_path)

    logger.info("Adding repeat annotations...")
    V = pl.read_parquet(UPath(config.input_path) / GROUPED_VARIANTS_FILENAME)
    genome = Genome(genome_path)
    V = V.with_columns(
        is_repeat=np.array(
            [
                genome(v["chrom"], v["pos"] - 1, v["pos"]).islower()
                for v in V.iter_rows(named=True)
            ]
        )
    )
    V.write_parquet(output_path / REPEAT_ANNOTATED_FILENAME)

    logger.info("Cleaning up genome file...")
    genome_path.unlink()
    logger.info(f"Deleted {genome_path}")


def _create_full_subset(V: pl.DataFrame) -> pl.DataFrame:
    """Create full subset (no subsampling).

    Parameters
    ----------
    V
        Pre-filtered dataframe for the split

    Returns
    -------
    pl.DataFrame
        The dataframe unchanged
    """
    return V


def _create_balanced_subset(
    V: pl.DataFrame,
    max_n: int,
    subsample_consequences: list[str],
    subsample_seed: int,
) -> pl.DataFrame:
    """Create balanced subset by sampling equally across consequence types.

    Parameters
    ----------
    V
        Pre-filtered dataframe for the split
    max_n
        Maximum total number of variants to sample
    subsample_consequences
        List of consequence types to include in balanced sample
    subsample_seed
        Random seed for sampling

    Returns
    -------
    pl.DataFrame
        Balanced subsampled dataframe sorted by coordinates
    """
    max_n_per_consequence = max_n // len(subsample_consequences)
    V = V.filter(~pl.col("is_repeat"))
    res = []
    for consequence in subsample_consequences:
        V_consequence = V.filter(pl.col("consequence") == consequence)
        n = min(max_n_per_consequence, len(V_consequence))
        res.append(V_consequence.sample(n=n, shuffle=True, seed=subsample_seed))
    return pl.concat(res).sort(COORDINATES)


def _create_consequence_subset(
    V: pl.DataFrame,
    consequence: str,
    max_n: int,
    subsample_seed: int,
) -> pl.DataFrame:
    """Create subsampled subset for a specific consequence type.

    Parameters
    ----------
    V
        Pre-filtered dataframe for the split
    consequence
        Specific consequence type to filter by
    max_n
        Maximum number of variants to sample
    subsample_seed
        Random seed for sampling

    Returns
    -------
    pl.DataFrame
        Subsampled dataframe filtered by consequence type, sorted by coordinates
    """
    V = V.filter(~pl.col("is_repeat"), pl.col("consequence") == consequence)
    n = min(max_n, len(V))
    return V.sample(n=n, shuffle=True, seed=subsample_seed).sort(COORDINATES)


def _write_subset(
    dataset_output_path: UPath,
    dataset_config: str,
    split: str,
    subset_fn: Callable[..., pl.DataFrame],
    *args: Any,
) -> None:
    """Create a subset and write it to a parquet file.

    Parameters
    ----------
    dataset_output_path
        Base output path for datasets
    dataset_config
        Name of the dataset configuration (e.g., "full", "10k_balanced")
    split
        Name of the split (e.g., "validation", "test")
    subset_fn
        Function to create the subset (e.g., _create_full_subset, _create_balanced_subset)
    *args
        Arguments to pass to subset_fn
    """
    split_data = subset_fn(*args)
    split_path = dataset_output_path / dataset_config / f"{split}.parquet"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_data.write_parquet(split_path)


def create_and_publish_dataset(config: CreateAndPublishDatasetConfig) -> None:
    """Create dataset configs, generate README, and upload to HuggingFace."""
    logger.info(f"Starting create_and_publish_dataset step; {config=}")

    repeat_path = UPath(config.input_path) / REPEAT_ANNOTATED_FILENAME
    output_path = UPath(config.output_path)
    dataset_output_path = output_path / "dataset"
    dataset_output_path.mkdir(parents=True, exist_ok=True)

    V = pl.read_parquet(repeat_path)
    splits = list(config.split_chroms.keys())

    # Pre-compute split dataframes once to avoid redundant filtering
    logger.info("Pre-filtering variants by split chromosomes...")
    split_dataframes = {
        split: V.filter(pl.col("chrom").is_in(config.split_chroms[split]))
        for split in splits
    }

    # Build dataset_configs list for README generation
    dataset_configs = ["full"]
    for max_n_key in config.max_n.keys():
        dataset_configs.append(f"{max_n_key}_balanced")
        for consequence in config.subsample_consequences:
            dataset_configs.append(f"{max_n_key}_{consequence}")

    logger.info("Creating dataset configurations...")
    for split in splits:
        # Create "full" dataset_config
        _write_subset(
            dataset_output_path,
            "full",
            split,
            _create_full_subset,
            split_dataframes[split],
        )

        # Create subsampled dataset_configs
        for max_n_key in config.max_n.keys():
            max_n = config.max_n[max_n_key]

            # Create "balanced" dataset_config
            _write_subset(
                dataset_output_path,
                f"{max_n_key}_balanced",
                split,
                _create_balanced_subset,
                split_dataframes[split],
                max_n,
                config.subsample_consequences,
                config.subsample_seed,
            )

            # Create consequence dataset_configs
            for consequence in config.subsample_consequences:
                _write_subset(
                    dataset_output_path,
                    f"{max_n_key}_{consequence}",
                    split,
                    _create_consequence_subset,
                    split_dataframes[split],
                    consequence,
                    max_n,
                    config.subsample_seed,
                )

    logger.info("Creating README...")
    readme_path = dataset_output_path / "README.md"

    frontmatter_configs = []
    for dataset_config in dataset_configs:
        config_entry: dict[str, Any] = {
            "config_name": dataset_config,
            "data_files": [
                {"split": split, "path": f"{dataset_config}/{split}.parquet"}
                for split in splits
            ],
        }
        if dataset_config == config.default_dataset_config:
            config_entry["default"] = True
        frontmatter_configs.append(config_entry)

    template_path = Path(__file__).parent / "README.jinja"
    with open(template_path, "r") as f:
        template_str = f.read()
    template = Template(template_str)

    rendered = template.render(
        license="apache-2.0",
        tags=["biology", "genomics", "dna", "variant-effect-prediction"],
        configs=frontmatter_configs,
        split_chroms=config.split_chroms,
        num_consequences=len(config.subsample_consequences),
        max_n_keys=list(config.max_n.keys()),
        consequences=config.subsample_consequences,
    )

    with open(readme_path, "w") as f:
        f.write(rendered)

    logger.info("Uploading to HuggingFace...")
    api = HfApi()
    api.upload_large_folder(
        repo_id=config.output_hf_path,
        repo_type="dataset",
        folder_path=str(dataset_output_path),
    )


class MaizeAlleleFrequencyDatasetPipeline:
    """Pipeline class for maize allele frequency dataset creation."""

    def __init__(self, config: MaizeAlleleFrequencyDatasetConfig):
        """Initialize the pipeline.

        Parameters
        ----------
        config
            Pipeline configuration containing step configs
        """
        self.config = config
        logger.info(f"MaizeAlleleFrequencyDatasetPipeline config: {self.config}")

    def process_variants(self) -> ExecutorStep:
        """Process raw variants data and compute allele frequencies."""
        return ExecutorStep(
            name="process_variants",
            fn=process_variants,
            config=replace(
                self.config.process_variants, output_path=this_output_path()
            ),
            description="Process raw variants and compute allele frequencies",
        )

    def ensembl_vep_annotation(self) -> ExecutorStep:
        """Annotate variants with Ensembl VEP."""
        return ExecutorStep(
            name="ensembl_vep_annotation",
            fn=ensembl_vep_annotation,
            config=replace(
                self.config.ensembl_vep_annotation,
                input_path=output_path_of(self.process_variants()),
                output_path=this_output_path(),
            ),
            description="Annotate variants with Ensembl VEP",
        )

    def group_consequences(self) -> ExecutorStep:
        """Group consequences according to configuration."""
        return ExecutorStep(
            name="group_consequences",
            fn=group_consequences,
            config=replace(
                self.config.group_consequences,
                input_path=output_path_of(self.ensembl_vep_annotation()),
                output_path=this_output_path(),
            ),
            description="Group consequence types",
        )

    def add_genome_repeats(self) -> ExecutorStep:
        """Add genome and repeat annotations."""
        return ExecutorStep(
            name="add_genome_repeats",
            fn=add_genome_repeats,
            config=replace(
                self.config.add_genome_repeats,
                input_path=output_path_of(self.group_consequences()),
                output_path=this_output_path(),
            ),
            description="Add genome and repeat annotations",
        )

    def create_and_publish_dataset(self) -> ExecutorStep:
        """Create dataset configs and publish to HuggingFace."""
        return ExecutorStep(
            name="create_and_publish_dataset",
            fn=create_and_publish_dataset,
            config=replace(
                self.config.create_and_publish_dataset,
                input_path=output_path_of(self.add_genome_repeats()),
                output_path=this_output_path(),
            ),
            description="Create dataset configs and publish to HuggingFace",
        )

    def last_step(self) -> ExecutorStep:
        """Return the final step in the pipeline."""
        return self.create_and_publish_dataset()


def main():
    """Main entry point for the maize allele frequency dataset pipeline."""
    import draccus
    from src.utils.logging_utils import filter_known_warnings, initialize_logging
    from src.exec import executor_main
    from src.io.hf import initialize_hf_path

    initialize_logging()
    filter_known_warnings()

    logger.info("Starting maize allele frequency dataset pipeline")

    cfg = draccus.parse(config_class=MaizeAlleleFrequencyDatasetConfig)

    if cfg.executor.prefix is None:
        raise ValueError("Executor prefix must be set")
    # Only initialize HF path if the prefix is an HF path
    prefix_path = UPath(cfg.executor.prefix)
    if prefix_path.protocol == "hf":
        initialize_hf_path(cfg.executor.prefix)

    pipeline = MaizeAlleleFrequencyDatasetPipeline(cfg)
    step = pipeline.last_step()
    executor_main(cfg.executor, [step], init_logging=False)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
