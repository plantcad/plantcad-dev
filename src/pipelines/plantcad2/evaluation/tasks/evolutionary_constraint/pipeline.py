"""Evolutionary constraint evaluation pipeline using thalas execution pattern."""

import logging
from dataclasses import replace
import pandas as pd
import ray
import xarray as xr
from upath import UPath
from thalas.execution import ExecutorStep, output_path_of, this_output_path

from src.pipelines.plantcad2.evaluation.config import (
    PipelineConfig,
    DownsampleDatasetConfig,
    GenerateLogitsConfig,
    GenerateScoresConfig,
    ComputeRocConfig,
)
from src.pipelines.plantcad2.evaluation.common import (
    compute_plantcad_scores,
    compute_roc_auc,
    load_and_downsample_dataset,
    generate_model_logits,
)
from src.utils.pipeline import AsyncLock, save_step_json, load_step_json

logger = logging.getLogger("ray")


def downsample_dataset(config: DownsampleDatasetConfig) -> None:
    """Load and downsample the HuggingFace dataset."""
    logger.info(f"Starting dataset downsampling task; {config=}")

    # Setup dataset directory using executor-managed output path
    output_path = UPath(config.output_path)

    # Load and downsample dataset
    dataset_path, num_samples = load_and_downsample_dataset(
        dataset_path=config.dataset_path,
        dataset_subdir=config.dataset_subdir,
        dataset_split=config.dataset_split,
        dataset_dir=output_path / "dataset",
        sample_size=config.sample_size,
    )

    # Save step data
    step_data = {
        "num_samples": num_samples,
        "dataset_filename": dataset_path.name,
        "dataset_relpath": dataset_path.relative_to(output_path).path,
    }
    save_step_json(output_path, step_data)


def generate_logits(config: GenerateLogitsConfig) -> None:
    """Generate logits using either fake random data or real model inference."""
    logger.info(f"Starting logits generation task; {config=}")

    # Load step data from previous step
    input_step = load_step_json(UPath(config.input_path))

    # Construct dataset path from the previous step's output
    dataset_path = UPath(config.input_path) / input_step["dataset_relpath"]

    output_path = UPath(config.output_path)

    # Set output path for this step
    logits_output_dir = output_path / "logits"

    # Use the new generate_model_logits function with simulation_mode
    if config.simulation_mode:
        logits_path = generate_model_logits(
            dataset_path=dataset_path,
            output_dir=logits_output_dir,
            model_path=config.model_path,
            device=config.device,
            token_idx=config.token_idx,
            batch_size=config.batch_size,
            simulation_mode=config.simulation_mode,
        )
    else:
        # Wrap for distributed processing
        remote_generate_logits = ray.remote(num_gpus=1)(generate_model_logits)

        # Create lock for HF writes
        lock = AsyncLock.remote()

        # Submit tasks for all workers
        futures = []
        for worker_id in range(config.num_workers):
            future = remote_generate_logits.remote(
                dataset_path=dataset_path,
                output_dir=logits_output_dir,
                model_path=config.model_path,
                device=config.device,
                token_idx=config.token_idx,
                batch_size=config.batch_size,
                simulation_mode=config.simulation_mode,
                worker_id=worker_id,
                num_workers=config.num_workers,
                lock=lock,
            )
            futures.append(future)

        # Wait for all tasks to complete and get file paths
        results = ray.get(futures)

        # Combine all worker output files into single logits.zarr
        worker_logits = []

        # Load and combine logits from all worker files
        for worker_file in results:
            worker_logits.append(xr.open_zarr(worker_file))

        if worker_logits:
            combined_logits = xr.concat(worker_logits, dim="sample")
            logits_path = logits_output_dir / "logits.zarr"
            combined_logits.to_zarr(
                logits_path, zarr_format=2, consolidated=True, mode="w"
            )
            logger.info(
                f"Combined {len(worker_logits)} worker outputs into {logits_path}"
            )
        else:
            raise ValueError("No worker outputs found to combine")

    # Save logits output location and token index for next step
    output_step = {
        "logits_relpath": logits_path.relative_to(output_path).path,
        "token_idx": config.token_idx,
    }

    # Save step data
    save_step_json(output_path, output_step)


def generate_scores(config: GenerateScoresConfig) -> None:
    """Generate scores for plantcad evaluation."""
    logger.info(f"Starting scores generation task; {config=}")

    # Load step data from previous step
    input_step = load_step_json(UPath(config.input_path))

    output_path = UPath(config.output_path)

    # Construct logits path from the previous step's output
    logits_path = UPath(config.input_path) / input_step["logits_relpath"]
    logits = xr.open_zarr(logits_path)

    token_idx = input_step["token_idx"]

    predictions = compute_plantcad_scores(
        logits=logits,
        token_idx=token_idx,
    )

    # Save prediction data as parquet
    predictions.to_parquet(output_path / "predictions.parquet")

    # Save predictions file location for ROC computation
    output_step = {
        "predictions_relpath": "predictions.parquet",
    }

    # Save step data
    save_step_json(output_path, output_step)


def compute_roc(config: ComputeRocConfig) -> None:
    """Compute and print ROC AUC score."""
    logger.info(f"Starting ROC computation task; {config=}")

    # Load step data from previous step
    input_step = load_step_json(UPath(config.input_path))

    # Load prediction data from parquet
    predictions = pd.read_parquet(
        UPath(config.input_path) / input_step["predictions_relpath"]
    )
    y_true = predictions["label"].values
    y_score = predictions["plantcad_scores"].values

    results = compute_roc_auc(y_true, y_score)

    # Save only relevant results for this step (all fields expected by main pipeline)
    output_step = {
        "roc_auc": results.roc_auc,
        "num_samples": results.num_samples,
        "num_positive": results.num_positive,
        "num_negative": results.num_negative,
    }

    # Save final step data
    save_step_json(UPath(config.output_path), output_step)


class EvolutionaryConstraintPipeline:
    """Pipeline class for evolutionary constraint evaluation."""

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline.

        Parameters
        ----------
        config
            Pipeline configuration containing step configs
        """
        self.steps_config = config.tasks.evolutionary_constraint
        logger.info(f"EvolutionaryConstraintPipeline config: {self.steps_config}")

    def downsample_dataset(self) -> ExecutorStep:
        """Load and downsample the HuggingFace dataset."""
        return ExecutorStep(
            name="downsample_dataset",
            fn=downsample_dataset,
            config=replace(
                self.steps_config.downsample_dataset, output_path=this_output_path()
            ),
            description="Load and downsample dataset",
        )

    def generate_logits(self) -> ExecutorStep:
        """Generate logits using the pre-trained model."""
        return ExecutorStep(
            name="generate_logits",
            fn=generate_logits,
            config=replace(
                self.steps_config.generate_logits,
                input_path=output_path_of(self.downsample_dataset()),
                output_path=this_output_path(),
            ),
            description="Generate model logits",
        )

    def generate_scores(self) -> ExecutorStep:
        """Generate scores for plantcad evaluation."""
        return ExecutorStep(
            name="generate_scores",
            fn=generate_scores,
            config=replace(
                self.steps_config.generate_scores,
                input_path=output_path_of(self.generate_logits()),
                output_path=this_output_path(),
            ),
            description="Generate plantcad scores",
        )

    def compute_roc(self) -> ExecutorStep:
        """Compute and print ROC AUC score."""
        return ExecutorStep(
            name="compute_roc",
            fn=compute_roc,
            config=replace(
                self.steps_config.compute_roc,
                input_path=output_path_of(self.generate_scores()),
                output_path=this_output_path(),
            ),
            description="Compute ROC AUC",
        )

    def last_step(self) -> ExecutorStep:
        return self.compute_roc()
