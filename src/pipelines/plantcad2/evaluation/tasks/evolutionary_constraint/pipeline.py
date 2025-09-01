"""Evolutionary constraint evaluation pipeline using thalas execution pattern."""

import logging
import pickle
from dataclasses import replace
import numpy as np
import pandas as pd
import ray
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
from src.io import open_file

logger = logging.getLogger("ray")


def downsample_dataset(config: DownsampleDatasetConfig) -> None:
    """Load and downsample the HuggingFace dataset."""
    logger.info("Starting Evolutionary Constraint evaluation pipeline")
    logger.info(f"Loaded task configuration:\n{config}")

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

    # Save pipeline data
    pipeline_data = {
        "config": config,
        "num_samples": num_samples,
        "dataset_filename": dataset_path.name,
        "dataset_relpath": dataset_path.relative_to(output_path),
    }
    with open_file(output_path / "pipeline_data", "wb") as f:
        pickle.dump(pipeline_data, f)


def generate_logits(config: GenerateLogitsConfig) -> None:
    """Generate logits using either fake random data or real model inference."""

    # Load pipeline data from previous step
    with open_file(UPath(config.input_path) / "pipeline_data", "rb") as f:
        pipeline_data = pickle.load(f)

    # Construct dataset path from the previous step's output
    dataset_path = UPath(config.input_path) / pipeline_data["dataset_relpath"]

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
            )
            futures.append(future)

        # Wait for all tasks to complete and get file paths
        results = ray.get(futures)

        # Combine all worker output files into single logits.tsv
        combined_logits = []

        # Load and combine logits from all worker files
        for worker_file in results:
            if worker_file.exists():
                worker_logits = pd.read_csv(worker_file, sep="\t", header=None).values
                combined_logits.append(worker_logits)

        if combined_logits:
            final_logits = np.vstack(combined_logits)
            logits_path = logits_output_dir / "logits.tsv"
            pd.DataFrame(final_logits).to_csv(
                logits_path, sep="\t", header=False, index=False
            )
            logger.info(
                f"Combined {len(combined_logits)} worker outputs into {logits_path}"
            )
        else:
            raise ValueError("No worker outputs found to combine")

    pipeline_data.update(
        {
            "logits_relpath": logits_path.relative_to(output_path),
            "token_idx": config.token_idx,
        }
    )

    # Save updated pipeline data
    with open_file(output_path / "pipeline_data", "wb") as f:
        pickle.dump(pipeline_data, f)


def generate_scores(config: GenerateScoresConfig) -> None:
    """Generate scores for plantcad evaluation."""
    # Load pipeline data from previous step
    with open_file(UPath(config.input_path) / "pipeline_data", "rb") as f:
        pipeline_data = pickle.load(f)

    output_path = UPath(config.output_path)

    # Construct logits path from the previous step's output
    logits_path = UPath(config.input_path) / pipeline_data["logits_relpath"]
    logits_matrix = pd.read_csv(logits_path, sep="\t", header=None).values

    token_idx = pipeline_data["token_idx"]

    # Get downsampled dataset path from two steps upstream in pipeline
    dataset_path = UPath(config.dataset_path)

    _, y_true, y_scores = compute_plantcad_scores(
        dataset_path=dataset_path,
        logits_matrix=logits_matrix,
        output_dir=output_path / "scores",
        token_idx=token_idx,
    )

    # Update pipeline data
    pipeline_data.update(
        {
            "y_true": y_true,
            "y_scores": y_scores,
        }
    )

    # Save updated pipeline data
    with open_file(output_path / "pipeline_data", "wb") as f:
        pickle.dump(pipeline_data, f)


def compute_roc(config: ComputeRocConfig) -> None:
    """Compute and print ROC AUC score."""
    # Load pipeline data from previous step
    with open_file(UPath(config.input_path) / "pipeline_data", "rb") as f:
        pipeline_data = pickle.load(f)

    results = compute_roc_auc(pipeline_data["y_true"], pipeline_data["y_scores"])

    # Update pipeline data
    pipeline_data.update(
        {
            "results": results,
        }
    )

    # Final logging
    logger.info("Pipeline completed successfully!")
    logger.info(f"Final ROC AUC: {results.roc_auc:.4f}")
    logger.info(f"Processed {results.num_samples} samples")
    logger.info(f"Results saved to: {config.output_path}")

    # Save final pipeline data
    with open_file(UPath(config.output_path) / "pipeline_data", "wb") as f:
        pickle.dump(pipeline_data, f)


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
                dataset_path=output_path_of(self.downsample_dataset()),
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
