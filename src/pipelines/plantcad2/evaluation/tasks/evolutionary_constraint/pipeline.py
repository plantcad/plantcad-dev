"""Evolutionary constraint evaluation pipeline using thalas execution pattern."""

import logging
import pickle
from pathlib import Path
from dataclasses import replace
import numpy as np
import ray
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

logger = logging.getLogger(__name__)


def downsample_dataset(config: DownsampleDatasetConfig) -> None:
    """Load and downsample the HuggingFace dataset."""
    logger.info("Starting Evolutionary Constraint evaluation pipeline")
    logger.info(f"Loaded task configuration:\n{config}")

    # Setup dataset directory using executor-managed output path
    dataset_dir = Path(config.output_path) / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset directory: {dataset_dir}")

    # Load and downsample dataset
    dataset_path, num_samples = load_and_downsample_dataset(
        dataset_path=config.dataset_path,
        dataset_subdir=config.dataset_subdir,
        dataset_split=config.dataset_split,
        dataset_dir=dataset_dir,
        sample_size=config.sample_size,
    )

    # Save pipeline data
    pipeline_data = {
        "config": config,
        "num_samples": num_samples,
        "dataset_filename": dataset_path.name,
        "dataset_relpath": Path("dataset") / dataset_path.name,
    }

    output_path = Path(config.output_path) / "pipeline_data"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(pipeline_data, f)


def generate_logits(config: GenerateLogitsConfig) -> None:
    """Generate logits using either fake random data or real model inference."""

    # Load pipeline data from previous step
    with open(Path(config.input_path) / "pipeline_data", "rb") as f:
        pipeline_data = pickle.load(f)

    # Construct dataset path from the previous step's output
    dataset_path = Path(config.input_path) / pipeline_data["dataset_relpath"]

    logits_output_dir = Path(config.output_path) / "logits"

    # Use the new generate_model_logits function with simulation_mode
    if config.simulation_mode:
        generate_model_logits(
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
        logits_output_dir.mkdir(parents=True, exist_ok=True)
        combined_logits = []

        # Load and combine logits from all worker files
        for worker_file in results:
            if worker_file.exists():
                worker_logits = np.loadtxt(worker_file, delimiter="\t")
                if len(worker_logits.shape) == 1:
                    worker_logits = worker_logits.reshape(1, -1)
                combined_logits.append(worker_logits)

        if combined_logits:
            final_logits = np.vstack(combined_logits)
            final_logits_path = logits_output_dir / "logits.tsv"
            np.savetxt(final_logits_path, final_logits, delimiter="\t")
            logger.info(
                f"Combined {len(combined_logits)} worker outputs into {final_logits_path}"
            )
        else:
            logger.warning("No worker outputs found to combine")

    # TODO: Figure out the correct way to fetch inputs from two steps upstream
    # For now, pass the dataset_path forward in pipeline_data
    pipeline_data.update(
        {
            "dataset_path": dataset_path,
            "logits_relpath": Path("logits") / "logits.tsv",
            "token_idx": config.token_idx,
        }
    )

    # Save updated pipeline data
    output_path = Path(config.output_path) / "pipeline_data"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(pipeline_data, f)


def generate_scores(config: GenerateScoresConfig) -> None:
    """Generate scores for plantcad evaluation."""
    # Load pipeline data from previous step
    with open(Path(config.input_path) / "pipeline_data", "rb") as f:
        pipeline_data = pickle.load(f)

    # Construct logits path from the previous step's output
    logits_path = Path(config.input_path) / pipeline_data["logits_relpath"]
    logits_matrix = np.loadtxt(logits_path, delimiter="\t")

    token_idx = pipeline_data["token_idx"]

    # Get dataset path from pipeline_data (passed forward from generate_logits)
    # TODO: This is temporary - need proper way to access inputs from two steps upstream
    dataset_path = pipeline_data["dataset_path"]

    _, y_true, y_scores = compute_plantcad_scores(
        dataset_path=dataset_path,
        logits_matrix=logits_matrix,
        output_dir=Path(config.output_path) / "scores",
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
    output_path = Path(config.output_path) / "pipeline_data"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(pipeline_data, f)


def compute_roc(config: ComputeRocConfig) -> None:
    """Compute and print ROC AUC score."""
    # Load pipeline data from previous step
    with open(Path(config.input_path) / "pipeline_data", "rb") as f:
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
    output_path = Path(config.output_path) / "pipeline_data"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
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
            name="evolutionary_downsample_dataset",
            fn=downsample_dataset,
            config=replace(
                self.steps_config.downsample_dataset, output_path=this_output_path()
            ),
            description="Load and downsample dataset",
        )

    def generate_logits(self) -> ExecutorStep:
        """Generate logits using the pre-trained model."""
        return ExecutorStep(
            name="evolutionary_generate_logits",
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
            name="evolutionary_generate_scores",
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
            name="evolutionary_compute_roc",
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
