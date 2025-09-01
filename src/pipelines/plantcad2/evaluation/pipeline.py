"""Combined evaluation pipeline that orchestrates PlantCAD2 evaluation tasks."""

import logging
import pickle
import draccus
from upath import UPath
from thalas.execution import ExecutorStep
from src.io import initialize_path, open_file
from src.exec import executor_main
from src.pipelines.plantcad2.evaluation.config import PipelineConfig
from src.pipelines.plantcad2.evaluation.tasks.evolutionary_constraint.pipeline import (
    EvolutionaryConstraintPipeline,
)
from src.log import initialize_logging

logger = logging.getLogger("ray")


class EvaluationPipeline:
    """Pipeline class for PlantCAD2 evaluation tasks."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        logger.info(f"EvaluationPipeline config: {self.config}")
        self.evolutionary_constraint_pipeline = EvolutionaryConstraintPipeline(
            self.config
        )

    def evolutionary_constraint(self) -> ExecutorStep:
        """Run the evolutionary constraint evaluation task."""
        return self.evolutionary_constraint_pipeline.last_step()


def main():
    """Main entry point for the evaluation pipeline."""
    initialize_logging()
    logger.info("Starting evaluation pipeline")

    # Parse configurations from command line
    cfg = draccus.parse(config_class=PipelineConfig)

    # If the executor prefix is on HF, create the repository for it first or Thalas will fail with, e.g.:
    # > FileNotFoundError: plantcad/_dev_biolm_demo/evolutionary_downsample_dataset-be132f/.executor_info (repository not found).
    initialize_path(cfg.executor.prefix)

    # Initialize the pipeline
    pipeline = EvaluationPipeline(cfg)

    # Fetch the final step
    step = pipeline.evolutionary_constraint()

    # Run the pipeline via Thalas/Ray
    executor = executor_main(cfg.executor, [step], init_logging=False)

    # Fetch the final step output path
    final_step_output = UPath(executor.output_paths[step]) / "pipeline_data"
    logger.info(f"Final step output path: {final_step_output}")
    with open_file(final_step_output, "rb") as f:
        pipeline_data = pickle.load(f)

    # Summarize results
    results = pipeline_data["results"]
    logger.info("Pipeline complete! Results summary:")
    logger.info(f"  ROC AUC: {results.roc_auc:.4f}")
    logger.info(
        f"  Samples: {results.num_samples} ({results.num_positive} positive, {results.num_negative} negative)"
    )
    logger.info(f"  Dataset: {pipeline_data.get('dataset_filename', 'N/A')}")

    logger.info("Evaluation pipeline complete.")


if __name__ == "__main__":
    main()
