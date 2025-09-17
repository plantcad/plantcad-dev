"""Combined evaluation pipeline that orchestrates PlantCAD2 evaluation tasks."""

import json
import logging
import draccus
from upath import UPath
from thalas.execution import ExecutorStep
from src.io.hf import initialize_hf_path
from src.exec import executor_main
from src.pipelines.plantcad2.evaluation.config import PipelineConfig
from src.pipelines.plantcad2.evaluation.tasks.evolutionary_constraint.pipeline import (
    EvolutionaryConstraintPipeline,
)
from src.utils.logging_utils import filter_known_warnings, initialize_logging

logger = logging.getLogger("ray")


class EvaluationPipeline:
    """Pipeline class for PlantCAD2 evaluation tasks."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        logger.info(f"EvaluationPipeline config: {self.config}")
        self.evolutionary_constraint_pipeline = EvolutionaryConstraintPipeline(
            self.config
        )

    def run_local_simulation(self) -> bool:
        """Check if the pipeline is running in local simulation mode."""
        return self.config.tasks.evolutionary_constraint.generate_logits.simulation_mode

    def evolutionary_constraint(self) -> ExecutorStep:
        """Run the evolutionary constraint evaluation task."""
        return self.evolutionary_constraint_pipeline.last_step()


def main():
    """Main entry point for the evaluation pipeline."""
    initialize_logging()
    filter_known_warnings()

    logger.info("Starting evaluation pipeline")

    # Parse configurations from command line
    cfg = draccus.parse(config_class=PipelineConfig)

    # If the executor prefix is on HF, create the repository for it first or Thalas will fail with, e.g.:
    # > FileNotFoundError: plantcad/_dev_pc2_eval/evolutionary_downsample_dataset-be132f/.executor_info (repository not found).
    initialize_hf_path(cfg.executor.prefix)

    # Initialize the pipeline
    pipeline = EvaluationPipeline(cfg)

    # Fetch the final step
    step = pipeline.evolutionary_constraint()

    # Run the pipeline via Thalas/Ray
    executor = executor_main(cfg.executor, [step], init_logging=False)

    # Summarize results
    results = json.loads(
        (UPath(executor.output_paths[step]) / "step.json").read_text(encoding="utf-8")
    )
    logger.info(f"Pipeline complete! Results: {results}")


if __name__ == "__main__":
    main()
