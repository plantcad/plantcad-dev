"""Combined evaluation pipeline that orchestrates PlantCAD2 evaluation tasks."""

import logging
from pathlib import Path
import pickle
import draccus
from thalas.execution import ExecutorStep
from src import io
from src.exec import executor_main
from src.pipelines.plantcad2.evaluation.config import PipelineConfig
from src.pipelines.plantcad2.evaluation.tasks.evolutionary_constraint.pipeline import (
    EvolutionaryConstraintPipeline,
)
from src.log import init_logging

logger = logging.getLogger(__name__)


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
    init_logging()
    logger.info("Starting evaluation pipeline")

    cfg = draccus.parse(config_class=PipelineConfig)

    if cfg.executor.prefix is not None:
        if cfg.executor.prefix.startswith("hf://"):
            import pyarrow as pa

            pa.fs.FSSpecHandler(io.filesystem())
            hf_repo = io.HfRepo.from_url(cfg.executor.prefix)
            logger.info(f"Creating repository on Hugging Face: {hf_repo.url()}")
            io.create_on_hub(hf_repo, exist_ok=True)
            logger.info(f"Repository created: {hf_repo.url()}")

    pipeline = EvaluationPipeline(cfg)

    step = pipeline.evolutionary_constraint()

    executor = executor_main(cfg.executor, [step], init_logging=False)

    final_step_output = Path(executor.output_paths[step]) / "pipeline_data"
    logger.info(f"Final step output path: {final_step_output}")

    with open(final_step_output, "rb") as f:
        pipeline_data = pickle.load(f)

    # Summarize results
    results = pipeline_data["results"]
    logger.info("Pipeline complete! Results summary:")
    logger.info(f"  ROC AUC: {results.roc_auc:.4f}")
    logger.info(
        f"  Samples: {results.num_samples} ({results.num_positive} positive, {results.num_negative} negative)"
    )
    logger.info(f"  Dataset: {pipeline_data.get('dataset_filename', 'N/A')}")


if __name__ == "__main__":
    main()
