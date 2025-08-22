"""Combined evaluation pipeline that orchestrates PlantCAD2 evaluation tasks."""

import argparse
import logging
import pickle
from pathlib import Path
from dataclasses import replace
from omegaconf import OmegaConf
from typing import Any
import ray

from thalas.execution import Executor, ExecutorStep, this_output_path

from src.pipelines.plantcad2.evaluation.config import PipelineConfig
from src.pipelines.plantcad2.evaluation.tasks.evolutionary_constraint.pipeline import (
    EvolutionaryConstraintPipeline,
)
from src.log import init_logging

logger = logging.getLogger(__name__)


def run_evolutionary_constraint(
    config: PipelineConfig,
    force_run_failed: bool = False,
    local_prefix: str = "/tmp/evolutionary_constraint",
) -> dict[str, Any]:
    """Run the evolutionary constraint evaluation task.

    Parameters
    ----------
    config
        Pipeline configuration

    Returns
    -------
    dict[str, Any]
        Results dictionary from the pipeline execution
    """
    logger.info("Running evolutionary constraint evaluation...")

    # Initialize Ray if not already initialized
    ray.init(
        address="local",
        namespace="pipeline",
        ignore_reinit_error=True,
        resources={"head_node": 1},
    )

    # Create evolutionary constraint pipeline
    evolutionary_pipeline = EvolutionaryConstraintPipeline(config)

    # Define the final step once - it will automatically pull in its dependencies
    compute_roc_step = evolutionary_pipeline.compute_roc()

    # Run the pipeline with thalas Executor
    executor = Executor(prefix=local_prefix, executor_info_base_path=local_prefix)
    executor.run(
        steps=[
            compute_roc_step
        ],  # Only pass the final step - dependencies will be resolved automatically
        force_run_failed=force_run_failed,
    )

    # Load final results from the last step
    final_step_output = Path(executor.output_paths[compute_roc_step]) / "pipeline_data"
    logger.info(f"Final step output path: {final_step_output}")

    with open(final_step_output, "rb") as f:
        pipeline_data = pickle.load(f)

    results = {
        "roc_auc": pipeline_data["results"].roc_auc,
        "num_samples": pipeline_data["results"].num_samples,
        "results": pipeline_data["results"],
    }

    logger.info(f"Evolutionary constraint completed. ROC AUC: {results['roc_auc']:.4f}")

    return results


class EvaluationPipeline:
    """Pipeline class for PlantCAD2 evaluation tasks."""

    def __init__(self, config: PipelineConfig):
        """Initialize the evaluation pipeline.

        Parameters
        ----------
        config
            Pipeline configuration
        """
        self.config = config
        logger.info(f"EvaluationPipeline config: {self.config}")

    def run_evolutionary_constraint(self) -> ExecutorStep:
        """Run the evolutionary constraint evaluation task."""
        config = replace(self.config, output_path=this_output_path())
        return ExecutorStep(
            name="evaluation_run_evolutionary_constraint",
            fn=run_evolutionary_constraint,
            config=config,
            description="Run evolutionary constraint evaluation",
        )


def resolve_omega_conf(config_path: str, overrides: str) -> dict[str, Any]:
    """Load and resolve OmegaConf configuration with overrides.

    Parameters
    ----------
    config_path
        Path to the YAML configuration file
    overrides
        Comma-separated config overrides string

    Returns
    -------
    dict[str, Any]
        Resolved configuration dictionary
    """
    # Load base configuration from file
    base_config = OmegaConf.load(config_path)

    # Apply overrides if provided
    if overrides:
        override_config = OmegaConf.from_dotlist(overrides.split(","))
        merged_config = OmegaConf.merge(base_config, override_config)
    else:
        merged_config = base_config

    # Convert to container and resolve interpolations
    resolved_cfg = OmegaConf.to_container(merged_config, resolve=True)
    assert isinstance(resolved_cfg, dict)

    return resolved_cfg


def resolve_pydantic_conf(resolved_cfg: dict[str, Any]) -> PipelineConfig:
    """Convert resolved configuration dictionary to Pydantic objects.

    This function dynamically validates and creates the PipelineConfig by directly
    constructing it from the dictionary. Pydantic dataclasses automatically
    handle nested dictionary structures and convert them to the appropriate
    dataclass objects, providing validation and type checking.

    Parameters
    ----------
    resolved_cfg
        Resolved configuration dictionary from OmegaConf

    Returns
    -------
    PipelineConfig
        Structured pipeline configuration with Pydantic objects
    """
    # Pydantic dataclasses can automatically construct nested objects
    # from dictionaries, providing validation and type conversion
    return PipelineConfig(**resolved_cfg)


def main():
    """Main entry point for the evaluation pipeline."""
    parser = argparse.ArgumentParser(description="PlantCAD2 Evaluation Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="src/pipelines/plantcad2/evaluation/configs/config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--config-overrides",
        type=str,
        default="",
        help="Comma-separated config overrides (e.g., 'tasks.evolutionary_constraint.downsample_dataset.sample_size=500')",
    )
    parser.add_argument(
        "--force-run-failed",
        action="store_true",
        help="Force re-run of previously failed steps",
    )
    parser.add_argument(
        "--local-prefix",
        type=str,
        default="/tmp/evolutionary_constraint",
        help="Local directory prefix for pipeline execution",
    )

    args = parser.parse_args()

    init_logging()

    # Load and resolve OmegaConf configuration
    logger.info(f"Loading config from: {args.config}")
    resolved_omega_conf = resolve_omega_conf(args.config, args.config_overrides)

    # Convert to Pydantic configuration objects
    config = resolve_pydantic_conf(resolved_omega_conf)

    logger.info(f"Final configuration: {config}")

    # Run the evaluation
    results = run_evolutionary_constraint(
        config, force_run_failed=args.force_run_failed, local_prefix=args.local_prefix
    )
    logger.info(f"Evaluation completed with ROC AUC: {results['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
