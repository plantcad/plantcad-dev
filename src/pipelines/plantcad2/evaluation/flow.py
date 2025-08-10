"""Combined evaluation flow that orchestrates multiple PlantCAD2 evaluation tasks."""

import inspect
import logging
from typing import Type
from metaflow import FlowSpec, Runner, step
from metaflow.client.core import MetaflowData

from src.flow import BaseFlow
from src.pipelines.plantcad2.evaluation.tasks.evolutionary_constraint.flow import (
    EvolutionaryConstraintFlow,
)
from src.pipelines.plantcad2.evaluation.tasks.splice_acceptor.flow import (
    SpliceAcceptorFlow,
)
from src.log import init_logging

logger = logging.getLogger(__name__)


def run_flow(flow_class: Type[FlowSpec], **kwargs) -> MetaflowData:
    """
    Run a Metaflow FlowSpec class and return the data from the end step.

    Parameters
    ----------
    flow_class
        The FlowSpec class to execute
    **kwargs : Any
        Additional arguments that you would pass to python myflow.py after the run command,
        in particular, any parameters accepted by the flow.

    Returns
    -------
    MetaflowData
        The data object from the end step of the completed flow
    """
    task_flow_file = inspect.getfile(flow_class)

    with Runner(task_flow_file).run(**kwargs) as running:
        run = running.run
        return run["end"].task.data


class EvaluationFlow(BaseFlow):
    """
    Combined evaluation flow that runs both evolutionary constraint and splice acceptor tasks.

    This flow demonstrates the pattern of using Metaflow's Runner to execute child flows,
    as discussed in: https://github.com/Netflix/metaflow/issues/2538

    The approach allows for modular task composition while maintaining proper isolation
    between different evaluation tasks.
    """

    @step
    def start(self):
        """Initialize the evaluation pipeline."""
        logger.info("Starting combined PlantCAD2 evaluation pipeline")

        # Resolve configuration to validate it early
        self.config = self.resolve_config()
        logger.info(
            f"Configuration loaded with tasks: {list(self.config.tasks.__dict__.keys())}"
        )

        self.next(self.run_evolutionary_constraint, self.run_splice_acceptor)

    @step
    def run_evolutionary_constraint(self):
        """Run the evolutionary constraint evaluation task."""
        logger.info("Running evolutionary constraint evaluation...")

        end_data = run_flow(EvolutionaryConstraintFlow, overrides=self.overrides)

        # Store results from the task
        self.evolutionary_results = {
            "roc_auc": end_data.results.roc_auc,
            "num_samples": end_data.results.num_samples,
            "output_dir": str(end_data.output_dir),
        }

        logger.info(
            f"Evolutionary constraint completed. ROC AUC: {self.evolutionary_results['roc_auc']:.4f}"
        )

        self.next(self.combine_results)

    @step
    def run_splice_acceptor(self):
        """Run the splice acceptor evaluation task."""
        logger.info("Running splice acceptor evaluation...")

        end_data = run_flow(SpliceAcceptorFlow, overrides=self.overrides)

        # Store results from the task
        self.splice_acceptor_results = {
            "roc_auc": end_data.results.roc_auc,
            "num_samples": end_data.results.num_samples,
            "output_dir": str(end_data.output_dir),
        }

        logger.info(
            f"Splice acceptor completed. ROC AUC: {self.splice_acceptor_results['roc_auc']:.4f}"
        )

        self.next(self.combine_results)

    @step
    def combine_results(self, inputs):
        """Combine results from both evaluation tasks."""
        # Merge results from parallel tasks
        self.evolutionary_results = (
            inputs.run_evolutionary_constraint.evolutionary_results
        )
        self.splice_acceptor_results = (
            inputs.run_splice_acceptor.splice_acceptor_results
        )

        logger.info("=" * 60)
        logger.info("COMBINED EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(
            f"Evolutionary Constraint ROC AUC: {self.evolutionary_results['roc_auc']:.4f}"
        )
        logger.info(f"  Samples: {self.evolutionary_results['num_samples']}")
        logger.info(f"  Output: {self.evolutionary_results['output_dir']}")
        logger.info("")
        logger.info(
            f"Splice Acceptor ROC AUC: {self.splice_acceptor_results['roc_auc']:.4f}"
        )
        logger.info(f"  Samples: {self.splice_acceptor_results['num_samples']}")
        logger.info(f"  Output: {self.splice_acceptor_results['output_dir']}")
        logger.info("=" * 60)

        self.next(self.end)

    @step
    def end(self):
        """Final step of the evaluation pipeline."""
        logger.info("Combined PlantCAD2 evaluation pipeline completed successfully!")
        logger.info("All task results are available in the flow artifacts.")


if __name__ == "__main__":
    init_logging()
    EvaluationFlow()
