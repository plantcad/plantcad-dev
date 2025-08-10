import logging
from pathlib import Path

from metaflow import step

from src.pipelines.plantcad2.evaluation.common import (
    compute_plantcad_scores,
    compute_roc_auc,
    generate_model_logits,
    load_and_downsample_dataset,
)
from src.flow import BaseFlow
from src.log import init_logging

logger = logging.getLogger(__name__)


class SpliceAcceptorFlow(BaseFlow):
    """A Metaflow pipeline for splice acceptor evaluation."""

    @step
    def start(self):
        """Initialize the pipeline and setup directories."""
        logger.info("Starting Splice Acceptor evaluation pipeline")

        self.config = super().resolve_config()
        self.task_config = self.config.tasks.splice_acceptor

        logger.info(f"Loaded task configuration:\n{self.task_config}")

        self.output_dir = Path(self.task_config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup log directory for SLURM jobs
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Resolved output directory: {self.output_dir}")
        logger.info(f"Log directory: {self.log_dir}")

        self.next(self.downsample_dataset)

    @step
    def downsample_dataset(self):
        """Load and downsample the HuggingFace dataset."""
        self.dataset_path, self.num_samples = load_and_downsample_dataset(
            repo_id=self.task_config.repo_id,
            dataset_subdir=self.task_config.dataset_subdir,
            dataset_split=self.task_config.dataset_split,
            output_dir=self.output_dir,
            sample_size=self.task_config.sample_size,
        )
        self.next(self.generate_logits)

    @step
    def generate_logits(self):
        """Generate logits using the pre-trained model with optional SLURM execution."""
        self.logits_path, self.logits_matrix = generate_model_logits(
            dataset_path=self.dataset_path,
            output_dir=self.output_dir,
            model_path=self.task_config.model_path,
            device=self.task_config.device,
            token_idx=self.task_config.token_idx,
            batch_size=self.task_config.batch_size,
            execution_mode=self.task_config.execution_mode,
            slurm_config=self.task_config.slurm_config.model_dump(),
            log_dir=self.log_dir,
        )
        self.next(self.compute_scores)

    @step
    def compute_scores(self):
        """Compute scores for plantcad evaluation."""
        self.scored_dataset_path, self.y_true, self.y_scores = compute_plantcad_scores(
            dataset_path=self.dataset_path,
            logits_matrix=self.logits_matrix,
            output_dir=self.output_dir,
            token_idx=self.task_config.token_idx,
        )
        self.next(self.compute_roc)

    @step
    def compute_roc(self):
        """Compute and print ROC AUC score."""
        self.results = compute_roc_auc(self.y_true, self.y_scores)
        self.next(self.end)

    @step
    def end(self):
        """Final step of the pipeline."""
        logger.info("Pipeline completed successfully!")
        logger.info(f"Final ROC AUC: {self.results.roc_auc:.4f}")
        logger.info(f"Processed {self.results.num_samples} samples")
        logger.info(f"Results saved to: {self.output_dir}")


def main():
    SpliceAcceptorFlow()


if __name__ == "__main__":
    init_logging()
    main()
