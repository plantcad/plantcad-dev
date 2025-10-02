"""Combined evaluation pipeline orchestrating zero-shot evaluation tasks."""

from __future__ import annotations

import json
import logging

import draccus
from upath import UPath

from src.exec import executor_main
from src.io.hf import initialize_hf_path
from src.utils.logging_utils import filter_known_warnings, initialize_logging
from thalas.execution import ExecutorStep

from .config import PipelineConfig
from .tasks.zero_shot import TwoStepTaskPipeline, build_task_pipelines

logger = logging.getLogger("ray")


class EvaluationPipeline:
    """Pipeline coordinating all zero-shot evaluation tasks."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        logger.info("EvaluationPipeline config: %s", self.config)
        self._task_pipelines: list[TwoStepTaskPipeline] = build_task_pipelines(config)
        self._task_steps: list[tuple[str, ExecutorStep, ExecutorStep]] = []

    def build(self) -> list[ExecutorStep]:
        evaluate_steps: list[ExecutorStep] = []
        self._task_steps = []
        for task in self._task_pipelines:
            compute_step, evaluate_step = task.build_steps()
            self._task_steps.append((task.name, compute_step, evaluate_step))
            evaluate_steps.append(evaluate_step)
        return evaluate_steps

    @property
    def evaluate_steps(self) -> list[tuple[str, ExecutorStep]]:
        return [(name, evaluate_step) for name, _, evaluate_step in self._task_steps]


def main() -> None:
    initialize_logging()
    filter_known_warnings()

    logger.info("Starting evaluation_v2 pipeline")
    cfg = draccus.parse(config_class=PipelineConfig)

    if cfg.executor.prefix is None:
        raise ValueError("Executor prefix must be set")
    initialize_hf_path(cfg.executor.prefix)

    pipeline = EvaluationPipeline(cfg)
    final_steps = pipeline.build()
    executor = executor_main(cfg.executor, final_steps, init_logging=False)

    for name, step in pipeline.evaluate_steps:
        result_path = UPath(executor.output_paths[step]) / "step.json"
        results = json.loads(result_path.read_text(encoding="utf-8"))
        logger.info("Task %s complete: %s", name, results)


if __name__ == "__main__":
    main()
