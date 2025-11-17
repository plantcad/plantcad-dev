from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import Any

import draccus
import pandas as pd
from upath import UPath

from src.exec import executor_main
from src.io.api import initialize_path
from src.utils.logging_utils import filter_known_warnings, initialize_logging
from thalas.execution import Executor, ExecutorStep, InputName, this_output_path

from src.pipelines.plantcad2.evaluation.config import (
    CoreNonCoreTaskConfig,
    EvoConsTaskConfig,
    MotifTaskConfig,
    PipelineConfig,
    StructuralVariantTaskConfig,
    TaskConfig,
)
from src.pipelines.plantcad2.evaluation.tasks import (
    core_noncore_task,
    evo_cons_task,
    motif_task,
    sv_task,
)

logger = logging.getLogger("ray")


def align_mask_indexes(
    defaults: dict[str, Any], model_context_length: int, seq_length: int
) -> dict[str, Any]:
    """Adjust mask indexes for shorter model context lengths.

    If model context is shorter than sequence length, recenter mask indexes
    to account for the center-cropping that will occur in load_task_data.
    """
    if model_context_length >= seq_length:
        return defaults

    if "mask_token_indexes" not in defaults:
        raise ValueError(
            f"mask_token_indexes required in defaults when model_context_length "
            f"({model_context_length}) < seq_length ({seq_length})"
        )

    original_indexes = defaults["mask_token_indexes"]
    offset = (seq_length - model_context_length) // 2
    adjusted_indexes = [idx - offset for idx in original_indexes]

    logger.info(
        f"Adjusting mask_token_indexes for context_length={model_context_length}: "
        f"{original_indexes} -> {adjusted_indexes} (offset={offset})"
    )
    return {**defaults, "mask_token_indexes": adjusted_indexes}


# Motif configuration defaults by length
MOTIF_L1_DEFAULTS: dict[str, Any] = {"mask_token_indexes": [4095], "motif_len": 1}
MOTIF_L2_DEFAULTS: dict[str, Any] = {"mask_token_indexes": [4095, 4096], "motif_len": 2}
MOTIF_L3_DEFAULTS: dict[str, Any] = {
    "mask_token_indexes": [4094, 4095, 4096],
    "motif_len": 3,
}

TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    "conservation_within_andropogoneae": MOTIF_L1_DEFAULTS,
    "conservation_within_poaceae_non_tis": MOTIF_L1_DEFAULTS,
    "conservation_within_poaceae_tis": MOTIF_L1_DEFAULTS,
    "acceptor_recovery": MOTIF_L2_DEFAULTS,
    "donor_recovery": MOTIF_L2_DEFAULTS,
    "tis_recovery": MOTIF_L3_DEFAULTS,
    "tts_recovery": MOTIF_L3_DEFAULTS,
    "acceptor_core_noncore_classification": MOTIF_L2_DEFAULTS,
    "donor_core_noncore_classification": MOTIF_L2_DEFAULTS,
    "tis_core_noncore_classification": MOTIF_L3_DEFAULTS,
    "tts_core_noncore_classification": MOTIF_L3_DEFAULTS,
}

TASK_REGISTRY: dict[str, tuple[type[TaskConfig], Callable[..., Any]]] = {
    "conservation_within_andropogoneae": (EvoConsTaskConfig, evo_cons_task),
    "conservation_within_poaceae_non_tis": (EvoConsTaskConfig, evo_cons_task),
    "conservation_within_poaceae_tis": (EvoConsTaskConfig, evo_cons_task),
    "acceptor_recovery": (MotifTaskConfig, motif_task),
    "donor_recovery": (MotifTaskConfig, motif_task),
    "tis_recovery": (MotifTaskConfig, motif_task),
    "tts_recovery": (MotifTaskConfig, motif_task),
    "structural_variant_effect_prediction": (StructuralVariantTaskConfig, sv_task),
    "acceptor_core_noncore_classification": (CoreNonCoreTaskConfig, core_noncore_task),
    "donor_core_noncore_classification": (CoreNonCoreTaskConfig, core_noncore_task),
    "tis_core_noncore_classification": (CoreNonCoreTaskConfig, core_noncore_task),
    "tts_core_noncore_classification": (CoreNonCoreTaskConfig, core_noncore_task),
}


def build_task_steps(
    config: PipelineConfig,
) -> list[ExecutorStep | InputName]:
    """Build executor steps from splits or use pre-configured tasks."""
    tasks = config.tasks
    if not tasks:
        # Build task configs from splits + models
        tasks = []
        for split_config in config.splits:
            if split_config.task not in TASK_REGISTRY:
                raise ValueError(f"Unknown task: {split_config.task}")

            config_cls = TASK_REGISTRY[split_config.task][0]
            task_defaults = TASK_DEFAULTS.get(split_config.task, {})

            for model_config in config.models:
                # Align mask indexes for model context length, then apply overrides
                defaults = align_mask_indexes(
                    defaults=task_defaults,
                    model_context_length=model_config.context_length,
                    seq_length=split_config.seq_length,
                )
                task_params = {**defaults, **split_config.overrides}
                runtime_config = config_cls(
                    repo_id=split_config.repo_id,
                    task=split_config.task,
                    split=split_config.split,
                    seq_column=split_config.seq_column,
                    label_column=split_config.label_column,
                    num_workers=config.compute.num_workers,
                    model_path=model_config.path,
                    model_type=model_config.type,
                    model_subfolder=model_config.subfolder,
                    model_context_length=model_config.context_length,
                    model_name=model_config.name,
                    model_motif_inference_mode=model_config.motif_inference_mode,
                    seq_length=split_config.seq_length,
                    device=config.compute.device,
                    batch_size=config.compute.batch_size,
                    sample_rate=config.sampling.rate if config.sampling else None,
                    sample_max_size=config.sampling.max_size
                    if config.sampling
                    else None,
                    sample_seed=config.sampling.seed if config.sampling else 0,
                    output_path=this_output_path(),
                    **task_params,
                )
                tasks.append(runtime_config)

    # Convert task configs to executor steps
    steps = []
    for task_config in tasks:
        if task_config.task not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_config.task}")

        assert task_config.model_name, "model_name must be set"
        step = ExecutorStep(
            name=(
                "__".join(
                    [
                        part.replace("/", "__").lower()
                        for part in [
                            task_config.task,
                            task_config.split,
                            task_config.model_name,
                        ]
                    ]
                )
            ),
            fn=TASK_REGISTRY[task_config.task][1],
            config=replace(task_config, output_path=this_output_path()),
            description=f"{task_config.task} ({task_config.split}, {task_config.model_name})",
        )
        steps.append(step)

    return steps


class EvaluationPipeline:
    """Pipeline coordinating all evaluation tasks."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        config_str = json.dumps(asdict(config), indent=2, default=str)
        logger.info(f"EvaluationPipeline config:\n{config_str}")
        self.steps = build_task_steps(config)

    def collect_results(self, executor: Executor) -> pd.DataFrame:
        """Collect results from all completed tasks into a DataFrame.

        Parameters
        ----------
        executor : Executor
            The executor instance with output_paths for each step

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per task, including task name, split, and metrics
        """
        results_list = []
        for step in self.steps:
            assert isinstance(step, ExecutorStep)
            result_path = UPath(executor.output_paths[step]) / "step.json"
            metrics = json.loads(result_path.read_text(encoding="utf-8"))

            # Extract task metadata from step config
            task_config = step.config
            result_row = {
                "dataset": task_config.repo_id,
                "model": task_config.model_path,
                "task": task_config.task,
                "split": task_config.split,
                **metrics,
            }
            results_list.append(result_row)

        return pd.DataFrame(results_list)


def main() -> None:
    initialize_logging()
    filter_known_warnings()

    logger.info("Starting evaluation pipeline")
    cfg = draccus.parse(config_class=PipelineConfig)

    if cfg.executor.prefix is None:
        raise ValueError("Executor prefix must be set")
    initialize_path(cfg.executor.prefix)

    pipeline = EvaluationPipeline(cfg)
    executor = executor_main(cfg.executor, steps=pipeline.steps, init_logging=False)

    results_df = pipeline.collect_results(executor)
    logger.info("All tasks complete:\n%s", results_df.to_string())


if __name__ == "__main__":
    main()
