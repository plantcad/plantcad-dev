"""Zero-shot evaluation steps used by the evaluation_v2 pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable, Iterable

import numpy as np
import ray
from sklearn.metrics import average_precision_score, auc, roc_curve
from upath import UPath
from thalas.execution import ExecutorStep, output_path_of, this_output_path

from src.utils.pipeline_utils import load_step_json, save_step_json
from src.utils.ray_utils import get_available_gpus
from src.zero_shot_eval import (
    compute_core_noncore_scores,
    compute_evo_cons_probs,
    compute_motif_probs,
    compute_ref_auroc,
    compute_sv_scores,
    compute_true_tokens_from_seq,
    load_zero_shot_dataframe,
    motif_accuracy_from_probs,
    reference_base_scores,
    token_accuracy_from_probs,
)

from ..config import (
    PipelineConfig,
    CoreNonCoreEvalConfig,
    CoreNonCoreScoreConfig,
    EvoConsEvalConfig,
    EvoConsProbsConfig,
    MotifEvalConfig,
    MotifProbsConfig,
    StructuralVariantEvalConfig,
    StructuralVariantScoreConfig,
)

logger = logging.getLogger("ray")


def _resolve_num_workers(config_value: int | None) -> int:
    if config_value is not None:
        if config_value <= 0:
            raise ValueError(f"Invalid number of workers: {config_value}")
        logger.info("Using %s workers (configured)", config_value)
        return config_value

    available = get_available_gpus()
    if available is None or available <= 0:
        raise ValueError("No GPUs available in the Ray cluster; set num_workers explicitly")
    logger.info("Using %s workers (detected GPUs)", available)
    return available


def _write_npz(path: UPath, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_npz_arrays(
    base_path: UPath, filenames: Iterable[str], key: str
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for name in filenames:
        data = np.load(base_path / name, allow_pickle=False)
        indices.append(data["example_idx"].astype(int))
        values.append(data[key])
    if not indices:
        return np.zeros(0, dtype=int), np.zeros((0,), dtype=np.float32)
    concat_idx = np.concatenate(indices, axis=0)
    concat_values = np.concatenate(values, axis=0)
    order = np.argsort(concat_idx)
    return concat_idx[order], concat_values[order]


def _compute_evo_cons_worker(
    config: EvoConsProbsConfig, worker_id: int, num_workers: int, output_dir: str
) -> str:
    df = load_zero_shot_dataframe(
        config.repo_id,
        config.task,
        config.split,
        worker_id=worker_id,
        num_workers=num_workers,
    )
    example_idx, probs = compute_evo_cons_probs(
        df,
        model=config.model,
        device=config.device,
        token_idx=config.token_idx,
        batch_size=config.batch_size,
        seq_column=config.seq_column,
    )
    filename = f"probs_{worker_id:05d}.npz"
    _write_npz(UPath(output_dir) / filename, example_idx=example_idx, probs=probs)
    return filename


def compute_evo_cons_probs_step(config: EvoConsProbsConfig) -> None:
    output_path = UPath(config.output_path)
    num_workers = _resolve_num_workers(config.num_workers)
    remote_fn = ray.remote(num_gpus=config.gpus_per_worker)(_compute_evo_cons_worker)
    futures = [
        remote_fn.remote(config, worker_id, num_workers, str(output_path))
        for worker_id in range(num_workers)
    ]
    files = ray.get(futures)
    save_step_json(
        output_path,
        {
            "files": files,
            "num_workers": num_workers,
        },
    )


def evaluate_evo_cons_step(config: EvoConsEvalConfig) -> None:
    step_meta = load_step_json(UPath(config.input_path))
    base_path = UPath(config.input_path)
    _, probs = _load_npz_arrays(base_path, step_meta.get("files", []), "probs")
    df = load_zero_shot_dataframe(config.repo_id, config.task, config.split)
    df = df.sort_values("example_idx").reset_index(drop=True)
    if len(probs) != len(df):
        raise ValueError(
            f"Mismatch between probabilities ({len(probs)}) and dataset size ({len(df)})"
        )
    roc_auc = compute_ref_auroc(df, probs, config.token_idx, config.seq_column)
    y_true = df[config.label_column].astype(int).to_numpy()
    pr_scores = reference_base_scores(df, probs, config.token_idx, config.seq_column)
    auprc = float(average_precision_score(y_true, pr_scores))
    save_step_json(
        UPath(config.output_path),
        {
            "auroc": roc_auc,
            "auprc": auprc,
            "num_examples": len(df),
        },
    )


def _compute_motif_worker(
    config: MotifProbsConfig, worker_id: int, num_workers: int, output_dir: str
) -> str:
    df = load_zero_shot_dataframe(
        config.repo_id,
        config.task,
        config.split,
        worker_id=worker_id,
        num_workers=num_workers,
    )
    example_idx, probs = compute_motif_probs(
        df,
        model=config.model,
        device=config.device,
        batch_size=config.batch_size,
        seq_column=config.seq_column,
        mask_idx=config.mask_idx,
    )
    filename = f"motif_probs_{worker_id:05d}.npz"
    _write_npz(UPath(output_dir) / filename, example_idx=example_idx, probs=probs)
    return filename


def compute_motif_probs_step(config: MotifProbsConfig) -> None:
    output_path = UPath(config.output_path)
    num_workers = _resolve_num_workers(config.num_workers)
    remote_fn = ray.remote(num_gpus=config.gpus_per_worker)(_compute_motif_worker)
    futures = [
        remote_fn.remote(config, worker_id, num_workers, str(output_path))
        for worker_id in range(num_workers)
    ]
    files = ray.get(futures)
    save_step_json(output_path, {"files": files, "num_workers": num_workers})


def evaluate_motif_step(config: MotifEvalConfig) -> None:
    step_meta = load_step_json(UPath(config.input_path))
    base_path = UPath(config.input_path)
    _, probs = _load_npz_arrays(base_path, step_meta.get("files", []), "probs")
    df = load_zero_shot_dataframe(config.repo_id, config.task, config.split)
    df = df.sort_values("example_idx").reset_index(drop=True)
    if len(probs) != len(df):
        raise ValueError(
            f"Mismatch between probabilities ({len(probs)}) and dataset size ({len(df)})"
        )
    flat_probs = probs.reshape(-1, probs.shape[-1])
    true_tokens = compute_true_tokens_from_seq(df[config.seq_column], config.mask_idx)
    token_acc = token_accuracy_from_probs(flat_probs, true_tokens)
    motif_acc = motif_accuracy_from_probs(flat_probs, true_tokens, config.motif_len)
    save_step_json(
        UPath(config.output_path),
        {
            "token_accuracy": token_acc,
            "motif_accuracy": motif_acc,
            "num_examples": len(df),
        },
    )


def _compute_sv_worker(
    config: StructuralVariantScoreConfig,
    worker_id: int,
    num_workers: int,
    output_dir: str,
) -> str:
    df = load_zero_shot_dataframe(
        config.repo_id,
        config.task,
        config.split,
        worker_id=worker_id,
        num_workers=num_workers,
    )
    result = compute_sv_scores(
        df,
        model=config.model,
        device=config.device,
        batch_size=config.batch_size,
        flanking=config.flanking,
    )
    filename = f"sv_scores_{worker_id:05d}.npz"
    _write_npz(
        UPath(output_dir) / filename,
        example_idx=result.example_idx,
        scores=result.scores,
    )
    return filename


def compute_sv_scores_step(config: StructuralVariantScoreConfig) -> None:
    output_path = UPath(config.output_path)
    num_workers = _resolve_num_workers(config.num_workers)
    remote_fn = ray.remote(num_gpus=config.gpus_per_worker)(_compute_sv_worker)
    futures = [
        remote_fn.remote(config, worker_id, num_workers, str(output_path))
        for worker_id in range(num_workers)
    ]
    files = ray.get(futures)
    save_step_json(output_path, {"files": files, "num_workers": num_workers})


def evaluate_sv_step(config: StructuralVariantEvalConfig) -> None:
    step_meta = load_step_json(UPath(config.input_path))
    base_path = UPath(config.input_path)
    _, scores = _load_npz_arrays(base_path, step_meta.get("files", []), "scores")
    df = load_zero_shot_dataframe(config.repo_id, config.task, config.split)
    df = df.sort_values("example_idx").reset_index(drop=True)
    if len(scores) != len(df):
        raise ValueError(
            f"Mismatch between scores ({len(scores)}) and dataset size ({len(df)})"
        )
    y_true = df[config.label_column].astype(int).to_numpy()
    auprc = float(average_precision_score(y_true, scores))
    save_step_json(
        UPath(config.output_path),
        {
            "auprc": auprc,
            "num_examples": len(df),
        },
    )


def _compute_core_noncore_worker(
    config: CoreNonCoreScoreConfig,
    worker_id: int,
    num_workers: int,
    output_dir: str,
) -> str:
    df = load_zero_shot_dataframe(
        config.repo_id,
        config.task,
        config.split,
        worker_id=worker_id,
        num_workers=num_workers,
    )
    example_idx, scores = compute_core_noncore_scores(
        df,
        model=config.model,
        device=config.device,
        batch_size=config.batch_size,
        seq_column=config.seq_column,
        mask_idx=config.mask_idx,
        motif_len=config.motif_len,
    )
    filename = f"core_noncore_scores_{worker_id:05d}.npz"
    _write_npz(UPath(output_dir) / filename, example_idx=example_idx, scores=scores)
    return filename


def compute_core_noncore_scores_step(config: CoreNonCoreScoreConfig) -> None:
    output_path = UPath(config.output_path)
    num_workers = _resolve_num_workers(config.num_workers)
    remote_fn = ray.remote(num_gpus=config.gpus_per_worker)(_compute_core_noncore_worker)
    futures = [
        remote_fn.remote(config, worker_id, num_workers, str(output_path))
        for worker_id in range(num_workers)
    ]
    files = ray.get(futures)
    save_step_json(output_path, {"files": files, "num_workers": num_workers})


def evaluate_core_noncore_step(config: CoreNonCoreEvalConfig) -> None:
    step_meta = load_step_json(UPath(config.input_path))
    base_path = UPath(config.input_path)
    _, scores = _load_npz_arrays(base_path, step_meta.get("files", []), "scores")
    df = load_zero_shot_dataframe(config.repo_id, config.task, config.split)
    df = df.sort_values("example_idx").reset_index(drop=True)
    if len(scores) != len(df):
        raise ValueError(
            f"Mismatch between scores ({len(scores)}) and dataset size ({len(df)})"
        )
    y_true = df[config.label_column].astype(int).to_numpy()
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = float(auc(fpr, tpr))
    auprc = float(average_precision_score(y_true, scores))
    save_step_json(
        UPath(config.output_path),
        {
            "auroc": roc_auc,
            "auprc": auprc,
            "num_examples": len(df),
        },
    )


@dataclass
class TwoStepTaskPipeline:
    name: str
    compute_fn: Callable[[object], None]
    evaluate_fn: Callable[[object], None]
    compute_config: object
    evaluate_config: object
    compute_description: str
    evaluate_description: str

    def build_steps(self) -> tuple[ExecutorStep, ExecutorStep]:
        compute_step = ExecutorStep(
            name=f"{self.name}_compute",
            fn=self.compute_fn,
            config=replace(self.compute_config, output_path=this_output_path()),
            description=self.compute_description,
        )
        evaluate_step = ExecutorStep(
            name=f"{self.name}_evaluate",
            fn=self.evaluate_fn,
            config=replace(
                self.evaluate_config,
                input_path=output_path_of(compute_step),
                output_path=this_output_path(),
            ),
            description=self.evaluate_description,
        )
        return compute_step, evaluate_step


def build_task_pipelines(config: PipelineConfig) -> list[TwoStepTaskPipeline]:
    return [
        TwoStepTaskPipeline(
            name="evo_cons",
            compute_fn=compute_evo_cons_probs_step,
            evaluate_fn=evaluate_evo_cons_step,
            compute_config=config.tasks.evo_cons.compute_probs,
            evaluate_config=config.tasks.evo_cons.evaluate,
            compute_description="Compute evolutionary constraint probabilities",
            evaluate_description="Evaluate evolutionary constraint metrics",
        ),
        TwoStepTaskPipeline(
            name="motif",
            compute_fn=compute_motif_probs_step,
            evaluate_fn=evaluate_motif_step,
            compute_config=config.tasks.motif.compute_probs,
            evaluate_config=config.tasks.motif.evaluate,
            compute_description="Compute motif masked-token probabilities",
            evaluate_description="Evaluate motif accuracy metrics",
        ),
        TwoStepTaskPipeline(
            name="sv_effect",
            compute_fn=compute_sv_scores_step,
            evaluate_fn=evaluate_sv_step,
            compute_config=config.tasks.sv_effect.compute_scores,
            evaluate_config=config.tasks.sv_effect.evaluate,
            compute_description="Compute structural variant effect scores",
            evaluate_description="Evaluate structural variant metrics",
        ),
        TwoStepTaskPipeline(
            name="core_noncore",
            compute_fn=compute_core_noncore_scores_step,
            evaluate_fn=evaluate_core_noncore_step,
            compute_config=config.tasks.core_noncore.compute_scores,
            evaluate_config=config.tasks.core_noncore.evaluate,
            compute_description="Compute core vs non-core scores",
            evaluate_description="Evaluate core vs non-core metrics",
        ),
    ]


__all__ = [
    "TwoStepTaskPipeline",
    "build_task_pipelines",
    "compute_core_noncore_scores_step",
    "compute_evo_cons_probs_step",
    "compute_motif_probs_step",
    "compute_sv_scores_step",
    "evaluate_core_noncore_step",
    "evaluate_evo_cons_step",
    "evaluate_motif_step",
    "evaluate_sv_step",
]
