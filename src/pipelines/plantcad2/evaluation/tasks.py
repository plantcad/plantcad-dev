from __future__ import annotations

import logging
from typing import Callable, TypeVar

import pandas as pd
import ray
import xarray as xr
from sklearn.metrics import average_precision_score, auc, roc_curve
from upath import UPath

from src.io.api import read_xarray_mfdataset, read_xarray_netcdf, write_xarray_netcdf
from src.utils.pipeline_utils import save_step_json
from src.utils.ray_utils import num_cluster_gpus, num_cpus_per_node
from src.pipelines.plantcad2.evaluation.utils import (
    compute_core_noncore_scores,
    compute_evo_cons_probs,
    compute_motif_probs,
    compute_ref_auroc,
    compute_sv_scores,
    compute_true_tokens_from_seq,
    load_task_data,
    motif_accuracy_from_probs,
    reference_base_scores,
    token_accuracy_from_probs,
)

from src.pipelines.plantcad2.evaluation.config import (
    CoreNonCoreTaskConfig,
    EvoConsTaskConfig,
    MotifTaskConfig,
    StructuralVariantTaskConfig,
    TaskConfig,
)

logger = logging.getLogger("ray")

TaskConfigT = TypeVar("TaskConfigT", bound=TaskConfig)


def _resolve_num_workers(config_value: int | None) -> int:
    if config_value is not None:
        if config_value <= 0:
            raise ValueError(f"Invalid number of workers: {config_value}")
        logger.info(f"Using {config_value} workers (configured)")
        return config_value

    available = num_cluster_gpus()
    if available <= 0:
        raise ValueError("No GPUs available in Ray cluster")
    logger.info(f"Using {available} workers (detected GPUs)")
    return available


def _write_netcdf(path: UPath, ds: xr.Dataset) -> None:
    write_xarray_netcdf(ds, path)


def _merge_predictions(
    worker_dir: str, worker_files: list[str], output_path: str, config: TaskConfigT
) -> str:
    predictions = read_xarray_mfdataset(worker_dir, concat_dim="sample", glob="*.nc")
    # Materialize chunked arrays (as dask) in memory
    predictions = predictions.compute()
    logger.info(
        f"[task={config.task}, split={config.split}] Loaded predictions:\n{predictions}"
    )

    labels = load_task_data(config)
    labels_ds = (
        labels.set_index("example_idx").to_xarray().rename({"example_idx": "sample"})
    )
    logger.info(
        f"[task={config.task}, split={config.split}] Loaded labels:\n{labels_ds}"
    )

    results = xr.merge([predictions, labels_ds], join="inner")
    logger.info(
        f"[task={config.task}, split={config.split}] Merged dataset:\n{results}"
    )

    assert len(results.sample) == len(predictions.sample) == len(labels), (
        "Predictions and labels must match 1:1; "
        f"got {len(results.sample)} predictions, {len(labels)} labels, "
        f"and {len(predictions.sample)} results after merge"
    )

    _write_netcdf(UPath(output_path), results)
    return output_path


def _write_worker_results(
    config: TaskConfigT,
    worker_id: int,
    num_workers: int,
    output_dir: str,
    compute_fn: Callable,
    result_column: str,
    dims: list[str],
    to_list: bool = False,
) -> str:
    df = load_task_data(config, worker_id=worker_id, num_workers=num_workers)
    example_idx, result = compute_fn(df, config)
    da = xr.DataArray(result, dims=dims)
    da = da.assign_coords({"sample": example_idx})
    worker_ds = da.to_dataset(name=result_column)
    filename = f"worker_{worker_id:05d}.nc"
    _write_netcdf(UPath(output_dir) / filename, worker_ds)
    return filename


def _generate_results(
    config: TaskConfigT,
    worker: Callable[[TaskConfigT, int, int, str], str],
    output_dir: str,
) -> str:
    # Launch workers to run inference, defaulting to as
    # many workers as there are GPUs in the cluster
    num_workers = _resolve_num_workers(config.num_workers)
    worker_dir = str(UPath(output_dir) / "predictions")
    pred_fn = ray.remote(num_gpus=1)(worker)
    pred_futures = [
        pred_fn.remote(config, worker_id, num_workers, worker_dir)
        for worker_id in range(num_workers)
    ]
    pred_files = ray.get(pred_futures)

    # Merge predictions and join to labels on a single worker that
    # uses all CPUs available on a single node
    predictions_path = str(UPath(output_dir) / "predictions.nc")
    num_cpus = num_cpus_per_node()
    merge_fn = ray.remote(num_cpus=num_cpus)(_merge_predictions)
    merge_future = merge_fn.remote(worker_dir, pred_files, predictions_path, config)
    result_path = ray.get(merge_future)
    return result_path


def _compute_evo_cons_worker(
    config: EvoConsTaskConfig, worker_id: int, num_workers: int, output_dir: str
) -> str:
    return _write_worker_results(
        config=config,
        worker_id=worker_id,
        num_workers=num_workers,
        output_dir=output_dir,
        compute_fn=compute_evo_cons_probs,
        result_column="probs",
        dims=["sample", "nucleotide"],
        to_list=True,
    )


def evo_cons_task(config: EvoConsTaskConfig) -> None:
    output_path = UPath(config.output_path)
    results_path = _generate_results(
        config=config, worker=_compute_evo_cons_worker, output_dir=str(output_path)
    )
    results = read_xarray_netcdf(results_path)

    probs_da = results["probs"]
    assert probs_da.dims == ("sample", "nucleotide"), (
        f"Expected probs dims ('sample', 'nucleotide'), got {probs_da.dims}"
    )
    assert probs_da.sizes["nucleotide"] == 4, (
        f"Expected nucleotide dimension size 4, got {probs_da.sizes['nucleotide']}"
    )
    probs = probs_da.values
    n_examples = len(results.sample)
    # Extract 1D variables for metrics computation
    results_df = pd.DataFrame(
        {
            config.seq_column: results[config.seq_column].values,
            "label": results[config.label_column].values,
        }
    )
    roc_auc = compute_ref_auroc(
        results_df, probs, config.mask_token_index, config.seq_column
    )
    y_true = results[config.label_column].values.astype(int)
    pr_scores = reference_base_scores(
        results_df, probs, config.mask_token_index, config.seq_column
    )
    auprc = float(average_precision_score(y_true, pr_scores))

    num_positive = int(y_true.sum())
    num_negative = int((1 - y_true).sum())
    balance = float(num_positive / len(y_true))

    save_step_json(
        output_path,
        {
            "auroc": roc_auc,
            "auprc": auprc,
            "num_examples": n_examples,
            "num_positive_examples": num_positive,
            "num_negative_examples": num_negative,
            "balance": balance,
        },
    )


def _compute_motif_worker(
    config: MotifTaskConfig, worker_id: int, num_workers: int, output_dir: str
) -> str:
    return _write_worker_results(
        config=config,
        worker_id=worker_id,
        num_workers=num_workers,
        output_dir=output_dir,
        compute_fn=compute_motif_probs,
        result_column="probs",
        dims=["sample", "position", "nucleotide"],
        to_list=True,
    )


def motif_task(config: MotifTaskConfig) -> None:
    output_path = UPath(config.output_path)
    results_path = _generate_results(
        config=config, worker=_compute_motif_worker, output_dir=str(output_path)
    )
    results = read_xarray_netcdf(results_path)

    probs_da = results["probs"]
    assert probs_da.dims == ("sample", "position", "nucleotide"), (
        f"Expected probs dims ('sample', 'position', 'nucleotide'), got {probs_da.dims}"
    )
    assert probs_da.sizes["position"] == config.motif_len, (
        f"Expected position dimension size {config.motif_len}, got {probs_da.sizes['position']}"
    )
    assert probs_da.sizes["nucleotide"] == 4, (
        f"Expected nucleotide dimension size 4, got {probs_da.sizes['nucleotide']}"
    )
    probs = probs_da.values
    n_examples = len(results.sample)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    sequences = results[config.seq_column].values
    true_tokens = compute_true_tokens_from_seq(
        pd.Series(sequences), config.mask_token_indexes
    )
    token_acc = token_accuracy_from_probs(flat_probs, true_tokens)
    motif_acc = motif_accuracy_from_probs(flat_probs, true_tokens, config.motif_len)

    save_step_json(
        output_path,
        {
            "token_accuracy": token_acc,
            "motif_accuracy": motif_acc,
            "num_examples": n_examples,
        },
    )


def _compute_sv_worker(
    config: StructuralVariantTaskConfig,
    worker_id: int,
    num_workers: int,
    output_dir: str,
) -> str:
    df = load_task_data(config, worker_id=worker_id, num_workers=num_workers)
    result = compute_sv_scores(df, config)
    worker_ds = xr.Dataset(
        {"scores": (["sample"], result.scores)},
        coords={"sample": result.example_idx},
    )
    filename = f"worker_{worker_id:05d}.nc"
    _write_netcdf(UPath(output_dir) / filename, worker_ds)
    return filename


def sv_task(config: StructuralVariantTaskConfig) -> None:
    output_path = UPath(config.output_path)
    results_path = _generate_results(
        config=config, worker=_compute_sv_worker, output_dir=str(output_path)
    )
    results = read_xarray_netcdf(results_path)

    scores_da = results["scores"]
    assert scores_da.dims == ("sample",), (
        f"Expected scores dims ('sample',), got {scores_da.dims}"
    )
    scores = scores_da.values
    y_true = results[config.label_column].values.astype(int)
    auprc = float(average_precision_score(y_true, scores))

    num_positive = int(y_true.sum())
    num_negative = int((1 - y_true).sum())
    balance = float(num_positive / len(y_true))

    save_step_json(
        output_path,
        {
            "auprc": auprc,
            "num_examples": len(results.sample),
            "num_positive_examples": num_positive,
            "num_negative_examples": num_negative,
            "balance": balance,
        },
    )


def _compute_core_noncore_worker(
    config: CoreNonCoreTaskConfig,
    worker_id: int,
    num_workers: int,
    output_dir: str,
) -> str:
    return _write_worker_results(
        config=config,
        worker_id=worker_id,
        num_workers=num_workers,
        output_dir=output_dir,
        compute_fn=compute_core_noncore_scores,
        result_column="scores",
        dims=["sample"],
        to_list=False,
    )


def core_noncore_task(config: CoreNonCoreTaskConfig) -> None:
    output_path = UPath(config.output_path)
    results_path = _generate_results(
        config=config, worker=_compute_core_noncore_worker, output_dir=str(output_path)
    )
    results = read_xarray_netcdf(results_path)

    scores_da = results["scores"]
    assert scores_da.dims == ("sample",), (
        f"Expected scores dims ('sample',), got {scores_da.dims}"
    )
    scores = scores_da.values
    y_true = results[config.label_column].values.astype(int)
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = float(auc(fpr, tpr))
    auprc = float(average_precision_score(y_true, scores))

    num_positive = int(y_true.sum())
    num_negative = int((1 - y_true).sum())
    balance = float(num_positive / len(y_true))

    save_step_json(
        output_path,
        {
            "auroc": roc_auc,
            "auprc": auprc,
            "num_examples": len(results.sample),
            "num_positive_examples": num_positive,
            "num_negative_examples": num_negative,
            "balance": balance,
        },
    )
