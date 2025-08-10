"""Shared evaluation utilities for PlantCAD2 tasks.

This module centralizes logic used by multiple evaluation tasks, including:
- dataset loading/downsampling from Hugging Face
- model/tokenizer loading
- logits generation for masked LM scoring
- score merging by reference nucleotide
- ROC AUC computation and result reporting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from numpy.typing import NDArray
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.hub import load_model_for_masked_lm, load_tokenizer
from src.pipelines.plantcad2.evaluation.data import SequenceDataset
from src.execution.slurm import run_slurm_function, process_group

logger = logging.getLogger(__name__)


def get_evaluation_config_path() -> str:
    """Get the path to the PlantCAD2 evaluation config.yaml file."""
    return str(Path(__file__).parent / "configs" / "config.yaml")


def _generate_logits_for_dataframe(
    df: pd.DataFrame,
    model_path: str,
    device: str | torch.device,
    token_idx: int,
    batch_size: int,
    rank_info: str = "",
) -> NDArray[np.floating]:
    """Core logits generation logic for a DataFrame partition."""
    if len(df) == 0:
        logger.warning(f"No data for {rank_info}")
        return np.array([])

    model, tokenizer = _load_model_and_tokenizer(model_path=model_path, device=device)

    sequences = df["sequences"].tolist()
    name_column = (
        "pos" if "pos" in df.columns else ("name" if "name" in df.columns else None)
    )
    names = df[name_column].tolist() if name_column else list(range(len(df)))

    dataset = SequenceDataset(
        sequences=sequences,
        tokenizer=tokenizer,
        names=names,
        mask_token_id=token_idx,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    nucleotides = list("acgt")
    all_logits: list[NDArray[np.floating]] = []

    desc = f"{rank_info} generating logits" if rank_info else "Generating logits"
    for batch in tqdm(loader, desc=desc):
        cur_ids = batch["input_ids"].to(device)
        cur_ids = cur_ids.squeeze(1)

        with torch.inference_mode():
            outputs = model(input_ids=cur_ids)
            batch_logits = outputs.logits

            logits = batch_logits[
                :,
                token_idx,
                [tokenizer.get_vocab()[nc] for nc in nucleotides],
            ]
            probs = torch.nn.functional.softmax(logits.cpu(), dim=1).numpy()
            all_logits.append(probs)

    return np.vstack(all_logits) if all_logits else np.array([])


def _generate_logits_task(
    dataset_path: Path,
    output_dir: Path,
    model_path: str,
    device: str | torch.device,
    token_idx: int,
    batch_size: int,
) -> tuple[Path, NDArray[np.floating]]:
    """SLURM task for generating logits on a data partition."""
    pg = process_group()
    logger.info(f"SLURM task {pg.rank}/{pg.world_size}: generating logits")

    df = pd.read_parquet(dataset_path)
    # Partition data for this rank
    df_partition = df.iloc[pg.rank :: pg.world_size]

    logits_matrix = _generate_logits_for_dataframe(
        df=df_partition,
        model_path=model_path,
        device=device,
        token_idx=token_idx,
        batch_size=batch_size,
        rank_info=f"Rank {pg.rank}",
    )

    logits_path = output_dir / f"logits_rank_{pg.rank}.tsv"
    if len(logits_matrix) > 0:
        np.savetxt(logits_path, logits_matrix, delimiter="\t")

    logger.info(f"Rank {pg.rank}: Generated logits for {len(logits_matrix)} sequences")

    return logits_path, logits_matrix


@dataclass
class EvaluationResults:
    """Results from task evaluation.

    Attributes
    ----------
    roc_auc
        Area under the ROC curve
    num_samples
        Total number of samples evaluated
    num_positive
        Number of positive labels
    num_negative
        Number of negative labels
    fpr
        False positive rates for each threshold
    tpr
        True positive rates for each threshold
    thresholds
        Decision thresholds used to compute ROC
    """

    roc_auc: float
    num_samples: int
    num_positive: int
    num_negative: int
    fpr: NDArray[np.floating]
    tpr: NDArray[np.floating]
    thresholds: NDArray[np.floating]


def _load_model_and_tokenizer(model_path: str, device: str | torch.device) -> tuple:
    """Load model and tokenizer with appropriate dtype handling.

    Parameters
    ----------
    model_path
        Hugging Face model identifier or local path.
    device
        Device to place the model on.

    Returns
    -------
    tuple
        Tuple of (model, tokenizer).
    """

    model = load_model_for_masked_lm(path=model_path, dtype=torch.bfloat16).to(device)
    tokenizer = load_tokenizer(path=model_path)
    return model, tokenizer


def _validate_sequence_lengths(df: pd.DataFrame) -> int:
    """Validate that all sequences have the same length and return it."""
    sequence_lengths = df["sequences"].str.len()
    logger.info("Sequence length summary:")
    logger.info(f"  Min: {sequence_lengths.min()}")
    logger.info(f"  Max: {sequence_lengths.max()}")
    logger.info(f"  Mean: {sequence_lengths.mean():.2f}")

    unique_lengths = sequence_lengths.unique()
    if len(unique_lengths) > 1:
        raise ValueError(
            "Found sequences with different lengths. All sequences must have the same length. "
            f"Lengths found: {sorted(unique_lengths)}"
        )

    seq_length = unique_lengths[0]
    logger.info(f"✓ All sequences are {seq_length} characters long")
    return seq_length


def load_and_downsample_dataset(
    repo_id: str,
    dataset_subdir: str,
    dataset_split: str,
    output_dir: Path,
    sample_size: int | None = None,
) -> tuple[Path, int]:
    """Load dataset from Hugging Face and optionally downsample.

    Parameters
    ----------
    repo_id
        HuggingFace repository ID for the dataset.
    dataset_subdir
        Subdirectory within the HF dataset repo (e.g., "Evolutionary_constraint", "Acceptor").
    dataset_split
        The dataset split to load (e.g., "train", "valid", "test").
    output_dir
        Directory to save the downsampled dataset.
    sample_size
        Number of samples to downsample to. If None, uses the full dataset.

    Returns
    -------
    tuple[Path, int]
        Tuple of (dataset_path, num_samples).
    """

    logger.info("Loading and downsampling dataset")

    data = load_dataset(
        repo_id,
        data_files={dataset_split: f"{dataset_subdir}/{dataset_split}.tsv"},
    )
    df = data[dataset_split].to_pandas()

    logger.info(f"Original dataset size: {len(df)}")

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Downsampled to: {len(df)} samples")

    dataset_path = output_dir / f"downsampled_{dataset_split}.parquet"
    df.to_parquet(dataset_path)
    logger.info(f"Saved downsampled dataset to: {dataset_path}")

    return dataset_path, len(df)


def generate_model_logits(
    dataset_path: Path,
    output_dir: Path,
    model_path: str,
    device: str | torch.device,
    token_idx: int,
    batch_size: int,
    execution_mode: str = "local",
    slurm_config: dict | None = None,
    log_dir: Path | None = None,
) -> tuple[Path, NDArray[np.floating]]:
    """Generate logits using the pre-trained model with optional distributed execution.

    Parameters
    ----------
    dataset_path
        Path to the parquet dataset containing columns `sequences` and a name/id column.
    output_dir
        Directory where outputs will be written. Must exist.
    model_path
        Hugging Face model identifier or local path for the masked LM.
    device
        Device for running inference.
    token_idx
        Index of the masked token position to score.
    batch_size
        Batch size for the DataLoader.
    execution_mode
        Execution mode: "local" or "slurm". Defaults to "local".
    slurm_config
        SLURM configuration dict. Required if execution_mode is "slurm".
    log_dir
        Directory for SLURM job logs. Required if execution_mode is "slurm".

    Returns
    -------
    tuple[Path, NDArray[np.floating]]
        Tuple of (logits_path, logits_matrix).
    """

    logger.info(
        f"Generating logits with pre-trained model using {execution_mode} execution"
    )

    if execution_mode == "local":
        df = pd.read_parquet(dataset_path)
        _validate_sequence_lengths(df)

        logger.info(f"Processing {len(df)} sequences")

        logits_matrix = _generate_logits_for_dataframe(
            df=df,
            model_path=model_path,
            device=device,
            token_idx=token_idx,
            batch_size=batch_size,
        )

        logits_path = output_dir / "logits.tsv"
        np.savetxt(logits_path, logits_matrix, delimiter="\t")

        logger.info(f"Generated logits for {len(logits_matrix)} sequences")
        logger.info(f"Saved logits to: {logits_path}")

        return logits_path, logits_matrix

    elif execution_mode == "slurm":
        if slurm_config is None or log_dir is None:
            raise ValueError(
                "slurm_config and log_dir are required for SLURM execution"
            )

        # Create a partial function for SLURM execution
        task_func = partial(
            _generate_logits_task,
            dataset_path=dataset_path,
            output_dir=output_dir,
            model_path=model_path,
            device=device,
            token_idx=token_idx,
            batch_size=batch_size,
        )

        # Execute the task on SLURM
        results = run_slurm_function(
            task_func,
            log_dir=log_dir,
            partition=slurm_config["partition"],
            timeout_min=slurm_config["timeout_min"],
            nodes=slurm_config["nodes"],
            tasks_per_node=slurm_config["tasks_per_node"],
            cluster=slurm_config.get("cluster", "slurm"),
        )

        # Combine results from all ranks
        all_logits = []
        for logits_path, logits_matrix in results:
            if len(logits_matrix) > 0:
                all_logits.append(logits_matrix)

        combined_logits = np.vstack(all_logits) if all_logits else np.array([])

        # Save combined results
        final_logits_path = output_dir / "logits.tsv"
        if len(combined_logits) > 0:
            np.savetxt(final_logits_path, combined_logits, delimiter="\t")

        logger.info(f"Combined logits from all ranks: {len(combined_logits)} sequences")
        logger.info(f"Saved combined logits to: {final_logits_path}")

        return final_logits_path, combined_logits

    else:
        raise ValueError(
            f"Invalid execution_mode: {execution_mode}. Must be 'local' or 'slurm'."
        )


def compute_plantcad_scores(
    dataset_path: Path,
    logits_matrix: NDArray[np.floating],
    output_dir: Path,
    token_idx: int,
) -> tuple[Path, NDArray[np.floating], NDArray[np.floating]]:
    """Compute scores by selecting the reference nucleotide probability.

    Parameters
    ----------
    dataset_path
        Path to the parquet dataset containing columns `sequences`, `label`.
    logits_matrix
        Logits probabilities with columns ordered as ["A", "C", "G", "T"].
    output_dir
        Directory where the scored dataset parquet will be written.
    token_idx
        Index of the masked token position used to choose the reference base.

    Returns
    -------
    tuple[Path, NDArray[np.floating], NDArray[np.floating]]
        Tuple of (scored_dataset_path, y_true, y_scores).
    """

    logger.info("Merging scores with labels")

    df = pd.read_parquet(dataset_path)
    logits_df = pd.DataFrame(logits_matrix, columns=["A", "C", "G", "T"])

    ref_nucleotides = df["sequences"].str[token_idx]
    logger.info("Reference nucleotide distribution:")
    ref_counts = ref_nucleotides.value_counts()
    for nuc, count in ref_counts.items():
        logger.info(f"  {nuc}: {count}")

    scores = df.apply(
        lambda row: logits_df.loc[row.name, ref_nucleotides.loc[row.name]]
        if ref_nucleotides.loc[row.name] in "ATCG"
        else 0,
        axis=1,
    )

    df["plantcad_scores"] = scores

    scored_dataset_path = output_dir / "dataset_with_scores.parquet"
    df.to_parquet(scored_dataset_path)

    logger.info(f"Generated scores for {len(scores)} samples")
    logger.info(f"Non-zero scores: {sum(scores != 0)}")
    logger.info(f"Saved scored dataset to: {scored_dataset_path}")

    return scored_dataset_path, df["label"].values, scores.values


def compute_roc_auc(
    y_true: NDArray[np.floating], y_scores: NDArray[np.floating]
) -> EvaluationResults:
    """Compute ROC AUC and related metrics."""

    logger.info("Computing ROC AUC score")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    results = EvaluationResults(
        roc_auc=roc_auc,
        num_samples=len(y_true),
        num_positive=int(sum(y_true)),
        num_negative=int(sum(1 - y_true)),
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"ROC AUC Score: {results.roc_auc:.4f}")
    logger.info(f"Number of samples: {results.num_samples}")
    logger.info(f"Positive labels: {results.num_positive}")
    logger.info(f"Negative labels: {results.num_negative}")
    logger.info("=" * 50)

    return results
