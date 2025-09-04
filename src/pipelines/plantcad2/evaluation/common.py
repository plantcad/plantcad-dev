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

import numpy as np
import pandas as pd
import ray
import torch
import xarray as xr
from upath import UPath
from datasets import load_dataset
from numpy.typing import NDArray
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from src.hub import load_model_for_masked_lm, load_tokenizer
from src.pipelines.plantcad2.evaluation.data import SequenceDataset
from src.io.api import read_pandas_parquet, write_pandas_parquet, write_xarray_netcdf
from src.io.hf import lock_hf_path


logger = logging.getLogger("ray")


def simulate_logits(df: pd.DataFrame) -> xr.Dataset:
    """Generate fake random logits for testing without GPU.

    Parameters
    ----------
    df
        DataFrame containing sequence data

    Returns
    -------
    xr.Dataset
        Dataset with 'sequence' and 'logit' variables, indexed by 'samples' dimension
    """
    if len(df) == 0:
        logger.warning("No data provided")
        return xr.Dataset()

    num_samples = len(df)
    logger.info(f"Generating fake logits for {num_samples} sequences")

    # Generate random probabilities for A, C, G, T using Dirichlet distribution
    rng = np.random.default_rng(42)  # For reproducible results
    logits_matrix = rng.dirichlet([1, 1, 1, 1], size=num_samples)

    nucleotides = ["A", "C", "G", "T"]
    sequences = df["sequences"].tolist()
    labels = df["label"].values

    return xr.Dataset(
        {
            "sequence": ("samples", sequences),
            "logit": (("samples", "nucleotides"), logits_matrix),
            "label": ("samples", labels),
        },
        coords={"nucleotides": nucleotides},
    )


def generate_logits(
    df: pd.DataFrame,
    model_path: str,
    device: str | torch.device,
    token_idx: int,
    batch_size: int,
) -> xr.Dataset:
    """Generate real logits using pre-trained model on single GPU.

    Parameters
    ----------
    df
        DataFrame containing sequence data
    model_path
        Hugging Face model identifier or local path
    device
        Device for running inference
    token_idx
        Index of the masked token position to score
    batch_size
        Batch size for the DataLoader

    Returns
    -------
    xr.Dataset
        Dataset with 'sequence' and 'logit' variables, indexed by 'samples' dimension
    """
    if len(df) == 0:
        logger.warning("No data provided")
        return xr.Dataset()

    logger.info(f"Loading model and tokenizer from {model_path} on {device=}")
    model, tokenizer = _load_model_and_tokenizer(model_path=model_path, device=device)

    sequences = df["sequences"].tolist()
    names = (df["chrom"].astype(str) + ":" + df["pos"].astype(str)).tolist()

    dataset = SequenceDataset(
        sequences=sequences,
        tokenizer=tokenizer,
        names=names,
        mask_token_id=token_idx,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    nucleotides = ["A", "C", "G", "T"]
    all_logits: list[NDArray[np.floating]] = []

    total_batches = len(loader)
    log_interval = max(1, total_batches // 20)  # Log every 5% of batches

    logger.info(f"Generating logits for {len(df)} sequences")
    logger.info(f"Processing {total_batches} batches with batch size {batch_size}")

    for batch_idx, batch in enumerate(loader):
        cur_ids = batch["input_ids"].to(device)
        cur_ids = cur_ids.squeeze(1)

        with torch.inference_mode():
            outputs = model(input_ids=cur_ids)
            batch_logits = outputs.logits

            logits = batch_logits[
                :,
                token_idx,
                [tokenizer.get_vocab()[nc.lower()] for nc in nucleotides],
            ]
            probs = torch.nn.functional.softmax(logits.cpu(), dim=1).numpy()
            all_logits.append(probs)

        # Log progress every 5% of batches
        if batch_idx % log_interval == 0 or batch_idx == total_batches - 1:
            progress_pct = (batch_idx + 1) / total_batches * 100
            logger.info(
                f"Processed batch {batch_idx + 1}/{total_batches} ({progress_pct:.1f}%)"
            )

    if not all_logits:
        return xr.Dataset()

    logits_matrix = np.vstack(all_logits)
    labels = df["label"].values

    return xr.Dataset(
        {
            "sequence": ("samples", sequences),
            "logit": (("samples", "nucleotides"), logits_matrix),
            "label": ("samples", labels),
        },
        coords={"nucleotides": nucleotides},
    )


@dataclass
class EvaluationResults:
    """Results from task evaluation.

    Attributes
    ----------
    roc_auc
        Area under the ROC curve
    num_samples
        Total number of samples evaluated (after filtering)
    num_positive
        Number of positive labels
    num_negative
        Number of negative labels
    num_nan
        Number of NaN scores that were filtered out
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
    num_nan: int
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
    dataset_path: str,
    dataset_subdir: str,
    dataset_split: str,
    dataset_dir: UPath,
    sample_size: int | None = None,
) -> tuple[UPath, int]:
    """Load dataset from Hugging Face and optionally downsample.

    Parameters
    ----------
    dataset_path
        HuggingFace repository ID or path for the dataset.
    dataset_subdir
        Subdirectory within the HF dataset repo (e.g., "Evolutionary_constraint", "Acceptor").
    dataset_split
        The dataset split to load (e.g., "train", "valid", "test").
    dataset_dir
        Directory to save the downsampled dataset.
    sample_size
        Number of samples to downsample to. If None, uses the full dataset.

    Returns
    -------
    tuple[UPath, int]
        Tuple of (dataset_path, num_samples).
    """

    logger.info("Loading and downsampling dataset")

    data = load_dataset(
        dataset_path,
        data_files={dataset_split: f"{dataset_subdir}/{dataset_split}.tsv"},
    )
    df = data[dataset_split].to_pandas()

    logger.info(f"Original dataset size: {len(df)}")

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Downsampled to: {len(df)} samples")

    dataset_path = dataset_dir / f"downsampled_{dataset_split}.parquet"
    write_pandas_parquet(df, dataset_path)
    logger.info(f"Saved downsampled dataset to: {dataset_path}")

    return dataset_path, len(df)


def generate_model_logits(
    dataset_path: UPath,
    output_dir: UPath,
    model_path: str,
    device: str | torch.device,
    token_idx: int,
    batch_size: int,
    simulation_mode: bool = True,
    worker_id: int | None = None,
    num_workers: int | None = None,
    lock: ray.actor.ActorHandle | None = None,
) -> UPath:
    """Generate logits using either fake random data or real model inference.

    Parameters
    ----------
    dataset_path
        Path to the parquet dataset containing columns `sequence` and a name/id column.
    output_dir
        Directory where outputs will be written. Must exist.
    model_path
        Hugging Face model identifier or local path for the masked LM.
    device
        Device for running inference (only used if simulation_mode=False).
    token_idx
        Index of the masked token position to score.
    batch_size
        Batch size for the DataLoader (only used if simulation_mode=False).
    simulation_mode
        If True, generate fake random logits for testing. If False, use real model inference.
    worker_id
        Worker ID for distributed processing (when None, processes all data).
    num_workers
        Total number of workers for distributed processing.
    lock
        Ray actor handle for HF write locking (optional).

    Returns
    -------
    UPath
        Path to the logits file written under output_dir.
    """
    if not simulation_mode:
        if worker_id is None:
            raise ValueError("worker_id must be provided")
        if num_workers is None:
            raise ValueError("num_workers must be provided")

    df = read_pandas_parquet(dataset_path)
    _validate_sequence_lengths(df)

    logger.info(f"Processing {len(df)} sequences")

    output_dir.mkdir(parents=True, exist_ok=True)

    if simulation_mode:
        logger.info("Generating fake logits for testing (simulation_mode=True)")
        logits = simulate_logits(df)
        logits_path = output_dir / "logits.nc"
        write_xarray_netcdf(logits, logits_path)
    else:
        logger.info("Generating real logits using pre-trained model")

        # Filter data for distributed processing
        df = df.iloc[worker_id::num_workers]
        logger.info(
            f"Worker {worker_id + 1}/{num_workers}: processing {len(df)} sequences"
        )

        logits = generate_logits(
            df=df,
            model_path=model_path,
            device=device,
            token_idx=token_idx,
            batch_size=batch_size,
        )

        logits_path = output_dir / f"logits_{worker_id:05d}.nc"

        with lock_hf_path(logits_path, lock) as locked_path:
            logger.info(f"Writing logits to netcdf file at {locked_path}")
            write_xarray_netcdf(logits, locked_path)

    logger.info(f"Generated logits for {logits.sizes['samples']} sequences")

    return logits_path


def compute_plantcad_scores(
    logits: xr.Dataset,
    token_idx: int,
) -> pd.DataFrame:
    """Compute scores by selecting the reference nucleotide probability.

    Parameters
    ----------
    logits
        Xarray dataset containing 'sequence', 'logit', and 'label' variables.
    token_idx
        Index of the masked token position used to choose the reference base.

    Returns
    -------
    pd.DataFrame
        DataFrame with sequence, labels, and computed PlantCAD scores.
    """

    logger.info("Computing PlantCAD scores from logits")

    sequences = logits.sequence.values
    logits_matrix = logits.logit.values
    nucleotides = logits.nucleotides.values
    labels = logits.label.values

    # Create mapping from nucleotide to index
    nuc_to_idx = {nuc: i for i, nuc in enumerate(nucleotides)}

    ref_nucleotides = pd.Series([seq[token_idx] for seq in sequences])
    logger.info("Reference nucleotide distribution:")
    ref_counts = ref_nucleotides.value_counts()
    for nuc, count in ref_counts.items():
        logger.info(f"  {nuc}: {count}")

    # Compute scores by selecting reference nucleotide probability
    scores = []
    for i, ref_nuc in enumerate(ref_nucleotides):
        if ref_nuc in nuc_to_idx:
            score = logits_matrix[i, nuc_to_idx[ref_nuc]]
        else:
            score = np.nan
        scores.append(score)

    scores = np.array(scores)

    # Create scored dataset
    df = pd.DataFrame(
        {
            "sequence": sequences,
            "label": labels,
            "plantcad_score": scores,
        }
    )

    logger.info(f"Generated scores for {len(scores)} samples")
    logger.info(f"Non-zero scores: {sum(scores != 0)}")

    return df


def compute_roc_auc(
    y_true: NDArray[np.floating], y_score: NDArray[np.floating]
) -> EvaluationResults:
    """Compute ROC AUC and related metrics."""

    logger.info("Computing ROC AUC score")

    # Filter out NaN values
    valid_mask = ~np.isnan(y_score)
    num_nan = int((~valid_mask).sum())

    if num_nan > 0:
        pct = num_nan / len(y_score) * 100
        logger.warning(f"Excluding {num_nan} NaN scores ({pct:.1f}%)")
        y_true, y_score = y_true[valid_mask], y_score[valid_mask]

        if len(y_score) == 0:
            raise ValueError("All scores are NaN")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    results = EvaluationResults(
        roc_auc=roc_auc,
        num_samples=len(y_true),
        num_positive=int(sum(y_true)),
        num_negative=int(sum(1 - y_true)),
        num_nan=num_nan,
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
