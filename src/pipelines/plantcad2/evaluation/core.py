from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from numpy.typing import NDArray
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm
from upath import UPath

from src.io.api import resolve_local_path
from src.pipelines.plantcad2.evaluation.config import (
    CoreNonCoreTaskConfig,
    EvoConsTaskConfig,
    ModelType,
    MotifInferenceMode,
    MotifTaskConfig,
    MultiMaskTaskConfig,
    StructuralVariantTaskConfig,
    TaskConfig,
)
from src.utils.hf_utils import load_hf_dataset

logger = logging.getLogger("ray")

TaskConfigT = TypeVar("TaskConfigT", bound=TaskConfig)

# =============================================================================
# Constants
# =============================================================================

NUCLEOTIDES = ("A", "C", "G", "T")
NUCLEOTIDES_LOWER = tuple(n.lower() for n in NUCLEOTIDES)
NUCLEOTIDE_TO_INDEX = {b: i for i, b in enumerate(NUCLEOTIDES)}
N_NUCLEOTIDES = len(NUCLEOTIDES)

# Complement mapping
COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")
RC_PROB_FLIP = [3, 2, 1, 0]  # Maps [A,C,G,T] → [T,G,C,A]


# =============================================================================
# Common Utilities
# =============================================================================


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def _flip_rc_probs(probs: NDArray[np.floating]) -> NDArray[np.floating]:
    """Flip RC probs: reverse position order and swap complement nucleotides."""
    if probs.ndim != 3:
        raise ValueError(f"Expected 3D array, got {probs.shape=}")
    if probs.shape[2] != N_NUCLEOTIDES:
        raise ValueError(f"Expected last dim size {N_NUCLEOTIDES}, got {probs.shape=}")
    return probs[:, ::-1, :][:, :, RC_PROB_FLIP]


def _validate_device(device: str) -> str:
    if device == "cpu":
        return device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{device}' requested but CUDA is not available."
            )
        return device
    raise ValueError(f"Unsupported device: {device}. Use 'cpu' or 'cuda[:N]'.")


def _optimal_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major >= 8:
        return torch.bfloat16
    if major >= 6:
        return torch.float16
    return torch.float32


# =============================================================================
# Model Loading
# =============================================================================

MODEL_TYPE_TO_CLASS: dict[ModelType, type[PreTrainedModel]] = {
    ModelType.mlm: AutoModelForMaskedLM,
    ModelType.clm: AutoModelForCausalLM,
}


def _resolve_model_path(model_path: str) -> str | None:
    """Resolve model path to local filesystem path if possible.

    Returns the local path for local paths or HF repo IDs, None for remote paths.
    """
    upath = UPath(model_path)
    if not upath.protocol:
        # Path is for local filesystem or HF repo id
        return model_path
    if upath.protocol == "file":
        # Path is local fsspec url beginning with file://
        return upath.path
    # Path cannot be resolved to the local filesystem
    return None


def _is_remote_model_path(model_path: str) -> bool:
    """Check if model path is a remote path that needs downloading."""
    return _resolve_model_path(model_path) is None


def _download_model_checkpoint(model_path: str) -> str:
    """Download a remote model checkpoint to local cache if not already cached.

    This should be called once per node before workers start to avoid
    race conditions on the local filesystem.

    Parameters
    ----------
    model_path : str
        Remote model path (e.g., gs://bucket/path or s3://bucket/path)

    Returns
    -------
    str
        Local path to the downloaded model checkpoint
    """
    logger.info(f"Downloading remote model checkpoint: {model_path}")
    local_path = str(
        resolve_local_path(UPath(model_path), kind="directory", force=False)
    )
    logger.info(f"Model checkpoint at: {local_path}")
    return local_path


def _load_model(
    model_path: str, device: str, model_type: ModelType, subfolder: str = ""
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if model_type not in MODEL_TYPE_TO_CLASS:
        raise ValueError(f"Unsupported model type: {model_type}")
    model_cls = MODEL_TYPE_TO_CLASS[model_type]
    dtype = _optimal_dtype(device)
    logger.info(
        f"Loading {model_type.upper()} model {model_path} with {dtype=}, {device=}"
    )

    # Resolve to local path - remote models should have been pre-downloaded
    # before workers start
    local_path = _resolve_model_path(model_path)
    if local_path is None:
        # For remote paths, resolve to cached local path (should already exist)
        # Use force=False to avoid re-downloading if already cached
        local_path = str(
            resolve_local_path(UPath(model_path), kind="directory", force=False)
        )
        logger.info(f"Using cached model checkpoint at: {local_path}")

    model = model_cls.from_pretrained(
        local_path, subfolder=subfolder, trust_remote_code=True, torch_dtype=dtype
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        local_path, subfolder=subfolder, trust_remote_code=True
    )
    return model, tokenizer


# =============================================================================
# Data Loading
# =============================================================================


class MultiMaskDataset(TorchDataset):
    def __init__(
        self,
        sequences: pd.Series,
        tokenizer: PreTrainedTokenizer,
        mask_idx: Sequence[int],
    ):
        self.sequences = sequences.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mask_idx = list(mask_idx)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        seq = self.sequences.iloc[i]
        enc = self.tokenizer(
            seq,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc["input_ids"]
        if input_ids.size(1) <= max(self.mask_idx):
            raise ValueError("mask index out of range")
        unmasked_ids = input_ids.clone()
        input_ids[0, self.mask_idx] = self.tokenizer.mask_token_id
        return {"masked_ids": input_ids, "unmasked_ids": unmasked_ids}


@dataclass
class TaskData:
    """Pre-fetched task data containing cached file info."""

    dataset_cache_files: list[dict]
    model_cache_path: str | None


def fetch_task_data(config: TaskConfigT) -> TaskData:
    """Pre-fetch task data and model checkpoint to local cache.

    This function runs once per node before workers start to avoid
    unnecessary cloud storage I/O and potential local filesystem race
    conditions between workers on the same nodes.

    Parameters
    ----------
    config : TaskConfigT
        Task configuration containing dataset and model info

    Returns
    -------
    TaskData
        Contains dataset cache file info and local model path (if downloaded)
    """
    task_desc = f"[task={config.task}, split={config.split}]"

    # Pre-fetch HF dataset to local cache
    logger.info(
        f"{task_desc} Pre-fetching HF task data from repository {config.repo_id} ..."
    )
    dataset = load_hf_dataset(config.repo_id, config.task, split=config.split)
    assert isinstance(dataset, Dataset)
    logger.info(
        f"{task_desc} Pre-fetched {len(dataset)} examples "
        f"from repository {config.repo_id} to local cache:\n{dataset.cache_files}"
    )

    # Pre-download remote model checkpoint to local cache
    model_cache_path: str | None = None
    if _is_remote_model_path(config.model_path):
        logger.info(
            f"{task_desc} Pre-downloading remote model checkpoint: {config.model_path}"
        )
        model_cache_path = _download_model_checkpoint(config.model_path)
        logger.info(f"{task_desc} Model checkpoint cached at: {model_cache_path}")
    else:
        model_cache_path = config.model_path

    # Verify model can be loaded, which will trigger a download if not already cached
    logger.info(f"{task_desc} Verifying model at: {model_cache_path}")
    _load_model(
        model_cache_path,
        device=config.device,
        model_type=config.model_type,
        subfolder=config.model_subfolder,
    )
    logger.info(f"{task_desc} Successfully verified model")

    return TaskData(
        dataset_cache_files=dataset.cache_files,
        model_cache_path=model_cache_path,
    )


def center_crop_sequences(
    df: pd.DataFrame, seq_column: str, seq_length: int, model_context_length: int
) -> pd.DataFrame:
    """Center-crop sequences if model context is shorter than sequence length."""
    if model_context_length >= seq_length:
        return df

    if seq_length % 2 != 0 or model_context_length % 2 != 0:
        raise ValueError(
            f"Both lengths must be even for center-cropping: "
            f"seq_length={seq_length}, model_context_length={model_context_length}"
        )

    start = (seq_length - model_context_length) // 2
    end = start + model_context_length
    left_removed, right_removed = start, seq_length - end
    assert left_removed == right_removed, (
        f"Unequal tokens removed: {left_removed=}, {right_removed=}"
    )
    logger.info(
        f"Center-cropping sequences [{start}:{end}] "
        f"({seq_length=}, {model_context_length=})"
    )
    df = df.copy()
    df[seq_column] = df[seq_column].str[start:end]
    return df


def load_task_data(
    config: TaskConfigT,
    *,
    worker_id: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    dataset = load_hf_dataset(config.repo_id, config.task, split=config.split)
    assert isinstance(dataset, Dataset)
    dataset = dataset.map(lambda _, idx: {"example_idx": idx}, with_indices=True)

    if config.sample_rate is not None:
        n = int(len(dataset) * config.sample_rate)
        dataset = dataset.shuffle(seed=config.sample_seed).select(range(n))
    if config.sample_max_size is not None and len(dataset) > config.sample_max_size:
        dataset = dataset.select(range(config.sample_max_size))

    if worker_id is not None and num_workers is not None:
        dataset = dataset.shard(num_shards=num_workers, index=worker_id)
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)
    if "example_idx" not in df.columns:
        raise KeyError("example_idx column missing after dataset preparation")
    df["example_idx"] = df["example_idx"].astype(int)

    df = center_crop_sequences(
        df=df,
        seq_column=config.seq_column,
        seq_length=config.seq_length,
        model_context_length=config.model_context_length,
    )
    return df


# =============================================================================
# MLM-Specific Functions
# =============================================================================


def _compute_mlm_probs_for_positions(
    df: pd.DataFrame,
    config: MultiMaskTaskConfig,
    desc: str,
) -> Tuple[NDArray[np.int_], NDArray[np.floating], list[int]]:
    """Compute masked probabilities for specified positions using MLM."""
    ids, positions, dev, model, tokenizer = _prepare_masked_inference(df, config)
    probs = _run_masked_probs(
        sequences=df[config.seq_column],
        positions=positions,
        tokenizer=tokenizer,
        model=model,
        device=dev,
        batch_size=config.batch_size,
        model_type=config.model_type,
        desc=desc,
    )
    if len(ids) != len(probs):
        raise ValueError(
            f"Length mismatch: ids has {len(ids)} elements but probs has shape {probs.shape}"
        )
    return ids, probs, positions


def _compute_mlm_sv_scores(
    df: pd.DataFrame,
    config: StructuralVariantTaskConfig,
) -> StructuralVariantResult:
    """Compute structural variant scores using MLM pseudo-perplexity."""
    dev = _validate_device(config.device)
    model, tokenizer = _load_model(
        model_path=config.model_path,
        device=dev,
        model_type=ModelType.mlm,
        subfolder=config.model_subfolder,
    )
    ref_probs = _unmasked_probs(
        sequences=df["RefSeq"],
        tokenizer=tokenizer,
        model=model,
        device=dev,
        batch_size=config.batch_size,
        desc=f"[{config.task}/{config.split}] Ref (unmasked)",
    )
    mut_probs = _unmasked_probs(
        sequences=df["MutSeq"],
        tokenizer=tokenizer,
        model=model,
        device=dev,
        batch_size=config.batch_size,
        desc=f"[{config.task}/{config.split}] Mut (unmasked)",
    )
    scores = structural_variant_boundary_scores(
        df, ref_probs, mut_probs, config.flanking
    )
    return StructuralVariantResult(
        example_idx=df["example_idx"].to_numpy(),
        scores=scores,
        ref_probs=ref_probs,
        mut_probs=mut_probs,
    )


def _masked_probs(
    model: AutoModelForMaskedLM,
    tokenizer: PreTrainedTokenizer,
    loader: DataLoader,
    device: str,
    *,
    masks_per_sequence: int,
    model_type: ModelType,
    desc: str,
) -> NDArray[np.floating]:
    """Compute nucleotide probabilities at masked positions.

    Parameters
    ----------
    model : AutoModelForMaskedLM
        Masked language model for inference
    tokenizer : PreTrainedTokenizer
        Tokenizer matching the model
    loader : DataLoader
        DataLoader providing batches with 'masked_ids' and 'unmasked_ids' keys
    device : str
        Device for model inference (e.g., 'cuda', 'cpu')
    masks_per_sequence : int
        Number of masked positions per sequence
    model_type : ModelType
        Model architecture type (mlm or clm)
    desc : str
        Description for progress bar

    Returns
    -------
    NDArray[np.floating]
        Probability array of shape (num_sequences, masks_per_sequence, N_NUCLEOTIDES)
        where N_NUCLEOTIDES is typically 4 (A, C, G, T)
    """
    idxs = [tokenizer.get_vocab()[n] for n in NUCLEOTIDES_LOWER]
    grouped_probs: list[NDArray[np.floating]] = []
    for batch in tqdm(loader, desc=desc):
        masked_ids = batch["masked_ids"].to(device).squeeze(1)
        unmasked_ids = batch["unmasked_ids"].to(device).squeeze(1)
        assert masked_ids.ndim == 2, (
            f"Expected 2D masked_ids (batch, seq_len), got shape {masked_ids.shape}"
        )
        batch_size = masked_ids.size(0)

        # CLM uses unmasked_ids for inference (needs real tokens for context);
        # MLM uses masked_ids (mask tokens signal positions to predict)
        inference_ids = unmasked_ids if model_type == ModelType.clm else masked_ids

        with torch.inference_mode():
            logits = model(input_ids=inference_ids).logits

        # logits shape: (batch, seq_len, vocab_size)
        assert logits.ndim == 3, (
            f"Expected 3D logits (batch, seq_len, vocab_size), got shape {logits.shape}"
        )
        vocab_size = logits.size(-1)

        # For CLM, logits[i] predicts token[i+1], so shift right by 1 to align
        # with mask positions (prepend nan column so logits[i] corresponds to token[i])
        if model_type == ModelType.clm:
            nan_col = torch.full(
                (logits.size(0), 1, logits.size(2)), float("nan"), device=device
            )
            shifted_logits = torch.cat([nan_col, logits[:, :-1, :]], dim=1)
        else:
            shifted_logits = logits

        # Transform masked_pos: (batch, seq_len) bool -> (batch, seq_len, vocab_size)
        # unsqueeze(-1) adds singleton vocab dim, expand replicates it to match logits
        # Always use masked_ids to identify positions of interest
        masked_pos = (
            (masked_ids == tokenizer.mask_token_id)
            .unsqueeze(-1)
            .expand(-1, -1, shifted_logits.size(-1))
        )

        # Find logits for masked positions
        masked_logits = torch.masked_select(shifted_logits, masked_pos)
        assert masked_logits.shape == (
            expected_shape := (batch_size * masks_per_sequence * vocab_size,)
        ), f"Expected masked_logits shape {expected_shape}, got {masked_logits.shape}"

        # Reshape to (batch_size * masks_per_sequence, vocab_size)
        masked_logits = masked_logits.view(-1, vocab_size)
        assert masked_logits.shape == (
            expected_shape := (
                batch_size * masks_per_sequence,
                vocab_size,
            )
        ), f"Expected masked_logits shape {expected_shape}, got {masked_logits.shape}"
        assert not torch.isnan(masked_logits).any(), "Masked logits contains NaN values"

        probs = torch.softmax(masked_logits[:, idxs].float(), dim=-1).cpu().numpy()
        grouped_probs.append(probs)

    if not grouped_probs:
        return np.zeros((0, masks_per_sequence, len(idxs)), dtype=np.float32)

    stacked = np.vstack(grouped_probs)
    if stacked.shape[0] % masks_per_sequence != 0:
        raise ValueError(
            "Number of masked positions not divisible by masks_per_sequence"
        )
    result = stacked.reshape(-1, masks_per_sequence, stacked.shape[-1])
    assert result.ndim == 3, f"Expected 3D array, got shape {result.shape}"
    assert result.shape[1] == masks_per_sequence, (
        f"Expected masks_per_sequence={masks_per_sequence}, got shape {result.shape}"
    )
    assert result.shape[2] == N_NUCLEOTIDES, (
        f"Expected {N_NUCLEOTIDES} nucleotides, got shape {result.shape}"
    )
    return result


def _unmasked_probs(
    sequences: pd.Series,
    tokenizer: PreTrainedTokenizer,
    model: AutoModelForMaskedLM,
    device: str,
    batch_size: int,
    *,
    desc: str,
) -> NDArray[np.floating]:
    """Compute nucleotide probabilities at all positions (unmasked inference).

    Parameters
    ----------
    sequences : pd.Series
        Series of DNA sequences to process
    tokenizer : PreTrainedTokenizer
        Tokenizer matching the model
    model : AutoModelForMaskedLM
        Masked language model for inference
    device : str
        Device for model inference (e.g., 'cuda', 'cpu')
    batch_size : int
        Number of sequences to process per batch
    desc : str
        Description for progress bar

    Returns
    -------
    NDArray[np.floating]
        Probability array of shape (num_sequences, seq_len, N_NUCLEOTIDES)
        where N_NUCLEOTIDES is typically 4 (A, C, G, T)
    """
    idxs = [tokenizer.get_vocab()[n] for n in NUCLEOTIDES_LOWER]
    seqs = sequences.astype(str).tolist()
    all_probs: Optional[NDArray[np.floating]] = None
    for i in tqdm(range(0, len(seqs), batch_size), desc=desc):
        batch = seqs[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=False,
            padding=False,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc["input_ids"].to(device)
        with torch.inference_mode():
            out = model(input_ids=input_ids)
        logits = out.logits[..., idxs]
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        if all_probs is None:
            seq_len = probs.shape[1]
            all_probs = np.zeros((len(seqs), seq_len, N_NUCLEOTIDES), dtype=np.float32)
        all_probs[i : i + len(batch), :, :] = probs
    if all_probs is None:
        return np.zeros((0, 0, len(idxs)), dtype=np.float32)
    assert all_probs.ndim == 3, f"Expected 3D array, got shape {all_probs.shape}"
    assert all_probs.shape[0] == len(seqs), (
        f"Expected num_sequences={len(seqs)}, got shape {all_probs.shape}"
    )
    assert all_probs.shape[2] == len(idxs), (
        f"Expected {len(idxs)} nucleotides, got shape {all_probs.shape}"
    )
    return all_probs


# =============================================================================
# CLM-Specific Functions (Stubs)
# =============================================================================


def _compute_clm_probs_for_positions(
    df: pd.DataFrame,
    config: MultiMaskTaskConfig,
    desc: str,
    mode: MotifInferenceMode,
) -> Tuple[NDArray[np.int_], NDArray[np.floating], list[int]]:
    """Compute CLM probabilities using forward, reverse complement, or averaged inference.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing sequences
    config : MultiMaskTaskConfig
        Task configuration
    desc : str
        Description for progress bar
    mode : MotifInferenceMode
        Inference mode: fwd_only (forward only), rc_only (reverse complement only),
        or fwd_rc_avg (average of both)
    """
    ids, positions, dev, model, tokenizer = _prepare_masked_inference(df, config)
    sequences = df[config.seq_column]
    seq_len = len(sequences.iloc[0])

    fwd_probs: NDArray[np.floating] | None = None
    rc_probs: NDArray[np.floating] | None = None

    # Forward pass
    if mode in (MotifInferenceMode.fwd_only, MotifInferenceMode.fwd_rc_avg):
        fwd_probs = _run_masked_probs(
            sequences=sequences,
            positions=positions,
            tokenizer=tokenizer,
            model=model,
            device=dev,
            batch_size=config.batch_size,
            model_type=config.model_type,
            desc=f"{desc} (fwd)",
        )

    # Reverse complement pass
    if mode in (MotifInferenceMode.rc_only, MotifInferenceMode.fwd_rc_avg):
        rc_sequences = sequences.apply(reverse_complement)
        rc_positions = [seq_len - 1 - p for p in reversed(positions)]
        assert all(p >= 0 for p in rc_positions), (
            f"Reverse complement positions must be non-negative: {rc_positions}"
        )
        rc_probs = _run_masked_probs(
            sequences=rc_sequences,
            positions=rc_positions,
            tokenizer=tokenizer,
            model=model,
            device=dev,
            batch_size=config.batch_size,
            model_type=config.model_type,
            desc=f"{desc} (rc)",
        )

    # Compute final probabilities based on mode
    if mode == MotifInferenceMode.fwd_only:
        assert fwd_probs is not None
        probs = fwd_probs
    elif mode == MotifInferenceMode.rc_only:
        assert rc_probs is not None
        probs = _flip_rc_probs(rc_probs)
    elif mode == MotifInferenceMode.fwd_rc_avg:
        assert fwd_probs is not None and rc_probs is not None
        if fwd_probs.shape != rc_probs.shape:
            raise AssertionError(
                f"Forward and RC prob shape mismatch: {fwd_probs.shape} != {rc_probs.shape}"
            )
        # TODO: average logits or renormalize
        probs = (fwd_probs + _flip_rc_probs(rc_probs)) / 2
    else:
        raise ValueError(f"Unsupported motif inference mode: {mode}")

    if len(ids) != len(probs):
        raise ValueError(
            f"Length mismatch: ids has {len(ids)} elements but probs has shape {probs.shape}"
        )
    return ids, probs, positions


def _compute_clm_sv_scores(
    df: pd.DataFrame,
    config: StructuralVariantTaskConfig,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute structural variant scores using CLM perplexity.

    Returns perplexity-based scores comparing reference and mutant sequences.
    """
    raise NotImplementedError("CLM SV scores not yet implemented")


# =============================================================================
# Shared Scoring Utilities
# =============================================================================


def structural_variant_boundary_scores(
    df: pd.DataFrame,
    ref_probs: NDArray[np.floating],
    mut_probs: NDArray[np.floating],
    flanking: int,
) -> NDArray[np.floating]:
    base_idx = NUCLEOTIDE_TO_INDEX
    length = ref_probs.shape[1]
    if length == 0:
        return np.zeros(len(df), dtype=float)
    center0 = length // 2
    mut_left0 = list(range(center0 - flanking, center0))
    mut_right0 = list(range(center0, center0 + flanking))
    scores = np.zeros(len(df), dtype=float)
    for i, row in enumerate(df.itertuples(index=False)):
        left1 = int(getattr(row, "left"))
        right1 = int(getattr(row, "right"))
        left_end = left1 - 1
        left_ref = list(range(left_end - (flanking - 1), left_end + 1))
        right_start = right1 + 1
        right_ref = list(range(right_start, right_start + flanking))

        vals: list[float] = []
        mut_full = getattr(row, "MutSeq")
        mut_start0 = mut_left0[0]
        center_seq = mut_full[mut_start0 : mut_start0 + 2 * flanking]

        for k in range(flanking):
            p_ref1 = left_ref[k]
            p_mut0 = mut_left0[k]
            b = center_seq[k].upper()
            if b in base_idx:
                j = base_idx[b]
                r = ref_probs[i, p_ref1 - 1, j]
                m = mut_probs[i, p_mut0, j]
                vals.append(float(np.log(max(m, 1e-12) / max(r, 1e-12))))
            else:
                vals.append(0.0)

            p_ref1 = right_ref[k]
            p_mut0 = mut_right0[k]
            b = center_seq[flanking + k].upper()
            if b in base_idx:
                j = base_idx[b]
                r = ref_probs[i, p_ref1 - 1, j]
                m = mut_probs[i, p_mut0, j]
                vals.append(float(np.log(max(m, 1e-12) / max(r, 1e-12))))
            else:
                vals.append(0.0)
        scores[i] = float(np.mean(vals)) * -1
    return scores


def _validate_sequences(
    df: pd.DataFrame, seq_column: str, mask_token_indexes: Sequence[int]
) -> None:
    if len(df) == 0:
        return

    # Check that all sequences have the same length
    sequences = df[seq_column].astype(str)
    lengths = sequences.str.len()
    unique_lengths = lengths.unique()

    if len(unique_lengths) > 1:
        raise ValueError(
            f"All sequences must have the same length; found {len(unique_lengths)} "
            f"different lengths: {sorted(unique_lengths)}"
        )

    # Check that mask token indexes are centered
    seq_len = unique_lengths[0]
    n_masks = len(mask_token_indexes)
    start = (seq_len - n_masks) // 2
    expected = list(range(start, start + n_masks))

    if list(mask_token_indexes) != expected:
        raise ValueError(
            f"Mask token indexes must be centered in sequences of length {seq_len}; "
            f"expected {expected}, got {list(mask_token_indexes)}"
        )


def _compute_probs_for_positions(
    df: pd.DataFrame,
    config: MultiMaskTaskConfig,
    desc: str,
) -> Tuple[NDArray[np.int_], NDArray[np.floating], list[int]]:
    """Compute probabilities at specified positions using MLM or CLM.

    Dispatches to the appropriate implementation based on config.model_type.
    """
    if config.model_type == ModelType.mlm:
        return _compute_mlm_probs_for_positions(df, config, desc)
    if config.model_type == ModelType.clm:
        return _compute_clm_probs_for_positions(
            df, config, desc, mode=config.model_motif_inference_mode
        )
    raise ValueError(f"Unsupported model type: {config.model_type}")


def _prepare_masked_inference(
    df: pd.DataFrame,
    config: MultiMaskTaskConfig,
) -> Tuple[NDArray[np.int_], list[int], str, PreTrainedModel, PreTrainedTokenizer]:
    """Shared setup for masked inference (MLM and CLM)."""
    ids = df["example_idx"].to_numpy()
    positions = [int(x) for x in config.mask_token_indexes]
    _validate_sequences(df, config.seq_column, positions)
    dev = _validate_device(config.device)
    model, tokenizer = _load_model(
        model_path=config.model_path,
        device=dev,
        model_type=config.model_type,
        subfolder=config.model_subfolder,
    )
    return ids, positions, dev, model, tokenizer


def _run_masked_probs(
    sequences: pd.Series,
    positions: list[int],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
    batch_size: int,
    model_type: ModelType,
    desc: str,
) -> NDArray[np.floating]:
    """Run masked inference on sequences and return probabilities."""
    dataset = MultiMaskDataset(sequences, tokenizer, positions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return _masked_probs(
        model=model,
        tokenizer=tokenizer,
        loader=loader,
        device=device,
        masks_per_sequence=len(positions),
        model_type=model_type,
        desc=desc,
    )


def compute_true_tokens_from_seq(
    sequences: pd.Series, positions: Sequence[int]
) -> NDArray[np.str_]:
    tokens: list[str] = []
    for seq in sequences.astype(str).tolist():
        for pos in positions:
            tokens.append(seq[pos].upper())
    return np.array(tokens)


def token_accuracy_from_probs(
    probs: NDArray[np.floating], true_tokens: NDArray[np.str_]
) -> float:
    pred_idx = probs.argmax(axis=1)
    nuc = np.array(NUCLEOTIDES)
    pred = nuc[pred_idx]
    valid = np.isin(true_tokens, nuc)
    if not valid.any():
        return 0.0
    return float((pred[valid] == true_tokens[valid]).mean())


def motif_accuracy_from_probs(
    probs: NDArray[np.floating],
    true_tokens: NDArray[np.str_],
    motif_len: int,
) -> float:
    nuc = np.array(NUCLEOTIDES)
    pred = nuc[probs.argmax(axis=1)]
    n = len(true_tokens)
    if n % motif_len != 0:
        raise ValueError("Total masked positions not divisible by motif length")
    pred_groups = pred.reshape(-1, motif_len)
    true_groups = true_tokens.reshape(-1, motif_len)
    valid_groups = np.all(np.isin(true_groups, nuc), axis=1)
    if not valid_groups.any():
        return 0.0
    return float(
        np.all(pred_groups[valid_groups] == true_groups[valid_groups], axis=1).mean()
    )


def compute_ref_auroc(
    df: pd.DataFrame,
    probs: NDArray[np.floating],
    token_idx: int,
    seq_col: str,
) -> float:
    ref_series = df[seq_col].str[token_idx].str.upper()
    y_true = df["label"].astype(int)
    nuc = list(NUCLEOTIDES)
    ref_map = {b: i for i, b in enumerate(nuc)}
    scores = np.zeros(len(df), dtype=float)
    valid = ref_series.isin(nuc)
    idx = valid[valid].index
    scores[idx] = probs[valid.values, [ref_map[b] for b in ref_series[valid]]]
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(auc(fpr, tpr))


def reference_base_scores(
    df: pd.DataFrame,
    probs: NDArray[np.floating],
    token_idx: int,
    seq_col: str,
) -> NDArray[np.floating]:
    ref_series = df[seq_col].str[token_idx].str.upper()
    nuc = list(NUCLEOTIDES)
    ref_map = {b: i for i, b in enumerate(nuc)}
    scores = np.zeros(len(df), dtype=float)
    valid = ref_series.isin(nuc)
    idx = valid[valid].index
    scores[idx] = probs[valid.values, [ref_map[b] for b in ref_series[valid]]]
    return scores


def average_true_probabilities(
    probs: NDArray[np.floating],
    true_tokens: NDArray[np.str_],
    motif_len: int,
    agg: Literal["mean", "product"] = "product",
) -> NDArray[np.floating]:
    nuc = np.array(NUCLEOTIDES)
    n = len(true_tokens)
    if n % motif_len != 0:
        raise ValueError("Total masked positions not divisible by motif length")
    idx_map = {b: i for i, b in enumerate(nuc)}
    idxs = np.array([idx_map.get(t, -1) for t in true_tokens], dtype=int)
    row_idx = np.arange(probs.shape[0])
    token_probs = np.zeros(probs.shape[0], dtype=float)
    valid = idxs >= 0
    token_probs[valid] = probs[row_idx[valid], idxs[valid]]
    grouped = token_probs.reshape(-1, motif_len)
    if agg == "mean":
        return grouped.mean(axis=1)
    elif agg == "product":
        return grouped.prod(axis=1)
    else:
        raise ValueError(f"Unsupported aggregation method: {agg}")


# =============================================================================
# Primary Scoring Functions
# =============================================================================


def compute_evo_cons_probs(
    df: pd.DataFrame,
    config: EvoConsTaskConfig,
) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
    if len(df) == 0:
        return np.zeros(0, dtype=int), np.zeros((0, N_NUCLEOTIDES), dtype=np.float32)
    desc = f"[{config.task}/{config.split}] Masked logits @ {config.mask_token_index}"
    ids, probs, _ = _compute_probs_for_positions(df, config, desc)
    # Reshape from (num_examples, 1, N_NUCLEOTIDES) to (num_examples, N_NUCLEOTIDES)
    probs = probs.reshape(-1, probs.shape[-1])
    if len(ids) != len(probs):
        raise ValueError(
            f"Length mismatch: ids has {len(ids)} elements but probs has shape {probs.shape}"
        )
    return ids, probs


def compute_motif_probs(
    df: pd.DataFrame,
    config: MotifTaskConfig,
) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
    if len(df) == 0:
        motif_len = len(config.mask_token_indexes)
        return (
            np.zeros(0, dtype=int),
            np.zeros((0, motif_len, N_NUCLEOTIDES), dtype=np.float32),
        )
    desc = f"[{config.task}/{config.split}] Masked logits motif_len={len(config.mask_token_indexes)}"
    ids, probs, _ = _compute_probs_for_positions(df, config, desc)
    if len(ids) != len(probs):
        raise ValueError(
            f"Length mismatch: ids has {len(ids)} elements but probs has shape {probs.shape}"
        )
    return ids, probs


@dataclass
class StructuralVariantResult:
    example_idx: NDArray[np.int_]
    scores: NDArray[np.floating]
    ref_probs: NDArray[np.floating]
    mut_probs: NDArray[np.floating]


def compute_sv_scores(
    df: pd.DataFrame,
    config: StructuralVariantTaskConfig,
) -> StructuralVariantResult:
    """Compute structural variant scores using MLM or CLM.

    Dispatches to the appropriate implementation based on config.model_type.
    """
    if len(df) == 0:
        return StructuralVariantResult(
            example_idx=np.zeros(0, dtype=int),
            scores=np.zeros(0, dtype=float),
            ref_probs=np.zeros((0, 0, N_NUCLEOTIDES), dtype=np.float32),
            mut_probs=np.zeros((0, 0, N_NUCLEOTIDES), dtype=np.float32),
        )

    required = ["RefSeq", "MutSeq", "left", "right", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if config.model_type == ModelType.mlm:
        return _compute_mlm_sv_scores(df, config)
    if config.model_type == ModelType.clm:
        raise NotImplementedError("CLM SV scores not yet implemented")
    raise ValueError(f"Unsupported model type: {config.model_type}")


def compute_core_noncore_scores(
    df: pd.DataFrame,
    config: CoreNonCoreTaskConfig,
) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
    if len(df) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    desc = f"[{config.task}/{config.split}] Masked logits motif_len={config.motif_len}"
    ids, probs, positions = _compute_probs_for_positions(df, config, desc)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    true_tokens = compute_true_tokens_from_seq(df[config.seq_column], positions)
    scores = average_true_probabilities(flat_probs, true_tokens, config.motif_len)
    if len(ids) != len(scores):
        raise ValueError(
            f"Length mismatch: ids has {len(ids)} elements but scores has {len(scores)} elements"
        )
    return ids, scores
