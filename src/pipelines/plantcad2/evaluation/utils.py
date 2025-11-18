from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from numpy.typing import NDArray
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm

from src.pipelines.plantcad2.evaluation.config import (
    CoreNonCoreTaskConfig,
    EvoConsTaskConfig,
    MotifTaskConfig,
    StructuralVariantTaskConfig,
    TaskConfig,
)
from src.utils.hf_utils import load_hf_dataset

logger = logging.getLogger("ray")

TaskConfigT = TypeVar("TaskConfigT", bound=TaskConfig)

NUCLEOTIDES = ("A", "C", "G", "T")
NUCLEOTIDES_LOWER = tuple(n.lower() for n in NUCLEOTIDES)
NUCLEOTIDE_TO_INDEX = {b: i for i, b in enumerate(NUCLEOTIDES)}
N_NUCLEOTIDES = len(NUCLEOTIDES)


def _require_cuda(device: str) -> str:
    if not (torch.cuda.is_available() and device.startswith("cuda")):
        raise RuntimeError(
            "CUDA is required for zero-shot evaluation; CPU is not supported. "
            "Set device='cuda:0' (or similar) and ensure a CUDA GPU is available."
        )
    return device


def _optimal_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major >= 8:
        return torch.bfloat16
    if major >= 6:
        return torch.float16
    return torch.float32


def _load_model(
    model_name: str, device: str
) -> Tuple[AutoModelForMaskedLM, PreTrainedTokenizer]:
    dtype = _optimal_dtype()
    logger.info(f"Loading model {model_name} with {dtype=}, {device=}")
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()  # Redundant with from_pretrained but included for clarity
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


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
        input_ids[0, self.mask_idx] = self.tokenizer.mask_token_id
        return {"masked_ids": input_ids}


class SingleMaskDataset(MultiMaskDataset):
    def __init__(
        self, sequences: pd.Series, tokenizer: PreTrainedTokenizer, token_idx: int
    ):
        super().__init__(sequences, tokenizer, [token_idx])


def fetch_task_data(config: TaskConfigT) -> list[dict]:
    logger.info(
        f"[task={config.task}, split={config.split}] Pre-fetching HF task data from repository {config.repo_id} ..."
    )
    dataset = load_hf_dataset(config.repo_id, config.task, split=config.split)
    assert isinstance(dataset, Dataset)
    result = dataset.cache_files
    logger.info(
        f"[task={config.task}, split={config.split}] Pre-fetched {len(dataset)} examples "
        f"from repository {config.repo_id} to local cache:\n{dataset.cache_files}"
    )
    return result


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
    return df


def _masked_probs(
    model: AutoModelForMaskedLM,
    tokenizer: PreTrainedTokenizer,
    loader: DataLoader,
    device: str,
    *,
    masks_per_sequence: int,
    desc: str,
) -> NDArray[np.floating]:
    idxs = [tokenizer.get_vocab()[n] for n in NUCLEOTIDES_LOWER]
    grouped_probs: list[NDArray[np.floating]] = []
    for batch in tqdm(loader, desc=desc):
        cur_ids = batch["masked_ids"].to(device).squeeze(1)
        with torch.inference_mode():
            logits = model(input_ids=cur_ids).logits
        masked_pos = (
            (cur_ids == tokenizer.mask_token_id)
            .unsqueeze(-1)
            .expand(-1, -1, logits.size(-1))
        )
        masked_logits = torch.masked_select(logits, masked_pos).view(
            -1, logits.size(-1)
        )
        probs = torch.softmax(masked_logits[:, idxs].float(), dim=-1).cpu().numpy()
        grouped_probs.append(probs)

    if not grouped_probs:
        return np.zeros((0, masks_per_sequence, len(idxs)), dtype=np.float32)

    stacked = np.vstack(grouped_probs)
    if stacked.shape[0] % masks_per_sequence != 0:
        raise ValueError(
            "Number of masked positions not divisible by masks_per_sequence"
        )
    return stacked.reshape(-1, masks_per_sequence, stacked.shape[-1])


def _unmasked_probs(
    sequences: pd.Series,
    tokenizer: PreTrainedTokenizer,
    model: AutoModelForMaskedLM,
    device: str,
    batch_size: int,
    *,
    desc: str,
) -> NDArray[np.floating]:
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
        return np.zeros((0, 0, N_NUCLEOTIDES), dtype=np.float32)
    return all_probs


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
    """Validate sequence lengths and mask token positions.

    Raises
    ------
    ValueError
        If sequences have different lengths or mask tokens are not centered
    """
    if len(df) == 0:
        return

    sequences = df[seq_column].astype(str)
    lengths = sequences.str.len()
    unique_lengths = lengths.unique()

    if len(unique_lengths) > 1:
        raise ValueError(
            f"All sequences must have the same length; found {len(unique_lengths)} "
            f"different lengths: {sorted(unique_lengths)}"
        )

    seq_len = unique_lengths[0]
    n_masks = len(mask_token_indexes)
    start = (seq_len - n_masks) // 2
    expected = list(range(start, start + n_masks))

    if list(mask_token_indexes) != expected:
        raise ValueError(
            f"Mask token indexes must be centered in sequences of length {seq_len}; "
            f"expected {expected}, got {list(mask_token_indexes)}"
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
    return token_probs.reshape(-1, motif_len).mean(axis=1)


def compute_evo_cons_probs(
    df: pd.DataFrame,
    config: EvoConsTaskConfig,
) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
    if len(df) == 0:
        return np.zeros(0, dtype=int), np.zeros((0, N_NUCLEOTIDES), dtype=np.float32)
    _validate_sequences(df, config.seq_column, [config.mask_token_index])
    dev = _require_cuda(config.device)
    model, tokenizer = _load_model(config.model, dev)
    dataset = SingleMaskDataset(
        df[config.seq_column], tokenizer, config.mask_token_index
    )
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=1
    )
    probs = _masked_probs(
        model,
        tokenizer,
        loader,
        dev,
        masks_per_sequence=1,
        desc=f"[{config.task}/{config.split}] Masked logits @ {config.mask_token_index}",
    )
    probs = probs.reshape(-1, probs.shape[-1])
    return df["example_idx"].to_numpy(), probs


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
    positions = [int(x) for x in config.mask_token_indexes]
    _validate_sequences(df, config.seq_column, positions)
    dev = _require_cuda(config.device)
    model, tokenizer = _load_model(config.model, dev)
    dataset = MultiMaskDataset(df[config.seq_column], tokenizer, positions)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=1
    )
    probs = _masked_probs(
        model,
        tokenizer,
        loader,
        dev,
        masks_per_sequence=len(positions),
        desc=f"[{config.task}/{config.split}] Masked logits motif_len={len(positions)}",
    )
    return df["example_idx"].to_numpy(), probs


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

    dev = _require_cuda(config.device)
    model, tokenizer = _load_model(config.model, dev)
    ref_probs = _unmasked_probs(
        df["RefSeq"],
        tokenizer,
        model,
        dev,
        config.batch_size,
        desc=f"[{config.task}/{config.split}] Ref (unmasked)",
    )
    mut_probs = _unmasked_probs(
        df["MutSeq"],
        tokenizer,
        model,
        dev,
        config.batch_size,
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


def compute_core_noncore_scores(
    df: pd.DataFrame,
    config: CoreNonCoreTaskConfig,
) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
    if len(df) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    positions = [int(x) for x in config.mask_token_indexes]
    if len(positions) != config.motif_len:
        raise ValueError("mask_idx count must equal motif_len")
    _validate_sequences(df, config.seq_column, positions)
    dev = _require_cuda(config.device)
    model, tokenizer = _load_model(config.model, dev)
    dataset = MultiMaskDataset(df[config.seq_column], tokenizer, positions)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=1
    )
    probs = _masked_probs(
        model,
        tokenizer,
        loader,
        dev,
        masks_per_sequence=len(positions),
        desc=f"[{config.task}/{config.split}] Masked logits motif_len={config.motif_len}",
    )
    flat_probs = probs.reshape(-1, probs.shape[-1])
    true_tokens = compute_true_tokens_from_seq(df[config.seq_column], positions)
    scores = average_true_probabilities(flat_probs, true_tokens, config.motif_len)
    return df["example_idx"].to_numpy(), scores
