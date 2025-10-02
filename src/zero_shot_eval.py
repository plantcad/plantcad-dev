"""Zero-shot evaluation utilities for PlantCAD2.

This module exposes helpers that can be reused both from the CLI entry point
(`zero-shot-eval.py`) and from distributed pipelines.  Compared to the
PlantCaduceus reference implementation the helpers here additionally support
dataset sharding via :func:`datasets.Dataset.shard`, enabling distributed
execution across Ray workers.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from numpy.typing import NDArray
from sklearn.metrics import auc, average_precision_score, roc_curve
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

NUCLEOTIDES = ("A", "C", "G", "T")
NUCLEOTIDES_LOWER = tuple(n.lower() for n in NUCLEOTIDES)
NUCLEOTIDE_TO_INDEX = {b: i for i, b in enumerate(NUCLEOTIDES)}


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


def _load_model(model_name: str, device: str):
    dtype = _optimal_dtype()
    logger.warning(
        "Loading model with dtype %s (no auto-fallback). Incompatible weights or kernels may error.",
        dtype,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype
    )
    model.to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, tokenizer


class SingleMaskDataset(TorchDataset):
    def __init__(self, sequences: pd.Series, tokenizer, token_idx: int):
        self.sequences = sequences.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.token_idx = token_idx

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int):
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
        if input_ids.size(1) <= self.token_idx:
            raise ValueError(
                f"token_idx {self.token_idx} out of range for sequence length {input_ids.size(1)}"
            )
        input_ids[0, self.token_idx] = self.tokenizer.mask_token_id
        return {"masked_ids": input_ids}


class MultiMaskDataset(TorchDataset):
    def __init__(self, sequences: pd.Series, tokenizer, mask_idx: Sequence[int]):
        self.sequences = sequences.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mask_idx = list(mask_idx)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int):
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


def load_zero_shot_dataframe(
    repo_id: str,
    task: str,
    split: str,
    *,
    worker_id: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Load a dataset split and attach a stable row index.

    Parameters
    ----------
    repo_id
        Hugging Face dataset repository identifier.
    task
        Dataset configuration name within the repository.
    split
        Split name to load.
    worker_id, num_workers
        Optional sharding parameters. When provided the dataset is sharded
        deterministically so that each worker receives a disjoint subset.
    """

    dataset: Dataset = load_dataset(repo_id, task, split=split)
    dataset = dataset.map(lambda _, idx: {"example_idx": idx}, with_indices=True)
    if worker_id is not None and num_workers is not None:
        dataset = dataset.shard(num_shards=num_workers, index=worker_id)
    df = dataset.to_pandas()
    if "example_idx" not in df.columns:
        raise KeyError("example_idx column missing after dataset preparation")
    df["example_idx"] = df["example_idx"].astype(int)
    return df


def _masked_probs(
    model,
    tokenizer,
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
        masked_pos = (cur_ids == tokenizer.mask_token_id).unsqueeze(-1).expand(
            -1, -1, logits.size(-1)
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
        raise ValueError("Number of masked positions not divisible by masks_per_sequence")
    return stacked.reshape(-1, masks_per_sequence, stacked.shape[-1])


def _unmasked_probs(
    sequences: pd.Series,
    tokenizer,
    model,
    device: str,
    batch_size: int,
    *,
    desc: str,
) -> NDArray[np.floating]:
    idxs = [tokenizer.get_vocab()[n] for n in NUCLEOTIDES_LOWER]
    seqs = sequences.astype(str).tolist()
    first_len: Optional[int] = None
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
        if first_len is None:
            first_len = probs.shape[1]
            all_probs = np.zeros((len(seqs), first_len, 4), dtype=np.float32)
        else:
            if probs.shape[1] != first_len:
                raise ValueError(
                    "All sequences must have the same length;"
                    f" got {probs.shape[1]} vs {first_len}"
                )
        all_probs[i : i + len(batch), :, :] = probs
    if all_probs is None:
        return np.zeros((0, 0, 4), dtype=np.float32)
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
    *,
    model: str,
    device: str,
    token_idx: int,
    batch_size: int,
    seq_column: str,
) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
    if len(df) == 0:
        return np.zeros(0, dtype=int), np.zeros((0, len(NUCLEOTIDES)), dtype=np.float32)
    dev = _require_cuda(device)
    model_, tokenizer = _load_model(model, dev)
    dataset = SingleMaskDataset(df[seq_column], tokenizer, token_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    probs = _masked_probs(
        model_, tokenizer, loader, dev, masks_per_sequence=1, desc=f"Masked logits @ {token_idx}"
    )
    probs = probs.reshape(-1, probs.shape[-1])
    return df["example_idx"].to_numpy(), probs


def compute_motif_probs(
    df: pd.DataFrame,
    *,
    model: str,
    device: str,
    batch_size: int,
    seq_column: str,
    mask_idx: Sequence[int],
) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
    if len(df) == 0:
        motif_len = len(mask_idx)
        return (
            np.zeros(0, dtype=int),
            np.zeros((0, motif_len, len(NUCLEOTIDES)), dtype=np.float32),
        )
    positions = [int(x) for x in mask_idx]
    dev = _require_cuda(device)
    model_, tokenizer = _load_model(model, dev)
    dataset = MultiMaskDataset(df[seq_column], tokenizer, positions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    probs = _masked_probs(
        model_,
        tokenizer,
        loader,
        dev,
        masks_per_sequence=len(positions),
        desc=f"Masked logits motif_len={len(positions)}",
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
    *,
    model: str,
    device: str,
    batch_size: int,
    flanking: int,
) -> StructuralVariantResult:
    if len(df) == 0:
        return StructuralVariantResult(
            example_idx=np.zeros(0, dtype=int),
            scores=np.zeros(0, dtype=float),
            ref_probs=np.zeros((0, 0, 4), dtype=np.float32),
            mut_probs=np.zeros((0, 0, 4), dtype=np.float32),
        )

    required = ["RefSeq", "MutSeq", "left", "right", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    dev = _require_cuda(device)
    model_, tokenizer = _load_model(model, dev)
    ref_probs = _unmasked_probs(
        df["RefSeq"], tokenizer, model_, dev, batch_size, desc="Ref (unmasked)"
    )
    mut_probs = _unmasked_probs(
        df["MutSeq"], tokenizer, model_, dev, batch_size, desc="Mut (unmasked)"
    )
    scores = structural_variant_boundary_scores(df, ref_probs, mut_probs, flanking)
    return StructuralVariantResult(
        example_idx=df["example_idx"].to_numpy(),
        scores=scores,
        ref_probs=ref_probs,
        mut_probs=mut_probs,
    )


def compute_core_noncore_scores(
    df: pd.DataFrame,
    *,
    model: str,
    device: str,
    batch_size: int,
    seq_column: str,
    mask_idx: Sequence[int],
    motif_len: int,
) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
    if len(df) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    positions = [int(x) for x in mask_idx]
    if len(positions) != motif_len:
        raise ValueError("mask_idx count must equal motif_len")
    dev = _require_cuda(device)
    model_, tokenizer = _load_model(model, dev)
    dataset = MultiMaskDataset(df[seq_column], tokenizer, positions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    probs = _masked_probs(
        model_,
        tokenizer,
        loader,
        dev,
        masks_per_sequence=len(positions),
        desc=f"Masked logits motif_len={motif_len}",
    )
    flat_probs = probs.reshape(-1, probs.shape[-1])
    true_tokens = compute_true_tokens_from_seq(df[seq_column], positions)
    scores = average_true_probabilities(flat_probs, true_tokens, motif_len)
    return df["example_idx"].to_numpy(), scores


@dataclass
class ZeroShotEvalCLI:
    """CLI entry points mirroring the original PlantCaduceus script."""

    def evo_cons(
        self,
        repo_id: str,
        task: str,
        split: str = "valid",
        model: str = "kuleshov-group/PlantCAD2-Small-l24-d0768",
        device: str = "cuda:0",
        token_idx: int = 255,
        batch_size: int = 128,
        seq_column: str = "sequence",
        save_logits: Optional[str] = None,
        logits_path: Optional[str] = None,
        metrics_json: Optional[str] = None,
        worker_id: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logger.info("Loading dataset")
        if logits_path is not None:
            probs = pd.read_csv(logits_path, sep="\t").values
            df = load_zero_shot_dataframe(repo_id, task, split, worker_id=worker_id, num_workers=num_workers)
        else:
            df = load_zero_shot_dataframe(repo_id, task, split, worker_id=worker_id, num_workers=num_workers)
            example_idx, probs = compute_evo_cons_probs(
                df,
                model=model,
                device=device,
                token_idx=token_idx,
                batch_size=batch_size,
                seq_column=seq_column,
            )
            if save_logits:
                pd.DataFrame(probs, columns=list(NUCLEOTIDES)).to_csv(
                    save_logits, sep="\t", index=False
                )
                logger.info("Saved logits TSV to %s", save_logits)
            if len(example_idx) != len(df):
                raise ValueError("Example index count mismatch")

        if probs.shape[0] != len(df):
            raise ValueError(
                f"Row mismatch: probs={probs.shape[0]} examples={len(df)}"
            )
        roc_auc = compute_ref_auroc(df, probs, token_idx, seq_column)
        y_true = df["label"].astype(int).to_numpy()
        pr_scores = reference_base_scores(df, probs, token_idx, seq_column)
        auprc = float(average_precision_score(y_true, pr_scores))
        print(f"AUROC\t{roc_auc:.6f}")
        print(f"AUPRC\t{auprc:.6f}")
        if metrics_json:
            with open(metrics_json, "w", encoding="utf-8") as fh:
                json.dump({"auroc": roc_auc, "auprc": auprc, "token_idx": token_idx}, fh, indent=2)

    def motif_acc(
        self,
        repo_id: str,
        task: str,
        split: str = "valid",
        model: str = "kuleshov-group/PlantCAD2-Small-l24-d0768",
        device: str = "cuda:0",
        mask_idx: Sequence[int] = (255, 256, 257),
        motif_len: int = 3,
        batch_size: int = 128,
        seq_column: str = "sequence",
        save_logits: Optional[str] = None,
        logits_path: Optional[str] = None,
        metrics_json: Optional[str] = None,
        worker_id: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logger.info("Loading dataset")
        df = load_zero_shot_dataframe(repo_id, task, split, worker_id=worker_id, num_workers=num_workers)
        positions = [int(x) for x in mask_idx]
        if logits_path is not None:
            probs = pd.read_csv(logits_path, sep="\t").values.reshape(
                len(df), len(positions), -1
            )
        else:
            example_idx, probs = compute_motif_probs(
                df,
                model=model,
                device=device,
                batch_size=batch_size,
                seq_column=seq_column,
                mask_idx=positions,
            )
            if save_logits:
                flat = probs.reshape(-1, probs.shape[-1])
                pd.DataFrame(flat, columns=list(NUCLEOTIDES)).to_csv(
                    save_logits, sep="\t", index=False
                )
                logger.info("Saved logits TSV to %s", save_logits)
            if len(example_idx) != len(df):
                raise ValueError("Example index count mismatch")

        if probs.shape[0] != len(df):
            raise ValueError(
                f"Row mismatch: probs={probs.shape[0]} expected={len(df)}"
            )
        flat_probs = probs.reshape(-1, probs.shape[-1])
        true_tokens = compute_true_tokens_from_seq(df[seq_column], positions)
        token_acc = token_accuracy_from_probs(flat_probs, true_tokens)
        motif_acc_value = motif_accuracy_from_probs(flat_probs, true_tokens, motif_len)
        print(f"token_accuracy\t{token_acc:.6f}")
        print(f"motif_accuracy\t{motif_acc_value:.6f}")
        if metrics_json:
            with open(metrics_json, "w", encoding="utf-8") as fh:
                json.dump(
                    {"token_accuracy": token_acc, "motif_accuracy": motif_acc_value},
                    fh,
                    indent=2,
                )

    def sv_effect(
        self,
        repo_id: str,
        task: str,
        split: str = "valid",
        model: str = "kuleshov-group/PlantCAD2-Small-l24-d0768",
        device: str = "cuda:0",
        batch_size: int = 64,
        flanking: int = 5,
        output: Optional[str] = None,
        save_ref_logits: Optional[str] = None,
        save_mut_logits: Optional[str] = None,
        worker_id: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        df = load_zero_shot_dataframe(repo_id, task, split, worker_id=worker_id, num_workers=num_workers)
        result = compute_sv_scores(
            df,
            model=model,
            device=device,
            batch_size=batch_size,
            flanking=flanking,
        )
        if len(result.example_idx) != len(df):
            raise ValueError("Example index count mismatch")
        if save_ref_logits:
            np.savez_compressed(save_ref_logits, logits=result.ref_probs)
        if save_mut_logits:
            np.savez_compressed(save_mut_logits, logits=result.mut_probs)
        y_true = df["label"].astype(int).to_numpy()
        auprc = float(average_precision_score(y_true, result.scores))
        print(f"AUPRC\t{auprc:.6f}")
        if output:
            out_df = df.copy()
            out_df["score"] = result.scores
            out_df.drop(columns=["Left5_Positions", "Right5_Positions"], errors="ignore")
            out_df.to_csv(output, sep="\t", index=False)

    def core_noncore(
        self,
        repo_id: str,
        task: str,
        split: str = "valid",
        model: str = "kuleshov-group/PlantCAD2-Small-l24-d0768",
        device: str = "cuda:0",
        mask_idx: Sequence[int] = (255, 256, 257),
        motif_len: int = 3,
        batch_size: int = 128,
        seq_column: str = "sequence",
        label_column: str = "label",
        save_logits: Optional[str] = None,
        logits_path: Optional[str] = None,
        metrics_json: Optional[str] = None,
        worker_id: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        df = load_zero_shot_dataframe(repo_id, task, split, worker_id=worker_id, num_workers=num_workers)
        positions = [int(x) for x in mask_idx]
        if logits_path is not None:
            probs = pd.read_csv(logits_path, sep="\t").values.reshape(
                len(df), len(positions), -1
            )
            flat_probs = probs.reshape(-1, probs.shape[-1])
            true_tokens = compute_true_tokens_from_seq(df[seq_column], positions)
            scores = average_true_probabilities(flat_probs, true_tokens, motif_len)
        else:
            example_idx, scores = compute_core_noncore_scores(
                df,
                model=model,
                device=device,
                batch_size=batch_size,
                seq_column=seq_column,
                mask_idx=positions,
                motif_len=motif_len,
            )
            if len(example_idx) != len(df):
                raise ValueError("Example index count mismatch")
            if save_logits:
                raise ValueError(
                    "save_logits is not supported when logits are computed inside the CLI;"
                    " use compute_motif_probs if logits are required"
                )

        y_true = df[label_column].astype(int).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = float(auc(fpr, tpr))
        auprc = float(average_precision_score(y_true, scores))
        print(f"AUROC\t{roc_auc:.6f}")
        print(f"AUPRC\t{auprc:.6f}")
        if metrics_json:
            with open(metrics_json, "w", encoding="utf-8") as fh:
                json.dump({"auroc": roc_auc, "auprc": auprc}, fh, indent=2)


__all__ = [
    "ZeroShotEvalCLI",
    "average_true_probabilities",
    "compute_core_noncore_scores",
    "compute_evo_cons_probs",
    "compute_motif_probs",
    "compute_ref_auroc",
    "compute_sv_scores",
    "compute_true_tokens_from_seq",
    "load_zero_shot_dataframe",
    "motif_accuracy_from_probs",
    "reference_base_scores",
    "structural_variant_boundary_scores",
    "token_accuracy_from_probs",
]
