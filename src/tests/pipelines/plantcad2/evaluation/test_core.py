"""Tests for masked probability computation."""

import contextlib
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock, patch

from src.pipelines.plantcad2.evaluation.config import (
    EvoConsTaskConfig,
    ModelType,
    MotifInferenceMode,
    MotifTaskConfig,
)
from src.pipelines.plantcad2.evaluation.core import (
    center_crop_sequences,
    compute_evo_cons_probs,
    compute_motif_probs,
    reverse_complement,
    N_NUCLEOTIDES,
    NUCLEOTIDES,
)

# Token IDs (offset from 0 to catch index bugs)
TOKEN_A, TOKEN_C, TOKEN_G, TOKEN_T = 10, 11, 12, 13
MASK_TOKEN = 99
UNK_TOKEN = 0


class MockTokenizer:
    """Char-level tokenizer with offset token IDs."""

    def __init__(self):
        self._vocab = {"a": TOKEN_A, "c": TOKEN_C, "g": TOKEN_G, "t": TOKEN_T}
        self.mask_token_id = MASK_TOKEN
        self.unk_token_id = UNK_TOKEN

    def get_vocab(self):
        return self._vocab

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            ids = [[self._vocab.get(c.lower(), UNK_TOKEN) for c in s] for s in text]
            return {"input_ids": torch.tensor(ids)}
        ids = [self._vocab.get(c.lower(), UNK_TOKEN) for c in text]
        return {"input_ids": torch.tensor([ids])}


class MockModel:
    """Mock model that predicts T for motif_len positions after each A.

    Logic:
    - For each 'A' at position i, predict 'T' for positions i+1 through i+motif_len.
    - Otherwise, predict 'C'.

    For MLM: logits[pos] predicts token at pos.
    For CLM: logits[pos] predicts token at pos+1 (next token prediction).
    """

    def __init__(self, motif_len: int, model_type: ModelType):
        self.motif_len = motif_len
        self.model_type = model_type

    def eval(self):
        pass

    def to(self, **kwargs):
        return self

    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, 100), -10.0)

        for i in range(batch_size):
            seq = input_ids[i]

            # Find target positions where we want T predictions
            t_positions = set()
            for pos in range(seq_len):
                if seq[pos] == TOKEN_A:
                    for offset in range(1, self.motif_len + 1):
                        if pos + offset < seq_len:
                            t_positions.add(pos + offset)

            # Set logits - for CLM, shift logit position back by 1
            logit_offset = -1 if self.model_type == ModelType.clm else 0
            for pos in range(seq_len):
                logit_pos = pos + logit_offset
                if logit_pos < 0:
                    continue
                if pos in t_positions:
                    logits[i, logit_pos, TOKEN_T] = 10.0
                else:
                    logits[i, logit_pos, TOKEN_C] = 10.0

        return MagicMock(logits=logits)


@contextlib.contextmanager
def mock_model(motif_len: int, model_type: ModelType = ModelType.mlm):
    model_class = {
        ModelType.mlm: "AutoModelForMaskedLM",
        ModelType.clm: "AutoModelForCausalLM",
    }[model_type]
    with (
        patch(
            f"src.pipelines.plantcad2.evaluation.core.{model_class}.from_pretrained",
            return_value=MockModel(motif_len, model_type),
        ),
        patch(
            "src.pipelines.plantcad2.evaluation.core.AutoTokenizer.from_pretrained",
            return_value=MockTokenizer(),
        ),
    ):
        yield


def _base_config(
    mask_indexes: list[int],
    model_type: ModelType = ModelType.mlm,
    inference_mode: MotifInferenceMode = MotifInferenceMode.fwd_only,
) -> dict:
    return dict(
        task="test",
        split="test",
        repo_id="test/repo",
        model_path="test/model",
        model_type=model_type,
        model_context_length=8192,
        model_name="test-model",
        model_motif_inference_mode=inference_mode,
        seq_length=8192,
        device="cpu",
        batch_size=4,
        num_workers=None,
        seq_column="sequence",
        label_column="label",
        mask_token_indexes=mask_indexes,
        motif_len=len(mask_indexes),
    )


def _softmax_probs(*high_indices: int) -> list[float]:
    """Return softmax probabilities with high logit at specified indices."""
    logits = np.full(N_NUCLEOTIDES, -10.0)
    for idx in high_indices:
        logits[idx] = 10.0
    exp = np.exp(logits - logits.max())
    return (exp / exp.sum()).tolist()


# Nucleotide indices in output probs
A, C, G, T = [NUCLEOTIDES.index(n) for n in ("A", "C", "G", "T")]

# Softmax probability vectors with high logit at the corresponding nucleotide
pA, pC, pG, pT = [_softmax_probs(i) for i in (A, C, G, T)]


@pytest.mark.parametrize(
    "model_type",
    [
        pytest.param(ModelType.mlm, id="mlm"),
        pytest.param(ModelType.clm, id="clm"),
    ],
)
def test_compute_evo_cons_probs(model_type):
    # Seq 0: A at pos 1 -> predict T at pos 2 (mask position)
    # Seq 1: no A -> predict C
    # Seq 2: A at pos 0 only (left of trigger) -> A@0 triggers T at pos 1, C at pos 2
    df = pd.DataFrame(
        {
            "example_idx": [0, 1, 2],
            "sequence": ["CACCCC", "CCCCCC", "ACCCCC"],
        }
    )
    config = EvoConsTaskConfig(**_base_config(mask_indexes=[2], model_type=model_type))

    with mock_model(motif_len=1, model_type=model_type):
        ids, probs = compute_evo_cons_probs(df, config)

    expected = np.array([pT, pC, pC])
    assert ids.tolist() == [0, 1, 2]
    np.testing.assert_allclose(probs, expected, atol=1e-5)


@pytest.mark.parametrize(
    "sequences,mask_indexes,expected",
    [
        pytest.param(
            # Seq 0: A at pos 1 -> predict T at pos 2 (mask position)
            # Seq 1: no A -> predict C
            # Seq 2: A at pos 0 only -> A@0 triggers T at 1, C at mask pos 2
            ["CACCCC", "CCCCCC", "ACCCCC"],
            [2],
            [[pT], [pC], [pC]],
            id="1bp",
        ),
        pytest.param(
            # Seq 0: A at pos 1 -> predict T at pos 2, 3 (mask positions)
            # Seq 1: no A -> predict CC
            # Seq 2: A at pos 0 only -> A@0 triggers T at 1,2 -> TC at masks
            ["CACCCC", "CCCCCC", "ACCCCC"],
            [2, 3],
            [[pT, pT], [pC, pC], [pT, pC]],
            id="2bp",
        ),
        pytest.param(
            # Seq 0: A at pos 1 -> predict T at pos 2, 3, 4 (mask positions)
            # Seq 1: no A -> predict CCC
            # Seq 2: A at pos 0 only -> A@0 triggers T at 1,2,3 -> TTC at masks
            ["CACCCCCC", "CCCCCCCC", "ACCCCCCC"],
            [2, 3, 4],
            [[pT, pT, pT], [pC, pC, pC], [pT, pT, pC]],
            id="3bp",
        ),
    ],
)
def test_compute_motif_probs(sequences, mask_indexes, expected):
    df = pd.DataFrame({"example_idx": [0, 1, 2], "sequence": sequences})
    config = MotifTaskConfig(**_base_config(mask_indexes=mask_indexes))
    motif_len = len(mask_indexes)

    with mock_model(motif_len=motif_len):
        ids, probs = compute_motif_probs(df, config)

    assert ids.tolist() == [0, 1, 2]
    assert probs.shape == (3, motif_len, N_NUCLEOTIDES)
    np.testing.assert_allclose(probs, np.array(expected), atol=1e-5)


@pytest.mark.parametrize(
    "inference_mode,sequences,expected",
    [
        pytest.param(
            # fwd_only: A at pos 2 triggers T at mask pos 3
            MotifInferenceMode.fwd_only,
            ["CCACCCC", "CCCCCCC", "CACCCCC"],
            [pT, pC, pC],
            id="fwd_only",
        ),
        pytest.param(
            # rc_only: RC'd inputs get RC'd back by core.py, mock predicts [pT, pC, pC],
            # then _flip_rc_probs swaps complements: T->A, C->G
            MotifInferenceMode.rc_only,
            ["GGGGTGG", "GGGGGGG", "GGGGGTG"],  # RC of fwd sequences
            [pA, pG, pG],
            id="rc_only",
        ),
    ],
)
def test_clm_motif_inference_mode(inference_mode, sequences, expected):
    # 7-char sequences with centered mask at pos 3 (odd length so RC preserves center)
    df = pd.DataFrame({"example_idx": [0, 1, 2], "sequence": sequences})
    config = MotifTaskConfig(
        **_base_config(
            mask_indexes=[3], model_type=ModelType.clm, inference_mode=inference_mode
        )
    )

    with mock_model(motif_len=1, model_type=ModelType.clm):
        ids, probs = compute_motif_probs(df, config)

    assert ids.tolist() == [0, 1, 2]
    np.testing.assert_allclose(probs.squeeze(), np.array(expected), atol=1e-5)


@pytest.mark.parametrize(
    "seq,seq_length,context_length,expected",
    [
        pytest.param("AABBCC", 6, 2, "BB", id="6_to_2"),
        pytest.param("AABBBBCC", 8, 4, "BBBB", id="8_to_4"),
        pytest.param("AABBBBCC", 8, 6, "ABBBBC", id="8_to_6"),
        pytest.param("AABBBBCC", 8, 8, "AABBBBCC", id="no_crop"),
        pytest.param("AABBBBCC", 8, 10, "AABBBBCC", id="context_exceeds_seq"),
        pytest.param(
            "A" * 3840 + "B" * 512 + "C" * 3840,
            8192,
            512,
            "B" * 512,
            id="8192_to_512",
        ),
    ],
)
def test_center_crop_sequences(seq, seq_length, context_length, expected):
    df = pd.DataFrame({"seq": [seq]})
    result = center_crop_sequences(df, "seq", seq_length, context_length)
    assert result["seq"].iloc[0] == expected


@pytest.mark.parametrize(
    "seq_length,context_length",
    [
        pytest.param(7, 4, id="odd_seq_length"),
        pytest.param(8, 5, id="odd_context_length"),
        pytest.param(7, 5, id="both_odd"),
    ],
)
def test_center_crop_sequences_rejects_odd_lengths(seq_length, context_length):
    df = pd.DataFrame({"seq": ["A" * seq_length]})
    with pytest.raises(ValueError, match="Both lengths must be even"):
        center_crop_sequences(df, "seq", seq_length, context_length)


def test_reverse_complement():
    rc = reverse_complement
    assert rc("ACGT") == "ACGT"
    assert rc("AAAA") == "TTTT"
    assert rc("CCGG") == "CCGG"
    assert rc("GcTA") == "TAgC"
    assert rc("acgt") == "acgt"
    assert rc("N") == "N"
    assert rc("ANCGT") == "ACGNT"
    assert rc(rc("ANCGTancgt")) == "ANCGTancgt"
