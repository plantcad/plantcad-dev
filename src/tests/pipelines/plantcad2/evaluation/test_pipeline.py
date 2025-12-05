import pytest

from src.pipelines.plantcad2.evaluation.pipeline import align_mask_indexes


def test_align_mask_indexes_no_adjustment():
    defaults = {"mask_token_indexes": [4095], "motif_len": 1}
    result = align_mask_indexes(defaults, model_context_length=8192, seq_length=8192)
    assert result == defaults


def test_align_mask_indexes_downscale_8192_to_512():
    # Original indexes centered in 8192: [4094, 4095, 4096]
    # Offset = (8192 - 512) // 2 = 3840
    # New indexes = [4094 - 3840, 4095 - 3840, 4096 - 3840] = [254, 255, 256]
    defaults = {"mask_token_indexes": [4094, 4095, 4096], "motif_len": 3}
    result = align_mask_indexes(defaults, model_context_length=512, seq_length=8192)
    assert result["mask_token_indexes"] == [254, 255, 256]
    assert result["motif_len"] == 3


def test_align_mask_indexes_raises_when_mask_token_indexes_missing():
    defaults = {"motif_len": 1}
    with pytest.raises(ValueError, match="mask_token_indexes required"):
        align_mask_indexes(defaults, model_context_length=512, seq_length=8192)
