import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from maize_missense.weights import make_weights, soft_top_fraction


def test_fractional_boundary_and_permutation_invariance() -> None:
    scores = np.array([9.0, 8.0, 5.0, 5.0, 5.0, 1.0])
    weights, stats = soft_top_fraction(scores, 0.5)
    np.testing.assert_allclose(weights, [1, 1, 1 / 3, 1 / 3, 1 / 3, 0])
    assert stats["target_k"] == 3
    assert stats["n_tied_at_cutoff"] == 3
    order = np.array([5, 3, 1, 2, 0, 4])
    permuted, _ = soft_top_fraction(scores[order], 0.5)
    np.testing.assert_allclose(permuted, weights[order])


def test_all_equal_scores_preserve_equivalence() -> None:
    weights, stats = soft_top_fraction(np.zeros(100), 0.01)
    np.testing.assert_allclose(weights, 0.01)
    assert stats["effective_n"] == pytest.approx(100)
    assert stats["weight_sum"] == pytest.approx(1)


@pytest.mark.parametrize(
    "values,fraction",
    [([], 0.1), ([1, np.nan], 0.5), ([1, np.inf], 0.5), ([1], 0.01), ([1, 2], 0), ([1, 2], 1.1)],
)
def test_invalid_inputs_fail(values: list[float], fraction: float) -> None:
    with pytest.raises(ValueError):
        soft_top_fraction(values, fraction)


def test_plantcad_abs_keeps_snp_ids(tmp_path: Path) -> None:
    source, output, report = (tmp_path / name for name in ["in.parquet", "out.parquet", "qc.json"])
    pl.DataFrame({"snp_id": ["z", "a", "b", "c"], "PlantCAD": [-5.0, 1.0, 5.0, 0.0]}).write_parquet(
        source
    )
    make_weights(source, output, report, "PlantCAD", 0.25, "abs")
    result = pl.read_parquet(output)
    assert result["snp_id"].to_list() == ["z", "a", "b", "c"]
    assert result["weight"].to_list() == [0.5, 0.0, 0.5, 0.0]
    assert json.loads(report.read_text())["transform"] == "abs"
