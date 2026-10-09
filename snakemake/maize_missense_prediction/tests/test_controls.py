import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from maize_missense.controls import (
    control_inputs,
    make_control_weights,
    permute_weights,
    plan_controls,
)
from maize_missense.weights import weight_fingerprint


def weight_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {"snp_id": [f"s{i:03}" for i in range(60)], "weight": [1.0] * 10 + [0.25] * 20 + [0.0] * 30}
    )


def test_shuffle_preserves_ties_and_is_reproducible_by_snp_id() -> None:
    source = weight_fixture()
    first = permute_weights(source, 20260929)
    assert first.equals(permute_weights(source.reverse(), 20260929))
    assert not first.equals(source)
    assert not first.equals(permute_weights(source, 20260930))
    np.testing.assert_array_equal(
        np.sort(first["weight"].to_numpy()), np.sort(source["weight"].to_numpy())
    )
    assert first["snp_id"].to_list() == sorted(source["snp_id"].to_list())


def test_shared_permutation_preserves_relative_weight_equivalence() -> None:
    source = weight_fixture()
    first = permute_weights(source, 7)
    scaled = permute_weights(source.with_columns(pl.col("weight") * 4), 7)
    assert weight_fingerprint(
        first["snp_id"].to_list(), first["weight"].to_numpy()
    ) == weight_fingerprint(scaled["snp_id"].to_list(), scaled["weight"].to_numpy())


def test_control_files_and_plan(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    weight_fixture().write_parquet(source)
    reports: list[Path] = []
    for setting, seed in [
        ("all_missense", None),
        ("shuffle/PlantCAD/0.01/seed_7", 7),
        ("shuffle/phyloP/0.01/seed_7", 7),
    ]:
        name = setting.replace("/", "_")
        output, report = tmp_path / f"{name}.parquet", tmp_path / f"{name}.json"
        make_control_weights(source, output, report, setting, seed)
        reports.append(report)
        saved = json.loads(report.read_text())
        if seed is None:
            assert pl.read_parquet(output)["weight"].to_list() == [1.0] * 60
            assert saved["effective_n"] == 60
        else:
            assert saved["weight_sum"] == 15
            assert saved["n_nonzero"] == 30
            assert saved["rng"] == "NumPy PCG64"
    output_plan = tmp_path / "plan.json"
    plan_controls(reports, output_plan)
    plan = json.loads(output_plan.read_text())
    assert plan["n_settings"] == 3 and plan["n_unique_kernels"] == 2
    assert control_inputs(output_plan.read_text(), "shuffle/phyloP/0.01/seed_7")["metrics"] == (
        "results/controls/evaluation_unique/shuffle/PlantCAD/0.01/seed_7/metrics.parquet"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        plan_controls(reports + reports[:1], output_plan)


def test_invalid_shuffle_inputs_fail() -> None:
    source = weight_fixture()
    with pytest.raises(ValueError, match="seed"):
        permute_weights(source, -1)
    with pytest.raises(ValueError, match="Duplicate"):
        permute_weights(pl.concat([source, source.head(1)]), 7)
    with pytest.raises(ValueError, match="missing"):
        permute_weights(source.with_columns(pl.lit(None, dtype=pl.Float64).alias("weight")), 7)
    with pytest.raises(ValueError, match="finite"):
        permute_weights(source.with_columns(pl.lit(float("nan")).alias("weight")), 7)
