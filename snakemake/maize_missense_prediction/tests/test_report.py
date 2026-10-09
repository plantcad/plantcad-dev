import json
from pathlib import Path

import polars as pl
import pytest

from maize_missense.report import summarize


def test_paired_summary_preserves_all_settings(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.parquet"
    coverage = tmp_path / "coverage.json"
    parquet, csv, report = (tmp_path / name for name in ["out.parquet", "out.csv", "out.json"])
    coverage.write_text("{}")
    pl.DataFrame(
        {
            "model": ["baseline", "PlantCAD/soft_top_0.01", "phyloP/soft_top_0.01"],
            "validation": ["CP"] * 3,
            "trait": ["PH"] * 3,
            "fold": ["NAM_H"] * 3,
            "r": [0.4, 0.6, 0.3],
            "MSE": [2.0, 1.0, 3.0],
        }
    ).write_parquet(metrics)
    summarize([metrics], coverage, [], parquet, csv, report, {})
    result = pl.read_parquet(parquet)
    assert result["delta_r"].to_list() == pytest.approx([0.0, 0.2, -0.1])
    assert result["delta_MSE"].to_list() == [0.0, -1.0, 1.0]
    assert pl.read_csv(csv).height == 3
    assert json.loads(report.read_text())["coverage"] == {}


def test_summary_rejects_missing_baseline(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.parquet"
    pl.DataFrame(
        {
            "model": ["PlantCAD"],
            "validation": ["CP"],
            "trait": ["PH"],
            "fold": ["NAM_H"],
            "r": [0.5],
            "MSE": [1.0],
        }
    ).write_parquet(metrics)
    with pytest.raises(ValueError, match="baseline"):
        summarize(
            [metrics],
            tmp_path / "unused",
            [],
            tmp_path / "out",
            tmp_path / "csv",
            tmp_path / "report",
            {},
        )


def test_summary_retains_control_provenance(tmp_path: Path) -> None:
    path = tmp_path / "metrics.parquet"
    pl.DataFrame(
        {
            "model": ["baseline", "all_missense", "shuffle/PlantCAD/0.01/seed_7"],
            "validation": ["CP"] * 3,
            "trait": ["PH"] * 3,
            "fold": ["NAM_H"] * 3,
            "r": [0.4, 0.5, 0.3],
            "MSE": [2.0, 1.0, 3.0],
        }
    ).write_parquet(path)
    coverage, controls = tmp_path / "coverage.json", tmp_path / "controls.json"
    coverage.write_text("{}")
    controls.write_text('{"n_unique_kernels": 2}')
    report = tmp_path / "report.json"
    summarize(
        [path],
        coverage,
        [],
        tmp_path / "out.parquet",
        tmp_path / "out.csv",
        report,
        {},
        control_plan_path=controls,
    )
    assert json.loads(report.read_text())["control_plan"]["n_unique_kernels"] == 2
