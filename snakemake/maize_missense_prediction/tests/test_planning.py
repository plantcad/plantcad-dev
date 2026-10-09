import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from maize_missense.planning import evaluation_inputs, plan_evaluations, relabel_evaluation
from maize_missense.weights import weight_fingerprint


def test_weight_identity_is_scale_and_order_invariant() -> None:
    original = weight_fingerprint(["a", "b", "c"], np.array([0.0, 0.1, 0.1]))
    assert original == weight_fingerprint(["c", "a", "b"], np.array([0.7, 0.0, 0.7]))
    assert original != weight_fingerprint(["a", "b", "c"], np.array([0.1, 0.1, 0.1]))
    assert original != weight_fingerprint(["z", "b", "c"], np.array([0.0, 0.1, 0.1]))


def test_weight_identity_does_not_merge_approximately_equal_values() -> None:
    original = weight_fingerprint(["a", "b"], np.array([1.0, 0.5]))
    assert original != weight_fingerprint(["a", "b"], np.array([1.0, 0.5 + 1e-12]))


def test_evaluation_plan_retains_all_labels_and_reuses_only_equal_weights(tmp_path: Path) -> None:
    reports = []
    for fraction, fingerprint in [(0.01, "same"), (0.05, "same"), (0.2, "different")]:
        path = tmp_path / f"{fraction}.json"
        path.write_text(
            json.dumps({"model": "phyloP", "fraction": fraction, "weight_fingerprint": fingerprint})
        )
        reports.append(path)
    output = tmp_path / "plan.json"
    plan_evaluations(reversed(reports), output)
    plan = json.loads(output.read_text())
    assert plan["n_settings"] == 3
    assert plan["n_unique_kernels"] == 2
    assert plan["canonical"]["phyloP/0.05"] == "phyloP/0.01"
    assert evaluation_inputs(output.read_text(), "phyloP", "0.05")["metrics"] == (
        "results/evaluation_unique/phyloP/0.01/metrics.parquet"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        plan_evaluations(reports + [reports[0]], output)


def test_relabeling_records_the_fit_source_without_changing_values(tmp_path: Path) -> None:
    source = tmp_path / "metrics.parquet"
    pl.DataFrame({"model": ["phyloP/soft_top_0.01"], "r": [0.7]}).write_parquet(source)
    report = tmp_path / "fit.json"
    report.write_text(json.dumps({"model": "phyloP/soft_top_0.01", "fits": {}}))
    output, predictions, provenance = (
        tmp_path / name for name in ["out.parquet", "pred.parquet", "out.json"]
    )
    relabel_evaluation(
        source, source, report, output, predictions, provenance, "phyloP/soft_top_0.05"
    )
    assert pl.read_parquet(output)["r"].to_list() == [0.7]
    assert pl.read_parquet(output)["model"].to_list() == ["phyloP/soft_top_0.05"]
    assert json.loads(provenance.read_text())["reused_fit"] is True
    assert json.loads(provenance.read_text())["fitted_model"] == "phyloP/soft_top_0.01"
