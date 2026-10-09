"""Small, deterministic tests; no cloud inputs or fitted models needed."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.container import BarContainer, ErrorbarContainer
from polars.testing import assert_frame_equal

from maize_missense.plots import attach_intervals, plot_results, plot_summary, summarize_plot_data
from maize_missense.uncertainty import (
    bootstrap_intervals,
    cluster_draws,
    cluster_moments,
    correlations_from_moments,
    sample_units,
    validate_predictions,
    write_uncertainty,
)


def prediction_fixture() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(47)
    nam = [f"Z{family:03}E{line:04}/PHZ51" for family in range(1, 4) for line in range(6)]
    ames = [
        f"P{parent}:id{parent}/{tester}"
        for parent in range(12)
        for tester in (["B47"] if parent < 4 else ["PHZ51"] if parent < 8 else ["B47", "PHZ51"])
    ]
    settings = {
        "baseline": 0.0,
        "all_missense": 0.15,
        "PlantCAD/soft_top_0.01": 0.3,
        "PlantCAD/soft_top_0.5": 0.5,
        "phyloP/soft_top_0.01": 0.0,
        "phyloP/soft_top_0.5": 0.3,
        "phastCons/soft_top_0.01": 0.1,
        "phastCons/soft_top_0.5": 0.2,
        "shuffle/PlantCAD/0.01/seed_7": 0.25,
        "shuffle/PlantCAD/0.5/seed_7": 0.4,
    }
    rows: list[dict[str, str | float]] = []
    for trait in ["PH", "DTS"]:
        observed = rng.normal(size=len(nam + ames))
        for validation in ["CP", "LOFO"]:
            ids = nam + ames if validation == "CP" else nam
            noise = rng.normal(size=len(ids))
            for i, sample_id in enumerate(ids):
                fold = (
                    ("NAM_H" if i < len(nam) else "Ames_H")
                    if validation == "CP"
                    else sample_id.split("E")[0]
                )
                for model, gain in settings.items():
                    rows.append(
                        {
                            "model": model,
                            "validation": validation,
                            "trait": trait,
                            "fold": fold,
                            "sample_id": sample_id,
                            "observed": float(observed[i]),
                            "predicted": float((0.6 + gain) * observed[i] + noise[i]),
                        }
                    )
    predictions = pl.DataFrame(rows)
    metrics = predictions.group_by("model", "validation", "trait", "fold").agg(
        pl.corr("observed", "predicted").alias("r"),
        ((pl.col("observed") - pl.col("predicted")) ** 2).mean().alias("MSE"),
        pl.len().alias("n_test"),
    )
    return predictions, metrics


def test_units_keep_parent_testers_together_and_fix_stratum_sizes() -> None:
    predictions, _ = prediction_fixture()
    units = sample_units(predictions)
    assert units.height == 34
    assert units["unit"].n_unique() == 15
    assert units.filter(pl.col("parent") == "P8:id8")["unit"].n_unique() == 1
    names, counts = cluster_draws(units, 300, 19)
    assert counts.shape == (300, 15)
    for frame in units.select("unit", "stratum").unique().partition_by("stratum"):
        indices = [names.index(name) for name in frame["unit"]]
        np.testing.assert_array_equal(counts[:, indices].sum(axis=1), np.full(300, len(indices)))
    # A shared parent always weights both tester hybrids identically.
    np.testing.assert_array_equal(cluster_draws(units.reverse(), 300, 19)[1], counts)
    assert not np.array_equal(cluster_draws(units, 300, 20)[1], counts)


def test_sufficient_statistics_match_explicit_cluster_resampling() -> None:
    y = np.array([2.0, 5.0, 1.0, -2.0, 4.0])
    p = np.array([[1.0, 5.0], [6.0, 2.0], [-1.0, 2.0], [0.0, -1.0], [2.0, 4.0]])
    groups = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    counts = np.array([2, 0, 1])
    moments = cluster_moments(y, p, groups, 3)
    actual = correlations_from_moments(np.einsum("g,gmk->mk", counts, moments))
    indices = np.repeat(np.arange(5), counts[groups])
    expected = np.array([np.corrcoef(y[indices], p[indices, m])[0, 1] for m in range(2)])
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert np.isnan(correlations_from_moments(np.zeros((1, 6)))).all()
    constant = cluster_moments(y, np.ones((5, 1)), groups, 3)
    assert np.isnan(correlations_from_moments(constant.sum(axis=0))).all()


def test_bootstrap_pairing_macro_means_and_row_order() -> None:
    predictions, metrics = prediction_fixture()
    intervals, _, report = bootstrap_intervals(predictions, metrics, n_resamples=300, seed=9)
    summary = attach_intervals(summarize_plot_data(metrics), intervals)
    assert report["n_valid"] == 300
    assert (intervals["ci_upper"] >= intervals["ci_lower"]).all()
    delta = intervals.filter(
        (pl.col("metric") == "delta_r")
        & (
            (pl.col("kind") == "baseline")
            | ((pl.col("method") == "phyloP") & (pl.col("fraction") == 0.01))
        )
    )
    assert delta["ci_lower"].to_list() == [0.0] * delta.height
    assert delta["ci_upper"].to_list() == [0.0] * delta.height
    assert intervals.filter(pl.col("kind") == "shuffle")["seed"].unique().to_list() == [7]
    assert summary.filter(pl.col("metric") == "delta_MSE")["ci_lower"].null_count() > 0
    repeated, _, _ = bootstrap_intervals(
        predictions.reverse(), metrics.reverse(), n_resamples=300, seed=9
    )
    assert_frame_equal(intervals, repeated, check_exact=False, abs_tol=1e-12)


def test_rejects_missing_pairs_observation_drift_and_metric_drift() -> None:
    predictions, metrics = prediction_fixture()
    with pytest.raises(ValueError, match="sample keys"):
        validate_predictions(predictions.slice(1), metrics)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_predictions(pl.concat([predictions, predictions.head(1)]), metrics)
    with pytest.raises(ValueError, match="observed values"):
        validate_predictions(
            predictions.with_columns(
                pl.when(pl.col("model") == "all_missense")
                .then(pl.col("observed") + 1)
                .otherwise(pl.col("observed"))
                .alias("observed")
            ),
            metrics,
        )
    with pytest.raises(ValueError, match="reproduce saved r"):
        validate_predictions(predictions, metrics.with_columns((pl.col("r") + 0.01).alias("r")))
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(
            predictions.with_columns(pl.lit(float("nan")).alias("predicted")), metrics
        )
    with pytest.raises(ValueError, match="test counts"):
        validate_predictions(
            predictions, metrics.with_columns((pl.col("n_test") + 1).alias("n_test"))
        )


def test_duplicate_traits_do_not_artificially_narrow_the_interval() -> None:
    predictions, metrics = prediction_fixture()
    predictions = predictions.filter(pl.col("trait") == "PH")
    metrics = metrics.filter(pl.col("trait") == "PH")
    predictions = pl.concat([predictions, predictions.with_columns(pl.lit("DTS").alias("trait"))])
    metrics = pl.concat([metrics, metrics.with_columns(pl.lit("DTS").alias("trait"))])
    intervals, _, _ = bootstrap_intervals(predictions, metrics, n_resamples=300, seed=9)
    for panel in ["CP/mean", "LOFO"]:
        selected = intervals.filter(pl.col("panel") == panel)
        keys = ["kind", "method", "fraction", "seed", "metric"]
        single = selected.filter(pl.col("trait") == "PH").sort(keys)
        averaged = selected.filter(pl.col("trait") == "ALL_TRAITS").sort(keys)
        np.testing.assert_allclose(single["ci_lower"], averaged["ci_lower"], atol=1e-14)
        np.testing.assert_allclose(single["ci_upper"], averaged["ci_upper"], atol=1e-14)


def test_plot_confidence_label_comes_from_data_not_hardcoded() -> None:
    predictions, metrics = prediction_fixture()
    intervals, _, _ = bootstrap_intervals(
        predictions, metrics, n_resamples=100, confidence_level=0.9
    )
    figure = plot_summary(attach_intervals(summarize_plot_data(metrics), intervals))
    try:
        assert any("pointwise 90%" in text.get_text() for text in figure.texts)
    finally:
        plt.close(figure)


def test_degenerate_resampling_is_not_silently_dropped() -> None:
    predictions, _ = prediction_fixture()
    predictions = predictions.filter(
        (pl.col("fold") != "Ames_H") | pl.col("sample_id").is_in(["P0:id0/B47", "P1:id1/B47"])
    )
    metrics = predictions.group_by("model", "validation", "trait", "fold").agg(
        pl.corr("observed", "predicted").alias("r"),
        ((pl.col("observed") - pl.col("predicted")) ** 2).mean().alias("MSE"),
        pl.len().alias("n_test"),
    )
    with pytest.raises(ValueError, match="degenerate"):
        bootstrap_intervals(predictions, metrics, n_resamples=100)


def test_rejects_bad_units_parameters_and_incomplete_intervals() -> None:
    predictions, metrics = prediction_fixture()
    with pytest.raises(ValueError, match="parent/tester"):
        sample_units(predictions.with_columns(pl.col("sample_id").str.replace_all("/", "_")))
    with pytest.raises(ValueError, match="LOFO membership"):
        sample_units(
            predictions.with_columns(
                pl.when(pl.col("validation") == "LOFO")
                .then(pl.lit("Z999"))
                .otherwise(pl.col("fold"))
                .alias("fold")
            )
        )
    with pytest.raises(ValueError, match="100"):
        cluster_draws(sample_units(predictions), 5, 1)
    with pytest.raises(ValueError, match="Confidence level"):
        bootstrap_intervals(predictions, metrics, confidence_level=1)
    intervals, _, _ = bootstrap_intervals(predictions, metrics, n_resamples=100)
    summary = summarize_plot_data(metrics)
    with pytest.raises(ValueError, match="exactly all"):
        attach_intervals(summary, intervals.slice(1))
    with pytest.raises(ValueError, match="point estimates"):
        attach_intervals(
            summary,
            intervals.with_columns((pl.col("point_estimate") + 0.1).alias("point_estimate")),
        )


def test_error_bars_not_bands_and_output_provenance(tmp_path: Path) -> None:
    predictions, metrics = prediction_fixture()
    pred_path, metric_path = tmp_path / "predictions.parquet", tmp_path / "metrics.parquet"
    ci_path, units_path, report_path = (
        tmp_path / "ci.parquet",
        tmp_path / "units.parquet",
        tmp_path / "report.json",
    )
    predictions.write_parquet(pred_path)
    metrics.write_parquet(metric_path)
    write_uncertainty(
        [pred_path],
        metric_path,
        ci_path,
        units_path,
        report_path,
        n_resamples=100,
        seed=1,
        confidence_level=0.95,
    )
    summary = attach_intervals(summarize_plot_data(metrics), pl.read_parquet(ci_path))
    figure = plot_summary(summary)
    try:
        for axis in figure.axes:
            assert axis.get_autoscaley_on()
            errors = [c for c in axis.containers if isinstance(c, ErrorbarContainer)]
            bars = [c for c in axis.containers if isinstance(c, BarContainer)]
            points = [line for line in axis.lines if line.get_marker() in {"o", "s", "^"}]
            assert len(errors) == len(points) == 4  # 3 scores and one shuffle.
            assert not bars
            assert all(line.get_linestyle() == "None" for line in points)
            assert all(container.has_yerr for container in errors)
            lower, upper = axis.get_ylim()
            for collection in axis.collections:
                interval_lower, interval_upper = collection.get_datalim(axis.transData).intervaly
                assert lower <= interval_lower <= interval_upper <= upper
            assert not any(
                isinstance(collection, PolyCollection) for collection in axis.collections
            )
        assert any("conditional" in text.get_text() for text in figure.texts)
    finally:
        plt.close(figure)
    plot_results(metric_path, tmp_path / "plots", ci_path)
    assert (
        pl.read_parquet(tmp_path / "plots/plot_data.parquet")["ci_method"].drop_nulls().n_unique()
        == 1
    )
    assert "inputs_sha256" in report_path.read_text()
    assert {path.suffix for path in (tmp_path / "plots").iterdir()} == {".svg", ".parquet"}
