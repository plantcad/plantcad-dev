from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from maize_missense.plots import ALL_TRAITS, plot_results, plot_summary, summarize_plot_data


def metric_fixture() -> pl.DataFrame:
    rows: list[dict[str, str | float]] = []
    for model, gain in [
        ("baseline", 0.0),
        ("PlantCAD/soft_top_0.01", 0.1),
        ("PlantCAD/soft_top_0.5", 0.2),
        ("phyloP/soft_top_0.01", -0.05),
        ("phyloP/soft_top_0.5", 0.05),
        ("phastCons/soft_top_0.01", 0.01),
        ("phastCons/soft_top_0.5", 0.02),
    ]:
        for validation, fold, r in [
            ("CP", "NAM_H", 0.2),
            ("CP", "Ames_H", 0.5),
            ("LOFO", "Z001", 0.1),
            ("LOFO", "Z002", 0.5),
        ]:
            rows.append(
                {
                    "model": model,
                    "validation": validation,
                    "fold": fold,
                    "trait": "PH",
                    "r": r + gain,
                    "MSE": 2.0 - gain,
                }
            )
    return pl.DataFrame(rows)


def test_plot_summary_keeps_directions_and_pairs_folds() -> None:
    summary = summarize_plot_data(metric_fixture())
    result = summary.filter(
        (pl.col("method") == "PlantCAD")
        & (pl.col("fraction") == 0.01)
        & (pl.col("trait") == "PH")
        & (pl.col("panel") != "CP/mean")
    )
    delta = result.filter(pl.col("metric") == "delta_r")
    assert delta["mean"].to_list() == pytest.approx([0.1, 0.1, 0.1])
    lofo = result.filter((pl.col("panel") == "LOFO") & (pl.col("metric") == "r"))
    assert lofo["mean"][0] == pytest.approx(0.4)
    assert lofo["q25"][0] == pytest.approx(0.3)
    assert lofo["q75"][0] == pytest.approx(0.5)
    assert lofo["n_folds"][0] == 2
    assert result.filter((pl.col("panel") == "CP/NAM_H") & (pl.col("metric") == "r"))["mean"][
        0
    ] == pytest.approx(0.3)
    # Paired differences have no spread here even though raw r varies by family.
    paired = delta.filter(pl.col("panel") == "LOFO")
    assert paired["q25"][0] == pytest.approx(paired["q75"][0])


def test_plot_summary_rejects_missing_or_undefined_folds() -> None:
    with pytest.raises(ValueError, match="baseline folds"):
        summarize_plot_data(metric_fixture().slice(1))
    with pytest.raises(ValueError, match="finite"):
        summarize_plot_data(metric_fixture().with_columns(pl.lit(float("nan")).alias("r")))
    with pytest.raises(ValueError, match="baseline"):
        summarize_plot_data(metric_fixture().filter(pl.col("model") != "baseline"))


def test_plot_summary_rejects_duplicate_and_unknown_settings() -> None:
    metrics = metric_fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_plot_data(pl.concat([metrics, metrics.head(1)]))
    with pytest.raises(ValueError, match="label"):
        summarize_plot_data(
            metrics.with_columns(pl.col("model").str.replace("PlantCAD", "unknown"))
        )
    with pytest.raises(ValueError, match="fraction grid"):
        summarize_plot_data(metrics.filter(pl.col("model") != "phyloP/soft_top_0.5"))


def test_plot_axes_and_output_files(tmp_path: Path) -> None:
    metrics = metric_fixture()
    figure = plot_summary(summarize_plot_data(metrics))
    try:
        assert len(figure.axes) == 4  # Two evaluations, all-trait mean + PH.
        assert figure.axes[0].get_xscale() == "linear"
        np.testing.assert_allclose(
            np.asarray(figure.axes[0].lines[0].get_ydata(), dtype=np.float64), [0.35, 0.35]
        )
        np.testing.assert_allclose(
            np.asarray(figure.axes[2].lines[0].get_ydata(), dtype=np.float64), [0.3, 0.3]
        )
        assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == ["1%", "50%"]
        points = figure.axes[0].lines[1:]
        assert [line.get_marker() for line in points] == ["o", "s", "^"]
        assert all(line.get_linestyle() == "None" for line in points)
        np.testing.assert_allclose(np.asarray(points[0].get_ydata()), [0.45, 0.55])
        np.testing.assert_allclose(np.asarray(points[0].get_xdata()), [-0.8 / 3, 1 - 0.8 / 3])
        assert all(not axis.patches and not axis.containers for axis in figure.axes)
        assert all(axis.get_ylim()[0] > 0 for axis in figure.axes)
        assert figure.axes[0].get_title(loc="left") == "All traits (equal-weight mean)"
        assert all(not axis.collections for axis in figure.axes)  # No invented macro CI.
        assert all("MSE" not in label.get_text() for label in figure.texts)
        assert {axis.get_ylabel() for axis in figure.axes} == {"", "Mean Pearson r"}
    finally:
        plt.close(figure)
    path = tmp_path / "metrics.parquet"
    metrics.write_parquet(path)
    plot_results(path, tmp_path / "plots")
    assert b"<svg" in (tmp_path / "plots" / "prediction_summary.svg").read_bytes()
    assert pl.read_parquet(tmp_path / "plots" / "plot_data.parquet").height == 112
    assert {path.suffix for path in (tmp_path / "plots").iterdir()} == {".svg", ".parquet"}


def test_controls_remain_separate_from_score_results_and_preserve_seed() -> None:
    metrics = metric_fixture()
    all_missense = metrics.filter(pl.col("model") == "baseline").with_columns(
        pl.lit("all_missense").alias("model"), (pl.col("r") + 0.03).alias("r")
    )
    random = metrics.filter(pl.col("model").str.starts_with("PlantCAD/")).with_columns(
        pl.col("model")
        .str.replace("PlantCAD/soft_top_", "shuffle/PlantCAD/")
        .add("/seed_20260929")
        .alias("model"),
        (pl.col("r") - 0.02).alias("r"),
    )
    combined = pl.concat([metrics, all_missense, random])
    summary = summarize_plot_data(combined)
    assert summary.height == 160
    shuffled = summary.filter(
        (pl.col("kind") == "shuffle")
        & (pl.col("panel") == "LOFO")
        & (pl.col("trait") == "PH")
        & (pl.col("metric") == "delta_r")
    )
    assert shuffled["seed"].to_list() == [20260929, 20260929]
    assert shuffled["n_folds"].to_list() == [2, 2]
    assert shuffled["mean"].to_list() == pytest.approx([0.08, 0.18])
    assert summary.filter(pl.col("kind") == "all_missense")["fraction"].unique().to_list() == [0.0]
    figure = plot_summary(summary)
    try:
        assert len(figure.axes[0].lines) == 6  # Two references and four point series.
        assert figure.axes[0].lines[1].get_linestyle() == ":"
        points = figure.axes[0].lines[2:]
        assert [line.get_marker() for line in points] == ["o", "o", "s", "^"]
        assert all(line.get_linestyle() == "None" for line in points)
        assert points[1].get_markerfacecolor() == "white"
        assert points[0].get_markerfacecolor() != "white"
        assert points[0].get_markeredgecolor() == points[1].get_markeredgecolor()
        np.testing.assert_allclose(np.asarray(points[1].get_xdata()), [-0.1, 0.9])
        assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == ["1%", "50%"]
        labels = {text.get_text() for legend in figure.legends for text in legend.get_texts()}
        assert "Genome-wide baseline (G₀)" in labels
        assert "Genome-wide + all missense" in labels
    finally:
        plt.close(figure)
    with pytest.raises(ValueError, match="complete fraction grid"):
        summarize_plot_data(
            combined.filter(pl.col("model") != "shuffle/PlantCAD/0.5/seed_20260929")
        )
    second_seed = random.with_columns(
        pl.col("model").str.replace("20260929", "20260930"),
        (pl.col("r") + 0.04).alias("r"),
    )
    macro_shuffles = (
        summarize_plot_data(pl.concat([combined, second_seed]))
        .filter(
            (pl.col("kind") == "shuffle")
            & (pl.col("panel") == "CP/mean")
            & (pl.col("trait") == ALL_TRAITS)
            & (pl.col("metric") == "r")
        )
        .sort("seed", "fraction")
    )
    assert macro_shuffles["seed"].to_list() == [20260929, 20260929, 20260930, 20260930]
    assert macro_shuffles["mean"].to_list() == pytest.approx([0.43, 0.53, 0.47, 0.57])


def test_macro_averages_balance_directions_and_traits_not_samples_or_families() -> None:
    ph = metric_fixture()
    gy = ph.with_columns(pl.lit("GY_adjusted").alias("trait"), (pl.col("r") + 0.1).alias("r"))
    dts = ph.with_columns(pl.lit("DTS").alias("trait"), (pl.col("r") - 0.1).alias("r"))
    # Yield has an extra usable family. It must not dominate the trait average.
    extra = gy.filter((pl.col("validation") == "LOFO") & (pl.col("fold") == "Z001"))
    extra = extra.with_columns(pl.lit("Z003").alias("fold"), (pl.col("r") + 0.3).alias("r"))
    metrics = pl.concat([ph, gy, extra, dts]).with_columns(
        pl.when(pl.col("fold") == "Ames_H").then(1000).otherwise(10).alias("n_test")
    )
    summary = summarize_plot_data(metrics)
    baseline = summary.filter((pl.col("kind") == "baseline") & (pl.col("metric") == "r"))
    cp = baseline.filter((pl.col("panel") == "CP/mean") & (pl.col("trait") == "PH"))
    assert cp["mean"][0] == pytest.approx((0.2 + 0.5) / 2)
    assert cp["n_folds"][0] == 2
    assert cp["n_directions"][0] == 2
    assert cp["aggregation"][0] == "mean_directions"
    macro = baseline.filter(pl.col("trait") == ALL_TRAITS)
    assert macro["n_traits"].to_list() == [3, 3]
    cross = macro.filter(pl.col("panel") == "CP/mean")
    assert cross["mean"][0] == pytest.approx((0.35 + 0.45 + 0.25) / 3)
    assert cross["n_folds"][0] == 6
    lofo = macro.filter(pl.col("panel") == "LOFO")
    assert lofo["mean"][0] == pytest.approx((0.3 + (0.2 + 0.6 + 0.5) / 3 + 0.2) / 3)
    assert lofo["mean"][0] != pytest.approx((0.1 + 0.5 + 0.2 + 0.6 + 0.5 + 0.0 + 0.4) / 7)
    assert lofo["n_folds"][0] == 7
    assert lofo["n_directions"][0] == 0
    assert lofo["aggregation"][0] == "mean_traits_of_family_means"
    assert macro["q25"].null_count() == macro.height
    assert macro["q75"].null_count() == macro.height
    assert summary.filter(
        (pl.col("trait") == ALL_TRAITS) & (pl.col("metric") == "delta_MSE")
    ).is_empty()
    gains = summary.filter(
        (pl.col("method") == "PlantCAD")
        & (pl.col("fraction") == 0.01)
        & (pl.col("trait") == ALL_TRAITS)
        & (pl.col("metric") == "delta_r")
    )
    assert gains["mean"].to_list() == pytest.approx([0.1, 0.1])
    assert_frame_equal(
        summary,
        summarize_plot_data(metrics.reverse()),
        check_exact=False,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    figure = plot_summary(summary)
    try:
        assert len(figure.axes) == 8
    finally:
        plt.close(figure)


def test_macro_averages_require_complete_evaluations_for_each_trait() -> None:
    metrics = metric_fixture()
    with pytest.raises(ValueError, match="both cross-panel directions"):
        summarize_plot_data(metrics.filter(pl.col("fold") != "NAM_H"))
    with pytest.raises(ValueError, match="LOFO"):
        summarize_plot_data(metrics.filter(pl.col("validation") != "LOFO"))
    with pytest.raises(ValueError, match="reserved"):
        summarize_plot_data(metrics.with_columns(pl.lit(ALL_TRAITS).alias("trait")))


@pytest.mark.parametrize("scale", [1.0, 0.001])
def test_y_limits_use_default_autoscaling_without_custom_padding(scale: float) -> None:
    metrics = metric_fixture().with_columns((0.3 + pl.col("r") * scale).alias("r"))
    figure = plot_summary(summarize_plot_data(metrics))
    try:
        for axis in figure.axes:
            assert axis.get_autoscaley_on()
            lower, upper = axis.dataLim.intervaly
            margin = float(plt.rcParams["axes.ymargin"]) * (upper - lower)
            np.testing.assert_allclose(axis.get_ylim(), [lower - margin, upper + margin])
    finally:
        plt.close(figure)


def test_points_use_independent_panel_scales_and_show_negative_values() -> None:
    metrics = metric_fixture()
    negative = metrics.with_columns(pl.lit("DTS").alias("trait"), (pl.col("r") - 0.6).alias("r"))
    metrics = pl.concat([metrics, negative]).with_columns(
        pl.when(pl.col("validation") == "LOFO")
        .then(pl.col("r") - 0.2)
        .otherwise(pl.col("r"))
        .alias("r")
    )
    summary = summarize_plot_data(metrics)
    figure = plot_summary(summary)
    try:
        assert len(figure.axes) == 6
        for column in range(3):
            assert figure.axes[column].get_ylim() != figure.axes[column + 3].get_ylim()
        assert figure.axes[2].get_ylim()[1] <= 0
        for axis in figure.axes:
            assert not any(
                axis.get_shared_y_axes().joined(axis, other)
                for other in figure.axes
                if other is not axis
            )
            lower, upper = axis.get_ylim()
            assert not axis.patches
            for line in axis.lines:
                values = np.asarray(line.get_ydata(), dtype=np.float64)
                assert np.all((lower <= values) & (values <= upper))
    finally:
        plt.close(figure)
