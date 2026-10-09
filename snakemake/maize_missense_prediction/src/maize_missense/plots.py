"""Descriptive, paired fold plots; no test-set selection or independence claims."""

import re
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from maize_missense.data import PathLike, require_unique

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"PlantCAD": "#7047B9", "phyloP": "#008A87", "phastCons": "#D57424"}
MARKERS = {"PlantCAD": "o", "phyloP": "s", "phastCons": "^"}
LABELS = {"PlantCAD": "PlantCAD |LLR|", "phyloP": "phyloP", "phastCons": "phastCons"}
PANELS = {
    "CP/mean": "Cross-panel · mean of both directions",
    "LOFO": "NAM · leave one family out",
}
ALL_TRAITS = "ALL_TRAITS"
TRAIT_LABELS = {
    ALL_TRAITS: "All traits (equal-weight mean)",
    "GY_adjusted": "Yield",
    "PH": "Plant height",
    "DTS": "Days to silking",
}
TITLES = {
    "r": ("Held-out prediction accuracy", "Pearson r"),
    "delta_r": ("Change in prediction accuracy", "Δ Pearson r vs G₀"),
    "delta_MSE": ("Change in prediction error", "Δ MSE vs G₀"),
}


def parse_model(label: str) -> tuple[str, str, float, int]:
    """Return kind, score method, fraction and seed (-1 denotes no shuffle)."""
    if label in {"baseline", "all_missense"}:
        return label, label, 0.0, -1
    if match := re.fullmatch(r"([^/]+)/soft_top_([0-9.]+)", label):
        method, fraction = match.groups()
        kind, seed = "score", -1
    elif match := re.fullmatch(r"shuffle/([^/]+)/([0-9.]+)/seed_([0-9]+)", label):
        method, fraction, raw_seed = match.groups()
        kind, seed = "shuffle", int(raw_seed)
    else:
        raise ValueError(f"Unexpected model/fraction label: {label}")
    if method not in COLORS or not 0 < float(fraction) <= 1:
        raise ValueError(f"Unexpected model/fraction label: {label}")
    return kind, method, float(fraction), seed


def summarize_plot_data(metrics: pl.DataFrame) -> pl.DataFrame:
    """Retain detailed summaries and add equal-direction / equal-trait means.

    Every setting must have exactly the baseline's folds. Nonfinite performance
    is an error, not an invitation to silently average a different set of folds.
    Differences are computed within folds before aggregation. IQRs describe
    heterogeneity across families, not confidence intervals or independent runs.
    Macro means are arithmetic means of correlations, not pooled correlations
    or Fisher-z means. Traits are averaged after directions/families, so each
    trait contributes equally even if its number of usable families differs.
    No cross-trait MSE average is computed: trait units are not comparable.
    """
    keys = ["validation", "trait", "fold"]
    require_unique(metrics, ["model", *keys], "plot model/fold results")
    if (
        metrics.is_empty()
        or metrics.select(
            pl.any_horizontal(pl.col("r", "MSE").is_null() | ~pl.col("r", "MSE").is_finite()).any()
        ).item()
    ):
        raise ValueError("Plot metrics must be nonempty and finite for every fold")
    baseline = metrics.filter(pl.col("model") == "baseline").select(
        *keys, pl.col("r").alias("baseline_r"), pl.col("MSE").alias("baseline_MSE")
    )
    if baseline.is_empty():
        raise ValueError("Plots require the G0 baseline")
    expected = baseline.select(keys).sort(keys)
    for frame in metrics.partition_by("model"):
        if not frame.select(keys).sort(keys).equals(expected):
            raise ValueError("Every plotted setting must have exactly the baseline folds")
    traits = baseline.select("trait").unique()
    if ALL_TRAITS in traits["trait"].to_list():
        raise ValueError(f"{ALL_TRAITS} is reserved for the derived trait average")
    expected_cp = traits.join(pl.DataFrame({"fold": ["NAM_H", "Ames_H"]}), how="cross")
    observed_cp = baseline.filter(pl.col("validation") == "CP").select("trait", "fold")
    lofo_traits = baseline.filter(pl.col("validation") == "LOFO").select("trait").unique()
    if not observed_cp.sort("trait", "fold").equals(expected_cp.sort("trait", "fold")) or (
        not lofo_traits.sort("trait").equals(traits.sort("trait"))
    ):
        raise ValueError("Each trait requires both cross-panel directions and LOFO")
    if metrics.filter(
        ~pl.col("validation").is_in(["CP", "LOFO"])
        | ((pl.col("validation") == "CP") & ~pl.col("fold").is_in(["NAM_H", "Ames_H"]))
    ).height:
        raise ValueError("Unexpected validation panel")
    labels = metrics["model"].unique().to_list()
    metadata = pl.DataFrame(
        [(label, *parse_model(label)) for label in labels],
        schema=["model", "kind", "method", "fraction", "seed"],
        orient="row",
    )
    data = metrics.select("model", *keys, "r", "MSE").join(baseline, on=keys, validate="m:1")
    data = data.join(metadata, on="model", validate="m:1")
    data = data.with_columns(
        pl.when(pl.col("validation") == "CP")
        .then(pl.concat_str("validation", "fold", separator="/"))
        .otherwise(pl.lit("LOFO"))
        .alias("panel"),
        (pl.col("r") - pl.col("baseline_r")).alias("delta_r"),
        (pl.col("MSE") - pl.col("baseline_MSE")).alias("delta_MSE"),
    )
    settings = data.filter(pl.col("kind") == "score").select("method", "fraction").unique()
    expected_settings = pl.DataFrame({"method": list(COLORS)}).join(
        settings.select("fraction").unique(), how="cross"
    )
    if settings.is_empty() or not settings.sort("method", "fraction").equals(
        expected_settings.sort("method", "fraction")
    ):
        raise ValueError("All three methods must have the same fraction grid")
    for shuffled in data.filter(pl.col("kind") == "shuffle").partition_by("method", "seed"):
        method = shuffled["method"][0]
        expected_fractions = settings.filter(pl.col("method") == method)["fraction"].sort()
        if not shuffled["fraction"].unique().sort().equals(expected_fractions):
            raise ValueError("Each shuffled method/seed must have the complete fraction grid")
    long = data.unpivot(
        on=list(TITLES),
        index=["panel", "trait", "kind", "method", "fraction", "seed", "fold"],
        variable_name="metric",
        value_name="value",
    )
    detailed = (
        long.group_by("panel", "trait", "kind", "method", "fraction", "seed", "metric")
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").quantile(0.25, interpolation="linear").alias("q25"),
            pl.col("value").quantile(0.75, interpolation="linear").alias("q75"),
            pl.len().alias("n_folds"),
        )
        .with_columns(
            pl.lit(1).alias("n_traits"),
            pl.when(pl.col("panel") == "LOFO").then(0).otherwise(1).alias("n_directions"),
            pl.when(pl.col("panel") == "LOFO")
            .then(pl.lit("mean_families"))
            .otherwise(pl.lit("single_direction"))
            .alias("aggregation"),
        )
    )
    return _add_macro_averages(detailed)


def _add_macro_averages(detailed: pl.DataFrame) -> pl.DataFrame:
    """Add balanced means without inventing uncertainty for pooled estimands."""
    settings = ["kind", "method", "fraction", "seed", "metric"]
    cross_panel = (
        detailed.filter(pl.col("panel").str.starts_with("CP/"))
        .group_by("trait", *settings)
        .agg(pl.col("mean").mean(), pl.col("n_folds").sum())
        .with_columns(
            pl.lit("CP/mean").alias("panel"),
            pl.lit(None, dtype=pl.Float64).alias("q25"),
            pl.lit(None, dtype=pl.Float64).alias("q75"),
            pl.lit(1).alias("n_traits"),
            pl.lit(2).alias("n_directions"),
            pl.lit("mean_directions").alias("aggregation"),
        )
        .select(detailed.columns)
    )
    per_trait = pl.concat([cross_panel, detailed.filter(pl.col("panel") == "LOFO")])
    across_traits = (
        per_trait.filter(pl.col("metric").is_in(["r", "delta_r"]))
        .group_by("panel", *settings)
        .agg(
            pl.col("mean").mean(),
            pl.col("n_folds").sum(),
            pl.col("trait").n_unique().cast(pl.Int32).alias("n_traits"),
            pl.col("n_directions").max(),
        )
        .with_columns(
            pl.lit(ALL_TRAITS).alias("trait"),
            pl.lit(None, dtype=pl.Float64).alias("q25"),
            pl.lit(None, dtype=pl.Float64).alias("q75"),
            pl.when(pl.col("panel") == "LOFO")
            .then(pl.lit("mean_traits_of_family_means"))
            .otherwise(pl.lit("mean_traits_of_direction_means"))
            .alias("aggregation"),
        )
        .select(detailed.columns)
    )
    return pl.concat([detailed, cross_panel, across_traits]).sort("panel", "trait", *settings)


def _point_series(data: pl.DataFrame) -> list[tuple[str, str, int]]:
    """Stable group order; every shuffle stays beside its originating model."""
    series: list[tuple[str, str, int]] = []
    for method in COLORS:
        if data.filter((pl.col("method") == method) & (pl.col("kind") == "score")).height:
            series.append(("score", method, -1))
        for seed in (
            data.filter((pl.col("method") == method) & (pl.col("kind") == "shuffle"))["seed"]
            .unique()
            .sort()
        ):
            series.append(("shuffle", method, int(seed)))
    return series


def _draw_panel(axis: Axes, data: pl.DataFrame, fractions: list[float]) -> None:
    """Dodged points with capped intervals and horizontal baseline references."""
    baseline = data.filter(pl.col("method") == "baseline")
    if baseline.height != 1:
        raise ValueError("Each plot panel needs exactly one baseline")
    reference = float(baseline["mean"][0])
    axis.axhline(reference, color="#50545B", linestyle="--", linewidth=1.3, zorder=4)
    has_intervals = "ci_lower" in data.columns
    unweighted = data.filter(pl.col("kind") == "all_missense")
    if unweighted.height:
        axis.axhline(
            float(unweighted["mean"][0]), color="#1B2631", linestyle=":", linewidth=1.6, zorder=4
        )
    series = _point_series(data)
    width = 0.8 / len(series)
    for index, (kind, method, seed) in enumerate(series):
        rows = data.filter(
            (pl.col("method") == method) & (pl.col("kind") == kind) & (pl.col("seed") == seed)
        ).sort("fraction")
        if rows["fraction"].to_list() != fractions:
            raise ValueError("Every point series must have exactly the displayed fraction grid")
        x = np.arange(len(fractions), dtype=np.float64) + (index - (len(series) - 1) / 2) * width
        color = COLORS[method]
        axis.plot(
            x,
            rows["mean"].to_numpy(),
            linestyle="none",
            marker=MARKERS[method],
            markersize=5,
            markerfacecolor=color if kind == "score" else "white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            zorder=6,
        )
        if has_intervals:
            _draw_intervals(axis, rows, x, color)
    axis.set_xticks(np.arange(len(fractions)), [f"{fraction * 100:g}%" for fraction in fractions])
    axis.set_xlim(-0.55, len(fractions) - 0.45)
    axis.minorticks_off()
    axis.grid(axis="y", color="#E3E6EB", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for spine in ["bottom", "left"]:
        axis.spines[spine].set_color("#D1D5DB")
    axis.tick_params(labelsize=9)


def _draw_intervals(axis: Axes, rows: pl.DataFrame, x: NDArray[np.float64], color: str) -> None:
    """Capped vertical intervals only, including asymmetric percentile CIs.

    The original estimate need not lie inside a percentile interval. Draw the
    whisker around its midpoint independently of the estimate's marker rather
    than clipping or forcing a negative matplotlib yerr to zero.
    """
    lower, upper = rows["ci_lower"].to_numpy(), rows["ci_upper"].to_numpy()
    axis.errorbar(
        x,
        (lower + upper) / 2,
        yerr=(upper - lower) / 2,
        fmt="none",
        ecolor=color,
        elinewidth=0.85,
        capsize=1.8,
        capthick=0.85,
        alpha=0.85,
        zorder=5,
    )


def attach_intervals(summary: pl.DataFrame, intervals: pl.DataFrame) -> pl.DataFrame:
    """Attach complete r/delta-r CIs, verifying their point estimands match."""
    keys = ["panel", "trait", "kind", "method", "fraction", "seed", "metric"]
    require_unique(intervals, keys, "bootstrap interval keys")
    expected = summary.filter(pl.col("metric").is_in(["r", "delta_r"]))
    if not intervals.select(keys).sort(keys).equals(expected.select(keys).sort(keys)):
        raise ValueError("Bootstrap intervals must cover exactly all r and delta-r summaries")
    merged = expected.join(intervals, on=keys, validate="1:1")
    if not np.allclose(
        merged["mean"].to_numpy(), merged["point_estimate"].to_numpy(), atol=1e-10, rtol=1e-9
    ):
        raise ValueError("Bootstrap point estimates disagree with plotted averages")
    if merged.filter(
        pl.any_horizontal(
            pl.col("ci_lower", "ci_upper").is_null() | ~pl.col("ci_lower", "ci_upper").is_finite()
        )
        | (pl.col("ci_lower") > pl.col("ci_upper"))
    ).height:
        raise ValueError("Bootstrap intervals must be finite and ordered")
    return summary.join(intervals.drop("point_estimate"), on=keys, how="left", validate="1:1")


def plot_summary(summary: pl.DataFrame) -> Figure:
    """Plot accuracy only: CP means / LOFO by all-trait mean and each trait."""
    interval_level: float | None = None
    if "ci_lower" in summary.columns:
        levels = summary["ci_level"].drop_nulls().unique()
        if len(levels) != 1 or not 0 < float(levels[0]) < 1:
            raise ValueError("Figure requires one valid confidence level")
        interval_level = float(levels[0])
    present_traits = summary["trait"].unique().to_list()
    traits = [trait for trait in TRAIT_LABELS if trait in present_traits]
    traits.extend(sorted(set(present_traits) - set(traits)))
    panels = [panel for panel in PANELS if panel in summary["panel"].to_list()]
    if not traits or not panels:
        raise ValueError("No results to plot")
    fractions = sorted(summary.filter(pl.col("kind") == "score")["fraction"].unique().to_list())
    if not fractions:
        raise ValueError("No score settings to plot")
    figure = plt.figure(figsize=(4.15 * len(traits), 3.0 * len(panels) + 2.5))
    figure.set_facecolor("white")
    grid = figure.add_gridspec(
        len(panels),
        len(traits),
        left=0.07,
        right=0.985,
        top=0.79,
        bottom=0.14,
        hspace=0.62,
        wspace=0.29,
    )
    data = summary.filter((pl.col("metric") == "r") & pl.col("panel").is_in(panels))
    # Let Matplotlib autoscale each subplot from its points, CIs and references.
    for row, panel in enumerate(panels):
        for column, trait in enumerate(traits):
            axis = figure.add_subplot(grid[row, column])
            subset = data.filter((pl.col("panel") == panel) & (pl.col("trait") == trait))
            _draw_panel(axis, subset, fractions)
            axis.set_title(TRAIT_LABELS.get(trait, trait), fontsize=11, loc="left")
            if trait == ALL_TRAITS:
                axis.set_facecolor("#F5F7FA")
            if column == 0:
                axis.set_ylabel("Mean Pearson r", fontsize=10)
                figure.text(
                    0.07,
                    axis.get_position().y1 + 0.05,
                    PANELS[panel],
                    fontsize=12,
                    weight="bold",
                )
            if row == len(panels) - 1:
                axis.set_xlabel("Soft-top fraction", fontsize=10)
    handles: list[Line2D] = []
    series = _point_series(data)
    for kind, method, seed in series:
        multiple_seeds = sum(s[0] == "shuffle" and s[1] == method for s in series) > 1
        label = LABELS[method] if kind == "score" else f"{method} shuffled"
        if kind == "shuffle" and multiple_seeds:
            label += f" (seed {seed})"
        color = COLORS[method]
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=MARKERS[method],
                markersize=5,
                markerfacecolor=color if kind == "score" else "white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                label=label,
            )
        )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(handles),
        frameon=False,
        fontsize=10,
    )
    references = [
        Line2D([0], [0], color="#50545B", linestyle="--", label="Genome-wide baseline (G₀)")
    ]
    if summary.filter(pl.col("kind") == "all_missense").height:
        references.append(
            Line2D([0], [0], color="#1B2631", linestyle=":", label="Genome-wide + all missense")
        )
    figure.legend(
        handles=references,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    figure.suptitle(
        "Held-out prediction accuracy · higher is better", fontsize=17, x=0.07, ha="left", y=0.99
    )
    figure.text(
        0.07,
        0.052,
        "Arithmetic means: equal directions for cross-panel; equal families within each trait for LOFO; then equal traits. No pooled correlations.",
        fontsize=9,
        color="#50545B",
    )
    figure.text(
        0.07,
        0.027,
        f"Error bars: pointwise {100 * interval_level:g}% cluster-bootstrap intervals, conditional on fitted models. Reference-line intervals remain in Parquet."
        if interval_level is not None
        else "One PlantCAD shuffle is a sanity check, not a permutation test. Detailed directions, fold counts and paired differences remain in Parquet."
        if summary.filter(pl.col("kind") == "shuffle").height
        else "Detailed directions, fold counts and paired differences remain in Parquet. Conservation 1–10% kernels are tied.",
        fontsize=9,
        color="#50545B",
    )
    if "ci_lower" in summary.columns:
        figure.text(
            0.07,
            0.006,
            "NAM: family blocks. Ames: parent blocks within tester-pattern strata. Residual kinship and training uncertainty are not covered. Single shuffle held fixed.",
            fontsize=8,
            color="#50545B",
        )
    return figure


def plot_results(
    metrics_path: PathLike,
    output_dir: PathLike,
    intervals_path: PathLike | None = None,
) -> None:
    """Save one SVG summary and paired aggregate values as Parquet."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = summarize_plot_data(pl.read_parquet(metrics_path))
    if intervals_path is not None:
        summary = attach_intervals(summary, pl.read_parquet(intervals_path))
    summary.write_parquet(destination / "plot_data.parquet")
    figure = plot_summary(summary)
    try:
        with matplotlib.rc_context({"svg.fonttype": "none"}):
            figure.savefig(destination / "prediction_summary.svg", facecolor="white")
    finally:
        plt.close(figure)
