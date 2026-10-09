"""Paired cluster bootstrap of frozen predictions, not repeated model training.

NAM families are exchangeable blocks. Ames parents are blocks stratified by
their observed tester combination. Draws are shared across traits, settings,
and validation schemes. LOFO resamples whole family results, without a second
within-family bootstrap (which would change the uncertainty target).
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from maize_missense.data import PathLike, require_unique, write_json
from maize_missense.plots import ALL_TRAITS, parse_model, summarize_plot_data

type FloatArray = NDArray[np.float64]
KEYS = ["validation", "trait", "fold", "sample_id"]
CI_KEYS = ["panel", "trait", "kind", "method", "fraction", "seed", "metric"]
CI_METHOD = "paired_stratified_cluster_percentile"
CI_SCOPE = "conditional_on_frozen_fits_and_observed_cluster_distributions"


def validate_predictions(predictions: pl.DataFrame, metrics: pl.DataFrame) -> pl.DataFrame:
    """Check exact pairing and reproduce saved fold metrics before resampling."""
    summarize_plot_data(metrics)  # Reuse completeness and balanced-design checks.
    require_unique(predictions, ["model", *KEYS], "prediction model/sample keys")
    if (
        predictions.is_empty()
        or predictions.select(pl.any_horizontal(pl.all().is_null()).any()).item()
    ):
        raise ValueError("Predictions must be nonempty and have no null values")
    if predictions.select(
        pl.any_horizontal(~pl.col("observed", "predicted").is_finite()).any()
    ).item():
        raise ValueError("Predictions must be finite")
    if set(predictions["model"].unique()) != set(metrics["model"].unique()):
        raise ValueError("Prediction and metric settings must match exactly")
    baseline = predictions.filter(pl.col("model") == "baseline").select(*KEYS, "observed")
    expected = baseline.sort(KEYS)
    for frame in predictions.partition_by("model"):
        if not frame.select(*KEYS, "observed").sort(KEYS).equals(expected):
            raise ValueError("Every setting must have identical observed values and sample keys")
    computed = predictions.group_by("model", "validation", "trait", "fold").agg(
        pl.corr("observed", "predicted").alias("computed_r"),
        ((pl.col("observed") - pl.col("predicted")) ** 2).mean().alias("computed_MSE"),
        pl.len().alias("computed_n"),
    )
    check = metrics.join(computed, on=["model", "validation", "trait", "fold"], validate="1:1")
    if check.height != metrics.height or computed.height != metrics.height:
        raise ValueError("Prediction and metric folds must match exactly")
    for metric in ["r", "MSE"]:
        if not np.allclose(
            check[metric].to_numpy(), check[f"computed_{metric}"].to_numpy(), atol=1e-10, rtol=1e-9
        ):
            raise ValueError(f"Predictions do not reproduce saved {metric}")
    if "n_test" in check.columns and check["n_test"].to_list() != check["computed_n"].to_list():
        raise ValueError("Predictions do not reproduce saved test counts")
    # Observations must agree between CP and LOFO and across duplicated settings.
    if (
        baseline.group_by("sample_id", "trait")
        .agg(pl.col("observed").n_unique())
        .filter(pl.col("observed") != 1)
        .height
    ):
        raise ValueError("Observed phenotypes differ between validation schemes")
    return (
        predictions.select(*KEYS, "model", "predicted")
        .pivot(on="model", index=KEYS, values="predicted")
        .join(baseline, on=KEYS, validate="1:1")
        .sort(KEYS)
    )


def sample_units(predictions: pl.DataFrame) -> pl.DataFrame:
    """Recover verified panel/family/parent/tester conventions from sample IDs.

    Ames parent IDs retain their accession suffix before '/tester'. NAM family
    IDs use the same prefix-before-E convention as the evaluation folds.
    """
    samples = (
        predictions.filter((pl.col("model") == "baseline") & (pl.col("validation") == "CP"))
        .select("sample_id", pl.col("fold").alias("population"))
        .unique()
        .sort("sample_id")
    )
    require_unique(samples, ["sample_id"], "sample populations")
    rows: list[dict[str, str]] = []
    for sample_id, population in samples.iter_rows():
        if "/" not in sample_id:
            raise ValueError("Sample IDs must contain parent/tester")
        parent, tester = sample_id.rsplit("/", 1)
        if not parent or not tester or population not in {"NAM_H", "Ames_H"}:
            raise ValueError("Invalid parent/tester or population")
        family = parent.split("E", 1)[0] if population == "NAM_H" else ""
        if population == "NAM_H" and ("E" not in parent or not family):
            raise ValueError("NAM sample IDs must contain a family prefix before E")
        rows.append(
            {
                "sample_id": sample_id,
                "population": population,
                "parent": parent,
                "tester": tester,
                "family": family,
                "unit": f"{population}/{family or parent}",
            }
        )
    if not rows:
        raise ValueError("No baseline cross-panel samples")
    units = pl.DataFrame(rows)
    ames = (
        units.filter(pl.col("population") == "Ames_H")
        .group_by("unit")
        .agg(pl.col("tester").unique().sort().str.join("+").alias("tester_pattern"))
    )
    units = (
        units.join(ames, on="unit", how="left", validate="m:1")
        .with_columns(
            pl.when(pl.col("population") == "NAM_H")
            .then(pl.lit("NAM/families"))
            .otherwise(pl.concat_str(pl.lit("Ames/"), pl.col("tester_pattern")))
            .alias("stratum")
        )
        .drop("tester_pattern")
    )
    lofo = (
        predictions.filter((pl.col("model") == "baseline") & (pl.col("validation") == "LOFO"))
        .select("sample_id", "fold")
        .unique()
    )
    checked = lofo.join(units, on="sample_id", how="left", validate="m:1")
    if checked.filter(
        pl.col("population").is_null()
        | (pl.col("population") != "NAM_H")
        | (pl.col("fold") != pl.col("family"))
    ).height:
        raise ValueError("LOFO membership must match the NAM family resampling blocks")
    if units.filter(pl.col("population") == "NAM_H")["unit"].n_unique() < 2:
        raise ValueError("At least two NAM families are required")
    return units.sort("sample_id")


def cluster_draws(units: pl.DataFrame, n_resamples: int, seed: int) -> tuple[list[str], FloatArray]:
    """Multinomial block counts preserving each tester-pattern stratum's size."""
    if n_resamples < 100 or seed < 0:
        raise ValueError("Use at least 100 bootstrap draws and a nonnegative seed")
    blocks = units.select("unit", "stratum").unique().sort("unit")
    require_unique(blocks, ["unit"], "resampling unit strata")
    names: list[str] = blocks["unit"].to_list()
    counts = np.zeros((n_resamples, len(names)), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    for stratum in sorted(blocks["stratum"].unique()):
        indices = np.flatnonzero(blocks["stratum"].to_numpy() == stratum)
        n = len(indices)
        counts[:, indices] = rng.multinomial(n, np.full(n, 1 / n), size=n_resamples)
    return names, counts


def cluster_moments(
    observed: FloatArray,
    predicted: FloatArray,
    groups: NDArray[np.int64],
    n_groups: int,
) -> FloatArray:
    """Six centered sufficient statistics per cluster and prediction setting."""
    y = observed - observed.mean()
    p = predicted - predicted.mean(axis=0)
    moments = np.zeros((n_groups, predicted.shape[1], 6), dtype=np.float64)
    values = np.stack(
        [
            np.ones_like(p),
            np.broadcast_to(y[:, None], p.shape),
            p,
            np.broadcast_to(y[:, None] ** 2, p.shape),
            p**2,
            y[:, None] * p,
        ],
        axis=-1,
    )
    np.add.at(moments, groups, values)
    return moments


def correlations_from_moments(moments: FloatArray) -> FloatArray:
    """Pearson r from weighted sufficient statistics; degeneracy remains NaN."""
    n, sy, sp, syy, spp, syp = np.moveaxis(moments, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        vy, vp = syy - sy * sy / n, spp - sp * sp / n
        r = (syp - sy * sp / n) / np.sqrt(vy * vp)
    valid = (n >= 2) & (vy > 0) & (vp > 0)
    return np.where(valid, np.clip(r, -1, 1), np.nan)


def bootstrap_replicates(
    wide: pl.DataFrame,
    metrics: pl.DataFrame,
    units: pl.DataFrame,
    models: list[str],
    names: list[str],
    counts: FloatArray,
) -> dict[tuple[str, str], FloatArray]:
    """Recompute whole-panel correlations and equal-family LOFO means per draw."""
    index = {name: i for i, name in enumerate(names)}
    annotated = wide.join(units.select("sample_id", "unit"), on="sample_id", validate="m:1")
    results: dict[tuple[str, str], FloatArray] = {}
    traits = sorted(metrics["trait"].unique())
    for trait in traits:
        for population in ["NAM_H", "Ames_H"]:
            data = annotated.filter(
                (pl.col("validation") == "CP")
                & (pl.col("trait") == trait)
                & (pl.col("fold") == population)
            ).sort("sample_id")
            groups = np.array([index[name] for name in data["unit"]], dtype=np.int64)
            moments = cluster_moments(
                data["observed"].to_numpy(), data.select(models).to_numpy(), groups, len(names)
            )
            weighted = (counts @ moments.reshape(len(names), -1)).reshape(
                len(counts), len(models), 6
            )
            results[(f"CP/{population}", trait)] = correlations_from_moments(weighted)
        results[("CP/mean", trait)] = (
            results[("CP/NAM_H", trait)] + results[("CP/Ames_H", trait)]
        ) / 2
        families = (
            metrics.filter((pl.col("validation") == "LOFO") & (pl.col("trait") == trait))
            .select("model", "fold", "r")
            .pivot(on="model", index="fold", values="r")
            .sort("fold")
        )
        family_indices = [index[f"NAM_H/{family}"] for family in families["fold"]]
        weights = counts[:, family_indices]
        with np.errstate(invalid="ignore", divide="ignore"):
            results[("LOFO", trait)] = (
                weights @ families.select(models).to_numpy() / weights.sum(axis=1)[:, None]
            )
    for panel in ["CP/mean", "LOFO"]:
        results[(panel, ALL_TRAITS)] = np.stack([results[(panel, t)] for t in traits]).mean(axis=0)
    return results


def bootstrap_intervals(
    predictions: pl.DataFrame,
    metrics: pl.DataFrame,
    *,
    n_resamples: int = 5000,
    seed: int = 20260930,
    confidence_level: float = 0.95,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """Pointwise percentile CIs for r and paired delta-r, with explicit scope."""
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must lie strictly between zero and one")
    wide = validate_predictions(predictions, metrics)
    units = sample_units(predictions)
    names, counts = cluster_draws(units, n_resamples, seed)
    models = sorted(metrics["model"].unique())
    baseline_index = models.index("baseline")
    replicates = bootstrap_replicates(wide, metrics, units, models, names, counts)
    # Never silently use different valid draws for different methods or traits.
    valid = np.logical_and.reduce([np.isfinite(a).all(axis=1) for a in replicates.values()])
    n_valid = int(valid.sum())
    if n_valid < 0.99 * n_resamples:
        raise ValueError("Over 1% of joint bootstrap draws are degenerate; review cluster design")
    point = bootstrap_replicates(wide, metrics, units, models, names, np.ones((1, len(names))))
    alpha = (1 - confidence_level) / 2
    rows: list[dict[str, str | float | int]] = []
    for (panel, trait), values in replicates.items():
        paired = values - values[:, baseline_index, None]
        point_values = point[(panel, trait)][0]
        for metric, draws, estimates in [
            ("r", values, point_values),
            ("delta_r", paired, point_values - point_values[baseline_index]),
        ]:
            lower, upper = np.quantile(draws[valid], [alpha, 1 - alpha], axis=0)
            for column, model in enumerate(models):
                kind, method, fraction, shuffle_seed = parse_model(model)
                rows.append(
                    {
                        "panel": panel,
                        "trait": trait,
                        "kind": kind,
                        "method": method,
                        "fraction": fraction,
                        "seed": shuffle_seed,
                        "metric": metric,
                        "point_estimate": float(estimates[column]),
                        "ci_lower": float(lower[column]),
                        "ci_upper": float(upper[column]),
                        "ci_level": confidence_level,
                        "n_bootstrap": n_resamples,
                        "n_bootstrap_valid": n_valid,
                        "bootstrap_seed": seed,
                        "ci_method": CI_METHOD,
                        "ci_scope": CI_SCOPE,
                    }
                )
    report: dict[str, object] = {
        "method": CI_METHOD,
        "scope": CI_SCOPE,
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
        "n_valid": n_valid,
        "n_joint_degenerate": n_resamples - n_valid,
        "seed": seed,
        "generator": "numpy.PCG64",
        "numpy_version": np.__version__,
        "polars_version": pl.__version__,
        "strata": units.group_by("population", "stratum")
        .agg(pl.len().alias("n_hybrids"), pl.col("unit").n_unique().alias("n_units"))
        .sort("population", "stratum")
        .to_dicts(),
        "pairing": "Identical cluster counts across methods, thresholds, traits and CP/LOFO",
        "averaging": "Equal directions; equal LOFO families; then equal traits; raw Pearson r",
        "limitations": [
            "Frozen fits: no training-sample, refitting or overlapping-training-fold uncertainty",
            "NAM: exchangeability across observed family blocks; within-family data held fixed",
            "Ames: parent blocks within fixed observed tester-pattern strata",
            "Does not account for residual relatedness between distinct families or parents",
            "Conditional on these traits, panels and testers; not inference to new traits or panels",
            "Pointwise intervals, not simultaneous bands or multiplicity-adjusted comparisons",
            "Shuffle intervals condition on the single fixed permutation, not shuffle variability",
        ],
    }
    return pl.DataFrame(rows).sort(CI_KEYS), units, report


def write_uncertainty(
    prediction_paths: Sequence[PathLike],
    metrics_path: PathLike,
    intervals_path: PathLike,
    units_path: PathLike,
    report_path: PathLike,
    *,
    n_resamples: int,
    seed: int,
    confidence_level: float,
) -> None:
    """Small Snakemake-facing wrapper with immutable input fingerprints."""
    predictions = pl.concat([pl.read_parquet(path) for path in prediction_paths])
    intervals, units, report = bootstrap_intervals(
        predictions,
        pl.read_parquet(metrics_path),
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
    )
    report["inputs_sha256"] = {
        str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in [metrics_path, *prediction_paths]
    }
    for path in [intervals_path, units_path, report_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    intervals.write_parquet(intervals_path)
    units.write_parquet(units_path)
    write_json(report_path, report)
