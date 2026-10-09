"""Paired fold comparisons, retaining every predefined score setting."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl

from maize_missense.data import PathLike, require_unique, write_json


def summarize(
    metric_paths: Iterable[PathLike],
    coverage_path: PathLike,
    weight_paths: Iterable[PathLike],
    parquet_path: PathLike,
    csv_path: PathLike,
    report_path: PathLike,
    config: Mapping[str, object],
    plan_path: PathLike | None = None,
    control_plan_path: PathLike | None = None,
) -> None:
    metrics = pl.concat([pl.read_parquet(path) for path in metric_paths])
    keys = ["validation", "trait", "fold"]
    require_unique(metrics, ["model", *keys], "model/fold results")
    baseline = metrics.filter(pl.col("model") == "baseline").select(
        *keys,
        pl.col("r").alias("baseline_r"),
        pl.col("MSE").alias("baseline_MSE"),
    )
    if baseline.height != metrics.select(keys).unique().height:
        raise ValueError("Every evaluated fold must have a baseline")
    metrics = metrics.join(baseline, on=keys, validate="m:1").with_columns(
        (pl.col("r") - pl.col("baseline_r")).alias("delta_r"),
        (pl.col("MSE") - pl.col("baseline_MSE")).alias("delta_MSE"),
    )
    metrics.write_parquet(parquet_path)
    metrics.write_csv(csv_path)
    write_json(
        report_path,
        {
            "config": config,
            "coverage": json.loads(Path(coverage_path).read_text()),
            "weights": [json.loads(Path(path).read_text()) for path in weight_paths],
            "evaluation_plan": json.loads(Path(plan_path).read_text()) if plan_path else None,
            "control_plan": json.loads(Path(control_plan_path).read_text())
            if control_plan_path
            else None,
            "interpretation": "All predeclared settings; no selection of the best test-fold transform.",
        },
    )
