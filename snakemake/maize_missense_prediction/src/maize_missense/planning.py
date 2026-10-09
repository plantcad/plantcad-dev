"""Reuse fits only when SNP identities and relative weights match exactly."""

import json
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from maize_missense.data import PathLike, write_json


def plan_evaluations(report_paths: Iterable[PathLike], output: PathLike) -> None:
    settings: list[tuple[str, float, str]] = []
    for path in report_paths:
        report = json.loads(Path(path).read_text())
        settings.append((report["model"], report["fraction"], report["weight_fingerprint"]))
    representatives: dict[str, str] = {}
    canonical: dict[str, str] = {}
    for model, fraction, fingerprint in sorted(settings):
        setting = f"{model}/{fraction}"
        if setting in canonical:
            raise ValueError(f"Duplicate evaluation setting: {setting}")
        canonical[setting] = representatives.setdefault(fingerprint, setting)
    if not canonical:
        raise ValueError("No evaluation settings")
    write_json(
        output,
        {
            "canonical": canonical,
            "n_settings": len(canonical),
            "n_unique_kernels": len(representatives),
            "equivalence": "Exact SNP-ID and max-normalized weight fingerprint",
        },
    )


def evaluation_inputs(plan_json: str, model: str, fraction: str) -> dict[str, str]:
    """Resolve the checkpoint's canonical fit outputs for one requested label."""
    canonical = json.loads(plan_json)["canonical"][f"{model}/{fraction}"]
    root = f"results/evaluation_unique/{canonical}"
    return {
        "metrics": f"{root}/metrics.parquet",
        "predictions": f"{root}/predictions.parquet",
        "report": f"{root}/fit.json",
    }


def relabel_evaluation(
    metrics_input: PathLike,
    predictions_input: PathLike,
    report_input: PathLike,
    metrics_output: PathLike,
    predictions_output: PathLike,
    report_output: PathLike,
    label: str,
) -> None:
    report = json.loads(Path(report_input).read_text())
    fitted_model = report["model"]
    for source, destination in [
        (metrics_input, metrics_output),
        (predictions_input, predictions_output),
    ]:
        table = pl.read_parquet(source)
        if table["model"].unique().to_list() != [fitted_model]:
            raise ValueError("The metric/prediction labels do not match the fit report")
        table.with_columns(pl.lit(label).alias("model")).write_parquet(destination)
    write_json(
        report_output,
        {
            **report,
            "model": label,
            "fitted_model": fitted_model,
            "reused_fit": fitted_model != label,
        },
    )
