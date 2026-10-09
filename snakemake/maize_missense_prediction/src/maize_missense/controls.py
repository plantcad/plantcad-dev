"""Score-free missense controls with explicit, reproducible weight matching."""

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

from maize_missense.data import PathLike, require_unique, write_json
from maize_missense.weights import weight_fingerprint


def permute_weights(weights: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Shuffle weights over sorted SNP IDs, preserving the exact weight multiset.

    The same seed gives the same SNP permutation across settings. This preserves
    their overlap structure and exact tied-kernel equivalences. No phenotype,
    score stratification, MAF matching or fold information enters randomization.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("The shuffle seed must be a nonnegative integer")
    frame = weights.select("snp_id", "weight").sort("snp_id")
    require_unique(frame, ["snp_id"], "control SNP IDs")
    if frame["snp_id"].null_count() or frame["weight"].null_count():
        raise ValueError("Control SNP IDs and weights cannot be missing")
    values = frame["weight"].to_numpy().astype(np.float64)
    # Reuse the production validator, including the nonempty/positive-weight checks.
    weight_fingerprint(frame["snp_id"].to_list(), values)
    shuffled = np.random.Generator(np.random.PCG64(seed)).permutation(values)
    return frame.with_columns(pl.Series("weight", shuffled))


def make_control_weights(
    source: PathLike,
    output: PathLike,
    report: PathLike,
    setting: str,
    seed: int | None = None,
) -> None:
    """Build all-missense weights or a single seeded weight permutation."""
    if seed is None:
        if setting != "all_missense":
            raise ValueError("An unweighted control must be named all_missense")
        frame = (
            pl.read_parquet(source, columns=["snp_id"])
            .sort("snp_id")
            .with_columns(pl.lit(1.0).alias("weight"))
        )
    else:
        frame = permute_weights(pl.read_parquet(source), seed)
    if frame["snp_id"].null_count():
        raise ValueError("Control SNP IDs cannot be missing")
    values = frame["weight"].to_numpy().astype(np.float64)
    fingerprint = weight_fingerprint(frame["snp_id"].to_list(), values)
    frame.write_parquet(output)
    write_json(
        report,
        {
            "setting": setting,
            "kind": "all_missense" if seed is None else "shuffle",
            "seed": seed,
            "rng": None if seed is None else "NumPy PCG64",
            "source": str(source),
            "n_variants": frame.height,
            "n_nonzero": int(np.count_nonzero(values)),
            "weight_sum": float(values.sum()),
            "effective_n": float(values.sum() ** 2 / np.square(values).sum()),
            "weight_fingerprint": fingerprint,
            "permutation_scope": "All matched, common-finite missense SNPs; canonical SNP-ID order",
        },
    )


def plan_controls(report_paths: Iterable[PathLike], output: PathLike) -> None:
    """Reuse control fits only for identical SNP identities and relative weights."""
    reports = [json.loads(Path(path).read_text()) for path in report_paths]
    canonical: dict[str, str] = {}
    representatives: dict[str, str] = {}
    for report in sorted(reports, key=lambda row: row["setting"]):
        setting = report["setting"]
        if setting in canonical:
            raise ValueError(f"Duplicate control setting: {setting}")
        canonical[setting] = representatives.setdefault(report["weight_fingerprint"], setting)
    if not canonical:
        raise ValueError("No control settings")
    write_json(
        output,
        {
            "canonical": canonical,
            "n_settings": len(canonical),
            "n_unique_kernels": len(representatives),
            "weights": reports,
            "interpretation": "A single shuffle is a sanity check, not a calibrated permutation test.",
        },
    )


def control_inputs(plan_json: str, setting: str) -> dict[str, str]:
    """Resolve the canonical control fit without changing the requested label."""
    canonical = json.loads(plan_json)["canonical"][setting]
    root = f"results/controls/evaluation_unique/{canonical}"
    return {
        name: f"{root}/{filename}"
        for name, filename in {
            "metrics": "metrics.parquet",
            "predictions": "predictions.parquet",
            "report": "fit.json",
        }.items()
    }
