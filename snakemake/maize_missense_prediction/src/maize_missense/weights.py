"""Deterministic soft top-k selection; tied variants always have equal weights."""

import hashlib
import json
import math
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from maize_missense.data import PathLike, write_json


class WeightStats(TypedDict):
    fraction: float
    n_variants: int
    target_k: int
    cutoff: float
    n_above: int
    n_tied_at_cutoff: int
    boundary_weight: float
    n_nonzero: int
    weight_sum: float
    effective_n: float


def weight_fingerprint(snp_ids: Sequence[str], weights: NDArray[np.float64]) -> str:
    """Identify exactly equal relative weights without a tolerance-based merge."""
    if len(snp_ids) != len(weights) or len(set(snp_ids)) != len(snp_ids):
        raise ValueError("Weights require one unique SNP ID per entry")
    if not len(weights) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    maximum = float(weights.max())
    if maximum <= 0:
        raise ValueError("At least one weight must be positive")
    ordered = pl.DataFrame({"snp_id": snp_ids, "weight": weights / maximum}).sort("snp_id")
    digest = hashlib.sha256(json.dumps(ordered["snp_id"].to_list()).encode())
    digest.update(ordered["weight"].to_numpy().astype("<f8").tobytes())
    return digest.hexdigest()


def soft_top_fraction(
    scores: ArrayLike, fraction: float
) -> tuple[NDArray[np.float64], WeightStats]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Scores must be a nonempty, one-dimensional finite array")
    if not 0 < fraction <= 1:
        raise ValueError("Fraction must lie in (0, 1]")
    k = math.floor(len(values) * fraction)
    if k < 1:
        raise ValueError("This fraction selects fewer than one variant")
    cutoff = float(np.partition(values, len(values) - k)[len(values) - k])
    above, tied = values > cutoff, values == cutoff
    n_above, n_tied = int(above.sum()), int(tied.sum())
    boundary_weight = (k - n_above) / n_tied
    weights = above.astype(np.float64) + tied * boundary_weight
    normalized = weights / k
    return weights, {
        "fraction": float(fraction),
        "n_variants": len(values),
        "target_k": k,
        "cutoff": cutoff,
        "n_above": n_above,
        "n_tied_at_cutoff": n_tied,
        "boundary_weight": boundary_weight,
        "n_nonzero": int(np.count_nonzero(weights)),
        "weight_sum": float(weights.sum()),
        "effective_n": float(1 / np.square(normalized).sum()),
    }


def make_weights(
    variants_path: PathLike,
    output: PathLike,
    report: PathLike,
    model: str,
    fraction: float,
    transform: str,
) -> None:
    variants = pl.read_parquet(variants_path)
    score = pl.col(model)
    if transform == "abs":
        score = score.abs()
    elif transform != "identity":
        raise ValueError(f"Unknown score transform: {transform}")
    values = variants.select(score).to_series().to_numpy()
    weights, stats = soft_top_fraction(values, fraction)
    variants.select("snp_id").with_columns(pl.Series("weight", weights)).write_parquet(output)
    write_json(
        report,
        {
            **stats,
            "model": model,
            "transform": transform,
            "weight_fingerprint": weight_fingerprint(variants["snp_id"].to_list(), weights),
        },
    )
