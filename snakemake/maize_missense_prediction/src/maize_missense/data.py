"""Polars data preparation and bounded-batch genotype-coordinate matching."""

import csv
import gzip
import hashlib
import json
import shutil
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl

COORDS = ["chrom", "pos", "ref", "alt"]
MODELS = ["PlantCAD", "phyloP", "phastCons"]
type PathLike = str | Path


def write_json(path: PathLike, value: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_digest(*paths: PathLike) -> str:
    """Include external function code in Snakemake's parameter rerun checks."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def download_hf(repo: str, revision: str, filename: str, output: PathLike) -> None:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, output)


def download_url(url: str, output: PathLike, decompress: bool = False) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, open(output, "wb") as dest:
        if decompress:
            with gzip.GzipFile(fileobj=response) as source:
                shutil.copyfileobj(source, dest, length=1024 * 1024)
        else:
            shutil.copyfileobj(response, dest, length=1024 * 1024)


def coordinates(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.with_columns(
        pl.col("chrom").cast(pl.String),
        pl.col("pos").cast(pl.Int64),
        pl.col("ref", "alt").str.to_uppercase(),
    )


def require_unique(frame: pl.DataFrame, keys: list[str], label: str) -> None:
    if frame.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError(f"Duplicate {label}; refusing an ambiguous or many-to-many join")


def prepare_scores(
    annotation_paths: Iterable[PathLike],
    score_path: PathLike,
    output: PathLike,
    report: PathLike,
    sources: Mapping[str, object],
) -> None:
    annotations = coordinates(pl.scan_parquet([str(path) for path in annotation_paths]))
    missense = (
        annotations.filter(pl.col("consequence") == "missense_variant")
        .select(COORDS)
        .collect(engine="streaming")
    )
    require_unique(missense, COORDS, "missense coordinates")
    score_scan = coordinates(
        pl.scan_parquet(score_path)
        .select(["CHR", "POS", "REF", "ALT"] + MODELS)
        .rename(dict(zip(["CHR", "POS", "REF", "ALT"], COORDS)))
    )
    scores = score_scan.join(missense.lazy(), on=COORDS, how="semi").collect(engine="streaming")
    require_unique(scores, COORDS, "score coordinates")
    available = {model: scores.filter(pl.col(model).is_finite()).height for model in MODELS}
    common = scores.filter(pl.all_horizontal(pl.col(MODELS).is_finite())).sort(COORDS)
    if not common.height:
        raise ValueError("No missense variants have finite scores from all three methods")
    common.write_parquet(output)
    write_json(
        report,
        {
            "sources": sources,
            "annotation_definition": "most-severe consequence == missense_variant",
            "n_annotated_missense": missense.height,
            "n_with_score_row": scores.height,
            "n_finite_by_model": available,
            "n_common_finite": common.height,
            "n_missing_score_row": missense.height - scores.height,
        },
    )


def map_genotypes(
    metadata_path: PathLike,
    score_path: PathLike,
    chain_path: PathLike,
    genome_path: PathLike,
    output: PathLike,
    report: PathLike,
) -> None:
    from liftover import ChainFile
    from pyfaidx import Fasta

    scores = pl.read_parquet(score_path)
    # Hold only the candidate missense set, not all 13M GDS rows.
    candidates = set(scores.select("chrom", "pos").iter_rows())
    valid_keys = set(scores.select(COORDS).iter_rows())
    converter = ChainFile(str(chain_path), one_based=True)
    genome = Fasta(str(genome_path), as_raw=True, sequence_always_upper=True)
    complement = str.maketrans("ACGT", "TGCA")
    nuclear_chromosomes = {str(i) for i in range(1, 11)}
    counts: Counter[str] = Counter()
    matches = []
    try:
        with gzip.open(metadata_path, "rt") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                counts["gds_rows"] += 1
                if counts["gds_rows"] % 1_000_000 == 0:
                    print(
                        f"Matched {len(matches):,} candidates after {counts['gds_rows']:,} GDS rows",
                        flush=True,
                    )
                chrom, pos = row["chrom"], int(row["pos"])
                alleles = row["alleles"].upper().replace(",", "/").split("/")
                if len(alleles) != 2 or any(a not in ("A", "C", "G", "T") for a in alleles):
                    counts["non_biallelic_snv"] += 1
                    continue
                if chrom not in nuclear_chromosomes:
                    counts["non_nuclear_chromosome"] += 1
                    continue
                hits = converter.convert_coordinate(chrom, pos)
                if not hits or len(hits) != 1:
                    counts["failed_or_ambiguous_liftover"] += 1
                    continue
                target_chrom, target_pos, strand = hits[0]
                target_chrom = str(target_chrom)
                if (target_chrom, target_pos) not in candidates:
                    counts["outside_scored_missense_positions"] += 1
                    continue
                if strand == "-":
                    alleles = [a.translate(complement) for a in alleles]
                reference = genome[target_chrom][target_pos - 1 : target_pos]
                swapped = False
                if alleles[0] == reference:
                    ref, alt = alleles
                elif alleles[1] == reference:
                    alt, ref = alleles
                    swapped = True
                else:
                    counts["reference_mismatch"] += 1
                    continue
                key = (target_chrom, target_pos, ref, alt)
                if key not in valid_keys:
                    counts["alternate_mismatch"] += 1
                    continue
                counts["matched_before_duplicate_filter"] += 1
                counts["matched_allele_order_swapped"] += int(swapped)
                matches.append(
                    (
                        row["snp_id"],
                        *key,
                        chrom,
                        pos,
                        row["alleles"],
                        strand,
                        swapped,
                    )
                )
    finally:
        genome.close()
    if not matches:
        raise ValueError(f"No genotype variants matched the scored missense set: {dict(counts)}")
    mapped = pl.DataFrame(
        matches,
        schema={
            "snp_id": pl.String,
            "chrom": pl.String,
            "pos": pl.Int64,
            "ref": pl.String,
            "alt": pl.String,
            "source_chrom": pl.String,
            "source_pos": pl.Int64,
            "source_alleles": pl.String,
            "liftover_strand": pl.String,
            "allele_order_swapped": pl.Boolean,
        },
        orient="row",
    )
    require_unique(mapped, ["snp_id"], "GDS SNP IDs")
    duplicated = mapped.select(pl.struct(COORDS).is_duplicated()).to_series()
    counts["ambiguous_duplicate_target_rows"] = int(duplicated.sum())
    mapped = mapped.filter(~duplicated).join(scores, on=COORDS, how="inner", validate="1:1")
    if not mapped.height:
        raise ValueError("All matched variants have ambiguous target coordinates")
    mapped.sort(COORDS).write_parquet(output)
    write_json(
        report,
        {
            "counts": dict(counts),
            "n_scored_missense_candidates": scores.height,
            "n_evaluation_variants": mapped.height,
            "fraction_candidates_matched": mapped.height / scores.height,
            "per_chromosome": mapped.group_by("chrom").len().sort("chrom").to_dicts(),
            "genome_build": "Zm-B73-REFERENCE-NAM-5.0",
            "gds_build": "AGPv4",
        },
    )
