import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from maize_missense.data import COORDS, map_genotypes, prepare_scores, source_digest


def test_source_digest_tracks_external_function_changes(tmp_path: Path) -> None:
    source = tmp_path / "function.py"
    source.write_text("original")
    original = source_digest(source)
    source.write_text("changed")
    assert source_digest(source) != original


def test_reverse_strand_liftover_complements_alleles(tmp_path: Path) -> None:
    genome, chain = tmp_path / "v5.fa", tmp_path / "lift.chain"
    genome.write_text(">1\n" + "A" * 30 + "\n")
    chain.write_text("chain 100 1 30 + 0 30 1 30 - 0 30 1\n30\n\n")
    metadata = tmp_path / "gds.tsv.gz"
    with gzip.open(metadata, "wt") as stream:
        stream.write("snp_id\tchrom\tpos\talleles\n11\t1\t1\tT/G\n")
    score_path = tmp_path / "scores.parquet"
    pl.DataFrame(
        {
            "chrom": ["1"],
            "pos": [30],
            "ref": ["A"],
            "alt": ["C"],
            "PlantCAD": [1.0],
            "phyloP": [1.0],
            "phastCons": [0.5],
        }
    ).write_parquet(score_path)
    output, report = tmp_path / "out.parquet", tmp_path / "report.json"
    map_genotypes(metadata, score_path, chain, genome, output, report)
    result = pl.read_parquet(output)
    assert result["pos"].to_list() == [30]
    assert result["liftover_strand"].to_list() == ["-"]
    assert result.select("ref", "alt").row(0) == ("A", "C")


def test_common_complete_missense_set_and_coverage(tmp_path: Path) -> None:
    annotated = tmp_path / "annotated.parquet"
    raw = tmp_path / "scores.parquet"
    output, report = tmp_path / "out.parquet", tmp_path / "report.json"
    pl.DataFrame(
        {
            "chrom": ["1"] * 5,
            "pos": [1, 2, 3, 4, 5],
            "ref": ["A"] * 5,
            "alt": ["C"] * 5,
            "consequence": ["missense_variant"] * 4 + ["intron_variant"],
        }
    ).write_parquet(annotated)
    pl.DataFrame(
        {
            "CHR": [1] * 4,
            "POS": [3, 1, 2, 5],
            "REF": ["A"] * 4,
            "ALT": ["C"] * 4,
            "PlantCAD": [-2.0, 1.0, 3.0, 4.0],
            "phyloP": [1.0, 1.0, float("nan"), 1.0],
            "phastCons": [None, 0.5, 0.5, 0.5],
        }
    ).write_parquet(raw)
    prepare_scores([annotated], raw, output, report, {})
    assert pl.read_parquet(output)["pos"].to_list() == [1]
    qc = json.loads(report.read_text())
    assert qc["n_annotated_missense"] == 4
    assert qc["n_missing_score_row"] == 1
    assert qc["n_finite_by_model"] == {"PlantCAD": 3, "phyloP": 2, "phastCons": 2}


def test_duplicate_score_keys_rejected(tmp_path: Path) -> None:
    annotations, raw = tmp_path / "ann.parquet", tmp_path / "raw.parquet"
    pl.DataFrame(
        {
            "chrom": ["1"],
            "pos": [1],
            "ref": ["A"],
            "alt": ["C"],
            "consequence": ["missense_variant"],
        }
    ).write_parquet(annotations)
    pl.DataFrame(
        {
            "CHR": [1, 1],
            "POS": [1, 1],
            "REF": ["A", "A"],
            "ALT": ["C", "C"],
            "PlantCAD": [1.0, 2.0],
            "phyloP": [1.0, 2.0],
            "phastCons": [0.5, 0.5],
        }
    ).write_parquet(raw)
    with pytest.raises(ValueError, match="Duplicate score"):
        prepare_scores([annotations], raw, tmp_path / "out", tmp_path / "report", {})


def test_mapping_handles_allele_order_and_duplicate_targets(tmp_path: Path) -> None:
    genome, chain = tmp_path / "v5.fa", tmp_path / "lift.chain"
    genome.write_text(">1\n" + "A" * 30 + "\n")
    chain.write_text("chain 100 1 30 + 0 30 1 30 + 0 30 1\n30\n\n")
    metadata = tmp_path / "gds.tsv.gz"
    with gzip.open(metadata, "wt") as stream:
        stream.write(
            "snp_id\tchrom\tpos\talleles\n11\t1\t1\tA/C\n12\t1\t2\tC/A\n"
            "13\t1\t3\tG/C\n14\t1\t4\tA/C\n15\t1\t4\tA/C\n16\t1\t5\tA/CG\n"
        )
    score_path = tmp_path / "scores.parquet"
    pl.DataFrame(
        {
            "chrom": ["1"] * 5,
            "pos": [1, 2, 3, 4, 5],
            "ref": ["A"] * 5,
            "alt": ["C"] * 5,
            "PlantCAD": [1.0] * 5,
            "phyloP": [1.0] * 5,
            "phastCons": [0.5] * 5,
        }
    ).write_parquet(score_path)
    output, report = tmp_path / "out.parquet", tmp_path / "report.json"
    map_genotypes(metadata, score_path, chain, genome, output, report)
    result = pl.read_parquet(output)
    assert result["snp_id"].to_list() == ["11", "12"]
    assert result["allele_order_swapped"].to_list() == [False, True]
    assert result.select(COORDS).height == 2
    assert json.loads(report.read_text())["counts"]["ambiguous_duplicate_target_rows"] == 2
