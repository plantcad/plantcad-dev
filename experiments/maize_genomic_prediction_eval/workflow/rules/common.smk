from Bio.Seq import Seq
from biofoundation.data import Genome
from liftover import ChainFile
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyBigWig
import seaborn as sns


CHROMS = [str(i) for i in range(1, 11)]


def reverse_complement(seq_str: str) -> str:
    """Reverse complement a DNA sequence."""
    return str(Seq(seq_str).reverse_complement())


def liftover_variant(row: tuple, converter: ChainFile) -> tuple[str, int, str, str]:
    """
    Lift over coordinates for a single variant.

    Parameters:
    -----------
    row : tuple
        Tuple of (chrom, pos, ref, alt) values
    converter : ChainFile
        ChainFile converter object

    Returns:
    --------
    tuple[str, int, str, str]
        Tuple of (chrom, pos, ref, alt). If conversion fails, pos == -1.
    """
    chrom, pos, ref, alt = row
    res = converter.convert_coordinate(chrom, pos)
    if res and len(res) == 1:
        new_chrom, new_pos, strand = res[0]
        if new_chrom not in CHROMS:
            return (new_chrom, -1, ref, alt)
        if strand == "-":
            ref = reverse_complement(ref)
            alt = reverse_complement(alt)
        return (new_chrom, new_pos, ref, alt)
    else:
        # Conversion failed
        return (chrom, -1, ref, alt)


def liftover_variants(V: pl.DataFrame, converter: ChainFile) -> pl.DataFrame:
    """
    Lift over variant coordinates between reference genomes using a chain file.

    Parameters:
    -----------
    V : pl.DataFrame
        Polars DataFrame with columns: chrom, pos, ref, alt
    converter : ChainFile
        ChainFile converter object

    Returns:
    --------
    pl.DataFrame
        New DataFrame with lifted-over coordinates (chrom, pos, ref, alt)
        Variants that fail to lift over have pos == -1
    """
    # Select columns and apply conversion to each row
    result = V.select(["chrom", "pos", "ref", "alt"]).map_rows(
        lambda row: liftover_variant(row, converter)
    ).rename({
        "column_0": "chrom",
        "column_1": "pos",
        "column_2": "ref",
        "column_3": "alt"
    })

    return result


def check_ref_alleles(V: pl.DataFrame, genome: Genome) -> pl.DataFrame:
    """
    Check that reference alleles match the reference genome sequence.
    Sets pos == -1 for variants where fasta_ref != ref.

    Parameters:
    -----------
    V : pl.DataFrame
        Polars DataFrame with columns: chrom, pos, ref, alt
    genome : Genome
        Genome object initialized with reference FASTA file

    Returns:
    --------
    pl.DataFrame
        DataFrame with pos set to -1 for mismatched ref alleles
    """
    def get_fasta_ref(row: tuple) -> str:
        """Get reference nucleotide from genome FASTA at position."""
        chrom, pos, ref, alt = row

        # Skip validation for failed liftover
        if pos == -1:
            return ""

        # Genome uses 0-based coordinates, variant coordinates are 1-based
        return genome(chrom, pos - 1, pos).upper()

    fasta_ref = V.select(["chrom", "pos", "ref", "alt"]).map_rows(
        get_fasta_ref
    ).rename({"map": "fasta_ref"})

    # Set pos == -1 when fasta_ref != ref (but keep existing pos == -1)
    result = V.with_columns(fasta_ref).with_columns(
        pl.when((pl.col("pos") != -1) & (pl.col("fasta_ref") != pl.col("ref")))
        .then(-1)
        .otherwise(pl.col("pos"))
        .alias("pos")
    ).drop("fasta_ref")

    return result


def compute_conservation_score(V: pl.DataFrame, bw) -> pl.DataFrame:
    """
    Compute conservation score for a variant.

    Parameters:
    -----------
    V : pl.DataFrame
        Polars DataFrame with columns: chrom, pos
    bw : pyBigWig.BigWig

    Returns:
    --------
    pl.DataFrame
        DataFrame with conservation score
    """

    def compute_score(x: tuple) -> float:
        chrom, pos = x
        return bw.values(chrom, pos-1, pos)[0]

    return V.map_rows(compute_score).rename({"map": "score"})


rule filter_defined_variants:
    input:
        "results/variants.parquet"
    output:
        "results/defined_variants.parquet"
    run:
        pl.read_parquet(input[0]).filter(pl.col("pos") != -1).write_parquet(output[0])


rule download_genome:
    output:
        "results/genome.fa.gz"
    shell:
        "wget -O {output} {config[genome_url]}"


rule download_input_data:
    output:
        "results/input_data/hybrids/G.rds",
        "results/input_data/hybrids/Q.rds",
        "results/input_data/hybrids/AGPv4_hybrids.gds",
        "results/input_data/NAM_H/pheno.rds",
        "results/input_data/Ames_H/pheno.rds"
    shell:
        "hf download {config[input_data_hf_repo_id]} --repo-type dataset --local-dir results/input_data"
