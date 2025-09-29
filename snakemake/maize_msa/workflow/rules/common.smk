from Bio import Phylo, SeqIO
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr


CHROMS = config["chroms"]


def contains_target(clade, target):
    """Return True if `target` is the name of this clade or in any descendant."""
    if clade.name == target:
        return True
    return any(contains_target(c, target) for c in clade.clades)


def reorder_clades(clade, target):
    """Sort subclades so those containing `target` come first, then recurse."""
    # False < True, so not-containing-target will sort after containing-target
    clade.clades.sort(key=lambda c: not contains_target(c, target))
    for sub in clade.clades:
        reorder_clades(sub, target)


def load_fasta(path, subset_chroms=None):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as handle:
        return pd.Series(
            {
                rec.id: str(rec.seq)
                for rec in SeqIO.parse(handle, "fasta")
                if subset_chroms is None or rec.id in subset_chroms
            }
        )


# msa_view requires chromosome 1 to be named zB73v5.1, etc. to match the MAF file
rule download_annotation:
    output:
        "results/annotation/{chrom}.gff",
    shell:
        """
        wget -O - https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/gff3/zea_mays/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.62.chromosome.{wildcards.chrom}.gff3.gz |
        gunzip -c |
        awk -F'\\t' 'BEGIN {{OFS=\"\\t\"}} {{if ($1 == \"{wildcards.chrom}\") $1 = \"zB73v5.{wildcards.chrom}\"; print}}' > {output}
        """


rule msa_view:
    input:
        "results/maf/{chrom}.maf",
        "results/annotation/{chrom}.gff",
    output:
        temp("results/msa_view/4d_codons/{chrom}.ss"),
        "results/msa_view/4d_sites/{chrom}.ss",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    shell:
        """
        msa_view {input[0]} --4d --features {input[1]} > {output[0]} &&
        msa_view {output[0]} --in-format SS --out-format SS --tuple-size 1 > {output[1]}
        """


rule msa_view_merge:
    input:
        chrom_ss=expand("results/msa_view/4d_sites/{chrom}.ss", chrom=CHROMS),
        species="results/dataset/species.txt",
    output:
        "results/msa_view/4d_sites/merged.ss",
    shell:
        """
        msa_view --unordered-ss --out-format SS --aggregate "$(paste -sd, {input.species})" {input.chrom_ss} > {output}
        """


rule phyloFit:
    input:
        "results/msa_view/4d_sites/merged.ss",
        "config/tree_topology.nh",
    output:
        "results/neutral.mod",
    params:
        "results/neutral",
    shell:
        """
        phyloFit \
        --tree {input[1]} \
        --msa-format SS \
        --out-root {params} \
        --EM \
        --precision MED \
        {input[0]}
        """


rule phyloFit_extract_tree:
    input:
        "results/neutral.mod",
    output:
        "results/neutral.nh",
    shell:
        """
        grep "^TREE" {input} | cut -d' ' -f2- > {output}
        """
