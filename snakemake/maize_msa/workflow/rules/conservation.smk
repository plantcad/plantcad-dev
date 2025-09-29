# takes like 15 min
rule phyloP:
    input:
        "results/neutral.mod",
        "results/maf/{chrom}.maf",
    output:
        temp("results/conservation/phyloP/{chrom}.wig"),
    wildcard_constraints:
        chrom="|".join(CHROMS),
    shell:
        "phyloP --wig-scores --method LRT --mode CONACC {input} > {output}"


# takes like 3 min
rule phastCons:
    input:
        "results/maf/{chrom}.maf",
        "results/neutral.mod",
    output:
        temp("results/conservation/phastCons_{target_coverage}_{expected_length}_{rho}/{chrom}.wig"),
    wildcard_constraints:
        chrom="|".join(CHROMS),
    shell:
        """
        phastCons \
        --seqname {wildcards.chrom} \
        --target-coverage {wildcards.target_coverage} \
        --expected-length {wildcards.expected_length} \
        --rho {wildcards.rho} \
        {input} > {output}
        """


rule merge_chroms:
    input:
        expand("results/conservation/{{conservation}}/{chrom}.wig", chrom=CHROMS),
    output:
        temp("results/conservation/{conservation}/merged.wig"),
    shell:
        "cat {input} > {output}"


rule wig_to_bigwig:
    input:
        "results/conservation/{conservation}/merged.wig",
        "config/chrom.sizes",
    output:
        temp("results/conservation/{conservation}/merged.bw"),
    shell:
        "wigToBigWig {input} {output}"


rule cp_phyloP:
    input:
        f"results/conservation/{config['phyloP']}/merged.bw",
    output:
        "results/dataset/phyloP.bw",
    shell:
        "cp {input} {output}"


rule cp_phastCons:
    input:
        f"results/conservation/{config['phastCons']}/merged.bw",
    output:
        "results/dataset/phastCons.bw",
    shell:
        "cp {input} {output}"
