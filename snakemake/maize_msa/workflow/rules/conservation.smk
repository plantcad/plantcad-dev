#rule download_phastCons:
#    output:
#        temp("results/conservation/phastCons/{chrom}.wig"),
#    wildcard_constraints:
#        chrom="|".join(CHROMS),
#    shell:
#        "wget -O {output} https://huggingface.co/datasets/plantcad/andropogoneae_alignment_raw_data/resolve/main/andropogoneae_phastcons/chr{wildcards.chrom}_phastCons.wig"


#rule download_phyloP:
#    output:
#        temp("results/conservation/phyloP/{chrom}.wig"),
#    wildcard_constraints:
#        chrom="|".join(CHROMS),
#    shell:
#        "wget -O {output} https://huggingface.co/datasets/plantcad/andropogoneae_alignment_raw_data/resolve/main/andropogoneae_phylop/chr{wildcards.chrom}_phyloP_SPH.wig"


#rule conservation_merge_chroms:
#    input:
#        expand("results/conservation/{{conservation}}/{chrom}.wig", chrom=CHROMS),
#    output:
#        temp("results/conservation/{conservation}/merged.wig"),
#    shell:
#        "cat {input} | sed 's/chrom=panand_chr/chrom=/g' > {output}"
#
#
#rule wig_to_bigwig:
#    input:
#        "results/conservation/{conservation}/merged.wig",
#        "config/chrom.sizes",
#    output:
#        "results/dataset/{conservation}.bw",
#    shell:
#        "wigToBigWig {input} {output}"


# msa_view requires chromosome 1 to be named zB73v5.1, etc. to match the MAF file
#rule download_annotation:
#    output:
#        "results/annotation/{chrom}.gff",
#    shell:
#        """
#        wget -O - https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/gff3/zea_mays/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.62.chromosome.{wildcards.chrom}.gff3.gz |
#        gunzip -c |
#        awk -F'\\t' 'BEGIN {{OFS=\"\\t\"}} {{if ($1 == \"{wildcards.chrom}\") $1 = \"zB73v5.{wildcards.chrom}\"; print}}' > {output}
#        """
#
#
#rule msa_view:
#    input:
#        "results/maf/{chrom}.maf",
#        "results/annotation/{chrom}.gff",
#    output:
#        temp("results/msa_view/4d_codons/{chrom}.ss"),
#        "results/msa_view/4d_sites/{chrom}.ss",
#    shell:
#        """
#        msa_view {input[0]} --4d --features {input[1]} > {output[0]} &&
#        msa_view {output[0]} --in-format SS --out-format SS --tuple-size 1 > {output[1]}
#        """
#
#
#rule phyloFit:
#    input:
#        "results/msa_view/4d_sites/{chrom}.ss",
#        "config/tree_topology.nh",
#    output:
#        "results/phyloFit/{chrom}.mod",
#    params:
#        "results/phyloFit/{chrom}",
#    shell:
#        """
#        phyloFit --tree {input[1]} --msa-format SS --out-root {params} --EM --precision MED {input[0]}
#        """

# takes like 15 min
rule phyloP:
    input:
        "config/neutral.mod",
        "results/maf/{chrom}.maf",
    output:
        "results/conservation/phyloP/{chrom}.wig",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    shell:
        "phyloP --wig-scores --method LRT --mode CONACC {input} > {output}"


#rule phyloP_v2:
#    input:
#        "results/phyloFit/{chrom}.mod",
#        "results/maf/{chrom}.maf",
#    output:
#        "results/phyloP_v2/{chrom}.wig",
#    shell:
#        "phyloP --wig-scores --method LRT --mode CONACC {input} > {output}"


# didn't finish in 1 hour
#rule phyloP_SPH:
#    input:
#        "config/neutral.mod",
#        "results/maf/{chrom}.maf",
#    output:
#        "results/conservation/phyloP_SPH/{chrom}.wig",
#    wildcard_constraints:
#        chrom="|".join(CHROMS),
#    log:
#        "logs/conservation/phyloP_SPH/{chrom}.log",
#    shell:
#        "phyloP --wig-scores --method SPH --mode CONACC --log {log} {input} > {output}"


# takes like 3 min when fixing target-coverage and expected-length
rule phastCons:
    input:
        "results/maf/{chrom}.maf",
        "config/neutral.mod",
    output:
        "results/conservation/phastCons_{target_coverage}_{expected_length}_{rho}/{chrom}.wig",
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

#        --target-coverage 0.05 \
#        --expected-length 12 \

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
        "results/conservation/{conservation}/merged.bw",
    shell:
        "wigToBigWig {input} {output}"
