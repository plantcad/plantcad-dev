rule download_phastCons:
    output:
        temp("results/conservation/phastCons/{chrom}.wig"),
    wildcard_constraints:
        chrom="|".join(CHROMS),
    shell:
        "wget -O {output} https://huggingface.co/datasets/plantcad/andropogoneae_alignment_raw_data/resolve/main/andropogoneae_phastcons/chr{wildcards.chrom}_phastCons.wig"


rule download_phyloP:
    output:
        temp("results/conservation/phyloP/{chrom}.wig"),
    wildcard_constraints:
        chrom="|".join(CHROMS),
    shell:
        "wget -O {output} https://huggingface.co/datasets/plantcad/andropogoneae_alignment_raw_data/resolve/main/andropogoneae_phylop/chr{wildcards.chrom}_phyloP_SPH.wig"


rule conservation_merge_chroms:
    input:
        expand("results/conservation/{{conservation}}/{chrom}.wig", chrom=CHROMS),
    output:
        temp("results/conservation/{conservation}/merged.wig"),
    shell:
        "cat {input} | sed 's/chrom=panand_chr/chrom=/g' > {output}"


rule wig_to_bigwig:
    input:
        "results/conservation/{conservation}/merged.wig",
        "config/chrom.sizes",
    output:
        "results/dataset/{conservation}.bw",
    shell:
        "wigToBigWig {input} {output}"
