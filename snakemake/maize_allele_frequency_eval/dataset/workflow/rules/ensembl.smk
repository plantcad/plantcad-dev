rule download_ensembl_vep_cache:
    output:
        directory("results/ensembl_vep_cache")
    params:
        url=config["ensembl_vep"]["cache_url"]
    shell:
        """
        mkdir -p {output}
        cd {output}
        wget {params.url}
        tar xzf $(basename {params.url})
        rm $(basename {params.url})
        """


rule make_ensembl_vep_input:
    input:
        "results/variants.parquet",
    output:
        temp("results/variants.ensembl_vep.input.tsv.gz"),
    threads: workflow.cores
    run:
        df = pd.read_parquet(input[0])
        df["start"] = df.pos
        df["end"] = df.start
        df["allele"] = df.ref + "/" + df.alt
        df["strand"] = "+"
        df.to_csv(
            output[0], sep="\t", header=False, index=False,
            columns=["chrom", "start", "end", "allele", "strand"],
        )


rule run_ensembl_vep:
    input:
        "results/variants.ensembl_vep.input.tsv.gz",
        "results/ensembl_vep_cache",
    output:
        temp("results/variants.ensembl_vep.output.tsv.gz"),
        temp("results/variants.ensembl_vep.output.tsv.gz_summary.html"),
    params:
        species=lambda wildcards: config["ensembl_vep"]["species"],
        version=lambda wildcards: config["ensembl_vep"]["cache_version"],
    container:
        "docker://ensemblorg/ensembl-vep:release_115.1"
    threads: workflow.cores
    shell:
        """
        vep -i {input[0]} -o {output[0]} --fork {threads} --cache \
        --dir_cache {input[1]} --format ensembl --species {params.species} \
        --most_severe --compress_output gzip --tab --distance 1000 --offline \
        --cache_version {params.version}
        """


rule process_ensembl_vep:
    input:
        "results/variants.parquet",
        "results/variants.ensembl_vep.output.tsv.gz",
    output:
        "results/variants.annot.parquet",
    run:
        V = pd.read_parquet(input[0])
        V2 = pd.read_csv(
            input[1], sep="\t", header=None, comment="#",
            usecols=[0, 6]
        ).rename(columns={0: "variant", 6: "consequence"})
        V2["chrom"] = V2.variant.str.split("_").str[0]
        V2["pos"] = V2.variant.str.split("_").str[1].astype(int)
        V2["ref"] = V2.variant.str.split("_").str[2].str.split("/").str[0]
        V2["alt"] = V2.variant.str.split("_").str[2].str.split("/").str[1]
        V2.drop(columns=["variant"], inplace=True)
        V = V.merge(V2, on=COORDINATES, how="left")
        V.to_parquet(output[0], index=False)
