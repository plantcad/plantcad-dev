rule extract_variant_coords:
    input:
        "results/input_data/hybrids/AGPv4_hybrids.gds"
    output:
        "results/variants_4.parquet"
    conda:
        "../envs/r-reml.yaml"
    shell:
        "Rscript workflow/scripts/extract_variant_coords.R "
        "--gds-file {input} --output {output}"


rule download_chain_file_4_to_5:
    output:
        "results/chain_file/4_to_5.chain"
    shell:
        "wget {config[chain_file_4_to_5_url]} -O {output}"


rule liftover_variants:
    input:
        "results/variants_4.parquet",
        "results/chain_file/4_to_5.chain",
        "results/genome.fa.gz"
    output:
        "results/variants.parquet"
    run:
        V = pl.read_parquet(input[0])
        converter = ChainFile(input[1], one_based=True)
        lifted_V = liftover_variants(V, converter)
        genome = Genome(input[2])
        validated_V = check_ref_alleles(lifted_V, genome)
        validated_V.write_parquet(output[0])


rule eval_baseline:
    input:
        g_file="results/input_data/hybrids/G.rds",
        q_file="results/input_data/hybrids/Q.rds",
        nam_pheno="results/input_data/NAM_H/pheno.rds",
        ames_pheno="results/input_data/Ames_H/pheno.rds"
    output:
        "results/metrics/baseline.parquet"
    conda:
        "../envs/r-reml.yaml"
    threads:
        workflow.cores
    shell:
        "Rscript workflow/scripts/eval_baseline_G0.R "
        "--hybrids-dir results/input_data/hybrids "
        "--nam-dir results/input_data/NAM_H "
        "--ames-dir results/input_data/Ames_H "
        "--output {output} "
        "--n-threads {threads}"


rule eval_score_model:
    input:
        scores_file="results/variant_scores/{model}.parquet",
        g_file="results/input_data/hybrids/G.rds",
        q_file="results/input_data/hybrids/Q.rds",
        gds_file="results/input_data/hybrids/AGPv4_hybrids.gds",
        nam_pheno="results/input_data/NAM_H/pheno.rds",
        ames_pheno="results/input_data/Ames_H/pheno.rds"
    output:
        "results/metrics/{model}.parquet"
    conda:
        "../envs/r-reml.yaml"
    threads:
        workflow.cores
    wildcard_constraints:
        model = "|".join(config["eval_models"])
    shell:
        "Rscript workflow/scripts/eval_score_model.R "
        "--hybrids-dir results/input_data/hybrids "
        "--nam-dir results/input_data/NAM_H "
        "--ames-dir results/input_data/Ames_H "
        "--gds-file {input.gds_file} "
        "--scores-file {input.scores_file} "
        "--output {output} "
        "--n-threads {threads}"
