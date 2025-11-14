# REML model evaluation rules (baseline and future models)

rule eval_baseline:
    input:
        g_file="results/input_data/hybrids/G.rds",
        q_file="results/input_data/hybrids/Q.rds",
        nam_pheno="results/input_data/NAM_H/pheno.rds",
        ames_pheno="results/input_data/Ames_H/pheno.rds"
    output:
        "results/CV_baseline.parquet"
    conda:
        "../envs/r-reml.yaml"
    threads:
        workflow.cores
    params:
        script="workflow/scripts/eval_baseline_G0.R",
        hybrids_dir="results/input_data/hybrids",
        nam_dir="results/input_data/NAM_H",
        ames_dir="results/input_data/Ames_H"
    shell:
        "Rscript {params.script} "
        "--hybrids-dir {params.hybrids_dir} "
        "--nam-dir {params.nam_dir} "
        "--ames-dir {params.ames_dir} "
        "--output {output} "
        "--n-threads {threads}"
