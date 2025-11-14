# Common utilities and data download rules

rule download_input_data:
    output:
        "results/input_data/hybrids/G.rds",
        "results/input_data/hybrids/Q.rds",
        "results/input_data/hybrids/AGPv4_hybrids.gds",
        "results/input_data/NAM_H/pheno.rds",
        "results/input_data/Ames_H/pheno.rds"
    shell:
        "hf download {config[input_data_hf_repo_id]} --repo-type dataset --local-dir results/input_data"
