rule download_conservation:
    output:
        "results/conservation/{conservation}.bw",
    params:
        lambda wildcards: config["conservation_models"][wildcards.conservation],
    wildcard_constraints:
        conservation="|".join(config["conservation_models"].keys()),
    shell:
        "wget {params} -O {output}"


rule score_conservation:
    input:
        "results/defined_variants.parquet",
        "results/conservation/{conservation}.bw",
    output:
        "results/defined_variants_scores/{conservation}.parquet",
    wildcard_constraints:
        conservation="|".join(config["conservation_models"].keys()),
    run:
        V = pl.read_parquet(input[0], columns=["chrom", "pos"])
        bw = pyBigWig.open(input[1])
        res = compute_conservation_score(V, bw)
        res.write_parquet(output[0])


rule score_hf_llr:
    input:
        "results/defined_variants.parquet",
        "results/genome.fa.gz",
    output:
        "results/defined_variants_scores/{model}_LLR.parquet",
    wildcard_constraints:
        model="|".join(config["hf_models"].keys()),
    threads: workflow.cores
    params:
        model_path=lambda wildcards: config["hf_models"][wildcards.model]["path"],
        context_size=lambda wildcards: config["hf_models"][wildcards.model]["context_size"],
        batch_size=lambda wildcards: config["hf_models"][wildcards.model]["per_device_eval_batch_size"],
        n_gpu=lambda wildcards: config["n_gpu"],
        n_workers=workflow.cores // config["n_gpu"],
    shell:
        "torchrun --nproc_per_node={params.n_gpu} workflow/scripts/score_hf_llr.py "
        "--variants {input[0]} "
        "--genome {input[1]} "
        "--model-path {params.model_path} "
        "--context-size {params.context_size} "
        "--batch-size {params.batch_size} "
        "--n-workers {params.n_workers} "
        "--output {output}"


rule abs_llr:
    input:
        "results/defined_variants_scores/{model}_LLR.parquet",
    output:
        "results/defined_variants_scores/{model}_absLLR.parquet",
    wildcard_constraints:
        model="|".join(config["hf_models"].keys()),
    run:
        pl.read_parquet(input[0]).with_columns(pl.col("score").abs()).write_parquet(output[0])


rule expand_scores:
    input:
        "results/variants.parquet",
        "results/defined_variants_scores/{model}.parquet",
    output:
        "results/variant_scores/{model}.parquet",
    run:
        variants = pl.read_parquet(input[0])
        defined_scores = pl.read_parquet(input[1])

        # Create mask for defined variants (pos != -1)
        mask = (variants["pos"] != -1).to_numpy()

        # Initialize full score array with NaN
        n_variants = len(variants)
        full_scores = np.full(n_variants, np.nan)

        # Apply scores where mask is True
        full_scores[mask] = defined_scores["score"].to_numpy()

        # Create output DataFrame with score column
        result = pl.DataFrame({"score": full_scores})
        result.write_parquet(output[0])


rule quantile_binarization:
    input:
        "results/variant_scores/{model}.parquet",
    output:
        "results/variant_scores/quantile_binarized/{model}/{q}.parquet",
    run:
        q = float(wildcards.q)
        x = pl.read_parquet(input[0])
        n_total = len(x)
        n_positive = int(q * n_total)

        new_x = (
            x
            # ensure that NaN values are treated as the smallest value
            .fill_nan(x["score"].min() - 1)
            .with_row_index("original_idx")
            .sample(fraction=1, seed=42, shuffle=True)
            .sort("score", descending=True, maintain_order=True)
            .with_columns(
                pl.when(pl.int_range(pl.len()) < n_positive)
                .then(1.0)
                .otherwise(0.0)
                .alias("score")
            )
            .sort("original_idx")
            .select("score")
        )
        new_x.write_parquet(output[0])


rule subtract_and_clip:
    input:
        "results/variant_scores/{model}.parquet",
    output:
        "results/variant_scores/subtract_and_clip/{model}/{val}.parquet",
    run:
        val = float(wildcards.val)
        (
            pl.read_parquet(input[0])
            .with_columns((pl.col("score") - val).clip(lower_bound=0).alias("score"))
            .write_parquet(output[0])
        )


rule quantile_set_to_zero:
    input:
        "results/variant_scores/{model}.parquet",
    output:
        "results/variant_scores/quantile_set_to_zero/{model}/{q}.parquet",
    run:
        q = float(wildcards.q)
        x = pl.read_parquet(input[0])
        n_total = len(x)
        n_top = int(q * n_total)

        new_x = (
            x
            # ensure that NaN values are treated as the smallest value
            .fill_nan(x["score"].min() - 1)
            .with_row_index("original_idx")
            .sample(fraction=1, seed=42, shuffle=True)
            .sort("score", descending=True, maintain_order=True)
            .with_columns(
                pl.when(pl.int_range(pl.len()) < n_top)
                .then(pl.col("score"))
                .otherwise(0.0)
                .alias("score")
            )
            .sort("original_idx")
            .select("score")
        )
        new_x.write_parquet(output[0])
