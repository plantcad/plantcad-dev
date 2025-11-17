rule download_conservation:
    output:
        "results/conservation/{conservation}.bw",
    params:
        lambda wildcards: config["conservation_models"][wildcards.conservation],
    wildcard_constraints:
        conservation="|".join(config["conservation_models"].keys()),
    shell:
        "wget {params} -O {output}"


rule conservation_score:
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
