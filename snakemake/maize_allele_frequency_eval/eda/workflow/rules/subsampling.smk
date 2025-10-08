# =============================================================================
# DATA PREPARATION AND SUBSAMPLING RULES
# =============================================================================
# Rules for creating complete dataset and subsampled datasets for stability analysis

rule create_complete_dataset:
    input:
        variants="results/variants.parquet",
        predictions=expand("results/predictions/{model}.parquet", model=config["pcad1_models"]),
    output:
        "results/complete_dataset.parquet",
    run:
        # Load variants (don't filter yet)
        variants = pl.read_parquet(input.variants)

        # Start with variants as base
        complete_data = variants

        # Add each model's predictions BEFORE filtering
        for model in config["pcad1_models"]:
            pred_path = f"results/predictions/{model}.parquet"
            if pred_path in input.predictions:
                predictions = pl.read_parquet(pred_path)

                # Apply sign flipping if needed
                if model in config["flip_sign_models"]:
                    predictions = predictions.with_columns(-pl.col("score"))

                # Add model score column
                complete_data = complete_data.with_columns(
                    predictions.select(pl.col("score").alias(model))
                )

        # NOW filter to analysis chromosomes
        complete_data = filter_analysis_chromosomes(complete_data)

        # Create binary labels for classification metrics (rare vs common)
        rare_threshold = config["rare_threshold"]
        common_threshold = config["common_threshold"]

        complete_data = complete_data.with_columns(
            pl.when(pl.col("AC") == rare_threshold["AC"])
            .then(True)
            .when(pl.col("AF") > common_threshold["AF"])
            .then(False)
            .otherwise(None)
            .alias("label")
        )

        # Assert no NaN values in model scores
        for model in config["pcad1_models"]:
            nan_count = complete_data.select(model).null_count().item()
            assert nan_count == 0, f"Found {nan_count} NaN values in {model} scores"

        # Sort by coordinates for consistency
        complete_data = complete_data.sort(COORDINATES)

        # Save complete dataset
        complete_data.write_parquet(output[0])

        print(f"Created complete dataset with {len(complete_data)} variants and {len(config['pcad1_models'])} models")


rule create_subsamples:
    input:
        "results/complete_dataset.parquet",
    output:
        "results/subsamples/frac_{fraction}_seed_{seed}.parquet",
    run:
        # Load complete dataset
        complete_data = pl.read_parquet(input[0])

        # Use polars built-in sampling with seed
        fraction = float(wildcards.fraction)
        seed = int(wildcards.seed)

        # Sample using polars (works fine for fraction=1.0)
        subsample = complete_data.sample(
            fraction=fraction,
            seed=seed
        )

        # Sort by coordinates for consistency
        subsample = subsample.sort(COORDINATES)

        # Save subsample
        subsample.write_parquet(output[0])

        print(f"Created subsample: {fraction} fraction, seed {seed}, {len(subsample)} variants")
