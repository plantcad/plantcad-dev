# =============================================================================
# METRICS COMPUTATION RULES
# =============================================================================
# Rules for computing various metrics on variants and subsampled data

rule compute_mean_af_quantile:
    input:
        "results/subsamples/frac_{fraction}_seed_{seed}.parquet",
    output:
        "results/metrics/mean_af_quantile/{model}_{fraction}_{seed}.parquet",
    run:

        # Load only required columns for mean AF quantile
        model_name = wildcards.model
        score_col = model_name
        data = pl.read_parquet(input[0], columns=["AF", score_col])

        # Compute mean AF at quantiles
        quantiles = config["analysis_quantiles"]
        results = compute_mean_af_at_quantile(data, score_col, quantiles)

        # Transform to consistent format with quantile in metric name
        metric_rows = []
        for row in results.iter_rows(named=True):
            metric_rows.append({
                "model": model_name,
                "fraction": float(wildcards.fraction),
                "seed": int(wildcards.seed),
                "metric": f"mean_af_at_q{row['quantile']}",  # e.g., "mean_af_at_q0.001", "mean_af_at_q0.01"
                "score": row["mean_af"]
            })

        result_df = pl.DataFrame(metric_rows)

        # Save results
        result_df.write_parquet(output[0])


rule compute_pearson_correlation:
    input:
        "results/subsamples/frac_{fraction}_seed_{seed}.parquet",
    output:
        "results/metrics/pearson_correlation/{model}_{fraction}_{seed}.parquet",
    run:

        # Load only required columns for Pearson correlation
        model_name = wildcards.model
        score_col = model_name
        data = pl.read_parquet(input[0], columns=["AF", score_col])

        # Compute Pearson correlation
        result = compute_pearson_correlation(data, score_col)

        # Create result DataFrame
        result_df = pl.DataFrame([{
            "model": model_name,
            "fraction": float(wildcards.fraction),
            "seed": int(wildcards.seed),
            "metric": "pearson_correlation",
            "score": result
        }])

        # Save results
        result_df.write_parquet(output[0])


rule compute_spearman_correlation:
    input:
        "results/subsamples/frac_{fraction}_seed_{seed}.parquet",
    output:
        "results/metrics/spearman_correlation/{model}_{fraction}_{seed}.parquet",
    run:

        # Load only required columns for Spearman correlation
        model_name = wildcards.model
        score_col = model_name
        data = pl.read_parquet(input[0], columns=["AF", score_col])

        # Compute Spearman correlation
        result = compute_spearman_correlation(data, score_col)

        # Create result DataFrame
        result_df = pl.DataFrame([{
            "model": model_name,
            "fraction": float(wildcards.fraction),
            "seed": int(wildcards.seed),
            "metric": "spearman_correlation",
            "score": result
        }])

        # Save results
        result_df.write_parquet(output[0])


rule compute_odds_ratio:
    input:
        "results/subsamples/frac_{fraction}_seed_{seed}.parquet",
    output:
        "results/metrics/odds_ratio/{model}_{fraction}_{seed}.parquet",
    run:

        # Load only required columns for odds ratio (label and score)
        model_name = wildcards.model
        score_col = model_name
        data = pl.read_parquet(input[0], columns=["label", score_col])

        # Compute odds ratio for different thresholds
        thresholds = config["or_thresholds"]
        results = []

        for threshold in thresholds:
            result = compute_odds_ratio(data, score_col, threshold)

            results.append({
                "model": model_name,
                "fraction": float(wildcards.fraction),
                "seed": int(wildcards.seed),
                "metric": f"odds_ratio_at_{threshold}",
                "score": result
            })

        result_df = pl.DataFrame(results)

        # Save results
        result_df.write_parquet(output[0])


rule compute_auroc:
    input:
        "results/subsamples/frac_{fraction}_seed_{seed}.parquet",
    output:
        "results/metrics/auroc/{model}_{fraction}_{seed}.parquet",
    run:

        # Load only required columns for AUROC (label and score)
        model_name = wildcards.model
        score_col = model_name
        data = pl.read_parquet(input[0], columns=["label", score_col])

        # Compute AUROC
        auroc = compute_auroc(data, score_col)

        # Create result DataFrame
        result_df = pl.DataFrame([{
            "model": model_name,
            "fraction": float(wildcards.fraction),
            "seed": int(wildcards.seed),
            "metric": "auroc",
            "score": auroc
        }])

        # Save results
        result_df.write_parquet(output[0])


rule compute_auprc:
    input:
        "results/subsamples/frac_{fraction}_seed_{seed}.parquet",
    output:
        "results/metrics/auprc/{model}_{fraction}_{seed}.parquet",
    run:

        # Load only required columns for AUPRC (label and score)
        model_name = wildcards.model
        score_col = model_name
        data = pl.read_parquet(input[0], columns=["label", score_col])

        # Compute AUPRC
        auprc = compute_auprc(data, score_col)

        # Create result DataFrame
        result_df = pl.DataFrame([{
            "model": model_name,
            "fraction": float(wildcards.fraction),
            "seed": int(wildcards.seed),
            "metric": "auprc",
            "score": auprc
        }])

        # Save results
        result_df.write_parquet(output[0])


rule aggregate_all_metrics:
    input:
        expand("results/metrics/{metric}/{model}_{fraction}_{seed}.parquet",
               metric=config["metrics_to_compute"],
               model=config["pcad1_models"],
               fraction=config["subsample_fractions"],
               seed=config["subsample_seeds"])
    output:
        "results/aggregated_metrics.parquet",
    run:
        # Load and combine all metric files
        all_metrics = []
        for file in input:
            df = pl.read_parquet(file)
            all_metrics.append(df)

        # Combine all metrics (now all have same column structure)
        combined_metrics = pl.concat(all_metrics)

        # Save aggregated results
        combined_metrics.write_parquet(output[0])


rule plot_aggregated_metrics:
    input:
        "results/aggregated_metrics.parquet",
    output:
        "results/metrics_plot.pdf",
    run:
        # Load aggregated metrics
        df = pl.read_parquet(input[0]).to_pandas()

        # Set up the plot style
        plt.rcParams['figure.figsize'] = (15, 10)

        # Create custom palette from config
        custom_palette = [config["model_colors"][model] for model in config["pcad1_models"]]

        # Create relplot with log scale for x-axis
        g = sns.relplot(
            data=df,
            x='fraction',
            y='score',
            col='metric',
            hue='model',
            errorbar='sd',
            kind='line',
            marker='o',
            markersize=8,
            linewidth=2,
            col_wrap=2,
            height=3,
            aspect=1.0,
            palette=custom_palette,
            facet_kws=dict(sharey=False),
        )

        # Set log scale for x-axis and clean up subtitles
        for ax in g.axes.flat:
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)

        # Clean up subtitles and customize x-axis ticks
        g.set_titles(col_template="{col_name}")
        g.set(xlim=(0.005, 1.5))

        # Add title
        g.fig.suptitle('PCAD1 Model Performance Metrics\nAcross Subsample Fractions\n(sd across seeds)',
                      y=1.02, fontsize=16, fontweight='bold')

        # Save plot
        plt.savefig(output[0], bbox_inches='tight')
        plt.close()

        print(f"Saved metrics plot to {output[0]}")
