# =============================================================================
# METRICS COMPUTATION RULES
# =============================================================================
# Refactored to compute all metrics for all models in a single job per subsample

rule compute_all_metrics:
    input:
        "results/subsamples/frac_{fraction}_seed_{seed}.parquet",
    output:
        "results/metrics/all_metrics_frac_{fraction}_seed_{seed}.parquet",
    run:
        # Load subsample with all required columns
        models = config["pcad1_models"]
        required_cols = ["AF", "label"] + models
        data = pl.read_parquet(input[0], columns=required_cols)

        # Get configuration
        quantiles = config["analysis_quantiles"]
        or_thresholds = config["or_thresholds"]
        fraction = float(wildcards.fraction)
        seed = int(wildcards.seed)

        # Collect all metric results
        all_results = []

        # Iterate over all models
        for model in models:
            score_col = model

            # 1. Mean AF at quantiles
            mean_af_results = compute_mean_af_at_quantile(
                data.select(["AF", score_col]),
                score_col,
                quantiles
            )
            for row in mean_af_results.iter_rows(named=True):
                all_results.append({
                    "model": model,
                    "fraction": fraction,
                    "seed": seed,
                    "metric": f"mean_af_at_q{row['quantile']}",
                    "score": row["mean_af"]
                })

            # 2. Pearson correlation
            pearson = compute_pearson_correlation(
                data.select(["AF", score_col]),
                score_col
            )
            all_results.append({
                "model": model,
                "fraction": fraction,
                "seed": seed,
                "metric": "pearson_correlation",
                "score": pearson
            })

            # 3. Spearman correlation
            spearman = compute_spearman_correlation(
                data.select(["AF", score_col]),
                score_col
            )
            all_results.append({
                "model": model,
                "fraction": fraction,
                "seed": seed,
                "metric": "spearman_correlation",
                "score": spearman
            })

            # 4. Odds ratio at different thresholds
            for threshold in or_thresholds:
                or_score = compute_odds_ratio(
                    data.select(["label", score_col]),
                    score_col,
                    threshold
                )
                all_results.append({
                    "model": model,
                    "fraction": fraction,
                    "seed": seed,
                    "metric": f"odds_ratio_at_{threshold}",
                    "score": or_score
                })

            # 5. AUROC
            auroc = compute_auroc(
                data.select(["label", score_col]),
                score_col
            )
            all_results.append({
                "model": model,
                "fraction": fraction,
                "seed": seed,
                "metric": "auroc",
                "score": auroc
            })

            # 6. AUPRC
            auprc = compute_auprc(
                data.select(["label", score_col]),
                score_col
            )
            all_results.append({
                "model": model,
                "fraction": fraction,
                "seed": seed,
                "metric": "auprc",
                "score": auprc
            })

        # Create and save combined results DataFrame
        result_df = pl.DataFrame(all_results)
        result_df.write_parquet(output[0])

        print(f"Computed all metrics for fraction={fraction}, seed={seed}: {len(all_results)} results")


rule aggregate_all_metrics:
    input:
        expand("results/metrics/all_metrics_frac_{fraction}_seed_{seed}.parquet",
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

        # Combine all metrics
        combined_metrics = pl.concat(all_metrics)

        # Save aggregated results
        combined_metrics.write_parquet(output[0])

        print(f"Aggregated {len(combined_metrics)} metric results from {len(input)} files")


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
        g.fig.suptitle('PCAD1 performance metrics (sd)',
                      y=1.02, fontsize=16, fontweight='bold')

        # Save plot
        plt.savefig(output[0], bbox_inches='tight')
        plt.close()

        print(f"Saved metrics plot to {output[0]}")
