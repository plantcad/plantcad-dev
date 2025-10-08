# =============================================================================
# METRICS COMPUTATION RULES
# =============================================================================
# Refactored to compute all metrics for all models and all seeds in a single job per sample size

rule compute_all_metrics_for_n:
    input:
        "results/complete_dataset.parquet",
    output:
        "results/metrics/all_metrics_n_{n}.parquet",
    run:
        # Get configuration
        models = config["pcad1_models"]
        required_cols = ["AF", "label"] + models
        quantiles = config["analysis_quantiles"]
        or_quantiles = config["or_quantiles"]
        n = int(wildcards.n)

        # Load complete dataset once with only required columns
        complete_data = pl.read_parquet(input[0], columns=required_cols)

        # Collect all metric results
        all_results = []

        # Generate seeds from count and iterate with progress bar
        n_seeds = config["n_subsample_seeds"]
        seeds = range(1, n_seeds + 1)

        for seed in tqdm(seeds, desc=f"Seeds (n={n})"):
            # Sample data for this seed
            data = complete_data.sample(n=n, seed=seed)

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
                        "n": n,
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
                    "n": n,
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
                    "n": n,
                    "seed": seed,
                    "metric": "spearman_correlation",
                    "score": spearman
                })

                # 4. Odds ratio at different quantiles
                for quantile in or_quantiles:
                    or_score = compute_odds_ratio(
                        data.select(["label", score_col]),
                        score_col,
                        quantile
                    )
                    all_results.append({
                        "model": model,
                        "n": n,
                        "seed": seed,
                        "metric": f"odds_ratio_at_q{quantile}",
                        "score": or_score
                    })

                # 5. AUROC
                auroc = compute_auroc(
                    data.select(["label", score_col]),
                    score_col
                )
                all_results.append({
                    "model": model,
                    "n": n,
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
                    "n": n,
                    "seed": seed,
                    "metric": "auprc",
                    "score": auprc
                })

        # Create and save combined results DataFrame
        result_df = pl.DataFrame(all_results)
        result_df.write_parquet(output[0])

        print(f"Computed all metrics for n={n}, all seeds: {len(all_results)} results")


rule aggregate_all_metrics:
    input:
        expand("results/metrics/all_metrics_n_{n}.parquet",
               n=config["subsample_n"])
    output:
        "results/aggregated_metrics.parquet",
    run:
        # Load and combine all metric files
        all_metrics = [pl.read_parquet(file) for file in input]

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

        # Create custom ordering for metrics based on config
        # Map config metric names to actual metric patterns in data
        metric_patterns = {
            'pearson_correlation': lambda m: m == 'pearson_correlation',
            'spearman_correlation': lambda m: m == 'spearman_correlation',
            'auroc': lambda m: m == 'auroc',
            'auprc': lambda m: m == 'auprc',
            'mean_af_quantile': lambda m: m.startswith('mean_af_at_q'),
            'odds_ratio': lambda m: m.startswith('odds_ratio_at_q'),
        }

        # Get the ordered list of metrics from config
        metrics_order_config = config['metrics_to_compute']

        # Build the ordered list of actual metric names from the data
        unique_metrics = df['metric'].unique()
        metrics_ordered = []
        for config_metric in metrics_order_config:
            if config_metric in metric_patterns:
                # Find all actual metrics matching this pattern
                matching = [m for m in unique_metrics if metric_patterns[config_metric](m)]
                metrics_ordered.extend(sorted(matching))  # Sort within category for consistency

        # Convert metric column to ordered categorical
        df['metric'] = pd.Categorical(df['metric'], categories=metrics_ordered, ordered=True)

        # Create relplot with log scale for x-axis
        g = sns.relplot(
            data=df,
            x='n',
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

        # Clean up subtitles
        g.set_titles(col_template="{col_name}")

        # Add title
        g.fig.suptitle('PCAD1 performance metrics by sample size (sd)',
                      y=1.02, fontsize=16, fontweight='bold')

        # Save plot
        plt.savefig(output[0], bbox_inches='tight')
        plt.close()

        print(f"Saved metrics plot to {output[0]}")
