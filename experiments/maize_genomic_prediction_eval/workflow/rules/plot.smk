rule summarize_metrics:
    input:
        "results/metrics/{model}.parquet"
    output:
        "results/summarized_metrics/{model}.parquet"
    wildcard_constraints:
        model="|".join(all_models)
    run:
        (
            pl.read_parquet(input[0])
            .group_by("validation", "trait")
            .agg([
                pl.col("r").mean().alias("mean_r"),
                pl.col("MSE").mean().alias("mean_MSE")
            ])
            .write_parquet(output[0])
        )


rule merge_metrics:
    input:
        expand("results/summarized_metrics/{model}.parquet", model=all_models)
    output:
        "results/summarized_metrics/merged.parquet"
    run:
        res = pl.concat([
            pl.read_parquet(path).with_columns(pl.lit(model).alias("model"))
            for path, model in zip(input, all_models)
        ])
        res.write_parquet(output[0])


rule plot_metrics:
    input:
        "results/summarized_metrics/merged.parquet"
    output:
        r_plot="results/plots/r_metrics.png",
        mse_plot="results/plots/mse_metrics.png"
    run:
        def plot_metric(data, metric_column, output_path):
            g = sns.catplot(
                data=data,
                kind="bar",
                y="model",
                x=metric_column,
                col="trait",
                row="validation",
                sharex=False,
                margin_titles=True,
            )
            g.savefig(output_path)
            plt.close()

        df = pl.read_parquet(input[0]).to_pandas()
        plot_metric(df, "mean_r", output.r_plot)
        plot_metric(df, "mean_MSE", output.mse_plot)
