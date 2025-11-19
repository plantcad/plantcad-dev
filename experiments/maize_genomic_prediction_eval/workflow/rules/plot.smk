rule merge_metrics:
    input:
        expand("results/metrics/{model}.parquet", model=all_models)
    output:
        "results/metrics/merged.parquet"
    run:
        res = pl.concat([
            pl.read_parquet(path).with_columns(pl.lit(model).alias("model"))
            for path, model in zip(input, all_models)
        ])
        res.write_parquet(output[0])


rule plot_metrics:
    input:
        "results/metrics/merged.parquet"
    output:
        r_plot="results/plots/r_metrics.png",
        mse_plot="results/plots/mse_metrics.png"
    run:
        def plot_metric(data, metric_column, output_path):
            g = sns.catplot(
                data=data,
                kind="point",
                y="model",
                x=metric_column,
                col="trait",
                row="validation",
                sharex=False,
                margin_titles=True,
                linestyle="none",
            )
            g.savefig(output_path)
            plt.close()

        df = pl.read_parquet(input[0]).to_pandas()
        trait_map = {
            "DTS": "days to silking",
            "PH": "plant height",
            "GY_adjusted": "grain yield adjusted for DTS",
        }
        df["trait"] = df["trait"].map(trait_map)
        plot_metric(df, "r", output.r_plot)
        plot_metric(df, "MSE", output.mse_plot)
