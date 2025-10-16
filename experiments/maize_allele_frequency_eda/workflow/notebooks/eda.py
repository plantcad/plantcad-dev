import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from scipy.stats import fisher_exact, pearsonr, spearmanr
    import seaborn as sns
    from tqdm import tqdm

    return fisher_exact, np, pearsonr, pl, plt, sns, spearmanr, tqdm


@app.cell
def _(fisher_exact, main_palette, model_renaming, pl, plot_dir, plt, sns):
    def get_odds_ratio(df, threshold_ns):
        rows = []
        negative_set = df.filter(~pl.col("label")).sort("score", descending=True)
        for n in threshold_ns:
            threshold = negative_set[n]["score"]
            group_counts = (
                # TODO: rethink with ties in mind
                df.group_by(["label", pl.col("score") >= threshold])
                .len()
                .sort(["label", "score"])["len"]
                .to_numpy()
                .reshape((2, 2))
            )
            odds_ratio, p_value = fisher_exact(group_counts, alternative="greater")
            rows.append([n, odds_ratio, p_value])
        return pl.DataFrame(rows, schema=["n", "Odds ratio", "p_value"], orient="row")

    def format_number(num):
        """
        Converts a number into a more readable format, using K for thousands, M for millions, etc.
        Args:
        - num: The number to format.

        Returns:
        - A formatted string representing the number.
        """
        if num >= 1e9:
            return f"{num / 1e9:.1f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.1f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.1f}K"
        else:
            return str(num)

    def barplot(
        df_pl,
        metric,
        title,
        groupby="Consequence",
        width=2,
        height=2,
        nrows=1,
        ncols=1,
        save_path=None,
        wspace=None,
        hspace=None,
        x=None,
        y=None,
        palette=main_palette,
    ):
        df = df_pl.to_pandas()
        df.Model = df.Model.replace(model_renaming)
        if groupby not in df.columns:
            df[groupby] = "all"
        groups = df[groupby].unique()
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=False,
            sharey=False,
            figsize=(width * ncols, height * nrows),
            squeeze=False,
            gridspec_kw={"wspace": wspace, "hspace": hspace},
        )

        for group, ax in zip(groups, axes.flat):
            df_g = df[df[groupby] == group].sort_values(metric, ascending=False)
            if metric not in ["Pearson", "Spearman"]:
                n_pos, n_neg = df_g.n_pos.iloc[0], df_g.n_neg.iloc[0]
            else:
                n = df_g.n.iloc[0]

            if metric == "AUROC":
                baseline = 0.5
            elif metric == "AUPRC":
                baseline = n_pos / (n_pos + n_neg)
            elif metric == "Odds ratio":
                baseline = 1
            elif metric in ["Pearson", "Spearman"]:
                baseline = 0

            g = sns.barplot(
                data=df_g,
                y="Model",
                x=metric,
                hue="Model",
                legend=False,
                palette=palette,
                ax=ax,
            )
            sns.despine()
            if metric not in ["Pearson", "Spearman"]:
                sample_size = f"n={format_number(n_pos)} vs. {format_number(n_neg)}"
            else:
                sample_size = f"n={format_number(n)}"
            subtitle = f"{group}\n{sample_size}" if len(groups) > 1 else sample_size
            g.set_title(subtitle, fontsize=10)
            g.set(xlim=baseline, ylabel="")

            for bar, model in zip(g.patches, df_g.Model):
                if metric == "Odds ratio":
                    text = f"{bar.get_width():.1f}"
                    if df_g[df_g.Model == model].p_value.iloc[0] >= 0.05:
                        text = text + " (NS)"
                else:
                    text = f"{bar.get_width():.3f}"

                g.text(
                    max(
                        bar.get_width(), baseline
                    ),  # X position, here at the end of the bar
                    bar.get_y()
                    + bar.get_height() / 2,  # Y position, in the middle of the bar
                    text,  # Text to be displayed, formatted to 3 decimal places
                    va="center",  # Vertical alignment
                )

        plt.suptitle(title, x=x, y=y, fontsize=11)
        if save_path is not None:
            plt.savefig(plot_dir + save_path, bbox_inches="tight")

        return g

    return barplot, get_odds_ratio


@app.cell
def _(np):
    models = [
        "PlantCAD",
        # "PCAD1-l20",
        # "PCAD1-l24",
        # "PCAD1-l28",
        "phastCons",
        "phyloP",
        "MSA_empirical_LLR",
        "GPN-Star-v1",
        "GPN-Star-v2",
        "GPN-Star-v3",
    ]

    flip_sign_models = [
        "PCAD1-l20",
        "PCAD1-l24",
        "PCAD1-l28",
        "PCAD1-l32",
        "GPN-Star-v1",
        "GPN-Star-v2",
        "GPN-Star-v3",
    ]

    model_renaming = {
        "PlantCAD": "PCAD1-l32",
        "MSA_empirical_LLR": "MSA-LLR",
        "GPN-Star-v1": "GPN-Star (thresh=0.05)",
        "GPN-Star-v2": "GPN-Star (thres=0.2)",
        "GPN-Star-v3": "GPN-Star (thresh=0)",
    }

    main_palette = {model_renaming.get(m, m): f"C{i}" for i, m in enumerate(models)}

    alt_palette = {
        "PCAD1-l20": "#6baed6",  # Light blue (20 layers)
        "PCAD1-l24": "#2171b5",  # Medium blue (24 layers)
        "PCAD1-l28": "#08519c",  # Medium-dark blue (28 layers)
        "PCAD1-l32": "#08306b",  # Dark blue (32 layers)
    }

    quantiles = np.logspace(-3, 0, 20)

    min_n = 100

    # ns = [30, 60, 90]

    consequence_bundles = {
        "splice-region": [
            "splice_polypyrimidine_tract_variant",
            "splice_region_variant",
            "splice_donor_region_variant",
            "splice_donor_variant",
            "splice_donor_5th_base_variant",
            "splice_acceptor_variant",
        ],
        "start-or-stop": {
            "stop_gained",
            "stop_lost",
            "start_lost",
        },
    }

    cs_renaming = {
        "5_prime_UTR": "5' UTR",
        "3_prime_UTR": "3' UTR",
        "upstream_gene": "upstream-of-gene",
        "downstream_gene": "downstream-of-gene",
        "non_coding_transcript_exon": "ncRNA",
    }
    return (
        alt_palette,
        consequence_bundles,
        cs_renaming,
        flip_sign_models,
        main_palette,
        min_n,
        model_renaming,
        models,
        quantiles,
    )


@app.cell
def _(consequence_bundles, cs_renaming, flip_sign_models, models, pl):
    def load_V():
        V = pl.read_parquet("../../results/variants.annotated.parquet")
        for model in models:
            score = pl.read_parquet(f"../../results/predictions/{model}.parquet")[
                "score"
            ]
            if model in flip_sign_models:
                score = -score
            V = V.with_columns(score.alias(model))
        return V

    V = load_V()
    print(len(V))
    V = V.drop_nulls(subset=models)
    print(len(V))
    V = V.filter(~pl.col("is_repeat"))
    print(len(V))
    V = V.sample(
        fraction=1, shuffle=True, seed=42
    )  # to simplify unbiased handling of ties

    for new_c, old_cs in consequence_bundles.items():
        V = V.with_columns(
            pl.when(pl.col("consequence").is_in(old_cs))
            .then(pl.lit(new_c))
            .otherwise(pl.col("consequence"))
            .alias("consequence")
        )

    V = V.with_columns(
        pl.col("consequence")
        .str.replace("_variant", "", literal=True)
        .replace(cs_renaming)
    )
    V
    return (V,)


@app.cell
def _(V):
    consequences = ["all"] + V["consequence"].value_counts().sort(
        "count", descending=True
    )["consequence"].to_list()
    consequences = [
        c for c in consequences if c != "stop_retained"
    ]  # very small proportion and unclear significance
    consequences
    return (consequences,)


@app.cell
def _(V, consequences, models, pl, quantiles, tqdm):
    # """
    res = []
    for consequence in tqdm(consequences):
        V2 = V if consequence == "all" else V.filter(consequence=consequence)
        for model in models:
            V3 = V2.sort(model, descending=True, maintain_order=True)
            for q in quantiles:
                V4 = V3.head(int(q * len(V3)))
                res.append([consequence, model, q, len(V4), V4["AF"].mean()])
    res = pl.DataFrame(
        res, ["consequence", "model", "q", "n", "Mean AF"], orient="row"
    ).with_columns(pl.col("consequence").str.replace("_variant", ""))
    res
    # """;
    return (res,)


@app.cell
def _(main_palette, min_n, model_renaming, pl, res, sns):
    # """
    sns.relplot(
        data=res.filter(pl.col("n") >= min_n).with_columns(
            pl.col("model").replace(model_renaming)
        ),
        x="q",
        y="Mean AF",
        hue="model",
        col="consequence",
        kind="line",
        # marker="o",
        col_wrap=4,
        height=2.0,
        facet_kws=dict(sharey=False),
        palette=main_palette,
    ).set_titles(col_template="{col_name}").set(xscale="log")
    # """;
    return


@app.cell
def _(main_palette, min_n, model_renaming, models3, pl, res, sns):
    # """
    sns.relplot(
        data=res.filter(
            pl.col("n") >= min_n, pl.col("model").is_in(models3)
        ).with_columns(pl.col("model").replace(model_renaming)),
        x="q",
        y="Mean AF",
        hue="model",
        col="consequence",
        kind="line",
        # marker="o",
        col_wrap=4,
        height=2.5,
        facet_kws=dict(sharey=False),
        palette=main_palette,
    ).set_titles(col_template="{col_name}").set(xscale="log")
    # """;
    return


@app.cell
def _(main_palette, min_n, model_renaming, models3, pl, res, sns):
    sns.relplot(
        data=res.filter(
            pl.col("n") >= min_n, pl.col("model").is_in(models3)
        ).with_columns(pl.col("model").replace(model_renaming)),
        x="q",
        y="Mean AF",
        hue="model",
        col="consequence",
        kind="line",
        # marker="o",
        col_wrap=4,
        height=2.5,
        facet_kws=dict(sharey=False),
        palette=main_palette,
    ).set_titles(col_template="{col_name}")
    return


@app.cell
def _(alt_palette, min_n, model_renaming, models2, pl, res, sns):
    sns.relplot(
        data=res.filter(
            pl.col("n") >= min_n, pl.col("model").is_in(models2)
        ).with_columns(pl.col("model").replace(model_renaming)),
        x="q",
        y="Mean AF",
        hue="model",
        col="consequence",
        kind="line",
        # marker="o",
        col_wrap=4,
        height=2.5,
        facet_kws=dict(sharey=False),
        palette=alt_palette,
    ).set_titles(col_template="{col_name}").set(xscale="log")
    return


@app.cell
def _(V):
    V["AC"].min()
    return


@app.cell
def _(V, pl):
    # """
    V5 = V.with_columns(
        pl.when(pl.col("AC") == 4)
        .then(True)
        .when(pl.col("AF") > 20 / 100)
        .then(False)
        .otherwise(None)
        .alias("label")
    ).drop_nulls(subset="label")
    V5["label"].value_counts()
    # """
    return (V5,)


@app.cell
def _(V5, consequences, get_odds_ratio, models, pl, tqdm):
    # """
    ns = [30, 90]

    res2 = []
    for c in tqdm(consequences):
        V_c = V5 if c == "all" else V5.filter(consequence=c)
        n_pos, n_neg = V_c["label"].sum(), (~V_c["label"]).sum()
        for m in models:
            odds_ratio = get_odds_ratio(
                V_c.select(["label", pl.col(m).alias("score")]), ns
            ).with_columns(
                Consequence=pl.lit(c),
                Model=pl.lit(m),
                n_pos=pl.lit(n_pos),
                n_neg=pl.lit(n_neg),
            )
            res2.append(odds_ratio)
    res2 = pl.concat(res2)
    res2
    # """
    return (res2,)


@app.cell
def _(barplot, pl, res2):
    # """
    barplot(
        res2.filter(
            pl.col("n") == 30,
            pl.col("p_value") < 0.05,
            pl.col("Consequence") == "all",
        ),
        "Odds ratio",
        "Maize AC=4 vs. AF > 20%",
        y=1.2,
    )
    # """
    return


@app.cell
def _(barplot, pl, res2):
    # """;
    barplot(
        res2.filter(
            pl.col("n") == 30,
            pl.col("p_value") < 0.05,
        ),
        "Odds ratio",
        "Maize AC=4 vs. AF > 20%",
        nrows=4,
        ncols=3,
        hspace=0.8,
        wspace=1.2,
        width=3,
        height=2.3,
        y=0.95,
    )
    # """
    return


@app.cell
def _(barplot, pl, res2):
    # """;
    barplot(
        res2.filter(
            pl.col("n") == 90,
            pl.col("p_value") < 0.05,
            pl.col("Consequence") == "all",
        ),
        "Odds ratio",
        "Maize AC=4 vs. AF > 20%",
        y=1.2,
    )
    # """
    return


@app.cell
def _(barplot, pl, res2):
    # """;
    barplot(
        res2.filter(
            pl.col("n") == 90,
            pl.col("p_value") < 0.05,
        ),
        "Odds ratio",
        "Maize AC=4 vs. AF > 20%",
        nrows=4,
        ncols=3,
        hspace=0.8,
        wspace=1.2,
        width=3.5,
        height=2.3,
        y=0.95,
    )
    # """
    return


@app.cell
def _(V, consequences, models, pearsonr, pl, spearmanr, tqdm):
    res3 = []
    for c2 in tqdm(consequences):
        V6 = V if c2 == "all" else V.filter(consequence=c2)
        for m2 in models:
            res3.append(
                [
                    c2,
                    m2,
                    len(V6),
                    pearsonr(V6["AF"], -V6[m2])[0],
                    spearmanr(V6["AF"], -V6[m2])[0],
                ]
            )
    res3 = pl.DataFrame(
        res3, ["Consequence", "Model", "n", "Pearson", "Spearman"], orient="row"
    ).with_columns(pl.col("Consequence").str.replace("_variant", ""))
    res3
    return (res3,)


@app.cell
def _(barplot, res3):
    barplot(
        res3,
        "Pearson",
        "Correlation with Maize AF",
        nrows=4,
        ncols=3,
        hspace=0.8,
        wspace=1.2,
        width=3.5,
        height=2.3,
        y=0.95,
    )
    return


@app.cell
def _(barplot, res3):
    barplot(
        res3,
        "Spearman",
        "Correlation with Maize AF",
        nrows=4,
        ncols=3,
        hspace=0.8,
        wspace=1.2,
        width=3.5,
        height=2.3,
        y=0.95,
    )
    return


@app.cell
def _():
    models2 = [
        "PCAD1-l20",
        "PCAD1-l24",
        "PCAD1-l28",
        "PlantCAD",
    ]

    models3 = [
        "PlantCAD",
        "phastCons",
        "phyloP",
        "MSA_empirical_LLR",
        # "GPN-Star-v1",
    ]
    return models2, models3


@app.cell
def _(alt_palette, barplot, models2, pl, res3):
    barplot(
        res3.filter(pl.col("Model").is_in(models2)),
        "Pearson",
        "Correlation with Maize AF",
        nrows=4,
        ncols=3,
        hspace=0.8,
        wspace=1.2,
        width=3.5,
        height=2.3,
        y=0.95,
        palette=alt_palette,
    )
    return


@app.cell
def _(barplot, models3, pl, res3):
    barplot(
        res3.filter(pl.col("Model").is_in(models3)),
        "Pearson",
        "Correlation with Maize AF",
        nrows=4,
        ncols=3,
        hspace=0.8,
        wspace=1.2,
        width=3.5,
        height=2.3,
        y=0.95,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
