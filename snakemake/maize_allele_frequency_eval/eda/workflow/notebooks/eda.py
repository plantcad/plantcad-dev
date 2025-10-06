import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from scipy.stats import fisher_exact
    import seaborn as sns
    from tqdm import tqdm

    return fisher_exact, np, pl, plt, sns, tqdm


@app.cell
def _(fisher_exact, palette, pl, plot_dir, plt, sns):
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
    ):
        df = df_pl.to_pandas()
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
            n_pos, n_neg = df_g.n_pos.iloc[0], df_g.n_neg.iloc[0]

            if metric == "AUROC":
                baseline = 0.5
            elif metric == "AUPRC":
                baseline = n_pos / (n_pos + n_neg)
            elif metric == "Odds ratio":
                baseline = 1

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
            sample_size = f"n={format_number(n_pos)} vs. {format_number(n_neg)}"
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
        "PCAD1-l20",
        "PCAD1-l24",
        "phastCons",
        "phyloP",
        "MSA_empirical_LLR",
        "GPN-Star-v1",
    ]

    flip_sign_models = [
        "PCAD1-l20",
        "PCAD1-l24",
        "GPN-Star-v1",
    ]

    palette = {m: f"C{i}" for i, m in enumerate(models)}

    quantiles = np.logspace(-4, 0, 10)

    ns = [30, 60, 90]

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
        consequence_bundles,
        cs_renaming,
        flip_sign_models,
        models,
        ns,
        palette,
        quantiles,
    )


@app.cell
def _(consequence_bundles, cs_renaming, flip_sign_models, models, pl):
    def load_V():
        V = pl.read_parquet("../../results/variants.parquet")
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
def _(pl, res, sns):
    # """
    min_n = 100

    g = sns.relplot(
        data=res.filter(pl.col("n") >= min_n),
        x="q",
        y="Mean AF",
        hue="model",
        col="consequence",
        kind="line",
        marker="o",
        col_wrap=4,
        height=2.0,
        facet_kws=dict(sharey=False),
    )
    g.set_titles(col_template="{col_name}")
    g.set(xscale="log")
    g
    # """;
    return


@app.cell
def _(V):
    V["AC"].min()
    return


@app.cell
def _(V, pl):
    V5 = V.with_columns(
        pl.when(pl.col("AC") == 4)
        .then(True)
        .when(pl.col("AF") > 20 / 100)
        .then(False)
        .otherwise(None)
        .alias("label")
    ).drop_nulls(subset="label")
    V5["label"].value_counts()
    return (V5,)


@app.cell
def _(V5, consequences, get_odds_ratio, models, ns, pl, tqdm):
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
    return (res2,)


@app.cell
def _(barplot, pl, res2):
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
    return


@app.cell
def _(barplot, pl, res2):
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
    return


@app.cell
def _(barplot, pl, res2):
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
        width=3,
        height=2.3,
        y=0.95,
    )
    return


@app.cell
def _(V, sns):
    sns.histplot(data=V, x="GPN-Star-v1")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
