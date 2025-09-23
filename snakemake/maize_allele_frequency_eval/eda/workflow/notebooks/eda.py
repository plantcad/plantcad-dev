import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import polars as pl
    import seaborn as sns
    from tqdm import tqdm

    return np, pl, sns, tqdm


@app.cell
def _(pl):
    models = [
        "PlantCAD",
        "MSA_empirical_LLR",
    ]

    def load_V():
        V = pl.read_parquet("../../results/variants.parquet")
        for model in models:
            score = pl.read_parquet(f"../../results/predictions/{model}.parquet")[
                "score"
            ]
            V = V.with_columns(score.alias(model))
        return V

    V = load_V()
    print(len(V))
    V = V.drop_nulls(subset=models)
    print(len(V))
    V = V.sample(fraction=1, shuffle=True, seed=42)
    V
    return V, models


@app.cell
def _(V, sns):
    sns.histplot(data=V, x="AF", bins=100)
    return


@app.cell
def _(V, sns):
    sns.histplot(data=V, x="MAF", bins=100)
    return


@app.cell
def _(V):
    V["consequence"].value_counts().sort("count", descending=True)
    return


@app.cell
def _(V, pl):
    V.group_by("consequence").agg(pl.len(), pl.mean("AF")).sort("AF")
    return


@app.cell
def _(V, sns):
    sns.histplot(data=V, x="PlantCAD", bins=100)
    return


@app.cell
def _(V, sns):
    sns.histplot(data=V, x="MSA_empirical_LLR", bins=100)
    return


@app.cell
def _(V, np):
    consequences = ["all"] + V["consequence"].value_counts().sort(
        "count", descending=True
    )["consequence"].to_list()
    # quantiles = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    quantiles = np.logspace(-4, 0, 10)
    return consequences, quantiles


@app.cell
def _(V, consequences, models, pl, quantiles, tqdm):
    res = []
    for consequence in tqdm(consequences):
        V2 = V if consequence == "all" else V.filter(consequence=consequence)
        for model in models:
            for q in quantiles:
                V3 = V2.sort(model, descending=True, maintain_order=True).head(
                    int(q * len(V2))
                )
                res.append([consequence, model, q, len(V3), V3["AF"].mean()])
    res = pl.DataFrame(
        res, ["consequence", "model", "q", "n", "Mean AF"], orient="row"
    ).with_columns(pl.col("consequence").str.replace("_variant", ""))
    res
    return (res,)


@app.cell
def _(pl, res, sns):
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
        height=3,
        facet_kws=dict(sharey=False),
    )
    g.set_titles(col_template="{col_name}")
    g.set(xscale="log")
    g
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
