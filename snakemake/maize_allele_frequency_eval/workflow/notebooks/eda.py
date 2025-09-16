import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import seaborn as sns
    from tqdm import tqdm
    return pl, sns, tqdm


@app.cell
def _(pl):
    V = pl.read_parquet("../../results/variants.annot.parquet")
    V
    return (V,)


@app.cell
def _(V):
    V["consequence"].value_counts().sort("count", descending=True)
    return


@app.cell
def _(V, pl):
    V.group_by("consequence").agg(pl.len(), pl.mean("MAF")).sort("MAF")
    return


@app.cell
def _(V, sns):
    sns.histplot(data=V, x="PlantCAD", bins=100)
    return


@app.cell
def _(V):
    consequences = ["all"] + V["consequence"].value_counts().sort("count", descending=True)["consequence"].to_list()
    quantiles = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    return consequences, quantiles


@app.cell
def _(V, consequences, pl, quantiles, tqdm):
    model = "PlantCAD"

    # TODO: to handle ties fairly should first shuffle V
    # and then sort by "model" and get variant q * len(V)
    # (important for phyloP and phastCons)
    # also: make all models be higher->more functional and use >

    res = []
    for consequence in tqdm(consequences):
        V2 = V if consequence == "all" else V.filter(consequence=consequence)
        for q in quantiles:
            V3 = V2.filter(pl.col(model) < pl.quantile(model, q))
            res.append([consequence, q, len(V3), V3["MAF"].mean()])
    res = (
        pl.DataFrame(res, ["consequence", "q", "n", "Mean MAF"], orient="row")
        .with_columns(pl.col("consequence").str.replace("_variant", ""))
    )
    res
    return (res,)


@app.cell
def _(pl, res, sns):
    min_n = 100

    g = sns.relplot(
        data=res.filter(pl.col("n") >= min_n),
        x="q",
        y="Mean MAF",
        col="consequence",
        kind="line",
        col_wrap=4,
        height=3,
        facet_kws=dict(sharey=False),
        markers=True,
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
