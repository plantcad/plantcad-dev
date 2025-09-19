rule pcad1_precomputed_score:
    input:
        "results/variants.annot.parquet",
    output:
        "results/predictions/PlantCAD.parquet",
    run:
        V = pd.read_parquet(input[0], columns=["PlantCAD"])
        V["score"] = -V[["PlantCAD"]]  # let's do higher -> more functional
        V = V[["score"]]
        V.to_parquet(output[0], index=False)


rule conservation_score:
    input:
        "results/variants.annot.parquet",
        lambda wildcards: config["conservation_models"][wildcards.model],
    output:
        "results/predictions/{model}.parquet",
    wildcard_constraints:
        model="|".join(config["conservation_models"]),
    run:
        import pyBigWig
        from tqdm import tqdm
        tqdm.pandas()

        V = pd.read_parquet(input[0], columns=["chrom", "pos"])
        bw = pyBigWig.open(input[1])
        V["score"] = V.progress_apply(
            lambda v: bw.values(v.chrom, v.pos - 1, v.pos)[0], axis=1,
        )
        V = V[["score"]]
        V.to_parquet(output[0], index=False)