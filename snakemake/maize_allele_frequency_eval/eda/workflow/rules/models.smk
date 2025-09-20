rule pcad1_precomputed_score:
    input:
        "results/variants.parquet",
    output:
        "results/predictions/PlantCAD.parquet",
    run:
        V = pl.read_parquet(input[0], columns=COORDINATES)
        score = (
            pl.read_parquet(
                config["raw_data_path"],
                columns=["CHR", "POS", "REF", "ALT", "PlantCAD"]
            )
            .rename({
                "CHR": "chrom", "POS": "pos", "REF": "ref", "ALT": "alt",
                "PlantCAD": "score"
            })
            .with_columns(
                pl.col("chrom").cast(str),
                -pl.col("score"),  # let's do higher -> more functional
            )
        )
        V = V.join(score, on=COORDINATES, how="left")[["score"]]
        V.write_parquet(output[0])


rule msa_empirical_LLR:
    input:
        "results/variants.parquet",
        config["msa_path"],
    output:
        "results/predictions/MSA_empirical_LLR.parquet",
    run:
        V = pd.read_parquet(input[0], columns=COORDINATES)
        MSA = zarr.open(input[1], mode="r")
        V["score"] = V.progress_apply(
            lambda v: run_vep_MSA_empirical_LLR(MSA, v.chrom, v.pos, v.ref, v.alt),
            axis=1
        )
        V = V[["score"]]
        V.to_parquet(output[0], index=False)


#rule conservation_score:
#    input:
#        "results/variants.annot.parquet",
#        lambda wildcards: config["conservation_models"][wildcards.model],
#    output:
#        "results/predictions/{model}.parquet",
#    wildcard_constraints:
#        model="|".join(config["conservation_models"]),
#    run:
#        import pyBigWig
#        from tqdm import tqdm
#        tqdm.pandas()
#
#        V = pd.read_parquet(input[0], columns=["chrom", "pos"])
#        bw = pyBigWig.open(input[1])
#        V["score"] = V.progress_apply(
#            lambda v: bw.values(v.chrom, v.pos - 1, v.pos)[0], axis=1,
#        )
#        V = V[["score"]]
#        V.to_parquet(output[0], index=False)
