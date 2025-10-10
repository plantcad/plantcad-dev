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
    threads: workflow.cores
    run:
        V = pd.read_parquet(input[0], columns=COORDINATES)

        chrom = V["chrom"].values
        pos = V["pos"].values
        ref = V["ref"].values
        alt = V["alt"].values
        msa = zarr.open(input[1], mode="r")

        with mp.Pool(processes=workflow.cores) as pool:
            vep_batch = pool.starmap(
                _run_vep_MSA_empirical_LLR_batch,
                [
                    (i, chrom, pos, ref, alt, msa)
                    for i in tqdm(range(len(chrom)))
                ],
            )

        V["score"] = vep_batch
        V["score"] = -V["score"]  # let's do higher -> more functional
        V[["score"]].to_parquet(output[0], index=False)


rule conservation_score:
    input:
        "results/variants.parquet",
        lambda wildcards: config["conservation_models"][wildcards.model],
    output:
        "results/predictions/{model}.parquet",
    wildcard_constraints:
        model="|".join(config["conservation_models"]),
    run:
        V = pd.read_parquet(input[0], columns=["chrom", "pos"])
        bw = pyBigWig.open(input[1])
        V["score"] = V.progress_apply(
            lambda v: bw.values(v.chrom, v.pos - 1, v.pos)[0], axis=1,
        )
        V = V[["score"]]
        V.to_parquet(output[0], index=False)


rule score_hf:
    input:
        "results/variants.parquet",
        "results/genome.fa.gz",
    output:
        "results/predictions/{model}.parquet",
    wildcard_constraints:
        model="|".join(config["hf_models"]),
    threads: workflow.cores
    run:
        model_cfg = config["hf_models"][wildcards.model]

        V = pd.read_parquet(input[0], columns=COORDINATES)
        dataset = Dataset.from_pandas(V, preserve_index=False)

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"])
        model = HFMaskedLM(AutoModelForMaskedLM.from_pretrained(
            model_cfg["path"], trust_remote_code=True,
        ))
        genome = Genome(input[1])

        inference_kwargs = dict(
            torch_compile=True,
            bf16_full_eval=True,
            dataloader_num_workers=threads,
            per_device_eval_batch_size=int(model_cfg["per_device_eval_batch_size"]),
            remove_unused_columns=False,
        )

        scores = run_llr_mlm(
            model,
            tokenizer,
            dataset,
            genome,
            window_size=int(model_cfg["context_size"]),
            data_transform_on_the_fly=True,
            inference_kwargs=inference_kwargs,
        )

        pd.DataFrame({"score": scores}).to_parquet(output[0], index=False)
