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
        assert model_cfg["kind"] == "MLM", "Only MLM models are supported"
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
