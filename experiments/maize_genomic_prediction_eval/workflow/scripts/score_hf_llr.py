#!/usr/bin/env python3
"""Score variants using HuggingFace masked language model LLR."""

import argparse
import pandas as pd
import polars as pl
from biofoundation.data import Genome
from biofoundation.inference import run_llr_mlm
from biofoundation.model.adapters.hf import HFMaskedLM, HFTokenizer
from datasets import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Score variants using HF MLM LLR")
    parser.add_argument(
        "--variants",
        required=True,
        help="Path to input variants parquet file",
    )
    parser.add_argument(
        "--genome",
        required=True,
        help="Path to genome FASTA file",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        required=True,
        help="Context window size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Per device evaluation batch size",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output parquet file",
    )

    args = parser.parse_args()

    # Load variants
    V = pd.read_parquet(args.variants, columns=["chrom", "pos", "ref", "alt"])
    dataset = Dataset.from_pandas(V, preserve_index=False)

    # Load tokenizer and model
    tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(args.model_path))
    model = HFMaskedLM(
        AutoModelForMaskedLM.from_pretrained(args.model_path, trust_remote_code=True)
    )
    genome = Genome(args.genome)

    # Configure inference
    inference_kwargs = dict(
        torch_compile=True,
        bf16_full_eval=True,
        dataloader_num_workers=args.n_workers,
        per_device_eval_batch_size=args.batch_size,
        remove_unused_columns=False,
    )

    # Run LLR scoring
    scores = run_llr_mlm(
        model,
        tokenizer,
        dataset,
        genome,
        window_size=args.context_size,
        data_transform_on_the_fly=True,
        inference_kwargs=inference_kwargs,
    )

    # Write output
    result = pl.DataFrame({"score": scores})
    result.write_parquet(args.output)


if __name__ == "__main__":
    main()
