import multiprocessing as mp
import numpy as np
import pandas as pd
import polars as pl
import pyBigWig
from biofoundation.data import Genome
from biofoundation.inference import run_llr_mlm
from biofoundation.model import HFMaskedLM
from datasets import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import zarr

tqdm.pandas()


COORDINATES = ["chrom", "pos", "ref", "alt"]
NUCLEOTIDES = list("ACGT")


def run_vep_MSA_empirical_LLR(MSA, chrom, pos, ref, alt, pseudocounts=1):
    msa = np.char.upper(MSA[chrom][pos - 1].view("S1"))
    assert msa[0] == ref.encode("ascii"), f"{ref=} does not match {msa[0]=}"
    msa = msa[1:]  # exclude target species
    ref_count = (msa == ref.encode("ascii")).sum() + pseudocounts
    alt_count = (msa == alt.encode("ascii")).sum() + pseudocounts
    ref_prob = ref_count / (ref_count + alt_count)
    alt_prob = alt_count / (ref_count + alt_count)
    return np.log(alt_prob) - np.log(ref_prob)


def _run_vep_MSA_empirical_LLR_batch(i, chrom, pos, ref, alt, msa):
    return run_vep_MSA_empirical_LLR(msa, chrom[i], pos[i], ref[i], alt[i])


rule download_genome:
    output:
        "results/genome.fa.gz",
    shell:
        "wget -O {output} {config[genome_url]}"
