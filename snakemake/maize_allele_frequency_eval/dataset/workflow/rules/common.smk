from biofoundation.data import Genome
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import yaml


COORDINATES = ["chrom", "pos", "ref", "alt"]
NUCLEOTIDES = list("ACGT")

SPLIT_CHROMS = config["split_chroms"]
SPLITS = list(SPLIT_CHROMS.keys())

CONFIGS = ["full"] + [
    f"{max_n}_{consequence}"
    for max_n in config["subsampling"]["max_n"].keys()
    for consequence in (config["consequences"] + ["all_consequences"])
]
