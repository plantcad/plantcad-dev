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
from scipy.stats import pearsonr, spearmanr, fisher_exact
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

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


# =============================================================================
# METRIC CALCULATION FUNCTIONS
# =============================================================================

def filter_analysis_chromosomes(df, chromosomes=None):
    """Filter to specified chromosomes for analysis."""
    if chromosomes is None:
        chromosomes = config["analysis_chromosomes"]
    return df.filter(pl.col("chrom").is_in(chromosomes))


def compute_mean_af_at_quantile(variants_df, score_col, quantiles):
    """
    Compute mean AF for variants above given score quantiles.

    Parameters:
    -----------
    variants_df : pl.DataFrame
        Variants with AF and score columns
    score_col : str
        Name of the score column
    quantiles : list
        List of quantile thresholds (e.g., [0.01, 0.05, 0.1, 0.25])

    Returns:
    --------
    pl.DataFrame
        Results with quantile, n_variants, mean_af columns
    """
    results = []

    for q in quantiles:
        # Get threshold for this quantile
        threshold = variants_df.select(score_col).quantile(1 - q, interpolation="higher").item()

        # Filter to variants above threshold
        top_variants = variants_df.filter(pl.col(score_col) >= threshold)

        # Calculate mean AF
        mean_af = top_variants.select("AF").mean().item()
        n_variants = len(top_variants)

        # Flip sign so higher values mean better performance (lower AF = more functional)
        results.append([q, n_variants, -mean_af])

    return pl.DataFrame(
        results,
        schema=["quantile", "n_variants", "mean_af"],
        orient="row"
    )


def compute_pearson_correlation(variants_df, score_col):
    """Compute Pearson correlation between AF and model scores (flipped for higher=better)."""
    # Convert to pandas for scipy
    df_pd = variants_df.select(["AF", score_col]).to_pandas()

    corr, _ = pearsonr(df_pd["AF"], df_pd[score_col])
    # Flip sign so higher correlation means better (more functional -> lower AF)
    return -corr


def compute_spearman_correlation(variants_df, score_col):
    """Compute Spearman correlation between AF and model scores (flipped for higher=better)."""
    # Convert to pandas for scipy
    df_pd = variants_df.select(["AF", score_col]).to_pandas()

    corr, _ = spearmanr(df_pd["AF"], df_pd[score_col])
    # Flip sign so higher correlation means better (more functional -> lower AF)
    return -corr


def compute_odds_ratio(variants_df, score_col, n_top_variants):
    """
    Compute odds ratio for rare vs common variants in the top n variants by score.

    Parameters:
    -----------
    variants_df : pl.DataFrame
        Variants with label and score columns (label column pre-computed)
    score_col : str
        Name of the score column
    n_top_variants : int
        Number of top variants to consider

    Returns:
    --------
    float
        Odds ratio from Fisher's exact test
    """
    # Filter to only rows with valid labels (drop null labels)
    df_with_labels = variants_df.filter(pl.col("label").is_not_null())

    # Sort by score (descending) and take top n
    top_variants = df_with_labels.sort(score_col, descending=True).head(n_top_variants)

    # Count rare vs common in top variants
    n_rare_top = len(top_variants.filter(pl.col("label") == True))
    n_common_top = len(top_variants.filter(pl.col("label") == False))

    # Count rare vs common in bottom variants (for comparison)
    bottom_variants = df_with_labels.sort(score_col, descending=True).tail(n_top_variants)
    n_rare_bottom = len(bottom_variants.filter(pl.col("label") == True))
    n_common_bottom = len(bottom_variants.filter(pl.col("label") == False))

    # Create contingency table
    contingency = [
        [n_rare_top, n_rare_bottom],
        [n_common_top, n_common_bottom]
    ]

    # Fisher's exact test
    odds_ratio, _ = fisher_exact(contingency, alternative="greater")

    return odds_ratio


def compute_auroc(variants_df, score_col):
    """
    Compute AUROC for rare vs common classification.

    Parameters:
    -----------
    variants_df : pl.DataFrame
        Variants with label and score columns (label column pre-computed)
    score_col : str
        Name of the score column

    Returns:
    --------
    float
        AUROC score
    """
    # Filter to only rows with valid labels (drop null labels)
    df_with_labels = variants_df.filter(pl.col("label").is_not_null())

    # Convert to pandas for sklearn
    df_pd = df_with_labels.select(["label", score_col]).to_pandas()

    try:
        auroc = roc_auc_score(df_pd["label"], df_pd[score_col])
        return auroc
    except ValueError:
        # This happens when only one class is present
        return None


def compute_auprc(variants_df, score_col):
    """
    Compute AUPRC for rare vs common classification.

    Parameters:
    -----------
    variants_df : pl.DataFrame
        Variants with label and score columns (label column pre-computed)
    score_col : str
        Name of the score column

    Returns:
    --------
    float
        AUPRC score
    """
    # Filter to only rows with valid labels (drop null labels)
    df_with_labels = variants_df.filter(pl.col("label").is_not_null())

    # Convert to pandas for sklearn
    df_pd = df_with_labels.select(["label", score_col]).to_pandas()

    try:
        auprc = average_precision_score(df_pd["label"], df_pd[score_col])
        return auprc
    except ValueError:
        # This happens when only one class is present
        return None
