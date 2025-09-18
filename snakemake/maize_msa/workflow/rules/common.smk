from Bio import Phylo, SeqIO
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr


CHROMS = config["chroms"]


def contains_target(clade, target):
    """Return True if `target` is the name of this clade or in any descendant."""
    if clade.name == target:
        return True
    return any(contains_target(c, target) for c in clade.clades)


def reorder_clades(clade, target):
    """Sort subclades so those containing `target` come first, then recurse."""
    # False < True, so not-containing-target will sort after containing-target
    clade.clades.sort(key=lambda c: not contains_target(c, target))
    for sub in clade.clades:
        reorder_clades(sub, target)


def load_fasta(path, subset_chroms=None):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as handle:
        return pd.Series(
            {
                rec.id: str(rec.seq)
                for rec in SeqIO.parse(handle, "fasta")
                if subset_chroms is None or rec.id in subset_chroms
            }
        )