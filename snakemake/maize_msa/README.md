# Processing maize MSA

Processes whole-genome alignment in MAF format into an array format (Zarr) and upload to Hugging Face.

Input data: https://huggingface.co/datasets/plantcad/andropogoneae_alignment_raw_data

Output data: https://huggingface.co/datasets/plantcad/andropogoneae_msa

## Setup

Requires mamba (or conda).

```bash
mamba env create -f env.yaml
mamba activate msa-maize-processing
```

Also requires `mafDuplicateFilter` from https://github.com/ComparativeGenomicsToolkit/mafTools.
(this fork contains the crucial option `-k` or `--keep-first`)
One convoluted way to install it:
```bash
# First install another version of mafTools
# https://github.com/dentearl/mafTools/tree/master
mamba env create -f workflow/envs/mafTools.yaml
mamba activate mafTools
# then install the version we want
cd path/to
git clone https://github.com/ComparativeGenomicsToolkit/mafTools.git
cd mafTools
make
export PATH=/path/to/mafTools/bin:$PATH
```

## Running

```bash
snakemake --cores all
```