rule download_maf:
    output:
        temp("results/maf_raw/{chrom}.maf.gz"),
    shell:
        "wget -O {output} https://huggingface.co/datasets/plantcad/andropogoneae_alignment_raw_data/resolve/main/andropogoneae_msa/panand_chr{wildcards.chrom}.maf.gz"


rule download_genome:
    output:
        temp("results/genome/{chrom}.fa"),
    shell:
        """
        wget -O - https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/fasta/zea_mays/dna/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.chromosome.{wildcards.chrom}.fa.gz \
        | gunzip -c > {output}
        """


rule extract_maf:
    input:
        "results/maf_raw/{chrom}.maf.gz",
    output:
        temp("results/maf_raw/{chrom}.maf"),
    shell:
        "gunzip -c {input} > {output}"


rule maf_filter_duplicates:
    input:
        "results/maf_raw/{chrom}.maf",
    output:
        temp("results/maf/{chrom}.maf"),
    shell:
        "mafDuplicateFilter --maf {input} -k > {output}"


# single-threaded but memory-intensive - consider not running too many in parallel
rule maf2fasta:
    input:
        "results/genome/{chrom}.fa",
        "results/maf/{chrom}.maf",
    output:
        temp("results/maf_fasta/{chrom}.fa"),
    threads:
        workflow.cores
    shell:
        "maf2fasta {input} fasta > {output}"


rule extract_species:
    input:
        "config/tree_topology.nh",
    output:
        "results/dataset/species.txt",
    run:
        tree = Phylo.read(input[0], "newick")
        reorder_clades(tree.root, config["target"])
        species = pd.DataFrame(index=[node.name for node in tree.get_terminals()])
        assert species.index.values[0] == config["target"]
        species.to_csv(output[0], header=False, columns=[])


rule make_msa_chrom:
    input:
        "results/maf_fasta/{chrom}.fa",
        "results/dataset/species.txt",
    output:
        temp("results/msa_chrom/{chrom}.npy"),
    threads:
        workflow.cores
    run:
        MSA = load_fasta(input[0])
        species = pd.read_csv(input[1], header=None).values.ravel()

        # need to handle the case where the MSA lacks a species
        # (didn't happen in human, mouse, drosophila but happened in chicken,
        # in some chromosomes)
        missing_species = set(species) - set(MSA.index)
        L = len(MSA.iloc[0])
        missing_seq = "-" * L
        for s in missing_species:
            MSA.loc[s] = missing_seq

        MSA = MSA[species]
        MSA = np.vstack(MSA.apply(
            lambda seq: np.frombuffer(seq.upper().encode("ascii"), dtype="S1")
        ))
        # let's only keep non-gaps in reference
        MSA = MSA[:, MSA[0]!=b'-']
        MSA = MSA.T
        vocab = np.frombuffer("ACGT-".encode('ascii'), dtype="S1")
        # decision: consider all "N" and similar as "-"
        # might not be the best, some aligners have a distinction
        # between N, or unaligned, and gap
        MSA[~np.isin(MSA, vocab)] = b"-"
        np.save(output[0], MSA)


rule merge_msa_chroms:
    input:
        lambda wildcards: expand("results/msa_chrom/{chrom}.npy", chrom=CHROMS),
    output:
        directory("results/msa.zarr"),
    threads: workflow.cores
    run:
        z = zarr.open_group(output[0], mode='w')
        for chrom, path in tqdm(zip(CHROMS, input), total=len(CHROMS)):
            data = np.load(path)
            z.create_dataset(
                chrom, data=data, chunks=(config["chunk_size"], data.shape[1])
            )


rule compress_msa:
    input:
        "results/msa.zarr",
    output:
        "results/dataset/msa.tar.gz",
    threads:
        workflow.cores
    shell:
        "tar cf - {input} | pigz -p {threads} > {output}"


rule species_metadata:
    input:
        "results/dataset/species.txt",
    output:
        "results/dataset/species_metadata.tsv",
    run:
        df = pd.read_csv(
            "hf://datasets/plantcad/andropogoneae_alignment_raw_data/keyFile.tsv",
            sep='\t',
            index_col=0
        )
        species_order = pd.read_csv(input[0], header=None).values.ravel()
        df = df.loc[species_order]
        df.to_csv(output[0], sep='\t')


rule cp_readme:
    input:
        "config/hf_readme.md",
    output:
        "results/dataset/README.md",
    shell:
        "cp {input} {output}"
