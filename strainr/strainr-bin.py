#!/usr/bin/env python

import multiprocessing as mp
import pathlib

import pandas as pd
from Bio import SeqIO

import utils

# from utils import _open


def generate_table(intermediate_results, strains):
    """Use the k-mer hits from classify in order to build a binning frame."""
    df = pd.DataFrame.from_dict(dict(intermediate_results)).T
    df.columns = strains
    return df


def top_bins(final_values, strains):
    """Use a dict of reads : strain list index and choose the top ones (sorted list)"""
    out_series = pd.Series(dict(final_values))
    top_strain_indices = list(out_series.value_counts().index)
    top_strain_names = [strains[i] for i in top_strain_indices if i != "NA"]
    return top_strain_names


def bin_helper(top_strain_names, bin_table, f1, f2, outdir, nbins=2):
    """strain_dict is the strain_id : set of reads"""
    # Make a directory for output
    bin_dir = outdir / "bins"
    bin_dir.mkdir(exist_ok=True, parents=True)

    strains_to_bin = top_strain_names[:nbins]
    print(f"Generating sequence files for the top {nbins} strains.")
    # most_reads_frame = (df != 0).sum().sort_values(ascending=False)

    procs, saved_strains = [], []
    for strain_id in strains_to_bin:
        if strain_id == "NA":
            continue

        # Keep track of binned strains
        saved_strains.append(strain_id)
        print(f"Binning reads for {strain_id}...")
        p = mp.Process(
            target=bin_single,
            args=(strain_id, bin_table, f1, f2, bin_dir),
        )
        p.start()
        procs.append(p)

    [p.join() for p in procs]

    # Return the set of strains and the list of processes
    return set(saved_strains), procs


def bin_single(strain, bin_table, forward_file, reverse_file, bin_dir):
    """Given a strain, use hit info to extract reads."""
    strain_accession = strain.split()[-1]  # Get GCF from name
    paired_files = [forward_file, reverse_file]

    for fidx, input_file in enumerate(paired_files):  # (0,R1), (1,R2)
        writefile_name = f"bin.{strain_accession}_R{fidx+1}.fastq"
        writefile = bin_dir / writefile_name

        if fidx == 0:
            reads = set(bin_table[bin_table[strain] > 0].index)
        else:
            # reads = set(bin_table[bin_table[strain] > 0].index.str.replace('/1','/2'))
            reads = set(
                bin_table[bin_table[strain] > 0].index.str.replace("1:N", "2:N")
            )  # Illumina extension

        with utils._open(input_file) as original, writefile.open("w") as newfasta:
            records = (
                r for r in SeqIO.parse(original, "fastq") if r.description in reads
            )  # TODO: r.description vs r.id vs fastqiterator
            count = SeqIO.write(records, newfasta, "fastq")

        print(f"Saved {count} records from {input_file} to {writefile}")

    return


def main_bin(intermediate_results, strains, final_values, f1, outdir):
    """Will become to main binning code"""
    bin_table = generate_table(intermediate_results, strains)
    top_strain_names = top_bins(final_values, strains)
    f1 = pathlib.Path(f1)  # TODO
    f2 = pathlib.Path(str(f1).replace("_R1", "_R2"))  # TODO
    bin_helper(top_strain_names, bin_table, f1, f2, outdir, nbins=2)
    return


# TODO
binflag = None
# if binflag:
#     main_bin(results_raw, strains, total_hits, fasta, outfile)
