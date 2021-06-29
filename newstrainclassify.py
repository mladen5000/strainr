#!/usr/bin/env python
import argparse
import multiprocessing as mp
import functools
import pathlib
import pickle
import random
import sys
import time
from mimetypes import guess_type
import gzip
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from Bio import SeqIO


def get_args():
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="input file",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--db",
        # required=True,
        help="Database file",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--procs",
        type=int,
        default=4,
        help="Number of cores to use (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=pathlib.Path,
        required=False,
        help="Output folder",
        default="strainr_out",
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="""
        Selection mode for diambiguation
        """,
        choices=["random", "max", "multinomial", "dirichlet"],
        type=str,
        default="random",
    )
    parser.add_argument("-a", "--thresh", help="", type=float, default=0.001)
    parser.add_argument(
        "--save-raw-hits",
        action="store_true",
        required=False,
        help="""
                        Save the intermediate results as a csv file containing each read's strain information.
                        """,
    )
    # Examples
    # parser.add_argument('--source_file', type=open)
    # parser.add_argument('--dest_file', type=argparse.FileType('w', encoding='latin-1'))
    # parser.add_argument('--datapath', type=pathlib.Path)
    return parser


def load_database(dbfile):
    """Load the database in dataframe format"""
    print("Loading Database...", file=sys.stderr)
    df = pd.read_pickle(dbfile)
    print(f"Database of {len(df.columns)} strains loaded")
    return df


def df_to_dict(df):
    """Convert database to dict for lookup"""
    hit_arrays = list(df.to_numpy())
    strain_ids = list(df.columns)
    kmers = df.index.to_list()
    db = dict(zip(kmers, hit_arrays))
    print("Finished building dbmap")
    return strain_ids, db


def get_kmer_len(df):
    """Obtain k-mer length of the database"""
    kmerlen = len(df.index[0])
    assert all(df.index.str.len() == kmerlen)
    return kmerlen


def get_rc(kmer):
    rc_kmer = kmer.reverse_complement()
    return kmer if kmer < rc_kmer else rc_kmer


def count_kmers(seqrecord):
    """Main function to assign strain hits to reads"""
    max_index = len(seqrecord.seq) - kmerlen + 1
    matched_kmer_strains = []
    s = seqrecord.seq
    with memoryview(bytes(s)) as seqview:
        for i in range(max_index):
            returned_strains = db.get(seqview[i : i + kmerlen])
            if returned_strains is not None:
                matched_kmer_strains.append(returned_strains)
    return seqrecord, sum(matched_kmer_strains)


def classify():
    """Call multiprocessing library to lookup k-mers"""
    t0 = time.time()
    records = SeqIO.parse(_open(f1), "fastq")
    print("Beginning classification")
    with mp.Pool(processes=procs) as pool:
        results = list(
            pool.map(
                count_kmers,
                records,
            ),
        )
    print(f"Ending classification: {time.time() - t0}s")
    return results


def raw_to_dict(raw_classified):
    """
    Go from list of tuples (SeqRecord,hits)
    to dict {ReadID:hits}
    """
    return {
        read.id: hits for read, hits in raw_classified if isinstance(hits, np.ndarray)
    }


def separate_hits(hitcounts):
    """Return maps of reads with 1 (clear), multiple (ambiguous), or no signal"""
    clear_hits, ambig_hits = {}, {}
    none_hits = []
    for read, hits in hitcounts:
        max_ind = np.argwhere(hits == np.max(hits)).flatten()
        if len(max_ind) == 1:
            clear_hits[read.id] = hits
        elif len(max_ind) > 1 and sum(hits) > 0:
            ambig_hits[read.id] = hits
        else:
            none_hits.append(read.id)
    print(
        f"Clear:{len(clear_hits)}, Ambiguous: {len(ambig_hits)}, None:{len(none_hits)}"
    )
    return clear_hits, ambig_hits, none_hits




def prior_counter(clear_hits):
    """Aggregate values"""
    return Counter(clear_hits.values())


def _open(infile):
    """Handle unknown file for gzip and non-gzip alike"""
    encoding = guess_type(str(infile))[1]  # uses file extension
    _open = functools.partial(gzip.open, mode="rt") if encoding == "gzip" else open
    file_object = _open(
        infile,
    )
    return file_object


def counter_to_array(prior_counter, nstrains):
    """
    Aggregate signals from reads with singular
    maximums and return a vector of probabilities for each strain
    """
    prior_array = np.zeros(nstrains)
    total = sum(prior_counter.values())
    for k, v in prior_counter.items():
        prior_array[k] = v / total

    prior_array[prior_array == 0] = 1e-20
    return prior_array


def parallel_resolve(hits, prior, selection):
    """Main function called by helper for parallel disambiguation"""
    # Treshold at max
    belowmax = hits < np.max(hits)
    hits[belowmax] = 0
    mlehits = hits * prior  # Apply prior

    if selection == "random":  # Weighted assignment
        return random.choices(range(len(hits)), weights=mlehits, k=1).pop()

    elif selection == "max":  # Maximum Likelihood assignment
        return np.argmax(mlehits)

    elif selection == "dirichlet":  # Dirichlet assignment
        mlehits[mlehits == 0] = 1e-10
        return np.argmax(rng.dirichlet(mlehits, 1))

    elif selection == "multinomial":  # Multinomial
        mlehits[mlehits == 0] = 1e-10
        return np.argmax(rng.multinomial(1, mlehits / sum(mlehits)))

    else:
        raise ValueError("Must select a selection mode")


def parallel_resolve_helper(ambig_hits, prior, selection="multinomial"):
    """
    Assign a strain to reads with ambiguous k-mer signals by maximum likelihood.
    Currently 3 options: random, max, and dirichlet. (dirichlet is slow and performs similar to random)
    For all 3, threshold spectra to only include maxima, multiply w/ prior.
    """
    new_clear, new_ambig = {}, {}
    mapfunc = functools.partial(
        parallel_resolve,
        prior=prior,
        selection=selection,
    )

    with mp.Pool(processes=procs) as pool:
        for read, outhits in zip(
            ambig_hits.keys(),
            pool.map(mapfunc, ambig_hits.values()),
        ):
            new_clear[read] = outhits

    return new_clear, new_ambig


def disambiguate(ambig_hits, prior, selection="multinomial"):
    """
    Assign a strain to reads with ambiguous k-mer signals by maximum likelihood.
    Currently 3 options: random, max, and dirichlet. (dirichlet is slow and performs similar to random)
    For all 3, threshold spectra to only include maxima, multiply w/ prior.
    """

    rng = np.random.default_rng()
    new_clear, new_ambig = {}, {}

    for read, hits in ambig_hits.items():
        # Treshold at max
        belowmax = hits < np.max(hits)
        hits[belowmax] = 0

        # Apply prior
        mlehits = hits * prior

        # Weighted assignment
        if selection == "random":
            select_random = random.choices(range(len(hits)), weights=mlehits, k=1).pop()
            new_clear[read] = select_random

        # Maximum Likelihood assignment
        elif selection == "max":
            select_max = np.argwhere(mlehits == np.max(mlehits)).flatten()
            if len(select_max) == 1:
                new_clear[read] = int(select_max)

        # Dirichlet assignment
        elif selection == "dirichlet":
            mlehits[mlehits == 0] = 1e-10
            select_dirichlet = rng.dirichlet(mlehits, 1)
            new_clear[read] = np.argmax(select_dirichlet)

        elif selection == "multinomial":
            mlehits[mlehits == 0] = 1e-10
            select_multi = rng.multinomial(1, mlehits / sum(mlehits))
            new_clear[read] = np.argmax(select_multi)

        else:
            raise ValueError("Must select a selection mode")

    assert len(ambig_hits) == len(new_clear) + len(new_ambig)
    return new_clear, new_ambig


def collect_reads(clear_hits, updated_hits, na_hits):
    """Assign the NA string to na and join all 3 dicts"""
    na = {k: "NA" for k in na_hits}
    all_dict = clear_hits | updated_hits | na
    print(len(all_dict), len(clear_hits), len(updated_hits), len(na_hits))
    # assert len(all_dict) == len(clear_hits) + len(updated_hits) + len(na_hits)
    return all_dict


def resolve_clear_hits(clear_hits):
    """
    INPUT: Reads whose arrays contain singular maxima - clear hits
    OUTPUT: The index/strain corresponding to the maximum value
    Replace numpy array with index"""
    return {k: int(np.argmax(v)) for k, v in clear_hits.items()}


def resolve_ambig_hits(ambig_hits):
    """Replace numpy array with index"""
    return {k: int(v[0]) for k, v in ambig_hits.items()}


def build_na_dict(na_hits):
    return {k: None for k in na_hits}


def normalize_counter(acounter):
    total = sum(acounter.values())
    return Counter({k: v / total for k, v in acounter.items()})


def threshold_by_relab(norm_counter_all, threshold=0.02):
    """
    Given a percentage cutoff [threshold], remove strains
    which do not meet the criteria and recalculate relab
    """
    thresh_results = Counter(
        {k: v for k, v in norm_counter_all.items() if v > threshold}
    )
    if thresh_results["NA"]:
        thresh_results.pop("NA")
    return normalize_counter(thresh_results)


def save_intermediate(intermediate_results, strains):
    """
    Take a dict of readid:hits and convert to a dataframe
    then save the output
    """
    df = pd.DataFrame.from_dict(intermediate_results, orient="index", columns=strains)
    df.to_csv("intermed.csv")
    return


def print_relab(acounter, nstrains=10, prefix=""):
    """Pretty print for counter"""
    print(f"\n{prefix}")
    for idx, ab in acounter.most_common(n=nstrains):
        try:
            print(strains[idx], "\t", round(ab, 5))
        except:
            print(idx, "\t", round(ab, 5))


def output_abundance(strain_names, idx_relab, outfile):
    """
    For each strain in the database,
    grab the relab gathered from classification, else print 0.0
    """
    full_relab = defaultdict(float)
    print(idx_relab)

    for idx, name in enumerate(strain_names):
        full_relab[name] = idx_relab[idx]
    if idx_relab["NA"]:
        full_relab["NA"] = idx_relab["NA"]

    with outfile.open(mode="w") as fh:
        for strain, ab in sorted(full_relab.items()):
            fh.write(f"{strain}\t{round(ab, 5)}\n")
    return


def main():
    """
    Execute main loop
    1. Classify
    2. Generate Initial Abundance estimate
    3. Reclassify ambiguous
    4. Collect reads for abundance
    5. Normalize, threshold, re-normalize
    6. Output
    """
    # Classify
    results_raw = classify()  # get list of (SecRecord,nparray)
    clear_hits, ambig_hits, na_hits = separate_hits(results_raw)  # parse into 3 dicts

    # Clear and Prior building
    assigned_clear = resolve_clear_hits(clear_hits)  # dict values are now single index
    cprior = prior_counter(assigned_clear)
    print_relab(normalize_counter(cprior), prefix="Prior Estimate")
    prior = counter_to_array(cprior, nstrains)  # counter to vector

    # Assign ambiguous
    new_clear, new_ambig = parallel_resolve_helper(ambig_hits, prior, mle_mode)
    total_hits = collect_reads(assigned_clear, new_clear, na_hits)

    # Build abundance and output
    final_hits = Counter(total_hits.values())
    print_relab(final_hits, prefix="Overall hits",)
    output_abundance(strains, final_hits, (outdir / "count_abundance.tsv"))

    final_relab = normalize_counter(final_hits)
    print_relab(final_relab, prefix="Overall abundance",)
    output_abundance(strains, final_relab, (outdir / "sample_abundance.tsv"))

    final_threshab = threshold_by_relab(final_relab, threshold=params["thresh"])
    print_relab(final_threshab, prefix="Overall relative abundance",)
    output_abundance(strains, final_threshab, (outdir / "intra_abundance.tsv"))
    return


if __name__ == "__main__":
    p = pathlib.Path().cwd()
    rng = np.random.default_rng()
    params = get_args().parse_args()

    # Parameters
    params = vars(params)
    procs = params["procs"]
    mle_mode = params["mode"]
    outdir = params["out"]
    if outdir:
        try:
            outdir.mkdir()
        except FileExistsError:
            print("Output directory already exists. Remove outdir to continue")

    df = load_database(params["db"])
    kmerlen = get_kmer_len(df)
    strains, db = df_to_dict(df)
    nstrains = len(strains)

    print(params)
    f1_files = params["input"]
    for file in f1_files:
        f1 = file
        main()

    """
    1. Make an output directory
    ~ sample cardinality
    2. abundance.tsv
    3. abundance2.tsv
    ~ tp/tn/stuff
    4. Save pd-intermediate
    5. Get strains -> reads mapping
    6. Write bins
    
    """
    # output = create_newdir(output)  # from str to path
    # reads_mcs = sample_cardinality(output, reads_mcs)  # hist.txt

    # # p_out = Path(output)
    # # abundance_file = "abundance.tsv" #+ p_out.stem + ".tsv"

    # abundance_file = output / "abundance.tsv"
    # output_abundance(strain_list, relab_count, abundance_file)

    # abundance_file2 = output / "abundance2.tsv"
    # output_abundance(strain_list, intra_relab, abundance_file2)
    # Save intermediate results
    # initial_results = raw_to_dict(results_raw)
    # save_intermediate(initial_results, strains)

# def output_abundance(strain_list, dt_abund, filepath):
#     """This will print the abundance profile based on all of the strains"""
#     strain_dict = show_all_strains(dt_abund, strain_list)
#     # with #open(filepath, "w") as f:

#     with filepath.open("w") as f:
#         for k, v in sorted(strain_dict.items(), key=lambda kv: kv[1]):
#             print(k, "\t", round(v, 4), file=f)
#     return
# def show_all_strains(relab, strain_ll):
#     strain_dict = defaultdict(float)
#     for i in strain_ll:
#         strain_dict[i] = relab[i]
#     return {k: v for k, v in sorted(strain_dict.items())}
#     # , key=lambda val: val[1], reverse=True) }
