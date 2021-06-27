#!/usr/bin/env python
import argparse
import multiprocessing as mp
import functools
import pathlib
import pickle
import random
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from Bio import SeqIO


def get_args():
    """
    # Examples
    parser.add_argument('--source_file', type=open)
    parser.add_argument('--dest_file', type=argparse.FileType('w', encoding='latin-1'))
    parser.add_argument('--datapath', type=pathlib.Path)
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "input",
    #     help="input file",
    #     type=str,
    # )
    parser.add_argument(
        "-j",
        "--reverse-pair",
        help="reverse reads",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--db",
        # required=True,
        help="Database file",
        type=int,
        default=31,
    )
    parser.add_argument(
        "-p",
        "--procs",
        type=int,
        default=1,
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
    return parser


# kmerset = {bytes(seqview[i : i + KMERLEN]) for i in range(max_index)}
def count_kmers(seqrecord):
    max_index = len(seqrecord.seq) - kmerlen + 1
    matched_kmer_strains = []
    with memoryview(bytes(seqrecord.seq)) as seqview:
        for i in range(max_index):
            returned_strains = db.get(seqview[i : i + kmerlen])
            if returned_strains is not None:
                matched_kmer_strains.append(returned_strains)
    return sum(matched_kmer_strains)


def load_database(dbfile):
    df = pd.read_pickle(dbfile)
    return df


def df_to_dict(df):
    hit_arrays = list(df.to_numpy())
    strain_ids = list(df.columns)
    kmers = df.index.to_list()
    db = dict(zip(kmers, hit_arrays))
    return strain_ids, db


def get_kmer_len(df):
    kmerlen = len(df.index[0])
    assert all(df.index.str.len() == kmerlen)
    return kmerlen


def classify():
    # Classify reads
    print("Begining classification")
    t0 = time.time()
    records = list(SeqIO.parse(f1, "fastq"))
    with mp.Pool(processes=procs) as pool:
        results = list(
            zip(
                records,
                pool.map(
                    count_kmers,
                    records,
                ),
            )
        )
    print(f"Ending classification: {time.time() - t0}s")
    return results


def separate_hits(hitcounts):
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


def print_relab(acounter, nstrains=10, prefix=""):
    """Pretty print for counter"""
    print(f"{prefix}")
    for k, v in acounter.most_common(n=nstrains):
        print(k, "\t", round(v, 5))


def prior_counter(clear_hits):
    return Counter(clear_hits.values())


def counter_to_array(prior_counter, nstrains):
    """
    Aggregate signals from reads with singular
    maximums and return a vector of probabilities for each strain
    """
    prior_array = np.zeros(nstrains)
    total = sum(prior_counter.values())
    for k, v in prior_counter.items():
        prior_array[k] = v / total
    return prior_array


def parallel_resolve(hits, prior, selection):
    # Treshold at max
    belowmax = hits < np.max(hits)
    hits[belowmax] = 0
    mlehits = hits * prior  # Apply prior

    if selection == "random":  # Weighted assignment
        return random.choices(range(len(hits)), weights=mlehits, k=1).pop()

    elif selection == "max":  # Maximum Likelihood assignment
        return int(np.argwhere(mlehits == np.max(mlehits)).flatten())

    elif selection == "dirichlet":  # Dirichlet assignment
        mlehits[mlehits == 0] = 1e-10
        return np.argmax(rng.dirichlet(mlehits, 1))

    elif selection == "multinomial":  # Multinomial
        mlehits[mlehits == 0] = 1e-10
        return np.argmax(rng.multinomial(1, mlehits / sum(mlehits)))

    else:
        raise ValueError("Must select a selection mode")




def parallel_resolve_helper(ambig_hits, prior, selection="multinomial"):
    new_clear, new_ambig = {}, {}
    mapfunc = functools.partial(parallel_resolve, prior=prior,selection=selection, )

    with mp.Pool(processes=procs) as pool:
        for read, outhits in zip(
            ambig_hits.keys(),
            pool.map(mapfunc, ambig_hits.values()),
        ):
            new_clear[read] = outhits

    return new_clear, new_ambig


def disambiguate(ambig_hits, prior, selection="multinomial"):
    """
    Description:
        Assign a strain to reads with ambiguous k-mer signals by maximum likelihood.
        Currently 3 options: random, max, and dirichlet. (dirichlet is slow and performs similar to random)
        For all 3, threshold spectra to only include maxima, multiply w/ prior.

    Selection mode:
        For random, randomly select strain from given maxima according to the intersection of
        prior and k-mer spectra.

        For max, assign strain from intersection of read's k-mer spectra and prior with the maximum prob.

        For dirichlet, draw from largest probability in dirichlet of length nstrains.

    Raises:
        ValueError: Incorrect selection mode

    Returns:
        dict: new_clear: Reads with assigned strains.
        dict: new_ambig: Reads with unassigned strains.

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
    na = {k: "NA" for k in na_hits}
    all_dict = clear_hits | updated_hits | na
    assert len(all_dict) == len(clear_hits) + len(updated_hits) + len(na_hits)
    return all_dict


def normalize(counter_abundance):
    total = sum(counter_abundance.values())
    return {read: hits / total for read, hits in counter_abundance.items()}


def threshold(norm_counter, threshval):
    threshed_counter = {k: v for k, v in norm_counter.items() if v > threshval}
    return normalize(threshed_counter)


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
    return normalize_counter(thresh_results)


if __name__ == "__main__":

    # Parameters
    params = vars(get_args().parse_args())
    kmerlen = 31
    f1 = "test_R1.fastq"
    # df = load_database("new_method.sdb")
    df = load_database("hflu_complete_genbank.db")
    # df = load_database("database.db")
    procs = 8

    # Initialize
    rng = np.random.default_rng()
    p = pathlib.Path().cwd()
    kmerlen = get_kmer_len(df)
    strains, db = df_to_dict(df)
    nstrains = len(strains)


    # Classify
    results_raw = classify()  # get list of (SecRecord,nparray)
    clear_hits, ambig_hits, na_hits = separate_hits(results_raw)  # parse into 3 dicts

    # Clear and Prior building
    assigned_clear = resolve_clear_hits(clear_hits)  # dict values are now single index
    cprior = prior_counter(assigned_clear)
    prior = counter_to_array(cprior, nstrains)  # counter to vector
    prior[prior == 0] = 1e-10

    # Assign ambiguous
    # new_clear, new_ambig = disambiguate(ambig_hits, prior)
    new_clear, new_ambig = parallel_resolve_helper(ambig_hits, prior, 'multinomial')
    if len(new_ambig) > 0:
        pass  # TODO

    total_hits = collect_reads(assigned_clear, new_clear, na_hits)

    # Build counter of hits
    final_hits = Counter(total_hits.values())
    print_relab(final_hits, prefix="Overall hits")

    final_relab = normalize_counter(final_hits)
    print_relab(final_relab, prefix="Overall abundance")

    final_threshab = threshold_by_relab(final_relab, threshold=0.02)
    print_relab(final_threshab, prefix="Overall relative abundance")
