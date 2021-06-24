# new
import sys
import time
import pickle
import pathlib
import argparse
import random

import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import Counter
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
    with mp.Pool() as pool:
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


def get_mode(hitcounts):
    clear_hits, ambig_hits = {}, {}
    none_hits = []
    for read, hits in hitcounts:
        max_ind = np.argwhere(hits == np.max(hits)).flatten()
        if len(max_ind) == 1:
            clear_hits[read.id] = hits
        elif len(max_ind) > 1:
            ambig_hits[read.id] = hits
        else:
            none_hits.append(read.id)
    print(
        f"Clear:{len(clear_hits)}, Ambiguous: {len(ambig_hits)}, None:{len(none_hits)}"
    )
    return clear_hits, ambig_hits, none_hits


def print_relab(posterior_prob, nstrains=10):
    """Displays Top 10 relative abundance strains"""
    print(f"\nRelative Abundance Estimation")
    for k, v in posterior_prob.most_common(n=nstrains):
        print(k, "\t", round(v, 5))
    print(f"\n")


def normalize_dist(countdict, num_strains):
    """
    Convert hits per strain into % of each strain.
    Also convert the dict of strain_index: value to a numpy array
    where the value is placed in the index
    example input: {2: 0.24, 4: 0.51, 10: 0.25}
    example output: [0,0,0.24,0,0.51,0,0,0,0,0,0.25,0,0,0]
    """
    prior_array = np.zeros(num_strains)
    total = sum(countdict.values())
    for k, v in countdict.items():
        prior_array[k] = v / total

    # return {k:v/total for k,v in countdict.items()}
    return prior_array


def generate_likelihood(clear_hits):
    """
    Aggregate clear hits in order to get initial estimate of distribution
    """
    unique_hit_distribution = Counter([np.argmax(v) for v in clear_hits.values()])
    nstrains = len(list(clear_hits.values())[0])  # TODO
    prior = normalize_dist(unique_hit_distribution, nstrains)
    return prior


def resolve_ties(ambig_hits, prior):
    updated_ambig = {read: np.argmax(hits * prior) for read, hits in ambig_hits.items()}
    # print(updated_ambig)
    assert len(updated_ambig) == len(ambig_hits)
    return updated_ambig


def assess_resolve_ties():
    """Not implemented, but use this in case of multiple rounds of resolution"""
    # TODO
    new_clear, new_ambig = {}, {}
    for read, hits in ambig_hits.items():
        mlehits = hits * prior
        max_ind = np.argwhere(mlehits == np.max(mlehits)).flatten()
        if len(max_ind) == 1:
            new_clear[read] = mlehits
        elif len(max_ind) > 1:
            new_ambig[read] = mlehits
    print(f"Ties Resolved:{len(new_clear)}, Still ambiguous: {len(new_ambig)}")
    return new_clear, new_ambig


def joined_abundance(clear_hits, new_ambig, na_hits, ambig_hits):
    all_dict = {}
    all_dict = clear_hits | new_ambig | na_hits
    assert len(all_dict) == len(clear_hits) + len(new_ambig) + len(na_hits)


if __name__ == "__main__":

    p = pathlib.Path().cwd()
    params = vars(get_args().parse_args())
    kmerlen = 31
    f1 = "short_R1.fastq"
    t0 = time.time()
    # df = load_database("new_method.sdb")
    # df = load_database("hflu_complete_genbank.db")
    df = load_database("database.db")
    t1 = time.time()
    kmerlen = get_kmer_len(df)
    strains, db = df_to_dict(df)
    t2 = time.time()
    print(f"Load DB: {t1-t0}, Make DF: {t2-t1}")
    results_raw = classify()
    t3 = time.time()
    clear_hits, ambig_hits, na_hits = get_mode(results_raw)
    t4 = time.time()
    print(f"classifying : {t3-t2}, separating : {t4-t3}")
    prior = generate_likelihood(clear_hits)
    new_ambig = resolve_ties(ambig_hits, prior)
    # print(ambig_hits)
