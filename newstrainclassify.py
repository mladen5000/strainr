# new
import sys
import time
import pickle
import pathlib
import argparse

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
    print("Begining maponly classification")
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


def generate_likelihood(clear_hits):
    fullhits = np.array(list(clear_hits.values()))
    print(fullhits)
    top_strains = np.argmax(fullhits, axis=0)
    print(top_strains)
    strain_counter = Counter(top_strains)
    print(strain_counter)
    for strain, num_occurences in strain_counter.items():
        percentage = num_occurences / len(top_strains)
        prior[strain] = percentage
    print(prior)
    return prior


# def resolve_ties(ambig_hits):
#     for k,v in ambig_hits.items():


if __name__ == "__main__":

    p = pathlib.Path().cwd()
    params = vars(get_args().parse_args())
    kmerlen = 31
    f1 = "test_R1.fastq"
    t0 = time.time()
    df = load_database("new_method.sdb")
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
    # for i,read in enumerate(SeqIO.parse("inputs/short_R1.fastq", "fastq")):
    #     print(i)
    #     kset = delayed(count_kmers)(read)
    #     res = delayed(get_hits)(kset)
    #     results.append(res)
    # results = delayed(final)(results)
    # resultsf = results.compute()
    # print(resultsf)
