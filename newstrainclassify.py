# new
import sys
import time
import pickle
import pathlib
import argparse
import random
import functools

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
    with mp.Pool(processes=16) as pool:
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


def print_relab(posterior_prob, nstrains=10):
    """Displays Top 10 relative abundance strains"""
    print(f"\nRelative Abundance Estimation")
    for k, v in posterior_prob.most_common(n=nstrains):
        print(k, "\t", round(v, 5))
    print(f"\n")


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


def disambiguate(ambig_hits, prior):
    """
    Take ambiguous values (arrays) and multiple with prior and take maxima

    """
    new_clear, new_ambig = {}, {}
    rng = np.random.default_rng()
    # Selection criteria
    selection = "random"

    for read, hits in ambig_hits.items():

        # Get max values of ambigs first
        belowmax = hits != np.max(hits)
        hits[belowmax] = 0
        # Apply prior
        mlehits = hits * prior

        # Probabilistic assignment
        if selection == "random":
            # select_random = rng.choice(len(hits), p=mlehits / sum(mlehits))
            select_random = random.choices(range(len(hits)), weights=mlehits, k=1).pop()
            new_clear[read] = select_random

        # Maximum Likelihood assignment
        elif selection == "max":
            select_max = np.argwhere(mlehits == np.max(mlehits)).flatten()
            if len(select_max) == 1:
                new_clear[read] = int(select_max)

            elif len(select_max) > 1:  # Do nothing
                new_ambig[read] = hits

            else:
                raise ValueError("Length 0 for final strain array")
        # TODO
        elif selection == "dirichlet":
            mlehits[mlehits == 0] = 1e-10
            select_dirichlet = rng.dirichlet(mlehits, 1)
            new_clear[read] = np.argmax(select_dirichlet)
        else:
            raise ValueError("Must select a selection mode")

    assert len(ambig_hits) == len(new_clear) + len(new_ambig)
    return new_clear, new_ambig


def joined_abundance(clear_hits, new_ambig, na_hits):
    clear = {k: np.argmax(v) for k, v in clear_hits.items()}
    na = {k: None for k in na_hits}
    all_dict = clear | new_ambig | na
    assert len(all_dict) == len(clear_hits) + len(new_ambig) + len(na_hits)
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


if __name__ == "__main__":

    p = pathlib.Path().cwd()
    params = vars(get_args().parse_args())
    kmerlen = 31
    f1 = "test_R1.fastq"
    t0 = time.time()
    # df = load_database("new_method.sdb")
    df = load_database("hflu_complete_genbank.db")
    # df = load_database("database.db")
    kmerlen = get_kmer_len(df)
    strains, db = df_to_dict(df)
    print(strains)
    nstrains = len(strains)
    results_raw = classify()  # get list of (SecRecord,nparray)
    clear_hits, ambig_hits, na_hits = separate_hits(results_raw)  # parse into 3 dicts

    # Clear and Prior building
    assigned_clear = resolve_clear_hits(clear_hits)  # dict values are now single index
    cprior = prior_counter(assigned_clear)
    prior = counter_to_array(cprior, nstrains)  # counter to vector
    prior[prior == 0] = 1e-10
    print(prior)

    # Assign ambig
    # function to identify arg max, then create a vector with just those 2 values and the rest 0s
    new_clear, new_ambig = disambiguate(ambig_hits, prior)

    if len(new_ambig) > 0:
        pass

    # Build counter of hits
    counter_hits = cprior + Counter(new_clear.values())
    total = sum(counter_hits.values())
    norm_hits = Counter({k: v / total for k, v in counter_hits.items()})
    print(norm_hits.most_common())

    # norm_prior = generate_initial_probability(clear_hits)
    # print("\nUpdating estimates..\n")
    # res_reads, ambig_reads, prior = disambiguate(clear_hits, ambig_hits)

    # print(len(res_reads))

    # assigned_ambig = resolve_ambig_hits(res_reads)
    # assigned_na = build_na_dict(na_hits)
    # all_dict = assigned_clear | assigned_ambig | assigned_na
    # assert len(all_dict) == len(assigned_clear) + len(assigned_ambig) + len(assigned_na)
    # print('hi')

    # all_dict = dict(sorted(all_dict.items(), key=lambda kv: kv[1]))
    # ccres = Counter(all_dict.values())
    # total = sum(ccres.values())
    # results = {k:v/total for k,v in ccres.items()}
    # print_relab(Counter(results))
    # print('hi2')
    # named_results = {strains[k]:v for k,v in results.items() if k is not None}
    # rawfinal = Counter(named_results).most_common()
    # print(rawfinal)
    # print(rawfinal)

    # prior = generate_likelihood(clear_hits)
    # cprior = Counter(prior)
    # print_relab(cprior)
    # new_ambig = resolve_ties(ambig_hits, prior)
    # all_dict = joined_abundance(clear_hits, new_ambig, na_hits)

    # relab = Counter(list(all_dict.values()))
    # # relab = {str(k):v for k,v in relab}
    # print(relab)
    # relab2 = normalize(relab)
    # print(relab2)
    # relab3 = threshold(relab2,0.02)
    # print(relab3)
    # for k,v in relab3.items():
    #     if k:
    #         print(strains[k],v)

    # print(ambig_hits)
