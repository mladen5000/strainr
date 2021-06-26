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


# def normalize_dist(countdict, num_strains):
#     """
#     Convert hits per strain into % of each strain.
#     Also convert the dict of strain_index: value to a numpy array
#     where the value is placed in the index
#     example input: {2: 0.24, 4: 0.51, 10: 0.25}
#     example output: [0,0,0.24,0,0.51,0,0,0,0,0,0.25,0,0,0]
#     """
#     prior_array = np.zeros(num_strains)
#     total = sum(countdict.values())
#     for k, v in countdict.items():
#         prior_array[k] = v / total

#     # return {k:v/total for k,v in countdict.items()}
#     return prior_array


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


# def normalize_dist(countdict):
#     total = sum(countdict.values())
#     # TODO: test out new dict
#     # return { k: v/total for k,v in countdict.items() }
#     for el in countdict.keys():
#         countdict[el] /= total
#     return countdict


def generate_likelihood(clear_hits):
    """
    Aggregate clear hits in order to get initial estimate of distribution
    """
    unique_hit_distribution = Counter([np.argmax(v) for v in clear_hits.values()])
    nstrains = len(list(clear_hits.values())[0])  # TODO
    prior = normalize_dist(unique_hit_distribution, nstrains)
    return prior


def resolve_ties(ambig_hits, prior):
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
            select_random = rng.choice(len(hits), p=mlehits / sum(mlehits))
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
            raise ValueError('Must select a selection mode')

    assert len(ambig_hits) == len(new_clear) + len(new_ambig)
    return new_clear, new_ambig


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


def generate_initial_probability(clear_hits):
    print(f"Generating initial probability estimate of strains")

    unique_hit_distribution = Counter([np.argmax(v) for v in clear_hits.values()])
    normalized_prob = normalize_dist(unique_hit_distribution)

    print_relab(normalized_prob)
    print(f"Initial model complete, updating ambiguous reads..")
    return normalized_prob


def mp_break_ties(kcount: list, prior: dict):
    """Intersect prior probability for all strains with available strains in that read and return the maximum strain"""
    # Get columns with max score
    max_ind = np.argwhere(hits == np.max(hits)).flatten()

    # Convert to vector of probabilities
    strain_to_prob = {strain_id: prior[strain_id] for strain_id in kcount}
    # Select max choice only
    selection_max = (max(strain_to_prob, key=strain_to_prob.get),)

    # random choice
    if sum(strain_to_prob.values()) > 0:
        selection_randweighted = tuple(
            random.choices(
                list(strain_to_prob.keys()), weights=strain_to_prob.values(), k=1
            )
        )
    else:
        selection_randweighted = selection_max

    return (
        # selection_randweighted
        selection_max
    )


def disambiguate(clear_hits: dict, ambig_reads: dict):
    """
    Disambiguate the remaining strains
    """
    t0 = time.time()
    resolved_reads, new_ambig = {}, {}

    # Generate initial probability from clear reads
    normalized_prob = generate_initial_probability(clear_hits)

    maximize_strain_probability = functools.partial(
        mp_break_ties, prior=normalized_prob
    )

    # if nproc ==1:
    # for read, argmax_strain in zip(
    #     ambig_reads.keys(), map(maximize_strain_probability, ambig_reads.values())
    # ):
    #     resolved_reads[read] = argmax_strain

    with mp.Pool(processes=16) as pool:
        for read, argmax_strain in zip(
            ambig_reads.keys(),
            pool.map(maximize_strain_probability, ambig_reads.values()),
        ):
            resolved_reads[read] = argmax_strain

    print(f"The time for tie_breaking is {time.time() - t0}")

    return resolved_reads, new_ambig, normalized_prob


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
    new_clear, new_ambig = resolve_ties(ambig_hits, prior)

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
