#!/usr/bin/env python

import functools
import multiprocessing as mp
import pathlib
import pickle
import random
import time
from collections import Counter, defaultdict
from typing import Any, Generator

import numpy as np
import pandas as pd
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from utils import _open

from strainr.parameter_config import process_arguments

SETCHUNKSIZE = 10000


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


def fast_count_kmers(seq_id: str, seq: bytes) -> tuple[str, np.ndarray]:
    """Main function to assign strain hits to reads"""
    matched_kmer_strains = []
    na_zeros = np.zeros(len(strains), dtype=np.uint8)
    max_index = len(seq) - kmerlen + 1
    with memoryview(seq) as seqview:
        for i in range(max_index):
            returned_strains = db.get(seqview[i : i + kmerlen])
            if returned_strains is not None:
                matched_kmer_strains.append(returned_strains)
    final_tally = sum(matched_kmer_strains)
    if isinstance(final_tally, np.ndarray):
        return seq_id, final_tally
    else:
        return seq_id, na_zeros


def fast_count_kmers_mpire(db: dict, seq_id: str, seq: bytes) -> tuple[str, np.ndarray]:
    """Main function to assign strain hits to reads"""
    # import mmh3
    matched_kmer_strains = []
    na_zeros = np.full(len(strains), 0)
    max_index = len(str(seq)) - kmerlen + 1
    with memoryview(seq) as seqview:
        for i in range(max_index):
            returned_strains = db.get(seqview[i : i + kmerlen])
            # returned_strains = db.get(mmh3.hash(bytes(seqview[i : i + kmerlen]), signed=False, seed=SEED))
            if returned_strains is not None:
                matched_kmer_strains.append(returned_strains)
    final_tally = sum(matched_kmer_strains)
    if isinstance(final_tally, np.ndarray):
        return seq_id, final_tally
    else:
        return seq_id, na_zeros


def fast_count_kmers_helper(seqtuple: tuple[str, bytes]):
    return fast_count_kmers(*seqtuple)


def FastqEncodedGenerator(inputfile: pathlib.Path) -> Generator:
    """
    Generate an updated generator expression, but without quality scores, and encodes sequences.
    This is a test to see if the linter will catch
    """
    with _open(inputfile) as f:
        for seq_id, seq, _ in FastqGeneralIterator(f):
            yield seq_id, bytes(seq, "utf-8")


def classify(input_file: pathlib.Path) -> list[tuple[str, np.ndarray]]:
    """Call multiprocessing library to lookup k-mers."""
    t0 = time.time()
    nreads = int(sum(1 for i in open(input_file, "rb")) / 4)
    print(f"Reads: {nreads}")

    # From 3-item generator to 2-item generator
    record_iter = (r for r in FastqEncodedGenerator(input_file))

    # Generate k-mers, lookup strain spectra in db, return sequence scores
    # with mpire.WorkerPool(n_jobs=args.procs, shared_objects=db) as pool:
    #     results = list(
    #         pool.imap_unordered(
    #             fast_count_kmers_mpire,
    #             record_iter,
    #             iterable_len=nreads,
    #             progress_bar=True,
    #         )
    #     )

    with mp.Pool(processes=args.procs) as pool:
        results = list(
            pool.imap_unordered(
                fast_count_kmers_helper, record_iter, chunksize=SETCHUNKSIZE
            )
        )

    print(f"Ending classification: {time.time() - t0}s")
    return results


def separate_hits(
    hitcounts: list[tuple[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """
    Return maps of reads with 1 (clear), multiple (ambiguous), or no signal.

    "Positions in the array correspond to bins for each strain, so we search
    for maxima along the vector, single max providing single strain evidence,
    multiple maxima as ambiguous, NA as NA, etc.
    """
    clear_hits: dict[str, np.ndarray] = {}
    ambig_hits: dict[str, np.ndarray] = {}
    core_reads: dict[str, np.ndarray] = {}
    none_hits: list[str] = []
    for read, hit_array in hitcounts:
        if np.all(hit_array == 0):
            none_hits.append(read)

        else:  # Find all max values and check for mulitiple maxima within the same array
            max_indices = np.argwhere(hit_array == np.max(hit_array)).flatten()

            if len(max_indices) == 1:  # Single max
                clear_hits[read] = hit_array

            elif len(max_indices) == len(hit_array):
                # print("Mladen is testing at 3am")
                # print(hit_array)
                core_reads[read] = hit_array

            elif len(max_indices) > 1 and sum(hit_array) > 0:
                ambig_hits[read] = hit_array

            else:
                raise Exception("This shouldnt occcur")

    print(f"Reads with likely assignment to a single strain: {len(clear_hits)}")
    print(f"Reads with multiple likely candidates: {len(ambig_hits)}")
    print(f"Reads with no hits to any reference genomes: {len(none_hits)}")
    print(f"Core reads: {len(core_reads)}")
    return clear_hits, ambig_hits, none_hits


def prior_counter(clear_hits: dict[str, int]) -> Counter[int]:
    """Aggregate the values which are indices corresponding to strain names"""
    return Counter(clear_hits.values())


def counter_to_array(prior_counter: Counter[int], nstrains: int) -> np.ndarray:
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
    """Call the resolution for parallel option selection.

    Main function called by helper for parallel disambiguation"""
    # Treshold at max
    belowmax = hits < np.max(hits)
    hits[belowmax] = 0
    mlehits = hits * prior  # Apply prior

    if selection == "random":  # Weighted assignment
        return random.choices(
            range(len(hits)),
            weights=mlehits,
            k=1,
        ).pop()

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


def parallel_resolve_helper(
    ambig_hits, prior, selection="multinomial"
) -> tuple[dict, dict]:
    """
    Assign a strain to reads with ambiguous k-mer signals by maximum likelihood.

    Currently 3 options: random, max,
    and dirichlet. (dirichlet is slow and performs similar to random)
    For all 3, threshold spectra to only include
    maxima, multiply w/ prior.
    """
    new_clear, new_ambig = {}, {}
    mapfunc = functools.partial(parallel_resolve, prior=prior, selection=selection)

    resolve_cores = max(1, args.procs // 4)

    with mp.Pool(processes=resolve_cores) as pool:
        for read, outhits in zip(
            ambig_hits.keys(), pool.map(mapfunc, ambig_hits.values())
        ):
            new_clear[read] = outhits

    return new_clear, new_ambig


def collect_reads(
    clear_hits: dict[str, int], updated_hits: dict[str, int], na_hits: list[str]
) -> dict[str, Any]:
    """Assign the NA string to na and join all 3 dicts."""
    np.full(len(strains), 0.0)
    na = {k: "NA" for k in na_hits}
    all_dict = clear_hits | updated_hits | na
    print(len(all_dict), len(clear_hits), len(updated_hits), len(na_hits))
    # assert len(all_dict) == len(clear_hits) +
    # len(updated_hits) + len(na_hits)
    return all_dict


def resolve_clear_hits(clear_hits) -> dict[str, int]:
    """
    INPUT: Reads whose arrays contain singular maxima - clear hits
    OUTPUT: The index/strain corresponding to the maximum value
    Replace numpy array with index"""
    return {k: int(np.argmax(v)) for k, v in clear_hits.items()}


def normalize_counter(acounter: Counter, remove_na=False) -> Counter[Any]:
    """Regardless of key type, return values that sum to 1."""
    if remove_na:
        acounter.pop("NA", None)
    total_counts = sum(acounter.values())
    norm_counter = Counter({k: v / total_counts for k, v in acounter.items()})
    return norm_counter


def threshold_by_relab(norm_counter_all, threshold=0.02):
    """
    Given a percentage cutoff [threshold], remove strains
    which do not meet the criteria and recalculate relab.
    """
    thresh_counter = {}
    for k, v in norm_counter_all.items():
        if v > threshold:
            thresh_counter[k] = v
        else:
            thresh_counter[k] = 0.0
    # thresh_results = Counter({k: v for k, v in norm_counter_all.items() if v > threshold})
    # if thresh_results["NA"]:
    #     thresh_results.pop("NA")
    return Counter(thresh_counter)
    # return normalize_counter(Counter(thresh_counter),remove_na=True)


def display_relab(
    acounter: Counter,
    nstrains: int = 10,
    template_string: str = "",
    display_na=True,
):
    """
    Pretty print for counters:
    Can work with either indices or names
    """
    print(f"\n\n{template_string}\n")

    choice_list = []
    for strain, abund in acounter.most_common(n=nstrains):
        if isinstance(strain, int):
            s_index = strain  # for clarity
            if abund > 0.0:
                print(f"{abund}\t{strains[s_index]}")
                choice_list.append(strains[s_index])
        elif isinstance(strain, str) and strain != "NA":
            if abund > 0.0:
                print(f"{abund}\t{strain}")
                choice_list.append(strain)

    if display_na and acounter["NA"] > 0.0:
        print(f"{acounter['NA']}\tNA\n")

    return choice_list


def translate_strain_indices_to_names(counter_indices, strain_names):
    """
    Convert a dictionary or counter from {strain_index: hits} to {strain_name: hits}.

    Args:
        counter_indices (dict or Counter): The dictionary or counter containing strain indices as keys and hits as values.
        strain_names (list): The list of strain names corresponding to the indices.

    Returns:
        Counter: The converted counter with strain names as keys and hits as values.
    """
    name_to_hits = {}
    for k_idx, v_hits in counter_indices.items():
        if k_idx != "NA" and isinstance(k_idx, int):
            name_to_hits[strain_names[k_idx]] = v_hits
        elif k_idx == "NA":
            continue
            # name_to_hits['NA'] = v_hits
        else:
            try:
                name_to_hits[strain_names[k_idx]] = v_hits
            except TypeError:
                print(
                    f"The value from either {k_idx} or {strain_names[k_idx]} is not the correct type."
                )
                print(f"The type is {type(k_idx)}")
    return Counter(name_to_hits)


def add_missing_strains(strain_names: list[str], final_hits: Counter[str]):
    """
    Provides a counter that has all the strains with 0 hits for completeness

    Args:
        strain_names (list[str]): List of strain names
        final_hits (Counter[str]): Counter object containing the hits for each strain

    Returns:
        Counter[str]: Counter object with all the strains, including those with 0 hits
    # Implementation code goes here

    Returns:
        Counter: Counter object with all the strains, including those with 0 hits
    """
    full_strain_relab: defaultdict = defaultdict(float)
    for strain in strain_names:
        full_strain_relab[strain] = final_hits[strain]
    full_strain_relab["NA"] = final_hits["NA"]
    return Counter(full_strain_relab)


# Function to make each counter into a pd.DataFrame
def counter_to_pandas(relab_counter, column_name):
    relab_df = pd.DataFrame.from_records(
        list(dict(relab_counter).items()), columns=["strain", column_name]
    ).set_index("strain")
    return relab_df


def output_results(
    results: dict[str, int], strains: list[str], outdir: pathlib.Path
) -> pd.DataFrame:
    """
    From {reads->strain_index} to {strain_name->rel. abundance}
    and returns a dataFrame.

    Args:
        results (dict[str, int]): A dictionary mapping reads to strain indices.
        strains (list[str]): A list of strain names.
        outdir (pathlib.Path): The output directory.

    Returns:
        pd.DataFrame: The resulting DataFrame containing strain names and relative abundances.
    """
    outdir.mkdir(parents=True, exist_ok=True)  # todo

    # Transform hit counts to rel. abundance through counters, and fetch name mappings.
    index_hits = Counter(results.values())
    name_hits: Counter[str] = translate_strain_indices_to_names(index_hits, strains)
    full_name_hits: Counter[str] = add_missing_strains(strains, name_hits)
    display_relab(full_name_hits, template_string="Overall hits")

    final_relab: Counter[str] = normalize_counter(full_name_hits, remove_na=False)
    display_relab(final_relab, template_string="Initial relative abundance ")

    final_threshab = threshold_by_relab(final_relab, threshold=args.thresh)
    final_threshab = normalize_counter(final_threshab, remove_na=True)
    choice_strains = display_relab(
        final_threshab, template_string="Post-thresholding relative abundance"
    )

    # Each abundance slice gets put into a df/series
    relab_columns = [
        counter_to_pandas(full_name_hits, "sample_hits"),
        # this should include NA
        counter_to_pandas(final_relab, "sample_relab"),
        counter_to_pandas(final_threshab, "intra_relab"),
    ]

    # Concatenate and sort
    results_table = pd.concat(relab_columns, axis=1).sort_values(
        by="sample_hits", ascending=False
    )

    results_table.to_csv((outdir / "abundance.tsv"), sep="\t")
    return results_table.loc[choice_strains]


def build_database(dbpath):
    """
    Build a database from a compressed dataframe.

    Parameters:
    dbpath (str): The path to the compressed dataframe.

    Returns:
    tuple: A tuple containing the database, strains, and kmer length.

    Raises:
    AssertionError: If the length of any kmer in the dataframe is not equal to kmerlen.
    """
    print("Loading Database.")

    global kmerlen, strains, db
    df = pd.read_pickle(dbpath)
    # kmerlen = len(df.index[0])
    kmerlen = 31
    strains = list(df.columns)

    # Convert to dictionary
    db = dict(zip(df.index, df.to_numpy()))

    # assert all(df.index.str.len() == kmerlen)  # Check all kmers
    print(f"Database of {len(strains)} strains loaded")
    return db, strains, kmerlen


def main():
    """
    Execute main loop.
    1. Classify
    2. Generate Initial Abundance estimate
    3. Reclassify ambiguous
    4. Collect reads for abundance
    5. Normalize, threshold, re-normalize
    6. Output
    """
    # Build
    results_raw: list[tuple[str, np.ndarray]] = classify(fasta)

    # TODO
    save_raw = True
    if save_raw:
        save_read_spectra(strains, results_raw)

    clear_hits, ambig_hits, na_hits = separate_hits(results_raw)

    #

    nn
    # dict values are now single index
    assigned_clear = resolve_clear_hits(clear_hits)
    cprior = prior_counter(assigned_clear)
    display_relab(normalize_counter(cprior), template_string="Prior Estimate")

    # Finalize
    prior = counter_to_array(cprior, len(strains))  # counter to vector
    new_clear, _ = parallel_resolve_helper(ambig_hits, prior, args.mode)
    total_hits: dict = collect_reads(assigned_clear, new_clear, na_hits)

    # Output
    if out:
        df_relabund = output_results(total_hits, strains, out)
        print(df_relabund[df_relabund["sample_hits"] > 0])
        print(f"Saving results to {out}")


def save_read_spectra(strains: list[str], results_raw):
    """Pickles the raw results from the k-mer db lookup"""
    import pickle

    # TODO - useful elsewhere
    # df = pd.DataFrame.from_dict(dict(results_raw),orient=index,columns=strains)
    df = pd.DataFrame(results_raw)
    print(df.head())
    out.mkdir(parents=True, exist_ok=True)  # todo
    output_file = pathlib.Path(out.parent / "raw_scores.pkl")
    with output_file.open("wb") as pf:
<<<<<<< HEAD:strainr/strainr-classify.py
        pickle.dump(results_raw, pf)  # , file=pf, protocol=pickle.HIGHEST_PROTOCOL)
=======
        pickle.dump(results_raw, pf) #, file=pf, protocol=pickle.HIGHEST_PROTOCOL)
>>>>>>> 5c49409 (Add strainr-pangenome script and associated functionalities):strainr/strainr-pangenome.py
        # pickle.dump(strains, file=pf, protocol=pickle.HIGHEST_PROTOCOL)
    return


if __name__ == "__main__":
    args = process_arguments.parse_args()
    db, strains, kmerlen = build_database(args.db)
    p = pathlib.Path().cwd()
    rng = np.random.default_rng()
    print("\n".join(f"{k} = {v}" for k, v in vars(args).items()))
    args.input.reverse()
    for in_fasta in args.input:
        t0 = time.time()
        fasta = pathlib.Path(in_fasta)
        out: pathlib.Path = args.out / str(fasta.stem)
        if not out.exists():
            print(f"Input file:{fasta}")
            main()
            print(f"Time for {fasta}: {time.time()-t0}")
<<<<<<< HEAD:strainr/strainr-classify.py


# if __name__ == "__main__":
#     p = pathlib.Path().cwd()
#     rng = np.random.default_rng()

#     args = get_args().parse_args()
#     db, strains, kmerlen = build_database(args.db)
#     print("\n".join(f"{k} = {v}" for k, v in vars(args).items()))

#     for in_fasta in args.input:
#         t0 = time.time()
#         fasta = pathlib.Path(in_fasta)
#         out = args.out / str(fasta.stem)
#         if not out.exists():
#             print(f"Input file:{fasta}")
#             main()
#             print(f"Time for {fasta}: {time.time()-t0}")
=======
>>>>>>> 5c49409 (Add strainr-pangenome script and associated functionalities):strainr/strainr-pangenome.py
