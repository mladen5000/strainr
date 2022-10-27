#!/usr/bin/env python
import functools
import multiprocessing as mp
import pathlib
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# Dummy class for args
class args():
    procs: int
    mode: str
    input: list[pathlib.Path]
    out: pathlib.Path 


def separate_hits(
    hitcounts: list[tuple[str, np.ndarray]]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """
    Return maps of reads with 1 (clear), multiple (ambiguous), or no signal.

    "Positions in the array correspond to bins for each strain, so we search
    for maxima along the vector, single max providing single strain evidence,
    multiple maxima as ambiguous, NA as NA, etc.
    """
    clear_hits: dict[str, np.ndarray] = {}
    ambig_hits: dict[str, np.ndarray] = {}
    none_hits: list[str] = []
    for read, hit_array in hitcounts:

        if np.all(hit_array == 0):
            none_hits.append(read)

        else:  # Find all max values and check for mulitiple maxima within the same array
            max_indices = np.argwhere(hit_array == np.max(hit_array)).flatten()

            if len(max_indices) == 1:  # Single max
                clear_hits[read] = hit_array

            elif len(max_indices) > 1 and sum(hit_array) > 0:
                ambig_hits[read] = hit_array

            else:
                raise Exception("This shouldnt occcur")

    print(f"Reads with likely assignment to a single strain: {len(clear_hits)}")
    print(f"Reads with multiple likely candidates: {len(ambig_hits)}")
    print(f"Reads with no hits to any reference genomes: {len(none_hits)}")
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
    acounter: Counter, nstrains: int = 10, template_string: str = "", display_na=True
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
    """Convert dict/counter from {strain_index: hits} to {strain_name:hits}"""
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
    """Provides a counter that has all the strains with 0 hits for completeness"""
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


def main():
    """
    Utilize raw results from classification for various analyses.
    """
    # Build

    clear_hits, ambig_hits, na_hits = separate_hits(results_raw)

    # Disambiguate
    assigned_clear = resolve_clear_hits(clear_hits)  # dict values are now single index
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


if __name__ == "__main__":
    p = pathlib.Path().cwd()
    rng = np.random.default_rng()
    for in_fasta in args.input:
        fasta = pathlib.Path(in_fasta)
        out = args.out / str(fasta.stem)
        if not out.exists():
            print(f"Input file:{fasta}")
            main()
