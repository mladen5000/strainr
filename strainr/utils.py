#!/usr/bin/env python

import functools
import gzip
import mimetypes
import pickle
import pathlib
from typing import Any, Dict, List, TextIO

import pandas as pd
from Bio.Seq import Seq # Assuming kmer is a BioPython Seq object


def open_file_transparently(file_path: pathlib.Path) -> TextIO:
    """Opens a file, transparently handling gzip compression.

    The function infers the compression type from the file extension.
    It supports plain text files and gzip compressed files.

    Args:
        file_path: The path to the file to be opened.

    Returns:
        A file object opened in text mode.
    """
    # Guess encoding based on file extension (e.g., .gz for gzip)
    guessed_type, encoding = mimetypes.guess_type(str(file_path))
    
    # Choose the appropriate open function
    if encoding == "gzip":
        open_func = functools.partial(gzip.open, mode="rt")
    else:
        open_func = functools.partial(open, mode="r") # Ensure text mode for regular files too
        
    file_object = open_func(file_path)
    return file_object


def get_canonical_kmer(kmer: Seq) -> Seq:
    """Computes the canonical representation of a k-mer.

    The canonical k-mer is the lexicographically smaller of the k-mer
    and its reverse complement. This is useful for ensuring that
    strand orientation does not affect k-mer identity.

    Args:
        kmer: The k-mer sequence (e.g., a Bio.Seq.Seq object).

    Returns:
        The canonical k-mer sequence.
    """
    reverse_complement_kmer = kmer.reverse_complement()
    return kmer if str(kmer) < str(reverse_complement_kmer) else reverse_complement_kmer


def pickle_intermediate_results(
    output_dir: pathlib.Path,
    raw_kmer_scores: List[Any],  # Or more specific type if known, e.g., List[Tuple[str, np.ndarray]]
    final_read_assignments: Dict[str, Any], # Or more specific type, e.g., Dict[str, int] for strain indices
    # strains_list: List[str] # This argument is currently unused
) -> None:
    """Pickles raw k-mer scores and final read assignments to disk.

    This function saves intermediate pipeline results for potential
    debugging or later analysis.

    Args:
        output_dir: The directory where the pickle files will be saved.
        raw_kmer_scores: A list or collection of raw k-mer scores
                         (structure depends on upstream processing).
        final_read_assignments: A dictionary mapping read identifiers to their
                                assigned strain identifiers or other assignment info.
        # strains_list: A list of strain names. Currently unused in this function.

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    
    raw_results_path = output_dir / "raw_kmer_scores.pkl"
    with open(raw_results_path, "wb") as fh:
        pickle.dump(raw_kmer_scores, fh)

    final_assignments_path = output_dir / "final_read_assignments.pkl"
    with open(final_assignments_path, "wb") as fh:
        pickle.dump(final_read_assignments, fh)
    return


def save_classification_results_to_dataframe(
    output_dir: pathlib.Path,
    intermediate_scores: Dict[str, List[float]], # Assuming scores are lists of floats per read
    final_assignments: Dict[str, str], # Read ID to assigned strain name (or "NA")
    strain_names: List[str]
) -> None:
    """Converts classification results to a Pandas DataFrame and pickles it.

    The DataFrame includes intermediate k-mer scores for each read against
    all strains and the final assigned strain name for each read.

    Args:
        output_dir: The directory where the pickled DataFrame will be saved.
        intermediate_scores: A dictionary mapping read IDs to lists of k-mer scores
                             against each strain.
        final_assignments: A dictionary mapping read IDs to their final assigned
                           strain name (or "NA" if unassigned).
        strain_names: A list of all strain names, corresponding to the order
                      of scores in `intermediate_scores`.

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    # Create DataFrame from intermediate scores
    # Ensuring column order matches strain_names for consistency
    results_df = pd.DataFrame.from_dict(
        intermediate_scores, orient="index", columns=strain_names
    )
    # Convert scores to integer type if appropriate, or float if not.
    # Assuming int for now as per original, but float might be more general.
    results_df = results_df.astype(int) 

    # Prepare series for final assignments to join with the main DataFrame
    # Original code: final_names = {k: strains[int(v)] for k, v in results.items() if v != "NA"}
    # This implies 'results' (now final_assignments) might have integer indices for strains if not "NA".
    # For clarity, if final_assignments already contains strain names, this step is simpler.
    # Assuming final_assignments directly maps read_id -> strain_name or "NA"
    assigned_strains_series = pd.Series(final_assignments).rename("final_assigned_strain")
    
    results_df = results_df.join(assigned_strains_series)

    dataframe_pickle_path = output_dir / "classification_results_table.pkl"
    results_df.to_pickle(dataframe_pickle_path)
    return

# Removed unused call_pickle variable and associated commented code.
