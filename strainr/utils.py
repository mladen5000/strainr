#!/usr/bin/env python

import gzip
import mimetypes
import pathlib
import pickle
import re
import shutil # Added for shutil.which
from typing import Dict, List, Tuple, Union, TextIO, BinaryIO

import numpy as np  # For type hinting np.ndarray
import pandas as pd
from Bio.Seq import Seq


def _get_sample_name(file_path: pathlib.Path) -> str:
    """
    Extract sample name from file path.

    Sanitizes the result to prevent path traversal attacks by removing
    potentially dangerous characters and patterns.

    Args:
        file_path: Path to the input file

    Returns:
        Sanitized sample name safe for use in filesystem paths

    Raises:
        ValueError: If the resulting sanitized name is empty
    """
    # Remove common suffixes and prefixes
    name = file_path.name
    for suffix in [".fastq", ".fasta", ".fq", ".fa", ".gz"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    # Remove common read pair indicators
    for pattern in ["_R1", "_R2", "_1", "_2"]:
        if pattern in name:
            name = name.split(pattern)[0]
            break

    # Sanitize to prevent path traversal attacks
    # Only allow alphanumeric characters, underscores, hyphens, and dots
    # This prevents "../", "//", "\\", and other path manipulation
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)

    # Remove leading/trailing dots and underscores (can be problematic)
    safe_name = safe_name.strip('._')

    # Ensure name is not empty after sanitization
    if not safe_name:
        # Fallback to a hash of the original name to ensure uniqueness
        import hashlib
        safe_name = f"sample_{hashlib.md5(name.encode()).hexdigest()[:8]}"

    return safe_name


def open_file_transparently(
    # TECH DEBT SUGGESTION:
    # This function's logic for handling both text/binary modes and
    # gzip/non-gzip files transparently can be complex, especially
    # concerning type hints (TextIO vs BinaryIO) and ensuring the
    # correct mode is used with gzip.open vs open.
    # The existing comments within the function highlight some of these
    # complexities (e.g., "This path is problematic for TextIO return hint").
    #
    # If this function becomes a source of bugs or is hard to maintain,
    # consider:
    #   - Separating into two functions: e.g., `open_text_file_transparently`
    #     and `open_binary_file_transparently`.
    #   - Simplifying the type handling if possible, or making behavior
    #     more explicit for callers.
    #   - Adding more comprehensive tests for various mode combinations
    #     and file types.
    file_path: Union[str, pathlib.Path],
    mode: str = "rt",
) -> Union[TextIO, BinaryIO]:
    """Opens a file, transparently handling gzip compression.

    Infers compression from file extension. Defaults to text read mode.

    Args:
        file_path: Path to the file.
        mode: File open mode (e.g., "rt", "rb", "wt", "wb"). Defaults to "rt".

    Returns:
        A file object (TextIO or BinaryIO depending on mode).

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If an I/O error occurs during opening.
        TypeError: If file_path is not a str or pathlib.Path.
    """
    if not isinstance(file_path, (str, pathlib.Path)):
        raise TypeError(
            f"file_path must be a string or pathlib.Path, not {type(file_path)}"
        )

    file_path = pathlib.Path(file_path)

    if not file_path.exists():  # Explicit check before mimetypes or open
        raise FileNotFoundError(f"File not found: {file_path}")

    guessed_type, encoding = mimetypes.guess_type(str(file_path))

    try:
        if encoding == "gzip":
            return gzip.open(file_path, mode=mode)  # type: ignore # gzip.open can return TextIO
        else:
            # For non-gzip, ensure 'b' is not in mode if we expect TextIO,
            # or ensure 't' is not in mode if we expect BinaryIO.
            # The type hint TextIO implies text mode.
            if "b" in mode:
                # This case would violate TextIO return if not for type: ignore.
                # For true transparent opening, the return type might need to be Union[TextIO, BinaryIO]
                # or the function should be split. For now, assume text mode is primary.
                # If mode is "rb", "wb", etc., this will return a BinaryIO.
                # The type hint is TextIO, so we prioritize text.
                # If a binary mode is passed, the user might get a BinaryIO despite TextIO hint.
                # This is a known complexity with such transparent openers.
                # For this refactor, we stick to the original intent of TextIO where possible.
                if "t" not in mode:  # if mode is purely binary e.g. "rb"
                    # This path is problematic for TextIO return hint.
                    # However, since original was TextIO, let's assume "rt" or "r" are typical.
                    pass  # Allow binary modes, but caller must be aware of return type change
            return open(file_path, mode=mode)
    except (IOError, OSError) as e:
        raise IOError(f"Error opening file {file_path} with mode '{mode}': {e}") from e


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
    raw_kmer_scores: List[
        Tuple[str, np.ndarray]
    ],  # e.g., List[Tuple[ReadId, CountVector]]
    final_read_assignments: Dict[
        str, Union[str, int]
    ],  # e.g., Dict[ReadId, Union[StrainName, StrainIndex]]
) -> None:
    """Pickles raw k-mer scores and final read assignments to disk.

    Args:
        output_dir: Directory to save pickle files.
        raw_kmer_scores: List of tuples, where each tuple contains a read ID (str)
                         and its associated k-mer scores/counts (np.ndarray).
        final_read_assignments: Dictionary mapping read IDs (str) to their final
                                assignment (e.g., strain name as str, strain index as int,
                                or an unassigned marker str).

    Raises:
        IOError: If an error occurs during file writing or pickling.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_results_path = output_dir / "raw_kmer_scores.pkl"
        with open(raw_results_path, "wb") as fh_raw:
            pickle.dump(raw_kmer_scores, fh_raw)
        print(f"Raw k-mer scores pickled to: {raw_results_path}")

        final_assignments_path = output_dir / "final_read_assignments.pkl"
        with open(final_assignments_path, "wb") as fh_final:
            pickle.dump(final_read_assignments, fh_final)
        print(f"Final read assignments pickled to: {final_assignments_path}")

    except (IOError, pickle.PicklingError) as e:
        raise IOError(
            f"Error pickling intermediate results to {output_dir}: {e}"
        ) from e


def save_classification_results_to_dataframe(
    output_dir: pathlib.Path,
    intermediate_scores: Dict[
        str, Union[List[float], np.ndarray]
    ],  # ReadID to scores per strain
    final_assignments: Dict[str, str],  # ReadID to assigned strain name (or "NA")
    strain_names: List[str],
) -> None:
    """Converts classification results to a Pandas DataFrame and pickles it.

    Args:
        output_dir: Directory to save the pickled DataFrame.
        intermediate_scores: Dictionary mapping read IDs (str) to a list or array
                             of scores against each strain (float or convertible).
        final_assignments: Dictionary mapping read IDs (str) to their final assigned
                           strain name (str) or an unassigned marker (e.g., "NA").
        strain_names: List of all strain names, defining the order of columns for scores.

    Raises:
        IOError: If an error occurs during DataFrame creation, file writing, or pickling.
        ValueError: If data for DataFrame creation is inconsistent.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create DataFrame from intermediate scores
        # Ensure column order matches strain_names for consistency
        results_df = pd.DataFrame.from_dict(
            intermediate_scores, orient="index", columns=strain_names
        )

        # Scores are often floats (probabilities, likelihoods, etc.)
        # Casting to float is safer than int if the nature of scores is not strictly integer.
        try:
            results_df = results_df.astype(float)
        except ValueError as e:
            # If scores cannot be cast to float (e.g., contain non-numeric strings erroneously)
            raise ValueError(
                f"Intermediate scores contain non-numeric values that cannot be cast to float. Error: {e}"
            ) from e

        # Prepare series for final assignments
        assigned_strains_series = pd.Series(
            final_assignments, name="final_assigned_strain"
        )

        # Join scores DataFrame with final assignments series
        # Use how='left' to keep all reads from results_df (scores table)
        # and add assignments where available. Reads in results_df but not in
        # assigned_strains_series will have NaN for 'final_assigned_strain'.
        results_df = results_df.join(assigned_strains_series, how="left")

        dataframe_pickle_path = output_dir / "classification_results_table.pkl"
        results_df.to_pickle(dataframe_pickle_path)
        print(f"Classification results DataFrame pickled to: {dataframe_pickle_path}")

    except (
        IOError,
        pickle.PicklingError,
        ValueError,
    ) as e:  # Added ValueError for DataFrame issues
        raise IOError(
            f"Error saving classification results to DataFrame at {output_dir}: {e}"
        ) from e


def check_external_commands(commands: List[str]) -> None:
    """
    Checks if specified external commands are available in the system's PATH.

    Args:
        commands: A list of command names (strings) to check.

    Raises:
        FileNotFoundError: If any of the specified commands are not found.
    """
    missing_commands = []
    for cmd in commands:
        if shutil.which(cmd) is None:
            missing_commands.append(cmd)

    if missing_commands:
        raise FileNotFoundError(
            f"The following required external command(s) were not found in PATH: {', '.join(missing_commands)}. "
            "Please ensure they are installed and accessible."
        )
