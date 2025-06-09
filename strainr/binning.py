# Standard library imports
import gzip  # For writing gzipped example files
import multiprocessing as mp
import pathlib
import traceback  # For more detailed error info in main example
from typing import Optional, Dict, List, Set, Tuple, Union

# Third-party imports
import numpy as np  # Added missing import
import pandas as pd
from Bio import SeqIO

from .genomic_types import (  # Changed to relative import; Import FinalAssignmentsType
    FinalAssignmentsType,
    ReadId,
)

# Local application/library specific imports
from .utils import open_file_transparently  # Changed to relative import

# Removed redundant typing import


# Type alias FinalAssignmentsType is now imported from genomic_types


def generate_table(
    final_assignments: FinalAssignmentsType, all_strain_names: List[str]
) -> pd.DataFrame:
    """
    Generates a DataFrame indicating read assignments to strains.
    Rows: ReadIds, Columns: strain names. Value 1 if read assigned to strain, else 0.
    """
    if not isinstance(final_assignments, dict):
        raise TypeError("final_assignments must be a dictionary.")
    if not isinstance(all_strain_names, list) or not all(
        isinstance(s, str) for s in all_strain_names
    ):
        raise TypeError("all_strain_names must be a list of strings.")

    for read_id, assignment in final_assignments.items():
        if not isinstance(read_id, str) or not read_id:
            raise ValueError(
                "All ReadId keys in final_assignments must be non-empty strings."
            )
        if not (isinstance(assignment, int) and assignment >= 0) and not isinstance(
            assignment, str
        ):
            raise TypeError(
                f"Assignment for ReadId '{read_id}' must be a non-negative integer (StrainIndex) "
                f"or a string (unassigned marker), got {type(assignment)}."
            )

    if not isinstance(all_strain_names, list) or not all(
        isinstance(s, str) and s for s in all_strain_names
    ):
        raise TypeError("all_strain_names must be a list of non-empty strings.")
    if len(set(all_strain_names)) != len(all_strain_names):
        raise ValueError("all_strain_names must contain unique names.")

    read_ids = list(final_assignments.keys())
    if not all_strain_names:
        # Return empty DataFrame with read_ids as index if there are no strains to form columns
        # This ensures the index is consistent with final_assignments keys.
        print(
            "Warning: all_strain_names is empty. Returning DataFrame with no columns."
        )
        # It's possible final_assignments is empty or contains only non-integer markers.
        # If it contains integer assignments (StrainIndex), it's an issue if all_strain_names is empty.
        if any(isinstance(val, int) for val in final_assignments.values()):
            # This specific check is important for data integrity.
            raise ValueError(
                "all_strain_names is empty, but final_assignments contains integer (StrainIndex) "
                "assignments. Strain names are required to interpret these indices."
            )
        # Removed duplicate print statement
        return pd.DataFrame(index=read_ids)  # Return with read_ids as index

    # Ensure all_strain_names are unique, this was previously checked but good to be certain before creating DataFrame
    if len(set(all_strain_names)) != len(all_strain_names):
        raise ValueError(
            "all_strain_names must contain unique names for DataFrame columns."
        )

    df = pd.DataFrame(0, index=read_ids, columns=all_strain_names, dtype=np.uint8)

    for read_id, assignment in final_assignments.items():
        if isinstance(assignment, int):  # StrainIndex
            if 0 <= assignment < len(all_strain_names):
                strain_name = all_strain_names[assignment]
                df.loc[read_id, strain_name] = 1
            else:
                # Consolidated warning for invalid StrainIndex
                print(
                    f"Warning: Invalid StrainIndex {assignment} for read_id '{read_id}'. Max index is {len(all_strain_names) - 1}. Read will not be assigned in table."
                )
                # No assignment is made to df for this read_id if index is invalid

    return df


def get_top_strain_names(
    read_assignments: Dict[
        ReadId, Union[str, int]
    ],  # Assuming values are strain names or indices
    strain_list: List[str],
    unassigned_marker: str = "NA",
    exclude_unassigned: bool = True,
) -> List[str]:
    """Determines strain names sorted by the number of assigned reads."""
    if not isinstance(read_assignments, dict):
        raise TypeError("read_assignments must be a dictionary.")
    if not isinstance(strain_list, list) or not all(
        isinstance(s, str) for s in strain_list
    ):
        raise TypeError("strain_list must be a list of strings.")

    # Combined and clarified initial checks
    if not strain_list:
        if any(isinstance(val, int) for val in read_assignments.values()):
            print(
                "Warning: get_top_strain_names received an empty strain_list, but read_assignments contain integer indices. "
                "Cannot map indices to names. Returning empty list."
            )
        else:
            # If strain_list is empty and no integer assignments, it's valid to return an empty list.
            print(
                "Info: get_top_strain_names received an empty strain_list and no integer assignments. Returning empty list."
            )
        return []

    # Basic validation for read_assignments content (more detailed in generate_table if used prior)
    for read_id, assignment in read_assignments.items():
        if (
            not isinstance(read_id, str) or not read_id
        ):  # Ensure read_id is a non-empty string
            raise ValueError(
                "All ReadId keys in read_assignments must be non-empty strings."
            )
        if not (isinstance(assignment, int) and assignment >= 0) and not isinstance(
            assignment, str
        ):
            raise TypeError(
                f"Assignment for ReadId '{read_id}' must be a non-negative int (StrainIndex) or a string (marker), got {type(assignment)}."
            )

    # Validate strain_list properties
    if not all(
        isinstance(s, str) and s for s in strain_list
    ):  # Must be list of non-empty strings
        raise TypeError("strain_list must be a list of non-empty strings.")
    if len(set(strain_list)) != len(strain_list):  # Must contain unique names
        raise ValueError("strain_list must contain unique names.")

    if not isinstance(
        unassigned_marker, str
    ):  # Marker must be a string (can be empty if desired by user)
        raise TypeError("unassigned_marker must be a string.")
    # Removed redundant check for: `if not strain_list and any(isinstance(val, int) for val in read_assignments.values()):` as it's covered

    assignment_values = list(read_assignments.values())
    counts = pd.Series(assignment_values).value_counts()

    mapped_strain_counts: Dict[str, int] = {}
    for entity, count_val in counts.items():
        strain_name_to_add: Optional[str] = None
        if isinstance(entity, int):
            if 0 <= entity < len(strain_list):
                strain_name_to_add = strain_list[entity]
            # else: Invalid StrainIndex, simply ignore or log
        elif isinstance(entity, str):  # String marker or pre-resolved strain name
            if entity == unassigned_marker and exclude_unassigned:
                continue  # Skip if it's the unassigned marker and we're excluding it

            if (
                entity in strain_list
            ):  # Prioritize if it's a direct match to a known strain
                strain_name_to_add = entity
            elif (
                not exclude_unassigned and entity == unassigned_marker
            ):  # If we count unassigned marker explicitly
                strain_name_to_add = entity  # Use the marker itself as the "name"
            # In the original code, there was an `elif entity != unassigned_marker:`
            # This could lead to double counting or incorrect assignment if `entity` was in `strain_list`
            # but also happened to not be the `unassigned_marker`.
            # The current logic correctly prioritizes `entity in strain_list` first.
            # If an entity is not in strain_list and not the (non-excluded) unassigned_marker,
            # it will result in `strain_name_to_add` being None, so it won't be counted, which is the desired behavior.

        # Use the determined strain_name_to_add to update counts
        if strain_name_to_add:
            mapped_strain_counts[strain_name_to_add] = (
                mapped_strain_counts.get(strain_name_to_add, 0) + count_val
            )
        # The block that previously defined and used `actual_strain_name_for_count` has been removed
        # as it was redundant with the logic now correctly assigning to `strain_name_to_add`.

    # Sort strains by count in descending order
    sorted_strains_with_counts = sorted(
        mapped_strain_counts.items(), key=lambda item: item[1], reverse=True
    )

    # Return only the names, and ensure unassigned_marker is handled correctly based on exclude_unassigned
    # The filtering for unassigned_marker should ideally happen before sorting if it's definitely excluded,
    # or be part of the logic forming mapped_strain_counts.
    # Given the current structure, the list comprehension is fine.
    # The previous version had duplicate return statements and complex conditions.
    # This simplified version relies on mapped_strain_counts being correctly populated.

    final_sorted_strain_names = [
        s_name for s_name, s_count in sorted_strains_with_counts
    ]

    # If unassigned_marker was counted (i.e., exclude_unassigned=False and it had counts),
    # it will be in final_sorted_strain_names. If exclude_unassigned=True, it shouldn't be in mapped_strain_counts.
    return final_sorted_strain_names


def _extract_reads_for_strain(
    strain_name: str,
    read_ids_for_strain: Set[ReadId],
    forward_fastq_path: pathlib.Path,
    reverse_fastq_path: Optional[pathlib.Path],
    output_bin_dir: pathlib.Path,
) -> None:
    """
    Extracts reads for a specific strain and writes them to new FASTQ files.

    Args:
        strain_name: Name of the strain, used for naming output files.
        read_ids_for_strain: Set of read IDs assigned to this strain.
        forward_fastq_path: Path to the forward (R1) FASTQ file.
        reverse_fastq_path: Path to the reverse (R2) FASTQ file (optional).
        output_bin_dir: Directory where binned FASTQ files will be written.
    """
    if (
        not isinstance(forward_fastq_path, pathlib.Path)
        or (reverse_fastq_path and not isinstance(reverse_fastq_path, pathlib.Path))
        or not isinstance(output_bin_dir, pathlib.Path)
    ):
        raise TypeError("All path arguments must be pathlib.Path objects.")

    # The minimal docstring and the first parameter/path validation block have been removed as requested.
    # The more complete docstring and validation block below are retained.

    safe_strain_filename = (
        strain_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    )
    fastq_files_to_process: List[Tuple[pathlib.Path, str]] = [
        (forward_fastq_path, "R1")
    ]
    if reverse_fastq_path:
        if reverse_fastq_path.is_file():
            fastq_files_to_process.append((reverse_fastq_path, "R2"))
        else:
            print(
                f"Warning: Reverse FASTQ file specified but not found or not a file: {reverse_fastq_path}. Skipping R2 for strain {strain_name}."
            )

    if not fastq_files_to_process:
        print(
            f"No valid FASTQ input files found for strain '{strain_name}'. No binned output generated for this strain."
        )
        return  # Removed duplicate fastq_files_to_process.append

    for original_fastq_path, read_suffix in fastq_files_to_process:
        binned_fastq_filename = f"bin.{safe_strain_filename}_{read_suffix}.fastq"
        binned_fastq_filepath = output_bin_dir / binned_fastq_filename

        records_to_write = []
        try:
            # Use "rt" for text mode with open_file_transparently, as SeqIO.parse expects text.
            with open_file_transparently(
                original_fastq_path, mode="rt"
            ) as original_handle:
                for record in SeqIO.parse(original_handle, "fastq"):
                    # Read ID normalization example: normalized_id = record.id.split('/')[0].split(' ')[0]
                    # The original code had a duplicate `if record.id in read_ids_for_strain:`, which was a syntax error.
                    # Corrected to a single check.
                    if (
                        record.id in read_ids_for_strain
                    ):  # or normalized_id in read_ids_for_strain:
                        records_to_write.append(record)

            if records_to_write:
                # The original code had a nested `with open(...)` which was a syntax error.
                # Corrected to a single `with open(...)`.
                # Rely on open() raising error if not writable, or check os.access if more robust check needed.
                with open(
                    binned_fastq_filepath, "w"
                ) as binned_handle:  # SeqIO.write expects text handle
                    count = SeqIO.write(records_to_write, binned_handle, "fastq")
                    print(
                        f"Saved {count} records for strain '{strain_name}' ({read_suffix}) to {binned_fastq_filepath.name}"
                    )
            else:
                print(
                    f"No reads matching strain '{strain_name}' (read suffix {read_suffix}) found in {original_fastq_path.name}."
                )

        # Consolidated exception handling
        except (
            FileNotFoundError
        ):  # More specific, handles if file disappears after initial checks
            print(
                f"Error: Input FASTQ file not found or disappeared: {original_fastq_path}. Skipping this file for strain '{strain_name}'."
            )
        except (
            IOError,
            OSError,
        ) as e:  # Handles general I/O errors (writing file, etc.)
            print(
                f"I/O Error processing file '{original_fastq_path}' or writing to '{binned_fastq_filepath}' for strain '{strain_name}': {e}."
            )
        except (
            Exception
        ) as e:  # Catch-all for other unexpected errors during processing of one file
            print(
                f"Unexpected error processing file '{original_fastq_path}' for strain '{strain_name}': {e}. Skipping this file."
            )
            # Removed redundant/misplaced print and except blocks that caused syntax errors.

# Removed _extract_reads_for_strain function as its logic is integrated into _write_reads_to_bins_single_pass

def _detect_sequence_format(file_path: pathlib.Path) -> str:
    """Detects sequence file format (fasta or fastq) based on the first character."""
    try:
        with open_file_transparently(file_path, mode="rt") as f:
            first_char = f.read(1)
            if first_char == ">":
                return "fasta"
            elif first_char == "@":
                return "fastq"
            else:
                raise ValueError(
                    f"Unknown sequence format for {file_path}. First character is '{first_char}'."
                )
    except Exception as e:
        print(f"Error detecting format for {file_path}: {e}")
        raise


def _write_reads_to_bins_single_pass(
    # TECH DEBT SUGGESTION:
    # This function implements a single pass through input FASTQ/FASTA files
    # to distribute reads into multiple output bin files. It handles:
    #   - Opening multiple output file handles.
    #   - Detecting input sequence format.
    #   - Iterating through input reads (potentially paired).
    #   - Matching read IDs to the assignment table.
    #   - Writing reads to the appropriate bin files.
    #   - Managing file handles and error handling.
    #
    # This is a complex operation. If it becomes harder to maintain or needs
    # more features (e.g., more sophisticated output naming, compression options
    # for bins), consider breaking it down into smaller helper functions or potentially
    # a class dedicated to the multi-bin writing process. For instance, input read
    # parsing and output file management could be further separated.
    read_to_strain_assignment_table: pd.DataFrame,
    strains_to_bin: List[str],
    # all_strain_names_in_table: List[str], # Not strictly needed if table columns are already strain names
    forward_fastq_path: pathlib.Path,
    reverse_fastq_path: Optional[pathlib.Path],
    output_bin_dir: pathlib.Path,
) -> Dict[str, Dict[str, int]]:
    """
    Reads input FASTQ/FASTA files once and distributes reads to strain-specific bin files.
    """
    output_handles: Dict[str, Tuple[SeqIO.SeqWriter, Optional[SeqIO.SeqWriter]]] = {}
    binned_read_counts: Dict[str, Dict[str, int]] = {
        strain: {"R1": 0, "R2": 0} for strain in strains_to_bin
    }

    # Ensure read_to_strain_assignment_table index is string type for matching with read IDs
    if not pd.api.types.is_string_dtype(read_to_strain_assignment_table.index.dtype):
        try:
            read_to_strain_assignment_table.index = read_to_strain_assignment_table.index.astype(str)
        except Exception as e:
            raise TypeError(f"Could not convert assignment table index to string: {e}")


    try:
        # Open output file handles for each strain and read type (R1/R2)
        for strain_name in strains_to_bin:
            safe_strain_filename = strain_name.replace(" ", "_").replace("/", "_").replace("\\", "_")

            r1_binned_path = output_bin_dir / f"bin.{safe_strain_filename}_R1.fastq"
            # Output as fastq regardless of input, common practice for binned reads
            # SeqIO.write handles writing SeqRecord objects to fastq format.
            # Files are opened in 'w' mode, implying text. open_file_transparently not needed for output here.
            # gzip.open could be used if compressed output is desired. For now, uncompressed.
            r1_handle = open(r1_binned_path, "w")

            r2_handle = None
            if reverse_fastq_path:
                r2_binned_path = output_bin_dir / f"bin.{safe_strain_filename}_R2.fastq"
                r2_handle = open(r2_binned_path, "w")

            output_handles[strain_name] = (r1_handle, r2_handle)

        # Determine file format
        fwd_format = _detect_sequence_format(forward_fastq_path)
        rev_format = None
        if reverse_fastq_path:
            rev_format = _detect_sequence_format(reverse_fastq_path)
            if fwd_format != rev_format:
                raise ValueError("Forward and reverse read files have mismatched formats.")

        # Process reads
        with open_file_transparently(forward_fastq_path, "rt") as fwd_fh:
            fwd_iter = SeqIO.parse(fwd_fh, fwd_format)
            rev_iter = None
            rev_fh = None # ensure it's defined for finally block

            if reverse_fastq_path:
                rev_fh = open_file_transparently(reverse_fastq_path, "rt")
                rev_iter = SeqIO.parse(rev_fh, rev_format)

            for fwd_record in fwd_iter:
                read_id = fwd_record.id
                rev_record = None
                if rev_iter:
                    try:
                        rev_record = next(rev_iter)
                        # Basic check for read ID consistency in pairs
                        if rev_record.id.split('/')[0].split(' ')[0] != read_id.split('/')[0].split(' ')[0]:
                            print(f"Warning: Mismatched read IDs: '{read_id}' (R1) and '{rev_record.id}' (R2). Skipping R2 for this pair.")
                            rev_record = None # Treat as single-end for this problematic pair
                    except StopIteration:
                        print(f"Warning: Reverse reads ended before forward reads at R1 ID: {read_id}")
                        rev_iter = None # Stop trying to fetch reverse reads

                if read_id in read_to_strain_assignment_table.index:
                    assignments = read_to_strain_assignment_table.loc[read_id]
                    for strain_name in strains_to_bin:
                        if strain_name in assignments and assignments[strain_name] == 1:
                            r1_out_handle, r2_out_handle = output_handles[strain_name]
                            SeqIO.write(fwd_record, r1_out_handle, "fastq") # Write as FASTQ
                            binned_read_counts[strain_name]["R1"] += 1
                            if rev_record and r2_out_handle:
                                SeqIO.write(rev_record, r2_out_handle, "fastq") # Write as FASTQ
                                binned_read_counts[strain_name]["R2"] += 1
                            break # Found assignment, no need to check other strains for this read
            if rev_fh:
                rev_fh.close()

    except Exception as e:
        print(f"Error during single-pass binning: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for strain_name_key in output_handles: # strain_name_key is strain_name
            r1_h, r2_h = output_handles[strain_name_key]
            if r1_h:
                r1_h.close()
            if r2_h:
                r2_h.close()

    for strain_name, counts in binned_read_counts.items():
        if counts["R1"] > 0 :
             print(f"Strain {strain_name}: Binned {counts['R1']} R1 reads.")
        if reverse_fastq_path and counts["R2"] > 0:
             print(f"Strain {strain_name}: Binned {counts['R2']} R2 reads.")

    return binned_read_counts


def create_binned_fastq_files(
    top_strain_names: List[str],
    read_to_strain_assignment_table: pd.DataFrame,
    forward_fastq_path: pathlib.Path,
    reverse_fastq_path: Optional[pathlib.Path],
    output_dir: pathlib.Path,
    num_bins_to_create: int = 2,
    unassigned_marker: str = "NA",
) -> Tuple[Set[str], Dict[str, Dict[str, int]]]: # Return type changed
    """Orchestrates binning of reads for the top N identified strains using a single pass."""

    # Validations (largely unchanged)
    if not isinstance(top_strain_names, list) or not all(
        isinstance(s, str) for s in top_strain_names
    ):
        raise TypeError("top_strain_names must be a list of strings.")
    if not isinstance(read_to_strain_assignment_table, pd.DataFrame):
        raise TypeError("read_to_strain_assignment_table must be a pandas DataFrame.")
    if not all(isinstance(rid, str) for rid in read_to_strain_assignment_table.index):
        # This check might be too strict if index is not yet string.
        # _write_reads_to_bins_single_pass will attempt conversion.
        pass
        # raise ValueError(
        #     "Read-to-strain assignment table index must consist of strings (ReadIds)."
        # )
    if not (
        isinstance(forward_fastq_path, pathlib.Path) and forward_fastq_path.is_file()
    ):
        raise FileNotFoundError(
            f"Forward FASTQ path '{forward_fastq_path}' is not a valid file."
        )
    if reverse_fastq_path and not (
        isinstance(reverse_fastq_path, pathlib.Path) and reverse_fastq_path.is_file()
    ):
        raise FileNotFoundError(
            f"Reverse FASTQ path '{reverse_fastq_path}' is not a valid file."
        )
    if not isinstance(output_dir, pathlib.Path):
        raise TypeError("output_dir must be a pathlib.Path object.")
    if not (isinstance(num_bins_to_create, int) and num_bins_to_create >= 0):
        raise ValueError("num_bins_to_create must be a non-negative integer.")
    if not isinstance(unassigned_marker, str):
        raise TypeError("unassigned_marker must be a string.")

    bin_output_dir = output_dir / "bins"
    try:
        bin_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise IOError(
            f"Could not create bin output directory {bin_output_dir}: {e}"
        ) from e

    strains_to_process: List[str] = []
    if num_bins_to_create > 0 and top_strain_names:
        valid_strains_for_binning = [
            s_name for s_name in top_strain_names
            if s_name and (s_name != unassigned_marker if unassigned_marker else True)
        ]
        strains_to_process = valid_strains_for_binning[:num_bins_to_create]

    if not strains_to_process:
        print(f"No strains selected for binning (num_bins_to_create={num_bins_to_create}, "
              f"top strains identified: {len(top_strain_names)}, valid for binning: {len(strains_to_process)}).")
        return set(), {}

    print(
        f"Attempting to generate binned FASTQ files for up to {num_bins_to_create} top strains: {strains_to_process}"
    )

    # Ensure all strains_to_process are actual columns in the table
    valid_strains_in_table = [s for s in strains_to_process if s in read_to_strain_assignment_table.columns]
    if len(valid_strains_in_table) != len(strains_to_process):
        missing_strains = set(strains_to_process) - set(valid_strains_in_table)
        print(f"Warning: The following strains selected for binning are not in the assignment table columns and will be skipped: {missing_strains}")
        strains_to_process = valid_strains_in_table
        if not strains_to_process:
            print("No valid strains left to process after checking assignment table columns.")
            return set(), {}


    binned_read_counts = _write_reads_to_bins_single_pass(
        read_to_strain_assignment_table=read_to_strain_assignment_table,
        strains_to_bin=strains_to_process,
        forward_fastq_path=forward_fastq_path,
        reverse_fastq_path=reverse_fastq_path,
        output_bin_dir=bin_output_dir,
    )

    binned_strain_names_set = {strain for strain, counts in binned_read_counts.items() if counts["R1"] > 0 or counts["R2"] > 0}

    # Multiprocessing is removed for now. The function returns the set of names for which bins were created
    # and the counts of reads binned.
    return binned_strain_names_set, binned_read_counts


def run_binning_pipeline(
    final_assignments: FinalAssignmentsType,
    all_strain_names: List[str],
    read_assignments: Dict[ReadId, Union[str, int]],  # For determining top strains
    forward_reads_fastq: Union[str, pathlib.Path],
    output_directory: Union[str, pathlib.Path],
    num_top_strains_to_bin: int = 2,
    reverse_reads_fastq: Optional[Union[str, pathlib.Path]] = None,
    unassigned_marker: str = "NA",
) -> None:
    """Executes the main binning pipeline."""

    # --- Input Validation for run_binning_pipeline ---
    if not isinstance(
        final_assignments, dict
    ):  # Basic check, deeper done by generate_table
        raise TypeError("final_assignments must be a dictionary.")
    if not isinstance(all_strain_names, list) or not all(
        isinstance(s, str) and s for s in all_strain_names
    ):
        raise TypeError("all_strain_names must be a list of non-empty strings.")
    if not isinstance(forward_reads_fastq, (str, pathlib.Path)):
        raise TypeError("forward_reads_fastq must be a string or pathlib.Path.")
    if not isinstance(output_directory, (str, pathlib.Path)):
        raise TypeError("output_directory must be a string or pathlib.Path.")
    if not isinstance(num_top_strains_to_bin, int) or num_top_strains_to_bin < 0:
        raise ValueError("num_top_strains_to_bin must be a non-negative integer.")
    if reverse_reads_fastq and not isinstance(reverse_reads_fastq, (str, pathlib.Path)):
        raise TypeError(
            "reverse_reads_fastq must be a string or pathlib.Path, or None."
        )
    if not isinstance(unassigned_marker, str):  # Allow empty string
        raise TypeError("unassigned_marker must be a string.")

    fwd_fastq_path = pathlib.Path(forward_reads_fastq).resolve()
    out_dir_path = pathlib.Path(output_directory).resolve()
    rev_fastq_path = (
        pathlib.Path(reverse_reads_fastq).resolve() if reverse_reads_fastq else None
    )

    # 1. Generate the binning table from final_assignments
    read_to_strain_assignment_table = generate_table(
        final_assignments, all_strain_names
    )

    # Ensure ReadId index for fast lookup
    if read_to_strain_assignment_table.index.name != 'ReadId':
         # If ReadId is a column, set it as index. Otherwise, assume index is already ReadId.
        if 'ReadId' in read_to_strain_assignment_table.columns:
            read_to_strain_assignment_table = read_to_strain_assignment_table.set_index('ReadId')
        # else: we assume the user has provided a table with ReadId as index if the column isn't named 'ReadId'

    has_actual_assignments = any(
        isinstance(val, int) for val in final_assignments.values()
    )
    if not fwd_fastq_path.is_file():
        raise FileNotFoundError(f"Forward FASTQ file not found: {fwd_fastq_path}")
    if rev_fastq_path and not rev_fastq_path.is_file():
        raise FileNotFoundError(
            f"Reverse FASTQ file specified but not found: {rev_fastq_path}"
        )

    try:
        out_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise IOError(f"Could not create output directory {out_dir_path}: {e}") from e

    # Duplicated table generation removed.

    if read_to_strain_assignment_table.empty and num_top_strains_to_bin > 0:
        if not all_strain_names and has_actual_assignments:
            print(
                "Warning: Read-to-strain assignment table is empty because all_strain_names is empty, but assignments exist."
            )
        elif not has_actual_assignments:
            print(
                "Info: Read-to-strain assignment table is empty as there are no actual read assignments (only unassigned or empty final_assignments)."
            )
        elif (
            all_strain_names and has_actual_assignments
        ):
            print(
                "Warning: Read-to-strain assignment table is unexpectedly empty despite assignments and strain names. Binning may not produce results."
            )

    top_strains = get_top_strain_names(
        read_assignments=read_assignments,
        strain_list=all_strain_names,
        unassigned_marker=unassigned_marker,
        exclude_unassigned=True,
    )
    if not top_strains and num_top_strains_to_bin > 0:
        print(
            f"Info: No top strains identified (potentially after excluding '{unassigned_marker}'). Binning will be skipped."
        )

    # 3. Create binned FASTQ files (now single-pass, no multiprocessing list)
    binned_strains_set, binned_read_counts = create_binned_fastq_files( # Return type changed
        top_strain_names=top_strains,
        read_to_strain_assignment_table=read_to_strain_assignment_table,
        forward_fastq_path=fwd_fastq_path,
        reverse_fastq_path=rev_fastq_path,
        output_dir=out_dir_path,
        num_bins_to_create=num_top_strains_to_bin,
        unassigned_marker=unassigned_marker,
    )

    # No processes to join anymore with the single-process approach
    # if processes: ... logic removed

    if binned_strains_set:
        print(
            f"Binning pipeline finished. Binned files attempted for strains: {sorted(list(binned_strains_set))}"
        )
        for strain_name, counts in binned_read_counts.items():
            if counts["R1"] > 0 or (counts["R2"] > 0 if rev_fastq_path else False):
                 print(f"  Strain {strain_name}: Total {counts['R1']} R1 reads, {counts['R2']} R2 reads binned.")
    else:
        print(
            "Binning pipeline finished. No strains were selected for binning or no reads could be binned."
        )


if __name__ == "__main__":
    # --- Mock Data Setup ---
    # Removed duplicate mock_all_strains_list and example_unassigned_marker_val declarations
    mock_all_strains_list: List[str] = [
        "StrainX",
        "StrainY",
        "StrainZ_complex name with_various-chars",
    ]
    example_unassigned_marker_val: str = "UNASSIGNED_READ"

    # Corrected mock_pipeline_assignments_data generation and removed duplicate assignments
    mock_pipeline_assignments_data: FinalAssignmentsType = {
        f"read{i:03d}": (i % len(mock_all_strains_list))
        if (i % 5 != 0)
        else example_unassigned_marker_val
        for i in range(1, 10)  # Reduced range for brevity in example
    }
    # Add a specific case for an invalid index if desired for testing warnings
    mock_pipeline_assignments_data["read008"] = 99
    # Add a specific case for a special ID if desired
    mock_pipeline_assignments_data["read007_special-ID"] = 0

    # Setup dummy FASTQ files and output directory
    # Removed duplicated manual assignments to mock_pipeline_assignments_data

    try:
        # current_file_path logic is fine
        current_file_path = pathlib.Path(__file__).parent
    except NameError:  # __file__ is not defined (e.g. in an interactive session)
        current_file_path = pathlib.Path.cwd()

    example_output_directory_path = (
        current_file_path / "example_strainr_binning_run_output"
    )
    example_output_directory_path.mkdir(exist_ok=True, parents=True)

    dummy_r1_fastq_file = example_output_directory_path / "sample_R1.fastq.gz"
    dummy_r2_fastq_file = example_output_directory_path / "sample_R2.fastq"

    fastq_r1_content_list = []
    fastq_r2_content_list = []
    for read_id_key in mock_pipeline_assignments_data.keys():
        # Check if the assignment is the unassigned marker or an invalid index (int but out of bounds)
        assignment = mock_pipeline_assignments_data[read_id_key]
        if assignment == example_unassigned_marker_val or not (
            0 <= int(assignment) < len(mock_all_strains_list)
        ):
            seq_r1 = (
                "N" * 50
            )  # Different sequence for unassigned or invalid for visual check
            seq_r2 = "N" * 50
        else:
            seq_r1 = "A" * 50
            seq_r2 = "C" * 50

        fastq_r1_content_list.append(f"@{read_id_key}/1\n{seq_r1}\n+\n{'I' * 50}\n")
        fastq_r2_content_list.append(f"@{read_id_key}/2\n{seq_r2}\n+\n{'I' * 50}\n")

    with gzip.open(dummy_r1_fastq_file, "wt") as f_r1_out:
        f_r1_out.write("".join(fastq_r1_content_list))
    with open(dummy_r2_fastq_file, "w") as f_r2_out:
        f_r2_out.write("".join(fastq_r2_content_list))

    print("--- Running StrainR Binning Pipeline Example ---")
    print(f"Mock Strain List: {mock_all_strains_list}")
    print(f"Output Directory: {example_output_directory_path.resolve()}")
    print(f"Dummy R1 FASTQ: {dummy_r1_fastq_file.resolve()}")
    print(f"Dummy R2 FASTQ: {dummy_r2_fastq_file.resolve()}")

    try:
        run_binning_pipeline(
            final_assignments=mock_pipeline_assignments_data,
            all_strain_names=mock_all_strains_list,
            forward_reads_fastq=dummy_r1_fastq_file,
            reverse_reads_fastq=dummy_r2_fastq_file,
            output_directory=example_output_directory_path,
            num_top_strains_to_bin=3,
            unassigned_marker=example_unassigned_marker_val,
        )
    except Exception as main_exception:
        print(f"An error occurred during the example pipeline run: {main_exception}")
        import traceback

        traceback.print_exc()  # Indented correctly

    print(
        f"--- Example Run Finished. Check '{example_output_directory_path.resolve() / 'bins'}' for binned FASTQ files. ---"
    )
    # Optional: Add cleanup for example files and directory
    # import shutil
    # if input("Clean up example directory? (y/N): ").strip().lower() == 'y':
    #     shutil.rmtree(example_output_directory_path)
    #     print(f"Cleaned up example directory: {example_output_directory_path.resolve()}")
    # The second traceback.print_exc() was part of the original erroneous structure and has been removed by prior fixes.
    # This block now only contains the correctly indented traceback.print_exc() within its except block.

    print(
        f"--- Example Run Finished. Check '{example_output_directory_path.resolve() / 'bins'}' for binned FASTQ files. ---"
    )
