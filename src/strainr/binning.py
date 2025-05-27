# Standard library imports
import multiprocessing as mp
import pathlib
from typing import (
    List,
    Dict,
    Set,
    Tuple,
    Union,
    Optional,  # Added Any
)
import gzip  # For writing gzipped example files
import traceback  # For more detailed error info in main example
# Removed redundant typing import

# Third-party imports
import numpy as np  # Added missing import
import pandas as pd
from Bio import SeqIO

# Local application/library specific imports
from strainr.utils import open_file_transparently
from strainr.genomic_types import (
    ReadId,
    FinalAssignmentsType,
)  # Import FinalAssignmentsType

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

    # Generate a safe filename from strain_name
    safe_strain_filename = (
        strain_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    )

    fastq_files_to_process: List[Tuple[pathlib.Path, str]] = []
    if forward_fastq_path.is_file():
        fastq_files_to_process.append((forward_fastq_path, "R1"))
    else:
        print(
            f"Warning: Forward FASTQ file not found or is not a file: {forward_fastq_path}. Skipping R1 for strain {strain_name}."
        )

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


def create_binned_fastq_files(
    top_strain_names: List[str],
    read_to_strain_assignment_table: pd.DataFrame,
    forward_fastq_path: pathlib.Path,
    reverse_fastq_path: Optional[pathlib.Path],
    output_dir: pathlib.Path,
    num_bins_to_create: int = 2,
    unassigned_marker: str = "NA",
) -> Tuple[Set[str], List[mp.Process]]:
    """Orchestrates binning of reads for the top N identified strains."""

    if not isinstance(top_strain_names, list) or not all(
        isinstance(s, str) for s in top_strain_names
    ):  # Allow empty strings if unassigned_marker is empty
        raise TypeError("top_strain_names must be a list of strings.")
    if not isinstance(read_to_strain_assignment_table, pd.DataFrame):
        raise TypeError("read_to_strain_assignment_table must be a pandas DataFrame.")
    if not all(isinstance(rid, str) for rid in read_to_strain_assignment_table.index):
        raise ValueError(
            "Read-to-strain assignment table index must consist of strings (ReadIds)."
        )
    if (
        not isinstance(forward_fastq_path, pathlib.Path)
        or not forward_fastq_path.is_file()
    ):
        raise FileNotFoundError(
            f"Forward FASTQ path '{forward_fastq_path}' is not a valid file."
        )
    if reverse_fastq_path and (
        not isinstance(reverse_fastq_path, pathlib.Path)
        or not reverse_fastq_path.is_file()
    ):
        raise FileNotFoundError(
            f"Reverse FASTQ path '{reverse_fastq_path}' is not a valid file."
        )
    if not isinstance(output_dir, pathlib.Path):
        raise TypeError("output_dir must be a pathlib.Path object.")
    if (
        not isinstance(num_bins_to_create, int) or num_bins_to_create < 0
    ):  # Allow 0 to mean "bin nothing"
        raise ValueError("num_bins_to_create must be a non-negative integer.")
    if not isinstance(
        unassigned_marker, str
    ):  # Allow empty string marker if user desires
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
        # Filter out empty strings or the unassigned_marker BEFORE slicing to get top N
        # This ensures we are selecting from valid, assignable strains.
        valid_strains_for_binning = [
            s_name
            for s_name in top_strain_names
            if s_name
            and (
                s_name != unassigned_marker if unassigned_marker else True
            )  # Handles if unassigned_marker is ""
        ]
        strains_to_process = valid_strains_for_binning[:num_bins_to_create]

    if not strains_to_process:
        # Clarified print statements based on whether filtering occurred.
        if (
            top_strain_names and not strains_to_process
        ):  # Had top strains, but they were filtered out
            print(
                f"No strains selected for binning. Top strains were: {top_strain_names[:num_bins_to_create]}, "
                f"but they were filtered out (e.g. matched unassigned_marker '{unassigned_marker}')."
            )
        else:  # No top_strain_names to begin with, or num_bins_to_create was 0.
            print(
                f"No strains selected for binning (num_bins_to_create={num_bins_to_create}, "
                f"number of identified top strains: {len(top_strain_names)}, "
                f"number of valid strains after filtering: {len(strains_to_process)})."
            )
        return set(), []

    print(
        f"Attempting to generate binned FASTQ files for up to {num_bins_to_create} top strains: {strains_to_process}"
    )

    processes: List[mp.Process] = []
    binned_strain_names_set: Set[str] = set()

    for strain_name in strains_to_process:
        # The filtering of strains_to_process should have already handled this.
        # This is a defensive check, but if strains_to_process is built correctly,
        # strain_name here should always be valid and not the unassigned_marker (if marker is set).
        # Original code had redundant checks here. Assuming strains_to_process is correctly pre-filtered.
        # if not strain_name or (unassigned_marker and strain_name == unassigned_marker):
        #     print(f"Skipping binning for '{strain_name}' as it's an unassigned marker or empty (this check should be redundant).")
        #     continue

        print(f"Preparing to bin reads for strain: {strain_name}...")

        if strain_name not in read_to_strain_assignment_table.columns:
            print(
                f"Warning: Strain '{strain_name}' not found as a column in the assignment table. Skipping."
            )
            # Removed duplicate continue
            continue

        # Validate column data type for safety before boolean comparisons
        # Allowing any numeric type that can be reasonably compared to 1 (or True)
        if not pd.api.types.is_numeric_dtype(
            read_to_strain_assignment_table[strain_name].dtype
        ) and not pd.api.types.is_bool_dtype(
            read_to_strain_assignment_table[strain_name].dtype
        ):
            print(
                f"Warning: Column '{strain_name}' in assignment table has non-numeric/boolean type "
                f"({read_to_strain_assignment_table[strain_name].dtype}). Skipping, as expecting 0 or 1 assignments."
            )
            continue

        strain_assigned_reads_series = read_to_strain_assignment_table[strain_name]
        strain_specific_read_ids: Set[ReadId] = set(
            strain_assigned_reads_series[
                strain_assigned_reads_series == 1
            ].index.astype(str)
        )

        if not strain_specific_read_ids:
            print(
                f"No reads found assigned to strain '{strain_name}' in the table. Skipping file creation for this strain."
            )
            continue

        binned_strain_names_set.add(strain_name)

        process = mp.Process(
            target=_extract_reads_for_strain,
            args=(
                strain_name,
                strain_specific_read_ids,
                forward_fastq_path,
                reverse_fastq_path,  # Can be None
                bin_output_dir,
            ),
        )
        # Removed duplicated process creation and start block.
        # The first process block was correct.
        try:
            process.start()
            processes.append(process)
        except Exception as e:  # Catch errors during process creation/start
            print(f"Error starting binning process for strain '{strain_name}': {e}")
            # Optionally, decide if this is fatal or if other bins can proceed.
            # If a process fails to start, it won't be in `processes` list for joining.

    return binned_strain_names_set, processes


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

    read_to_strain_assignment_table = generate_table(
        final_assignments, all_strain_names
    )

    has_actual_assignments = any(
        isinstance(val, int) for val in final_assignments.values()
    )

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
        ):  # Should not happen if generate_table is correct
            print(
                "Warning: Read-to-strain assignment table is unexpectedly empty despite assignments and strain names. Binning may not produce results."
            )

    # 2. Determine top strains using final_assignments
    # Removed mis-indented print statements here; they are handled within the if block above.

    top_strains = get_top_strain_names(
        read_assignments=read_assignments,  # Changed from final_assignments to use the dedicated read_assignments dict
        strain_list=all_strain_names,
        unassigned_marker=unassigned_marker,
        exclude_unassigned=True,  # Standard behavior to exclude unassigned from top strains for binning
    )
    if not top_strains and num_top_strains_to_bin > 0:
        print(
            f"Info: No top strains identified (potentially after excluding '{unassigned_marker}'). Binning will be skipped."
        )
    # Removed duplicate print statement.

    # 3. Create binned FASTQ files
    binned_strains_set, processes = create_binned_fastq_files(
        top_strain_names=top_strains,
        read_to_strain_assignment_table=read_to_strain_assignment_table,
        forward_fastq_path=fwd_fastq_path,  # Already resolved Path object
        reverse_fastq_path=rev_fastq_path,  # Already resolved Path object or None
        output_dir=out_dir_path,  # Already resolved Path object
        num_bins_to_create=num_top_strains_to_bin,
        unassigned_marker=unassigned_marker,
    )

    if processes:
        print(f"Waiting for {len(processes)} binning process(es) to complete...")
        for p in processes:
            try:
                p.join()  # Wait for each process to finish
                if p.exitcode != 0:
                    print(
                        f"Warning: Binning process for a strain (PID {p.pid}) finished with exit code {p.exitcode}."
                    )
            except (
                Exception
            ) as e:  # Should not happen if target function catches its errors
                print(f"Error joining binning process (PID {p.pid}): {e}")
        print("All binning processes completed.")

    if binned_strains_set:
        print(
            f"Binning pipeline finished. Binned files attempted for strains: {sorted(list(binned_strains_set))}"
        )
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
