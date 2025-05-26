# Standard library imports
import multiprocessing as mp
import pathlib
from typing import (
    List,
    Dict,
    Set,
    Tuple,
    Union,
    Optional,
    Any,
)  # Added Any for undefined generate_table output
import gzip # For writing gzipped example files
import traceback # For more detailed error info in main example
from typing import List, Dict, Set, Tuple, Union, Optional 

# Third-party imports
import pandas as pd
from Bio import SeqIO

# Local application/library specific imports
from strainr.utils import open_file_transparently
from strainr.genomic_types import ReadId, StrainIndex, FinalAssignmentsType # Import FinalAssignmentsType

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
            raise ValueError("All ReadId keys in final_assignments must be non-empty strings.")
        if not (isinstance(assignment, int) and assignment >= 0) and not isinstance(assignment, str):
            raise TypeError(
                f"Assignment for ReadId '{read_id}' must be a non-negative integer (StrainIndex) "
                f"or a string (unassigned marker), got {type(assignment)}."
            )

    if not isinstance(all_strain_names, list) or \
       not all(isinstance(s, str) and s for s in all_strain_names):
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
        if any(isinstance(val, int) for val in final_assignments.values()):
            raise ValueError(
                "all_strain_names is empty, but final_assignments contains integer (StrainIndex) "
                "assignments. Strain names are required to interpret indices."
            )
        print("Warning: all_strain_names is empty. Returning DataFrame with no columns.")
        return pd.DataFrame(index=read_ids)

    df = pd.DataFrame(0, index=read_ids, columns=all_strain_names, dtype=np.uint8)

    for read_id, assignment in final_assignments.items():
        if isinstance(assignment, int):  # StrainIndex
            if 0 <= assignment < len(all_strain_names):
                strain_name = all_strain_names[assignment]
                df.loc[read_id, strain_name] = 1
            else:
                print(
                    f"Warning: Invalid StrainIndex {assignment} for read_id {read_id}. Max index is {len(all_strain_names)-1}. Read will not be assigned to any strain in table."
                )
                print(f"Warning: Invalid StrainIndex {assignment} for read_id '{read_id}'. Max index is {len(all_strain_names)-1}. Read will not be assigned in table.")
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

    if not strain_list and any(
        isinstance(val, int) for val in read_assignments.values()
    ):
        print(
            "Warning: get_top_strain_names received an empty strain_list but read_assignments contain integer indices. Cannot map indices to names. Returning empty list."
        )
    # Basic validation for read_assignments content (more detailed in generate_table if used prior)
    for read_id, assignment in read_assignments.items():
        if not isinstance(read_id, str) or not read_id:
            raise ValueError("All ReadId keys in read_assignments must be non-empty strings.")
        if not (isinstance(assignment, int) and assignment >= 0) and not isinstance(assignment, str) :
            raise TypeError(f"Assignment for ReadId '{read_id}' must be a non-negative int or str.")

    if not isinstance(strain_list, list) or not all(isinstance(s, str) and s for s in strain_list):
        raise TypeError("strain_list must be a list of non-empty strings.")
    if len(set(strain_list)) != len(strain_list):
        raise ValueError("strain_list must contain unique names.")
    
    if not isinstance(unassigned_marker, str) or not unassigned_marker:
        raise ValueError("unassigned_marker must be a non-empty string.")

    if not strain_list and any(isinstance(val, int) for val in read_assignments.values()):
        print("Warning: get_top_strain_names received an empty strain_list but read_assignments contain integer indices. Cannot map indices to names. Returning empty list.")
        return []

    assignment_values = list(read_assignments.values())
    counts = pd.Series(assignment_values).value_counts()

    mapped_strain_counts: Dict[str, int] = {}
    for entity, count_val in counts.items():
        if isinstance(entity, int): 
            if 0 <= entity < len(strain_list):
                strain_name = strain_list[entity]
                mapped_strain_counts[strain_name] = (
                    mapped_strain_counts.get(strain_name, 0) + count_val
                )
            # else: Invalid StrainIndex, simply ignore or log
        elif isinstance(entity, str):  # String marker
                mapped_strain_counts[strain_name] = mapped_strain_counts.get(strain_name, 0) + count_val
            # else: Invalid StrainIndex, ignored for ranking
        elif isinstance(entity, str):
            if entity == unassigned_marker and exclude_unassigned:
                continue
            # If the string entity itself is a valid strain name (and not the unassigned marker)
            if entity in strain_list and entity != unassigned_marker:
                mapped_strain_counts[entity] = (
                    mapped_strain_counts.get(entity, 0) + count_val
                )
            # Other strings are ignored.
            if entity in strain_list: # Could be a pre-assigned name or the unassigned_marker if it's in strain_list
                 mapped_strain_counts[entity] = mapped_strain_counts.get(entity, 0) + count_val
            elif entity == unassigned_marker: # If unassigned_marker is not in strain_list and not excluded
                 mapped_strain_counts[entity] = mapped_strain_counts.get(entity, 0) + count_val


    # Sort strains by count in descending order
    sorted_strains = sorted(
        mapped_strain_counts.items(), key=lambda item: item[1], reverse=True
    )

    return [strain_name for strain_name, count_val in sorted_strains]
    sorted_strains = sorted(mapped_strain_counts.items(), key=lambda item: item[1], reverse=True)
    # Filter out the unassigned_marker from the final list if it was counted
    return [strain_name for strain_name, _ in sorted_strains if (strain_name != unassigned_marker or not exclude_unassigned)]


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

    """Extracts reads for a specific strain to new FASTQ files."""
    if not isinstance(strain_name, str) or not strain_name:
        raise ValueError("strain_name must be a non-empty string.")
    if not isinstance(read_ids_for_strain, set) or not all(isinstance(rid, str) and rid for rid in read_ids_for_strain):
        raise TypeError("read_ids_for_strain must be a set of non-empty strings.")
    if not isinstance(forward_fastq_path, pathlib.Path):
        raise TypeError("forward_fastq_path must be a pathlib.Path object.")
    if reverse_fastq_path and not isinstance(reverse_fastq_path, pathlib.Path):
        raise TypeError("reverse_fastq_path must be a pathlib.Path object or None.")
    if not isinstance(output_bin_dir, pathlib.Path):
        raise TypeError("output_bin_dir must be a pathlib.Path object.")

    if not forward_fastq_path.is_file():
        raise FileNotFoundError(f"Forward FASTQ file not found: {forward_fastq_path}")
    if reverse_fastq_path and not reverse_fastq_path.is_file():
        raise FileNotFoundError(f"Reverse FASTQ file specified but not found: {reverse_fastq_path}")
    if not output_bin_dir.is_dir(): # Assume it should exist before calling this worker
        # This check might be redundant if create_binned_fastq_files always creates it.
        # However, for robustness of the worker, it's a good check.
        # To be fully robust, it could try to create it, but that might be unexpected for a worker.
        raise FileNotFoundError(f"Output bin directory does not exist or is not a directory: {output_bin_dir}")


    safe_strain_filename = strain_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    fastq_files_to_process: List[Tuple[pathlib.Path, str]] = [(forward_fastq_path, "R1")]
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
        return
        fastq_files_to_process.append((reverse_fastq_path, "R2"))

    for original_fastq_path, read_suffix in fastq_files_to_process:
        binned_fastq_filename = f"bin.{safe_strain_filename}_{read_suffix}.fastq"
        binned_fastq_filepath = output_bin_dir / binned_fastq_filename

        records_to_write = []
        try:
            # Use "rt" for text mode with open_file_transparently, as SeqIO.parse expects text.
            with open_file_transparently(
                original_fastq_path, mode="rt"
            ) as original_handle:
            with open_file_transparently(original_fastq_path, mode="rt") as original_handle:
                for record in SeqIO.parse(original_handle, "fastq"):
                    # Read ID normalization might be needed if FASTQ IDs have suffixes like /1 or /2
                    # and read_ids_for_strain does not.
                    # Example: normalized_id = record.id.split('/')[0].split(' ')[0]
                    if (
                        record.id in read_ids_for_strain
                    ):  # or normalized_id in read_ids_for_strain:
                    if record.id in read_ids_for_strain:
                        records_to_write.append(record)

            if records_to_write:
                with open(
                    binned_fastq_filepath, "w"
                ) as binned_handle:  # SeqIO.write expects text handle
                # Check writability of directory more directly before opening file for write
                # This is a bit more involved, can check os.access(output_bin_dir, os.W_OK)
                # For now, rely on open() raising error if not writable.
                with open(binned_fastq_filepath, "w") as binned_handle:
                    count = SeqIO.write(records_to_write, binned_handle, "fastq")
                print(f"Saved {count} records for strain '{strain_name}' ({read_suffix}) to {binned_fastq_filepath.name}")
            else:
                print(
                    f"No reads matching strain '{strain_name}' (read suffix {read_suffix}) found in {original_fastq_path.name}."
                )

        except FileNotFoundError:
            print(
                f"Error: Input FASTQ file not found: {original_fastq_path}. Skipping this file for strain {strain_name}."
            )
        except Exception as e:
            print(
                f"Error processing file {original_fastq_path} for strain '{strain_name}': {e}. Skipping this file."
            )
                print(f"No reads matching strain '{strain_name}' (read suffix {read_suffix}) found in {original_fastq_path.name}.")
        except FileNotFoundError: # Should have been caught by initial checks, but defensive.
            print(f"Error: Input FASTQ file disappeared: {original_fastq_path}. Skipping.")
        except (IOError, OSError) as e:
            print(f"I/O Error processing file {original_fastq_path} or writing to {binned_fastq_filepath} for strain '{strain_name}': {e}.")
        except Exception as e: 
            print(f"Unexpected error processing file {original_fastq_path} for strain '{strain_name}': {e}. Records may not have been written.")


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

    if not isinstance(top_strain_names, list) or not all(isinstance(s, str) for s in top_strain_names): # Allow empty strings if unassigned_marker is empty
        raise TypeError("top_strain_names must be a list of strings.")
    if not isinstance(read_to_strain_assignment_table, pd.DataFrame):
        raise TypeError("read_to_strain_assignment_table must be a pandas DataFrame.")
    if not all(isinstance(rid, str) for rid in read_to_strain_assignment_table.index):
        raise ValueError("Read-to-strain assignment table index must consist of strings (ReadIds).")
    if not isinstance(forward_fastq_path, pathlib.Path) or not forward_fastq_path.is_file():
        raise FileNotFoundError(f"Forward FASTQ path '{forward_fastq_path}' is not a valid file.")
    if reverse_fastq_path and (not isinstance(reverse_fastq_path, pathlib.Path) or not reverse_fastq_path.is_file()):
        raise FileNotFoundError(f"Reverse FASTQ path '{reverse_fastq_path}' is not a valid file.")
    if not isinstance(output_dir, pathlib.Path):
        raise TypeError("output_dir must be a pathlib.Path object.")
    if not isinstance(num_bins_to_create, int) or num_bins_to_create < 0: # Allow 0 to mean "bin nothing"
        raise ValueError("num_bins_to_create must be a non-negative integer.")
    if not isinstance(unassigned_marker, str): # Allow empty string marker if user desires
        raise TypeError("unassigned_marker must be a string.")

    bin_output_dir = output_dir / "bins"
    try:
        bin_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise IOError(f"Could not create bin output directory {bin_output_dir}: {e}") from e

    strains_to_process = []
    if num_bins_to_create > 0 and top_strain_names:
        strains_to_process = top_strain_names[:num_bins_to_create]

        strains_to_process = [s for s in top_strain_names if s and s != unassigned_marker][:num_bins_to_create]
    
    if not strains_to_process:
        print(
            f"No strains selected for binning (num_bins_to_create={num_bins_to_create}, top_strain_names has {len(top_strain_names)} entries)."
        )
        print(f"No strains selected for binning (num_bins_to_create={num_bins_to_create}, filtered top_strain_names has {len(strains_to_process)} entries).")
        return set(), []

    print(
        f"Attempting to generate binned FASTQ files for up to {num_bins_to_create} top strains: {strains_to_process}"
    )

    processes: List[mp.Process] = []
    binned_strain_names_set: Set[str] = set()

    for strain_name in strains_to_process:
        if strain_name == unassigned_marker or not strain_name:
            print(
                f"Skipping binning for '{strain_name}' as it matches unassigned marker or is empty."
            )
        # Redundant check if strains_to_process is already filtered, but safe.
        if not strain_name or (strain_name == unassigned_marker and unassigned_marker): # Ensure unassigned_marker itself is not empty if used in check
            print(f"Skipping binning for '{strain_name}' as it's an unassigned marker or empty.")
            continue

        print(f"Preparing to bin reads for strain: {strain_name}...")

        if strain_name not in read_to_strain_assignment_table.columns:
            print(
                f"Warning: Strain '{strain_name}' not found as a column in the assignment table. Skipping."
            )
            continue

            print(f"Warning: Strain '{strain_name}' not found as a column in the assignment table. Skipping.")
            continue
        
        if read_to_strain_assignment_table[strain_name].dtype not in [np.uint8, np.int8, np.int16, np.int32, np.int64, int, bool]:
             print(f"Warning: Column '{strain_name}' in assignment table has non-integer/boolean type ({read_to_strain_assignment_table[strain_name].dtype}). Skipping, expecting 0 or 1.")
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
        process.start()
        processes.append(process)
        try:
            process = mp.Process(
                target=_extract_reads_for_strain,
                args=(
                    strain_name, strain_specific_read_ids,
                    forward_fastq_path, reverse_fastq_path, 
                    bin_output_dir,
                ),
            )
            process.start()
            processes.append(process)
        except Exception as e: # Catch errors during process creation/start
            print(f"Error starting binning process for strain '{strain_name}': {e}")
            # Optionally, decide if this is fatal or if other bins can proceed

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
    if not isinstance(final_assignments, dict): # Basic check, deeper done by generate_table
        raise TypeError("final_assignments must be a dictionary.")
    if not isinstance(all_strain_names, list) or not all(isinstance(s, str) and s for s in all_strain_names):
        raise TypeError("all_strain_names must be a list of non-empty strings.")
    if not isinstance(forward_reads_fastq, (str, pathlib.Path)):
        raise TypeError("forward_reads_fastq must be a string or pathlib.Path.")
    if not isinstance(output_directory, (str, pathlib.Path)):
        raise TypeError("output_directory must be a string or pathlib.Path.")
    if not isinstance(num_top_strains_to_bin, int) or num_top_strains_to_bin < 0:
        raise ValueError("num_top_strains_to_bin must be a non-negative integer.")
    if reverse_reads_fastq and not isinstance(reverse_reads_fastq, (str, pathlib.Path)):
        raise TypeError("reverse_reads_fastq must be a string or pathlib.Path, or None.")
    if not isinstance(unassigned_marker, str): # Allow empty string
        raise TypeError("unassigned_marker must be a string.")


    fwd_fastq_path = pathlib.Path(forward_reads_fastq).resolve()
    out_dir_path = pathlib.Path(output_directory).resolve()
    rev_fastq_path = pathlib.Path(reverse_reads_fastq).resolve() if reverse_reads_fastq else None

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
        raise FileNotFoundError(f"Reverse FASTQ file specified but not found: {rev_fastq_path}")
    
    try:
        out_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise IOError(f"Could not create output directory {out_dir_path}: {e}") from e


    read_to_strain_assignment_table = generate_table(final_assignments, all_strain_names)
    
    has_actual_assignments = any(isinstance(val, int) for val in final_assignments.values())
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
             print("Info: Read-to-strain assignment table is empty as there are no actual read assignments.")
        elif all_strain_names and has_actual_assignments: 
             print("Warning: Read-to-strain assignment table is unexpectedly empty despite assignments and strain names. Binning may not produce results.")

    top_strains = get_top_strain_names(
        read_assignments=final_assignments,
        strain_list=all_strain_names,
        unassigned_marker=unassigned_marker,
        exclude_unassigned=True,
    )
    if not top_strains and num_top_strains_to_bin > 0:
        print(
            f"Info: No top strains identified (excluding '{unassigned_marker}'). Binning will be skipped."
        )

    # 3. Create binned FASTQ files
        print(f"Info: No top strains identified (excluding '{unassigned_marker}'). Binning will be skipped.")
    
    binned_strains_set, processes = create_binned_fastq_files(
        top_strain_names=top_strains,
        read_to_strain_assignment_table=read_to_strain_assignment_table,
        forward_fastq_path=fwd_fastq_path, # Already resolved Path object
        reverse_fastq_path=rev_fastq_path, # Already resolved Path object or None
        output_dir=out_dir_path,           # Already resolved Path object
        num_bins_to_create=num_top_strains_to_bin,
        unassigned_marker=unassigned_marker,
    )

    if processes:
        print(f"Waiting for {len(processes)} binning process(es) to complete...")
        for p in processes:
            try:
                p.join() # Wait for each process to finish
                if p.exitcode != 0:
                    print(f"Warning: Binning process for a strain (PID {p.pid}) finished with exit code {p.exitcode}.")
            except Exception as e: # Should not happen if target function catches its errors
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
    mock_all_strains_list: List[str] = [
        "StrainX",
        "StrainY",
        "StrainZ_complex name with_various-chars",
    ]
    example_unassigned_marker_val: str = "UNASSIGNED_READ"
    mock_all_strains_list: List[str] = ["StrainX", "StrainY", "StrainZ_complex name with_various-chars"]
    example_unassigned_marker_val: str = "UNASSIGNED_READ" 

    mock_pipeline_assignments_data: FinalAssignmentsType = {
        f"read{i:03d}": (
            (i % len(mock_all_strains_list))
            if (i % 5 != 0)
            else example_unassigned_marker_val
        )
        for i in range(1, 31)
    }
    mock_pipeline_assignments_data["read001"] = 0  # StrainX
    mock_pipeline_assignments_data["read002"] = 1  # StrainY
    mock_pipeline_assignments_data["read003"] = 2  # StrainZ...
    mock_pipeline_assignments_data["read004"] = 0  # StrainX
    mock_pipeline_assignments_data["read005"] = example_unassigned_marker_val
    mock_pipeline_assignments_data["read006"] = 1  # StrainY
    mock_pipeline_assignments_data["read007_special-ID"] = 0  # StrainX
    mock_pipeline_assignments_data["read008"] = 99  # Invalid index for testing warning

    # Setup dummy FASTQ files and output directory
    mock_pipeline_assignments_data["read001"] = 0 
    mock_pipeline_assignments_data["read002"] = 1 
    mock_pipeline_assignments_data["read003"] = 2 
    mock_pipeline_assignments_data["read004"] = 0 
    mock_pipeline_assignments_data["read005"] = example_unassigned_marker_val 
    mock_pipeline_assignments_data["read006"] = 1 
    mock_pipeline_assignments_data["read007_special-ID"] = 0 
    mock_pipeline_assignments_data["read008"] = 99 

    try:
        current_file_path = pathlib.Path(__file__).parent
    except NameError:  # __file__ is not defined (e.g. in an interactive session)
    except NameError: 
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
        seq_r1 = "A" * 50
        seq_r2 = "C" * 50
        if mock_pipeline_assignments_data[read_id_key] == example_unassigned_marker_val:
            seq_r1 = "N" * 50  # Different sequence for unassigned for visual check
            seq_r2 = "N" * 50

        seq_r1 = 'A' * 50
        seq_r2 = 'C' * 50
        if mock_pipeline_assignments_data[read_id_key] == example_unassigned_marker_val :
            seq_r1 = 'N' * 50 
            seq_r2 = 'N' * 50
        
        fastq_r1_content_list.append(f"@{read_id_key}/1\n{seq_r1}\n+\n{'I'*50}\n")
        fastq_r2_content_list.append(f"@{read_id_key}/2\n{seq_r2}\n+\n{'I'*50}\n")

    with gzip.open(dummy_r1_fastq_file, "wt") as f_r1_out:
        f_r1_out.write("".join(fastq_r1_content_list))
    with open(dummy_r2_fastq_file, "w") as f_r2_out:
        f_r2_out.write("".join(fastq_r2_content_list))

    print(f"--- Running StrainR Binning Pipeline Example ---")
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

        traceback.print_exc()

    print(
        f"--- Example Run Finished. Check '{example_output_directory_path.resolve() / 'bins'}' for binned FASTQ files. ---"
    )
    # Optional: Add cleanup for example files and directory
    # import shutil
    # if input("Clean up example directory? (y/N): ").strip().lower() == 'y':
    #     shutil.rmtree(example_output_directory_path)
    #     print(f"Cleaned up example directory: {example_output_directory_path.resolve()}")
        traceback.print_exc() 
    
    print(f"--- Example Run Finished. Check '{example_output_directory_path.resolve() / 'bins'}' for binned FASTQ files. ---")
