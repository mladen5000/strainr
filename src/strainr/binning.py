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

# Third-party imports
import pandas as pd
from Bio import SeqIO

# Local application/library specific imports
from strainr.utils import open_file_transparently
from strainr.genomic_types import ReadId, StrainIndex

# Type alias for clarity
FinalAssignmentsType = Dict[ReadId, Union[StrainIndex, str]]


def generate_table(
    final_assignments: FinalAssignmentsType, all_strain_names: List[str]
) -> pd.DataFrame:
    """
    Generates a DataFrame indicating read assignments to strains.

    The DataFrame has ReadIds as its index and strain names as columns.
    A cell value of 1 indicates that the read is assigned to the strain,
    and 0 otherwise. Unassigned reads (marked by a string like "NA")
    will result in rows of all zeros for those reads.

    Args:
        final_assignments: A dictionary mapping ReadId to either a StrainIndex (int)
                           or a string marker (e.g., "NA") for unassigned reads.
        all_strain_names: A list of all strain names, defining the columns
                          of the output DataFrame. The order of names in this
                          list corresponds to StrainIndex.

    Returns:
        A Pandas DataFrame where rows are ReadIds, columns are strain names,
        and cell values are 1 if the read is assigned to that strain, else 0.
    """
    if not isinstance(final_assignments, dict):
        raise TypeError("final_assignments must be a dictionary.")
    if not isinstance(all_strain_names, list) or not all(
        isinstance(s, str) for s in all_strain_names
    ):
        raise TypeError("all_strain_names must be a list of strings.")

    read_ids = list(final_assignments.keys())
    if not all_strain_names:
        # Return empty DataFrame with read_ids as index if there are no strains to form columns
        # This ensures the index is consistent with final_assignments keys.
        print(
            "Warning: all_strain_names is empty. Returning DataFrame with no columns."
        )
        return pd.DataFrame(index=read_ids)

    # Initialize DataFrame with all zeros
    df = pd.DataFrame(0, index=read_ids, columns=all_strain_names)

    for read_id, assignment in final_assignments.items():
        if isinstance(assignment, int):  # StrainIndex
            if 0 <= assignment < len(all_strain_names):
                strain_name = all_strain_names[assignment]
                df.loc[read_id, strain_name] = 1
            else:
                print(
                    f"Warning: Invalid StrainIndex {assignment} for read_id {read_id}. Max index is {len(all_strain_names)-1}. Read will not be assigned to any strain in table."
                )
    return df


def get_top_strain_names(
    read_assignments: Dict[
        ReadId, Union[str, int]
    ],  # Assuming values are strain names or indices
    strain_list: List[str],
    unassigned_marker: str = "NA",
    exclude_unassigned: bool = True,
) -> List[str]:
    """
    Determines strain names sorted by the number of assigned reads.

    Args:
        read_assignments: A dictionary mapping ReadId to its assigned StrainIndex (int)
                          or a string marker (e.g., "NA") for unassigned reads.
        strain_list: A list of all strain names. Indices in `read_assignments`
                     refer to this list.
        unassigned_marker: The specific string value used to mark unassigned reads.
        exclude_unassigned: If True, unassigned reads are ignored when ranking strains.

    Returns:
        A list of strain names, sorted by read count (descending).
    """
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
        return []

    # Count occurrences of each assignment value (StrainIndex or unassigned_marker)
    assignment_values = list(read_assignments.values())
    counts = pd.Series(assignment_values).value_counts()

    mapped_strain_counts: Dict[str, int] = {}
    for entity, count_val in counts.items():
        if isinstance(entity, int):  # StrainIndex
            if 0 <= entity < len(strain_list):
                strain_name = strain_list[entity]
                mapped_strain_counts[strain_name] = (
                    mapped_strain_counts.get(strain_name, 0) + count_val
                )
            # else: Invalid StrainIndex, simply ignore or log
        elif isinstance(entity, str):  # String marker
            if entity == unassigned_marker and exclude_unassigned:
                continue
            # If the string entity itself is a valid strain name (and not the unassigned marker)
            if entity in strain_list and entity != unassigned_marker:
                mapped_strain_counts[entity] = (
                    mapped_strain_counts.get(entity, 0) + count_val
                )
            # Other strings are ignored.

    # Sort strains by count in descending order
    sorted_strains = sorted(
        mapped_strain_counts.items(), key=lambda item: item[1], reverse=True
    )

    return [strain_name for strain_name, count_val in sorted_strains]


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
                    # Read ID normalization might be needed if FASTQ IDs have suffixes like /1 or /2
                    # and read_ids_for_strain does not.
                    # Example: normalized_id = record.id.split('/')[0].split(' ')[0]
                    if (
                        record.id in read_ids_for_strain
                    ):  # or normalized_id in read_ids_for_strain:
                        records_to_write.append(record)

            if records_to_write:
                with open(
                    binned_fastq_filepath, "w"
                ) as binned_handle:  # SeqIO.write expects text handle
                    count = SeqIO.write(records_to_write, binned_handle, "fastq")
                print(
                    f"Saved {count} records for strain '{strain_name}' ({read_suffix}) "
                    f"to {binned_fastq_filepath.name}"
                )
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


def create_binned_fastq_files(
    top_strain_names: List[str],
    read_to_strain_assignment_table: pd.DataFrame,
    forward_fastq_path: pathlib.Path,
    reverse_fastq_path: Optional[pathlib.Path],
    output_dir: pathlib.Path,
    num_bins_to_create: int = 2,
    unassigned_marker: str = "NA",
) -> Tuple[Set[str], List[mp.Process]]:
    """
    Orchestrates binning of reads for the top N identified strains.

    Args:
        top_strain_names: List of strain names, typically sorted by abundance.
        read_to_strain_assignment_table: DataFrame with ReadIds as index,
                                         strain names as columns, and 1 (assigned)
                                         or 0 (not assigned) as values.
        forward_fastq_path: Path to the forward (R1) FASTQ file.
        reverse_fastq_path: Path to the reverse (R2) FASTQ file (optional).
        output_dir: Base directory for binned FASTQ files ("bins" subdirectory).
        num_bins_to_create: Max number of top strains for which to create bins.
        unassigned_marker: Marker used for unassigned reads (e.g., "NA"),
                           to ensure these are not processed as strains.

    Returns:
        Tuple: (set of strain names for which binning was attempted,
                list of mp.Process objects launched).
    """
    bin_output_dir = output_dir / "bins"
    bin_output_dir.mkdir(exist_ok=True, parents=True)

    strains_to_process = []
    if num_bins_to_create > 0 and top_strain_names:
        strains_to_process = top_strain_names[:num_bins_to_create]

    if not strains_to_process:
        print(
            f"No strains selected for binning (num_bins_to_create={num_bins_to_create}, top_strain_names has {len(top_strain_names)} entries)."
        )
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
            continue

        print(f"Preparing to bin reads for strain: {strain_name}...")

        if strain_name not in read_to_strain_assignment_table.columns:
            print(
                f"Warning: Strain '{strain_name}' not found as a column in the assignment table. Skipping."
            )
            continue

        strain_assigned_reads_series = read_to_strain_assignment_table[strain_name]
        # Ensure index elements are strings (ReadId type)
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
    """
    Executes the main binning pipeline.

    Args:
        final_assignments: Dictionary mapping ReadId to its assigned StrainIndex
                           or an unassigned marker string. Used for generating
                           the assignment table and determining top strains.
        all_strain_names: List of all possible strain names.
        forward_reads_fastq: Path to the forward (R1) FASTQ file.
        output_directory: Path for results (a "bins" subdirectory will be created).
        num_top_strains_to_bin: Number of top strains to select for binning.
        reverse_reads_fastq: Optional path to the reverse (R2) FASTQ file.
        unassigned_marker: String marker for unassigned reads.
    """
    fwd_fastq_path = pathlib.Path(forward_reads_fastq)
    out_dir_path = pathlib.Path(output_directory)
    rev_fastq_path = pathlib.Path(reverse_reads_fastq) if reverse_reads_fastq else None

    # 1. Generate the binning table from final_assignments
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
    binned_strains_set, processes = create_binned_fastq_files(
        top_strain_names=top_strains,
        read_to_strain_assignment_table=read_to_strain_assignment_table,
        forward_fastq_path=fwd_fastq_path,
        reverse_fastq_path=rev_fastq_path,
        output_dir=out_dir_path,
        num_bins_to_create=num_top_strains_to_bin,
        unassigned_marker=unassigned_marker,
    )

    if processes:
        print(f"Waiting for {len(processes)} binning process(es) to complete...")
        for p in processes:
            p.join()
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
    try:
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
        seq_r1 = "A" * 50
        seq_r2 = "C" * 50
        if mock_pipeline_assignments_data[read_id_key] == example_unassigned_marker_val:
            seq_r1 = "N" * 50  # Different sequence for unassigned for visual check
            seq_r2 = "N" * 50

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
