# Standard library imports
import multiprocessing as mp
import pathlib
from typing import List, Dict, Set, Tuple, Union, Any # Added Any for undefined generate_table output

# Third-party imports
import pandas as pd
from Bio import SeqIO

# Local application/library specific imports
from strainr.utils import open_file_transparently # Corrected import
# Assuming ReadId is defined in genomic_types, if not, define/import appropriately
from strainr.genomic_types import ReadId # Or from .genomic_types if preferred and structure allows


# Placeholder for the generate_table function if its definition is elsewhere.
# This is to allow type hinting for its presumed output.
# If generate_table is part of this module, it should be defined here.
def generate_table(intermediate_results: Any, strains: List[str]) -> pd.DataFrame:
    """
    Placeholder for a function that generates a table for binning.

    This function is assumed to convert raw classification results into a
    DataFrame where rows are read IDs and columns are strain names,
    with values indicating hits or scores.

    Args:
        intermediate_results: The raw results from the classification step.
                              The exact structure is dependent on its definition.
        strains: A list of strain names.

    Returns:
        A Pandas DataFrame suitable for use as `bin_table`.
    """
    # This is a mock implementation or should be imported if defined elsewhere.
    print(
        "Warning: Using placeholder for generate_table. "
        "Actual implementation needed for real functionality."
    )
    # Example: Assuming intermediate_results is Dict[ReadId, Dict[str, int]]
    # This is highly speculative and needs to match actual data.
    if isinstance(intermediate_results, dict) and strains:
        return pd.DataFrame.from_dict(intermediate_results, orient='index').reindex(columns=strains).fillna(0)
    return pd.DataFrame()


def get_top_strain_names(
    read_assignments: Dict[ReadId, Union[str, int]], # Assuming values are strain names or indices
    strain_list: List[str],
    exclude_unassigned: bool = True,
) -> List[str]:
    """
    Determines the most abundant strains based on read assignments.

    This function takes a dictionary of read assignments (where keys are read IDs
    and values are assigned strain identifiers/indices) and a list of all possible
    strain names. It returns a list of strain names sorted by the number of
    reads assigned to them, in descending order.

    Args:
        read_assignments: A dictionary where keys are read IDs (str) and values
                          are either strain names (str) or strain indices (int)
                          as found in `strain_list`. Can also include "NA" or
                          similar for unassigned reads.
        strain_list: A list of all strain names. If `read_assignments` uses
                     indices, this list is used for mapping indices to names.
        exclude_unassigned: If True (default), strains labeled as "NA" (or similar
                            markers for unassigned) are excluded from the ranked list.

    Returns:
        A list of strain names, sorted from the most assigned to the least assigned.
    """
    if not isinstance(read_assignments, dict):
        raise TypeError("read_assignments must be a dictionary.")
    if not isinstance(strain_list, list):
        raise TypeError("strain_list must be a list.")

    # Convert assignments to a Pandas Series for easy counting
    assignment_series = pd.Series(read_assignments)

    # Get value counts (number of reads per assigned strain/index)
    strain_counts = assignment_series.value_counts()

    top_assigned_entities = list(strain_counts.index)
    
    resolved_top_strain_names: List[str] = []
    for entity in top_assigned_entities:
        if exclude_unassigned and str(entity) == "NA":
            continue
        if isinstance(entity, int) and entity < len(strain_list):
            resolved_top_strain_names.append(strain_list[entity])
        elif isinstance(entity, str) and entity in strain_list:
             resolved_top_strain_names.append(entity)
        # else: can add warning for entities not mappable to strain_list

    return resolved_top_strain_names


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
        strain_name: The name or identifier of the strain for which to bin reads.
                     Used for naming output files.
        read_ids_for_strain: A set of read IDs assigned to this strain.
        forward_fastq_path: Path to the forward (R1) FASTQ file.
        reverse_fastq_path: Path to the reverse (R2) FASTQ file, if applicable.
        output_bin_dir: The directory where binned FASTQ files will be written.
    """
    # Generate a safe filename from strain_name (e.g., replacing spaces or special chars)
    safe_strain_filename = strain_name.replace(" ", "_").replace("/", "_")

    fastq_files_to_process: List[Tuple[pathlib.Path, str]] = []
    if forward_fastq_path.exists():
        fastq_files_to_process.append((forward_fastq_path, "R1"))
    if reverse_fastq_path and reverse_fastq_path.exists():
        fastq_files_to_process.append((reverse_fastq_path, "R2"))

    for original_fastq_path, read_suffix in fastq_files_to_process:
        binned_fastq_filename = f"bin.{safe_strain_filename}_{read_suffix}.fastq"
        binned_fastq_filepath = output_bin_dir / binned_fastq_filename

        records_to_write = []
        # Use open_file_transparently for reading potentially gzipped FASTQ
        with open_file_transparently(original_fastq_path) as original_handle:
            for record in SeqIO.parse(original_handle, "fastq"):
                # Matching by record.id is generally more robust than record.description
                # Paired-end read IDs might need normalization (e.g. removing /1, /2) if
                # IDs in read_ids_for_strain are normalized. Here, we assume exact match.
                if record.id in read_ids_for_strain:
                    records_to_write.append(record)
        
        with open(binned_fastq_filepath, "w") as binned_handle:
            count = SeqIO.write(records_to_write, binned_handle, "fastq")

        print(
            f"Saved {count} records for strain '{strain_name}' "
            f"from {original_fastq_path.name} to {binned_fastq_filepath.name}"
        )


def create_binned_fastq_files(
    top_strain_names: List[str],
    read_to_strain_assignment_table: pd.DataFrame, # Assumes index=ReadId, columns=strains
    forward_fastq_path: pathlib.Path,
    reverse_fastq_path: Optional[pathlib.Path],
    output_dir: pathlib.Path,
    num_bins_to_create: int = 2,
) -> Tuple[Set[str], List[mp.Process]]:
    """
    Orchestrates the binning of reads for the top N identified strains.

    For each of the top N strains, this function launches a separate process
    to extract reads assigned to that strain from the input FASTQ files
    and write them into new, strain-specific FASTQ files.

    Args:
        top_strain_names: A list of strain names, typically sorted by abundance.
        read_to_strain_assignment_table: A Pandas DataFrame where the index contains
                                         read IDs and columns represent strain names.
                                         Non-zero values indicate assignment/hits.
        forward_fastq_path: Path to the forward (R1) FASTQ file.
        reverse_fastq_path: Path to the reverse (R2) FASTQ file (optional).
        output_dir: The base directory where binned FASTQ files will be saved.
                    A subdirectory named "bins" will be created here.
        num_bins_to_create: The number of top strains for which to create bins.

    Returns:
        A tuple containing:
            - A set of strain names for which binning was attempted.
            - A list of the multiprocessing.Process objects launched.
    """
    bin_output_dir = output_dir / "bins"
    bin_output_dir.mkdir(exist_ok=True, parents=True)

    strains_selected_for_binning = top_strain_names[:num_bins_to_create]
    print(f"Generating binned FASTQ files for the top {len(strains_selected_for_binning)} strains.")

    processes: List[mp.Process] = []
    binned_strain_names: Set[str] = set()

    for strain_name in strains_selected_for_binning:
        if strain_name == "NA" or not strain_name: # Skip unassigned or empty names
            continue

        binned_strain_names.add(strain_name)
        print(f"Preparing to bin reads for strain: {strain_name}...")

        # Get read IDs associated with the current strain
        # Assumes that the table has strains as columns and read IDs as index.
        # Values > 0 indicate assignment.
        if strain_name not in read_to_strain_assignment_table.columns:
            print(f"Warning: Strain '{strain_name}' not found in assignment table. Skipping.")
            continue
            
        strain_specific_read_ids: Set[ReadId] = set(
            read_to_strain_assignment_table[
                read_to_strain_assignment_table[strain_name] > 0
            ].index
        )
        
        if not strain_specific_read_ids:
            print(f"No reads assigned to strain '{strain_name}'. Skipping file creation.")
            continue

        process = mp.Process(
            target=_extract_reads_for_strain,
            args=(
                strain_name,
                strain_specific_read_ids,
                forward_fastq_path,
                reverse_fastq_path,
                bin_output_dir,
            ),
        )
        process.start()
        processes.append(process)

    # Wait for all launched processes to complete (optional here, could be managed by caller)
    # for p in processes:
    #     p.join()
    # print("All binning processes launched.")

    return binned_strain_names, processes


def run_binning_pipeline(
    classification_results: Any, # Structure depends on upstream `classify` output
    all_strain_names: List[str],
    read_assignments: Dict[ReadId, Union[str, int]], # For determining top strains
    forward_reads_fastq: Union[str, pathlib.Path],
    output_directory: Union[str, pathlib.Path],
    num_top_strains_to_bin: int = 2,
    reverse_reads_fastq: Optional[Union[str, pathlib.Path]] = None,
) -> None:
    """
    Executes the main binning pipeline.

    This involves:
    1. Generating a table of read-to-strain assignments (if not already available).
    2. Identifying the top N most abundant strains.
    3. Creating binned FASTQ files for these top strains.

    Args:
        classification_results: Data from the classification step, used by
                                `generate_table`. The exact structure is
                                dependent on preceding pipeline steps.
        all_strain_names: A list of all possible strain names.
        read_assignments: A dictionary mapping read IDs to their assigned strain
                          (name or index) or "NA" for unassigned. Used to
                          determine top strains.
        forward_reads_fastq: Path to the forward (R1) FASTQ file.
        output_directory: Path to the directory where results (including a
                          "bins" subdirectory) will be saved.
        num_top_strains_to_bin: The number of top strains to select for binning.
        reverse_reads_fastq: Optional path to the reverse (R2) FASTQ file. If
                             provided, paired-end binning is attempted.
    """
    # Ensure paths are pathlib.Path objects
    fwd_fastq_path = pathlib.Path(forward_reads_fastq)
    out_dir_path = pathlib.Path(output_directory)
    rev_fastq_path = pathlib.Path(reverse_reads_fastq) if reverse_reads_fastq else None

    # 1. Generate the binning table (read assignments per strain)
    # The actual implementation of generate_table is crucial here.
    # This table should have read IDs as index and strains as columns.
    # Values might be counts or boolean indicators of hits.
    read_to_strain_assignment_table = generate_table(classification_results, all_strain_names)
    if read_to_strain_assignment_table.empty and num_top_strains_to_bin > 0:
        print("Warning: Read-to-strain assignment table is empty. Binning may not produce results.")


    # 2. Determine top strains
    top_strains = get_top_strain_names(read_assignments, all_strain_names)
    if not top_strains and num_top_strains_to_bin > 0:
        print("Warning: No top strains identified. Binning will be skipped.")
        return

    # 3. Create binned FASTQ files for the top N strains
    binned_strains, processes = create_binned_fastq_files(
        top_strain_names=top_strains,
        read_to_strain_assignment_table=read_to_strain_assignment_table,
        forward_fastq_path=fwd_fastq_path,
        reverse_fastq_path=rev_fastq_path,
        output_dir=out_dir_path,
        num_bins_to_create=num_top_strains_to_bin,
    )

    # Wait for all binning processes to complete
    for p in processes:
        p.join()
    
    print(f"Binning pipeline completed. Binned files for strains: {binned_strains}")


if __name__ == "__main__":
    # This section is for example usage or direct script execution.
    # Actual values need to be provided for the pipeline to run.

    # --- Example Placeholder Data (replace with actual data loading) ---
    # Mock classification_results (e.g., from a previous step)
    # This structure needs to match what generate_table expects.
    # For instance, if generate_table expects Dict[ReadId, Dict[strain_name, score]]:
    mock_classification_output: Dict[ReadId, Dict[str, int]] = {
        "read1": {"StrainA": 10, "StrainB": 2},
        "read2": {"StrainA": 1, "StrainC": 8},
        "read3": {"StrainB": 12},
        "read4": {"StrainA": 9, "StrainB": 1, "StrainC": 1},
        "read5": {"StrainC": 15},
    }
    
    # Mock list of all strain names
    mock_strains_list: List[str] = ["StrainA", "StrainB", "StrainC", "StrainD"]

    # Mock read_assignments (e.g., from a final assignment step)
    # Values could be strain names or indices if strain_list is used for mapping
    mock_final_assignments: Dict[ReadId, str] = {
        "read1": "StrainA",
        "read2": "StrainC",
        "read3": "StrainB",
        "read4": "StrainA",
        "read5": "StrainC",
        "read6": "NA", # Unassigned read
    }

    # Mock paths to FASTQ files and output directory
    # Create dummy FASTQ files for a runnable example
    example_output_dir = pathlib.Path("example_binning_output")
    example_output_dir.mkdir(exist_ok=True)
    
    dummy_r1_path = example_output_dir / "dummy_R1.fastq"
    dummy_r2_path = example_output_dir / "dummy_R2.fastq"

    with open(dummy_r1_path, "w") as f:
        f.write("@read1/1\nATGC\n+\n!!!!\n")
        f.write("@read2/1\nCGTA\n+\n!!!!\n")
        f.write("@read3/1\nTTTT\n+\n!!!!\n")
        f.write("@read4/1\nGGGG\n+\n!!!!\n")
        f.write("@read5/1\nCCCC\n+\n!!!!\n")
        f.write("@read6/1\nNNNN\n+\n!!!!\n")
        
    with open(dummy_r2_path, "w") as f:
        f.write("@read1/2\nGCAT\n+\n!!!!\n")
        f.write("@read2/2\nTACG\n+\n!!!!\n")
        f.write("@read3/2\nAAAA\n+\n!!!!\n")
        f.write("@read4/2\nCCCC\n+\n!!!!\n")
        f.write("@read5/2\nGGGG\n+\n!!!!\n")
        f.write("@read6/2\nNNNN\n+\n!!!!\n")

    print("Running binning pipeline with example data...")
    try:
        run_binning_pipeline(
            classification_results=mock_classification_output,
            all_strain_names=mock_strains_list,
            read_assignments=mock_final_assignments,
            forward_reads_fastq=dummy_r1_path,
            reverse_reads_fastq=dummy_r2_path,
            output_directory=example_output_dir,
            num_top_strains_to_bin=2,
        )
    except Exception as e:
        print(f"An error occurred during example pipeline run: {e}")
    finally:
        # Clean up dummy files
        # import shutil
        # if example_output_dir.exists():
        #     shutil.rmtree(example_output_dir)
        # print(f"Cleaned up example directory: {example_output_dir}")
        pass # Keep files for inspection for now
    
    # Note: The original duplicated main_bin calls and return statements in main_bin were removed.
    # The original if __name__ == "__main__": block was also incomplete.
    # This example provides a more runnable (though still mock) entry point.
