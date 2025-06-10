#!/usr/bin/env python
import argparse
import subprocess
import pathlib
import pickle
import sys
import logging
from typing import Tuple, List, Dict, Union

# It's good practice to ensure src directory is in PYTHONPATH if running from root,
# or use relative imports if the runner is structured as part of a package.
# For a simple script in root, direct imports might require PYTHONPATH manipulation
# or installing the package in editable mode.
# For now, assume src.strainr modules can be imported.
# If not, this might need adjustment (e.g. sys.path.append).
try:
    from strainr.binning import run_binning_pipeline

    # For get_sample_name, if needed directly by runner (classify.py already uses it for output naming)
    from strainr.utils import _get_sample_name as get_sample_name
except ImportError:
    # Fallback for finding src if script is in root and src is a subdir
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    try:
        from strainr.binning import run_binning_pipeline
        from strainr.utils import _get_sample_name as get_sample_name
    except ImportError as e:
        print(
            f"Error: Could not import StrainR modules. Make sure StrainR is installed or src is in PYTHONPATH: {e}"
        )
        sys.exit(1)


# Setup basic logging for the runner script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - PIPELINE - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_command(command: list, step_name: str):
    logger.info(f"Running {step_name} with command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"{step_name} completed successfully.")
        logger.debug(f"{step_name} STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.debug(f"{step_name} STDERR:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {step_name}:")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"STDOUT:\n{e.stdout}")
        logger.error(f"STDERR:\n{e.stderr}")
        raise  # Re-raise to stop the pipeline


def main():
    parser = argparse.ArgumentParser(description="StrainR Full Pipeline Runner Script")

    # --- DB Building Arguments ---
    db_group = parser.add_argument_group("Database Building Arguments")
    db_source_group = db_group.add_mutually_exclusive_group(required=True)
    db_source_group.add_argument(
        "--db_taxid", type=str, help="Species taxonomic ID for NCBI download."
    )
    db_source_group.add_argument(
        "--db_assembly_accessions",
        type=str,
        help="Path to file listing assembly accessions for NCBI.",
    )
    db_source_group.add_argument(
        "--db_genus", type=str, help="Genus name for NCBI download."
    )
    db_source_group.add_argument(
        "--db_custom_genomes_dir",
        type=str,
        help="Path to folder with custom genome FASTA files.",
    )

    db_group.add_argument(
        "--db_kmerlen", type=int, default=31, help="K-mer length for database."
    )
    db_group.add_argument(
        "--db_name_prefix",
        type=str,
        default="strainr_db",
        help="Output name prefix for the database (e.g., 'my_db').",
    )
    db_group.add_argument(
        "--db_procs",
        type=int,
        help="Number of cores for DB building (defaults to --procs if not set).",
    )
    db_group.add_argument(
        "--db_unique_taxid",
        action="store_true",
        help="Filter NCBI genomes for unique strain-level taxID.",
    )
    db_group.add_argument(
        "--db_assembly_levels",
        choices=["complete", "chromosome", "scaffold", "contig"],
        type=str,
        default="complete",
        help="Assembly level for NCBI download.",
    )
    db_group.add_argument(
        "--db_source_ncbi",
        choices=["refseq", "genbank"],
        type=str,
        default="refseq",
        help="NCBI source for downloads.",
    )

    # --- Classification Arguments ---
    classify_group = parser.add_argument_group("Classification Arguments")
    classify_group.add_argument(
        "--classify_input_forward",
        nargs="+",
        required=True,
        type=str,
        help="Input forward read file(s) (FASTA/FASTQ).",
    )
    classify_group.add_argument(
        "--classify_input_reverse",
        nargs="*",
        type=str,
        help="Optional input reverse read file(s).",
    )
    classify_group.add_argument(
        "--classify_out_dir",
        type=str,
        default="strainr_classification_out",
        help="Output directory for classification.",
    )
    classify_group.add_argument(
        "--classify_procs",
        type=int,
        help="Number of cores for classification (defaults to --procs if not set).",
    )
    classify_group.add_argument(
        "--classify_disambiguation_mode",
        choices=["random", "max", "multinomial", "dirichlet"],
        default="max",
        help="Disambiguation mode.",
    )
    classify_group.add_argument(
        "--classify_abundance_threshold",
        type=float,
        default=0.001,
        help="Abundance threshold.",
    )
    classify_group.add_argument(
        "--classify_save_raw_hits",
        action="store_true",
        help="Save raw k-mer hits during classification.",
    )

    # --- Binning Arguments ---
    bin_group = parser.add_argument_group("Binning Arguments")
    bin_group.add_argument(
        "--bin_num_top_strains",
        type=int,
        default=2,
        help="Number of top strains to create bins for.",
    )
    bin_group.add_argument(
        "--bin_out_dir",
        type=str,
        help="Output directory for binned FASTQ files (defaults to a 'bins' subdir in --classify_out_dir).",
    )
    bin_group.add_argument(
        "--skip_binning", action="store_true", help="Skip the binning step."
    )

    # --- General Arguments ---
    general_group = parser.add_argument_group("General Arguments")
    general_group.add_argument(
        "--procs",
        type=int,
        default=4,
        help="Default number of cores for all steps (can be overridden).",
    )
    general_group.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for sub-scripts."
    )

    args = parser.parse_args()

    # Resolve procs for each step
    db_procs = args.db_procs if args.db_procs is not None else args.procs
    classify_procs = (
        args.classify_procs if args.classify_procs is not None else args.procs
    )

    # --- Step 1: Build Database ---
    logger.info("--- Starting Step 1: Database Building ---")
    db_command = [
        "python",
        "src/strainr/build_db.py",
        "--kmerlen",
        str(args.db_kmerlen),
        "--out",
        args.db_name_prefix,
        "--procs",
        str(db_procs),
        "--assembly_levels",
        args.db_assembly_levels,
        "--source",
        args.db_source_ncbi,
    ]
    if args.db_taxid:
        db_command.extend(["--taxid", args.db_taxid])
    elif args.db_assembly_accessions:
        db_command.extend(["--assembly_accessions", args.db_assembly_accessions])
    elif args.db_genus:
        db_command.extend(["--genus", args.db_genus])
    elif args.db_custom_genomes_dir:
        db_command.extend(["--custom", args.db_custom_genomes_dir])

    if args.db_unique_taxid:
        db_command.append("--unique-taxid")
    if args.verbose:
        db_command.append("--verbose")  # Assuming build_db.py supports --verbose

    run_command(db_command, "Database Building")
    db_output_path = pathlib.Path(f"{args.db_name_prefix}.db.parquet").resolve()
    logger.info(f"Database built at: {db_output_path}")

    # --- Step 2: Classify Reads ---
    logger.info("--- Starting Step 2: Read Classification ---")

    # Create classification output directory
    classify_out_dir_path = pathlib.Path(args.classify_out_dir)
    classify_out_dir_path.mkdir(parents=True, exist_ok=True)

    base_classify_command = [
        "python",
        "src/strainr/classify.py",
        "--db",
        str(db_output_path),
        "--out",
        str(classify_out_dir_path),
        "--procs",
        str(classify_procs),
        "--mode",
        args.classify_disambiguation_mode,
        "--thresh",
        str(args.classify_abundance_threshold),
    ]
    if args.classify_save_raw_hits:
        base_classify_command.append("--save_raw_hits")
    if args.verbose:
        base_classify_command.append("--verbose")

    # Handle multiple input files for classification
    # The binning step will run per sample based on the outputs of classify.py
    num_fwd_files = len(args.classify_input_forward)
    num_rev_files = (
        len(args.classify_input_reverse) if args.classify_input_reverse else 0
    )

    if num_rev_files > 0 and num_fwd_files != num_rev_files:
        logger.error(
            "Mismatch in number of forward and reverse read files for classification."
        )
        sys.exit(1)

    processed_samples_data = []  # To store paths for binning

    for i in range(num_fwd_files):
        fwd_read_file = args.classify_input_forward[i]
        current_classify_command = list(base_classify_command)  # Start with a copy
        current_classify_command.extend(["--input_forward", fwd_read_file])

        sample_name_for_output = get_sample_name(pathlib.Path(fwd_read_file))
        logger.info(
            f"Classifying sample: {sample_name_for_output} (from {fwd_read_file})"
        )

        if args.classify_input_reverse and i < num_rev_files:
            rev_read_file = args.classify_input_reverse[i]
            current_classify_command.extend(["--input_reverse", rev_read_file])

        run_command(
            current_classify_command, f"Classification for {sample_name_for_output}"
        )

        # Store info for binning
        assignments_file = (
            classify_out_dir_path / f"{sample_name_for_output}_final_assignments.pkl"
        )
        strains_file = (
            classify_out_dir_path / f"{sample_name_for_output}_strain_names.txt"
        )

        if assignments_file.exists() and strains_file.exists():
            processed_samples_data.append({
                "sample_name": sample_name_for_output,
                "fwd_reads": fwd_read_file,
                "rev_reads": rev_read_file
                if (args.classify_input_reverse and i < num_rev_files)
                else None,
                "assignments_pkl": assignments_file,
                "strain_names_txt": strains_file,
                "classify_out_dir": classify_out_dir_path,
            })
        else:
            logger.warning(
                f"Output files for binning not found for sample {sample_name_for_output}. Skipping binning for this sample."
            )

    logger.info("All classification runs complete.")

    # --- Step 3: Perform Binning ---
    if args.skip_binning:
        logger.info("--- Skipping Step 3: Binning (as per user request) ---")
    elif not processed_samples_data:
        logger.info(
            "--- Skipping Step 3: Binning (no samples successfully classified for binning inputs) ---"
        )
    else:
        logger.info("--- Starting Step 3: Read Binning ---")
        for sample_data in processed_samples_data:
            logger.info(f"Performing binning for sample: {sample_data['sample_name']}")
            try:
                with open(sample_data["assignments_pkl"], "rb") as f:
                    final_assignments = pickle.load(f)

                with open(sample_data["strain_names_txt"], "r") as f:
                    all_strain_names = [line.strip() for line in f if line.strip()]

                if args.bin_out_dir:
                    # User specified a custom directory. run_binning_pipeline's callee will create a 'bins' subdir in it.
                    effective_bin_parent_dir = pathlib.Path(args.bin_out_dir)
                else:
                    # Default: use the classification output directory as the parent for the 'bins' subdir.
                    effective_bin_parent_dir = sample_data[
                        "classify_out_dir"
                    ]  # This is already a pathlib.Path

                effective_bin_parent_dir.mkdir(parents=True, exist_ok=True)

                # run_binning_pipeline expects:
                # final_assignments: FinalAssignmentsType (loaded)
                # all_strain_names: List[str] (loaded)
                # read_assignments: Dict[ReadId, Union[str, int]] (this is final_assignments)
                # forward_reads_fastq: Union[str, pathlib.Path]
                # output_directory: Union[str, pathlib.Path] (this is for bin_output_dir_actual / "bins")
                # num_top_strains_to_bin: int
                # reverse_reads_fastq: Optional
                # unassigned_marker: str (default "NA" in run_binning_pipeline)

                run_binning_pipeline(
                    final_assignments=final_assignments,
                    all_strain_names=all_strain_names,
                    read_assignments=final_assignments,  # This is used for get_top_strain_names
                    forward_reads_fastq=pathlib.Path(sample_data["fwd_reads"]),
                    output_directory=effective_bin_parent_dir,  # Pass the determined parent directory
                    num_top_strains_to_bin=args.bin_num_top_strains,
                    reverse_reads_fastq=pathlib.Path(sample_data["rev_reads"])
                    if sample_data["rev_reads"]
                    else None,
                )
                final_bin_location = effective_bin_parent_dir / "bins"
                logger.info(
                    f"Binning completed for sample: {sample_data['sample_name']}. Output in {final_bin_location}"
                )

            except Exception as e:
                logger.error(
                    f"Error during binning for sample {sample_data['sample_name']}: {e}"
                )
                # Continue to next sample if one fails

    logger.info("--- StrainR Pipeline Finished ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"StrainR pipeline failed: {e}")
        sys.exit(1)
