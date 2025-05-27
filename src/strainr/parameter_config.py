import argparse
import pathlib
# from typing import List # List is not used in type hints in this file


def process_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the Strainr application.

    Defines and parses arguments related to input files (forward, reverse),
    database path, output directory, processing options (cores, mode, threshold),
    and optional flags for binning and saving raw hits.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments
                            as attributes.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(  # Renamed for clarity
        description="Strainr: A tool for strain analysis using k-mer based methods."  # More descriptive
    )
    parser.add_argument(
        "input",
        help="One or more forward/unpaired FASTQ input file(s).",  # Clarified help
        nargs="+",
        type=pathlib.Path,  # Changed to pathlib.Path
    )
    parser.add_argument(
        "-r",
        "--reverse",
        help="Optional: One or more reverse FASTQ input file(s), corresponding to 'input'. (Feature: todo)",  # Clarified help
        nargs="+",  # Should match number of input files if provided, or be single for all
        type=pathlib.Path,  # Changed to pathlib.Path
        default=[],  # Provide a default empty list
    )
    parser.add_argument(
        "-d",
        "--db",
        help="Path to the StrainKmerDatabase file (Parquet format).",  # Clarified help
        type=pathlib.Path,  # Changed to pathlib.Path
        required=True,  # Assuming database is essential for most operations
    )
    parser.add_argument(
        "-p",
        "--procs",
        type=int,
        default=4,
        help="Number of cores to use (default: 4)",
    )
    parser.add_argument(  # Standardized to parser
        "-o",
        "--out",
        type=pathlib.Path,
        required=False,
        help="Output folder",
        default="strainr_out",
    )
    parser.add_argument(
        "-m",
        "--mode",
        help=" Selection mode for diambiguation ",
        choices=[
            "random",
            "max",
            "multinomial",
            "dirichlet",
        ],
        type=str,
        default="max",
    )
    parser.add_argument(  # Standardized to parser
        "-a",
        "--thresh",
        help="Abundance threshold for reporting strains (default: 0.001).",  # Improved help
        type=float,
        default=0.001,
    )
    parser.add_argument(  # Changed from args to parser
        "--bin",
        action="store_true",
        required=False,
        help=" Perform binning.  ",
    )
    parser.add_argument(  # Standardized to parser
        "--save-raw-hits",
        action="store_true",
        required=False,  # Default is False if not specified
        help="Save intermediate k-mer hit scores and final assignments as pickle files.",  # Clarified help
    )
    # Ensure all calls are to 'parser', not 'args'
    config_space = parser.parse_args()
    return config_space
