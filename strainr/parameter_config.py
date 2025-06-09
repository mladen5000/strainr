import argparse
import pathlib


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

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Strainr: A tool for strain analysis using k-mer based methods."
    )
    parser.add_argument(
        "input",
        help="One or more forward/unpaired FASTQ input file(s).",
        nargs="+",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-r",
        "--reverse",
        help="Optional: One or more reverse FASTQ input file(s), corresponding to 'input'. (Feature: todo)",
        nargs="+",
        type=pathlib.Path,
        default=[],
    )
    parser.add_argument(
        "-d",
        "--db",
        help="Path to the KmerStrainDatabase file (Parquet format).",  # Corrected
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--procs",
        type=int,
        default=4,
        help="Number of cores to use (default: 4)",
    )
    parser.add_argument(
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
        help="Selection mode for disambiguation",  # Corrected
        choices=[
            "random",
            "max",
            "multinomial",
            "dirichlet",
        ],
        type=str,
        default="max",
    )
    parser.add_argument(
        "-a",
        "--thresh",
        help="Abundance threshold for reporting strains (default: 0.001).",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--bin",
        action="store_true",
        required=False,
        help="Perform binning.",  # Corrected
    )
    parser.add_argument(
        "--save-raw-hits",
        action="store_true",
        required=False,
        help="Save intermediate k-mer hit scores and final assignments to output files (format may vary).",  # Corrected
    )
    config_space = parser.parse_args()
    return config_space
