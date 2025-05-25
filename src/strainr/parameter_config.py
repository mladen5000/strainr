import argparse
import pathlib


def process_arguments() -> argparse.ArgumentParser:
    """
    Get the arguments from the command line, and return them to the main function.

    Examples:
        args.add_argument('--source_file', type=open)
        args.add_argument('--dest_file',
        type=argparse.FileType('w', encoding='latin-1'))
        args.add_argument('--datapath', type=pathlib.Path)
    """

    args: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Strainr: A tool for strain analysis"
    )
    args.add_argument(
        "input",
        help="input file",
        nargs="+",
        type=str,
    )
    args.add_argument(
        "-r",
        "--reverse",
        help="reverse fastq file, todo",
        nargs="+",
        type=str,
    )
    args.add_argument(
        "-d",
        "--db",
        # required=True,
        help="Database file",
        type=str,
    )
    args.add_argument(
        "-p",
        "--procs",
        type=int,
        default=4,
        help="Number of cores to use (default: 4)",
    )
    args.add_argument(
        "-o",
        "--out",
        type=pathlib.Path,
        required=False,
        help="Output folder",
        default="strainr_out",
    )
    args.add_argument(
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
    args.add_argument(
        "-a",
        "--thresh",
        help="",
        type=float,
        default=0.001,
    )
    args.add_argument(
        "--bin",
        action="store_true",
        required=False,
        help=" Perform binning.  ",
    )
    args.add_argument(
        "--save-raw-hits",
        action="store_true",
        required=False,
        help=" Save the intermediate results as a csv file containing each read's strain information.",
    )
    config_space = args.parse_args()
    return config_space
