import dataclasses
import pathlib
import argparse  # For type hinting args
from typing import Union, Any  # Removed List, Added Any

# Assuming StrainKmerDatabase is correctly importable from this location
from .database import StrainKmerDatabase # Changed to relative import

# Assuming process_arguments is correctly importable
from .parameter_config import process_arguments # Changed to relative and specific import


class SequenceFile(pathlib.Path):
    """
    A pathlib.Path subclass representing a sequence file that must exist.
    """

    # _flavour = pathlib.Path._flavour  # type: ignore # Workaround for pathlib internals if needed # Commented out
    # This line caused AttributeError and is likely unnecessary as Path subclasses should inherit it.

    def __new__(cls, *args: Any, **kwargs: Any) -> "SequenceFile":
        # Path resolution happens in Path.__new__ or __init__
        # We can't easily do validation before super().__new__ if we rely on Path methods.
        # For this subclass, it's simpler to validate after the object is created.
        return super().__new__(cls, *args, **kwargs)  # type: ignore

    def __init__(self, path: Union[str, pathlib.Path]) -> None:
        """
        Initializes SequenceFile. The path itself is handled by pathlib.Path's init.
        This __init__ is primarily for post-initialization validation.

        Args:
            path: Path to the sequence file.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
        """
        # No need for super().__init__(path) if Path handles it in __new__
        # No need for self.path = path, as Path object itself is the path.
        if not self.exists():  # `self` is now the Path object
            raise FileNotFoundError(f"Sequence file does not exist: {self}")
        if not self.is_file():
            raise FileNotFoundError(f"Path is not a file: {self}")


def main() -> None:
    """
    Illustrative main entry point for the Strainr application.

    Note: This function is currently illustrative and demonstrates basic argument parsing
    and initialization of core classes. It makes assumptions about argument availability
    (e.g., 'k' for k-mer length, which is not defined in parameter_config.py) and
    processes only the first input file if multiple are provided.
    Further development is needed for full functionality.
    """
    args: argparse.Namespace = process_arguments() # Use directly imported function

    # Corrected argument access and assumptions
    # args.input is List[pathlib.Path], args.db is pathlib.Path, args.out is pathlib.Path

    if not args.input:
        print("Error: No input FASTQ files provided.")
        return  # Or raise error

    # For this illustrative main, process only the first input file.
    # A full implementation would likely iterate through args.input.
    try:
        # Use SequenceFile for validation, though args.input elements are already pathlib.Path
        # If validation is desired here, it should be applied to each file.
        # For simplicity, directly use the Path object from args.
        input_fasta_path: pathlib.Path = args.input[0]
        if (
            not input_fasta_path.is_file()
        ):  # Basic check, SequenceFile would do this too
            raise FileNotFoundError(
                f"Input file not found or is not a file: {input_fasta_path}"
            )
    except IndexError:
        print(
            "Error: Input file list is empty (this should be caught by argparse if nargs='+' and required)."
        )
        return

    results_output_dir: pathlib.Path = args.out  # Corrected from args.outdir
    database_file_path: pathlib.Path = args.db

    # Handling k-mer length:
    # 'args.k' is not defined in parameter_config.py.
    # KmerStrainDatabase expects 'expected_kmer_length'.
    # Runner has a default k=31. Let's assume we use a default or would add a CLI arg.
    # For this refactor, we'll use a placeholder/default if args.k is not available.

    # Placeholder: If k-mer length were an argument, it would be e.g. args.kmer_length
    # For now, let KmerStrainDatabase infer it or use its default if Runner's k is intended
    # for something else. Or, assume Runner's k is the one to use.
    # Let's assume the k-mer length for database loading should be specified or inferred by KmerStrainDatabase.
    # The Runner's 'k' might be for k-mer extraction if different from DB's.

    # Initialize KmerStrainDatabase. expected_kmer_length can be None for inference.
    # If a specific k-mer length is expected from CLI for DB, it should be added to process_arguments.
    # For now, let's assume we want the DB to use its intrinsic k-mer length.
    try:
        kmer_db = StrainKmerDatabase( # Updated class name
            database_filepath=database_file_path, expected_kmer_length=None
        )
    except Exception as e:
        print(f"Error initializing StrainKmerDatabase: {e}") # Updated class name
        return

    # Runner's k parameter (default 31) can be used for k-mer extraction logic within Runner.
    # If this 'k' needs to come from CLI, it should be added to parameter_config.py
    runner_k_value = getattr(args, "k", 31)  # Use 31 if no 'k' arg (which there isn't)

    print("Initializing Strainr run with: ")
    print(f"  Input FASTQ: {input_fasta_path}")
    print(f"  Output Directory: {results_output_dir}")
    print(f"  K-mer Database: {kmer_db.database_filepath} (k={kmer_db.kmer_length})")
    print(f"  Runner k-mer length (for processing): {runner_k_value}")

    try:
        # Initialize Runner
        # Runner expects fasta: pathlib.Path, kmer_database: KmerStrainDatabase, k: int
        strain_runner_instance = Runner(
            fasta=input_fasta_path, kmer_database=kmer_db, k=runner_k_value
        )
        print(f"Runner instance created: {strain_runner_instance}")
        # Further operations with strain_runner_instance would go here.
        # e.g., strain_runner_instance.start_analysis()
    except Exception as e:
        print(f"Error initializing or running Strainr Runner: {e}")
        return

    print("Main function finished (illustrative).")


@dataclasses.dataclass
class Runner:
    """
    Container class for encapsulating parameters and logic for a single
    strain analysis run.
    """

    fasta: pathlib.Path  # Path to the input FASTA/FASTQ file
    kmer_database: StrainKmerDatabase  # Instance of the k-mer database, updated class name
    k: int = 31  # k-mer length to use for analysis (e.g., k-mer extraction)

    # Removed commented-out block
