#!/usr/bin/env python
"""
StrainR Command-Line Interface: K-mer based strain classification tool.

This script processes single-end or paired-end sequence reads (FASTA/FASTQ),
classifies them against a k-mer database, resolves ambiguous assignments,
calculates strain abundances, and outputs the results.
"""

import argparse
import gzip
import logging
import multiprocessing as mp
import pathlib
import sys
from collections import Counter
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    TextIO,
    Callable,
    Set,
)

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from pydantic import BaseModel, Field, field_validator, model_validator

# Assuming these local modules are structured correctly within the 'strainr' package
from strainr.kmer_database import KmerStrainDatabase
from strainr.analyze import ClassificationAnalyzer
from strainr.genomic_types import (
    ReadId,
    CountVector,
    StrainIndex,
)

# Type aliases for better readability
ReadHitResults = List[Tuple[ReadId, CountVector]]

# Global constants
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_NUM_PROCESSES = 4
DEFAULT_ABUNDANCE_THRESHOLD = 0.001


class KmerExtractor:
    """Handles k-mer extraction with fallback between Rust and Python implementations."""

    def __init__(self):
        self._extract_func: Callable[[bytes, int], List[bytes]]
        self._rust_available: bool = False
        self._initialize_extractor()

    def _initialize_extractor(self) -> None:
        """Initialize the k-mer extraction function with Rust fallback to Python."""
        try:
            from kmer_counter_rs import extract_kmers_rs

            self._extract_func = extract_kmers_rs
            self._rust_available = True
            logging.getLogger(__name__).info(
                "Successfully imported Rust k-mer counter. Using Rust implementation."
            )
        except ImportError:
            self._extract_func = self._py_extract_canonical_kmers
            self._rust_available = False
            logging.getLogger(__name__).warning(
                "Rust k-mer counter not found. Using Python fallback."
            )
        except Exception as e:
            self._extract_func = self._py_extract_canonical_kmers
            self._rust_available = False
            logging.getLogger(__name__).error(
                f"Error importing Rust k-mer counter: {e}. Using Python fallback."
            )

    @staticmethod
    def _py_reverse_complement(dna_sequence: bytes) -> bytes:
        """Computes the reverse complement of a DNA sequence."""
        complement_map = {
            ord("A"): ord("T"),
            ord("T"): ord("A"),
            ord("C"): ord("G"),
            ord("G"): ord("C"),
            ord("N"): ord("N"),
        }
        return bytes(complement_map.get(base, base) for base in reversed(dna_sequence))

    def _py_extract_canonical_kmers(self, sequence: bytes, k: int) -> List[bytes]:
        """
        Extracts canonical k-mers from a DNA sequence using Python.
        A k-mer is canonical if it's lexicographically smaller than its reverse complement.
        """
        if k <= 0 or not sequence or len(sequence) < k:
            return []

        kmers: List[bytes] = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i : i + k]
            rc_kmer = self._py_reverse_complement(kmer)
            kmers.append(kmer if kmer <= rc_kmer else rc_kmer)

        return kmers

    def extract_kmers(self, sequence_bytes: bytes, k: int) -> List[bytes]:
        """Extract k-mers using the configured implementation."""
        if not sequence_bytes or len(sequence_bytes) < k:
            return []

        try:
            # Normalize sequence for consistent processing
            normalized_seq = sequence_bytes.upper()
            return self._extract_func(normalized_seq, k)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error during k-mer extraction: {e}")
            return []

    @property
    def is_rust_available(self) -> bool:
        """Check if Rust implementation is available."""
        return self._rust_available


# Global k-mer extractor instance
KMER_EXTRACTOR = KmerExtractor()


class SequenceFileProcessor:
    """Handles sequence file parsing and format detection."""

    @staticmethod
    def open_file_handle(file_path: pathlib.Path) -> TextIO:
        """Opens a file, handling .gz compression transparently."""
        if file_path.name.endswith(".gz"):
            return gzip.open(file_path, "rt")
        return file_path.open("rt")

    @staticmethod
    def detect_file_format(file_path: pathlib.Path) -> str:
        """Detects sequence file format (FASTA or FASTQ) based on the first line."""
        with SequenceFileProcessor.open_file_handle(file_path) as f:
            first_line = f.readline().strip()
            if first_line.startswith(">"):
                return "fasta"
            elif first_line.startswith("@"):
                return "fastq"
            else:
                raise ValueError(f"Unknown file format for file: {file_path}")

    @staticmethod
    def parse_sequence_files(
        fwd_reads_path: pathlib.Path,
        rev_reads_path: Optional[pathlib.Path] = None,
    ) -> Generator[Tuple[ReadId, bytes, bytes], None, None]:
        """
        Parses sequence files and yields reads.

        Yields:
            Tuple of (ReadId, forward_sequence_bytes, reverse_sequence_bytes).
            For single-end reads, reverse_sequence_bytes will be empty bytes.
        """
        logger = logging.getLogger(__name__)
        logger.info(
            f"Parsing files. Forward: {fwd_reads_path}, Reverse: {rev_reads_path or 'N/A'}"
        )

        fwd_format = SequenceFileProcessor.detect_file_format(fwd_reads_path)
        rev_format: Optional[str] = None

        if rev_reads_path:
            rev_format = SequenceFileProcessor.detect_file_format(rev_reads_path)
            if fwd_format != rev_format:
                raise ValueError(
                    f"Mismatched file formats: {fwd_reads_path} is {fwd_format}, "
                    f"but {rev_reads_path} is {rev_format}."
                )

        with SequenceFileProcessor.open_file_handle(fwd_reads_path) as fwd_fh:
            rev_fh: Optional[TextIO] = None
            try:
                if rev_reads_path:
                    rev_fh = SequenceFileProcessor.open_file_handle(rev_reads_path)

                # Create appropriate iterators based on format
                if fwd_format == "fasta":
                    fwd_iter = SeqIO.parse(fwd_fh, "fasta")
                    rev_iter = SeqIO.parse(rev_fh, "fasta") if rev_fh else None
                else:  # fastq
                    fwd_iter = FastqGeneralIterator(fwd_fh)
                    rev_iter = FastqGeneralIterator(rev_fh) if rev_fh else None

                for fwd_record in fwd_iter:
                    # Extract ID and sequence based on format
                    if fwd_format == "fasta":
                        fwd_id = fwd_record.id
                        fwd_seq_str = str(fwd_record.seq)
                    else:  # fastq
                        fwd_id, fwd_seq_str, _ = fwd_record

                    fwd_seq_bytes = fwd_seq_str.encode("utf-8")
                    rev_seq_bytes = b""

                    # Process reverse read if available
                    if rev_iter:
                        try:
                            rev_record = next(rev_iter)
                            if rev_format == "fasta":
                                rev_seq_str = str(rev_record.seq)
                            else:  # fastq
                                _, rev_seq_str, _ = rev_record
                            rev_seq_bytes = rev_seq_str.encode("utf-8")
                        except StopIteration:
                            logger.warning(
                                f"Reverse file ended before forward file at read {fwd_id}. "
                                "Treating remaining as single-end."
                            )
                            rev_iter = None

                    yield fwd_id, fwd_seq_bytes, rev_seq_bytes
            finally:
                if rev_fh:
                    rev_fh.close()


class CliArgs(BaseModel):
    """Pydantic model for command-line argument validation and management."""

    input_forward: Union[pathlib.Path, List[pathlib.Path]] = Field(
        description="Input forward read file(s) (FASTA/FASTQ, possibly gzipped)."
    )
    input_reverse: Optional[Union[pathlib.Path, List[pathlib.Path]]] = Field(
        default=None,
        description="Optional input reverse read file(s) for paired-end data.",
    )
    db_path: pathlib.Path = Field(
        description="Path to the k-mer database file (pickle format)."
    )
    num_processes: int = Field(
        default=DEFAULT_NUM_PROCESSES,
        ge=1,
        description="Number of processor cores to use.",
    )
    output_dir: pathlib.Path = Field(
        default=pathlib.Path("strainr_out"), description="Output directory."
    )
    disambiguation_mode: str = Field(
        default="max",
        pattern="^(random|max|multinomial|dirichlet)$",
        description="Selection mode for disambiguation of ambiguous reads.",
    )
    abundance_threshold: float = Field(
        default=DEFAULT_ABUNDANCE_THRESHOLD,
        ge=0.0,
        lt=1.0,
        description="Threshold for relative abundance filtering.",
    )
    perform_binning: bool = Field(
        default=False, description="Perform read binning (if implemented)."
    )
    save_raw_kmer_hits: bool = Field(
        default=False, description="Save raw k-mer hit counts per read to a CSV file."
    )
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE, ge=1, description="Chunk size for multiprocessing."
    )

    @field_validator("input_forward", "input_reverse", "db_path", mode="before")
    @classmethod
    def validate_paths_exist(
        cls, v: Union[str, List[str], None]
    ) -> Optional[Union[pathlib.Path, List[pathlib.Path]]]:
        """Ensures input file paths exist."""
        if v is None:
            return None

        if isinstance(v, list):
            return [cls._validate_single_path(path_str) for path_str in v]
        return cls._validate_single_path(str(v))

    @classmethod
    def _validate_single_path(cls, path_str: str) -> pathlib.Path:
        """Validates a single file path string."""
        path_obj = pathlib.Path(path_str)
        if not path_obj.exists():
            raise ValueError(f"File does not exist: {path_str}")
        if not path_obj.is_file():
            raise ValueError(f"Path is not a file: {path_str}")
        return path_obj

    @model_validator(mode="after")
    def validate_paired_read_consistency(self) -> "CliArgs":
        """Validates consistency between forward and reverse read file lists."""
        if self.input_reverse:
            fwd_is_list = isinstance(self.input_forward, list)
            rev_is_list = isinstance(self.input_reverse, list)

            if fwd_is_list != rev_is_list:
                raise ValueError(
                    "Both input_forward and input_reverse must be lists or both single files."
                )

            if fwd_is_list and rev_is_list:
                if len(self.input_forward) != len(self.input_reverse):
                    raise ValueError(
                        "Number of forward and reverse read files must match."
                    )
        return self


class AbundanceCalculator:
    """Handles abundance calculations and output formatting."""

    def __init__(self, strain_names: List[str], abundance_threshold: float):
        self.strain_names = strain_names
        self.abundance_threshold = abundance_threshold
        self.logger = logging.getLogger(__name__)

    def translate_assignments_to_names(
        self, final_assignments: Dict[ReadId, Union[StrainIndex, str]]
    ) -> Dict[ReadId, str]:
        """Translates final assignments (indices or 'NA') to strain names."""
        translated: Dict[ReadId, str] = {}
        for read_id, assignment in final_assignments.items():
            if isinstance(assignment, int):
                if 0 <= assignment < len(self.strain_names):
                    translated[read_id] = self.strain_names[assignment]
                else:
                    translated[read_id] = "Unassigned_BadIndex"
            else:
                translated[read_id] = str(assignment)
        return translated

    def calculate_abundances(
        self, final_named_assignments: Dict[ReadId, str], sample_name: str
    ) -> pd.DataFrame:
        """Calculates and formats abundance data into a DataFrame."""
        self.logger.info(f"Calculating abundances for sample: {sample_name}")

        hit_counts: Counter[str] = Counter(final_named_assignments.values())

        # Ensure all strains are represented
        for strain_name in self.strain_names:
            hit_counts.setdefault(strain_name, 0)
        hit_counts.setdefault("NA", 0)

        total_reads = sum(hit_counts.values())
        total_assigned = sum(
            count for strain, count in hit_counts.items() if strain != "NA"
        )

        table_data = []
        for strain_name in self.strain_names + ["NA"]:
            raw_hits = hit_counts[strain_name]

            # Sample relative abundance
            sample_relab = raw_hits / total_reads if total_reads > 0 else 0.0

            # Intra-sample relative abundance (for strains passing threshold)
            intra_relab = 0.0
            if strain_name != "NA" and sample_relab >= self.abundance_threshold:
                # Calculate denominator for intra-sample relab
                sum_hits_passing = sum(
                    count
                    for name, count in hit_counts.items()
                    if name != "NA"
                    and (count / total_reads if total_reads > 0 else 0)
                    >= self.abundance_threshold
                )
                if sum_hits_passing > 0:
                    intra_relab = raw_hits / sum_hits_passing

            table_data.append(
                {
                    "strain_name": strain_name,
                    "sample_hits": raw_hits,
                    "sample_relab": sample_relab,
                    "intra_relab": intra_relab if strain_name != "NA" else 0.0,
                }
            )

        abundance_df = pd.DataFrame(table_data).set_index("strain_name")
        abundance_df = abundance_df.sort_values(by="sample_hits", ascending=False)

        return abundance_df

    def save_abundance_report(
        self, abundance_df: pd.DataFrame, output_path: pathlib.Path
    ) -> None:
        """Save abundance report to TSV file."""
        abundance_df.to_csv(output_path, sep="\t", float_format="%.6f")
        self.logger.info(f"Abundance report saved to: {output_path}")

    def display_console_output(
        self, abundance_df: pd.DataFrame, sample_name: str, top_n: int = 10
    ) -> None:
        """Displays formatted abundance results to the console."""
        self.logger.info(f"Displaying top {top_n} abundances for sample: {sample_name}")
        print(f"\n--- Abundance Report for: {sample_name} ---")

        df_to_display = abundance_df[abundance_df["sample_hits"] > 0].copy()

        # Handle unassigned reads separately
        unassigned_info = ""
        if "NA" in df_to_display.index:
            na_row = df_to_display.loc["NA"]
            unassigned_info = (
                f"Unassigned Reads: {int(na_row['sample_hits'])} "
                f"({na_row['sample_relab']:.4f})"
            )
            df_to_display = df_to_display.drop(index="NA")

        print(df_to_display.head(top_n).to_string(float_format="%.4f"))
        if unassigned_info:
            print(unassigned_info)
        print("--- End of Report ---")


class KmerClassificationWorkflow:
    """Orchestrates the k-mer based strain classification workflow."""

    def __init__(self, args: CliArgs) -> None:
        """Initialize the workflow with validated arguments."""
        self.args = args
        self.database: Optional[KmerStrainDatabase] = None
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized KmerClassificationWorkflow")

    def _initialize_database(self) -> None:
        """Loads and initializes the KmerStrainDatabase."""
        self.logger.info(f"Loading k-mer database from: {self.args.db_path}")
        try:
            self.database = KmerStrainDatabase(self.args.db_path)
            self.logger.info(
                f"Database loaded: {self.database.num_kmers} k-mers, "
                f"{self.database.num_strains} strains, "
                f"k-mer length {self.database.kmer_length}."
            )
        except Exception as e:
            self.logger.error(f"Failed to load database: {e}")
            raise

    def _count_kmers_for_read(
        self, record_tuple: Tuple[ReadId, bytes, bytes]
    ) -> Tuple[ReadId, CountVector]:
        """Processes a single read to count k-mer occurrences against the database."""
        if self.database is None:
            raise RuntimeError("Database not initialized.")

        read_id, fwd_seq_bytes, rev_seq_bytes = record_tuple

        all_kmers: Set[bytes] = set()

        # Process forward sequence
        if fwd_seq_bytes:
            kmers_fwd = KMER_EXTRACTOR.extract_kmers(
                fwd_seq_bytes, self.database.kmer_length
            )
            all_kmers.update(kmers_fwd)

        # Process reverse sequence
        if rev_seq_bytes:
            kmers_rev = KMER_EXTRACTOR.extract_kmers(
                rev_seq_bytes, self.database.kmer_length
            )
            all_kmers.update(kmers_rev)

        # Count strain hits
        strain_counts = np.zeros(self.database.num_strains, dtype=np.uint8)

        for kmer_bytes in all_kmers:
            kmer_strain_counts = self.database.get_strain_counts_for_kmer(kmer_bytes)
            if kmer_strain_counts is not None:
                strain_counts += kmer_strain_counts

        return read_id, strain_counts

    def _classify_reads_parallel(
        self,
        fwd_reads_path: pathlib.Path,
        rev_reads_path: Optional[pathlib.Path] = None,
    ) -> ReadHitResults:
        """Classifies reads from input files in parallel."""
        self.logger.info(f"Starting parallel classification for {fwd_reads_path}")

        sequence_iterator = SequenceFileProcessor.parse_sequence_files(
            fwd_reads_path, rev_reads_path
        )

        with mp.Pool(processes=self.args.num_processes) as pool:
            results = pool.map(
                self._count_kmers_for_read,
                sequence_iterator,
                chunksize=self.args.chunk_size,
            )

        self.logger.info(f"Finished parallel classification for {fwd_reads_path}")
        return results

    def _save_raw_kmer_hits(
        self, raw_hit_results: ReadHitResults, sample_name: str
    ) -> None:
        """Saves raw k-mer hit counts per read to a CSV file."""
        if not raw_hit_results or self.database is None:
            self.logger.warning(f"No raw hit results to save for sample: {sample_name}")
            return

        self.logger.info(f"Saving raw k-mer hit counts for sample: {sample_name}")

        # Prepare data for DataFrame
        df_data = []
        for read_id, count_vector in raw_hit_results:
            row = {"read_id": read_id}
            for i, strain_name in enumerate(self.database.strain_names):
                row[strain_name] = count_vector[i]
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Save to CSV
        raw_hits_dir = self.args.output_dir / "raw_kmer_hits"
        raw_hits_dir.mkdir(parents=True, exist_ok=True)

        output_path = raw_hits_dir / f"{sample_name}_raw_hits.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"Raw k-mer hit counts saved to: {output_path}")

    def _get_sample_name(self, file_path: pathlib.Path) -> str:
        """Extract sample name from file path."""
        # Remove common suffixes and prefixes
        name = file_path.name
        for suffix in [".fastq", ".fasta", ".fq", ".fa", ".gz"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]

        # Remove common read pair indicators
        for pattern in ["_R1", "_R2", "_1", "_2"]:
            if pattern in name:
                name = name.split(pattern)[0]
                break

        return name

    def run_workflow(self) -> None:
        """Executes the main classification and analysis workflow."""
        self.logger.info("Starting StrainR workflow")

        try:
            # Initialize database
            self._initialize_database()
            if self.database is None:
                raise RuntimeError("Database initialization failed")

            # Initialize analyzer and abundance calculator
            analyzer = ClassificationAnalyzer(
                strain_names=self.database.strain_names,
                disambiguation_mode=self.args.disambiguation_mode,
                abundance_threshold=self.args.abundance_threshold,
                num_processes=self.args.num_processes,
            )

            abundance_calc = AbundanceCalculator(
                strain_names=self.database.strain_names,
                abundance_threshold=self.args.abundance_threshold,
            )

            # Prepare file lists
            fwd_files = (
                [self.args.input_forward]
                if isinstance(self.args.input_forward, pathlib.Path)
                else self.args.input_forward
            )
            rev_files = []
            if self.args.input_reverse:
                rev_files = (
                    [self.args.input_reverse]
                    if isinstance(self.args.input_reverse, pathlib.Path)
                    else self.args.input_reverse
                )

            # Ensure output directory exists
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

            # Process each sample
            for i, fwd_reads_path in enumerate(fwd_files):
                sample_name = self._get_sample_name(fwd_reads_path)
                self.logger.info(
                    f"Processing sample: {sample_name} (File: {fwd_reads_path.name})"
                )

                rev_reads_path: Optional[pathlib.Path] = None
                if rev_files and i < len(rev_files):
                    rev_reads_path = rev_files[i]

                # 1. Classify reads
                raw_kmer_hits = self._classify_reads_parallel(
                    fwd_reads_path, rev_reads_path
                )

                if self.args.save_raw_kmer_hits:
                    self._save_raw_kmer_hits(raw_kmer_hits, sample_name)

                # 2. Analyze classification results
                clear_hits, ambiguous_hits, no_hit_ids = (
                    analyzer.separate_hit_categories(raw_kmer_hits)
                )
                clear_assignments = analyzer.resolve_clear_hits_to_indices(clear_hits)

                # Calculate priors and resolve ambiguous hits
                prior_counts = analyzer.calculate_strain_prior_from_assignments(
                    clear_assignments
                )
                prior_probs = analyzer.convert_prior_counts_to_probability_vector(
                    prior_counts
                )
                resolved_ambiguous = analyzer.resolve_ambiguous_hits_parallel(
                    ambiguous_hits, prior_probs
                )

                # Combine all assignments
                final_assignments = analyzer.combine_assignments(
                    clear_assignments,
                    resolved_ambiguous,
                    no_hit_ids,
                    unassigned_marker="NA",
                )

                # 3. Calculate and output abundances
                final_named_assignments = abundance_calc.translate_assignments_to_names(
                    final_assignments
                )
                abundance_df = abundance_calc.calculate_abundances(
                    final_named_assignments, sample_name
                )

                # Save and display results
                output_path = (
                    self.args.output_dir / f"{sample_name}_abundance_report.tsv"
                )
                abundance_calc.save_abundance_report(abundance_df, output_path)
                abundance_calc.display_console_output(abundance_df, sample_name)

            self.logger.info("StrainR workflow completed successfully")

        except Exception as e:
            self.logger.critical(f"Workflow failed: {e}", exc_info=True)
            raise


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with appropriate level and format."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("strainr_classify.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_cli_arguments() -> CliArgs:
    """Parses and validates command-line arguments."""
    parser = argparse.ArgumentParser(
        description="StrainR: K-mer based strain classification tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_forward",
        help="Input forward read file(s) (FASTA/FASTQ, possibly gzipped)",
        nargs="+",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--input_reverse",
        help="Optional input reverse read file(s) for paired-end data",
        nargs="*",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--db",
        help="Path to the k-mer database file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--procs",
        help="Number of processor cores to use",
        type=int,
        default=DEFAULT_NUM_PROCESSES,
    )
    parser.add_argument(
        "--out",
        help="Output directory",
        type=str,
        default="strainr_out",
    )
    parser.add_argument(
        "--mode",
        help="Selection mode for disambiguation of ambiguous reads",
        choices=["random", "max", "multinomial", "dirichlet"],
        default="max",
    )
    parser.add_argument(
        "--thresh",
        help="Minimum relative abundance threshold",
        type=float,
        default=DEFAULT_ABUNDANCE_THRESHOLD,
    )
    parser.add_argument(
        "--bin",
        help="Perform read binning",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_raw_hits",
        help="Save raw k-mer hit counts per read to CSV",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--chunk_size",
        help="Chunk size for multiprocessing",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )
    parser.add_argument(
        "--verbose",
        help="Enable conda install pydanticverbose logging",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # Set up logging early
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    logger.info("Parsing command-line arguments")

    try:
        cli_args = CliArgs(
            input_forward=args.input_forward,
            input_reverse=args.input_reverse,
            db_path=args.db,
            num_processes=args.procs,
            output_dir=args.out,
            disambiguation_mode=args.mode,
            abundance_threshold=args.thresh,
            perform_binning=args.bin,
            save_raw_kmer_hits=args.save_raw_hits,
            chunk_size=args.chunk_size,
        )
        logger.info("Arguments validated successfully")
        return cli_args

    except ValueError as e:
        logger.error(f"Argument validation error: {e}")
        parser.print_help()
        sys.exit(1)


def main() -> None:
    """Main entry point for the StrainR classification script."""
    try:
        # Parse and validate arguments
        cli_args = parse_cli_arguments()

        # Initialize and run workflow
        workflow = KmerClassificationWorkflow(cli_args)
        workflow.run_workflow()

        logging.getLogger(__name__).info(
            "StrainR classification completed successfully"
        )

    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).critical(f"Critical error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
