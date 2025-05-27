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
from typing import Callable, Dict, Generator, List, Optional, Set, TextIO, Tuple, Union

import numpy as np
import pandas as pd

try:
    from Bio import SeqIO
    from Bio.SeqIO.QualityIO import FastqGeneralIterator
except ImportError as e:
    raise ImportError(
        "Biopython is required for this script. Please install it with 'pip install biopython'."
    ) from e
try:
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError as e:
    raise ImportError(
        "Pydantic is required for this script. Please install it with 'pip install pydantic'."
    ) from e

# Assuming these local modules are structured correctly within the 'strainr' package
from strainr.analyze import ClassificationAnalyzer
from strainr.genomic_types import CountVector, ReadId, StrainIndex
from strainr.database import StrainKmerDatabase # Updated import

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
    def _process_read_pair(
        fwd_record, rev_iter, logger
    ) -> Generator[Tuple[str, bytes, bytes], None, None]:
        """Processes a forward read and its corresponding reverse read."""
        if fwd_record is None:
            return
        fwd_id = (
            getattr(fwd_record, "id", "")
            if hasattr(fwd_record, "id")
            else fwd_record[0]
        )
        if not isinstance(fwd_id, str):
            fwd_id = str(fwd_id)
        fwd_seq_str = (
            str(getattr(fwd_record, "seq", ""))
            if hasattr(fwd_record, "seq")
            else fwd_record[1]
        )
        fwd_seq_bytes = fwd_seq_str.encode("utf-8")
        rev_seq_bytes = b""
        if rev_iter:
            try:
                rev_record = next(rev_iter)
                rev_seq_str = (
                    str(getattr(rev_record, "seq", ""))
                    if hasattr(rev_record, "seq")
                    else rev_record[1]
                )
                rev_seq_bytes = rev_seq_str.encode("utf-8")
            except StopIteration:
                logger.warning(
                    f"Reverse file ended before forward file at read {fwd_id}. Treating remaining as single-end."
                )
                rev_iter = None
        yield fwd_id, fwd_seq_bytes, rev_seq_bytes

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
            if rev_reads_path:
                rev_fh = SequenceFileProcessor.open_file_handle(rev_reads_path)
            try:
                if fwd_format == "fasta":
                    fwd_iter = SeqIO.parse(fwd_fh, "fasta")
                    rev_iter = SeqIO.parse(rev_fh, "fasta") if rev_fh else None
                    for fwd_record in fwd_iter:
                        yield from SequenceFileProcessor._process_read_pair(
                            fwd_record, rev_iter, logger
                        )
                else:  # fastq
                    fwd_iter = FastqGeneralIterator(fwd_fh)
                    rev_iter = FastqGeneralIterator(rev_fh) if rev_fh else None
                    for fwd_record in fwd_iter:
                        yield from SequenceFileProcessor._process_read_pair(
                            fwd_record, rev_iter, logger
                        )
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
        description="Path to the k-mer database file (Parquet format)."
    )
    num_processes: int = Field(
        default=DEFAULT_NUM_PROCESSES,
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
        description="Threshold for relative abundance filtering.",
    )
    perform_binning: bool = Field(
        default=False, description="Perform read binning (if implemented)."
    )
    save_raw_kmer_hits: bool = Field(
        default=False, description="Save raw k-mer hit counts per read to a CSV file."
    )
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE, description="Chunk size for multiprocessing."
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
                if (
                    isinstance(self.input_forward, list)
                    and isinstance(self.input_reverse, list)
                    and len(self.input_forward) != len(self.input_reverse)
                ):
                    raise ValueError(
                        "Number of forward and reverse read files must match."
                    )
        return self

    def __init__(self, **data):
        super().__init__()


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
        # sum(count for strain, count in hit_counts.items() if strain != "NA") # This sum was unused

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

            table_data.append({
                "strain_name": strain_name,
                "sample_hits": raw_hits,
                "sample_relab": sample_relab,
                "intra_relab": intra_relab if strain_name != "NA" else 0.0,
            })

        abundance_df = pd.DataFrame(table_data).set_index("strain_name")
        abundance_df = abundance_df.sort_values(by="sample_hits", ascending=False)

        return abundance_df

    def save_abundance_report(
        self, abundance_df: pd.DataFrame, output_path: pathlib.Path
    ) -> None:
        """Save abundance report to TSV file."""
        abundance_df.to_csv(output_path, sep="	", float_format="%.6f")
        self.logger.info(f"Abundance report saved to: {output_path}")

    def display_console_output(
        self, abundance_df: pd.DataFrame, sample_name: str, top_n: int = 10
    ) -> None:
        """Displays formatted abundance results to the console."""
        self.logger.info(f"Displaying top {top_n} abundances for sample: {sample_name}")
        print(f"
--- Abundance Report for: {sample_name} ---")

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
        self.database: Optional[StrainKmerDatabase] = None
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized KmerClassificationWorkflow")

    def _initialize_database(self) -> None:
        """Loads and initializes the StrainKmerDatabase."""
        self.logger.info(f"Loading k-mer database from: {self.args.db_path}")
        try:
            self.database = StrainKmerDatabase(self.args.db_path)
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
        if self.database is None:
            raise RuntimeError(
                "Database is not initialized before accessing its attributes."
            )
        for kmer_bytes in all_kmers:
            kmer_strain_counts = self.database.get_strain_counts_for_kmer(kmer_bytes)
            if kmer_strain_counts is not None:
                strain_counts += kmer_strain_counts
        # Ensure read_id is str
        if not isinstance(read_id, str):
            read_id = str(read_id)
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
        if self.database is None:
            raise RuntimeError(
                "Database is not initialized before accessing its attributes."
            )
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
        help="Enable verbose logging",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # Set up logging early
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    logger.info("Parsing command-line arguments")

    try:
        # Map CLI args to CliArgs model, handling lists and optionals
        input_forward = args.input_forward
        if isinstance(input_forward, list) and len(input_forward) == 1:
            input_forward = input_forward[0]
        input_reverse = args.input_reverse
        if input_reverse is not None:
            if len(input_reverse) == 0:
                input_reverse = None
            elif len(input_reverse) == 1:
                input_reverse = input_reverse[0]

        cli_args = CliArgs(
            input_forward=input_forward,
            input_reverse=input_reverse,
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
```

**2. File Path: `src/strainr/convert_db_format.py`**
New Content:
```python
"""
Converts a StrainKmerDatabase from pickle format to Parquet format.

This script provides a command-line interface to load a pandas DataFrame
from a pickle file and save it to a Parquet file, ensuring the index
(which typically contains k-mers) is preserved.
"""

import argparse
import pandas as pd
from pathlib import Path
import pickle # Added import

def convert_pickle_to_parquet(pickle_path: Path, parquet_path: Path) -> None:
    """
    Loads a DataFrame from a pickle file and saves it to a Parquet file.

    Args:
        pickle_path: Path to the input pickle file.
        parquet_path: Path for the output Parquet file.

    Raises:
        FileNotFoundError: If the pickle_path does not exist.
        Exception: For other errors during file loading or saving.
    """
    try:
        print(f"Loading DataFrame from pickle file: {pickle_path}")
        df = pd.read_pickle(pickle_path)
        
        if not isinstance(df, pd.DataFrame):
            print(f"Error: The file {pickle_path} did not contain a pandas DataFrame.")
            return

        print(f"Saving DataFrame to Parquet file: {parquet_path}")
        # Ensure the directory for the parquet file exists
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=True)
        
        print(f"Successfully converted {pickle_path} to {parquet_path}")

    except FileNotFoundError:
        print(f"Error: Input pickle file not found at {pickle_path}")
    except pd.errors.EmptyDataError:
        print(f"Error: The pickle file {pickle_path} is empty or does not contain a valid DataFrame.")
    except pickle.UnpicklingError: 
        print(f"Error: Failed to unpickle data from {pickle_path}. The file may be corrupted or not a pickle file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a StrainKmerDatabase from pickle format to Parquet format."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input pickle database file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path for the output Parquet database file.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    convert_pickle_to_parquet(input_path, output_path)
```

**3. File Path: `src/strainr/database.py`**
New Content:
```python
"""
K-mer database management for strain classification.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any # Tuple removed

import numpy as np
import pandas as pd  # For pd.errors.EmptyDataError

from strainr.genomic_types import (  # KmerCountDict is Dict[bytes, CountVector]
    CountVector,
    KmerCountDict,
)


class StrainKmerDatabase:
    """
    K-mer database for strain classification.

    This class manages a database of k-mer frequencies across multiple strains,
    enabling efficient lookup of strain-specific k-mer signatures.

    Attributes:
        database_path: Path to the Parquet database file
        kmer_length: Length of k-mers in the database
        strain_names: List of strain identifiers
        kmer_lookup_dict: Dictionary mapping k-mers to strain frequency vectors
        num_strains: Number of strains in database
        num_kmers: Number of unique k-mers in database
    """

    def __init__(self, database_path: Union[str, Path], kmer_length: int = 31) -> None:
        """
        Initialize strain database from a Parquet file.

        Args:
            database_path: Path to the Parquet file. The DataFrame stored in Parquet
                           is expected to have k-mers (typically strings or bytes)
                           as its index and strain names (strings) as its columns.
                           Cell values should be k-mer counts (numeric, convertible to np.uint8).
            kmer_length: Expected length of k-mers. This is validated against the
                         first k-mer found in the database. If a mismatch occurs,
                         a warning is printed, and the database's k-mer length is used.

        Raises:
            FileNotFoundError: If the database_path does not point to a valid file.
            RuntimeError: If loading or processing the database fails due to issues
                          like file corruption, incorrect format, empty data, or unexpected data types.
            ValueError: If the loaded database is empty or if k-mer length validation
                        reveals issues (though currently it warns and updates self.kmer_length).

        Example:
            >>> # db = StrainKmerDatabase("path/to/your/database.parquet", kmer_length=31)
            >>> # print(f"Loaded {db.num_strains} strains with {db.num_kmers} k-mers of length {db.kmer_length}")
        """
        self.database_path = (
            Path(database_path).resolve().expanduser()
        )  # Use resolve for absolute path
        self.kmer_length = kmer_length

        # Initialize attributes that will be set in _load_database or elsewhere
        self.strain_names: List[str] = []
        self.kmer_lookup_dict: KmerCountDict = {}  # Dict[bytes, CountVector]
        self.num_strains: int = 0
        self.num_kmers: int = 0

        self._validate_database_file()
        self._load_database()

    def _validate_database_file(self) -> None:
        """Validate that database_path points to an existing file."""
        if not self.database_path.is_file():  # Check if it's a file, not just exists
            raise FileNotFoundError(
                f"Database file not found or is not a file: {self.database_path}"
            )

    def _load_database(self) -> None:
        """Load and validate the k-mer database from Parquet file.

        Raises:
            RuntimeError: For errors during file reading or DataFrame processing.
            ValueError: If the database DataFrame is empty or structure is invalid.
        """
        print(f"Loading k-mer database from {self.database_path} (Parquet format)...")

        try:
            # It's common for k-mers to be strings in DataFrames from bioinformatics tools
            kmer_frequency_dataframe: pd.DataFrame = pd.read_parquet(self.database_path)
        except (
            FileNotFoundError
        ):  # Should be caught by _validate_database_file, but good practice
            raise RuntimeError(
                f"Database file disappeared after validation: {self.database_path}"
            )
        except (IOError, ValueError, pd.errors.EmptyDataError) as e: 
            raise RuntimeError(
                f"Failed to read or process Parquet database from {self.database_path}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Parquet database from {self.database_path} due to an unexpected error: {e}"
            ) from e

        if not isinstance(kmer_frequency_dataframe, pd.DataFrame):
            raise RuntimeError(
                f"Loaded database from {self.database_path} is not a pandas DataFrame. "
                f"Found type: {type(kmer_frequency_dataframe)}."
            )

        if kmer_frequency_dataframe.empty:
            raise ValueError(f"Database DataFrame from {self.database_path} is empty.")

        # Extract database components
        self.strain_names = list(kmer_frequency_dataframe.columns.astype(str))

        if not kmer_frequency_dataframe.index.is_unique:
            print(
                f"Warning: K-mer index in {self.database_path} is not unique. Counts for duplicate k-mers might be based on their last occurrence in the input DataFrame during conversion to dictionary."
            )

        kmer_sequences_from_index = kmer_frequency_dataframe.index

        if len(kmer_sequences_from_index) == 0:
            raise ValueError(
                f"Database DataFrame from {self.database_path} has no k-mers (empty index)."
            )

        first_kmer_in_index = kmer_sequences_from_index[0]
        kmer_needs_encoding: bool
        if isinstance(first_kmer_in_index, str):
            actual_kmer_length = len(first_kmer_in_index)
            kmer_needs_encoding = True
        elif isinstance(first_kmer_in_index, bytes):
            actual_kmer_length = len(first_kmer_in_index)
            kmer_needs_encoding = False
        else:
            raise ValueError(
                f"Unsupported k-mer type in DataFrame index: {type(first_kmer_in_index)}. "
                "Expected str or bytes."
            )

        if actual_kmer_length == 0:
            raise ValueError(
                f"First k-mer in database '{first_kmer_in_index}' has zero length. This is invalid."
            )

        if actual_kmer_length != self.kmer_length:
            print(
                f"Warning: Initial expected k-mer length was {self.kmer_length} for database {self.database_path}, "
                f"but found {actual_kmer_length} (based on first k-mer: '{first_kmer_in_index}'). "
                f"Using actual length from database: {actual_kmer_length}."
            )
            self.kmer_length = actual_kmer_length

        for idx, kmer_val in enumerate(kmer_sequences_from_index):
            if len(kmer_val) != self.kmer_length:
                raise ValueError(
                    f"Inconsistent k-mer length found in database {self.database_path} at index position {idx}. "
                    f"Expected {self.kmer_length}, found k-mer '{kmer_val}' (type: {type(kmer_val)}) with length {len(kmer_val)}."
                )
            if kmer_needs_encoding and not isinstance(kmer_val, str):
                raise ValueError(
                    f"K-mer at index position {idx} ('{kmer_val}') is type {type(kmer_val)}, expected str based on first k-mer."
                )
            if not kmer_needs_encoding and not isinstance(kmer_val, bytes):
                raise ValueError(
                    f"K-mer at index position {idx} ('{kmer_val}') is type {type(kmer_val)}, expected bytes based on first k-mer."
                )

        try:
            frequency_matrix = kmer_frequency_dataframe.to_numpy(dtype=np.uint8)
        except ValueError as e:
            raise RuntimeError(
                f"Could not convert DataFrame values to np.uint8 for {self.database_path}. Ensure all counts are valid integers within 0-255. Error: {e}"
            ) from e
        except Exception as e: 
            raise RuntimeError(
                f"Unexpected error converting DataFrame to NumPy array for {self.database_path}: {e}"
            ) from e

        self.kmer_lookup_dict.clear()
        skipped_kmers_count = 0
        for i, kmer_in_idx in enumerate(kmer_sequences_from_index):
            kmer_bytes: bytes
            if kmer_needs_encoding:
                try:
                    kmer_bytes = str(kmer_in_idx).encode("utf-8")
                except UnicodeEncodeError as e:
                    print(
                        f"Warning: Failed to encode k-mer '{kmer_in_idx}' (index {i}) to UTF-8 bytes: {e}. Skipping this k-mer."
                    )
                    skipped_kmers_count += 1
                    continue
            else:
                kmer_bytes = bytes(kmer_in_idx)

            if len(kmer_bytes) != self.kmer_length:
                print(
                    f"Warning: K-mer '{kmer_in_idx}' (index {i}) resulted in byte length {len(kmer_bytes)} after encoding/casting, expected {self.kmer_length}. Skipping this k-mer."
                )
                skipped_kmers_count += 1
                continue
            self.kmer_lookup_dict[kmer_bytes] = frequency_matrix[i]

        self.num_strains = len(self.strain_names)
        self.num_kmers = len(self.kmer_lookup_dict)

        if self.num_kmers == 0 and len(kmer_sequences_from_index) > 0:
            raise ValueError(
                f"No k-mers were successfully loaded into the lookup dictionary from {self.database_path} "
                f"(skipped {skipped_kmers_count} out of {len(kmer_sequences_from_index)}). "
                "This might be due to encoding issues or length mismatches after encoding. Check warnings."
            )

        print(
            f"Successfully loaded database: {self.num_strains} strains, "
            f"{self.num_kmers} k-mers (k={self.kmer_length}). "
            f"Skipped {skipped_kmers_count} k-mers during loading."
            if skipped_kmers_count > 0
            else ""
        )

    def lookup_kmer(self, kmer_sequence: bytes) -> Optional[CountVector]:
        """
        Look up strain frequency vector for a given k-mer.

        Args:
            kmer_sequence: K-mer sequence as bytes

        Returns:
            A NumPy array representing the CountVector for the k-mer if found,
            otherwise None.
        """
        if not isinstance(kmer_sequence, bytes):
            pass 
        return self.kmer_lookup_dict.get(kmer_sequence)

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loaded k-mer database.

        Returns:
            A dictionary containing key database statistics:
                - "num_strains" (int): Number of strains in the database.
                - "num_kmers" (int): Number of unique k-mers loaded into the lookup dictionary.
                - "kmer_length" (int): The validated length of k-mers in the database.
                - "database_path" (str): Absolute path to the database file.
                - "strain_names_preview" (List[str]): A preview of the first 5 strain names.
                - "total_strain_names" (int): Total number of strains (should match "num_strains").
        """
        return {
            "num_strains": self.num_strains,
            "num_kmers": self.num_kmers,
            "kmer_length": self.kmer_length,
            "database_path": str(
                self.database_path.resolve()
            ), 
            "strain_names_preview": self.strain_names[:5],
            "total_strain_names": len(self.strain_names),
        }

    def validate_kmer_length(self, test_kmer: Union[str, bytes]) -> bool:
        """
        Validates if a given k-mer (string or bytes) matches the database's k-mer length.

        If the input `test_kmer` is a string, its direct length is checked.
        If it's bytes, its byte length is checked. This method does not perform
        encoding; it assumes the provided form (str or bytes) is what needs checking.
        For checking against the database's internal byte representation of k-mers,
        ensure `test_kmer` is passed as bytes or handle encoding prior to calling.

        Args:
            test_kmer: The k-mer (string or bytes) to validate.

        Returns:
            True if the length of `test_kmer` matches `self.kmer_length`,
            False otherwise.
        """
        if not isinstance(test_kmer, (str, bytes)):
            return False
        return len(test_kmer) == self.kmer_length
```

**4. File Path: `src/strainr/kmer_database.py`**
New Content:
```python
import pathlib
from typing import Optional, Any # Tuple removed
from typing import List, Dict, Union


import numpy as np
import pandas as pd


class StrainKmerDb: 
    """
    Represents a database of k-mers and their corresponding strain frequency vectors.

    This class loads a k-mer database from a Parquet file. The DataFrame stored in Parquet
    is expected to have k-mers as its index (typically strings or bytes) and strain names
    as its columns. The values should be counts or frequencies (convertible to uint8).

    Attributes:
        database_filepath (pathlib.Path): Absolute path to the database file.
        kmer_length (int): The length of k-mers in the database. Determined from data
                           if not provided, or validated if provided.
        kmer_to_counts_map (KmerDatabaseDict): A dictionary mapping each k-mer (bytes)
                                             to its count vector (np.ndarray[np.uint8]).
        strain_names (List[str]): A list of strain names present in the database.
        num_strains (int): The number of strains in the database.
        num_kmers (int): The number of unique k-mers successfully loaded into the database.
    """

    def __init__(
        self,
        database_filepath: Union[str, pathlib.Path],
        expected_kmer_length: Optional[int] = None,
    ) -> None:
        """
        Initializes and loads the StrainKmerDb from a file.

        Args:
            database_filepath: Path to the Parquet file containing the k-mer database.
                               The DataFrame stored in Parquet should have k-mers (strings or bytes)
                               as its index and strain names (strings) as its columns.
                               Cell values should be numeric and convertible to `np.uint8`.
            expected_kmer_length: Optional. If provided, this length is enforced. K-mers in the
                                  database must match this length. If None, the k-mer length is
                                  inferred from the first k-mer in the database and then
                                  enforced for all other k-mers.

        Raises:
            FileNotFoundError: If the `database_filepath` does not exist or is not a file.
            ValueError: If the database is empty, if k-mers have inconsistent lengths,
                        if `expected_kmer_length` is provided and does not match the k-mer
                        lengths in the file, or if other data validation checks fail.
            TypeError: If the data in the DataFrame is not of the expected type (e.g., k-mer
                       index contains types other than str/bytes, or counts are not
                       convertible to uint8).
            RuntimeError: For lower-level issues during file reading (e.g., corrupted Parquet file),
                          often wrapping underlying exceptions like `IOError`, `ValueError`,
                          or specific PyArrow errors.
        """
        self.database_filepath = pathlib.Path(
            database_filepath
        ).resolve() 
        if not self.database_filepath.is_file():
            raise FileNotFoundError(
                f"Database file not found or is not a file: {self.database_filepath}"
            )

        self.kmer_length: int = 0
        self.kmer_to_counts_map = {}
        self.strain_names: List[str] = []
        self.num_strains: int = 0
        self.num_kmers: int = 0

        self._load_database(expected_kmer_length)

        print(
            f"Successfully loaded database from {self.database_filepath}
"
            f" - K-mer length: {self.kmer_length}
"
            f" - Number of k-mers: {self.num_kmers}
"
            f" - Number of strains: {self.num_strains}
"
            f" - Strain names preview: {', '.join(self.strain_names[:5])}{'...' if len(self.strain_names) > 5 else ''}"
        )

    def _load_database(self, expected_kmer_length: Optional[int]) -> None:
        """
        Internal method to load data from the Parquet database file.
        """
        print(f"Loading k-mer database from {self.database_filepath} (Parquet format)...")
        try:
            kmer_strain_df: pd.DataFrame = pd.read_parquet(self.database_filepath)
        except (IOError, ValueError, pd.errors.EmptyDataError) as e: 
            raise RuntimeError(
                f"Failed to read or process Parquet database file: {self.database_filepath}. File may be corrupted, empty, or not a valid Parquet file. Original error: {e}"
            ) from e
        except FileNotFoundError: 
            raise RuntimeError(
                f"Database file {self.database_filepath} vanished after initial check."
            ) from None
        except Exception as e: 
            raise RuntimeError(
                f"An unexpected error occurred while reading Parquet file {self.database_filepath}: {e}"
            ) from e

        if not isinstance(kmer_strain_df, pd.DataFrame):
            raise RuntimeError(
                f"Data loaded from {self.database_filepath} is not a pandas DataFrame (type: {type(kmer_strain_df)})."
            )

        if kmer_strain_df.empty:
            raise ValueError(f"Loaded database is empty: {self.database_filepath}")

        self.strain_names = list(kmer_strain_df.columns.astype(str))
        self.num_strains = len(self.strain_names)
        if self.num_strains == 0:
            raise ValueError("Database contains no strain information (no columns).")

        if kmer_strain_df.index.empty:
            raise ValueError("Database contains no k-mers (empty index).")

        if not kmer_strain_df.index.is_unique:
            print(
                f"Warning: K-mer index in {self.database_filepath} is not unique. Duplicates will be resolved by last occurrence when creating the lookup dictionary."
            )

        first_kmer_obj = kmer_strain_df.index[0]
        kmer_type_is_str: bool
        inferred_k_len: int
        if isinstance(first_kmer_obj, str):
            inferred_k_len = len(first_kmer_obj)
            kmer_type_is_str = True
        elif isinstance(first_kmer_obj, bytes):
            inferred_k_len = len(first_kmer_obj)
            kmer_type_is_str = False
        else:
            raise TypeError(
                f"Unsupported k-mer type in DataFrame index: {type(first_kmer_obj)}. Expected str or bytes."
            )

        if inferred_k_len == 0:
            raise ValueError(
                "First k-mer in database has zero length, which is invalid."
            )

        if expected_kmer_length is not None:
            if expected_kmer_length != inferred_k_len:
                raise ValueError(
                    f"Provided expected_kmer_length ({expected_kmer_length}) does not match "
                    f"length of first k-mer in database ({inferred_k_len})."
                )
            self.kmer_length = expected_kmer_length
        else:
            self.kmer_length = inferred_k_len
            print(f"K-mer length inferred from first k-mer: {self.kmer_length}")

        if (self.kmer_length <= 0):
            raise ValueError(
                f"Determined k-mer length ({self.kmer_length}) must be positive."
            )

        temp_kmer_map = {}
        try:
            count_matrix = kmer_strain_df.to_numpy(dtype=np.uint8)
        except ValueError as e:
            raise TypeError(
                f"Could not convert database values to count matrix (np.uint8). "
                f"Ensure all values are numeric and within 0-255. Error: {e}"
            ) from e

        skipped_kmers_count = 0
        for i, kmer_obj in enumerate(kmer_strain_df.index):
            kmer_bytes: bytes
            current_obj_len: int

            if kmer_type_is_str:
                if not isinstance(kmer_obj, str):
                    print(
                        f"Warning: Inconsistent k-mer type at index {i}. Expected str, got {type(kmer_obj)}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
                current_obj_len = len(kmer_obj)
                if current_obj_len != self.kmer_length:
                    print(
                        f"Warning: Inconsistent k-mer string length at index {i}. Expected {self.kmer_length}, k-mer '{kmer_obj}' has length {current_obj_len}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
                try:
                    kmer_bytes = kmer_obj.encode("utf-8")
                except UnicodeEncodeError as e:
                    print(
                        f"Warning: Failed to encode k-mer string '{kmer_obj}' (index {i}) to UTF-8 bytes. Error: {e}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
            else: 
                if not isinstance(kmer_obj, bytes):
                    print(
                        f"Warning: Inconsistent k-mer type at index {i}. Expected bytes, got {type(kmer_obj)}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
                kmer_bytes = kmer_obj
                current_obj_len = len(kmer_bytes)
                if current_obj_len != self.kmer_length:
                    print(
                        f"Warning: Inconsistent k-mer bytes length at index {i}. Expected {self.kmer_length}, k-mer {kmer_bytes!r} has length {current_obj_len}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
            temp_kmer_map[kmer_bytes] = count_matrix[i]

        self.kmer_to_counts_map = temp_kmer_map
        self.num_kmers = len(self.kmer_to_counts_map)

        if (
            self.num_kmers == 0
            and not kmer_strain_df.index.empty
            and skipped_kmers_count == len(kmer_strain_df.index)
        ):
            raise ValueError(
                f"No k-mers were successfully loaded into the database from {self.database_filepath} "
                f"(all {skipped_kmers_count} k-mers from input were skipped). "
                "This is likely due to encoding issues or consistent length mismatches. Check warnings."
            )
        if skipped_kmers_count > 0:
            print(
                f"Warning: Skipped {skipped_kmers_count} k-mers during loading due to type, length, or encoding issues."
            )

    def get_strain_counts_for_kmer(self, kmer: bytes) -> Optional[np.ndarray]:
        """
        Retrieves the strain count vector for a given k-mer. (Equivalent to lookup_kmer)

        Args:
            kmer: The k-mer (bytes) to look up.

        Returns:
            A NumPy array (CountVector) of uint8 counts for each strain if the
            k-mer is found, otherwise None.
        """
        if not isinstance(kmer, bytes):
            return None
        return self.kmer_to_counts_map.get(kmer)

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loaded k-mer database.
        (Method merged from StrainKmerDatabase in database.py)

        Returns:
            A dictionary containing key database statistics:
                - "num_strains" (int): Number of strains in the database.
                - "num_kmers" (int): Number of unique k-mers loaded.
                - "kmer_length" (int): The validated length of k-mers in the database.
                - "database_filepath" (str): Absolute path to the database file.
                - "strain_names_preview" (List[str]): A preview of the first 5 strain names.
                - "total_strain_names" (int): Total count of strain names.
        """
        return {
            "num_strains": self.num_strains,
            "num_kmers": self.num_kmers,
            "kmer_length": self.kmer_length,
            "database_filepath": str(self.database_filepath), 
            "strain_names_preview": self.strain_names[:5],
            "total_strain_names": self.num_strains,
        }

    def validate_kmer_length(self, test_kmer: Union[str, bytes]) -> bool:
        """
        Validates if a given k-mer (string or bytes) matches the database's k-mer length.
        (Method merged from StrainKmerDatabase in database.py)

        If the input `test_kmer` is a string, its direct length is checked.
        If it's bytes, its byte length is checked. This method does not perform
        encoding; it assumes the provided form (str or bytes) is what needs checking.

        Args:
            test_kmer: The k-mer (string or bytes) to validate.

        Returns:
            True if the length of `test_kmer` matches `self.kmer_length`,
            False otherwise. Returns False for non-str/bytes input.
        """
        if not isinstance(test_kmer, (str, bytes)):
            return False
        return len(test_kmer) == self.kmer_length

    def __len__(self) -> int:
        """Returns the number of unique k-mers in the database."""
        return self.num_kmers

    def __contains__(self, kmer: bytes) -> bool:
        """Checks if a k-mer (bytes) is present in the database."""
        if not isinstance(kmer, bytes):
            return False 
        return kmer in self.kmer_to_counts_map

if __name__ == "__main__":
    dummy_kmers_str = [("A" * 4), ("C" * 4), ("G" * 4), ("T" * 4)]
    dummy_strains = ["ExampleStrain1", "ExampleStrain2"]
    dummy_data_np = np.array([[10, 5], [3, 12], [8, 8], [0, 15]], dtype=np.uint8)
    dummy_df_str_idx = pd.DataFrame(
        dummy_data_np, index=dummy_kmers_str, columns=dummy_strains
    )
    try:
        script_dir = pathlib.Path(__file__).parent
    except NameError: 
        script_dir = pathlib.Path.cwd()
    dummy_db_output_dir = script_dir / "test_db_output_consolidated"
    dummy_db_output_dir.mkdir(exist_ok=True)

    dummy_db_path_str_parquet = dummy_db_output_dir / "dummy_strain_kmer_db_str_idx.parquet" 
    dummy_df_str_idx.to_parquet(dummy_db_path_str_parquet, index=True) 
    print(f"Created dummy Parquet database (string k-mers) at {dummy_db_path_str_parquet.resolve()}")

    try:
        print("
--- Testing consolidated StrainKmerDb (inferred length) from Parquet ---")
        db_inferred = StrainKmerDb(dummy_db_path_str_parquet) 
        kmer_to_find_bytes = b"AAAA"
        counts = db_inferred.get_strain_counts_for_kmer(kmer_to_find_bytes)
        print(f"Counts for {kmer_to_find_bytes.decode('utf-8', 'replace')}: {counts}")

        known_kmer_bytes = b"CCCC"
        if known_kmer_bytes in db_inferred:
            print(f"K-mer {known_kmer_bytes.decode()} is in the database.")

        print(f"Total k-mers in database: {len(db_inferred)}")
        print(f"Database stats: {db_inferred.get_database_stats()}")
        print(f"Is 'AAAA' valid length? {db_inferred.validate_kmer_length(b'AAAA')}")
        print(f"Is 'AAA' valid length? {db_inferred.validate_kmer_length(b'AAA')}")

        print("
--- Testing with expected_kmer_length provided (from Parquet) ---")
        db_expected_len = StrainKmerDb(dummy_db_path_str_parquet, expected_kmer_length=4) 
        counts_2 = db_expected_len.get_strain_counts_for_kmer(b"GGGG")
        print(f"Counts for b'GGGG': {counts_2}")

    except Exception as e:
        print(f"An error occurred during Parquet database testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dummy_db_path_str_parquet.exists():
            dummy_db_path_str_parquet.unlink()
        if dummy_db_output_dir.exists() and not any(dummy_db_output_dir.iterdir()):
            dummy_db_output_dir.rmdir()
        print("
Cleaned up dummy Parquet database files and directory (if empty).")

```

**5. File Path: `src/strainr/parameter_config.py`**
New Content:
```python
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
        help="Path to the KmerStrainDatabase file (Parquet format).", # Corrected
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
        help="Selection mode for disambiguation", # Corrected
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
        help="Perform binning.", # Corrected
    )
    parser.add_argument(
        "--save-raw-hits",
        action="store_true",
        required=False,
        help="Save intermediate k-mer hit scores and final assignments to output files (format may vary).", # Corrected
    )
    config_space = parser.parse_args()
    return config_space
```

--- END OF FILE CONTENT BLOCKS ---

Please process these instructions.
The content for `src/strainr/binning.py` is not included as the previous review indicated its specific correction was already applied.
If any of these files were actually already corrected as intended, the overwrite will simply re-apply the correct state.
This subtask focuses on ensuring these 5 files are in the state defined above.
