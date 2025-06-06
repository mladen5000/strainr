#!/usr/bin/env python
"""
StrainR Command-Line Interface: K-mer based strain classification tool.

This script processes single-end or paired-end sequence reads (FASTA/FASTQ),
classifies them against a k-mer database, resolves ambiguous assignments,
calculates strain abundances, and outputs the results.
"""

import gzip
import logging
import pathlib
import pickle
from collections import Counter
from typing import Callable, Dict, Generator, List, Optional, TextIO, Tuple, Union

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
from .genomic_types import (
    CountVector,
    ReadId,
    StrainIndex,
)  # Changed to relative import
from .output import AbundanceCalculator
from .utils import _get_sample_name


# Type aliases for better readability
ReadHitResults = List[Tuple[ReadId, CountVector]]

# Global constants
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_NUM_PROCESSES = 4
DEFAULT_ABUNDANCE_THRESHOLD = 0.001


class KmerExtractor:
    """Handles k-mer extraction with fallback between Rust and Python implementations."""

    PY_RC_TRANSLATE_TABLE = bytes.maketrans(b"ACGTN", b"TGCAN")

    def __init__(self):
        self._extract_func: Callable[[bytes, int], List[bytes]]
        self._rust_available: bool = False
        self._initialize_extractor()

    def _initialize_extractor(self) -> None:
        """Initialize the k-mer extraction function with Rust fallback to Python."""
        try:  # Rust implementation
            from kmer_counter_rs import extract_kmers_rs

            self._extract_func = extract_kmers_rs
            self._rust_available = True
            logging.getLogger(__name__).info(
                "Successfully imported Rust k-mer counter. Using Rust implementation."
            )
        except ImportError:  # Fallback to Python implementation
            self._extract_func = self._py_extract_canonical_kmers
            self._rust_available = False
            logging.getLogger(__name__).warning(
                "Rust k-mer counter not found. Using Python fallback."
            )
        except Exception as e:  # Fallback to Python implementation
            self._extract_func = self._py_extract_canonical_kmers
            self._rust_available = False
            logging.getLogger(__name__).error(
                f"Error importing Rust k-mer counter: {e}. Using Python fallback."
            )

    # @staticmethod
    # def _py_reverse_complement(dna_sequence: bytes) -> bytes:
    #     """Computes the reverse complement of a DNA sequence."""
    #     complement_map = {
    #         ord("A"): ord("T"),
    #         ord("T"): ord("A"),
    #         ord("C"): ord("G"),
    #         ord("G"): ord("C"),
    #         ord("N"): ord("N"),
    #     }
    #     return bytes(complement_map.get(base, base) for base in reversed(dna_sequence))

    _RC_TABLE = bytes.maketrans(b"ACGTacgt", b"TGCAtgca")

    @staticmethod
    def _py_reverse_complement(seq: bytes) -> bytes:
        """Efficient reverse complement for DNA k-mers as bytes."""
        return seq.translate(KmerExtractor._RC_TABLE)[::-1]

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

    input_forward: List[pathlib.Path] = Field(
        description="Input forward read file(s) (FASTA/FASTQ, possibly gzipped)."
    )
    input_reverse: Optional[List[pathlib.Path]] = Field(
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
        """Ensures input file paths exist.
        Pydantic calls this per field. The return type should be what Pydantic expects for that field
        after "before" validation, or Pydantic will attempt coercion.
        """
        if (
            v is None
        ):  # This case is for Optional fields like input_reverse if it's not provided.
            return None

        if isinstance(v, list):
            # This branch is for input_forward (List[str] from argparse)
            # and input_reverse (List[str] from argparse if provided)
            return [cls._validate_single_path(path_str) for path_str in v]

        # This branch is for db_path (str from argparse)
        # It's also a fallback, but argparse setup should prevent other types for these fields.
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
        if self.input_reverse:  # This is now Optional[List[pathlib.Path]]
            # self.input_forward is List[pathlib.Path]
            if len(self.input_forward) != len(self.input_reverse):
                raise ValueError("Number of forward and reverse read files must match.")
        return self

    # __init__ is not needed for Pydantic models unless custom logic beyond validation is required at instantiation.
    # Pydantic handles initialization from kwargs automatically.
    # def __init__(self, **data):
    # super().__init__() # This is incorrect for Pydantic V2. Use super().__init__(**data) if overriding.
    # For Pydantic, usually no custom __init__ is needed if only using field definitions and validators.
    # If it was `super().__init__(**data)` it would be fine, but it's not needed here.


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

    def run_workflow(self) -> None:
        # TECH DEBT SUGGESTION:
        # This method orchestrates the main StrainR classification workflow and is quite long.
        # It handles:
        #   - Database initialization.
        #   - Analyzer and AbundanceCalculator initialization.
        #   - Iterating through input files (samples).
        #   - For each sample:
        #       - Read classification (parallel processing).
        #       - Saving raw k-mer hits (optional).
        #       - Hit categorization and disambiguation via ClassificationAnalyzer.
        #       - Abundance calculation and reporting (reconstructing DataFrame logic).
        #
        # While an orchestrator method is expected to call many other components, if more
        # steps or complexity are added to the per-sample processing, consider breaking
        # down the loop body into smaller private methods. For example, a
        # `_process_single_sample(self, fwd_reads_path, rev_reads_path, sample_name, analyzer, abundance_calc)`
        # method could encapsulate the logic for one sample, improving readability of `run_workflow`.
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
                sample_name = _get_sample_name(fwd_reads_path)
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

                # Save final_assignments and strain_names for potential downstream use (e.g., binning)
                self.logger.info(
                    "Saving final_assignments and strain_names for downstream use."
                )

                assignments_output_path = (
                    self.args.output_dir / f"{sample_name}_final_assignments.pkl"
                )
                try:
                    with open(assignments_output_path, "wb") as f_assign:
                        pickle.dump(final_assignments, f_assign)
                    self.logger.info(
                        f"Final assignments pickled to: {assignments_output_path}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to pickle final_assignments: {e}")

                strain_names_output_path = (
                    self.args.output_dir / f"{sample_name}_strain_names.txt"
                )
                try:
                    with open(strain_names_output_path, "w") as f_strains:
                        for (
                            strain_name_item
                        ) in self.database.strain_names:  # Corrected variable name
                            f_strains.write(f"{strain_name_item}\n")
                    self.logger.info(
                        f"Strain names saved to: {strain_names_output_path}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to save strain_names: {e}")

                # 3. Calculate and output abundances
                # New sequence using output.AbundanceCalculator:
                final_named_assignments = (
                    abundance_calc.convert_assignments_to_strain_names(
                        final_assignments,
                        unassigned_marker="NA",  # Assuming "NA" is the marker
                    )
                )

                # To replicate the old abundance_df structure:
                raw_counts = abundance_calc.calculate_raw_abundances(
                    final_named_assignments,
                    exclude_unassigned=False,
                    unassigned_marker="NA",  # Get all counts including NA
                )

                # Prepare data for DataFrame similar to the original one in classify.py's AbundanceCalculator
                table_data = []
                total_reads_for_relab = sum(raw_counts.values())

                # Calculate sum of hits for strains passing threshold for intra_relab calculation
                sum_hits_passing_threshold = 0
                if total_reads_for_relab > 0:
                    for (
                        strain_name_iter
                    ) in self.database.strain_names:  # Iterate through known strains
                        count = raw_counts.get(strain_name_iter, 0)
                        sample_relab_temp = count / total_reads_for_relab
                        if sample_relab_temp >= self.args.abundance_threshold:
                            sum_hits_passing_threshold += count

                # Ensure all strains and NA are in the df. Use a combined list of known strain names and any potentially new names from raw_counts (e.g. "NA" or others if they appear)
                # However, the original logic iterated self.database.strain_names + ["NA"], so we stick to that to replicate behavior.
                all_strain_keys_for_df = self.database.strain_names + ["NA"]
                # Deduplicate if "NA" is somehow in self.database.strain_names (though unlikely)
                processed_keys = set()
                unique_keys_for_df = []
                for key in all_strain_keys_for_df:
                    if key not in processed_keys:
                        unique_keys_for_df.append(key)
                        processed_keys.add(key)
                # Also add any keys from raw_counts that might not be in the initial list (e.g. if "Unassigned_BadIndex" was produced)
                # For strict replication of original, we only consider self.database.strain_names + ["NA"]
                # The original code was: for strain_name_for_df in self.strain_names + ["NA"]:

                for strain_name_for_df in unique_keys_for_df:
                    raw_hits = raw_counts.get(strain_name_for_df, 0)
                    sample_relab = (
                        raw_hits / total_reads_for_relab
                        if total_reads_for_relab > 0
                        else 0.0
                    )

                    intra_relab = 0.0
                    # Original logic for intra_relab:
                    # if strain_name_for_df != "NA" and sample_relab >= self.abundance_threshold:
                    #    if sum_hits_passing_threshold > 0: # sum_hits_passing_threshold calculated above
                    #        intra_relab = raw_hits / sum_hits_passing_threshold
                    # The self.args.abundance_threshold is from the workflow args
                    if (
                        strain_name_for_df != "NA"
                        and sample_relab >= self.args.abundance_threshold
                    ):
                        if sum_hits_passing_threshold > 0:
                            intra_relab = raw_hits / sum_hits_passing_threshold

                    table_data.append({
                        "strain_name": strain_name_for_df,
                        "sample_hits": raw_hits,
                        "sample_relab": sample_relab,
                        "intra_relab": intra_relab,  # ensure NA has 0.0 as per original logic due to strain_name_for_df != "NA"
                    })

                abundance_df_reconstructed = pd.DataFrame(table_data).set_index(
                    "strain_name"
                )
                abundance_df_reconstructed = abundance_df_reconstructed.sort_values(
                    by="sample_hits", ascending=False
                )

                # Now save and display this reconstructed DataFrame
                output_path = (
                    self.args.output_dir / f"{sample_name}_abundance_report.tsv"
                )
                abundance_df_reconstructed.to_csv(
                    output_path, sep="	", float_format="%.6f"
                )
                self.logger.info(f"Abundance report saved to: {output_path}")

                # Display to console (replicate KmerClassificationWorkflow.AbundanceCalculator.display_console_output)
                self.logger.info(
                    f"Displaying top 10 abundances for sample: {sample_name}"
                )
                print(f" --- Abundance Report for: {sample_name} ---")
                # Make sure to use the reconstructed DataFrame
                df_to_display_reconstructed = abundance_df_reconstructed[
                    abundance_df_reconstructed["sample_hits"] > 0
                ].copy()
                unassigned_info_reconstructed = ""
                if "NA" in df_to_display_reconstructed.index:
                    na_row_reconstructed = df_to_display_reconstructed.loc["NA"]
                    unassigned_info_reconstructed = (
                        f"Unassigned Reads: {int(na_row_reconstructed['sample_hits'])} "
                        f"({na_row_reconstructed['sample_relab']:.4f})"
                    )
                    df_to_display_reconstructed = df_to_display_reconstructed.drop(
                        index="NA"
                    )

                # Default top_n=10 as in the original display_console_output method
                print(
                    df_to_display_reconstructed.head(10).to_string(float_format="%.4f")
                )
                if unassigned_info_reconstructed:
                    print(unassigned_info_reconstructed)
                print("--- End of Report ---")

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
        # Map CLI args to CliArgs model
        # argparse with nargs="+" ensures input_forward is always a list.
        # argparse with nargs="*" ensures input_reverse is a list or None (if not provided).
        # No need to convert single-item lists to single items.
        cli_args = CliArgs(
            input_forward=args.input_forward,  # Directly pass the list from argparse
            input_reverse=args.input_reverse
            if args.input_reverse
            else None,  # Pass list or None
            db_path=args.db,  # This is a single path string from argparse
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
