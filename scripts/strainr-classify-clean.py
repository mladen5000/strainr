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
from math import log2
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    TextIO,
)  # Added TextIO

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
)  # Assuming these are defined
from typing import Callable, Set  # Added Callable, Set


# Python fallback k-mer extraction functions
def _py_reverse_complement(dna_sequence: bytes) -> bytes:
    """Computes the reverse complement of a DNA sequence."""
    complement_map = {
        ord("A"): ord("T"),
        ord("T"): ord("A"),
        ord("C"): ord("G"),
        ord("G"): ord("C"),
        ord("N"): ord("N"),
    }
    # Apply complement and reverse, handling non-DNA bytes by leaving them as is
    return bytes(complement_map.get(base, base) for base in reversed(dna_sequence))


def _py_extract_canonical_kmers(sequence: bytes, k: int) -> List[bytes]:
    """
    Extracts canonical k-mers from a DNA sequence using Python.
    A k-mer is canonical if it's lexicographically smaller than its reverse complement.
    Assumes sequence is already normalized (e.g., uppercase).
    """
    if k == 0 or not sequence or len(sequence) < k:
        return []

    kmers: List[bytes] = []

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        # Assuming kmer is already normalized (e.g. uppercase DNA)
        # For needletail compatibility, it expects normalized input for canonical kmer generation
        # The Rust version uses needletail::sequence::normalize first.
        # Here, we assume the input `sequence` to this Python function should also be normalized if needed.
        rc_kmer_bytes = _py_reverse_complement(kmer)

        if kmer <= rc_kmer_bytes:
            kmers.append(kmer)
        else:
            kmers.append(rc_kmer_bytes)
    return kmers


# Global k-mer extraction function dispatcher
_extract_kmers_func: Callable[[bytes, int], List[bytes]]
RUST_KMER_COUNTER_AVAILABLE: bool

try:
    from kmer_counter_rs import extract_kmers_rs

    _extract_kmers_func = extract_kmers_rs
    RUST_KMER_COUNTER_AVAILABLE = True
    # Use logger after it's initialized
    # logger.info("Successfully imported Rust k-mer counter. Using Rust implementation.")
    print(
        "Successfully imported Rust k-mer counter (extract_kmers_rs). Using Rust implementation."
    )
except ImportError:
    RUST_KMER_COUNTER_AVAILABLE = False
    _extract_kmers_func = _py_extract_canonical_kmers
    # logger.warning("Rust k-mer counter not found. Using Python fallback.")
    print(
        "Warning: Rust k-mer counter (kmer_counter_rs.extract_kmers_rs) not found. Using Python fallback for k-mer extraction."
    )
except Exception as e:
    RUST_KMER_COUNTER_AVAILABLE = False
    _extract_kmers_func = _py_extract_canonical_kmers
    # logger.error(f"Error importing Rust k-mer counter: {e}. Using Python fallback.")
    print(f"Error importing Rust k-mer counter: {e}. Using Python fallback.")


SETCHUNKSIZE = 10000

# Set up logging
# Configure logging to write to a file and stream to stdout
logging.basicConfig(
    level=logging.INFO,  # Changed default level to INFO for less verbosity, DEBUG can be set via arg if needed
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("strainr_classify.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class CliArgs(BaseModel):
    """
    Pydantic model for command-line argument validation and management.

    Attributes:
        input_forward: Path(s) to forward read files.
        input_reverse: Optional path(s) to reverse read files for paired-end data.
        db_path: Path to the k-mer database file.
        num_processes: Number of processor cores to use.
        output_dir: Output directory for results.
        disambiguation_mode: Strategy for resolving ambiguous k-mer hits.
        abundance_threshold: Minimum relative abundance for filtering.
        perform_binning: Flag to perform read binning (functionality not fully implemented here).
        save_raw_kmer_hits: Flag to save raw k-mer hit counts per read.
    """

    input_forward: Union[pathlib.Path, List[pathlib.Path]] = Field(
        alias="input_forward",
        description="Input forward read file(s) (FASTA/FASTQ, possibly gzipped).",
    )
    input_reverse: Optional[Union[pathlib.Path, List[pathlib.Path]]] = Field(
        default=None,
        alias="input_reverse",
        description="Optional input reverse read file(s) for paired-end data.",
    )
    db_path: pathlib.Path = Field(
        alias="db", description="Path to the k-mer database file (pickle format)."
    )
    num_processes: int = Field(
        default=4, ge=1, alias="procs", description="Number of processor cores to use."
    )
    output_dir: pathlib.Path = Field(
        default=pathlib.Path("strainr_out"),
        alias="out",
        description="Output directory.",
    )
    disambiguation_mode: str = Field(
        default="max",
        pattern="^(random|max|multinomial|dirichlet)$",  # Ensures mode is one of the supported
        alias="mode",
        description="Selection mode for disambiguation of ambiguous reads.",
    )
    abundance_threshold: float = Field(
        default=0.001,
        ge=0.0,
        lt=1.0,
        alias="thresh",
        description="Threshold for relative abundance filtering.",
    )
    perform_binning: bool = Field(
        default=False, alias="bin", description="Perform read binning (if implemented)."
    )  # Added specific alias
    save_raw_kmer_hits: bool = Field(
        default=False,
        alias="save_raw_hits",
        description="Save raw k-mer hit counts per read to a CSV file.",
    )

    @field_validator("input_forward", "input_reverse", "db_path", mode="before")
    @classmethod
    def _validate_input_paths_exist(
        cls, v: Union[str, List[str], None]
    ) -> Optional[Union[pathlib.Path, List[pathlib.Path]]]:
        """Ensures input file paths exist."""
        if v is None:
            return None

        logger.debug(f"Validating file path(s): {v}")
        if isinstance(v, list):
            return [cls._validate_single_path_exists(path_str) for path_str in v]
        return cls._validate_single_path_exists(str(v))

    @classmethod
    def _validate_single_path_exists(cls, path_str: str) -> pathlib.Path:
        """Validates a single file path string."""
        path_obj = pathlib.Path(path_str)
        if not path_obj.exists():
            logger.error(f"File does not exist: {path_str}")
            raise ValueError(f"File does not exist: {path_str}")
        if not path_obj.is_file():
            logger.error(f"Path is not a file: {path_str}")
            raise ValueError(f"Path is not a file: {path_str}")
        return path_obj

    @model_validator(mode="after")
    def _check_paired_read_consistency(self) -> "CliArgs":
        """Validates consistency between forward and reverse read file lists."""
        logger.debug("Validating paired-end read consistency.")
        if self.input_reverse:  # Only if reverse reads are provided
            if isinstance(self.input_forward, list) != isinstance(
                self.input_reverse, list
            ):
                raise ValueError(
                    "If one of input_forward or input_reverse is a list, the other must also be a list."
                )
            if isinstance(self.input_forward, list) and isinstance(
                self.input_reverse, list
            ):  # Both are lists
                if len(self.input_forward) != len(self.input_reverse):
                    raise ValueError(
                        "The number of forward read files must match the number of reverse read files."
                    )
        return self


class KmerClassificationWorkflow:
    """
    Orchestrates the k-mer based strain classification workflow.

    This class encapsulates all steps of the process, from loading data and
    k-mer databases to performing classification, analyzing results, and
    generating output.
    """

    def __init__(self, args: CliArgs) -> None:
        """
        Initializes the KmerClassificationWorkflow.

        Args:
            args: A `CliArgs` object containing validated command-line arguments.
        """
        logger.info("Initializing KmerClassificationWorkflow with provided arguments.")
        self.args: CliArgs = args
        self.database: Optional[KmerStrainDatabase] = (
            None  # Initialized in build_database
        )

    def _open_file_handle(self, file_path: pathlib.Path) -> TextIO:
        """
        Opens a file, handling .gz compression transparently.

        Args:
            file_path: The path to the file to open.

        Returns:
            A file object opened in text mode (`TextIO`).
        """
        logger.debug(f"Opening file: {file_path}")
        if file_path.name.endswith(".gz"):
            return gzip.open(file_path, "rt")
        else:
            return file_path.open("rt")

    def _detect_sequence_file_format(self, file_path: pathlib.Path) -> str:
        """
        Detects sequence file format (FASTA or FASTQ) based on the first line.

        Args:
            file_path: The path to the sequence file.

        Returns:
            A string indicating the format ("fasta" or "fastq").

        Raises:
            ValueError: If the file format cannot be determined from the first line.
        """
        logger.debug(f"Detecting file format for: {file_path}")
        with self._open_file_handle(file_path) as f:
            first_line = f.readline().strip()
            if first_line.startswith(">"):
                logger.info(f"Detected FASTA format for {file_path}.")
                return "fasta"
            elif first_line.startswith("@"):
                logger.info(f"Detected FASTQ format for {file_path}.")
                return "fastq"
            else:
                logger.error(
                    f"Unknown file format for file: {file_path}. First line: '{first_line}'"
                )
                raise ValueError(f"Unknown file format for file: {file_path}")

    def _parse_sequence_files(
        self,
        fwd_reads_path: pathlib.Path,
        rev_reads_path: Optional[pathlib.Path] = None,
    ) -> Generator[Tuple[ReadId, bytes, bytes], None, None]:
        """
        Parses sequence files (FASTA/FASTQ, single or paired-end) and yields reads.

        Args:
            fwd_reads_path: Path to the forward (or single-end) read file.
            rev_reads_path: Optional path to the reverse read file for paired-end data.

        Yields:
            A tuple for each read (or pair): (ReadId, forward_sequence_bytes, reverse_sequence_bytes).
            For single-end reads, reverse_sequence_bytes will be an empty byte string (b"").
        """
        logger.info(
            f"Parsing sequence files. Forward: {fwd_reads_path}, Reverse: {rev_reads_path if rev_reads_path else 'N/A'}"
        )
        fwd_format = self._detect_sequence_file_format(fwd_reads_path)
        rev_format: Optional[str] = None
        if rev_reads_path:
            rev_format = self._detect_sequence_file_format(rev_reads_path)
            if fwd_format != rev_format:
                raise ValueError(
                    f"Mismatched file formats: {fwd_reads_path} is {fwd_format}, "
                    f"but {rev_reads_path} is {rev_format}."
                )

        with self._open_file_handle(fwd_reads_path) as fwd_fh:
            rev_fh: Optional[TextIO] = None
            try:
                if rev_reads_path:
                    rev_fh = self._open_file_handle(rev_reads_path)

                fwd_iter: Any  # Iterator for forward reads
                rev_iter: Optional[Any] = (
                    None  # Iterator for reverse reads, if applicable
                )

                if fwd_format == "fasta":
                    fwd_iter = SeqIO.parse(fwd_fh, "fasta")
                    if rev_fh:
                        rev_iter = SeqIO.parse(rev_fh, "fasta")
                elif fwd_format == "fastq":
                    fwd_iter = FastqGeneralIterator(fwd_fh)
                    if rev_fh:
                        rev_iter = FastqGeneralIterator(rev_fh)
                else:  # Should be caught by _detect_sequence_file_format
                    raise ValueError(f"Unsupported file format: {fwd_format}")

                for fwd_record in fwd_iter:
                    fwd_id: ReadId
                    fwd_seq_str: str

                    if fwd_format == "fasta":
                        fwd_id = fwd_record.id
                        fwd_seq_str = str(fwd_record.seq)
                    else:  # fastq
                        fwd_id, fwd_seq_str, _ = fwd_record  # type: ignore

                    fwd_seq_bytes = fwd_seq_str.encode("utf-8")
                    rev_seq_bytes = b""

                    if rev_iter:
                        try:
                            rev_record = next(rev_iter)
                            rev_seq_str: str
                            if rev_format == "fasta":
                                # Basic ID check for FASTA, could be more robust
                                # if fwd_id.split()[0] != rev_record.id.split()[0]:
                                #     logger.warning(f"Paired read IDs may not match: '{fwd_id}' and '{rev_record.id}'")
                                rev_seq_str = str(rev_record.seq)
                            else:  # fastq
                                _, rev_seq_str, _ = rev_record  # type: ignore
                            rev_seq_bytes = rev_seq_str.encode("utf-8")
                        except StopIteration:
                            logger.warning(
                                f"Reverse read file ended before forward file for ID {fwd_id}. Treating remaining as single-end."
                            )
                            rev_iter = None  # Stop trying to read from it

                    yield fwd_id, fwd_seq_bytes, rev_seq_bytes
            finally:
                if rev_fh:
                    rev_fh.close()

    def _initialize_database(self) -> None:
        """Loads and initializes the KmerStrainDatabase."""
        logger.info(f"Loading k-mer database from: {self.args.db_path}")
        self.database = KmerStrainDatabase(self.args.db_path)
        # kmer_length is automatically determined by KmerStrainDatabase
        logger.info(
            f"Database loaded: {self.database.num_kmers} k-mers, "
            f"{self.database.num_strains} strains, k-mer length {self.database.kmer_length}."
        )

    def _count_kmers_for_read_record(
        self, record_tuple: Tuple[ReadId, bytes, bytes]
    ) -> Tuple[ReadId, CountVector]:
        """
        Processes a single read (or read pair) to count k-mer occurrences against the database.

        Args:
            record_tuple: A tuple containing (ReadId, forward_sequence_bytes, reverse_sequence_bytes).

        Returns:
            A tuple containing (ReadId, CountVector) for the processed read.
        """
        if self.database is None:
            logger.error("Database not initialized before k-mer counting.")
            raise RuntimeError("Database not initialized.")

        read_id, fwd_seq_bytes, rev_seq_bytes = (
            record_tuple  # rev_seq_bytes is b"" for single-end
        )
        logger.debug(f"Processing k-mers for read: {read_id}")

        all_kmers_from_pair: Set[bytes] = set()

        if fwd_seq_bytes:
            # The Python fallback expects normalized sequences if the Rust version does.
            # Assuming sequences from _parse_sequence_files are raw, need normalization before Python kmer extraction.
            # The Rust function handles normalization internally.
            # For simplicity here, we'll assume _get_kmers_for_sequence handles normalization if Python is used.
            # This means _py_extract_canonical_kmers should ideally also normalize.
            # Let's add a basic normalization (uppercase) to _py_extract_canonical_kmers for now.
            # Or, ensure sequence_bytes passed to _extract_kmers_func are already normalized.
            # The Rust function `extract_kmers_rs` applies `Sequence::normalize(sequence, false);`
            # So, the Python fallback should do something similar if it wants to be a true equivalent.
            # For this step, _get_kmers_for_sequence will call the selected function.
            # If Python fallback is used, it includes _py_reverse_complement.
            # Normalization to uppercase should happen before _py_extract_canonical_kmers.

            # Normalization step (e.g., to uppercase ASCII if that's what needletail's normalize does)
            # This should ideally match what the Rust `Sequence::normalize` does.
            # For DNA, typically converting to uppercase is sufficient.
            normalized_fwd_seq = fwd_seq_bytes.upper()  # Basic normalization

            kmers_fwd = self._get_kmers_for_sequence(normalized_fwd_seq)
            all_kmers_from_pair.update(kmers_fwd)

        if rev_seq_bytes:
            normalized_rev_seq = rev_seq_bytes.upper()  # Basic normalization
            kmers_rev = self._get_kmers_for_sequence(normalized_rev_seq)
            all_kmers_from_pair.update(kmers_rev)

        current_read_strain_counts = np.zeros(self.database.num_strains, dtype=np.uint8)

        if not all_kmers_from_pair:
            return read_id, current_read_strain_counts

        for kmer_bytes in all_kmers_from_pair:
            strain_counts_for_kmer: Optional[CountVector] = (
                self.database.get_strain_counts_for_kmer(kmer_bytes)
            )
            if strain_counts_for_kmer is not None:
                current_read_strain_counts += strain_counts_for_kmer

        return read_id, current_read_strain_counts

    def _get_kmers_for_sequence(self, sequence_bytes: bytes) -> List[bytes]:
        """
        Extracts canonical k-mers from a single sequence using the configured k-mer counter.
        The input sequence_bytes to this function should be pre-normalized if the
        Python fallback is to be used effectively (e.g., converted to uppercase).
        The Rust version handles its own normalization.
        """
        if self.database is None:
            raise RuntimeError(
                "Database not initialized, cannot determine k-mer length."
            )

        if not sequence_bytes or len(sequence_bytes) < self.database.kmer_length:
            return []

        # Global _extract_kmers_func will point to either Rust or Python version
        try:
            # sequence_bytes passed to _extract_kmers_func should be what the function expects.
            # Rust's extract_kmers_rs does its own normalization.
            # _py_extract_canonical_kmers expects an already somewhat normalized sequence (e.g. uppercase).
            # The caller (_count_kmers_for_read_record) now does basic .upper() before calling this.
            return _extract_kmers_func(sequence_bytes, self.database.kmer_length)
        except Exception as e:
            logger.error(
                f"Error during k-mer extraction for sequence of length {len(sequence_bytes)} with k={self.database.kmer_length}: {e}"
            )
            # Depending on how critical, either re-raise or return empty / specific error object
            # For now, let's return empty to allow processing of other reads.
            return []

    def _classify_reads_parallel(
        self,
        fwd_reads_path: pathlib.Path,
        rev_reads_path: Optional[pathlib.Path] = None,
    ) -> ReadHitResults:
        """
        Classifies reads from input files in parallel using multiprocessing.

        Args:
            fwd_reads_path: Path to the forward (or single-end) read file.
            rev_reads_path: Optional path to the reverse read file.

        Returns:
            A list of tuples, where each tuple is (ReadId, CountVector),
            representing the k-mer hit profile for each read.
        """
        logger.info(f"Starting parallel classification for {fwd_reads_path}...")
        sequence_record_iterator = self._parse_sequence_files(
            fwd_reads_path, rev_reads_path
        )

        results: ReadHitResults
        with mp.Pool(processes=self.args.num_processes) as pool:
            results = pool.map(
                self._count_kmers_for_read_record,
                sequence_record_iterator,
                chunksize=SETCHUNKSIZE,  # Consider making SETCHUNKSIZE an Args parameter
            )
        logger.info(f"Finished parallel classification for {fwd_reads_path}.")
        return results

    def _save_raw_kmer_hit_counts(
        self, raw_hit_results: ReadHitResults, sample_name: str
    ) -> None:
        """Saves the raw k-mer hit counts per read to a CSV file."""
        logger.info(f"Saving raw k-mer hit counts for sample: {sample_name}")
        # Convert list of tuples to DataFrame
        # Each CountVector needs to be expanded or handled appropriately
        # For simplicity, this example might save a less detailed version or require adjustment
        # based on how CountVector should be represented in CSV.

        # A more practical save would be: ReadId, Strain1_hits, Strain2_hits, ...
        if not raw_hit_results:
            logger.warning(f"No raw hit results to save for sample: {sample_name}")
            return

        if self.database is None:
            logger.error(
                "Cannot save raw hits, database not initialized (needed for strain names)."
            )
            return

        df_data = []
        for read_id, count_vector in raw_hit_results:
            row = {"read_id": read_id}
            for i, strain_name in enumerate(self.database.strain_names):
                row[strain_name] = count_vector[i]
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Ensure output directory exists
        raw_hits_output_dir = self.args.output_dir / "raw_kmer_hits"
        raw_hits_output_dir.mkdir(parents=True, exist_ok=True)

        output_csv_path = raw_hits_output_dir / f"{sample_name}_raw_hits.csv"
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Raw k-mer hit counts saved to: {output_csv_path}")

    # --- Abundance calculation and output methods (kept within this class for now) ---
    # These methods would ideally be part of AbundanceCalculator from strainr.output,
    # but are included here due to previous issues with overwriting that file.
    # They are simplified versions for direct use.

    def _translate_assignments_to_names_for_output(
        self, final_assignments: Dict[ReadId, Union[StrainIndex, str]]
    ) -> Dict[ReadId, str]:
        """Translates final assignments (indices or 'NA') to strain names for output."""
        if self.database is None:
            raise RuntimeError("Database not initialized.")

        translated: Dict[ReadId, str] = {}
        for read_id, assignment in final_assignments.items():
            if isinstance(assignment, int):  # StrainIndex
                if 0 <= assignment < len(self.database.strain_names):
                    translated[read_id] = self.database.strain_names[assignment]
                else:
                    translated[read_id] = (
                        "Unassigned_BadIndex"  # Should not happen with proper assignment
                    )
            else:  # It's a string, likely "NA"
                translated[read_id] = assignment
        return translated

    def _calculate_output_abundances(
        self, final_named_assignments: Dict[ReadId, str], sample_name: str
    ) -> pd.DataFrame:
        """Calculates and formats abundance data into a DataFrame."""
        if self.database is None:
            raise RuntimeError("Database not initialized.")

        logger.info(f"Calculating abundances for sample: {sample_name}")
        hit_counts: Counter[str] = Counter(final_named_assignments.values())

        # Ensure all known strains and "NA" (if present) are in the Counter
        for strain_name in self.database.strain_names:
            hit_counts.setdefault(strain_name, 0)
        hit_counts.setdefault(
            "NA", 0
        )  # Assuming "NA" is the marker from ClassificationAnalyzer

        total_assigned_hits = sum(
            count for strain, count in hit_counts.items() if strain != "NA"
        )

        table_data = []
        for strain_name in self.database.strain_names + ["NA"]:
            raw_hits = hit_counts[strain_name]

            # Sample relab: relative to all assigned reads + NA reads
            sample_relab = (
                raw_hits / sum(hit_counts.values())
                if sum(hit_counts.values()) > 0
                else 0.0
            )

            # Intra relab: relative to assigned reads only, for strains passing threshold
            intra_relab = 0.0
            if strain_name != "NA" and sample_relab >= self.args.abundance_threshold:
                if total_assigned_hits > 0:  # Denominator for intra-sample relab
                    # Calculate sum of hits for strains passing threshold
                    sum_hits_passing_thresh = sum(
                        h_count
                        for s_name, h_count in hit_counts.items()
                        if s_name != "NA"
                        and (
                            h_count / sum(hit_counts.values())
                            if sum(hit_counts.values()) > 0
                            else 0
                        )
                        >= self.args.abundance_threshold
                    )
                    if sum_hits_passing_thresh > 0:
                        intra_relab = raw_hits / sum_hits_passing_thresh

            table_data.append(
                {
                    "strain_name": strain_name,
                    "sample_hits": raw_hits,
                    "sample_relab": sample_relab,
                    "intra_relab": (
                        intra_relab if strain_name != "NA" else 0.0
                    ),  # NA doesn't get intra_relab
                }
            )

        abundance_df = pd.DataFrame(table_data).set_index("strain_name")
        abundance_df = abundance_df.sort_values(by="sample_hits", ascending=False)

        # Save to TSV
        output_tsv_path = self.args.output_dir / f"{sample_name}_abundance_report.tsv"
        abundance_df.to_csv(output_tsv_path, sep="\t", float_format="%.6f")
        logger.info(f"Abundance report for {sample_name} saved to: {output_tsv_path}")

        return abundance_df

    def _display_console_output(
        self, abundance_df: pd.DataFrame, sample_name: str, top_n: int = 10
    ) -> None:
        """Displays formatted abundance results to the console."""
        logger.info(f"Displaying top {top_n} abundances for sample: {sample_name}")
        print(f"\n--- Abundance Report for: {sample_name} ---")

        # Display relevant columns, up to top_n strains + NA
        df_to_display = abundance_df[abundance_df["sample_hits"] > 0]

        unassigned_info = ""
        if "NA" in df_to_display.index:
            na_row = df_to_display.loc[["NA"]]
            unassigned_info = f"Unassigned Reads: {int(na_row['sample_hits'].iloc[0])} ({na_row['sample_relab'].iloc[0]:.4f})"
            df_to_display = df_to_display.drop(index="NA")

        print(df_to_display.head(top_n).to_string(float_format="%.4f"))
        if unassigned_info:
            print(unassigned_info)
        print("--- End of Report ---")

    def run_workflow(self) -> None:
        """
        Executes the main classification and analysis workflow.
        """
        logger.info("Starting StrainR workflow.")
        self._initialize_database()  # Loads self.database
        if (
            self.database is None
        ):  # Should be caught by KmerStrainDatabase init if path is bad
            logger.critical("Database initialization failed. Exiting.")
            return

        analyzer = ClassificationAnalyzer(
            strain_names=self.database.strain_names,  # Use names from loaded DB
            disambiguation_mode=self.args.disambiguation_mode,
            abundance_threshold=self.args.abundance_threshold,  # Pass along if needed by analyzer
            num_processes=self.args.num_processes,
        )

        # Handle single or multiple input files
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

        for i, fwd_reads_path in enumerate(fwd_files):
            sample_name = fwd_reads_path.name.split("_R1")[0].split(".")[
                0
            ]  # Basic sample naming
            logger.info(
                f"Processing sample: {sample_name} (File: {fwd_reads_path.name})"
            )

            rev_reads_path: Optional[pathlib.Path] = None
            if rev_files and i < len(rev_files):
                rev_reads_path = rev_files[i]

            # 1. Classify reads (k-mer counting)
            raw_kmer_hits: ReadHitResults = self._classify_reads_parallel(
                fwd_reads_path, rev_reads_path
            )

            if self.args.save_raw_kmer_hits:
                self._save_raw_kmer_hit_counts(raw_kmer_hits, sample_name)

            # 2. Analyze classification results
            clear_hits_vec, ambiguous_hits_vec, no_hit_ids = (
                analyzer.separate_hit_categories(raw_kmer_hits)
            )

            clear_assignments_idx = analyzer.resolve_clear_hits_to_indices(
                clear_hits_vec
            )

            prior_counts = analyzer.calculate_strain_prior_from_assignments(
                clear_assignments_idx
            )
            prior_prob_vector = analyzer.convert_prior_counts_to_probability_vector(
                prior_counts
            )

            resolved_ambiguous_idx = analyzer.resolve_ambiguous_hits_parallel(
                ambiguous_hits_vec, prior_prob_vector
            )

            final_assignments_idx: Dict[ReadId, Union[StrainIndex, str]] = (
                analyzer.combine_assignments(
                    clear_assignments_idx,
                    resolved_ambiguous_idx,
                    no_hit_ids,
                    unassigned_marker="NA",
                )
            )

            # 3. Calculate and output abundances (using internal methods for now)
            final_named_assignments = self._translate_assignments_to_names_for_output(
                final_assignments_idx
            )
            abundance_df = self._calculate_output_abundances(
                final_named_assignments, sample_name
            )
            self._display_console_output(abundance_df, sample_name)

        logger.info("StrainR workflow completed.")


def _parse_cli_arguments() -> CliArgs:
    """
    Parses command-line arguments using argparse and validates with Pydantic.

    Returns:
        A `CliArgs` Pydantic model instance populated with validated arguments.
    """
    logger.info("Parsing command-line arguments.")
    parser = argparse.ArgumentParser(
        description="StrainR: K-mer based strain classification tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_forward",
        help="Input forward read file(s) (FASTA/FASTQ, possibly gzipped). Can be one or more files.",
        nargs="+",  # Allows multiple forward files
        required=True,
        type=str,
    )
    parser.add_argument(
        "--input_reverse",
        help="Optional input reverse read file(s) for paired-end data. Must match number of forward files if provided.",
        nargs="*",  # Allows zero or more reverse files
        type=str,
        default=None,  # Important for Pydantic optional field
    )
    parser.add_argument(
        "--db",
        help="Path to the k-mer database file (Pandas DataFrame pickle format).",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--procs",
        help="Number of processor cores to use.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--out",
        help="Output directory.",
        type=str,
        default="strainr_out",
    )
    parser.add_argument(
        "--mode",
        help="Selection mode for disambiguation of ambiguous reads.",
        choices=list(ClassificationAnalyzer.SUPPORTED_DISAMBIGUATION_MODES),
        default="max",
    )
    parser.add_argument(
        "--thresh",
        help="Minimum relative abundance threshold for filtering and reporting.",
        type=float,
        default=0.001,
    )
    # --bin flag is not used by KmerClassificationWorkflow, but kept for Args model consistency
    parser.add_argument(
        "--bin",
        help="Perform read binning (functionality to be implemented separately).",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_raw_hits",
        help="Save raw k-mer hit counts per read to a CSV file in the output directory.",
        action="store_true",
        default=False,
    )

    parsed_args = parser.parse_args()

    # Convert to Pydantic model for validation and type coercion
    try:
        cli_args_validated = CliArgs(
            input_forward=parsed_args.input_forward,
            input_reverse=parsed_args.input_reverse,  # Pydantic handles None if not provided
            db=parsed_args.db,  # Pydantic validator will handle Path conversion
            procs=parsed_args.procs,
            out=parsed_args.out,  # Pydantic validator will handle Path conversion
            mode=parsed_args.mode,
            thresh=parsed_args.thresh,
            bin=parsed_args.bin,
            save_raw_hits=parsed_args.save_raw_hits,
        )
    except ValueError as e:  # Pydantic validation error
        logger.error(f"Argument validation error: {e}")
        parser.print_help()
        sys.exit(1)

    return cli_args_validated


if __name__ == "__main__":
    logger.info("StrainR Classification Script Started.")

    # 1. Parse and validate arguments
    cli_args = _parse_cli_arguments()

    # 2. Initialize workflow orchestrator
    workflow_processor = KmerClassificationWorkflow(args=cli_args)

    # 3. Run the main workflow
    try:
        workflow_processor.run_workflow()
    except Exception as e:  # Catch any unhandled exceptions during workflow execution
        logger.critical(
            f"A critical error occurred during the StrainR workflow: {e}", exc_info=True
        )
        sys.exit(1)  # Exit with error status

    logger.info("StrainR Classification Script Finished Successfully.")
