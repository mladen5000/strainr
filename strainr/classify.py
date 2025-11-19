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
import pickle
import sys
from typing import Callable, Generator, Optional, TextIO, Union

import numpy as np
import pandas as pd

from strainr import ClassificationAnalyzer, StrainKmerDatabase

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
from .genomic_types import CountVector  # Changed to relative import
from .genomic_types import ReadId  # Changed to relative import
from .output import AbundanceCalculator
from .utils import _get_sample_name

# Type aliases for better readability
ReadHitResults = list[tuple[ReadId, CountVector]]

# Global constants
# DEFAULT_CHUNK_SIZE: Number of reads to process in each chunk for parallel processing.
# Larger values use more memory but reduce overhead; smaller values increase parallelism.
# 10,000 reads provides good balance for typical FASTQ files (~2-5MB per chunk).
DEFAULT_CHUNK_SIZE = 10000

# DEFAULT_NUM_PROCESSES: Default number of parallel worker processes.
# Set to 4 as a conservative default that works on most systems.
DEFAULT_NUM_PROCESSES = 4

# DEFAULT_ABUNDANCE_THRESHOLD: Minimum relative abundance to report a strain (0.1%).
# Strains below this threshold are filtered out to reduce noise in results.
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
        try:
            from kmer_counter_rs import extract_kmers_rs as rust_extract_kmers

            # Wrapper to match expected signature if rust function differs
            # The rust function is extract_kmers_rs(seq_bytes, k, skip_n)
            def rust_wrapper(sequence: bytes, k: int, skip_n: bool) -> List[bytes]:
                return rust_extract_kmers(sequence, k, skip_n)

            self._extract_func = rust_wrapper
            self._rust_available = True
            logging.getLogger(__name__).info(
                "Successfully imported Rust k-mer counter. Using Rust implementation."
            )
        except ImportError:
            self._extract_func = self._py_extract_canonical_kmers
            self._rust_available = False
            logging.getLogger(__name__).warning(
                "Rust k-mer counter not found. Using Python fallback (does not support skip_n_kmers parameter via kmer_counter_rs)."
            )
        except Exception as e:
            self._extract_func = self._py_extract_canonical_kmers
            self._rust_available = False
            logging.getLogger(__name__).error(
                f"Error importing Rust k-mer counter: {e}. Using Python fallback."
            )

    _RC_TABLE = bytes.maketrans(b"ACGTacgt", b"TGCAtgca")

    @staticmethod
    def _py_reverse_complement(seq: bytes) -> bytes:
        """Efficient reverse complement for DNA k-mers as bytes."""
        return seq.translate(KmerExtractor._RC_TABLE)[::-1]

    def _py_extract_canonical_kmers(
        self, sequence: bytes, k: int, skip_n_kmers: bool
    ) -> list[bytes]:
        """
        Extracts canonical k-mers from a DNA sequence using Python.
        A k-mer is canonical if it's lexicographically smaller than its reverse complement.
        Includes logic to skip k-mers containing 'N' if skip_n_kmers is True.
        """
        if k <= 0 or not sequence or len(sequence) < k:
            return []

        kmers: list[bytes] = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i : i + k]
            if skip_n_kmers and b"N" in kmer:  # Check for 'N' before reverse complement
                continue
            rc_kmer = self._py_reverse_complement(kmer)
            kmers.append(kmer if kmer <= rc_kmer else rc_kmer)
        return kmers

    def extract_kmers_with_quality(
        self, sequence: bytes, quality_scores: Optional[str], k: int,
        skip_n_kmers: bool, min_quality: int = 20
    ) -> list[bytes]:
        """
        Extract k-mers with quality filtering for FASTQ data.

        Only extracts k-mers where ALL bases meet the minimum quality threshold.
        This is scientifically important because low-quality bases can cause
        false k-mer matches and incorrect strain assignments.

        Args:
            sequence: DNA sequence as bytes
            quality_scores: Phred quality string (ASCII encoded), or None for FASTA
            k: K-mer length
            skip_n_kmers: Whether to skip k-mers containing 'N'
            min_quality: Minimum Phred quality score (default 20 = 99% accuracy)

        Returns:
            List of canonical k-mers passing quality filters
        """
        if not sequence or len(sequence) < k:
            return []

        # If no quality scores (FASTA), use standard extraction
        if quality_scores is None or len(quality_scores) == 0:
            return self.extract_kmers(sequence, k, skip_n_kmers)

        # Extract k-mers with quality filtering
        kmers: list[bytes] = []
        sequence_upper = sequence.upper()

        for i in range(len(sequence_upper) - k + 1):
            kmer = sequence_upper[i : i + k]

            # Skip if contains N
            if skip_n_kmers and b"N" in kmer:
                continue

            # Check quality scores for this k-mer's bases
            kmer_quals = quality_scores[i : i + k]
            if len(kmer_quals) < k:  # Safety check
                continue

            # Convert Phred+33 ASCII to quality scores
            # All bases in k-mer must meet minimum quality
            if all(ord(q) - 33 >= min_quality for q in kmer_quals):
                rc_kmer = self._py_reverse_complement(kmer)
                kmers.append(kmer if kmer <= rc_kmer else rc_kmer)

        return kmers

    def extract_kmers(
        self, sequence_bytes: bytes, k: int, skip_n_kmers: bool
    ) -> list[bytes]:
        """Extract k-mers using the configured implementation."""
        if not sequence_bytes or len(sequence_bytes) < k:
            return []

        try:
            normalized_seq = sequence_bytes.upper()
            # Pass skip_n_kmers to the selected extraction function
            return self._extract_func(normalized_seq, k, skip_n_kmers)
        except (
            TypeError
        ) as te:  # Catch if the underlying function doesn't take skip_n_kmers
            if "_py_extract_canonical_kmers" in str(te) or "rust_wrapper" in str(
                te
            ):  # Check if it's our functions
                # This might happen if the Rust wrapper or Python fallback was called incorrectly
                # or if the Rust function signature changes.
                logging.getLogger(__name__).error(
                    f"K-mer extraction function called with incorrect arguments: {te}. Falling back for this call."
                )
                # Fallback to Python version that we know accepts skip_n_kmers for this call
                # Or, if Rust is available but wrapper failed, this is more complex.
                # For simplicity, let's assume Python fallback is safest if signature mismatch.
                return self._py_extract_canonical_kmers(normalized_seq, k, skip_n_kmers)
            else:  # Other TypeError
                logging.getLogger(__name__).error(
                    f"Error during k-mer extraction: {te}"
                )
                return []
        except (ValueError, MemoryError, RuntimeError) as e:
            # ValueError: Invalid sequence data
            # MemoryError: Sequence too large
            # RuntimeError: Issues with extraction logic
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
    ) -> Generator[tuple[str, bytes, bytes], None, None]:
        """Processes a forward read and its corresponding reverse read."""
        if fwd_record is None:
            return
        fwd_id = (
            getattr(fwd_record, "id", "")
            if hasattr(fwd_record, "id")
            else (
                fwd_record[0]
                if isinstance(fwd_record, (tuple, list))
                else str(fwd_record)
            )
        )
        if not isinstance(fwd_id, str):
            fwd_id = str(fwd_id)
        # Normalize read ID
        fwd_id = fwd_id.split(' ')[0].split('/')[0]
        fwd_seq_str = (
            str(getattr(fwd_record, "seq", ""))
            if hasattr(fwd_record, "seq")
            else (
                fwd_record[1]
                if isinstance(fwd_record, (tuple, list)) and len(fwd_record) > 1
                else ""
            )
        )
        fwd_seq_bytes = fwd_seq_str.encode("utf-8")
        rev_seq_bytes = b""
        if rev_iter:
            try:
                rev_record = next(rev_iter)
                rev_seq_str = (
                    str(getattr(rev_record, "seq", ""))
                    if hasattr(rev_record, "seq")
                    else (
                        rev_record[1]
                        if isinstance(rev_record, (tuple, list)) and len(rev_record) > 1
                        else ""
                    )
                )
                rev_seq_bytes = rev_seq_str.encode("utf-8")
            except StopIteration:
                logger.warning(
                    f"Reverse file ended before forward file at read {fwd_id}. Treating remaining as single-end."
                )
                rev_iter = None
        if fwd_seq_bytes: # Only yield if forward sequence is not empty
            yield fwd_id, fwd_seq_bytes, rev_seq_bytes

    @staticmethod
    def parse_sequence_files(
        fwd_reads_path: pathlib.Path,
        rev_reads_path: Optional[pathlib.Path] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,  # Added chunk_size parameter
    ) -> Generator[
        List[Tuple[ReadId, bytes, bytes]], None, None
    ]:  # Yields List of Tuples
        """
        Parses sequence files and yields chunks of reads.

        Args:
            fwd_reads_path: Path to the forward reads file.
            rev_reads_path: Optional path to the reverse reads file.
            chunk_size: Number of reads to include in each chunk.

        Yields:
            A list of read tuples. Each tuple contains (ReadId, forward_sequence_bytes, reverse_sequence_bytes).
            For single-end reads, reverse_sequence_bytes will be empty bytes.
        """
        logger = logging.getLogger(__name__)
        logger.info(
            f"Parsing files in chunks of {chunk_size}. Forward: {fwd_reads_path}, Reverse: {rev_reads_path or 'N/A'}"
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
            chunk: list[tuple[ReadId, bytes, bytes]] = []
            read_counter = 0

            if rev_reads_path:
                rev_fh = SequenceFileProcessor.open_file_handle(rev_reads_path)
            try:
                fwd_iter_provider = None
                rev_iter_provider = None

                if fwd_format == "fasta":
                    fwd_iter_provider = SeqIO.parse(fwd_fh, "fasta")
                    if rev_fh:
                        rev_iter_provider = SeqIO.parse(rev_fh, "fasta")
                else:  # fastq
                    fwd_iter_provider = FastqGeneralIterator(fwd_fh)
                    if rev_fh:
                        rev_iter_provider = FastqGeneralIterator(rev_fh)

                if fwd_iter_provider is None:  # Should not happen with current logic
                    return

                for fwd_record in fwd_iter_provider:
                    for read_tuple in SequenceFileProcessor._process_read_pair(
                        fwd_record, rev_iter_provider, logger
                    ):
                        chunk.append(read_tuple)
                        read_counter += 1
                        if read_counter % chunk_size == 0:
                            yield chunk
                            chunk = []

                if chunk:  # Yield any remaining reads
                    yield chunk

            finally:
                if rev_fh:
                    rev_fh.close()


class CliArgs(BaseModel):
    """Pydantic model for command-line argument validation and management."""

    input_forward: list[pathlib.Path] = Field(
        description="Input forward read file(s) (FASTA/FASTQ, possibly gzipped)."
    )
    input_reverse: Optional[list[pathlib.Path]] = Field(
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
    min_base_quality: int = Field(
        default=20,
        ge=0,
        le=60,
        description="Minimum Phred quality score for bases in k-mers (FASTQ only). K-mers containing bases below this quality are excluded. Default 20 (99% accuracy)."
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


class KmerClassificationWorkflow:
    """Orchestrates the k-mer based strain classification workflow."""

    def __init__(self, args: CliArgs) -> None:
        """Initialize the workflow with validated arguments."""
        self.args = args
        self.database: Optional[StrainKmerDatabase] = None
        self.kmer_coverage: dict[bytes, int] = {}  # Track k-mer observation counts for scientific validation
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized KmerClassificationWorkflow")

    def _initialize_database(self) -> None:
        """Loads and initializes the StrainKmerDatabase."""
        # Validate database path exists before attempting to load
        if not self.args.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.args.db_path}")
        if not self.args.db_path.is_file():
            raise ValueError(f"Database path is not a file: {self.args.db_path}")

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
        self, record_tuple: tuple[ReadId, bytes, bytes]
    ) -> tuple[ReadId, CountVector]:
        """Processes a single read to count k-mer occurrences against the database."""
        if self.database is None:
            raise RuntimeError("Database not initialized.")

        read_id, fwd_seq_bytes, rev_seq_bytes = record_tuple
        from collections import Counter
        kmer_counts: Counter = Counter()  # Track k-mer multiplicities within this read

        if self.database is None:  # Should not happen if workflow is correct
            raise RuntimeError("Database not initialized in _count_kmers_for_read.")

        k_len = self.database.kmer_length
        # Use db_skip_n_kmers from database metadata if available, otherwise default to False
        skip_n = (
            self.database.db_skip_n_kmers
            if self.database.db_skip_n_kmers is not None
            else False
        )

        # Process forward sequence
        if fwd_seq_bytes:
            kmers_fwd = KMER_EXTRACTOR.extract_kmers(fwd_seq_bytes, k_len, skip_n)
            kmer_counts.update(kmers_fwd)
        # Process reverse sequence
        if rev_seq_bytes:
            kmers_rev = KMER_EXTRACTOR.extract_kmers(rev_seq_bytes, k_len, skip_n)
            kmer_counts.update(kmers_rev)

        # Count strain hits (use unique k-mers for classification)
        strain_counts = np.zeros(self.database.num_strains, dtype=np.uint8)
        for kmer_bytes in kmer_counts.keys():
            if self.database and hasattr(self.database, "get_strain_counts_for_kmer"):
                kmer_strain_counts = self.database.get_strain_counts_for_kmer(
                    kmer_bytes
                )
                if kmer_strain_counts is not None:
                    strain_counts += kmer_strain_counts
        # Ensure read_id is str
        if not isinstance(read_id, str):
            read_id = str(read_id)
        return read_id, strain_counts.astype(np.uint8)

    def _classify_reads_parallel(
        self,
        fwd_reads_path: pathlib.Path,
        rev_reads_path: Optional[pathlib.Path] = None,
    ) -> ReadHitResults:
        """Classifies reads from input files in parallel."""
        self.logger.info(f"Starting parallel classification for {fwd_reads_path}")

        if self.database is None:  # Guard against uninitialized database
            self.logger.error("Database not initialized before classification.")
            raise RuntimeError("Database not initialized.")

        analyzer = ClassificationAnalyzer(  # Create an analyzer instance for this run
            strain_names=self.database.strain_names,
            disambiguation_mode=self.args.disambiguation_mode,  # This will be used later
            abundance_threshold=self.args.abundance_threshold,
            num_processes=self.args.num_processes,
        )

        read_chunk_iterator = SequenceFileProcessor.parse_sequence_files(
            fwd_reads_path, rev_reads_path, chunk_size=self.args.chunk_size
        )

        accumulated_clear_assignments: dict[ReadId, int] = {}
        accumulated_no_hit_ids: list[ReadId] = []
        ambiguous_hit_files: list[pathlib.Path] = []

        temp_ambiguous_dir = self.args.output_dir / "temp_ambiguous_chunks"
        temp_ambiguous_dir.mkdir(parents=True, exist_ok=True)
        chunk_idx = 0

        # Pass 1: Classify reads, separate clear hits, store ambiguous hits to disk
        with mp.Pool(processes=self.args.num_processes) as pool:
            for read_chunk in read_chunk_iterator:
                if not read_chunk:
                    continue

                chunk_hit_results: ReadHitResults = pool.map(
                    self._count_kmers_for_read, read_chunk
                )

                if self.args.save_raw_kmer_hits:  # Call save_raw_kmer_hits per chunk
                    # Assuming fwd_reads_path gives a basis for sample_name
                    sample_name_for_raw_hits = _get_sample_name(fwd_reads_path)
                    self._save_raw_kmer_hits(
                        chunk_hit_results, sample_name_for_raw_hits, chunk_idx
                    )

                # Process this chunk's results immediately
                clear_hits_chunk, ambiguous_hits_chunk_dict, no_hit_ids_chunk = (
                    analyzer.separate_hit_categories(chunk_hit_results)
                )

                clear_assignments_chunk = analyzer.resolve_clear_hits_to_indices(
                    clear_hits_chunk
                )
                accumulated_clear_assignments.update(clear_assignments_chunk)
                accumulated_no_hit_ids.extend(no_hit_ids_chunk)

                if ambiguous_hits_chunk_dict:
                    chunk_file = temp_ambiguous_dir / f"ambiguous_chunk_{chunk_idx}.pkl"
                    with open(chunk_file, "wb") as f:
                        pickle.dump(ambiguous_hits_chunk_dict, f)
                    ambiguous_hit_files.append(chunk_file)
                chunk_idx += 1

        self.logger.info(
            f"Finished first pass of classification for {fwd_reads_path}. "
            f"Clear assignments: {len(accumulated_clear_assignments)}, "
            f"No hits: {len(accumulated_no_hit_ids)}, "
            f"Ambiguous chunks: {len(ambiguous_hit_files)}"
        )

        return (
            accumulated_clear_assignments,
            accumulated_no_hit_ids,
            ambiguous_hit_files,
            temp_ambiguous_dir,
        )

    def _save_raw_kmer_hits(
        self, chunk_hit_results: ReadHitResults, sample_name: str, chunk_idx: int
    ) -> None:
        """Saves raw k-mer hit counts for a chunk of reads to a CSV file."""
        if not chunk_hit_results or self.database is None:
            # self.logger.warning(f"No raw hit results to save for sample: {sample_name}, chunk {chunk_idx}")
            return  # Reduce noise for empty chunks

        # self.logger.info(f"Saving raw k-mer hit counts for sample: {sample_name}, chunk {chunk_idx}")
        df_data = []
        for read_id, count_vector in chunk_hit_results:
            row = {"read_id": read_id}
            # Ensure self.database and self.database.strain_names are valid
            if self.database and hasattr(self.database, "strain_names"):
                strain_names = self.database.strain_names
            else:  # Fallback, though database should be initialized
                strain_names = [f"strain_{j}" for j in range(len(count_vector))]

            for i, strain_name_val in enumerate(strain_names):
                if i < len(count_vector):
                    row[strain_name_val] = count_vector[i]
            df_data.append(row)

        if not df_data:  # Possible if all count_vectors were empty or names issue
            return

        df = pd.DataFrame(df_data)

        raw_hits_dir = self.args.output_dir / "raw_kmer_hits" / sample_name
        raw_hits_dir.mkdir(parents=True, exist_ok=True)

        output_path = raw_hits_dir / f"raw_hits_chunk_{chunk_idx}.csv"
        df.to_csv(output_path, index=False)
        # self.logger.info(f"Raw k-mer hit counts for chunk {chunk_idx} saved to: {output_path}")

    def _save_reproducibility_metadata(self) -> None:
        """
        Save analysis metadata for reproducibility.

        Saves: software version, execution timestamp, all parameters,
        database info. Critical for scientific reproducibility.
        """
        import json
        from datetime import datetime
        import sys

        metadata = {
            "strainr_version": "1.0.0",  # TODO: Load from package __version__
            "python_version": sys.version,
            "execution_timestamp": datetime.now().isoformat(),
            "parameters": {
                "db_path": str(self.args.db_path),
                "num_processes": self.args.num_processes,
                "disambiguation_mode": self.args.disambiguation_mode,
                "abundance_threshold": self.args.abundance_threshold,
                "chunk_size": self.args.chunk_size,
                "min_base_quality": self.args.min_base_quality,
                "perform_binning": self.args.perform_binning,
                "save_raw_kmer_hits": self.args.save_raw_kmer_hits,
            },
            "database_info": {
                "kmer_length": self.database.kmer_length if self.database else None,
                "num_strains": self.database.num_strains if self.database else None,
                "num_kmers": self.database.num_kmers if self.database else None,
                "skip_n_kmers": self.database.db_skip_n_kmers if self.database else None,
            },
            "input_files": {
                "forward": [str(p) for p in (self.args.input_forward if isinstance(self.args.input_forward, list) else [self.args.input_forward])],
                "reverse": [str(p) for p in self.args.input_reverse] if self.args.input_reverse else None,
            },
        }

        metadata_file = self.args.output_dir / "analysis_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved reproducibility metadata to {metadata_file}")

    def run_workflow(self) -> None:
        # This method orchestrates the main StrainR classification workflow.
        # It handles database initialization, iterating through input files,
        # parallel read classification (chunk-wise), prior calculation,
        # ambiguous hit resolution (chunk-wise from temp files),
        # and final abundance calculation and reporting.
        # It handles:
        #   - Database initialization.
        #   - Analyzer and AbundanceCalculator initialization.
        """Executes the main classification and analysis workflow."""
        self.logger.info("Starting StrainR workflow")

        try:
            self._initialize_database()
            # Save reproducibility metadata after database is loaded
            self._save_reproducibility_metadata()
            if (
                self.database is None
            ):  # Should be caught by _initialize_database, but defensive check
                self.logger.critical("Database initialization failed critically.")
                return  # Cannot proceed

            # Analyzer is now created per-sample or per-run if its state is purely transient
            # For prior calculation, it's better to have one analyzer instance if it holds no state
            # or re-initialize as needed. The current ClassificationAnalyzer is stateless beyond init args.
            analyzer = ClassificationAnalyzer(
                strain_names=self.database.strain_names,
                disambiguation_mode=self.args.disambiguation_mode,
                abundance_threshold=self.args.abundance_threshold,  # Not used by analyzer directly
                num_processes=self.args.num_processes,
            )
            abundance_calc = AbundanceCalculator(
                strain_names=self.database.strain_names,
                abundance_threshold=self.args.abundance_threshold,
            )

            # Prepare file lists (current Pydantic model handles single item to list conversion)
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
                if rev_files and i < len(rev_files):  # Ensure index is valid
                    rev_reads_path = rev_files[i]

                # 1. Classify reads (Pass 1: get clear assignments and temp ambiguous files)
                # _classify_reads_parallel now returns:
                # all_clear_assignments, all_no_hit_ids, ambiguous_hit_files_paths, temp_ambiguous_dir
                (
                    all_clear_assignments,
                    all_no_hit_ids,
                    ambiguous_hit_files,
                    temp_ambiguous_dir,
                ) = self._classify_reads_parallel(fwd_reads_path, rev_reads_path)

                # Note: _save_raw_kmer_hits is not called here with all_hit_results anymore.
                # If raw hits need to be saved, _classify_reads_parallel should handle it per chunk
                # or this part needs significant rework. For now, focusing on memory for classification.
                # The new _save_raw_kmer_hits is designed to be called per chunk if needed inside _classify_reads_parallel
                # but it's not currently called from there to reduce I/O during the parallel phase.
                # If self.args.save_raw_kmer_hits is True, one might iterate and load chunks again, or save them
                # during the initial pool.map (but that adds I/O to the CPU-bound task).
                # For this refactor, I'll assume saving raw hits is secondary to memory efficiency of classification.

                # 2. Calculate global priors from all clear assignments
                prior_counts = analyzer.calculate_strain_prior_from_assignments(
                    all_clear_assignments
                )
                prior_probs = analyzer.convert_prior_counts_to_probability_vector(
                    prior_counts
                )

                # 3. Resolve ambiguous hits (Pass 2: iterate through temp ambiguous files)
                resolved_ambiguous_assignments_all_chunks: Dict[ReadId, int] = {}
                for ambig_chunk_file in ambiguous_hit_files:
                    with open(ambig_chunk_file, "rb") as f:
                        ambiguous_hits_chunk_dict: Dict[ReadId, CountVector] = (
                            pickle.load(f)
                        )

                    if ambiguous_hits_chunk_dict:  # Ensure it's not empty
                        resolved_chunk = analyzer.resolve_ambiguous_hits_parallel(
                            ambiguous_hits_chunk_dict, prior_probs
                        )
                        resolved_ambiguous_assignments_all_chunks.update(resolved_chunk)

                    try:  # Clean up individual chunk file
                        ambig_chunk_file.unlink()
                    except OSError as e:
                        self.logger.warning(
                            f"Could not delete temp ambiguous chunk file {ambig_chunk_file}: {e}"
                        )

                try:  # Clean up temp directory
                    if temp_ambiguous_dir.exists() and not any(
                        temp_ambiguous_dir.iterdir()
                    ):
                        temp_ambiguous_dir.rmdir()
                    elif (
                        temp_ambiguous_dir.exists()
                    ):  # If somehow files remain (e.g. error during deletion)
                        self.logger.warning(
                            f"Temporary ambiguous directory {temp_ambiguous_dir} is not empty after processing. Manual cleanup might be needed."
                        )
                except OSError as e:
                    self.logger.warning(
                        f"Could not delete temp ambiguous directory {temp_ambiguous_dir}: {e}"
                    )

                # 4. Combine all assignments
                final_assignments = analyzer.combine_assignments(
                    all_clear_assignments,
                    resolved_ambiguous_assignments_all_chunks,
                    all_no_hit_ids,
                    unassigned_marker="NA",
                )

                # 5. Save final_assignments and strain_names for potential downstream use
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
                        strain_names = (
                            self.database.strain_names
                            if self.database and hasattr(self.database, "strain_names")
                            else []
                        )
                        for strain_name_item in strain_names:  # Corrected variable name
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
                        {
                            k: v for k, v in final_assignments.items()
                        },  # Convert to proper dict type
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
                    for strain_name_iter in (
                        self.database.strain_names if self.database else []
                    ):  # Iterate through known strains
                        count = raw_counts.get(strain_name_iter, 0)
                        sample_relab_temp = count / total_reads_for_relab
                        if sample_relab_temp >= self.args.abundance_threshold:
                            sum_hits_passing_threshold += count

                # Ensure all strains and NA are in the df. Use a combined list of known strain names and any potentially new names from raw_counts (e.g. "NA" or others if they appear)
                # However, the original logic iterated self.database.strain_names + ["NA"], so we stick to that to replicate behavior.
                all_strain_keys_for_df = (
                    self.database.strain_names if self.database else []
                ) + ["NA"]
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
