#!/usr/bin/env python
"""
StrainR Pangenome Analysis Script.

This script provides a workflow for k-mer based analysis of sequence reads,
potentially with a focus on pangenome contexts (though the specific "pangenome"
aspect isn't fully elaborated in the original functions beyond standard classification).

It involves:
1. Loading a k-mer database.
2. Parsing input FASTQ/FASTA files (single or paired-end).
3. Classifying reads by counting k-mer hits against the database.
4. Analyzing classification results:
    - Separating clear, ambiguous, and no-hit reads.
    - Calculating priors from clear hits.
    - Resolving ambiguous hits using various disambiguation strategies.
    - Combining all assignments.
5. Calculating and reporting strain abundances.
"""

import multiprocessing as mp
import pathlib
import pickle
import time
import logging  # Added for logging
import sys  # Added for logging handler

from collections import Counter
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from Bio.SeqIO.QualityIO import FastqGeneralIterator

# StrainR module imports
from strainr.utils import open_file_transparently  # Corrected import
from strainr.database import StrainKmerDatabase  # Corrected import
from strainr.analyze import ClassificationAnalyzer
from strainr.genomic_types import ReadId, CountVector, StrainIndex, ReadHitResults
from typing import Callable  # Added Callable, Set

# Argument parsing setup (assuming it returns an argparse-like namespace)
from strainr.parameter_config import process_arguments  # This will be used as is


# --- Python K-mer Extraction Fallback ---
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
        rc_kmer_bytes = _py_reverse_complement(kmer)
        if kmer <= rc_kmer_bytes:
            kmers.append(kmer)
        else:
            kmers.append(rc_kmer_bytes)
    return kmers


_extract_kmers_func: Callable[[bytes, int], List[bytes]]
RUST_KMER_COUNTER_AVAILABLE: bool

try:
    from kmer_counter_rs import extract_kmers_rs

    _extract_kmers_func = extract_kmers_rs
    RUST_KMER_COUNTER_AVAILABLE = True
    # Using print here as logger might not be configured when this module is imported elsewhere
    print(
        "Successfully imported Rust k-mer counter for pangenome script. Using Rust implementation."
    )
except ImportError:
    RUST_KMER_COUNTER_AVAILABLE = False
    _extract_kmers_func = _py_extract_canonical_kmers
    print(
        "Warning: Rust k-mer counter (kmer_counter_rs.extract_kmers_rs) not found for pangenome script. Using Python fallback."
    )
except Exception as e:
    RUST_KMER_COUNTER_AVAILABLE = False
    _extract_kmers_func = _py_extract_canonical_kmers
    print(
        f"Error importing Rust k-mer counter for pangenome script: {e}. Using Python fallback."
    )


# --- Configuration ---
SETCHUNKSIZE = 10000  # For multiprocessing map functions

# --- Logging Setup ---
logger = logging.getLogger(__name__)
# Ensure logging is configured if not already done by other modules or if this is a standalone entry point
if not logger.hasHandlers():  # Check if handlers are already configured
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("strainr_pangenome.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


class PangenomeAnalysisWorkflow:
    """
    Orchestrates the pangenome analysis workflow.

    This class encapsulates database loading, read processing, k-mer classification,
    statistical analysis of hits, and abundance reporting.
    """

    UNASSIGNED_READ_MARKER = "NA"  # Marker for unassigned reads

    def __init__(self, args: Any) -> None:  # args type depends on process_arguments
        """
        Initializes the PangenomeAnalysisWorkflow.

        Args:
            args: Command-line arguments, typically an `argparse.Namespace`
                  or a Pydantic model if `process_arguments` was refactored.
                  Expected to have attributes like `db` (db_path), `procs`,
                  `mode` (disambiguation_mode), `thresh` (abundance_threshold),
                  `input` (input_files), `out` (output_dir), `save_raw_hits`.
        """
        logger.info("Initializing PangenomeAnalysisWorkflow.")
        self.args = args
        self.database: Optional[StrainKmerDatabase] = None
        self.analyzer: Optional[ClassificationAnalyzer] = None

        # Ensure output directory exists
        self.output_dir: pathlib.Path = pathlib.Path(self.args.out)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize RNG for methods that might need it (e.g., certain disambiguation modes)
        # This rng is specific to this workflow instance if needed locally,
        # ClassificationAnalyzer also has its own.
        self.rng = np.random.default_rng()

    def _initialize_database(self) -> None:
        """Loads the k-mer database using StrainKmerDatabase."""
        logger.info(f"Loading k-mer database from: {self.args.db}")
        self.database = StrainKmerDatabase(database_filepath=self.args.db)
        logger.info(
            f"Database loaded: {self.database.num_kmers} k-mers, "
            f"{self.database.num_strains} strains, k-mer length {self.database.kmer_length}."
        )
        # Initialize ClassificationAnalyzer once the database (and thus strain names) is available
        self.analyzer = ClassificationAnalyzer(
            strain_names=self.database.strain_names,
            disambiguation_mode=self.args.mode,
            # abundance_threshold is not used by analyzer, but by local output functions
            num_processes=self.args.procs,
        )

    def _parse_fastq_sequences(
        self, fastq_file_path: pathlib.Path
    ) -> Generator[Tuple[ReadId, bytes], None, None]:
        """
        Parses a FASTQ file and yields read IDs and their sequences as bytes.
        Uses `open_file_transparently` for potentially gzipped files.

        Args:
            fastq_file_path: Path to the input FASTQ file.

        Yields:
            Tuples of (ReadId, sequence_bytes).
        """
        logger.info(f"Parsing FASTQ file: {fastq_file_path}")
        with open_file_transparently(fastq_file_path) as file_handle:
            for read_id, sequence_str, _ in FastqGeneralIterator(file_handle):
                yield read_id, sequence_str.encode("utf-8")  # Ensure bytes

    def _count_kmers_for_read(
        self, read_data: Tuple[ReadId, bytes]
    ) -> Tuple[ReadId, CountVector]:
        """
        Counts k-mer occurrences for a single read against the loaded database.

        Args:
            read_data: A tuple containing (ReadId, sequence_bytes).

        Returns:
            A tuple (ReadId, CountVector) for the read.
        """
        if self.database is None:
            raise RuntimeError("Database not initialized prior to k-mer counting.")

        read_id, seq_bytes = read_data
        # Initialize a zero vector for this read's k-mer counts
        current_read_strain_counts = np.zeros(self.database.num_strains, dtype=np.uint8)

        if not seq_bytes or len(seq_bytes) < self.database.kmer_length:
            return read_id, current_read_strain_counts  # Sequence too short or empty

        # Normalize sequence (e.g., to uppercase) before k-mer extraction if Python fallback is used
        # The Rust function handles its own normalization.
        normalized_seq_bytes = seq_bytes.upper()

        kmers_from_read: List[bytes] = self._get_kmers_for_sequence(
            normalized_seq_bytes
        )

        if not kmers_from_read:
            return read_id, current_read_strain_counts

        for kmer_bytes in kmers_from_read:
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
            # The caller (_count_kmers_for_read) now does basic .upper() before calling this.
            return _extract_kmers_func(sequence_bytes, self.database.kmer_length)
        except Exception as e:
            logger.error(
                f"Error during k-mer extraction for sequence of length {len(sequence_bytes)} with k={self.database.kmer_length}: {e}"
            )
            return []

    def _classify_reads_in_file(self, fastq_file_path: pathlib.Path) -> ReadHitResults:
        """
        Classifies all reads in a FASTQ file using multiprocessing.

        Args:
            fastq_file_path: Path to the input FASTQ file.

        Returns:
            A list of (ReadId, CountVector) tuples.
        """
        start_time = time.time()
        logger.info(f"Starting classification for {fastq_file_path}...")

        # Count reads for progress (can be slow for large gzipped files)
        # n_reads = 0
        # with open_file_transparently(fastq_file_path) as f_count:
        #     for _ in FastqGeneralIterator(f_count):
        #         n_reads +=1
        # logger.info(f"Processing approximately {n_reads} reads from {fastq_file_path}.")
        # Decided to remove read counting for performance, can be added back if progress is essential

        sequence_iterator = self._parse_fastq_sequences(fastq_file_path)

        results: ReadHitResults
        with mp.Pool(processes=self.args.procs) as pool:
            results = pool.map(
                self._count_kmers_for_read, sequence_iterator, chunksize=SETCHUNKSIZE
            )

        elapsed_time = time.time() - start_time
        logger.info(
            f"Classification of {fastq_file_path} finished in {elapsed_time:.2f}s. Classified {len(results)} reads."
        )
        return results

    def _save_raw_kmer_spectra(
        self, raw_hit_results: ReadHitResults, sample_name: str
    ) -> None:
        """Saves raw k-mer hit spectra (CountVectors per read) to a pickle file."""
        if self.database is None:
            raise RuntimeError("Database not initialized.")

        spectra_output_dir = self.output_dir / "kmer_spectra"
        spectra_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = spectra_output_dir / f"{sample_name}_raw_kmer_spectra.pkl"

        # Structure to save: dict of read_id -> CountVector
        # And also save strain names for context
        data_to_save = {
            "strain_names": self.database.strain_names,
            "read_spectra": dict(raw_hit_results),
        }

        with output_file.open("wb") as pf:
            pickle.dump(data_to_save, pf)
        logger.info(f"Raw k-mer spectra saved to: {output_file}")

    # --- Abundance Calculation and Output (Refactored In-Place) ---
    def _output_abundance_results(
        self,
        final_assignments: Dict[
            ReadId, Union[StrainIndex, str]
        ],  # From ClassificationAnalyzer
        sample_name: str,
    ) -> pd.DataFrame:
        """
        Calculates abundances and generates output files and console display.
        This method consolidates the logic from the original script's output functions.

        Args:
            final_assignments: Dictionary mapping ReadId to assigned StrainIndex or "NA".
            sample_name: Name of the sample being processed, for output file naming.

        Returns:
            A Pandas DataFrame containing the abundance report.
        """
        if self.database is None or self.analyzer is None:
            raise RuntimeError(
                "Database or Analyzer not initialized before outputting results."
            )

        # 1. Translate final assignments (indices/"NA") to strain names
        # This is a bit redundant if final_assignments already contains names for assigned,
        # but ensures "NA" and indices are handled correctly.
        named_assignments: Dict[ReadId, str] = {}
        for read_id, assignment in final_assignments.items():
            if isinstance(assignment, int):  # StrainIndex
                if 0 <= assignment < len(self.database.strain_names):
                    named_assignments[read_id] = self.database.strain_names[assignment]
                else:  # Should not happen if analyzer works correctly
                    named_assignments[read_id] = self.UNASSIGNED_READ_MARKER
            else:  # It's a string, should be self.UNASSIGNED_READ_MARKER
                named_assignments[read_id] = str(assignment)

        # 2. Calculate hit counts per strain name
        hit_counts: Counter[str] = Counter(named_assignments.values())
        for s_name in self.database.strain_names:  # Ensure all strains are present
            hit_counts.setdefault(s_name, 0)
        hit_counts.setdefault(self.UNASSIGNED_READ_MARKER, 0)

        # 3. Calculate sample relative abundance (normalized by all reads, including NA)
        total_reads_for_sample_relab = sum(hit_counts.values())
        sample_relab: Dict[str, float] = {
            name: count / total_reads_for_sample_relab
            if total_reads_for_sample_relab > 0
            else 0.0
            for name, count in hit_counts.items()
        }

        # 4. Calculate intra-sample relative abundance (normalized by assigned reads, post-threshold)
        # Strains below threshold are excluded from this normalization base.
        counts_for_intra_relab = Counter()
        for strain_name, rel_ab in sample_relab.items():
            if (
                strain_name != self.UNASSIGNED_READ_MARKER
                and rel_ab >= self.args.thresh
            ):
                counts_for_intra_relab[strain_name] = hit_counts[strain_name]

        total_hits_for_intra_relab = sum(counts_for_intra_relab.values())
        intra_relab: Dict[str, float] = {
            name: count / total_hits_for_intra_relab
            if total_hits_for_intra_relab > 0
            else 0.0
            for name, count in counts_for_intra_relab.items()
        }
        # Ensure all original strains are in intra_relab dict, with 0 if they didn't pass
        for s_name in self.database.strain_names:
            intra_relab.setdefault(s_name, 0.0)
        intra_relab.setdefault(
            self.UNASSIGNED_READ_MARKER, 0.0
        )  # NA has 0 intra-sample relab

        # 5. Create DataFrame
        report_data = []
        # Ensure order: known strains first, then "NA"
        ordered_strain_names_for_report = self.database.strain_names + [
            self.UNASSIGNED_READ_MARKER
        ]

        for strain_name in ordered_strain_names_for_report:
            if strain_name not in hit_counts:
                continue  # Skip if a name (e.g. NA) had no hits at all

            report_data.append({
                "strain_name": strain_name,
                "sample_hits": hit_counts.get(strain_name, 0),
                "sample_relab": sample_relab.get(strain_name, 0.0),
                "intra_relab": intra_relab.get(strain_name, 0.0)
                if strain_name != self.UNASSIGNED_READ_MARKER
                else 0.0,
            })

        abundance_df = pd.DataFrame(report_data).set_index("strain_name")
        # Sort by sample_hits, ensuring NA is last if present
        if self.UNASSIGNED_READ_MARKER in abundance_df.index:
            df_assigned = abundance_df.drop(self.UNASSIGNED_READ_MARKER).sort_values(
                by="sample_hits", ascending=False
            )
            df_unassigned = abundance_df.loc[[self.UNASSIGNED_READ_MARKER]]
            abundance_df_sorted = pd.concat([df_assigned, df_unassigned])
        else:
            abundance_df_sorted = abundance_df.sort_values(
                by="sample_hits", ascending=False
            )

        # Save to TSV
        output_tsv_path = self.output_dir / f"{sample_name}_abundance.tsv"
        abundance_df_sorted.to_csv(output_tsv_path, sep="\t", float_format="%.6f")
        logger.info(f"Abundance report for {sample_name} saved to: {output_tsv_path}")

        # 6. Display to console
        self._display_console_report(abundance_df_sorted, sample_name)

        return abundance_df_sorted

    def _display_console_report(
        self, abundance_df: pd.DataFrame, sample_name: str, top_n: int = 10
    ) -> None:
        """Displays a formatted abundance report to the console."""
        logger.info(f"Displaying abundance summary for {sample_name}:")
        print(f"\n--- Top {top_n} Strain Abundances for Sample: {sample_name} ---")

        # Filter out zero abundances for display unless it's the Unassigned category
        df_display = abundance_df[
            (abundance_df["sample_hits"] > 0)
            | (abundance_df.index == self.UNASSIGNED_READ_MARKER)
        ]

        # Separate Unassigned for specific display
        unassigned_info_str = ""
        if self.UNASSIGNED_READ_MARKER in df_display.index:
            unassigned_row = df_display.loc[self.UNASSIGNED_READ_MARKER]
            unassigned_info_str = (
                f"{self.UNASSIGNED_READ_MARKER:<30} "
                f"{int(unassigned_row['sample_hits']):<12} "
                f"{unassigned_row['sample_relab']:.4f}"
            )  # intra_relab for Unassigned is typically 0 or not meaningful
            df_display = df_display.drop(index=self.UNASSIGNED_READ_MARKER)

        # Display header
        print(
            f"{'Strain Name':<30} {'Sample Hits':<12} {'Sample RelAb':<12} {'Intra-Sample RelAb':<18}"
        )
        print("-" * 75)

        for strain_name, row_data in df_display.head(top_n).iterrows():
            print(
                f"{str(strain_name):<30} "
                f"{int(row_data['sample_hits']):<12} "
                f"{row_data['sample_relab']:.4f}           "  # Align
                f"{row_data['intra_relab']:.4f}"
            )

        if unassigned_info_str:
            print("-" * 75)
            print(unassigned_info_str)
        print("--- End of Report ---")

    def run(self) -> None:
        """
        Main execution method for the pangenome analysis workflow.
        """
        logger.info("Pangenome analysis workflow started.")
        self._initialize_database()
        if (
            self.database is None or self.analyzer is None
        ):  # Should have been initialized
            logger.critical("Database or Analyzer initialization failed. Exiting.")
            sys.exit(1)

        # Assuming args.input is a list of forward read files from process_arguments
        # This part needs to align with how process_arguments structures 'input'
        # If it's a single file, wrap it in a list for iteration
        input_files_fwd = self.args.input
        if not isinstance(input_files_fwd, list):
            input_files_fwd = [input_files_fwd]

        # Paired-end logic needs corresponding reverse files if provided by process_arguments
        # For simplicity, this example assumes single-end or that pairing is handled by process_arguments
        # and `input_files_fwd` would contain tuples or be processed accordingly.
        # The original script had `args.input.reverse()` and then iterated.
        # Here, we'll iterate through what `process_arguments` gives.

        for fwd_fastq_path_str in input_files_fwd:
            fwd_fastq_path = pathlib.Path(
                fwd_fastq_path_str
            )  # Ensure it's a Path object
            sample_name = fwd_fastq_path.name.split("_R1")[0].split(".")[
                0
            ]  # Basic sample naming from fwd read
            logger.info(
                f"Processing sample: {sample_name} (File: {fwd_fastq_path.name})"
            )

            # This script doesn't explicitly handle paired-end reads in its main loop structure.
            # The _classify_reads_in_file and _parse_fastq_sequences can, but the loop here is simple.
            # If paired-end is a use case, argument parsing and file iteration need adjustment.

            # 1. Classify reads
            raw_kmer_hits: ReadHitResults = self._classify_reads_in_file(fwd_fastq_path)

            if not raw_kmer_hits:
                logger.warning(
                    f"No reads classified for {sample_name}. Skipping further analysis for this sample."
                )
                continue

            # 2. Optionally save raw k-mer spectra (CountVectors per read)
            if self.args.save_raw_hits:  # Assuming `save_raw_hits` is an arg
                self._save_raw_kmer_spectra(raw_kmer_hits, sample_name)

            # 3. Analyze classification results using ClassificationAnalyzer
            clear_hits_vec, ambiguous_hits_vec, no_hit_ids = (
                self.analyzer.separate_hit_categories(raw_kmer_hits)
            )

            clear_assignments_idx = self.analyzer.resolve_clear_hits_to_indices(
                clear_hits_vec
            )

            prior_counts = self.analyzer.calculate_strain_prior_from_assignments(
                clear_assignments_idx
            )
            prior_prob_vector = (
                self.analyzer.convert_prior_counts_to_probability_vector(prior_counts)
            )

            resolved_ambiguous_idx = self.analyzer.resolve_ambiguous_hits_parallel(
                ambiguous_hits_vec, prior_prob_vector
            )

            final_assignments: Dict[ReadId, Union[StrainIndex, str]] = (
                self.analyzer.combine_assignments(
                    clear_assignments_idx,
                    resolved_ambiguous_idx,
                    no_hit_ids,
                    unassigned_marker=self.UNASSIGNED_READ_MARKER,
                )
            )

            # 4. Output abundance results
            self._output_abundance_results(final_assignments, sample_name)

        logger.info("Pangenome analysis workflow finished.")


if __name__ == "__main__":
    logger.info("StrainR Pangenome Script Started.")

    # Parse arguments using the function from strainr.parameter_config
    # This assumes process_arguments.parse_args() returns an object
    # compatible with PangenomeAnalysisWorkflow constructor (e.g., has .db, .procs etc.)
    args_namespace = process_arguments.parse_args()

    workflow = PangenomeAnalysisWorkflow(args=args_namespace)
    try:
        workflow.run()
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)
    logger.info("StrainR Pangenome Script Finished Successfully.")
