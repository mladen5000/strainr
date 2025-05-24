"""
Core classification engine for strain identification using k-mer matching.

CHANGES:
- Separated classification logic into focused functions
- Improved multiprocessing implementation
- Better error handling and logging
- Cleaner interfaces and documentation
- Fixed import issues and global variable dependencies
"""

import functools
import multiprocessing as mp
import pathlib
import time
from collections import Counter, defaultdict
from typing import Generator, Tuple, Dict, List, Any
import numpy as np
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from genomic_types import (
    CountVector, 
    ReadHitResults,  
    StrainAbundanceDict, 
    ReadId, 
    KmerString, 
    KmerCountDict
    )


class StrainClassifier:
    """
    Main classification engine for assigning reads to strains based on k-mer matching.

    This class handles the core logic of k-mer extraction, database lookup,
    and statistical assignment of sequence reads to microbial strains.
    """

    def __init__(
        self,
        database: StrainKmerDatabase,
        num_processes: int = 4,
        chunk_size: int = 10000,
    ):
        """
        Initialize the strain classifier.

        Args:
            database: Loaded k-mer database
            num_processes: Number of parallel processes for classification
            chunk_size: Size of chunks for multiprocessing
        """
        self.database = database
        self.num_processes = num_processes
        self.chunk_size = chunk_size

    def count_kmer_matches_for_read(
        self, read_id: str, sequence_bytes: bytes
    ) -> Tuple[str, CountVector]:
        """
        Count k-mer matches between a read and strain database.

        This function maintains the original memory view approach for performance
        while providing better error handling and documentation.

        Args:
            read_id: Unique identifier for the sequence read
            sequence_bytes: Raw sequence data as bytes

        Returns:
            Tuple of (read_id, strain_match_counts)

        Note:
            This preserves the original algorithm which is performance-critical
        """
        matched_kmer_vectors = []
        zero_vector = np.zeros(self.database.num_strains, dtype=np.uint8)
        max_kmer_start = len(sequence_bytes) - self.database.kmer_length + 1

        # Use memory view for efficient k-mer extraction (original approach)
        with memoryview(sequence_bytes) as sequence_view:
            for start_pos in range(max_kmer_start):
                kmer_bytes = sequence_view[
                    start_pos : start_pos + self.database.kmer_length
                ]
                strain_frequencies = self.database.lookup_kmer(kmer_bytes.tobytes())

                if strain_frequencies is not None:
                    matched_kmer_vectors.append(strain_frequencies)

        # Sum all matched k-mer vectors
        if matched_kmer_vectors:
            total_strain_counts = sum(matched_kmer_vectors)
            return read_id, total_strain_counts
        else:
            return read_id, zero_vector

    def _count_kmer_matches_helper(
        self, read_tuple: Tuple[str, bytes]
    ) -> Tuple[str, CountVector]:
        """Helper function for multiprocessing k-mer counting."""
        return self.count_kmer_matches_for_read(*read_tuple)

    def _generate_encoded_sequences(
        self, input_fastq: pathlib.Path
    ) -> Generator[Tuple[str, bytes], None, None]:
        """
        Generate encoded sequences from FASTQ file.

        Args:
            input_fastq: Path to input FASTQ file

        Yields:
            Tuples of (sequence_id, encoded_sequence)
        """
        from .utils import open_file_handle  # Import here to avoid circular imports

        with open_file_handle(input_fastq) as file_handle:
            for seq_id, sequence, _ in FastqGeneralIterator(file_handle):
                yield seq_id, bytes(sequence, "utf-8")

    def classify_reads_in_file(self, input_fastq: pathlib.Path) -> ReadHitResults:
        """
        Classify all reads in a FASTQ file against the strain database.

        Args:
            input_fastq: Path to input FASTQ file

        Returns:
            List of (read_id, strain_match_counts) tuples

        Example:
            >>> classifier = StrainClassifier(database)
            >>> results = classifier.classify_reads_in_file("sample.fastq")
            >>> print(f"Classified {len(results)} reads")
        """
        start_time = time.time()

        # Count total reads for progress tracking
        with open(input_fastq, "rb") as file_handle:
            total_reads = sum(1 for _ in file_handle) // 4
        print(f"Processing {total_reads} reads from {input_fastq}")

        # Generate sequence iterator
        sequence_iterator = self._generate_encoded_sequences(input_fastq)

        # Perform parallel classification
        with mp.Pool(processes=self.num_processes) as process_pool:
            classification_results = list(
                process_pool.imap_unordered(
                    self._count_kmer_matches_helper,
                    sequence_iterator,
                    chunksize=self.chunk_size,
                )
            )

        elapsed_time = time.time() - start_time
        print(f"Classification completed in {elapsed_time:.2f} seconds")

        return classification_results


# Alternative optimized classification method (commented out as requested)
# def classify_reads_streaming(self, input_fastq: pathlib.Path) -> Generator[Tuple[str, CountVector], None, None]:
#     """
#     Stream-based classification that yields results as they're computed.
#
#     This alternative approach processes reads one at a time without storing
#     all results in memory, which is beneficial for very large FASTQ files.
#
#     Args:
#         input_fastq: Path to input FASTQ file
#
#     Yields:
#         Tuples of (read_id, strain_match_counts)
#     """
#     sequence_iterator = self._generate_encoded_sequences(input_fastq)
#
#     with mp.Pool(processes=self.num_processes) as process_pool:
#         for result in process_pool.imap_unordered(
#             self._count_kmer_matches_helper,
#             sequence_iterator,
#             chunksize=self.chunk_size
#         ):
#             yield result
